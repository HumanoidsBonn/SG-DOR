#!/usr/bin/env python3
"""Analyze how varied the occlusion supervision is (fruits + directions + positives).

Motivation:
  - Occlusion leaf losses (potential/rank/consistency) are currently gated to "dominant" directions (union>=tau)
    to avoid drowning in trivial negatives.
  - We want to understand the supervised fruit/direction distribution under the current sampling regime and
    whether it makes sense to expand supervision to mid/low-occlusion directions (e.g. via stratified direction
    sampling or soft union-based weighting).

This script simulates the dataset sampling (BFS + occl-aug + extra supervised fruits) and then:
  - buckets directions into low/mid/high based on union thresholds (default: <=0.25, (0.25,0.5), >=0.5)
  - materializes the *selected* leaf set per fruit (cap K) using the same geometry-only heuristic as the dataset
  - measures, per bucket:
      - fraction of directions that have at least one positive leaf (pot>=eps)
      - fraction of directions that have any rank-mass (k3 sum after eps gating)
      - positive pair density among (leaf,dir) within that bucket (how imbalanced potential BCE would be)
  - reports fruit-level variety (union_max distribution) among supervised fruits.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _mkdir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
  import json

  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _quantiles(arr: np.ndarray, qs: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
  if arr.size == 0:
    return {f"q{int(q*100):02d}": float("nan") for q in qs}
  vals = np.quantile(arr, list(qs))
  out: Dict[str, float] = {}
  for q, v in zip(qs, vals):
    out[f"q{int(q*100):02d}"] = float(v)
  return out


def _summary(arr: np.ndarray) -> Dict[str, float]:
  if arr.size == 0:
    return {"count": 0}
  return {
    "count": int(arr.size),
    "mean": float(arr.mean()),
    "std": float(arr.std()),
    "min": float(arr.min()),
    "max": float(arr.max()),
    **_quantiles(arr),
  }


def _drop_nodes_by_semantic_id(
  nodes: List[Dict[str, Any]],
  edges: List[Tuple[int, int, str]],
  drop_sem_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
  """Filter nodes/edges and remap indices to 0..N'-1."""
  drop_set = set(int(x) for x in drop_sem_ids)
  keep_old: List[int] = []
  for i, n in enumerate(nodes):
    if int(n.get("semantic_id", -1)) in drop_set:
      continue
    keep_old.append(int(i))
  old_to_new = {old: new for new, old in enumerate(keep_old)}
  nodes_new = [nodes[old] for old in keep_old]
  edges_new: List[Tuple[int, int, str]] = []
  for u, v, lab in edges:
    u = int(u)
    v = int(v)
    if u not in old_to_new or v not in old_to_new:
      continue
    edges_new.append((old_to_new[u], old_to_new[v], lab))
  return nodes_new, edges_new


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analyze variety of occlusion supervision (fruit/direction buckets and positives).")
  p.add_argument("--data_root", required=True, help="Dataset root containing data/*_graph.json and pointcloud npz")
  p.add_argument("--split", default="val", choices=["train", "val", "all"], help="Dataset split")
  p.add_argument("--max_scenes", type=int, default=200, help="How many scenes to analyze (default: 200)")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes when max_scenes is set")
  p.add_argument("--seed", type=int, default=0, help="Random seed")
  p.add_argument("--samples_per_scene", type=int, default=3, help="How many subgraph samples per scene")

  # Subgraph sampling config.
  p.add_argument("--subgraph_size", type=int, default=288)
  p.add_argument("--subgraph_size_min", type=int, default=None)
  p.add_argument("--hard_pair_constraint", action="store_true", default=True)
  p.add_argument("--no_hard_pair_constraint", action="store_false", dest="hard_pair_constraint")
  p.add_argument("--drop_shoots", action="store_true", help="Approximate train_oracle --drop_shoots for JSON sampling")

  # Occl-aug config.
  p.add_argument("--occl_aug_enabled", action="store_true")
  p.add_argument("--occl_sidecar_dir", default=None)
  p.add_argument("--occl_aug_num_fruits", type=int, default=4)
  p.add_argument("--occl_aug_union_tau", type=float, default=0.5)
  p.add_argument("--occl_aug_resample_tries", type=int, default=5)
  p.add_argument("--occl_aug_max_dist", type=float, default=0.2)
  p.add_argument("--occl_aug_max_pos_leaves", type=int, default=16)
  p.add_argument("--occl_aug_max_neg_leaves", type=int, default=8)
  p.add_argument("--occl_aug_max_total_leaves", type=int, default=96)
  p.add_argument("--occl_aug_reserve_nodes", type=int, default=64)

  # Extra occlusion supervision selection (same as dataset).
  p.add_argument("--occl_supervise_max_fruits_per_scene", type=int, default=12)
  p.add_argument("--occl_supervise_remain_rate_min", type=float, default=0.9)
  p.add_argument("--occl_supervise_min_dom_dirs", type=int, default=1)

  # Leaf-set cap (occl_pair_index cap).
  p.add_argument("--max_leaves_per_fruit", type=int, default=64)

  # Thresholds.
  p.add_argument("--union_tau", type=float, default=0.5, help="High/dominant union threshold")
  p.add_argument("--union_tau_low", type=float, default=0.25, help="Low union threshold for bucketing (<=low, mid in between)")
  p.add_argument("--pot_eps", type=float, default=0.05, help="Positive leaf threshold for potential")
  p.add_argument("--actionable_eps", type=float, default=0.15, help="Higher threshold for actionable leaf stats")

  p.add_argument("--out_dir", default=None, help="Output dir (default: analysis/occl_supervision_variety_<ts>)")
  return p.parse_args()


def main() -> int:
  args = parse_args()
  random.seed(args.seed)
  rng = random.Random(args.seed)
  np_rng = np.random.default_rng(args.seed)

  # Ensure repo root on path.
  repo_root = Path(__file__).resolve().parents[2]
  sys.path.append(str(repo_root))

  from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset

  ds = OraclePepperDataset(
    data_root=args.data_root,
    split=args.split,
    subgraph_enabled=True,
    subgraph_size=args.subgraph_size,
    subgraph_size_min=args.subgraph_size_min,
    use_4mm=True,
    hard_pair_constraint=args.hard_pair_constraint,
    occl_aug_enabled=args.occl_aug_enabled,
    occl_sidecar_dir=args.occl_sidecar_dir,
    occl_aug_num_fruits=args.occl_aug_num_fruits,
    occl_aug_union_tau=args.occl_aug_union_tau,
    occl_aug_resample_tries=args.occl_aug_resample_tries,
    occl_aug_max_dist=args.occl_aug_max_dist,
    occl_aug_max_pos_leaves=args.occl_aug_max_pos_leaves,
    occl_aug_max_neg_leaves=args.occl_aug_max_neg_leaves,
    occl_aug_max_total_leaves=args.occl_aug_max_total_leaves,
    occl_aug_reserve_nodes=args.occl_aug_reserve_nodes,
  )

  scene_indices = list(ds.indices)
  if args.split == "all":
    scene_indices = list(range(len(ds.pc_files)))
  if args.max_scenes is not None and args.max_scenes < len(scene_indices):
    if args.sample_random:
      scene_indices = rng.sample(scene_indices, k=args.max_scenes)
    else:
      scene_indices = scene_indices[: args.max_scenes]

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"occl_supervision_variety_{ts}"
  out_dir = out_dir.resolve()
  _mkdir(out_dir)

  # Per fruit aggregates.
  fruit_union_max: List[float] = []
  fruit_union_mean: List[float] = []
  fruit_n_dirs_high: List[int] = []
  fruit_n_dirs_mid: List[int] = []
  fruit_n_dirs_low: List[int] = []
  fruit_n_leaves_selected: List[int] = []

  # Global direction-bucket counters.
  buckets = ("low", "mid", "high")
  dir_count: Dict[str, int] = {b: 0 for b in buckets}
  dir_has_pos: Dict[str, int] = {b: 0 for b in buckets}
  dir_has_act: Dict[str, int] = {b: 0 for b in buckets}
  dir_has_mass: Dict[str, int] = {b: 0 for b in buckets}

  # Pair-density counters (how imbalanced potential BCE would be if trained on that bucket).
  pair_total: Dict[str, int] = {b: 0 for b in buckets}
  pair_pos: Dict[str, int] = {b: 0 for b in buckets}
  pair_act: Dict[str, int] = {b: 0 for b in buckets}

  # Scene/sample counters.
  n_scenes_used = 0
  n_json_missing = 0
  n_sidecar_missing = 0
  n_supervised_fruits = 0

  for scene_idx in scene_indices:
    pc_filename = ds.pc_files[scene_idx]
    scene_id = pc_filename.split("_")[0] + "_" + pc_filename.split("_")[1]
    json_path = Path(ds.data_dir) / f"{scene_id}_graph.json"
    if not json_path.exists():
      n_json_missing += 1
      continue
    sidecar = ds._load_occl_sidecar(scene_id)
    if sidecar is None:
      n_sidecar_missing += 1
      continue

    # Load JSON.
    nodes, edges, _ = ds.sampler.load_json_graph(str(json_path))
    if args.drop_shoots:
      nodes, edges = _drop_nodes_by_semantic_id(nodes, edges, drop_sem_ids=[5])
    if not nodes:
      continue

    # Load pointcloud and compute point-derived centroids + size proxy per instance.
    pc_path = Path(ds.data_dir) / pc_filename
    try:
      z = np.load(str(pc_path))
      pts = z["points"].astype(np.float32, copy=False)
      sem = z["semantics"].astype(np.int32, copy=False)
      inst = z["instances"].astype(np.int32, copy=False)
    except Exception:
      continue
    mask = inst > 0
    if args.drop_shoots:
      mask = mask & (sem != 5)  # semantics in npz are 1-based
    pts = pts[mask]
    inst = inst[mask]
    if pts.shape[0] == 0:
      continue
    order_inst = np.argsort(inst)
    inst_s = inst[order_inst]
    pts_s = pts[order_inst]
    uniq_i, starts_i, counts_i = np.unique(inst_s, return_index=True, return_counts=True)
    centroid_pc: Dict[int, np.ndarray] = {}
    radius_pc: Dict[int, float] = {}
    for iid, s0, c0 in zip(uniq_i.tolist(), starts_i.tolist(), counts_i.tolist()):
      seg = pts_s[s0 : s0 + c0]
      if seg.shape[0] == 0:
        continue
      c = seg.mean(axis=0)
      dif = seg - c[None, :]
      r = float(np.sqrt((dif * dif).sum(axis=1)).max()) if seg.shape[0] > 0 else 0.0
      centroid_pc[int(iid)] = c.astype(np.float32, copy=False)
      radius_pc[int(iid)] = float(r)

    # Sidecar union mapping.
    fruit_iid_arr = sidecar.get("fruit_iid", np.asarray([], dtype=np.int32))
    fruit_union_arr = sidecar.get("fruit_union", None)
    if fruit_union_arr is None or fruit_iid_arr.size == 0:
      continue
    fruit_iid_to_row = {int(iid): int(r) for r, iid in enumerate(fruit_iid_arr.tolist())}

    # Pair tensors.
    pf = sidecar.get("pair_fruit_iid", np.asarray([], dtype=np.int32))
    pl = sidecar.get("pair_leaf_iid", np.asarray([], dtype=np.int32))
    pot_all = sidecar.get("pair_potential", None)
    exc_all = sidecar.get("pair_exclusive", None)
    k_rank_all = sidecar.get("pair_k_rank", None)
    if pot_all is None or k_rank_all is None or pf.size == 0:
      continue
    k_sum_all = k_rank_all.sum(axis=-1).astype(np.float32, copy=False)  # [R,D]
    if exc_all is None:
      exc_all = k_rank_all[:, :, 0].astype(np.float32, copy=False)

    # Sort pairs by fruit for fast slicing.
    order = np.argsort(pf)
    pf_s = pf[order]
    pl_s = pl[order]
    pot_s = pot_all[order].astype(np.float32, copy=False)
    exc_s = exc_all[order].astype(np.float32, copy=False)
    k_sum_s = k_sum_all[order].astype(np.float32, copy=False)
    uniq_f, starts, counts = np.unique(pf_s, return_index=True, return_counts=True)
    fruit_to_slice = {int(i): (int(s), int(c)) for i, s, c in zip(uniq_f.tolist(), starts.tolist(), counts.tolist())}

    for _ in range(int(args.samples_per_scene)):
      target_size = int(args.subgraph_size)
      if args.subgraph_size_min is not None and args.subgraph_size_min < args.subgraph_size:
        target_size = int(np_rng.integers(args.subgraph_size_min, args.subgraph_size + 1))

      core_target = int(target_size)
      if args.occl_aug_enabled and int(args.occl_aug_reserve_nodes) > 0:
        core_target = max(1, int(target_size) - int(args.occl_aug_reserve_nodes))

      selected_base = ds.sampler.bfs_sample(nodes, edges, target_size=core_target, seed_policy="stem_bias")
      selected = selected_base
      debug: Dict[str, Any] = {}
      if args.occl_aug_enabled:
        selected = ds._augment_subgraph_with_occlusion(scene_id, nodes, edges, selected_base, target_size=target_size, debug=debug)

      # Collect leaf+fruit instance IDs (present in this sample).
      leaf_iids: List[int] = []
      fruit_iids: List[int] = []
      for idx in selected:
        n = nodes[int(idx)]
        iid = n.get("instance_id")
        if iid is None:
          continue
        sid = int(n.get("semantic_id", -1))
        if sid == 2:
          leaf_iids.append(int(iid))
        elif sid == 4:
          fruit_iids.append(int(iid))

      if not fruit_iids or not leaf_iids:
        continue

      # Keep only iids with point-derived centroids.
      leaf_iids_kept: List[int] = []
      leaf_centroids: List[np.ndarray] = []
      leaf_sizes: List[float] = []
      for liid in leaf_iids:
        c = centroid_pc.get(int(liid))
        if c is None:
          continue
        leaf_iids_kept.append(int(liid))
        leaf_centroids.append(c)
        leaf_sizes.append(float(radius_pc.get(int(liid), 0.0)))
      if not leaf_iids_kept:
        continue
      fruit_centroid_by_iid: Dict[int, np.ndarray] = {}
      for fiid in fruit_iids:
        c = centroid_pc.get(int(fiid))
        if c is None:
          continue
        fruit_centroid_by_iid[int(fiid)] = c
      if not fruit_centroid_by_iid:
        continue

      leaf_iids_present: Set[int] = set(int(x) for x in leaf_iids_kept)
      leaf_centroids_a = np.stack(leaf_centroids, axis=0)  # [L,3]
      leaf_sizes_a = np.asarray(leaf_sizes, dtype=np.float32)  # [L]
      leaf_iids_a = np.asarray(leaf_iids_kept, dtype=np.int64)  # [L]

      # Focus fruits: prefer occl-aug focus list, else random fruits present.
      focus_fruit_iids: List[int] = []
      if args.occl_aug_enabled and debug.get("focus_fruits"):
        for jidx in debug.get("focus_fruits", []) or []:
          try:
            focus_fruit_iids.append(int(nodes[int(jidx)]["instance_id"]))
          except Exception:
            continue
      else:
        k = min(max(int(args.occl_aug_num_fruits), 1), len(fruit_iids))
        focus_fruit_iids = rng.sample(fruit_iids, k=k)

      # Filter focus fruits to those that exist in sidecar.
      fruit_iids_used: List[int] = []
      for iid in focus_fruit_iids:
        if int(iid) not in fruit_iid_to_row:
          continue
        if int(iid) not in fruit_centroid_by_iid:
          continue
        fruit_iids_used.append(int(iid))
      if not fruit_iids_used:
        continue

      # Optional extra supervised fruits (same policy as dataset).
      if int(args.occl_supervise_max_fruits_per_scene) > 0:
        max_total = int(args.occl_supervise_max_fruits_per_scene)
        tau = float(args.union_tau)
        min_dom_dirs = max(int(args.occl_supervise_min_dom_dirs), 1)
        remain_min = float(args.occl_supervise_remain_rate_min)
        used_set = set(int(x) for x in fruit_iids_used)

        def _present_union_for_fruit_iid(fiid: int) -> np.ndarray:
          sl = fruit_to_slice.get(int(fiid))
          if sl is None:
            return np.zeros((int(fruit_union_arr.shape[1]),), dtype=np.float32)
          s0, c0 = sl
          leaf_i = pl_s[s0 : s0 + c0]
          keep = np.isin(leaf_i, list(leaf_iids_present))
          if not bool(np.any(keep)):
            return np.zeros((int(fruit_union_arr.shape[1]),), dtype=np.float32)
          u = exc_s[s0 : s0 + c0][keep].sum(axis=0)
          return np.clip(u, 0.0, 1.0).astype(np.float32, copy=False)

        cand_scored: List[Tuple[float, int]] = []
        for fiid in fruit_iids:
          fiid = int(fiid)
          if fiid in used_set:
            continue
          if fiid not in fruit_iid_to_row or fiid not in fruit_centroid_by_iid:
            continue
          row = fruit_iid_to_row.get(fiid)
          if row is None:
            continue
          uv = fruit_union_arr[row].astype(np.float32, copy=False)
          if float(uv.max()) < tau:
            continue
          dom_mask = (uv >= tau)
          if int(dom_mask.sum()) < int(min_dom_dirs):
            continue
          u_present = _present_union_for_fruit_iid(fiid)
          remain_rate = float((u_present[dom_mask] >= tau).mean()) if bool(dom_mask.any()) else 0.0
          if remain_rate < remain_min:
            continue
          cand_scored.append((float(uv.max()), fiid))

        cand_scored.sort(key=lambda t: t[0], reverse=True)
        for _, fiid in cand_scored:
          if len(fruit_iids_used) >= max_total:
            break
          fruit_iids_used.append(int(fiid))

      cap = max(int(args.max_leaves_per_fruit), 1)

      # Per supervised fruit: bucket directions and count positives among selected leaves.
      for fiid in fruit_iids_used:
        fiid = int(fiid)
        row = fruit_iid_to_row.get(fiid)
        if row is None:
          continue
        fruit_c = fruit_centroid_by_iid.get(fiid)
        if fruit_c is None:
          continue
        union_gt = fruit_union_arr[row].astype(np.float32, copy=False)  # [D]
        D = int(union_gt.shape[0])

        # Candidate leaf pool: leaves within max_dist (in sampled subgraph).
        d = np.linalg.norm(leaf_centroids_a - fruit_c[None, :], axis=1)
        cand_mask = (d <= float(args.occl_aug_max_dist))
        cand_iids = leaf_iids_a[cand_mask]
        if cand_iids.size == 0:
          continue

        # Select up to cap leaves (geom-only stratified; matches dataset default).
        if int(cand_iids.size) <= cap:
          picked = [int(x) for x in cand_iids.tolist()]
        else:
          try:
            scene_num = int(str(scene_id).split("_")[1])
          except Exception:
            scene_num = 0
          seed = (int(scene_num) * 1000003 + int(fiid)) & 0xFFFFFFFF
          rng_np = np.random.default_rng(seed)
          d_cand = d[cand_mask].astype(np.float32, copy=False)
          sizes_cand = leaf_sizes_a[cand_mask].astype(np.float32, copy=False)
          score = sizes_cand / (d_cand + 1e-3)
          order = np.argsort(-score)  # indices into cand
          n = int(cand_iids.shape[0])
          k_total = min(int(cap), n)
          k_low = max(1, int(round(0.125 * float(k_total))))
          k_mid = k_low
          k_top = k_total - k_mid - k_low
          if k_top < 1:
            k_top = max(1, k_total - 2)
            k_mid = max(1, (k_total - k_top) // 2)
            k_low = k_total - k_top - k_mid
          picked_set: Set[int] = set()
          picked = []
          for oi in order[:k_top].tolist():
            liid = int(cand_iids[int(oi)])
            picked.append(liid)
            picked_set.add(liid)
          mid_lo = int(0.25 * float(n))
          mid_hi = int(0.75 * float(n))
          mid_pool = [int(cand_iids[int(oi)]) for oi in order[mid_lo:mid_hi].tolist() if int(cand_iids[int(oi)]) not in picked_set]
          if mid_pool and k_mid > 0:
            take = min(k_mid, len(mid_pool))
            mid_sel = rng_np.choice(np.asarray(mid_pool, dtype=np.int64), size=take, replace=False)
            for liid in mid_sel.tolist():
              picked.append(int(liid))
              picked_set.add(int(liid))
          low_pool = [int(cand_iids[int(oi)]) for oi in order[mid_hi:].tolist() if int(cand_iids[int(oi)]) not in picked_set]
          if low_pool and k_low > 0:
            take = min(k_low, len(low_pool))
            low_sel = rng_np.choice(np.asarray(low_pool, dtype=np.int64), size=take, replace=False)
            for liid in low_sel.tolist():
              picked.append(int(liid))
              picked_set.add(int(liid))
          if len(picked) < k_total:
            rem = [int(x) for x in cand_iids.tolist() if int(x) not in picked_set]
            if rem:
              take = min(k_total - len(picked), len(rem))
              fill_sel = rng_np.choice(np.asarray(rem, dtype=np.int64), size=take, replace=False)
              for liid in fill_sel.tolist():
                picked.append(int(liid))
                picked_set.add(int(liid))

        L_sel = int(len(picked))
        if L_sel == 0:
          continue

        # Materialize pot/k_sum for selected leaves (include zeros for leaves absent in sidecar).
        pot_sel = np.zeros((L_sel, D), dtype=np.float32)
        k_sum_sel = np.zeros((L_sel, D), dtype=np.float32)
        sl = fruit_to_slice.get(fiid)
        if sl is not None:
          s0, c0 = sl
          leaf_i = pl_s[s0 : s0 + c0].astype(np.int64, copy=False)
          pot_f = pot_s[s0 : s0 + c0].astype(np.float32, copy=False)
          k_f = k_sum_s[s0 : s0 + c0].astype(np.float32, copy=False)
          # Leaf iid -> local row in slice.
          leaf_to_local = {int(li): int(j) for j, li in enumerate(leaf_i.tolist())}
          for j, liid in enumerate(picked):
            r = leaf_to_local.get(int(liid))
            if r is None:
              continue
            pot_sel[j] = pot_f[r]
            k_sum_sel[j] = k_f[r]

        # Buckets on union.
        tau_hi = float(args.union_tau)
        tau_lo = float(args.union_tau_low)
        m_high = union_gt >= tau_hi
        m_low = union_gt <= tau_lo
        m_mid = (~m_high) & (~m_low)

        # Per-fruit stats.
        fruit_union_max.append(float(union_gt.max()))
        fruit_union_mean.append(float(union_gt.mean()))
        fruit_n_dirs_high.append(int(m_high.sum()))
        fruit_n_dirs_mid.append(int(m_mid.sum()))
        fruit_n_dirs_low.append(int(m_low.sum()))
        fruit_n_leaves_selected.append(int(L_sel))

        # Leaf positives and rank-mass per direction.
        pos_mask = pot_sel >= float(args.pot_eps)
        act_mask = pot_sel >= float(args.actionable_eps)
        pos_count_dir = pos_mask.sum(axis=0)  # [D]
        act_count_dir = act_mask.sum(axis=0)  # [D]

        mass = k_sum_sel.copy()
        mass[~pos_mask] = 0.0  # match training gating for rank mass
        sum_mass_dir = mass.sum(axis=0)  # [D]

        def _acc(bucket_mask: np.ndarray, name: str) -> None:
          n_dir = int(bucket_mask.sum())
          if n_dir <= 0:
            return
          dir_count[name] += n_dir
          has_p = (pos_count_dir[bucket_mask] > 0).sum()
          has_a = (act_count_dir[bucket_mask] > 0).sum()
          has_m = (sum_mass_dir[bucket_mask] > 0).sum()
          dir_has_pos[name] += int(has_p)
          dir_has_act[name] += int(has_a)
          dir_has_mass[name] += int(has_m)
          pair_total[name] += int(L_sel * n_dir)
          pair_pos[name] += int(pos_mask[:, bucket_mask].sum())
          pair_act[name] += int(act_mask[:, bucket_mask].sum())

        _acc(m_low, "low")
        _acc(m_mid, "mid")
        _acc(m_high, "high")

        n_supervised_fruits += 1

    n_scenes_used += 1

  payload: Dict[str, Any] = {
    "args": vars(args),
    "scenes_used": int(n_scenes_used),
    "json_missing": int(n_json_missing),
    "sidecar_missing": int(n_sidecar_missing),
    "supervised_fruits_count": int(n_supervised_fruits),
    "fruit_union_max": _summary(np.asarray(fruit_union_max, dtype=np.float32)),
    "fruit_union_mean": _summary(np.asarray(fruit_union_mean, dtype=np.float32)),
    "fruit_n_dirs_high": _summary(np.asarray(fruit_n_dirs_high, dtype=np.float32)),
    "fruit_n_dirs_mid": _summary(np.asarray(fruit_n_dirs_mid, dtype=np.float32)),
    "fruit_n_dirs_low": _summary(np.asarray(fruit_n_dirs_low, dtype=np.float32)),
    "fruit_n_leaves_selected": _summary(np.asarray(fruit_n_leaves_selected, dtype=np.float32)),
    "dir_bucket_counts": dir_count,
    "dir_bucket_has_pos": dir_has_pos,
    "dir_bucket_has_act": dir_has_act,
    "dir_bucket_has_mass": dir_has_mass,
    "pair_bucket_total": pair_total,
    "pair_bucket_pos": pair_pos,
    "pair_bucket_act": pair_act,
  }

  # Derived rates (easier to interpret).
  rates: Dict[str, Any] = {}
  for b in buckets:
    n_dir = float(dir_count[b])
    rates[f"{b}_dir_has_pos_rate"] = float(dir_has_pos[b]) / n_dir if n_dir > 0 else float("nan")
    rates[f"{b}_dir_has_act_rate"] = float(dir_has_act[b]) / n_dir if n_dir > 0 else float("nan")
    rates[f"{b}_dir_has_mass_rate"] = float(dir_has_mass[b]) / n_dir if n_dir > 0 else float("nan")
    n_pair = float(pair_total[b])
    rates[f"{b}_pair_pos_rate"] = float(pair_pos[b]) / n_pair if n_pair > 0 else float("nan")
    rates[f"{b}_pair_act_rate"] = float(pair_act[b]) / n_pair if n_pair > 0 else float("nan")
  payload["rates"] = rates

  _save_json(out_dir / "summary.json", payload)
  print(f"Wrote: {out_dir / 'summary.json'}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

