#!/usr/bin/env python3
"""Analyze bias/GT-leak risk of occlusion leaf-set construction for the SA/attention blocks.

In the oracle pointcloud pipeline, occlusion supervision is materialized as:
  - occl_fruit_idx / occl_fruit_union  (per supervised fruit, per direction)
  - occl_pair_index / occl_pair_*      (per supervised fruit-leaf pair, per direction)

The self-attention + direction-conditioned scorer only "sees" leaves that appear in occl_pair_index.
To keep memory bounded, occl_pair_index is capped to `occl_targets_max_leaves_per_fruit` leaves per fruit.

Legacy dataset policy (OraclePepperDataset, `--occl_leafset_policy sidecar_priority`):
  1) Build candidate pool: leaves within `occl_aug_max_dist` of the fruit centroid, *within the sampled subgraph*.
  2) Priority include occl-aug leaves (pos then neg), then randomly fill to the cap.

New default dataset policy (OraclePepperDataset, `--occl_leafset_policy geom_stratified`):
  1) Build candidate pool: leaves within `occl_aug_max_dist` of the fruit centroid, *within the sampled subgraph*.
  2) If cand>cap: pick leaves via a geometry-only stratified heuristic (top/mid/low by size/distance score).

This script compares these policies against a "pure random" policy:
  - sample up to K leaves uniformly from the candidate pool (no GT-driven priority)

We report how much these policies differ in terms of:
  - candidate set overlap (Jaccard)
  - union explainability under the selected leaf set (dominant dirs)
  - dominant-direction remain-rate under the selected leaf set (recall ceiling proxy)
  - density of positive supervision among selected leaf-dir pairs (pot>=eps)
  - optional k3 (top-3) mass coverage under the selected leaf set

This helps decide whether the current policy is "just sampling" (little effect)
or a significant label-aware selection that may not match inference-time behavior.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
try:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: E402
except Exception:
  plt = None


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


def _plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 100, range_: Optional[Tuple[float, float]] = None) -> None:
  if plt is None:
    return
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(1, 1, 1)
  if range_ is None:
    ax.hist(values, bins=bins, edgecolor="none")
  else:
    ax.hist(values, bins=bins, range=range_, edgecolor="none")
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("count")
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


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
  p = argparse.ArgumentParser(description="Analyze bias of occlusion leaf-set selection policy (sidecar-priority vs geom-stratified vs random).")
  p.add_argument("--data_root", required=True, help="Dataset root containing data/*.json")
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

  # Occl-aug config (needed to reproduce the "priority leaves" set).
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

  # Occlusion supervision selection (extra fruits).
  p.add_argument("--occl_supervise_max_fruits_per_scene", type=int, default=12)
  p.add_argument("--occl_supervise_remain_rate_min", type=float, default=0.9)
  p.add_argument("--occl_supervise_min_dom_dirs", type=int, default=1)

  # Leaf-set cap (this is what can create bias).
  p.add_argument("--max_leaves_per_fruit", type=int, default=64)

  # Thresholds for stats.
  p.add_argument("--union_tau", type=float, default=0.5)
  p.add_argument("--pot_eps", type=float, default=0.05)
  p.add_argument("--actionable_eps", type=float, default=0.15)

  p.add_argument("--out_dir", default=None, help="Output dir (default: analysis/occl_leafset_bias_<ts>)")
  return p.parse_args()


def _jaccard(a: Set[int], b: Set[int]) -> float:
  if not a and not b:
    return 1.0
  if not a or not b:
    return 0.0
  inter = len(a & b)
  uni = len(a | b)
  return float(inter) / float(uni) if uni > 0 else 0.0


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
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"occl_leafset_bias_{ts}"
  out_dir = out_dir.resolve()
  plots_dir = out_dir / "plots"
  _mkdir(plots_dir)

  # Aggregates (per supervised fruit).
  cand_sizes: List[int] = []
  picked_sizes_priority: List[int] = []
  picked_sizes_geom: List[int] = []
  picked_sizes_random: List[int] = []
  frac_cand_gt_cap: List[float] = []
  jaccard_priority_vs_random: List[float] = []
  jaccard_geom_vs_random: List[float] = []
  jaccard_geom_vs_priority: List[float] = []

  # Explainability on dominant directions (selected leaf sets only).
  explain_frac_dom_priority: List[float] = []
  explain_frac_dom_geom: List[float] = []
  explain_frac_dom_random: List[float] = []
  remain_dom_priority: List[float] = []
  remain_dom_geom: List[float] = []
  remain_dom_random: List[float] = []

  # K3 coverage (selected leaves only) on dominant directions.
  k3_cov_dom_priority: List[float] = []
  k3_cov_dom_geom: List[float] = []
  k3_cov_dom_random: List[float] = []

  # Positive density among selected leaf-dir pairs on dominant directions.
  pos_pair_rate_dom_priority: List[float] = []
  pos_pair_rate_dom_geom: List[float] = []
  pos_pair_rate_dom_random: List[float] = []
  act_pair_rate_dom_priority: List[float] = []
  act_pair_rate_dom_geom: List[float] = []
  act_pair_rate_dom_random: List[float] = []

  # How much "priority" comes from GT-derived aug leaves.
  frac_selected_from_aug_pos: List[float] = []
  frac_selected_from_aug_any: List[float] = []

  n_scenes_used = 0
  n_json_missing = 0
  n_sidecar_missing = 0
  n_fruits_total = 0

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

    # Point-derived centroids and size proxy (matches dataset "deployable" geometry-only leafset selection):
    # centroid_pc = mean(points), radius_pc = max distance from centroid.
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

    nodes, edges, _ = ds.sampler.load_json_graph(str(json_path))
    if args.drop_shoots:
      nodes, edges = _drop_nodes_by_semantic_id(nodes, edges, drop_sem_ids=[5])
    if not nodes:
      continue

    # Fruit union mapping.
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

      # Collect leaf+fruit instance IDs from the selected JSON nodes.
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

      # Focus fruits (match OraclePepperDataset): prefer occl-aug focus list, else random fruits present.
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
          s, c = sl
          leaf_i = pl_s[s : s + c]
          keep = np.isin(leaf_i, list(leaf_iids_present))
          if not bool(np.any(keep)):
            return np.zeros((int(fruit_union_arr.shape[1]),), dtype=np.float32)
          u = exc_s[s : s + c][keep].sum(axis=0)
          return np.clip(u, 0.0, 1.0).astype(np.float32, copy=False)

        cand_scored: List[Tuple[float, int]] = []
        for fiid in fruit_iids:
          fiid = int(fiid)
          if fiid in used_set:
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

      # Priority leaves from occl-aug (instance ids).
      aug_pos_iids: List[int] = []
      aug_neg_iids: List[int] = []
      if args.occl_aug_enabled and debug:
        seen: Set[int] = set()
        for jidx in debug.get("pos_leaves", []) or []:
          try:
            liid = int(nodes[int(jidx)]["instance_id"])
          except Exception:
            continue
          if liid <= 0 or liid in seen:
            continue
          seen.add(liid)
          aug_pos_iids.append(liid)
        for jidx in debug.get("neg_leaves", []) or []:
          try:
            liid = int(nodes[int(jidx)]["instance_id"])
          except Exception:
            continue
          if liid <= 0 or liid in seen:
            continue
          seen.add(liid)
          aug_neg_iids.append(liid)

      # Per supervised fruit: build candidate pool, then compare leaf-set selection policies.
      for fiid in fruit_iids_used:
        fiid = int(fiid)
        row = fruit_iid_to_row.get(fiid)
        if row is None:
          continue
        fruit_c = fruit_centroid_by_iid.get(fiid)
        if fruit_c is None:
          continue

        # Candidate pool: all leaves within max_dist (in the sampled subgraph).
        d = np.linalg.norm(leaf_centroids_a - fruit_c[None, :], axis=1)
        cand_mask = (d <= float(args.occl_aug_max_dist))
        cand = leaf_iids_a[cand_mask]
        if cand.size == 0:
          continue
        cand_set: Set[int] = set(int(x) for x in cand.tolist())
        cand_sizes.append(int(cand.size))
        frac_cand_gt_cap.append(float(int(cand.size) > int(args.max_leaves_per_fruit)))

        # Policy A: priority (current dataset policy).
        picked_p: List[int] = []
        picked_p_set: Set[int] = set()
        cap = max(int(args.max_leaves_per_fruit), 1)

        def _try_add(iid: int) -> None:
          if iid in picked_p_set:
            return
          if iid not in cand_set:
            return
          picked_p_set.add(iid)
          picked_p.append(iid)

        for liid in aug_pos_iids:
          if len(picked_p) >= cap:
            break
          _try_add(int(liid))
        for liid in aug_neg_iids:
          if len(picked_p) >= cap:
            break
          _try_add(int(liid))
        if len(picked_p) < cap:
          rem = [int(x) for x in cand_set if int(x) not in picked_p_set]
          if rem:
            k = min(cap - len(picked_p), len(rem))
            picked_p.extend(rng.sample(rem, k=k))
            picked_p_set = set(picked_p)

        # Policy B: pure random.
        k = min(cap, int(cand.size))
        picked_r = rng.sample(list(cand_set), k=k) if k > 0 else []
        picked_r_set = set(int(x) for x in picked_r)

        # Policy C: geometry-only stratified (matches OraclePepperDataset default).
        picked_g: List[int] = []
        picked_g_set: Set[int] = set()
        if int(cand.size) <= cap:
          picked_g = [int(x) for x in cand.tolist()]
          picked_g_set = set(picked_g)
        else:
          # Stable per (scene, fruit) seed (same as dataset).
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

          n = int(cand.shape[0])
          k_total = min(int(cap), n)
          k_low = max(1, int(round(0.125 * float(k_total))))
          k_mid = k_low
          k_top = k_total - k_mid - k_low
          if k_top < 1:
            k_top = max(1, k_total - 2)
            k_mid = max(1, (k_total - k_top) // 2)
            k_low = k_total - k_top - k_mid

          # 1) top (deterministic)
          for oi in order[:k_top].tolist():
            liid = int(cand[int(oi)])
            picked_g.append(liid)
            picked_g_set.add(liid)

          # 2) mid (sample from 25-75% ranks)
          mid_lo = int(0.25 * float(n))
          mid_hi = int(0.75 * float(n))
          mid_pool = [int(cand[int(oi)]) for oi in order[mid_lo:mid_hi].tolist() if int(cand[int(oi)]) not in picked_g_set]
          if mid_pool and k_mid > 0:
            take = min(k_mid, len(mid_pool))
            mid_sel = rng_np.choice(np.asarray(mid_pool, dtype=np.int64), size=take, replace=False)
            for liid in mid_sel.tolist():
              picked_g.append(int(liid))
              picked_g_set.add(int(liid))

          # 3) low (sample from lowest quartile)
          low_pool = [int(cand[int(oi)]) for oi in order[mid_hi:].tolist() if int(cand[int(oi)]) not in picked_g_set]
          if low_pool and k_low > 0:
            take = min(k_low, len(low_pool))
            low_sel = rng_np.choice(np.asarray(low_pool, dtype=np.int64), size=take, replace=False)
            for liid in low_sel.tolist():
              picked_g.append(int(liid))
              picked_g_set.add(int(liid))

          # 4) fill remainder uniformly
          if len(picked_g) < k_total:
            rem = [int(x) for x in cand.tolist() if int(x) not in picked_g_set]
            if rem:
              take = min(k_total - len(picked_g), len(rem))
              fill_sel = rng_np.choice(np.asarray(rem, dtype=np.int64), size=take, replace=False)
              for liid in fill_sel.tolist():
                picked_g.append(int(liid))
                picked_g_set.add(int(liid))

        picked_sizes_priority.append(int(len(picked_p_set)))
        picked_sizes_geom.append(int(len(picked_g_set)))
        picked_sizes_random.append(int(len(picked_r_set)))
        jaccard_priority_vs_random.append(_jaccard(picked_p_set, picked_r_set))
        jaccard_geom_vs_random.append(_jaccard(picked_g_set, picked_r_set))
        jaccard_geom_vs_priority.append(_jaccard(picked_g_set, picked_p_set))

        if picked_p_set:
          ap = set(int(x) for x in aug_pos_iids)
          an = set(int(x) for x in aug_neg_iids)
          frac_selected_from_aug_pos.append(float(len(picked_p_set & ap)) / float(len(picked_p_set)))
          frac_selected_from_aug_any.append(float(len(picked_p_set & (ap | an))) / float(len(picked_p_set)))

        # Pull fruit slice from sidecar.
        sl = fruit_to_slice.get(fiid)
        if sl is None:
          continue
        s, c = sl
        leaf_i = pl_s[s : s + c]
        pot = pot_s[s : s + c]
        exc = exc_s[s : s + c]
        k_sum = k_sum_s[s : s + c]

        union_gt = fruit_union_arr[row].astype(np.float32, copy=False)
        dom_mask = (union_gt >= float(args.union_tau))
        if not bool(np.any(dom_mask)):
          # No dominant dirs => leaf losses/metrics are not evaluated; still count fruit.
          n_fruits_total += 1
          continue
        n_dom = int(dom_mask.sum())

        def _metrics_for_sel(sel: Set[int]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
          # Returns: (explain_frac_mean, remain_rate, k3_cov_mean, pos_pair_rate, act_pair_rate, union_present_mean)
          keep = np.isin(leaf_i, list(sel))
          if not bool(np.any(keep)):
            return None, None, None, None, None, None
          exc_sel = exc[keep].sum(axis=0)
          exc_sel = np.clip(exc_sel, 0.0, 1.0).astype(np.float32, copy=False)
          # Explainability fraction (dominant dirs).
          denom = np.clip(union_gt, 1e-6, None)
          frac = np.clip(exc_sel / denom, 0.0, 1.0)
          explain = float(frac[dom_mask].mean()) if n_dom > 0 else None
          remain = float((exc_sel[dom_mask] >= float(args.union_tau)).mean()) if n_dom > 0 else None
          # K3 coverage on dominant dirs.
          k3_sel = k_sum[keep].sum(axis=0)
          k3_tot = k_sum.sum(axis=0)
          k3_cov = np.divide(k3_sel, np.clip(k3_tot, 1e-12, None))
          k3_cov_m = float(k3_cov[dom_mask].mean()) if n_dom > 0 else None
          # Positive density among selected leaf-dir pairs (dominant dirs).
          pot_sel = pot[keep]
          pos_pairs = int((pot_sel[:, dom_mask] >= float(args.pot_eps)).sum())
          act_pairs = int((pot_sel[:, dom_mask] >= float(args.actionable_eps)).sum())
          total_pairs = int(pot_sel.shape[0] * n_dom)
          pos_rate = float(pos_pairs) / float(total_pairs) if total_pairs > 0 else None
          act_rate = float(act_pairs) / float(total_pairs) if total_pairs > 0 else None
          union_present_mean = float(exc_sel[dom_mask].mean()) if n_dom > 0 else None
          return explain, remain, k3_cov_m, pos_rate, act_rate, union_present_mean

        e_p, r_p, k3_p, pos_p, act_p, _ = _metrics_for_sel(picked_p_set)
        e_g, r_g, k3_g, pos_g, act_g, _ = _metrics_for_sel(picked_g_set)
        e_r, r_r, k3_r, pos_r, act_r, _ = _metrics_for_sel(picked_r_set)
        if e_p is not None:
          explain_frac_dom_priority.append(float(e_p))
        if e_g is not None:
          explain_frac_dom_geom.append(float(e_g))
        if e_r is not None:
          explain_frac_dom_random.append(float(e_r))
        if r_p is not None:
          remain_dom_priority.append(float(r_p))
        if r_g is not None:
          remain_dom_geom.append(float(r_g))
        if r_r is not None:
          remain_dom_random.append(float(r_r))
        if k3_p is not None:
          k3_cov_dom_priority.append(float(k3_p))
        if k3_g is not None:
          k3_cov_dom_geom.append(float(k3_g))
        if k3_r is not None:
          k3_cov_dom_random.append(float(k3_r))
        if pos_p is not None:
          pos_pair_rate_dom_priority.append(float(pos_p))
        if pos_g is not None:
          pos_pair_rate_dom_geom.append(float(pos_g))
        if pos_r is not None:
          pos_pair_rate_dom_random.append(float(pos_r))
        if act_p is not None:
          act_pair_rate_dom_priority.append(float(act_p))
        if act_g is not None:
          act_pair_rate_dom_geom.append(float(act_g))
        if act_r is not None:
          act_pair_rate_dom_random.append(float(act_r))

        n_fruits_total += 1

    n_scenes_used += 1

  payload: Dict[str, Any] = {
    "args": vars(args),
    "scenes_used": int(n_scenes_used),
    "json_missing": int(n_json_missing),
    "sidecar_missing": int(n_sidecar_missing),
    "supervised_fruits_count": int(n_fruits_total),
    "cand_size": _summary(np.asarray(cand_sizes, dtype=np.float32)),
    "picked_size_priority": _summary(np.asarray(picked_sizes_priority, dtype=np.float32)),
    "picked_size_geom_stratified": _summary(np.asarray(picked_sizes_geom, dtype=np.float32)),
    "picked_size_random": _summary(np.asarray(picked_sizes_random, dtype=np.float32)),
    "frac_cand_gt_cap": _summary(np.asarray(frac_cand_gt_cap, dtype=np.float32)),
    "jaccard_priority_vs_random": _summary(np.asarray(jaccard_priority_vs_random, dtype=np.float32)),
    "jaccard_geom_vs_random": _summary(np.asarray(jaccard_geom_vs_random, dtype=np.float32)),
    "jaccard_geom_vs_priority": _summary(np.asarray(jaccard_geom_vs_priority, dtype=np.float32)),
    "dominant_explain_frac_priority": _summary(np.asarray(explain_frac_dom_priority, dtype=np.float32)),
    "dominant_explain_frac_geom_stratified": _summary(np.asarray(explain_frac_dom_geom, dtype=np.float32)),
    "dominant_explain_frac_random": _summary(np.asarray(explain_frac_dom_random, dtype=np.float32)),
    "dominant_remain_rate_priority": _summary(np.asarray(remain_dom_priority, dtype=np.float32)),
    "dominant_remain_rate_geom_stratified": _summary(np.asarray(remain_dom_geom, dtype=np.float32)),
    "dominant_remain_rate_random": _summary(np.asarray(remain_dom_random, dtype=np.float32)),
    "k3_cov_dom_priority": _summary(np.asarray(k3_cov_dom_priority, dtype=np.float32)),
    "k3_cov_dom_geom_stratified": _summary(np.asarray(k3_cov_dom_geom, dtype=np.float32)),
    "k3_cov_dom_random": _summary(np.asarray(k3_cov_dom_random, dtype=np.float32)),
    "pos_pair_rate_dom_priority": _summary(np.asarray(pos_pair_rate_dom_priority, dtype=np.float32)),
    "pos_pair_rate_dom_geom_stratified": _summary(np.asarray(pos_pair_rate_dom_geom, dtype=np.float32)),
    "pos_pair_rate_dom_random": _summary(np.asarray(pos_pair_rate_dom_random, dtype=np.float32)),
    "act_pair_rate_dom_priority": _summary(np.asarray(act_pair_rate_dom_priority, dtype=np.float32)),
    "act_pair_rate_dom_geom_stratified": _summary(np.asarray(act_pair_rate_dom_geom, dtype=np.float32)),
    "act_pair_rate_dom_random": _summary(np.asarray(act_pair_rate_dom_random, dtype=np.float32)),
    "frac_selected_from_aug_pos": _summary(np.asarray(frac_selected_from_aug_pos, dtype=np.float32)),
    "frac_selected_from_aug_any": _summary(np.asarray(frac_selected_from_aug_any, dtype=np.float32)),
  }

  _save_json(out_dir / "summary.json", payload)

  # Light plots for quick inspection.
  _plot_hist(np.asarray(cand_sizes, dtype=np.float32), plots_dir / "cand_size.png", "Candidate leaves per fruit (within max_dist)", "cand_size", bins=80)
  _plot_hist(np.asarray(jaccard_priority_vs_random, dtype=np.float32), plots_dir / "jaccard_priority_vs_random.png", "Jaccard(picked_priority, picked_random)", "jaccard", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(jaccard_geom_vs_random, dtype=np.float32), plots_dir / "jaccard_geom_vs_random.png", "Jaccard(picked_geom, picked_random)", "jaccard", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(jaccard_geom_vs_priority, dtype=np.float32), plots_dir / "jaccard_geom_vs_priority.png", "Jaccard(picked_geom, picked_priority)", "jaccard", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(explain_frac_dom_priority, dtype=np.float32), plots_dir / "explain_frac_dom_priority.png", "Explain frac on dominant dirs (priority)", "explain_frac", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(explain_frac_dom_geom, dtype=np.float32), plots_dir / "explain_frac_dom_geom.png", "Explain frac on dominant dirs (geom_stratified)", "explain_frac", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(explain_frac_dom_random, dtype=np.float32), plots_dir / "explain_frac_dom_random.png", "Explain frac on dominant dirs (random)", "explain_frac", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(remain_dom_priority, dtype=np.float32), plots_dir / "remain_dom_priority.png", "Remain-rate on dominant dirs (priority)", "remain_rate", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(remain_dom_geom, dtype=np.float32), plots_dir / "remain_dom_geom.png", "Remain-rate on dominant dirs (geom_stratified)", "remain_rate", bins=80, range_=(0.0, 1.0))
  _plot_hist(np.asarray(remain_dom_random, dtype=np.float32), plots_dir / "remain_dom_random.png", "Remain-rate on dominant dirs (random)", "remain_rate", bins=80, range_=(0.0, 1.0))

  print(f"Wrote: {out_dir / 'summary.json'}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
