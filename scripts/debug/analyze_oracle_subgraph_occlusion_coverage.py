#!/usr/bin/env python3
"""Analyze occlusion coverage under oracle JSON subgraph sampling.

This answers: "If we train on BFS subgraphs, do we even *see* the true occluder leaves?"

We simulate the subgraph sampling step (stem-biased BFS), optionally followed by the
occlusion-augmented injection we added in `OraclePepperDataset`.

Coverage definitions (per fruit, per direction):
  - exclusive coverage: sum(exclusive occlusion mass for leaves present in subgraph) / union
    This is exactly the fraction of union occlusion attributable to leaves present, because
    union is computed from the frontmost occluder and exclusive is an attribution of union.
  - k3 coverage: sum(k-rank(1..3) mass for leaves present) / sum(k-rank(1..3) total mass)
    This measures how much of the "backup occluders" (non-stealing signal) we retain.

We report stats both over:
  - all directions with union > 0
  - "dominant" directions where union >= union_tau (default: 0.5)

Outputs:
  - summary.json with aggregate stats
  - plots/ histograms (if matplotlib is installed)
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


def _hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
  if values.size == 0:
    return np.zeros((bins.size - 1,), dtype=np.int64)
  h, _ = np.histogram(values, bins=bins)
  return h.astype(np.int64, copy=False)


def _plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 100, range_: Tuple[float, float] = (0.0, 1.0)) -> None:
  if plt is None:
    return
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.hist(values, bins=bins, range=range_, edgecolor="none")
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("count")
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)

def _pick_focus_fruits_like_occl_aug(
  nodes: Sequence[Dict[str, Any]],
  selected: Sequence[int],
  *,
  num_fruits: int,
  resample_tries: int,
  union_tau: float,
  fruit_iid_to_row: Dict[int, int],
  fruit_union_arr: Optional[np.ndarray],
) -> List[int]:
  """
  Choose "focus fruits" from a sampled subgraph, mirroring the occl-aug selection logic:
  - draw uniformly at random from fruit nodes already present in the subgraph
  - if union sidecars exist and at least one high-union fruit is available (max_d union>=tau),
    resample a few times to include at least one such fruit

  Returns JSON node indices (not instance IDs).
  """
  FRUIT = 4  # JSON semantic IDs are 1-based.
  fruits_in_sel = [i for i in selected if int(nodes[int(i)].get("semantic_id", -1)) == FRUIT]
  if not fruits_in_sel:
    return []
  k = min(int(num_fruits), len(fruits_in_sel))
  if k <= 0:
    return []

  # Identify which fruit JSON indices correspond to "high union" fruits in the sidecar.
  high_union_set: Set[int] = set()
  if fruit_union_arr is not None:
    for jidx in fruits_in_sel:
      try:
        iid = int(nodes[int(jidx)].get("instance_id", -1))
      except Exception:
        iid = -1
      row = fruit_iid_to_row.get(iid, None) if iid > 0 else None
      if row is None:
        continue
      try:
        u = np.asarray(fruit_union_arr[int(row)], dtype=np.float32)
      except Exception:
        continue
      if bool((u >= float(union_tau)).any()):
        high_union_set.add(int(jidx))

  # If no "high union" fruit exists, fall back to a single random draw.
  if not high_union_set:
    return [int(x) for x in random.sample(fruits_in_sel, k=k)]

  # Guardrail: try a few resamples to include at least one high-union fruit.
  best = None
  for _ in range(int(max(1, resample_tries))):
    cand = [int(x) for x in random.sample(fruits_in_sel, k=k)]
    best = cand
    if any(int(x) in high_union_set for x in cand):
      return cand
  return best if best is not None else [int(x) for x in random.sample(fruits_in_sel, k=k)]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analyze occlusion coverage under oracle subgraph sampling.")
  p.add_argument("--data_root", required=True, help="Dataset root containing data/*.json")
  p.add_argument("--split", default="train", choices=["train", "val", "all"], help="Dataset split (matches OraclePepperDataset)")
  p.add_argument("--max_scenes", type=int, default=200, help="How many scenes to analyze (default: 200)")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes when max_scenes is set")
  p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")

  # Sampling config (match train_oracle defaults).
  p.add_argument("--subgraph_size", type=int, default=256, help="Subgraph size (nodes)")
  p.add_argument("--subgraph_size_min", type=int, default=None, help="Optional min for variable-size sampling")
  p.add_argument("--samples_per_scene", type=int, default=3, help="How many subgraph samples per scene")
  p.add_argument("--hard_pair_constraint", action="store_true", default=True, help="Enforce fruit<->ped hard constraint")
  p.add_argument("--no_hard_pair_constraint", action="store_false", dest="hard_pair_constraint")
  p.add_argument("--drop_shoots", action="store_true", help="Match train_oracle --drop_shoots: remove shoot nodes (semantic_id=5) before sampling")

  # Dominant + coverage thresholds.
  p.add_argument("--union_tau", type=float, default=0.5, help="Dominant-direction threshold on union occlusion")
  p.add_argument("--cov_tau_list", default="0.5,0.9,0.95", help="Comma-separated coverage thresholds to report")

  # Sidecar discovery.
  p.add_argument("--occl_sidecar_dir", default=None, help="Optional sidecar dir (auto-detect if omitted)")

  # Compare baseline BFS vs occl-aug sampling.
  p.add_argument("--occl_aug_enabled", action="store_true", help="Apply occlusion-augmented injection after BFS")
  p.add_argument("--occl_aug_num_fruits", type=int, default=3)
  p.add_argument("--occl_aug_union_tau", type=float, default=0.5)
  p.add_argument("--occl_aug_resample_tries", type=int, default=5)
  p.add_argument("--occl_aug_max_dist", type=float, default=0.2)
  p.add_argument("--occl_aug_max_pos_leaves", type=int, default=16)
  p.add_argument("--occl_aug_max_neg_leaves", type=int, default=8)
  p.add_argument("--occl_aug_max_total_leaves", type=int, default=96)
  p.add_argument("--occl_aug_reserve_nodes", type=int, default=0, help="Reserve nodes for occl-aug (BFS core uses subgraph_size - reserve)")

  # Extra occlusion supervision selection (matches OraclePepperDataset behavior).
  p.add_argument("--occl_supervise_max_fruits_per_scene", type=int, default=0, help="If >0, supervise additional fruits with explainable dominant dirs")
  p.add_argument("--occl_supervise_remain_rate_min", type=float, default=0.9, help="Min remain-rate for extra supervised fruits")
  p.add_argument("--occl_supervise_min_dom_dirs", type=int, default=1, help="Min #dominant dirs (union>=tau) required for extra supervised fruits")
  p.add_argument("--occl_targets_max_leaves_per_fruit", type=int, default=64, help="Max leaves per fruit for occlusion pair supervision (approx)")

  p.add_argument("--out_dir", default=None, help="Output directory (default: analysis/oracle_subgraph_occl_cov_<ts>)")
  return p.parse_args()


def _parse_float_list(s: str) -> List[float]:
  out: List[float] = []
  for part in (s or "").split(","):
    part = part.strip()
    if not part:
      continue
    out.append(float(part))
  return out


def _semantic_counts(nodes: List[Dict[str, Any]], selected_indices: List[int]) -> Dict[str, int]:
  # JSON semantic IDs are 1-based in this dataset version.
  names = {1: "stem", 2: "leaf", 3: "ped", 4: "fruit", 5: "shoot"}
  out = {v: 0 for v in names.values()}
  for idx in selected_indices:
    sid = int(nodes[idx].get("semantic_id", -1))
    name = names.get(sid, f"sem{sid}")
    out[name] = int(out.get(name, 0)) + 1
  out["total"] = int(len(selected_indices))
  return out


def _leaf_fraction_within_focus_zone(
  nodes: List[Dict[str, Any]],
  leaf_indices: List[int],
  focus_fruit_indices: List[int],
  max_dist: float,
) -> float:
  """Fraction of leaf nodes within max_dist of any focus fruit centroid."""
  if not leaf_indices or not focus_fruit_indices:
    return float("nan")
  fruit_centroids: List[np.ndarray] = []
  for fi in focus_fruit_indices:
    c = nodes[int(fi)].get("centroid")
    if c is None:
      continue
    fruit_centroids.append(np.asarray(c, dtype=np.float32))
  if not fruit_centroids:
    return float("nan")
  fruit_c = np.stack(fruit_centroids, axis=0)  # [F,3]

  leaf_centroids: List[np.ndarray] = []
  for li in leaf_indices:
    c = nodes[int(li)].get("centroid")
    if c is None:
      continue
    leaf_centroids.append(np.asarray(c, dtype=np.float32))
  if not leaf_centroids:
    return float("nan")
  leaf_c = np.stack(leaf_centroids, axis=0)  # [L,3]
  d = np.linalg.norm(leaf_c[:, None, :] - fruit_c[None, :, :], axis=2)  # [L,F]
  m = (d.min(axis=1) <= float(max_dist))
  return float(m.mean())


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


def main() -> int:
  args = parse_args()
  # Seed the *global* RNG because JSONSubgraphSampler (and our occl-aug injector)
  # uses the module-level `random` functions.
  random.seed(args.seed)
  rng = random.Random(args.seed)
  np_rng = np.random.default_rng(args.seed)

  # Ensure repo root is on sys.path for `conda run python ...` usage.
  repo_root = Path(__file__).resolve().parents[2]
  sys.path.append(str(repo_root))

  # Import here so running this script doesn't force torch/open3d in minimal envs.
  from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset

  # We instantiate the dataset only to reuse:
  #  - consistent split selection (seed=42 inside)
  #  - JSONSubgraphSampler wiring
  #  - sidecar auto-detection helpers
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
    # OraclePepperDataset doesn't have a native "all" mode, but indices is already
    # the selected split. For "all", just use full pc_files list.
    scene_indices = list(range(len(ds.pc_files)))
  if args.max_scenes is not None and args.max_scenes < len(scene_indices):
    if args.sample_random:
      scene_indices = rng.sample(scene_indices, k=args.max_scenes)
    else:
      scene_indices = scene_indices[: args.max_scenes]

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  mode = "occl_aug" if args.occl_aug_enabled else "bfs"
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"oracle_subgraph_occl_cov_{mode}_{ts}"
  out_dir = out_dir.resolve()
  plots_dir = out_dir / "plots"
  _mkdir(plots_dir)

  cov_taus = _parse_float_list(args.cov_tau_list)

  # Aggregate stats.
  subgraph_totals: List[int] = []
  subgraph_stems: List[int] = []
  subgraph_leaves: List[int] = []
  subgraph_peds: List[int] = []
  subgraph_fruits: List[int] = []
  subgraph_shoots: List[int] = []

  # Fruit-level stats (per sampled subgraph).
  fruits_with_sidecar_per_subgraph: List[int] = []
  fruits_high_union_per_subgraph: List[int] = []
  dom_dirs_per_subgraph: List[int] = []
  # Per-fruit stats (pooled over fruits, across all sampled subgraphs).
  fruit_dom_dir_count_all: List[int] = []
  fruit_remain_rate_all: List[float] = []
  fruit_remain_rate_high_union: List[float] = []

  # Extra occlusion supervision density stats (matches OraclePepperDataset selection).
  sup_focus_used_per_subgraph: List[int] = []
  sup_extra_eligible_per_subgraph: List[int] = []
  sup_extra_added_per_subgraph: List[int] = []
  sup_total_per_subgraph: List[int] = []
  sup_dom_dirs_total_per_subgraph: List[int] = []
  sup_pairs_total_per_subgraph: List[int] = []
  sup_pairs_focus_per_subgraph: List[int] = []
  sup_pairs_extra_per_subgraph: List[int] = []

  # Occl-aug "clusteredness"/shift stats (only computed when occl_aug_enabled).
  leaf_focus_frac_base: List[float] = []
  leaf_focus_frac_final: List[float] = []
  leaf_injected_focus_frac: List[float] = []
  # When BFS is used (occl_aug disabled), we still compute a focus-zone "clusteredness" probe
  # by selecting focus fruits with the same guardrail logic and reporting the resulting leaf fraction.
  leaf_focus_frac_bfs: List[float] = []

  base_kept_frac_total: List[float] = []
  base_kept_frac_leaf: List[float] = []
  base_kept_frac_fruit: List[float] = []
  base_kept_frac_ped: List[float] = []
  base_kept_frac_stem: List[float] = []
  injected_total_nodes: List[int] = []
  injected_leaf_nodes: List[int] = []
  evicted_total_nodes: List[int] = []

  fruit_union_all: List[float] = []
  fruit_union_dom: List[float] = []

  cov_excl_all: List[float] = []
  cov_excl_dom: List[float] = []
  cov_k3_all: List[float] = []
  cov_k3_dom: List[float] = []

  top1_present_all: List[float] = []
  top1_present_dom: List[float] = []

  # Direction "explainability cap" at the dominant threshold:
  # For each GT-dominant direction (union>=tau), we compute the union occlusion that is
  # attributable to leaves present in the sampled subgraph:
  #   union_present = sum_{l present} exclusive_{l,d}
  # and count the fraction of dominant directions that remain dominant under this present-only union:
  #   remain = 1[union_present >= tau]
  union_present_dom: List[float] = []
  dominant_remain: List[float] = []
  # Same, but restricted to occl-aug focus fruits (the ones supervised during training).
  union_present_dom_focus: List[float] = []
  dominant_remain_focus: List[float] = []

  n_dirs_total_all = 0
  n_dirs_total_dom = 0

  n_scenes_used = 0
  n_sidecar_missing = 0
  n_json_missing = 0

  for scene_idx in scene_indices:
    pc_filename = ds.pc_files[scene_idx]
    scene_id = pc_filename.split("_")[0] + "_" + pc_filename.split("_")[1]
    json_path = Path(ds.data_dir) / f"{scene_id}_graph.json"
    if not json_path.exists():
      n_json_missing += 1
      continue
    # Sidecar is optional; we'll compute coverage only when present.
    sidecar = ds._load_occl_sidecar(scene_id)
    if sidecar is None:
      n_sidecar_missing += 1
    nodes, edges, _ = ds.sampler.load_json_graph(str(json_path))
    if args.drop_shoots:
      nodes, edges = _drop_nodes_by_semantic_id(nodes, edges, drop_sem_ids=[5])
    if not nodes:
      continue

    # Pre-index sidecar fruit data for this scene.
    if sidecar is not None and sidecar["fruit_iid"].size > 0:
      fruit_iid_arr = sidecar["fruit_iid"]
      fruit_union_arr = sidecar["fruit_union"]  # [F,D]
      fruit_iid_to_row = {int(iid): i for i, iid in enumerate(fruit_iid_arr.tolist())}
    else:
      fruit_iid_to_row = {}
      fruit_union_arr = None

    # Pre-index sidecar pairs by fruit.
    if sidecar is not None and sidecar["pair_fruit_iid"].size > 0:
      pf = sidecar["pair_fruit_iid"]
      pl = sidecar["pair_leaf_iid"]
      pk = sidecar["pair_k_rank"]  # [P,D,3]
      pp = sidecar["pair_potential"]
      # Sort by fruit to allow fast slicing.
      order = np.argsort(pf)
      pf_s = pf[order]
      pl_s = pl[order]
      pk_s = pk[order]
      pp_s = pp[order]
      uniq_f, starts, counts = np.unique(pf_s, return_index=True, return_counts=True)
      fruit_to_slice = {int(i): (int(s), int(c)) for i, s, c in zip(uniq_f.tolist(), starts.tolist(), counts.tolist())}
    else:
      fruit_to_slice = {}
      pf_s = pl_s = pk_s = pp_s = None

    for _ in range(int(args.samples_per_scene)):
      target_size = int(args.subgraph_size)
      if args.subgraph_size_min is not None and args.subgraph_size_min < args.subgraph_size:
        target_size = int(np_rng.integers(args.subgraph_size_min, args.subgraph_size + 1))

      # Match OraclePepperDataset behavior: optionally reserve capacity for occl-aug by sampling
      # a smaller BFS core and then augmenting back up to target_size.
      core_target = int(target_size)
      if args.occl_aug_enabled and int(args.occl_aug_reserve_nodes) > 0:
        core_target = max(1, int(target_size) - int(args.occl_aug_reserve_nodes))

      selected_base = ds.sampler.bfs_sample(nodes, edges, target_size=core_target, seed_policy="stem_bias")
      selected = selected_base
      focus_fruit_iid_set = set()
      focus_fruits: List[int] = []
      if args.occl_aug_enabled:
        debug: Dict[str, Any] = {}
        selected = ds._augment_subgraph_with_occlusion(scene_id, nodes, edges, selected_base, target_size=target_size, debug=debug)

        focus_fruits = [int(x) for x in debug.get("focus_fruits", [])]
        # Convert focus fruit JSON indices -> instance IDs (used later to compute focus-only cap metrics).
        focus_fruit_iids: List[int] = []
        for jidx in focus_fruits:
          try:
            iid = int(nodes[int(jidx)].get("instance_id", -1))
          except Exception:
            iid = -1
          if iid > 0:
            focus_fruit_iids.append(iid)
        focus_fruit_iid_set = set(focus_fruit_iids)

        base_set = set(int(x) for x in selected_base)
        final_set = set(int(x) for x in selected)
        kept = base_set & final_set

        if base_set:
          base_kept_frac_total.append(len(kept) / float(len(base_set)))

        def _type_counts(sel: Sequence[int]) -> Dict[int, int]:
          out: Dict[int, int] = {}
          for idx in sel:
            sid = int(nodes[int(idx)].get("semantic_id", -1))
            out[sid] = int(out.get(sid, 0)) + 1
          return out

        base_counts = _type_counts(selected_base)
        kept_counts = _type_counts(list(kept))
        for sid, bucket in [
          (2, base_kept_frac_leaf),
          (4, base_kept_frac_fruit),
          (3, base_kept_frac_ped),
          (1, base_kept_frac_stem),
        ]:
          denom = base_counts.get(sid, 0)
          if denom > 0:
            bucket.append(kept_counts.get(sid, 0) / float(denom))

        leaf_base = [i for i in selected_base if int(nodes[int(i)].get("semantic_id", -1)) == 2]
        leaf_final = [i for i in selected if int(nodes[int(i)].get("semantic_id", -1)) == 2]
        leaf_focus_frac_base.append(_leaf_fraction_within_focus_zone(nodes, leaf_base, focus_fruits, max_dist=float(args.occl_aug_max_dist)))
        leaf_focus_frac_final.append(_leaf_fraction_within_focus_zone(nodes, leaf_final, focus_fruits, max_dist=float(args.occl_aug_max_dist)))

        injected = final_set - base_set
        leaf_injected = [i for i in injected if int(nodes[int(i)].get("semantic_id", -1)) == 2]
        injected_total_nodes.append(int(len(injected)))
        injected_leaf_nodes.append(int(len(leaf_injected)))
        evicted_total_nodes.append(int(len(base_set - final_set)))
        if leaf_injected:
          leaf_injected_focus_frac.append(_leaf_fraction_within_focus_zone(nodes, leaf_injected, focus_fruits, max_dist=float(args.occl_aug_max_dist)))
      else:
        # BFS-only: pick focus fruits using the same guardrail logic, but do not inject/evict nodes.
        # Important: JSONSubgraphSampler uses the *global* RNG, so we must not perturb it here,
        # otherwise the "BFS" distribution changes. We snapshot/restore around focus sampling.
        _state = random.getstate()
        try:
          focus_fruits = _pick_focus_fruits_like_occl_aug(
            nodes,
            selected,
            num_fruits=int(args.occl_aug_num_fruits),
            resample_tries=int(args.occl_aug_resample_tries),
            union_tau=float(args.occl_aug_union_tau),
            fruit_iid_to_row=fruit_iid_to_row,
            fruit_union_arr=fruit_union_arr,
          )
        finally:
          random.setstate(_state)
        leaf_sel = [i for i in selected if int(nodes[int(i)].get("semantic_id", -1)) == 2]
        leaf_focus_frac_bfs.append(_leaf_fraction_within_focus_zone(nodes, leaf_sel, focus_fruits, max_dist=float(args.occl_aug_max_dist)))

      counts = _semantic_counts(nodes, selected)
      subgraph_totals.append(counts["total"])
      subgraph_stems.append(counts.get("stem", 0))
      subgraph_leaves.append(counts.get("leaf", 0))
      subgraph_peds.append(counts.get("ped", 0))
      subgraph_fruits.append(counts.get("fruit", 0))
      subgraph_shoots.append(counts.get("shoot", 0))

      if sidecar is None:
        continue

      # Selected leaves instance_ids.
      leaf_iids_sel: List[int] = []
      fruit_iids_sel: List[int] = []
      leaf_centroids_sel: List[np.ndarray] = []
      fruit_centroids_sel: Dict[int, np.ndarray] = {}
      for idx in selected:
        n = nodes[idx]
        iid = n.get("instance_id")
        if iid is None:
          continue
        sid = int(n.get("semantic_id", -1))
        if sid == 2:
          leaf_iids_sel.append(int(iid))
          leaf_centroids_sel.append(np.asarray(n.get("centroid"), dtype=np.float32))
        elif sid == 4:
          fruit_iids_sel.append(int(iid))
          fruit_centroids_sel[int(iid)] = np.asarray(n.get("centroid"), dtype=np.float32)
      leaf_iids_set = set(leaf_iids_sel)
      leaf_centroids_a = np.stack(leaf_centroids_sel, axis=0) if leaf_centroids_sel else None  # [L,3] or None

      # Per-sample fruit info (used for extra-supervision simulation).
      fruit_info: Dict[int, Dict[str, Any]] = {}
      # Focus fruit set for this sampled subgraph (matches OraclePepperDataset fallback behavior).
      # If occl-aug is enabled, we already have a focus list from the debug dict; otherwise sample random fruits present.
      focus_iid_set: set[int]
      if focus_fruit_iid_set:
        focus_iid_set = set(int(x) for x in focus_fruit_iid_set)
      else:
        keys = [int(x) for x in fruit_iids_sel]
        if keys:
          k = min(max(int(args.occl_aug_num_fruits), 1), len(keys))
          focus_iid_set = set(int(x) for x in rng.sample(keys, k=k))
        else:
          focus_iid_set = set()

      for fruit_iid in fruit_iids_sel:
        # Union vector.
        row = fruit_iid_to_row.get(int(fruit_iid))
        if row is None or fruit_union_arr is None:
          continue
        union = fruit_union_arr[row].astype(np.float32, copy=False)  # [D]
        if union.size == 0:
          continue

        # Pair slice.
        sl = fruit_to_slice.get(int(fruit_iid))
        if sl is None or pk_s is None or pl_s is None:
          continue
        s0, c0 = sl
        leaf_iids = pl_s[s0 : s0 + c0].astype(np.int32, copy=False)  # [P_f]
        k_rank = pk_s[s0 : s0 + c0].astype(np.float32, copy=False)  # [P_f,D,3]
        # exclusive is rank0 by construction
        excl = k_rank[:, :, 0]  # [P_f,D]
        k_sum = k_rank.sum(axis=2)  # [P_f,D]

        present_mask = np.isin(leaf_iids, list(leaf_iids_set))

        excl_present = excl[present_mask].sum(axis=0)  # [D]
        ksum_present = k_sum[present_mask].sum(axis=0)  # [D]
        ksum_total = k_sum.sum(axis=0)  # [D]

        # Fruit-level dominant dirs and "remain rate" under present leaves (matches OraclePepperDataset logic).
        tau = float(args.union_tau)
        dom_mask = (union >= tau)
        dom_count = int(dom_mask.sum())
        union_max = float(union.max())
        u_pres = np.clip(excl_present, 0.0, 1.0)
        remain_rate = float((u_pres[dom_mask] >= tau).mean()) if bool(dom_mask.any()) else 0.0
        fruit_info[int(fruit_iid)] = {
          "union_max": union_max,
          "dom_count": dom_count,
          "remain_rate": remain_rate,
        }
        fruit_dom_dir_count_all.append(dom_count)
        fruit_remain_rate_all.append(remain_rate)
        if union_max >= tau:
          fruit_remain_rate_high_union.append(remain_rate)

        # Union>0 directions.
        m_all = union > 0
        if bool(m_all.any()):
          u_all = union[m_all]
          cov_e = np.clip(excl_present[m_all] / (u_all + 1e-12), 0.0, 1.0)
          cov_excl_all.extend(cov_e.tolist())
          fruit_union_all.extend(u_all.tolist())
          n_dirs_total_all += int(u_all.size)

          denom = ksum_total[m_all]
          m_k = denom > 0
          if bool(m_k.any()):
            cov_k = np.clip(ksum_present[m_all][m_k] / (denom[m_k] + 1e-12), 0.0, 1.0)
            cov_k3_all.extend(cov_k.tolist())

          # Top1 presence rate (by exclusive).
          # For each dir, find leaf with max exclusive (ties arbitrary).
          top_leaf = leaf_iids[np.argmax(excl[:, m_all], axis=0)]
          top_present = np.isin(top_leaf, list(leaf_iids_set)).astype(np.float32)
          top1_present_all.extend(top_present.tolist())

        # Dominant directions.
        m_dom = union >= float(args.union_tau)
        if bool(m_dom.any()):
          u_dom = union[m_dom]
          u_pres_dom = np.clip(excl_present[m_dom], 0.0, 1.0)
          union_present_dom.extend(u_pres_dom.tolist())
          dominant_remain.extend((u_pres_dom >= float(args.union_tau)).astype(np.float32).tolist())
          if int(fruit_iid) in focus_iid_set:
            union_present_dom_focus.extend(u_pres_dom.tolist())
            dominant_remain_focus.extend((u_pres_dom >= float(args.union_tau)).astype(np.float32).tolist())

          cov_e = np.clip(excl_present[m_dom] / (u_dom + 1e-12), 0.0, 1.0)
          cov_excl_dom.extend(cov_e.tolist())
          fruit_union_dom.extend(u_dom.tolist())
          n_dirs_total_dom += int(u_dom.size)

          denom = ksum_total[m_dom]
          m_k = denom > 0
          if bool(m_k.any()):
            cov_k = np.clip(ksum_present[m_dom][m_k] / (denom[m_k] + 1e-12), 0.0, 1.0)
            cov_k3_dom.extend(cov_k.tolist())

          top_leaf = leaf_iids[np.argmax(excl[:, m_dom], axis=0)]
          top_present = np.isin(top_leaf, list(leaf_iids_set)).astype(np.float32)
          top1_present_dom.extend(top_present.tolist())

      # Fruit-level counts + extra-supervision density (per sampled subgraph).
      tau = float(args.union_tau)
      n_fruit_sidecar = int(len(fruit_info))
      fruits_with_sidecar_per_subgraph.append(n_fruit_sidecar)
      n_high = int(sum(1 for v in fruit_info.values() if float(v.get("union_max", 0.0)) >= tau))
      fruits_high_union_per_subgraph.append(n_high)
      dom_dirs_per_subgraph.append(int(sum(int(v.get("dom_count", 0)) for v in fruit_info.values())))

      # Match OraclePepperDataset: if occl-aug provides an explicit focus list, use it; otherwise
      # fall back to sampling random fruits present in this subgraph.
      focus_used = [int(fiid) for fiid in fruit_info.keys() if int(fiid) in focus_iid_set]
      focus_used_set = set(focus_used)

      min_dom_dirs = max(int(args.occl_supervise_min_dom_dirs), 1)
      remain_min = float(args.occl_supervise_remain_rate_min)
      eligible: List[Tuple[float, int]] = []
      for fiid, info in fruit_info.items():
        if int(fiid) in focus_used_set:
          continue
        if float(info.get("union_max", 0.0)) < tau:
          continue
        if int(info.get("dom_count", 0)) < min_dom_dirs:
          continue
        if float(info.get("remain_rate", 0.0)) < remain_min:
          continue
        eligible.append((float(info.get("union_max", 0.0)), int(fiid)))
      eligible.sort(key=lambda t: t[0], reverse=True)

      max_total = int(args.occl_supervise_max_fruits_per_scene)
      extra_added: List[int] = []
      if max_total > 0 and len(focus_used) > 0:
        space = max(0, max_total - len(focus_used))
        extra_added = [int(fiid) for _, fiid in eligible[:space]]

      sup_focus_used_per_subgraph.append(int(len(focus_used)))
      sup_extra_eligible_per_subgraph.append(int(len(eligible)))
      sup_extra_added_per_subgraph.append(int(len(extra_added)))
      sup_total_per_subgraph.append(int(len(focus_used) + len(extra_added)))

      dom_dirs_sup = int(sum(int(fruit_info[int(fiid)]["dom_count"]) for fiid in focus_used + extra_added))
      sup_dom_dirs_total_per_subgraph.append(dom_dirs_sup)

      # Approximate occlusion pair supervision density: for each supervised fruit, count nearby leaves within max_dist,
      # then clamp by max_leaves_per_fruit (matches dataset's picked-leaf budget).
      max_leaves = max(int(args.occl_targets_max_leaves_per_fruit), 1)
      max_dist = float(args.occl_aug_max_dist)

      def _picked_for_fruit(fiid: int) -> int:
        if leaf_centroids_a is None:
          return 0
        fc = fruit_centroids_sel.get(int(fiid))
        if fc is None:
          return 0
        d = np.linalg.norm(leaf_centroids_a - fc[None, :], axis=1)
        cand = int((d <= max_dist).sum())
        return int(min(max_leaves, cand))

      pairs_focus = int(sum(_picked_for_fruit(int(fiid)) for fiid in focus_used))
      pairs_extra = int(sum(_picked_for_fruit(int(fiid)) for fiid in extra_added))
      sup_pairs_focus_per_subgraph.append(pairs_focus)
      sup_pairs_extra_per_subgraph.append(pairs_extra)
      sup_pairs_total_per_subgraph.append(int(pairs_focus + pairs_extra))

    n_scenes_used += 1

  # Convert to arrays.
  subgraph_totals_a = np.asarray(subgraph_totals, dtype=np.float32)
  subgraph_stems_a = np.asarray(subgraph_stems, dtype=np.float32)
  subgraph_leaves_a = np.asarray(subgraph_leaves, dtype=np.float32)
  subgraph_peds_a = np.asarray(subgraph_peds, dtype=np.float32)
  subgraph_fruits_a = np.asarray(subgraph_fruits, dtype=np.float32)
  subgraph_shoots_a = np.asarray(subgraph_shoots, dtype=np.float32)

  union_all_a = np.asarray(fruit_union_all, dtype=np.float32)
  union_dom_a = np.asarray(fruit_union_dom, dtype=np.float32)

  cov_excl_all_a = np.asarray(cov_excl_all, dtype=np.float32)
  cov_excl_dom_a = np.asarray(cov_excl_dom, dtype=np.float32)
  cov_k3_all_a = np.asarray(cov_k3_all, dtype=np.float32)
  cov_k3_dom_a = np.asarray(cov_k3_dom, dtype=np.float32)
  top1_all_a = np.asarray(top1_present_all, dtype=np.float32)
  top1_dom_a = np.asarray(top1_present_dom, dtype=np.float32)
  union_present_dom_a = np.asarray(union_present_dom, dtype=np.float32)
  dominant_remain_a = np.asarray(dominant_remain, dtype=np.float32)
  union_present_dom_focus_a = np.asarray(union_present_dom_focus, dtype=np.float32)
  dominant_remain_focus_a = np.asarray(dominant_remain_focus, dtype=np.float32)

  # Fruit-level arrays.
  fruits_with_sidecar_a = np.asarray(fruits_with_sidecar_per_subgraph, dtype=np.float32)
  fruits_high_union_a = np.asarray(fruits_high_union_per_subgraph, dtype=np.float32)
  dom_dirs_per_subgraph_a = np.asarray(dom_dirs_per_subgraph, dtype=np.float32)
  fruit_dom_dir_count_all_a = np.asarray(fruit_dom_dir_count_all, dtype=np.float32)
  fruit_remain_rate_all_a = np.asarray(fruit_remain_rate_all, dtype=np.float32)
  fruit_remain_rate_high_union_a = np.asarray(fruit_remain_rate_high_union, dtype=np.float32)

  sup_focus_used_a = np.asarray(sup_focus_used_per_subgraph, dtype=np.float32)
  sup_extra_eligible_a = np.asarray(sup_extra_eligible_per_subgraph, dtype=np.float32)
  sup_extra_added_a = np.asarray(sup_extra_added_per_subgraph, dtype=np.float32)
  sup_total_a = np.asarray(sup_total_per_subgraph, dtype=np.float32)
  sup_dom_dirs_total_a = np.asarray(sup_dom_dirs_total_per_subgraph, dtype=np.float32)
  sup_pairs_total_a = np.asarray(sup_pairs_total_per_subgraph, dtype=np.float32)
  sup_pairs_focus_a = np.asarray(sup_pairs_focus_per_subgraph, dtype=np.float32)
  sup_pairs_extra_a = np.asarray(sup_pairs_extra_per_subgraph, dtype=np.float32)

  # Threshold rates.
  thr: Dict[str, Any] = {}
  for t in cov_taus:
    thr[f"excl_all_ge_{t:g}"] = float((cov_excl_all_a >= t).mean()) if cov_excl_all_a.size else float("nan")
    thr[f"excl_dom_ge_{t:g}"] = float((cov_excl_dom_a >= t).mean()) if cov_excl_dom_a.size else float("nan")
    thr[f"k3_all_ge_{t:g}"] = float((cov_k3_all_a >= t).mean()) if cov_k3_all_a.size else float("nan")
    thr[f"k3_dom_ge_{t:g}"] = float((cov_k3_dom_a >= t).mean()) if cov_k3_dom_a.size else float("nan")

  summary: Dict[str, Any] = {
    "mode": mode,
    "data_root": str(Path(args.data_root).resolve()),
    "split": args.split,
    "scenes_requested": int(len(scene_indices)),
    "scenes_used": int(n_scenes_used),
    "json_missing": int(n_json_missing),
    "sidecar_missing": int(n_sidecar_missing),
    "samples_per_scene": int(args.samples_per_scene),
    "subgraph": {
      "size": int(args.subgraph_size),
      "size_min": None if args.subgraph_size_min is None else int(args.subgraph_size_min),
      "hard_pair_constraint": bool(args.hard_pair_constraint),
      "drop_shoots": bool(args.drop_shoots),
      "counts_total": _summary(subgraph_totals_a),
      "counts_stem": _summary(subgraph_stems_a),
      "counts_leaf": _summary(subgraph_leaves_a),
      "counts_ped": _summary(subgraph_peds_a),
      "counts_fruit": _summary(subgraph_fruits_a),
      "counts_shoot": _summary(subgraph_shoots_a),
    },
    "occlusion": {
      "union_tau": float(args.union_tau),
      "n_dirs_all": int(n_dirs_total_all),
      "n_dirs_dom": int(n_dirs_total_dom),
      "union_all": _summary(union_all_a),
      "union_dom": _summary(union_dom_a),
      "union_present_dom": _summary(union_present_dom_a),
      "dominant_remain_rate": _summary(dominant_remain_a),
      "union_present_dom_focus": _summary(union_present_dom_focus_a),
      "dominant_remain_rate_focus": _summary(dominant_remain_focus_a),
      "coverage_exclusive_all": _summary(cov_excl_all_a),
      "coverage_exclusive_dom": _summary(cov_excl_dom_a),
      "coverage_k3_all": _summary(cov_k3_all_a),
      "coverage_k3_dom": _summary(cov_k3_dom_a),
      "top1_present_all": _summary(top1_all_a),
      "top1_present_dom": _summary(top1_dom_a),
      "threshold_rates": thr,
      "fruit_level": {
        "fruits_with_sidecar_per_subgraph": _summary(fruits_with_sidecar_a),
        "high_union_fruits_per_subgraph": _summary(fruits_high_union_a),
        "dominant_dirs_per_subgraph": _summary(dom_dirs_per_subgraph_a),
        "dom_dir_count_per_fruit": _summary(fruit_dom_dir_count_all_a),
        "remain_rate_per_fruit": _summary(fruit_remain_rate_all_a),
        "remain_rate_per_high_union_fruit": _summary(fruit_remain_rate_high_union_a),
      },
      "extra_supervision": {
        "max_fruits_per_scene": int(args.occl_supervise_max_fruits_per_scene),
        "remain_rate_min": float(args.occl_supervise_remain_rate_min),
        "min_dom_dirs": int(args.occl_supervise_min_dom_dirs),
        "max_leaves_per_fruit": int(args.occl_targets_max_leaves_per_fruit),
        "focus_used_per_subgraph": _summary(sup_focus_used_a),
        "extra_eligible_per_subgraph": _summary(sup_extra_eligible_a),
        "extra_added_per_subgraph": _summary(sup_extra_added_a),
        "total_supervised_per_subgraph": _summary(sup_total_a),
        "supervised_dom_dirs_per_subgraph": _summary(sup_dom_dirs_total_a),
        "supervised_pairs_total_per_subgraph": _summary(sup_pairs_total_a),
        "supervised_pairs_focus_per_subgraph": _summary(sup_pairs_focus_a),
        "supervised_pairs_extra_per_subgraph": _summary(sup_pairs_extra_a),
      },
    },
  }

  def _finite(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
      return a
    m = np.isfinite(a)
    return a[m]

  if args.occl_aug_enabled:

    leaf_focus_frac_base_a = _finite(np.asarray(leaf_focus_frac_base, dtype=np.float32))
    leaf_focus_frac_final_a = _finite(np.asarray(leaf_focus_frac_final, dtype=np.float32))
    leaf_injected_focus_frac_a = _finite(np.asarray(leaf_injected_focus_frac, dtype=np.float32))
    base_kept_frac_total_a = _finite(np.asarray(base_kept_frac_total, dtype=np.float32))
    base_kept_frac_leaf_a = _finite(np.asarray(base_kept_frac_leaf, dtype=np.float32))
    base_kept_frac_fruit_a = _finite(np.asarray(base_kept_frac_fruit, dtype=np.float32))
    base_kept_frac_ped_a = _finite(np.asarray(base_kept_frac_ped, dtype=np.float32))
    base_kept_frac_stem_a = _finite(np.asarray(base_kept_frac_stem, dtype=np.float32))
    injected_total_nodes_a = _finite(np.asarray(injected_total_nodes, dtype=np.float32))
    injected_leaf_nodes_a = _finite(np.asarray(injected_leaf_nodes, dtype=np.float32))
    evicted_total_nodes_a = _finite(np.asarray(evicted_total_nodes, dtype=np.float32))

    summary["occl_aug_shift"] = {
      "occl_aug_max_dist": float(args.occl_aug_max_dist),
      "leaf_focus_frac_base": _summary(leaf_focus_frac_base_a),
      "leaf_focus_frac_final": _summary(leaf_focus_frac_final_a),
      "leaf_injected_focus_frac": _summary(leaf_injected_focus_frac_a),
      "base_kept_frac_total": _summary(base_kept_frac_total_a),
      "base_kept_frac_leaf": _summary(base_kept_frac_leaf_a),
      "base_kept_frac_fruit": _summary(base_kept_frac_fruit_a),
      "base_kept_frac_ped": _summary(base_kept_frac_ped_a),
      "base_kept_frac_stem": _summary(base_kept_frac_stem_a),
      "injected_total_nodes": _summary(injected_total_nodes_a),
      "injected_leaf_nodes": _summary(injected_leaf_nodes_a),
      "evicted_total_nodes": _summary(evicted_total_nodes_a),
    }

    _plot_hist(leaf_focus_frac_base_a, plots_dir / "leaf_focus_frac_base.png", "Leaf fraction within focus-fruit 0.2m zone (base BFS)", "fraction")
    _plot_hist(leaf_focus_frac_final_a, plots_dir / "leaf_focus_frac_final.png", "Leaf fraction within focus-fruit 0.2m zone (final occl-aug)", "fraction")
    _plot_hist(base_kept_frac_total_a, plots_dir / "base_kept_frac_total.png", "Fraction of BFS base nodes retained after occl-aug", "fraction")
    _plot_hist(injected_total_nodes_a, plots_dir / "injected_total_nodes.png", "Injected nodes per subgraph (final - base)", "count", bins=50, range_=(0, float(args.subgraph_size)))
  else:
    leaf_focus_frac_bfs_a = _finite(np.asarray(leaf_focus_frac_bfs, dtype=np.float32))
    summary["focus_probe_bfs"] = {
      "occl_aug_max_dist": float(args.occl_aug_max_dist),
      "leaf_focus_frac": _summary(leaf_focus_frac_bfs_a),
    }
    _plot_hist(leaf_focus_frac_bfs_a, plots_dir / "leaf_focus_frac_bfs.png", "Leaf fraction within focus-fruit 0.2m zone (BFS)", "fraction")

  _save_json(out_dir / "summary.json", summary)

  # Plots.
  _plot_hist(cov_excl_all_a, plots_dir / "cov_exclusive_all.png", "Exclusive Coverage (all union>0 dirs)", "coverage")
  _plot_hist(cov_excl_dom_a, plots_dir / "cov_exclusive_dom.png", f"Exclusive Coverage (dominant dirs union>={args.union_tau:g})", "coverage")
  _plot_hist(cov_k3_dom_a, plots_dir / "cov_k3_dom.png", f"K3 Coverage (dominant dirs union>={args.union_tau:g})", "coverage")
  _plot_hist(union_present_dom_a, plots_dir / "union_present_dom.png", f"Present-only Union (dominant dirs union>={args.union_tau:g})", "union_present")
  _plot_hist(union_present_dom_focus_a, plots_dir / "union_present_dom_focus.png", f"Present-only Union (FOCUS fruits; dominant dirs union>={args.union_tau:g})", "union_present")
  _plot_hist(subgraph_fruits_a, plots_dir / "subgraph_fruits.png", "Fruits per subgraph", "count", bins=50, range_=(0, float(args.subgraph_size)))
  _plot_hist(subgraph_leaves_a, plots_dir / "subgraph_leaves.png", "Leaves per subgraph", "count", bins=50, range_=(0, float(args.subgraph_size)))
  _plot_hist(fruits_high_union_a, plots_dir / "high_union_fruits_per_subgraph.png", f"High-union fruits per subgraph (union_max>={args.union_tau:g})", "count", bins=50, range_=(0, float(args.subgraph_size)))
  _plot_hist(sup_total_a, plots_dir / "supervised_fruits_per_subgraph.png", "Supervised fruits per subgraph (focus + extra)", "count", bins=50, range_=(0, float(args.subgraph_size)))

  print(f"Wrote: {out_dir}/summary.json")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
