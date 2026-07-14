#!/usr/bin/env python3
"""Analyze occlusion overlap distributions for raw/weighted targets and axis-pair modes.

This script summarizes per-direction and global statistics to help choose bin thresholds,
diagnose class imbalance, and sanity-check the occlusion values stored in the dataset JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
try:
  import matplotlib.pyplot as plt
except ImportError as exc:
  plt = None


@dataclass
class StatsCfg:
  none_thresh: float
  high_thresh: float


DEFAULT_CANDIDATE_TESTS = [
  {
    "name": "base",
    "max_dist": 0.15,
    "thin_keep_prob": 1.0,
    "none_keep_prob": 1.0,
    "extra_neg_ratio": 0.0,
    "ring_max_dist": 0.3,
    "ring_frac": 0.7,
  },
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Analyze occlusion overlap distributions.")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
  parser.add_argument(
    "--out",
    default=None,
    help="Output JSON path (default: <dataset_dir>/occlusion_stats.json)",
  )
  parser.add_argument(
    "--num-scenes",
    type=int,
    default=None,
    help="Optional limit on number of scenes to analyze",
  )
  parser.add_argument(
    "--sample-random",
    action="store_true",
    help="Sample scenes randomly instead of taking the first N",
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
  parser.add_argument("--none-thresh", type=float, default=0.03, help="Upper bound for none bin")
  parser.add_argument("--high-thresh", type=float, default=0.1, help="Lower bound for high bin")
  parser.add_argument(
    "--none-mode",
    choices=["axes3", "axes9"],
    default="axes3",
    help="Direction set used for all-none candidate filtering in simulations",
  )
  parser.add_argument(
    "--weight",
    type=float,
    default=None,
    help="Occlusion weight factor for mapping raw thresholds to weighted thresholds (defaults to dataset value).",
  )
  return parser.parse_args()


def _select_scenes(graphs: List[Dict[str, Any]], num_scenes: int | None, sample_random: bool, seed: int) -> List[Dict[str, Any]]:
  if num_scenes is None or num_scenes >= len(graphs):
    return graphs
  if not sample_random:
    return graphs[:num_scenes]
  rng = random.Random(seed)
  indices = list(range(len(graphs)))
  rng.shuffle(indices)
  return [graphs[i] for i in indices[:num_scenes]]


def _flatten_values(values: np.ndarray) -> np.ndarray:
  if values.size == 0:
    return values
  return values.reshape(-1)


def _quantiles(arr: np.ndarray) -> Dict[str, float]:
  if arr.size == 0:
    return {k: float("nan") for k in ("q05", "q25", "q50", "q75", "q95")}
  qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
  return {"q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4])}


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
  if arr.size == 0:
    return {
      "count": 0,
      "mean": float("nan"),
      "median": float("nan"),
      "std": float("nan"),
      "min": float("nan"),
      "max": float("nan"),
      "zero_rate": float("nan"),
      **_quantiles(arr),
    }
  return {
    "count": int(arr.size),
    "mean": float(arr.mean()),
    "median": float(np.median(arr)),
    "std": float(arr.std()),
    "min": float(arr.min()),
    "max": float(arr.max()),
    "zero_rate": float((arr == 0.0).mean()),
    **_quantiles(arr),
  }


def _axis_stats(arr: np.ndarray) -> Dict[str, Any]:
  if arr.size == 0:
    return {f"axis_{i}": _summary_stats(arr) for i in range(3)}
  stats = {}
  for i in range(arr.shape[1]):
    stats[f"axis_{i}"] = _summary_stats(arr[:, i])
  return stats


def _axis_pairs(mode: str) -> List[tuple[int, int]]:
  if mode == "axes3":
    return [(0, 1), (2, 3), (4, 5)]
  if mode == "axes9":
    return [
      (0, 1),    # X
      (2, 3),    # Y
      (4, 5),    # Z
      (9, 6),    # X+Y vs -X-Y
      (8, 7),    # X-Y vs -X+Y
      (13, 10),  # X+Z vs -X-Z
      (12, 11),  # X-Z vs -X+Z
      (17, 14),  # Y+Z vs -Y-Z
      (16, 15),  # Y-Z vs -Y+Z
    ]
  raise ValueError(f"Unknown dir_mode: {mode}")


def _collapse_axes(values: np.ndarray, mode: str) -> np.ndarray:
  """Collapse +/- pairs into axis magnitudes."""
  if values.size == 0:
    return values.reshape(0, len(_axis_pairs(mode)))
  pairs = _axis_pairs(mode)
  collapsed = np.zeros((values.shape[0], len(pairs)), dtype=float)
  for i, (pos, neg) in enumerate(pairs):
    collapsed[:, i] = np.maximum(values[:, pos], values[:, neg])
  return collapsed


def _bin_counts(arr: np.ndarray, cfg: StatsCfg) -> Dict[str, int]:
  if arr.size == 0:
    return {"none": 0, "low": 0, "high": 0}
  none_mask = arr <= cfg.none_thresh
  high_mask = arr >= cfg.high_thresh
  low_mask = (~none_mask) & (~high_mask)
  return {
    "none": int(none_mask.sum()),
    "low": int(low_mask.sum()),
    "high": int(high_mask.sum()),
  }


def _per_direction_stats(values: np.ndarray, cfg: StatsCfg) -> Dict[str, Any]:
  stats = {}
  for i in range(values.shape[1]):
    arr = values[:, i]
    stats[f"dir_{i}"] = {
      "summary": _summary_stats(arr),
      "bin_counts": _bin_counts(arr, cfg),
    }
  return stats


def _nonzero_hist(values: np.ndarray, thresh: float) -> Dict[str, int]:
  if values.size == 0:
    return {}
  counts = (values > thresh).sum(axis=1)
  hist: Dict[str, int] = {}
  for c in counts.tolist():
    key = str(int(c))
    hist[key] = hist.get(key, 0) + 1
  return hist


def analyze(values: np.ndarray, cfg: StatsCfg) -> Dict[str, Any]:
  flat = _flatten_values(values)
  return {
    "overall": {
      "summary": _summary_stats(flat),
      "bin_counts": _bin_counts(flat, cfg),
      "nonzero_dir_hist": _nonzero_hist(values, cfg.none_thresh),
    },
    "per_direction": _per_direction_stats(values, cfg),
  }


def _sample_without_replacement(indices: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
  if num_samples <= 0 or indices.size == 0:
    return indices[:0]
  if indices.size <= num_samples:
    return indices
  perm = rng.permutation(indices.size)
  return indices[perm[:num_samples]]


def _candidate_stats(values: np.ndarray, cfg: StatsCfg) -> Dict[str, Any]:
  return analyze(values, cfg)


def _candidate_pair_stats(counts: List[int]) -> Dict[str, float]:
  if not counts:
    return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
  arr = np.asarray(counts, dtype=float)
  return {
    "mean": float(arr.mean()),
    "median": float(np.median(arr)),
    "min": float(arr.min()),
    "max": float(arr.max()),
  }


def main() -> None:
  args = parse_args()
  dataset_path = Path(args.dataset_path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)

  out_path = Path(args.out) if args.out else dataset_path.parent / "occlusion_stats.json"
  cfg = StatsCfg(none_thresh=args.none_thresh, high_thresh=args.high_thresh)
  default_weight = 0.1

  with dataset_path.open("r", encoding="utf-8") as f:
    graphs_all = json.load(f)
  total_scenes = len(graphs_all)

  graphs = _select_scenes(graphs_all, args.num_scenes, args.sample_random, args.seed)

  raw_list: List[List[float]] = []
  weighted_list: List[List[float]] = []
  fruit_extents: List[List[float]] = []
  leaf_extents: List[List[float]] = []
  fruit_radii: List[float] = []
  edge_dists: List[float] = []
  edge_dist_ratios: List[float] = []
  missing_raw = 0
  missing_weighted = 0
  len_mismatch = 0
  total_edges = 0
  weight_vals: List[float] = []

  node_by_id_list: List[Dict[Any, Dict[str, Any]]] = []
  for graph in graphs:
    nodes = graph.get("nodes", [])
    node_by_id = {n.get("id"): n for n in nodes}
    node_by_id_list.append(node_by_id)
    fruit_radius_by_id: Dict[Any, float] = {}
    for n in nodes:
      n_type = n.get("type")
      ext = n.get("obb", {}).get("extents")
      if ext is None:
        continue
      if n_type == "fruit":
        fruit_extents.append(ext)
        radius = 0.5 * float(np.linalg.norm(ext))
        fruit_radii.append(radius)
        fruit_radius_by_id[n.get("id")] = radius
      elif n_type == "leaf":
        leaf_extents.append(ext)

    occl = graph.get("occlusion") or {}
    if "weight" in occl:
      weight_vals.append(float(occl["weight"]))
    edges = occl.get("edges", [])
    for e in edges:
      total_edges += 1
      raw = e.get("raw_overlap")
      weighted = e.get("weighted_overlap")
      if raw is None:
        missing_raw += 1
        continue
      if weighted is None:
        missing_weighted += 1
        continue
      if len(raw) != len(weighted):
        len_mismatch += 1
        continue
      raw_list.append(raw)
      weighted_list.append(weighted)
      leaf_id = e.get("leaf_id")
      fruit_id = e.get("fruit_id")
      leaf = node_by_id.get(leaf_id)
      fruit = node_by_id.get(fruit_id)
      if leaf and fruit:
        lc = np.asarray(leaf.get("centroid", leaf.get("center", [0.0, 0.0, 0.0])), dtype=float)
        fc = np.asarray(fruit.get("centroid", fruit.get("center", [0.0, 0.0, 0.0])), dtype=float)
        dist = float(np.linalg.norm(lc - fc))
        edge_dists.append(dist)
        radius = fruit_radius_by_id.get(fruit_id)
        if radius is not None and radius > 1e-8:
          edge_dist_ratios.append(dist / radius)

  raw_arr = np.asarray(raw_list, dtype=float)
  weighted_arr = np.asarray(weighted_list, dtype=float)
  fruit_ext_arr = np.asarray(fruit_extents, dtype=float)
  leaf_ext_arr = np.asarray(leaf_extents, dtype=float)
  fruit_radius_arr = np.asarray(fruit_radii, dtype=float)
  dist_arr = np.asarray(edge_dists, dtype=float)
  dist_ratio_arr = np.asarray(edge_dist_ratios, dtype=float)

  weight_source = "default"
  weight_used = default_weight
  if args.weight is not None:
    weight_source = "cli"
    weight_used = float(args.weight)
  elif weight_vals:
    weight_source = "dataset"
    weight_used = float(np.median(np.asarray(weight_vals, dtype=float)))

  weight_mismatch = 0
  if weight_vals:
    for w in weight_vals:
      if abs(w - weight_used) > 1e-6:
        weight_mismatch += 1

  # Collapse +/- directions into axis-level magnitudes.
  raw_axes3 = _collapse_axes(raw_arr, "axes3")
  weighted_axes3 = _collapse_axes(weighted_arr, "axes3")
  raw_axes9 = _collapse_axes(raw_arr, "axes9")
  weighted_axes9 = _collapse_axes(weighted_arr, "axes9")

  ratio_quantiles = _quantiles(dist_ratio_arr)
  weighted_thresholds = {
    "weight": weight_used,
    "none": {},
    "high": {},
  }
  for key, ratio in ratio_quantiles.items():
    if np.isnan(ratio):
      weighted_thresholds["none"][key] = float("nan")
      weighted_thresholds["high"][key] = float("nan")
      continue
    weight_fac = 1.0 / (1.0 + weight_used * ratio * ratio)
    weighted_thresholds["none"][key] = float(args.none_thresh * weight_fac)
    weighted_thresholds["high"][key] = float(args.high_thresh * weight_fac)

  report = {
    "dataset_path": str(dataset_path),
    "num_scenes_total": total_scenes,
    "num_scenes_used": len(graphs),
    "total_edges_seen": total_edges,
    "edges_used": len(raw_list),
    "missing_raw": missing_raw,
    "missing_weighted": missing_weighted,
    "length_mismatch": len_mismatch,
    "weight_used": weight_used,
    "weight_source": weight_source,
    "weight_mismatch_count": weight_mismatch,
    "thresholds": {"none": cfg.none_thresh, "high": cfg.high_thresh},
    "geometry": {
      "fruit_extents": {
        "axis": _axis_stats(fruit_ext_arr),
        "norm": _summary_stats(np.linalg.norm(fruit_ext_arr, axis=1) if fruit_ext_arr.size else fruit_ext_arr),
      },
      "leaf_extents": {
        "axis": _axis_stats(leaf_ext_arr),
        "norm": _summary_stats(np.linalg.norm(leaf_ext_arr, axis=1) if leaf_ext_arr.size else leaf_ext_arr),
      },
      "fruit_radius": _summary_stats(fruit_radius_arr),
    },
    "distance": {
      "dist": _summary_stats(dist_arr),
      "dist_ratio": _summary_stats(dist_ratio_arr),
      "dist_ratio_quantiles": ratio_quantiles,
      "weighted_thresholds": weighted_thresholds,
    },
    "raw_axes3": analyze(raw_axes3, cfg),
    "raw_axes9": analyze(raw_axes9, cfg),
    "weighted_axes3": analyze(weighted_axes3, cfg),
    "weighted_axes9": analyze(weighted_axes9, cfg),
  }

  # Histogram outputs (all values + nonzero-only).
  if plt is None:
    print("matplotlib not available; skipping histogram PNG outputs.")
  elif raw_arr.size > 0 and weighted_arr.size > 0:
    out_dir = dataset_path.parent / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_flat = raw_arr.reshape(-1)
    weighted_flat = weighted_arr.reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(raw_flat, bins=80, color="#4c72b0", alpha=0.8)
    axes[0].set_title("Raw overlaps (all)")
    axes[0].set_xlabel("overlap")
    axes[0].set_ylabel("count")
    axes[1].hist(weighted_flat, bins=80, color="#dd8452", alpha=0.8)
    axes[1].set_title("Weighted overlaps (all)")
    axes[1].set_xlabel("overlap")
    fig.tight_layout()
    hist_all_path = out_dir / "occlusion_value_hist.png"
    fig.savefig(hist_all_path, dpi=200)
    plt.close(fig)

    raw_nz = raw_flat[raw_flat > 0.0]
    weighted_nz = weighted_flat[weighted_flat > 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(raw_nz, bins=80, color="#4c72b0", alpha=0.8)
    axes[0].set_title("Raw overlaps (>0)")
    axes[0].set_xlabel("overlap")
    axes[0].set_ylabel("count")
    axes[1].hist(weighted_nz, bins=80, color="#dd8452", alpha=0.8)
    axes[1].set_title("Weighted overlaps (>0)")
    axes[1].set_xlabel("overlap")
    fig.tight_layout()
    hist_nz_path = out_dir / "occlusion_value_hist_nonzero.png"
    fig.savefig(hist_nz_path, dpi=200)
    plt.close(fig)
    print(f"Saved occlusion histograms to {hist_all_path} and {hist_nz_path}")

  # Candidate handling diagnostics (simulate sampling).
  candidate_tests: List[Dict[str, Any]] = []
  none_pairs = _axis_pairs(str(args.none_mode).lower())

  # Precompute occlusion maps for fast lookup.
  occl_maps: List[Dict[tuple[int, int], List[float]]] = []
  for graph in graphs:
    occl = graph.get("occlusion") or {}
    edges = occl.get("edges", [])
    mapping: Dict[tuple[int, int], List[float]] = {}
    for e in edges:
      leaf_id = e.get("leaf_id")
      fruit_id = e.get("fruit_id")
      raw = e.get("raw_overlap")
      weighted = e.get("weighted_overlap")
      if raw is None or weighted is None:
        continue
      mapping[(leaf_id, fruit_id)] = [raw, weighted]
    occl_maps.append(mapping)

  for test_idx, test_cfg in enumerate(DEFAULT_CANDIDATE_TESTS):
    rng = np.random.default_rng(args.seed + test_idx)
    raw_vals: List[List[float]] = []
    weighted_vals: List[List[float]] = []
    cand_per_fruit: List[int] = []

    for graph, occl_map, node_by_id in zip(graphs, occl_maps, node_by_id_list):
      nodes = graph.get("nodes", [])
      leaf_ids = [n.get("id") for n in nodes if n.get("type") == "leaf"]
      fruit_ids = [n.get("id") for n in nodes if n.get("type") == "fruit"]
      if not leaf_ids or not fruit_ids:
        continue

      leaf_pos = np.asarray([node_by_id.get(lid, {}).get("centroid", [0.0, 0.0, 0.0]) for lid in leaf_ids], dtype=float)
      fruit_pos = np.asarray([node_by_id.get(fid, {}).get("centroid", [0.0, 0.0, 0.0]) for fid in fruit_ids], dtype=float)
      if leaf_pos.size == 0 or fruit_pos.size == 0:
        continue

      dists = np.linalg.norm(leaf_pos[:, None, :] - fruit_pos[None, :, :], axis=-1)

      for j, f_id in enumerate(fruit_ids):
        base_mask = dists[:, j] <= test_cfg["max_dist"]
        base_idx = np.where(base_mask)[0]
        if base_idx.size == 0:
          continue

        extra_idx = np.array([], dtype=int)
        if test_cfg["extra_neg_ratio"] > 0.0:
          n_extra = int(np.ceil(base_idx.size * test_cfg["extra_neg_ratio"]))
          ring_mask = (dists[:, j] > test_cfg["max_dist"]) & (dists[:, j] <= test_cfg["ring_max_dist"])
          far_mask = dists[:, j] > test_cfg["ring_max_dist"]
          ring = np.where(ring_mask)[0]
          far = np.where(far_mask)[0]
          n_ring = int(round(n_extra * test_cfg["ring_frac"]))
          n_far = max(0, n_extra - n_ring)
          ring_sample = _sample_without_replacement(ring, n_ring, rng)
          far_sample = _sample_without_replacement(far, n_far, rng)
          if ring_sample.size + far_sample.size > 0:
            extra_idx = np.concatenate([ring_sample, far_sample], axis=0)

        cand_idx = np.concatenate([base_idx, extra_idx], axis=0) if extra_idx.size > 0 else base_idx

        # Thinning (train-mode style)
        if test_cfg["thin_keep_prob"] < 1.0 and cand_idx.size > 0:
          keep = rng.random(cand_idx.size) < test_cfg["thin_keep_prob"]
          cand_idx = cand_idx[keep]
          if cand_idx.size == 0:
            cand_idx = base_idx[:1]

        kept = 0
        fallback = None

        for li in cand_idx.tolist():
          leaf_id = leaf_ids[li]
          key = (leaf_id, f_id)
          if key in occl_map:
            raw_vals_local, weighted_vals_local = occl_map[key]
          else:
            raw_vals_local = [0.0] * 18
            weighted_vals_local = [0.0] * 18
          if fallback is None:
            fallback = (raw_vals_local, weighted_vals_local)

          if test_cfg["none_keep_prob"] < 1.0:
            raw_arr_local = np.asarray(raw_vals_local, dtype=float)
            all_none = True
            for pos, neg in none_pairs:
              if max(raw_arr_local[pos], raw_arr_local[neg]) > cfg.none_thresh:
                all_none = False
                break
            if all_none and rng.random() >= test_cfg["none_keep_prob"]:
              continue

          raw_vals.append(raw_vals_local)
          weighted_vals.append(weighted_vals_local)
          kept += 1

        if kept == 0 and fallback is not None:
          raw_vals.append(fallback[0])
          weighted_vals.append(fallback[1])
          kept = 1

        cand_per_fruit.append(kept)

    raw_vals_arr = np.asarray(raw_vals, dtype=float)
    weighted_vals_arr = np.asarray(weighted_vals, dtype=float)

    def _slice(vals: np.ndarray, mode: str) -> np.ndarray:
      return _collapse_axes(vals, mode)

    candidate_tests.append(
      {
        "name": test_cfg["name"],
        "config": test_cfg,
        "candidate_pairs": int(raw_vals_arr.shape[0]),
        "candidates_per_fruit": _candidate_pair_stats(cand_per_fruit),
        "raw_axes3": _candidate_stats(_slice(raw_vals_arr, "axes3"), cfg),
        "raw_axes9": _candidate_stats(_slice(raw_vals_arr, "axes9"), cfg),
        "weighted_axes3": _candidate_stats(_slice(weighted_vals_arr, "axes3"), cfg),
        "weighted_axes9": _candidate_stats(_slice(weighted_vals_arr, "axes9"), cfg),
      }
    )

  report["candidate_tests"] = candidate_tests

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(f"Saved occlusion stats to {out_path}")


if __name__ == "__main__":
  main()
