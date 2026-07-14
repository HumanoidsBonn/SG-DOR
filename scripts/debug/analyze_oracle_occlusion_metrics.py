#!/usr/bin/env python3
"""Analyze oracle/pointcloud occlusion metrics stored in per-scene *_graph.json files.

This is intended for the "industrial_v2_800_bio_colors" style data layout:
  <data_root>/data/scene_XXXXX_graph.json

Outputs:
  - summary.json: key scalar stats + quantiles
  - plots/*.png: histograms + a few scatter plots

Focus:
  - Fruit GT: node["occlusion_targets"]["union_raw"]  (per fruit, per direction)
  - Pair GT:  graph["occlusion"]["edges"][*]["raw_overlap"], ["weighted_overlap"]
  - Distance vs magnitude, sparsity, "dominance" (union vs max-occluder), etc.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Matplotlib is optional but strongly preferred for saved histograms.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
try:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: E402
except Exception:
  plt = None


def _load_json(path: Path) -> Dict[str, Any]:
  # Prefer faster parsers if installed.
  try:
    import orjson  # type: ignore

    return orjson.loads(path.read_bytes())
  except Exception:
    pass
  try:
    import ujson  # type: ignore

    with path.open("r") as f:
      return ujson.load(f)
  except Exception:
    import json

    with path.open("r") as f:
      return json.load(f)


def _safe_np(a: Any, dtype: Any = np.float32) -> np.ndarray:
  try:
    return np.asarray(a, dtype=dtype)
  except Exception:
    return np.asarray([], dtype=dtype)


def _quantiles(arr: np.ndarray, qs: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
  if arr.size == 0:
    return {f"q{int(q*100):02d}": float("nan") for q in qs}
  out = {}
  vals = np.quantile(arr, list(qs))
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
    "zero_rate": float((arr == 0).mean()),
    **_quantiles(arr),
  }


def _mkdir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
  import json

  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hist_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
  if values.size == 0:
    return np.zeros((bins.size - 1,), dtype=np.int64)
  h, _ = np.histogram(values, bins=bins)
  return h.astype(np.int64)


def _plot_hist_counts(
  counts: np.ndarray,
  bins: np.ndarray,
  out_path: Path,
  title: str,
  xlabel: str,
  logy: bool = False,
) -> None:
  if plt is None:
    return
  centers = 0.5 * (bins[:-1] + bins[1:])
  widths = (bins[1:] - bins[:-1])
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.bar(centers, counts, width=widths, align="center", edgecolor="none")
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("count")
  if logy:
    ax.set_yscale("log")
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


def _plot_scatter(
  x: np.ndarray,
  y: np.ndarray,
  out_path: Path,
  title: str,
  xlabel: str,
  ylabel: str,
  max_points: int = 200_000,
) -> None:
  if plt is None or x.size == 0 or y.size == 0:
    return
  n = min(int(x.size), int(y.size))
  if n == 0:
    return
  if n > max_points:
    idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    xs = x[idx]
    ys = y[idx]
  else:
    xs = x[:n]
    ys = y[:n]
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(xs, ys, s=2, alpha=0.25)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


@dataclass
class DirRadiusStats:
  radius: float
  ratio_max_r_over_max_all: List[float]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analyze occlusion metrics in oracle pointcloud JSON graphs.")
  p.add_argument("--data_root", required=True, help="Root folder containing data/scene_XXXXX_graph.json")
  p.add_argument("--out_dir", default=None, help="Output directory (default: analysis/oracle_occlusion_metrics_<ts>)")
  p.add_argument("--max_scenes", type=int, default=None, help="Optional cap on number of scenes")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes (requires --max_scenes)")
  p.add_argument("--seed", type=int, default=0, help="Random seed (sampling + plots)")
  p.add_argument("--scene_glob", default="scene_*_graph.json", help="Glob pattern for per-scene graph JSONs")
  p.add_argument("--eps_list", default="0,1e-4,1e-3,1e-2,3e-2,1e-1", help="Comma-separated eps thresholds for 'significant occluder' stats")
  p.add_argument("--radii_list", default="0.05,0.1,0.15,0.2", help="Comma-separated radii for distance sensitivity (must be <= dataset max_dist)")
  p.add_argument("--dominant_tau_list", default="0.03,0.05,0.1,0.2", help="Comma-separated union_raw thresholds for 'dominant direction' counts")
  p.add_argument("--tau_sweep_step", type=float, default=0.01, help="Step size for union_raw threshold sweep in [0,1]")
  p.add_argument("--eps_sweep_step", type=float, default=0.01, help="Step size for max_leaf_raw threshold sweep in [0,1]")
  p.add_argument("--topk_list", default="1,3,5", help="Comma-separated top-K leaf counts for 'explains union' analysis (noisy-or)")
  return p.parse_args()


def _parse_float_list(s: str) -> List[float]:
  out: List[float] = []
  for part in (s or "").split(","):
    part = part.strip()
    if not part:
      continue
    out.append(float(part))
  return out


def main() -> int:
  args = parse_args()
  rng = random.Random(args.seed)
  np_rng = np.random.default_rng(args.seed)

  data_root = Path(args.data_root).expanduser().resolve()
  data_dir = data_root / "data"
  if not data_dir.exists():
    data_dir = data_root

  files = sorted(data_dir.glob(args.scene_glob))
  if not files:
    raise SystemExit(f"No graph JSONs found under {data_dir} with glob={args.scene_glob!r}")

  if args.max_scenes is not None and args.max_scenes < len(files):
    if args.sample_random:
      files = rng.sample(files, k=args.max_scenes)
    else:
      files = files[: args.max_scenes]

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"oracle_occlusion_metrics_{ts}"
  out_dir = out_dir.resolve()
  plots_dir = out_dir / "plots"
  _mkdir(plots_dir)

  eps_list = _parse_float_list(args.eps_list)
  radii_list = _parse_float_list(args.radii_list)
  dominant_tau_list = _parse_float_list(args.dominant_tau_list)
  topk_list = [int(x) for x in _parse_float_list(args.topk_list)]
  topk_list = sorted({k for k in topk_list if k >= 1})

  # Sweeps (for choosing thresholds).
  tau_sweep_step = float(args.tau_sweep_step)
  eps_sweep_step = float(args.eps_sweep_step)
  if tau_sweep_step <= 0 or eps_sweep_step <= 0:
    raise SystemExit("tau_sweep_step and eps_sweep_step must be > 0")
  tau_grid = np.arange(0.0, 1.0 + 1e-9, tau_sweep_step, dtype=np.float32)
  eps_grid = np.arange(0.0, 1.0 + 1e-9, eps_sweep_step, dtype=np.float32)

  # Global accumulators (kept compact).
  union_vals_all: List[np.ndarray] = []
  union_vals_axes6: List[np.ndarray] = []
  union_max_per_fruit: List[np.ndarray] = []
  fruit_radius_vals: List[np.ndarray] = []
  union_per_fruit: List[np.ndarray] = []  # [F, D] stacked later
  max_leaf_per_fruit: List[np.ndarray] = []  # [F, D] stacked later (max raw overlap per dir)

  edge_max_raw_all: List[np.ndarray] = []
  edge_max_weighted_all: List[np.ndarray] = []
  edge_dist_all: List[np.ndarray] = []
  edge_num_pos_dirs_all: List[np.ndarray] = []

  # Fruit-direction aggregates (only where max_raw > 0).
  ratio_union_over_max: List[np.ndarray] = []
  ratio_union_over_sum: List[np.ndarray] = []
  ratio_union_over_noisyor: List[np.ndarray] = []
  # How much of union is explained by top-K occluders (noisy-or topK / union).
  topk_explains_union: Dict[int, List[np.ndarray]] = {k: [] for k in topk_list}
  # How much of union is retained if we ignore per-leaf overlaps below eps (noisy-or_eps / union).
  eps_retains_union: Dict[float, List[np.ndarray]] = {eps: [] for eps in eps_list if eps > 0}

  max_raw_all_fruitdir: List[np.ndarray] = []
  union_all_fruitdir: List[np.ndarray] = []
  sum_raw_all_fruitdir: List[np.ndarray] = []
  noisy_or_all_fruitdir: List[np.ndarray] = []

  # Distance sensitivity: for each radius, keep ratios (max within r / max overall) for fruit-directions.
  radius_ratio_payload: Dict[float, List[np.ndarray]] = {r: [] for r in radii_list}

  # Dominant direction counts (per fruit) for various union thresholds.
  dominant_counts_by_tau: Dict[float, List[np.ndarray]] = {t: [] for t in dominant_tau_list}

  # Candidate leaf counts (all leaves within radius r per fruit, regardless of overlap).
  leaf_candidates_by_r: Dict[float, List[np.ndarray]] = {r: [] for r in radii_list}

  # High-level counts and sanity checks.
  num_scenes = 0
  num_scenes_missing_occl = 0
  num_fruits = 0
  num_leaves = 0
  num_edges_occl = 0

  occl_max_dist_values: List[float] = []
  occl_weight_values: List[float] = []
  dir_templates: Optional[np.ndarray] = None
  dir_max_abs_diff = 0.0

  # Hist-friendly derived stats.
  occluders_per_fruit: List[np.ndarray] = []
  occluders_per_fruit_eps: Dict[float, List[np.ndarray]] = {e: [] for e in eps_list}
  edge_pos_dirs_eps: Dict[float, List[np.ndarray]] = {e: [] for e in eps_list}

  # Main loop.
  for i, path in enumerate(files):
    if (i + 1) % 25 == 0 or i == 0 or (i + 1) == len(files):
      print(f"[{i+1:4d}/{len(files)}] {path.name}", flush=True)
    g = _load_json(path)
    num_scenes += 1

    nodes = g.get("nodes", [])
    fruits = [n for n in nodes if n.get("type") == "fruit"]
    leaves = [n for n in nodes if n.get("type") == "leaf"]
    num_fruits += len(fruits)
    num_leaves += len(leaves)

    # Candidate leaf counts by radius (spatial, not using occlusion edges).
    # This helps reason about subgraph budgets if we include all leaves within radius of each fruit.
    try:
      fruit_centroids = np.asarray([n.get("centroid") for n in fruits], dtype=np.float32)
      leaf_centroids = np.asarray([n.get("centroid") for n in leaves], dtype=np.float32)
      if fruit_centroids.ndim == 2 and fruit_centroids.shape[1] == 3 and leaf_centroids.ndim == 2 and leaf_centroids.shape[1] == 3:
        dmat = np.linalg.norm(fruit_centroids[:, None, :] - leaf_centroids[None, :, :], axis=-1)
        for r in radii_list:
          leaf_candidates_by_r[r].append((dmat <= float(r)).sum(axis=1).astype(np.int32))
    except Exception:
      pass

    occl = g.get("occlusion")
    if not isinstance(occl, dict):
      num_scenes_missing_occl += 1
      continue

    dirs = _safe_np(occl.get("dirs", []), dtype=np.float32)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
      num_scenes_missing_occl += 1
      continue
    if dir_templates is None:
      dir_templates = dirs
    else:
      if dirs.shape == dir_templates.shape:
        dir_max_abs_diff = max(dir_max_abs_diff, float(np.max(np.abs(dirs - dir_templates))))
      else:
        dir_max_abs_diff = float("nan")

    max_dist = float(occl.get("max_dist", float("nan")))
    weight = float(occl.get("weight", float("nan")))
    occl_max_dist_values.append(max_dist)
    occl_weight_values.append(weight)

    # Fruit union targets.
    fruit_ids: List[str] = []
    union_arr_list: List[List[float]] = []
    fruit_radius_list: List[float] = []
    for n in fruits:
      fid = str(n.get("id"))
      occl_t = n.get("occlusion_targets") or {}
      union = occl_t.get("union_raw", [])
      if not isinstance(union, list) or len(union) != int(dirs.shape[0]):
        continue
      fruit_ids.append(fid)
      union_arr_list.append([float(x) for x in union])

      # A rough scale proxy: 0.5*||extents|| if present.
      ext = None
      if isinstance(n.get("obb"), dict):
        ext = n["obb"].get("extents")
      if isinstance(ext, list) and len(ext) == 3:
        e = np.asarray(ext, dtype=np.float32)
        fruit_radius_list.append(float(0.5 * np.linalg.norm(e)))
      else:
        fruit_radius_list.append(float("nan"))

    if not fruit_ids:
      continue
    union_arr = np.asarray(union_arr_list, dtype=np.float32)  # [F, D]
    union_vals_all.append(union_arr.reshape(-1))
    union_vals_axes6.append(union_arr[:, : min(6, union_arr.shape[1])].reshape(-1))
    union_max_per_fruit.append(union_arr.max(axis=1))
    fruit_radius_vals.append(np.asarray(fruit_radius_list, dtype=np.float32))
    union_per_fruit.append(union_arr)

    for tau in dominant_tau_list:
      dominant_counts_by_tau[tau].append((union_arr > tau).sum(axis=1).astype(np.int32))

    # Pair edges.
    edges = occl.get("edges", [])
    if not isinstance(edges, list) or not edges:
      continue

    # Extract arrays for vectorized stats.
    try:
      dist_arr = np.asarray([float(e.get("dist", float("nan"))) for e in edges], dtype=np.float32)
      raw_arr = np.asarray([e.get("raw_overlap", []) for e in edges], dtype=np.float32)
      w_arr = np.asarray([e.get("weighted_overlap", []) for e in edges], dtype=np.float32)
      fruit_edge_ids = [str(e.get("fruit_id")) for e in edges]
    except Exception:
      continue

    if raw_arr.ndim != 2 or raw_arr.shape[1] != union_arr.shape[1]:
      continue
    if w_arr.shape != raw_arr.shape:
      continue

    num_edges_occl += int(raw_arr.shape[0])

    edge_dist_all.append(dist_arr)
    edge_max_raw = raw_arr.max(axis=1)
    edge_max_w = w_arr.max(axis=1)
    edge_max_raw_all.append(edge_max_raw)
    edge_max_weighted_all.append(edge_max_w)
    edge_num_pos_dirs_all.append((raw_arr > 0).sum(axis=1).astype(np.int16))

    # eps-based edge dir counts (learnability/imbalance proxy).
    for eps in eps_list:
      edge_pos_dirs_eps[eps].append((raw_arr > eps).sum(axis=1).astype(np.int16))

    # Map fruit_id -> index in union_arr
    fruit_id_to_idx = {fid: j for j, fid in enumerate(fruit_ids)}
    fruit_idx = np.asarray([fruit_id_to_idx.get(fid, -1) for fid in fruit_edge_ids], dtype=np.int32)
    valid = fruit_idx >= 0
    if not valid.any():
      continue

    fruit_idx = fruit_idx[valid]
    dist_arr = dist_arr[valid]
    raw_arr = raw_arr[valid]

    # Group edges by fruit efficiently (sort by fruit_idx once).
    order = np.argsort(fruit_idx)
    fruit_idx_s = fruit_idx[order]
    dist_s = dist_arr[order]
    raw_s = raw_arr[order]

    uniq, start, counts = np.unique(fruit_idx_s, return_index=True, return_counts=True)

    # per-fruit occluder counts (raw>0 in ANY direction, by construction of edges list)
    occluders_per_fruit.append(counts.astype(np.int32))

    # per-fruit occluders above eps (using edge max)
    edge_max_s = raw_s.max(axis=1)
    # We'll compute per-fruit by slicing (counts is per-fruit).
    for u, s0, c0 in zip(uniq, start, counts):
      sl = slice(int(s0), int(s0 + c0))
      em = edge_max_s[sl]
      for eps in eps_list:
        occluders_per_fruit_eps[eps].append(np.asarray([(em > eps).sum()], dtype=np.int32))

    # Fruit-direction aggregates + distance sensitivity
    # Also keep a per-fruit per-direction max leaf overlap array aligned with union_arr.
    max_raw_per_fruit = np.zeros_like(union_arr, dtype=np.float32)
    for u, s0, c0 in zip(uniq, start, counts):
      sl = slice(int(s0), int(s0 + c0))
      raw_f = raw_s[sl]  # [E_f, D]
      dist_f = dist_s[sl]  # [E_f]

      max_raw = raw_f.max(axis=0)
      sum_raw = raw_f.sum(axis=0)
      max_raw_per_fruit[int(u)] = max_raw.astype(np.float32, copy=False)

      # noisy-or approximation from per-leaf overlaps
      # Use float64 here: in float32, (1.0 - 1e-12) rounds back to 1.0, which
      # can produce log1p(-1.0) = -inf and spam warnings.
      raw_clip = np.clip(raw_f.astype(np.float64), 0.0, 1.0 - 1e-12)
      log_prod = np.log1p(-raw_clip).sum(axis=0)  # <= 0
      noisy_or = (1.0 - np.exp(log_prod)).astype(np.float32)

      union = union_arr[int(u)]

      # Only keep directions where there is any occlusion (max_raw>0).
      mask = max_raw > 0
      if not mask.any():
        continue

      union_m = union[mask]
      max_m = max_raw[mask]
      sum_m = sum_raw[mask]
      noisy_m = noisy_or[mask]

      union_all_fruitdir.append(union_m.astype(np.float32))
      max_raw_all_fruitdir.append(max_m.astype(np.float32))
      sum_raw_all_fruitdir.append(sum_m.astype(np.float32))
      noisy_or_all_fruitdir.append(noisy_m.astype(np.float32))

      ratio_union_over_max.append((union_m / (max_m + 1e-12)).astype(np.float32))
      ratio_union_over_sum.append((union_m / (sum_m + 1e-12)).astype(np.float32))
      ratio_union_over_noisyor.append((union_m / (noisy_m + 1e-12)).astype(np.float32))

      # Top-K explains union (via noisy-or on top-K raw overlaps per direction).
      # We compute this for directions with any occlusion (mask=max_raw>0).
      if topk_list:
        # raw_clip is float64 for stability; use it for partitioning too.
        for k in topk_list:
          kk = min(int(k), int(raw_clip.shape[0]))
          if kk <= 0:
            continue
          if kk == 1:
            noisy_topk = max_raw.astype(np.float32, copy=False)
          else:
            topk = np.partition(raw_clip, -kk, axis=0)[-kk:, :]  # [kk, D]
            noisy_topk = (1.0 - np.prod(1.0 - topk, axis=0)).astype(np.float32)
          topk_explains_union[k].append((noisy_topk[mask] / (union_m + 1e-12)).astype(np.float32))

      # Thresholding: how much union remains if we drop tiny per-leaf overlaps (<eps)?
      # Use noisy-or as a proxy for union (union/noisy-or is ~1 in this dataset).
      for eps in eps_list:
        if eps <= 0:
          continue
        m_eps = raw_clip >= float(eps)
        if not m_eps.any():
          eps_retains_union[eps].append(np.zeros_like(union_m, dtype=np.float32))
          continue
        log_prod_eps = (np.log1p(-raw_clip) * m_eps).sum(axis=0)
        noisy_eps = (1.0 - np.exp(log_prod_eps)).astype(np.float32)
        eps_retains_union[eps].append((noisy_eps[mask] / (union_m + 1e-12)).astype(np.float32))

      # Distance sensitivity: how much of max_raw can be recovered with only edges <= r.
      max_all = max_m
      for r in radii_list:
        m_r = dist_f <= r
        if not m_r.any():
          # no edges within radius -> ratio 0
          radius_ratio_payload[r].append(np.zeros_like(max_all, dtype=np.float32))
          continue
        max_r = raw_f[m_r].max(axis=0)[mask]
        radius_ratio_payload[r].append((max_r / (max_all + 1e-12)).astype(np.float32))

    max_leaf_per_fruit.append(max_raw_per_fruit)

  # Concatenate global arrays.
  def cat(xs: List[np.ndarray], dtype: Any = np.float32) -> np.ndarray:
    if not xs:
      return np.asarray([], dtype=dtype)
    return np.concatenate(xs).astype(dtype, copy=False)

  union_all = cat(union_vals_all)
  union_axes6 = cat(union_vals_axes6)
  union_max = cat(union_max_per_fruit)
  fruit_radius = cat(fruit_radius_vals)

  edge_max_raw = cat(edge_max_raw_all)
  edge_max_w = cat(edge_max_weighted_all)
  edge_dist = cat(edge_dist_all)
  edge_pos_dirs = cat(edge_num_pos_dirs_all, dtype=np.int16)

  ratio_um = cat(ratio_union_over_max)
  ratio_us = cat(ratio_union_over_sum)
  ratio_un = cat(ratio_union_over_noisyor)

  max_fd = cat(max_raw_all_fruitdir)
  union_fd = cat(union_all_fruitdir)
  sum_fd = cat(sum_raw_all_fruitdir)
  noisy_fd = cat(noisy_or_all_fruitdir)

  occl_per_fruit = cat(occluders_per_fruit, dtype=np.int32)

  # eps-based arrays
  edge_pos_dirs_by_eps = {eps: cat(edge_pos_dirs_eps[eps], dtype=np.int16) for eps in eps_list}
  occl_per_fruit_by_eps = {eps: cat(occluders_per_fruit_eps[eps], dtype=np.int32) for eps in eps_list}
  dom_counts_by_tau = {tau: cat(dominant_counts_by_tau[tau], dtype=np.int32) for tau in dominant_tau_list}
  radius_ratio_by_r = {r: cat(radius_ratio_payload[r], dtype=np.float32) for r in radii_list}
  leaf_candidates_by_r_cat = {r: cat(leaf_candidates_by_r[r], dtype=np.int32) for r in radii_list}

  # Align per-fruit arrays for sweeps / joint thresholding.
  union_pf = np.concatenate(union_per_fruit, axis=0).astype(np.float32, copy=False) if union_per_fruit else np.zeros((0, 0), dtype=np.float32)
  max_pf = np.concatenate(max_leaf_per_fruit, axis=0).astype(np.float32, copy=False) if max_leaf_per_fruit else np.zeros((0, 0), dtype=np.float32)
  if union_pf.size and max_pf.size and union_pf.shape != max_pf.shape:
    raise SystemExit(f"Internal error: union_pf shape {union_pf.shape} != max_pf shape {max_pf.shape}")

  # Union-dominance sweep: how many directions per fruit exceed tau?
  dominant_sweep: List[Dict[str, Any]] = []
  if union_pf.size:
    # counts: [F, T]
    dom_counts = (union_pf[:, :, None] > tau_grid[None, None, :]).sum(axis=1).astype(np.int16)
    for j, tau in enumerate(tau_grid.tolist()):
      c = dom_counts[:, j].astype(np.float32)
      dominant_sweep.append(
        {
          "tau": float(tau),
          "dir_pos_rate": float((union_pf.reshape(-1) > float(tau)).mean()),
          "mean_dirs_per_fruit": float(c.mean()),
          "q10": float(np.quantile(c, 0.10)),
          "q50": float(np.quantile(c, 0.50)),
          "q90": float(np.quantile(c, 0.90)),
          "frac_fruits_with_0": float((c == 0).mean()),
          "frac_fruits_with_all": float((c == union_pf.shape[1]).mean()),
        }
      )

  # Max-leaf sweep: for each eps, how many directions per fruit have a leaf with raw_overlap>eps?
  maxleaf_sweep: List[Dict[str, Any]] = []
  if max_pf.size:
    ml_counts = (max_pf[:, :, None] > eps_grid[None, None, :]).sum(axis=1).astype(np.int16)
    for j, eps in enumerate(eps_grid.tolist()):
      c = ml_counts[:, j].astype(np.float32)
      maxleaf_sweep.append(
        {
          "eps": float(eps),
          "dir_pos_rate": float((max_pf.reshape(-1) > float(eps)).mean()),
          "mean_dirs_per_fruit": float(c.mean()),
          "q10": float(np.quantile(c, 0.10)),
          "q50": float(np.quantile(c, 0.50)),
          "q90": float(np.quantile(c, 0.90)),
          "frac_fruits_with_0": float((c == 0).mean()),
          "frac_fruits_with_all": float((c == max_pf.shape[1]).mean()),
        }
      )

  # Joint sweep (limited): for each tau, what fraction of dominant dirs have a "relevant" occluder (max_leaf>eps)?
  # Use eps_list (not full grid) to keep JSON compact and the analysis interpretable.
  joint_dominant_summary: Dict[str, Any] = {}
  if union_pf.size and max_pf.size and eps_list:
    union_flat = union_pf.reshape(-1)
    max_flat = max_pf.reshape(-1)
    for eps in [e for e in eps_list if e > 0]:
      eps_key = str(eps)
      joint_dominant_summary[eps_key] = []
      for tau in tau_grid.tolist():
        dom = union_flat > float(tau)
        denom = float(dom.mean())
        if denom <= 0:
          joint_dominant_summary[eps_key].append({"tau": float(tau), "frac_dom_with_maxleaf_gt_eps": float("nan")})
          continue
        frac = float(((dom) & (max_flat > float(eps))).mean() / denom)
        joint_dominant_summary[eps_key].append({"tau": float(tau), "frac_dom_with_maxleaf_gt_eps": frac})

  # Plots
  np_rng = np.random.default_rng(args.seed)
  if plt is not None:
    bins01 = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    _plot_hist_counts(_hist_counts(union_all, bins01), bins01, plots_dir / "union_raw_all.png", "Fruit union_raw (all directions)", "union_raw", logy=True)
    _plot_hist_counts(_hist_counts(union_axes6, bins01), bins01, plots_dir / "union_raw_axes6.png", "Fruit union_raw (axes6 only)", "union_raw", logy=True)
    _plot_hist_counts(_hist_counts(union_max, bins01), bins01, plots_dir / "union_raw_max_per_fruit.png", "Fruit max(union_raw) across directions", "max union_raw", logy=True)

    # Edge distributions (max per edge is more learnability-relevant than all zeros)
    _plot_hist_counts(_hist_counts(edge_max_raw, bins01), bins01, plots_dir / "pair_max_raw_per_edge.png", "Pair GT: max(raw_overlap) per leaf-fruit edge", "max raw_overlap", logy=True)
    _plot_hist_counts(_hist_counts(edge_max_w, bins01), bins01, plots_dir / "pair_max_weighted_per_edge.png", "Pair GT: max(weighted_overlap) per leaf-fruit edge", "max weighted_overlap", logy=True)

    # Positive-dir counts
    bins_dirs = np.arange(-0.5, (int(dir_templates.shape[0]) if dir_templates is not None else 18) + 1.5, 1.0, dtype=np.float32)
    _plot_hist_counts(_hist_counts(edge_pos_dirs.astype(np.float32), bins_dirs), bins_dirs, plots_dir / "edge_num_pos_dirs_raw.png", "Per edge: #dirs with raw_overlap>0", "#positive dirs", logy=True)

    # Distance
    dmax = float(np.nanmax(edge_dist)) if edge_dist.size else 0.2
    bins_dist = np.linspace(0.0, max(dmax, 1e-6), 200, dtype=np.float32)
    _plot_hist_counts(_hist_counts(edge_dist, bins_dist), bins_dist, plots_dir / "edge_dist.png", "Occlusion edge dist (leaf center to fruit center)", "dist (m)", logy=True)

    # Ratios
    bins_ratio = np.linspace(0.0, 5.0, 201, dtype=np.float32)
    _plot_hist_counts(_hist_counts(ratio_um, bins_ratio), bins_ratio, plots_dir / "ratio_union_over_max.png", "Fruit-dir: union_raw / max_leaf_raw (>=1)", "union/max", logy=True)
    _plot_hist_counts(_hist_counts(ratio_us, bins01), bins01, plots_dir / "ratio_union_over_sum.png", "Fruit-dir: union_raw / sum_leaf_raw (<=1)", "union/sum", logy=True)
    _plot_hist_counts(_hist_counts(ratio_un, bins_ratio), bins_ratio, plots_dir / "ratio_union_over_noisyor.png", "Fruit-dir: union_raw / noisy_or(raw) (approx ~1 if match)", "union/noisy_or", logy=True)

    # Scatter: dist vs max_raw (subsample)
    _plot_scatter(edge_dist, edge_max_raw, plots_dir / "scatter_dist_vs_edge_max_raw.png", "dist vs max(raw_overlap) per edge", "dist (m)", "max raw_overlap")

    # Scatter: union vs max_raw at fruit-dir level
    _plot_scatter(max_fd, union_fd, plots_dir / "scatter_max_leaf_vs_union.png", "max leaf overlap vs union_raw (fruit-dir)", "max leaf raw_overlap", "union_raw")

    # Distance sensitivity hist per radius
    for r in radii_list:
      rr = radius_ratio_by_r.get(r, np.asarray([], dtype=np.float32))
      _plot_hist_counts(_hist_counts(rr, bins01), bins01, plots_dir / f"ratio_max_within_{r:.2f}_over_all.png", f"Fruit-dir: max_raw(dist<= {r:.2f}) / max_raw(all)", f"ratio @r={r:.2f}", logy=True)

    # Top-K explains union (noisy-or topK / union)
    for k, vals_list in topk_explains_union.items():
      vals = cat(vals_list)
      bins_explain = np.linspace(0.0, 1.2, 241, dtype=np.float32)
      _plot_hist_counts(_hist_counts(vals, bins_explain), bins_explain, plots_dir / f"top{k}_explains_union.png", f"Top-{k} leaves explain union (noisy-or top{k} / union_raw)", f"top{k} / union", logy=True)

    # eps retains union (noisy-or eps-filtered / union)
    for eps, vals_list in eps_retains_union.items():
      vals = cat(vals_list)
      bins_ret = np.linspace(0.0, 1.05, 211, dtype=np.float32)
      _plot_hist_counts(_hist_counts(vals, bins_ret), bins_ret, plots_dir / f"eps_{eps}_retains_union.png", f"Retained union if drop per-leaf raw< {eps} (noisy-or_eps / union_raw)", f"retained fraction @eps={eps}", logy=True)

    # Dominant direction counts per fruit
    bins_dom = np.arange(-0.5, (int(dir_templates.shape[0]) if dir_templates is not None else 18) + 1.5, 1.0, dtype=np.float32)
    for tau in dominant_tau_list:
      dc = dom_counts_by_tau.get(tau, np.asarray([], dtype=np.int32)).astype(np.float32)
      _plot_hist_counts(_hist_counts(dc, bins_dom), bins_dom, plots_dir / f"dominant_dir_count_tau_{tau:.3f}.png", f"Per fruit: #dirs with union_raw > {tau:.3f}", "#dominant dirs", logy=True)

  # Summaries
  summary: Dict[str, Any] = {
    "input": {
      "data_root": str(data_root),
      "data_dir": str(data_dir),
      "num_files_scanned": len(files),
      "max_scenes": args.max_scenes,
      "seed": args.seed,
    },
    "counts": {
      "num_scenes": num_scenes,
      "num_scenes_missing_occlusion": num_scenes_missing_occl,
      "num_fruits_total": num_fruits,
      "num_leaves_total": num_leaves,
      "num_occlusion_edges_total": num_edges_occl,
    },
    "occlusion_config": {
      "max_dist_unique": sorted({float(x) for x in occl_max_dist_values if not math.isnan(float(x))}),
      "weight_unique": sorted({float(x) for x in occl_weight_values if not math.isnan(float(x))}),
      "dir_max_abs_diff_vs_first_scene": float(dir_max_abs_diff),
      "num_dirs": int(dir_templates.shape[0]) if dir_templates is not None else None,
    },
    "union_raw": {
      "all_values": _summary(union_all),
      "axes6_values": _summary(union_axes6),
      "max_per_fruit": _summary(union_max),
      "dominant_sweep": dominant_sweep,
      "dominant_dir_count_by_tau": {str(tau): _summary(dom_counts_by_tau[tau].astype(np.float32)) for tau in dominant_tau_list},
    },
    "pair_gt": {
      "edge_dist": _summary(edge_dist),
      "edge_max_raw": _summary(edge_max_raw),
      "edge_max_weighted": _summary(edge_max_w),
      "edge_num_pos_dirs_raw_gt0": _summary(edge_pos_dirs.astype(np.float32)),
    },
    "fruit_dir_relationships": {
      "union_over_max": _summary(ratio_um),
      "union_over_sum": _summary(ratio_us),
      "union_over_noisy_or": _summary(ratio_un),
      "topk_explains_union(noisy_or_topk/union)": {str(k): _summary(cat(v).astype(np.float32)) for k, v in topk_explains_union.items()},
      "eps_retains_union(noisy_or_eps/union)": {str(eps): _summary(cat(v).astype(np.float32)) for eps, v in eps_retains_union.items()},
      "max_leaf_raw": _summary(max_fd),
      "union_raw_masked(max_raw>0)": _summary(union_fd),
      "sum_leaf_raw": _summary(sum_fd),
      "noisy_or(raw)": _summary(noisy_fd),
    },
    "occluders_per_fruit": {
      "any_overlap_edge_count": _summary(occl_per_fruit.astype(np.float32)),
    },
    "leaf_candidates_per_fruit": {
      str(r): _summary(leaf_candidates_by_r_cat[r].astype(np.float32)) for r in radii_list
    },
    "max_leaf_raw": {
      "maxleaf_sweep": maxleaf_sweep,
      "joint_dominant_frac_with_maxleaf_gt_eps": joint_dominant_summary,
    },
    "notes": [
      "Occlusion edges list contains ONLY leaf-fruit pairs with any raw_overlap>0 in at least one direction.",
      "union_raw is GT union-of-shadows; sum(raw_overlap) is an upper bound; max(raw_overlap) is a lower bound.",
      "Distance sensitivity ratios use max_raw computed from existing occlusion edges (already limited to dataset max_dist).",
    ],
  }

  # Add eps/tau/radius breakdowns
  summary["occluders_per_fruit"]["by_eps_edge_max_raw"] = {str(eps): _summary(occl_per_fruit_by_eps[eps].astype(np.float32)) for eps in eps_list}
  summary["pair_gt"]["edge_num_pos_dirs_raw_gt_eps"] = {str(eps): _summary(edge_pos_dirs_by_eps[eps].astype(np.float32)) for eps in eps_list}
  summary["distance_sensitivity"] = {str(r): _summary(radius_ratio_by_r[r]) for r in radii_list}

  # Conditional distance stats: where are the 'important' occluders located?
  summary["pair_gt"]["edge_dist_given_edge_max_raw_gt_eps"] = {}
  for eps in eps_list:
    m = edge_max_raw > float(eps)
    summary["pair_gt"]["edge_dist_given_edge_max_raw_gt_eps"][str(eps)] = _summary(edge_dist[m])

  # Conditional overlap stats by distance cutoff.
  summary["pair_gt"]["edge_max_raw_given_dist_le_r"] = {}
  for r in radii_list:
    m = edge_dist <= float(r)
    summary["pair_gt"]["edge_max_raw_given_dist_le_r"][str(r)] = _summary(edge_max_raw[m])

  # Fruit radius proxy (for visualization of distances)
  fruit_radius_clean = fruit_radius[~np.isnan(fruit_radius)]
  summary["fruit_radius_proxy"] = _summary(fruit_radius_clean) if fruit_radius_clean.size else {"count": 0}

  _save_json(out_dir / "summary.json", summary)

  # Write a quick human-oriented markdown too.
  lines: List[str] = []
  lines.append("# Oracle Occlusion Metrics Summary\n")
  lines.append(f"- data_root: `{data_root}`\n")
  lines.append(f"- scanned scenes: {num_scenes} (missing occlusion: {num_scenes_missing_occl})\n")
  lines.append(f"- total fruits: {num_fruits}\n")
  lines.append(f"- total leaves: {num_leaves}\n")
  lines.append(f"- total occlusion edges (leaf-fruit with any overlap): {num_edges_occl}\n")
  lines.append(f"- unique max_dist values: {summary['occlusion_config']['max_dist_unique']}\n")
  lines.append(f"- unique weight values: {summary['occlusion_config']['weight_unique']}\n")
  lines.append(f"- num_dirs: {summary['occlusion_config']['num_dirs']}\n")
  lines.append(f"- dir max abs diff vs first scene: {summary['occlusion_config']['dir_max_abs_diff_vs_first_scene']}\n")
  lines.append("\n## Quick Read\n")
  if fruit_radius_clean.size:
    rmean = summary["fruit_radius_proxy"].get("mean", float("nan"))
    lines.append(f"- fruit radius proxy (0.5*||extents||): mean ~ {rmean:.4f} m\n")
  lines.append(f"- union_raw(all): mean={summary['union_raw']['all_values'].get('mean', float('nan')):.4f}, zero_rate={summary['union_raw']['all_values'].get('zero_rate', float('nan')):.3f}\n")
  lines.append(f"- edge max raw_overlap: mean={summary['pair_gt']['edge_max_raw'].get('mean', float('nan')):.4f}, q95={summary['pair_gt']['edge_max_raw'].get('q95', float('nan')):.4f}\n")
  lines.append(f"- union/max ratio (fruit-dir): median={summary['fruit_dir_relationships']['union_over_max'].get('q50', float('nan')):.3f}, q90={summary['fruit_dir_relationships']['union_over_max'].get('q90', float('nan')):.3f}\n")
  lines.append("\n## Files\n")
  lines.append(f"- JSON: `{out_dir / 'summary.json'}`\n")
  lines.append(f"- plots: `{plots_dir}`\n")
  (out_dir / "summary.md").write_text("".join(lines))

  print(f"Wrote: {out_dir / 'summary.json'}", flush=True)
  print(f"Wrote: {out_dir / 'summary.md'}", flush=True)
  print(f"Plots: {plots_dir}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
