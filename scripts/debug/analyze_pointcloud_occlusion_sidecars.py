#!/usr/bin/env python3
"""Analyze pointcloud z-buffer occlusion sidecars.

Inputs are `.npz` sidecars produced by:
  scripts/dataset_generation/postprocess_pointcloud_occlusion_zbuffer.py

These sidecars store (per scene):
  - fruit_union: [F, D] union occlusion fraction per fruit and direction
  - pair_potential: [P, D] leaf->fruit "potential" occlusion (non-stealing)
  - pair_exclusive: [P, D] leaf->fruit exclusive/frontmost occlusion
  - pair_k_rank: [P, D, 3] k-buffer ranks (k=3); rank0 equals exclusive

This script aggregates distributions and a few task-design-relevant summaries:
  - Dominant-direction threshold sweep on fruit_union
  - Occlusion concentration: top-K exclusive leaves explain how much union?
  - Redundancy: how much of union has backup occluders (rank2/rank3)?
  - "Dust" sweep: threshold on potential, measure how much union remains if we
    keep only leaves with potential >= eps (measured via exclusive mass).

Optionally compares to the OBB-based union_raw stored in the original per-scene
graph JSONs (if --data_root is provided).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _load_json(path: Path) -> Dict[str, Any]:
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


def _quantiles(arr: np.ndarray, qs: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
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
    "zero_rate": float((arr == 0).mean()),
    **_quantiles(arr),
  }


def _hist_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
  if values.size == 0:
    return np.zeros((bins.size - 1,), dtype=np.int64)
  h, _ = np.histogram(values, bins=bins)
  return h.astype(np.int64, copy=False)


def _plot_hist_counts(counts: np.ndarray, bins: np.ndarray, out_path: Path, title: str, xlabel: str, logy: bool = False) -> None:
  if plt is None:
    return
  centers = 0.5 * (bins[:-1] + bins[1:])
  widths = bins[1:] - bins[:-1]
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


def _plot_scatter(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, max_points: int = 200_000) -> None:
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


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analyze pointcloud occlusion z-buffer sidecars.")
  p.add_argument("--sidecar_dir", required=True, help="Directory containing scene_XXXXX_occl_zbuf_*.npz")
  p.add_argument("--out_dir", default=None, help="Output dir (default: analysis/pointcloud_occlusion_sidecars_<ts>)")
  p.add_argument("--max_scenes", type=int, default=None, help="Optional cap on number of scenes")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes when max_scenes is set")
  p.add_argument("--seed", type=int, default=0)

  p.add_argument("--hist_bins", type=int, default=200, help="Histogram bins over [0,1]")
  p.add_argument("--tau_sweep_step", type=float, default=0.05, help="Step size for dominant-direction tau sweep")
  p.add_argument("--dominant_tau_list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7", help="Comma-separated union thresholds for 'dominant dir' counts")
  p.add_argument("--topk_list", default="1,3,5,10", help="Comma-separated K for top-K exclusive coverage")
  p.add_argument("--retain_eps_list", default="0,0.002,0.004,0.01,0.02,0.03,0.05", help="Comma-separated eps on potential for union retention analysis")
  p.add_argument("--union_tau_list", default="0.1,0.3,0.5,0.7", help="Comma-separated union thresholds for conditional retention summaries")

  # Optional OBB comparison.
  p.add_argument("--data_root", default=None, help="If set, compare against JSON OBB union_raw in <data_root>/data/scene_XXXXX_graph.json")
  p.add_argument("--compare_max_scenes", type=int, default=None, help="Optional cap for OBB comparison (if data_root is large)")
  return p.parse_args()


def _parse_float_list(s: str) -> List[float]:
  out: List[float] = []
  for part in (s or "").split(","):
    part = part.strip()
    if not part:
      continue
    out.append(float(part))
  return out


def _parse_int_list(s: str) -> List[int]:
  out: List[int] = []
  for part in (s or "").split(","):
    part = part.strip()
    if not part:
      continue
    out.append(int(part))
  return out


def _scene_id_from_sidecar(path: Path) -> str:
  # Expected: scene_XXXXX_occl_zbuf_k3_ps4mm_d4mm.npz
  return path.name.split("_occl_zbuf")[0]


def main() -> int:
  args = parse_args()
  rng = random.Random(args.seed)

  sidecar_dir = Path(args.sidecar_dir).expanduser().resolve()
  files = sorted(sidecar_dir.glob("scene_*_occl_zbuf_*.npz"))
  if not files:
    raise SystemExit(f"No sidecars found in {sidecar_dir}")

  if args.max_scenes is not None and args.max_scenes < len(files):
    if args.sample_random:
      files = rng.sample(files, k=args.max_scenes)
    else:
      files = files[: args.max_scenes]

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"pointcloud_occlusion_sidecars_{ts}"
  out_dir = out_dir.resolve()
  plots_dir = out_dir / "plots"
  _mkdir(plots_dir)

  bins = np.linspace(0.0, 1.0, int(args.hist_bins) + 1, dtype=np.float32)

  dominant_tau_list = _parse_float_list(args.dominant_tau_list)
  topk_list = sorted(set(_parse_int_list(args.topk_list)))
  retain_eps_list = sorted(set(_parse_float_list(args.retain_eps_list)))
  union_tau_list = sorted(set(_parse_float_list(args.union_tau_list)))
  tau_step = float(args.tau_sweep_step)
  tau_sweep = np.arange(0.0, 1.0 + 1e-9, tau_step, dtype=np.float32)

  # Accumulators
  fruit_union_vals: List[np.ndarray] = []
  fruit_union_dir_sums: Optional[np.ndarray] = None
  fruit_union_dir_counts: Optional[np.ndarray] = None
  fruit_pix_vals: List[np.ndarray] = []

  # Histograms for pair metrics (flattened over all pairs and dirs).
  hist_potential_all = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_exclusive_all = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_k1_all = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_k2_all = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_k3_all = np.zeros((bins.size - 1,), dtype=np.int64)

  hist_potential_nz = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_exclusive_nz = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_k2_nz = np.zeros((bins.size - 1,), dtype=np.int64)
  hist_k3_nz = np.zeros((bins.size - 1,), dtype=np.int64)

  total_pairs = 0
  total_fruits = 0
  total_pair_dirs = 0
  total_fruit_dirs = 0

  # Exclusive concentration + redundancy stats collected as raw arrays (fruit-dir level).
  topk_ratios: Dict[int, List[float]] = {k: [] for k in topk_list}
  min_k_for_90: List[int] = []

  ratio_k2_over_union: List[float] = []
  ratio_k3_over_union: List[float] = []
  k2_abs: List[float] = []
  k3_abs: List[float] = []

  # Potential dust sweep: retained union ratio (using exclusive mass) per eps.
  retained_union_ratio_by_eps: Dict[float, List[float]] = {eps: [] for eps in retain_eps_list}
  leafcount_by_eps: Dict[float, List[int]] = {eps: [] for eps in retain_eps_list}

  # Conditional retention histograms: for each eps, summarize retained/union only for directions where union >= tau.
  retained_ratio_hist_by_eps_tau: Dict[float, Dict[float, np.ndarray]] = {
    eps: {tau: np.zeros((bins.size - 1,), dtype=np.int64) for tau in union_tau_list} for eps in retain_eps_list
  }
  # Conditional leaf-count histograms (counts per fruit-direction where union >= tau):
  # index is leafcount clipped to [0, 255] (255 means >=255).
  leafcount_hist_by_eps_tau: Dict[float, Dict[float, np.ndarray]] = {
    eps: {tau: np.zeros((256,), dtype=np.int64) for tau in union_tau_list} for eps in retain_eps_list
  }

  # Consistency checks (assume constant across files).
  pixel_size_m = None
  delta_m = None
  max_dist_m = None
  dirs_ref: Optional[np.ndarray] = None

  # Optional OBB comparison
  obb_pc_union_flat: List[np.ndarray] = []
  obb_union_flat: List[np.ndarray] = []
  compare_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
  compare_data_dir = (compare_root / "data") if compare_root is not None else None
  compare_limit = int(args.compare_max_scenes) if args.compare_max_scenes is not None else None

  for fi, fpath in enumerate(files):
    z = np.load(fpath)
    if pixel_size_m is None:
      pixel_size_m = float(z["pixel_size_m"]) if "pixel_size_m" in z else None
      delta_m = float(z["delta_m"]) if "delta_m" in z else None
      max_dist_m = float(z["max_dist_m"]) if "max_dist_m" in z else None
      dirs_ref = z["dirs"].astype(np.float32, copy=False)
    else:
      if "pixel_size_m" in z and abs(float(z["pixel_size_m"]) - float(pixel_size_m)) > 1e-9:
        raise SystemExit(f"pixel_size mismatch in {fpath}")
      if "delta_m" in z and abs(float(z["delta_m"]) - float(delta_m)) > 1e-9:
        raise SystemExit(f"delta mismatch in {fpath}")
      if "max_dist_m" in z and abs(float(z["max_dist_m"]) - float(max_dist_m)) > 1e-6:
        raise SystemExit(f"max_dist mismatch in {fpath}")
      if dirs_ref is not None and "dirs" in z:
        dirs_cur = z["dirs"].astype(np.float32, copy=False)
        if dirs_cur.shape != dirs_ref.shape or np.max(np.abs(dirs_cur - dirs_ref)) > 1e-6:
          raise SystemExit(f"dirs mismatch in {fpath}")

    fruit_union = z["fruit_union"].astype(np.float32, copy=False)  # [F,D]
    fruit_pix = z["fruit_pixel_count"].astype(np.int32, copy=False)  # [F,D]
    total_fruits += int(fruit_union.shape[0])
    total_fruit_dirs += int(fruit_union.size)
    fruit_union_vals.append(fruit_union.reshape(-1))
    fruit_pix_vals.append(fruit_pix.reshape(-1).astype(np.float32))

    D = int(fruit_union.shape[1])
    if fruit_union_dir_sums is None:
      fruit_union_dir_sums = fruit_union.sum(axis=0).astype(np.float64)
      fruit_union_dir_counts = np.full((D,), fruit_union.shape[0], dtype=np.int64)
    else:
      fruit_union_dir_sums += fruit_union.sum(axis=0).astype(np.float64)
      fruit_union_dir_counts += np.full((D,), fruit_union.shape[0], dtype=np.int64)

    # Pair metrics.
    pair_potential = z["pair_potential"].astype(np.float32, copy=False)  # [P,D]
    pair_exclusive = z["pair_exclusive"].astype(np.float32, copy=False)  # [P,D]
    pair_k_rank = z["pair_k_rank"].astype(np.float32, copy=False)  # [P,D,3]
    P = int(pair_potential.shape[0])
    total_pairs += P
    total_pair_dirs += int(pair_potential.size)

    # Global histograms (all values incl zeros).
    hist_potential_all += _hist_counts(pair_potential.reshape(-1), bins)
    hist_exclusive_all += _hist_counts(pair_exclusive.reshape(-1), bins)
    hist_k1_all += _hist_counts(pair_k_rank[:, :, 0].reshape(-1), bins)
    hist_k2_all += _hist_counts(pair_k_rank[:, :, 1].reshape(-1), bins)
    hist_k3_all += _hist_counts(pair_k_rank[:, :, 2].reshape(-1), bins)

    # Non-zero hists (helpful because many directions are 0 for a given pair).
    pv = pair_potential.reshape(-1)
    ex = pair_exclusive.reshape(-1)
    k2 = pair_k_rank[:, :, 1].reshape(-1)
    k3 = pair_k_rank[:, :, 2].reshape(-1)
    if pv.size:
      hist_potential_nz += _hist_counts(pv[pv > 0], bins)
      hist_exclusive_nz += _hist_counts(ex[ex > 0], bins)
      hist_k2_nz += _hist_counts(k2[k2 > 0], bins)
      hist_k3_nz += _hist_counts(k3[k3 > 0], bins)

    # Per-fruit aggregation for concentration / redundancy / retention sweeps.
    fruit_iid = z["fruit_iid"].astype(np.int32, copy=False)
    pair_fruit_iid = z["pair_fruit_iid"].astype(np.int32, copy=False)

    # Group pair rows by fruit iid (within this scene).
    if P > 0 and fruit_iid.size > 0:
      # Sort by fruit iid for cheap slicing.
      order = np.argsort(pair_fruit_iid, kind="mergesort")
      pf_sorted = pair_fruit_iid[order]
      pot_sorted = pair_potential[order]
      ex_sorted = pair_exclusive[order]
      k_sorted = pair_k_rank[order]

      uniq, start_idx, counts = np.unique(pf_sorted, return_index=True, return_counts=True)
      fruit_to_union = {int(iid): fruit_union[j] for j, iid in enumerate(fruit_iid.tolist())}

      for iid, s, c in zip(uniq.tolist(), start_idx.tolist(), counts.tolist()):
        union_vec = fruit_to_union.get(int(iid))
        if union_vec is None:
          continue
        pot_mat = pot_sorted[s : s + c]  # [L,D]
        ex_mat = ex_sorted[s : s + c]    # [L,D]
        k_mat = k_sorted[s : s + c]      # [L,D,3]

        union_mask = union_vec > 0
        if not np.any(union_mask):
          continue

        # Redundancy: u2/u1 and u3/u1.
        u1 = union_vec[union_mask]
        u2 = k_mat[:, union_mask, 1].sum(axis=0)
        u3 = k_mat[:, union_mask, 2].sum(axis=0)
        k2_abs.extend(u2.astype(np.float32).tolist())
        k3_abs.extend(u3.astype(np.float32).tolist())
        ratio_k2_over_union.extend((u2 / (u1 + 1e-12)).astype(np.float32).tolist())
        ratio_k3_over_union.extend((u3 / (u1 + 1e-12)).astype(np.float32).tolist())

        # Exclusive concentration: top-K explains union.
        # For each direction independently, take top-K exclusive values.
        for di in np.nonzero(union_mask)[0].tolist():
          u = float(union_vec[di])
          col = ex_mat[:, di].astype(np.float32, copy=False)
          if col.size == 0 or u <= 0:
            continue
          # To avoid full sort for large L, use partial partition for max K.
          kmax = topk_list[-1]
          k_use = min(int(kmax), int(col.size))
          if k_use <= 0:
            continue
          # Largest k_use values (unsorted) then sort descending within that small set.
          part = np.partition(col, kth=col.size - k_use)[-k_use:]
          part_sorted = np.sort(part)[::-1]
          cums = np.cumsum(part_sorted)
          for k in topk_list:
            kk = min(int(k), int(cums.size))
            topk_ratios[k].append(float(cums[kk - 1] / u))
          # Minimal K to reach 90% union (cap at full leaf count).
          full_sorted = None
          # Only do a full sort when needed (rarely), based on k_use.
          if col.size <= 256:
            full_sorted = np.sort(col)[::-1]
            cs = np.cumsum(full_sorted)
            need = float(0.9 * u)
            k90 = int(np.searchsorted(cs, need, side="left") + 1)
            min_k_for_90.append(k90)

        # Potential eps retention sweep: keep leaves with potential>=eps, measure retained union via exclusive.
        # We compute per direction: retained = sum(exclusive * (potential>=eps)).
        for eps in retain_eps_list:
          mask = pot_mat >= float(eps)
          retained = (ex_mat * mask).sum(axis=0)
          ratio_full = retained / (union_vec + 1e-12)
          ratio = ratio_full[union_mask]
          retained_union_ratio_by_eps[eps].extend(ratio.astype(np.float32).tolist())
          leafcount = mask[:, union_mask].sum(axis=0)  # per dir
          leafcount_by_eps[eps].extend(leafcount.astype(np.int32).tolist())

          # Conditional: only directions with union >= tau.
          for tau in union_tau_list:
            m = union_vec >= float(tau)
            if not np.any(m):
              continue
            r = ratio_full[m].astype(np.float32, copy=False)
            # r is in [0,1] by construction.
            retained_ratio_hist_by_eps_tau[eps][tau] += _hist_counts(r, bins)

            # Leafcount distribution under the same condition.
            lc = mask[:, m].sum(axis=0).astype(np.int32, copy=False)
            if lc.size:
              lc_clip = np.clip(lc, 0, 255)
              leafcount_hist_by_eps_tau[eps][tau] += np.bincount(lc_clip, minlength=256).astype(np.int64, copy=False)

    # Optional OBB comparison for union_raw.
    if compare_root is not None and compare_data_dir is not None:
      if compare_limit is not None and len(obb_pc_union_flat) >= compare_limit:
        continue
      scene_id = _scene_id_from_sidecar(fpath)
      jpath = compare_data_dir / f"{scene_id}_graph.json"
      if not jpath.exists():
        continue
      g = _load_json(jpath)
      nodes = g.get("nodes", [])
      # map instance_id -> union_raw (18)
      inst_to_union: Dict[int, np.ndarray] = {}
      for n in nodes:
        if n.get("type") != "fruit":
          continue
        iid = n.get("instance_id")
        ot = n.get("occlusion_targets") or {}
        ur = ot.get("union_raw")
        if iid is None or ur is None:
          continue
        inst_to_union[int(iid)] = np.asarray(ur, dtype=np.float32)
      if not inst_to_union:
        continue
      # align
      pc_u = fruit_union
      iids = fruit_iid.tolist()
      obb_u = []
      pc_u_match = []
      for j, iid in enumerate(iids):
        uu = inst_to_union.get(int(iid))
        if uu is None:
          continue
        if uu.shape[0] != pc_u.shape[1]:
          continue
        obb_u.append(uu[None, :])
        pc_u_match.append(pc_u[j : j + 1])
      if obb_u and pc_u_match:
        obb_union_flat.append(np.concatenate(obb_u, axis=0).reshape(-1))
        obb_pc_union_flat.append(np.concatenate(pc_u_match, axis=0).reshape(-1))

    if (fi + 1) % 50 == 0 or fi == 0 or (fi + 1) == len(files):
      print(f"[{fi+1:4d}/{len(files)}]", fpath.name, flush=True)

  fruit_union_all = np.concatenate(fruit_union_vals, axis=0) if fruit_union_vals else np.asarray([], dtype=np.float32)
  fruit_pix_all = np.concatenate(fruit_pix_vals, axis=0) if fruit_pix_vals else np.asarray([], dtype=np.float32)

  # Direction means.
  dir_means: Optional[np.ndarray] = None
  if fruit_union_dir_sums is not None and fruit_union_dir_counts is not None:
    dir_means = (fruit_union_dir_sums / np.maximum(fruit_union_dir_counts, 1)).astype(np.float32)

  # Dominant-direction sweep (#dirs above tau per fruit).
  fruit_union_by_fruit = fruit_union_all.reshape(-1, int(dirs_ref.shape[0]) if dirs_ref is not None else -1) if fruit_union_all.size and dirs_ref is not None else None
  dominant_sweep: Dict[str, Dict[str, float]] = {}
  dominant_fixed: Dict[str, Dict[str, float]] = {}
  if fruit_union_by_fruit is not None:
    for tau in tau_sweep.tolist():
      cnt = (fruit_union_by_fruit > float(tau)).sum(axis=1).astype(np.float32)
      dominant_sweep[f"{tau:.3f}"] = _summary(cnt)
    for tau in dominant_tau_list:
      cnt = (fruit_union_by_fruit > float(tau)).sum(axis=1).astype(np.float32)
      dominant_fixed[f"{tau:.3f}"] = _summary(cnt)

  # Summaries for pair metrics via hist (approx) + basic exact stats on flattened arrays would be too big;
  # we keep hist + a few cheap scalars based on counts.
  def _hist_to_summary(counts: np.ndarray, bins: np.ndarray) -> Dict[str, float]:
    total = int(counts.sum())
    if total <= 0:
      return {"count": 0}
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = float((centers * counts).sum() / total)
    # Approx quantiles from cumulative counts.
    c = np.cumsum(counts)
    qmap = {}
    for q in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
      tgt = int(round(q * (total - 1)))
      idx = int(np.searchsorted(c, tgt, side="left"))
      idx = max(0, min(idx, centers.size - 1))
      qmap[f"q{int(q*100):02d}"] = float(centers[idx])
    return {"count": total, "mean": mean, **qmap}

  def _leafcount_hist_to_summary(counts: np.ndarray) -> Dict[str, float]:
    total = int(counts.sum())
    if total <= 0:
      return {"count": 0}
    xs = np.arange(counts.size, dtype=np.float32)
    mean = float((xs * counts).sum() / total)
    zero_rate = float(counts[0] / total)
    c = np.cumsum(counts)
    qmap = {}
    for q in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
      tgt = int(round(q * (total - 1)))
      idx = int(np.searchsorted(c, tgt, side="left"))
      idx = max(0, min(idx, counts.size - 1))
      qmap[f"q{int(q*100):02d}"] = float(idx)
    return {"count": total, "mean": mean, "zero_rate": zero_rate, **qmap}

  pair_metric_summary = {
    "potential_all": _hist_to_summary(hist_potential_all, bins),
    "exclusive_all": _hist_to_summary(hist_exclusive_all, bins),
    "k1_all": _hist_to_summary(hist_k1_all, bins),
    "k2_all": _hist_to_summary(hist_k2_all, bins),
    "k3_all": _hist_to_summary(hist_k3_all, bins),
    "potential_nz": _hist_to_summary(hist_potential_nz, bins),
    "exclusive_nz": _hist_to_summary(hist_exclusive_nz, bins),
    "k2_nz": _hist_to_summary(hist_k2_nz, bins),
    "k3_nz": _hist_to_summary(hist_k3_nz, bins),
  }

  summary: Dict[str, Any] = {
    "input": {
      "sidecar_dir": str(sidecar_dir),
      "num_scene_files": len(files),
      "pixel_size_m": pixel_size_m,
      "delta_m": delta_m,
      "max_dist_m": max_dist_m,
      "dirs_shape": list(dirs_ref.shape) if dirs_ref is not None else None,
      "data_root_compare": str(compare_root) if compare_root is not None else None,
    },
    "counts": {
      "total_fruits": total_fruits,
      "total_pairs": total_pairs,
      "total_fruit_dirs": total_fruit_dirs,
      "total_pair_dirs": total_pair_dirs,
    },
    "fruit_union": {
      "all_values": _summary(fruit_union_all),
      "axes6_values": _summary(fruit_union_all.reshape(-1, dirs_ref.shape[0])[:, :6].reshape(-1)) if (fruit_union_all.size and dirs_ref is not None and dirs_ref.shape[0] >= 6) else {"count": 0},
      "pixel_counts": _summary(fruit_pix_all),
      "dir_means": dir_means.tolist() if dir_means is not None else None,
    },
    "dominant_direction_counts_per_fruit": {
      "tau_sweep_step": tau_step,
      "sweep": dominant_sweep,
      "fixed": dominant_fixed,
    },
    "pair_metrics_hist": {
      "bins": bins.tolist(),
      "potential_all": hist_potential_all.tolist(),
      "exclusive_all": hist_exclusive_all.tolist(),
      "k1_all": hist_k1_all.tolist(),
      "k2_all": hist_k2_all.tolist(),
      "k3_all": hist_k3_all.tolist(),
      "potential_nz": hist_potential_nz.tolist(),
      "exclusive_nz": hist_exclusive_nz.tolist(),
      "k2_nz": hist_k2_nz.tolist(),
      "k3_nz": hist_k3_nz.tolist(),
    },
    "pair_metrics_summary_approx_from_hist": pair_metric_summary,
    "exclusive_concentration": {
      "topk_list": topk_list,
      "ratio_topk_over_union": {str(k): _summary(np.asarray(v, dtype=np.float32)) for k, v in topk_ratios.items()},
      "min_k_for_90pct_union(if_leaf_count<=256)": _summary(np.asarray(min_k_for_90, dtype=np.float32)),
    },
    "kbuffer_redundancy": {
      "k2_abs": _summary(np.asarray(k2_abs, dtype=np.float32)),
      "k3_abs": _summary(np.asarray(k3_abs, dtype=np.float32)),
      "k2_over_union": _summary(np.asarray(ratio_k2_over_union, dtype=np.float32)),
      "k3_over_union": _summary(np.asarray(ratio_k3_over_union, dtype=np.float32)),
    },
    "retain_union_by_potential_eps": {
      "eps_list": retain_eps_list,
      "retained_union_ratio": {str(eps): _summary(np.asarray(v, dtype=np.float32)) for eps, v in retained_union_ratio_by_eps.items()},
      "leafcount_per_fruit_dir": {str(eps): _summary(np.asarray(v, dtype=np.float32)) for eps, v in leafcount_by_eps.items()},
    },
    "retain_union_by_potential_eps_conditioned_on_union_tau": {
      "union_tau_list": union_tau_list,
      "ratio_summary_approx_from_hist": {
        str(eps): {str(tau): _hist_to_summary(h, bins) for tau, h in tau_map.items()} for eps, tau_map in retained_ratio_hist_by_eps_tau.items()
      },
      "leafcount_summary_from_hist": {
        str(eps): {str(tau): _leafcount_hist_to_summary(h) for tau, h in tau_map.items()} for eps, tau_map in leafcount_hist_by_eps_tau.items()
      },
    },
  }

  # Optional union_raw comparison.
  if obb_pc_union_flat and obb_union_flat:
    pc_flat = np.concatenate(obb_pc_union_flat, axis=0)
    obb_flat = np.concatenate(obb_union_flat, axis=0)
    if pc_flat.size and obb_flat.size:
      diff = pc_flat - obb_flat
      corr = float(np.corrcoef(pc_flat, obb_flat)[0, 1]) if pc_flat.size > 10 else float("nan")
      summary["compare_obb_union_raw"] = {
        "count": int(min(pc_flat.size, obb_flat.size)),
        "pc_union": _summary(pc_flat.astype(np.float32)),
        "obb_union_raw": _summary(obb_flat.astype(np.float32)),
        "pc_minus_obb": _summary(diff.astype(np.float32)),
        "pearson_corr": corr,
      }
      _plot_scatter(pc_flat, obb_flat, plots_dir / "pc_union_vs_obb_union_raw_scatter.png", "PC union vs OBB union_raw", "pc_union", "obb_union_raw")

  _save_json(out_dir / "summary.json", summary)

  # Plots (fruit union + pair metric hists + a few core analyses).
  _plot_hist_counts(_hist_counts(fruit_union_all, bins), bins, plots_dir / "fruit_union_hist.png", "Fruit union occlusion (PC z-buffer)", "union", logy=True)
  _plot_hist_counts(hist_potential_all, bins, plots_dir / "pair_potential_hist_all.png", "Pair potential occlusion (all values)", "potential", logy=True)
  _plot_hist_counts(hist_exclusive_all, bins, plots_dir / "pair_exclusive_hist_all.png", "Pair exclusive occlusion (all values)", "exclusive", logy=True)
  _plot_hist_counts(hist_k2_all, bins, plots_dir / "pair_k2_hist_all.png", "Pair k-rank2 occlusion (all values)", "k2", logy=True)
  _plot_hist_counts(hist_k3_all, bins, plots_dir / "pair_k3_hist_all.png", "Pair k-rank3 occlusion (all values)", "k3", logy=True)

  _plot_hist_counts(hist_potential_nz, bins, plots_dir / "pair_potential_hist_nonzero.png", "Pair potential occlusion (nonzero only)", "potential", logy=True)
  _plot_hist_counts(hist_exclusive_nz, bins, plots_dir / "pair_exclusive_hist_nonzero.png", "Pair exclusive occlusion (nonzero only)", "exclusive", logy=True)

  # Concentration histograms.
  for k in topk_list:
    vals = np.asarray(topk_ratios.get(k, []), dtype=np.float32)
    if vals.size:
      _plot_hist_counts(_hist_counts(vals, bins), bins, plots_dir / f"exclusive_top{k}_over_union_hist.png", f"Top-{k} exclusive mass / union", "ratio", logy=True)

  # Retained-union ratio vs eps plots.
  for eps in retain_eps_list:
    vals = np.asarray(retained_union_ratio_by_eps.get(eps, []), dtype=np.float32)
    if vals.size:
      _plot_hist_counts(_hist_counts(vals, bins), bins, plots_dir / f"retained_union_ratio_eps_{eps:.3f}.png", f"Retained union ratio (potential>= {eps:.3f})", "retained/union", logy=True)

  print(f"Wrote: {out_dir / 'summary.json'}", flush=True)
  print(f"Plots: {plots_dir}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
