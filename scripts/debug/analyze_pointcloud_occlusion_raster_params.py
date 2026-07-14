#!/usr/bin/env python3
"""Choose pixel_size + delta for pointcloud-based z-buffer occlusion.

We want parameters grounded in the dataset's point sampling resolution (4mm voxelized clouds).

This script samples fruits from scenes and, for each fruit + direction:
  - builds a fruit-local frame (same as OBB occlusion: +X -> stem in XY plane, +Z up)
  - projects fruit points and candidate leaf points onto (u,v) plane
  - builds a z-buffer (min depth) per pixel

It reports:
  - Fruit depth-map coverage vs pixel_size:
      coverage = (#pixels with fruit depth) / (#pixels inside fruit convex hull)
  - Occlusion value sensitivity vs delta (tolerance):
      occl(delta) = frac of fruit pixels where leaf depth is closer than fruit depth by delta
  - Depth margin stats dz = z_fruit - z_leaf for occluding pixels (delta=0),
    to see how much occlusion is near-tie and thus noisy under voxelization.

Outputs:
  - summary.json
  - plots/*.png
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
try:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: E402
except Exception:
  plt = None


def _load_json(path: Path) -> Dict[str, Any]:
  try:
    import orjson  # type: ignore

    return orjson.loads(path.read_bytes())
  except Exception:
    pass
  import json

  with path.open("r") as f:
    return json.load(f)


def _mkdir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
  import json

  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary(arr: np.ndarray) -> Dict[str, float]:
  if arr.size == 0:
    return {"count": 0}
  qs = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
  return {
    "count": int(arr.size),
    "mean": float(arr.mean()),
    "std": float(arr.std()),
    "min": float(arr.min()),
    "max": float(arr.max()),
    "q00": float(qs[0]),
    "q01": float(qs[1]),
    "q05": float(qs[2]),
    "q10": float(qs[3]),
    "q25": float(qs[4]),
    "q50": float(qs[5]),
    "q75": float(qs[6]),
    "q90": float(qs[7]),
    "q95": float(qs[8]),
    "q99": float(qs[9]),
    "q100": float(qs[10]),
    "zero_rate": float((arr == 0).mean()),
  }


def _plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 200, logy: bool = False) -> None:
  if plt is None or values.size == 0:
    return
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.hist(values, bins=bins)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("count")
  if logy:
    ax.set_yscale("log")
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


def _plot_box_by_group(groups: Dict[str, np.ndarray], out_path: Path, title: str, ylabel: str) -> None:
  if plt is None or not groups:
    return
  labels = list(groups.keys())
  data = [groups[k] for k in labels]
  fig = plt.figure(figsize=(12, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.boxplot(data, labels=labels, showfliers=False)
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


def _cross2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
  return a[0] * b[1] - a[1] * b[0]


def convex_hull(points_2d: np.ndarray) -> List[Tuple[float, float]]:
  """Monotonic chain convex hull. points_2d: [N,2]."""
  if points_2d.size == 0:
    return []
  if points_2d.shape[0] <= 1:
    return [(float(points_2d[0, 0]), float(points_2d[0, 1]))]

  pts = sorted({(float(p[0]), float(p[1])) for p in points_2d})
  if len(pts) <= 1:
    return pts

  def build_half(seq):
    half: List[Tuple[float, float]] = []
    for p in seq:
      while len(half) >= 2 and _cross2d(
        (half[-1][0] - half[-2][0], half[-1][1] - half[-2][1]),
        (p[0] - half[-1][0], p[1] - half[-1][1]),
      ) <= 0:
        half.pop()
      half.append(p)
    return half

  lower = build_half(pts)
  upper = build_half(reversed(pts))
  return lower[:-1] + upper[:-1]


def point_in_poly(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
  """Ray crossing test (even-odd). poly must be in order."""
  inside = False
  n = len(poly)
  if n < 3:
    return False
  j = n - 1
  for i in range(n):
    xi, yi = poly[i]
    xj, yj = poly[j]
    # Edge straddles y?
    if (yi > y) != (yj > y):
      # X coordinate of intersection of edge with ray y=const
      x_int = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
      if x < x_int:
        inside = not inside
    j = i
  return inside


def rasterized_hull_pixel_count(poly: Sequence[Tuple[float, float]], x0: float, y0: float, pix: float) -> int:
  if len(poly) < 3:
    return 0
  xs = [p[0] for p in poly]
  ys = [p[1] for p in poly]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  ix0 = int(math.floor((min_x - x0) / pix))
  ix1 = int(math.floor((max_x - x0) / pix))
  iy0 = int(math.floor((min_y - y0) / pix))
  iy1 = int(math.floor((max_y - y0) / pix))
  cnt = 0
  for iy in range(iy0, iy1 + 1):
    yc = y0 + (iy + 0.5) * pix
    for ix in range(ix0, ix1 + 1):
      xc = x0 + (ix + 0.5) * pix
      if point_in_poly(xc, yc, poly):
        cnt += 1
  return cnt


def build_occlusion_dirs_axes6() -> np.ndarray:
  base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  dirs = []
  for v in base:
    dirs.append(v)
    dirs.append(-v)
  return np.stack(dirs, axis=0)  # [6,3]


def build_dir_bases(dirs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  bases = []
  z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
  x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  for d in dirs:
    d = d.astype(np.float32)
    d = d / (np.linalg.norm(d) + 1e-12)
    tmp = z if abs(float(np.dot(d, z))) < 0.9 else x
    u = np.cross(tmp, d)
    if float(np.linalg.norm(u)) < 1e-6:
      tmp = x
      u = np.cross(tmp, d)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(d, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    bases.append((d, u.astype(np.float32), v.astype(np.float32)))
  return bases


def fruit_local_frame(fruit_center: np.ndarray, stem_center: np.ndarray) -> np.ndarray:
  x_vec = np.asarray([stem_center[0] - fruit_center[0], stem_center[1] - fruit_center[1], 0.0], dtype=np.float32)
  if float(np.linalg.norm(x_vec)) < 1e-6:
    x_vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  x_unit = x_vec / (np.linalg.norm(x_vec) + 1e-12)
  z_unit = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
  y_vec = np.cross(z_unit, x_unit)
  if float(np.linalg.norm(y_vec)) < 1e-6:
    y_vec = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
  y_unit = y_vec / (np.linalg.norm(y_vec) + 1e-12)
  x_unit = np.cross(y_unit, z_unit)
  x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-12)
  return np.vstack([x_unit, y_unit, z_unit]).astype(np.float32)  # rows


def zbuffer_min(keys: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Return unique keys + min z for each key. keys must be int64."""
  if keys.size == 0:
    return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
  order = np.argsort(keys)
  k = keys[order]
  zv = z[order].astype(np.float32, copy=False)
  uniq, idx = np.unique(k, return_index=True)
  zmin = np.minimum.reduceat(zv, idx)
  return uniq.astype(np.int64, copy=False), zmin.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analyze pixel_size + delta for pointcloud z-buffer occlusion.")
  p.add_argument("--data_root", required=True, help="Root folder containing data/scene_XXXXX_pc_4mm.npz and *_graph.json")
  p.add_argument("--out_dir", default=None, help="Output dir (default: analysis/pointcloud_occl_params_<ts>)")
  p.add_argument("--use_4mm", action="store_true", help="Use *_pc_4mm.npz (default)")
  p.add_argument("--max_scenes", type=int, default=50, help="Max scenes to scan")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--fruits_per_scene", type=int, default=8, help="How many fruits to sample per scene")
  p.add_argument("--dirs_mode", choices=["axes6"], default="axes6")
  p.add_argument("--pixel_sizes", default="0.004,0.008,0.012,0.016", help="Comma-separated pixel sizes in meters")
  p.add_argument("--deltas", default="0.0,0.002,0.004", help="Comma-separated delta tolerances in meters")
  p.add_argument("--max_dist", type=float, default=None, help="Leaf candidate radius (defaults to graph['occlusion']['max_dist'])")
  p.add_argument("--max_leaf_points", type=int, default=0, help="Optional cap: randomly sample this many leaf points per fruit (0=disable)")
  p.add_argument("--max_fruit_points", type=int, default=0, help="Optional cap: randomly sample this many fruit points per fruit (0=disable)")
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

  suffix = "_pc_4mm.npz" if args.use_4mm or True else "_pc.npz"
  pc_files = sorted(data_dir.glob(f"*{suffix}"))
  if not pc_files:
    raise SystemExit(f"No pointcloud npz files found under {data_dir} with suffix {suffix!r}")

  if args.max_scenes is not None and args.max_scenes < len(pc_files):
    if args.sample_random:
      pc_files = rng.sample(pc_files, k=args.max_scenes)
    else:
      pc_files = pc_files[: args.max_scenes]

  pixel_sizes = _parse_float_list(args.pixel_sizes)
  deltas = _parse_float_list(args.deltas)
  pixel_sizes = sorted({float(x) for x in pixel_sizes if x > 0})
  deltas = sorted({float(x) for x in deltas if x >= 0})
  if not pixel_sizes or not deltas:
    raise SystemExit("Need at least one pixel_size (>0) and delta (>=0).")

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / f"pointcloud_occl_params_{ts}"
  out_dir = out_dir.resolve()
  plots_dir = out_dir / "plots"
  _mkdir(plots_dir)

  if args.dirs_mode == "axes6":
    dirs = build_occlusion_dirs_axes6()
  else:
    raise SystemExit(f"Unsupported dirs_mode: {args.dirs_mode}")
  dir_bases = build_dir_bases(dirs)

  # Accumulators per pixel size
  coverage_by_pix: Dict[float, List[float]] = {p: [] for p in pixel_sizes}
  fruit_pixels_by_pix: Dict[float, List[int]] = {p: [] for p in pixel_sizes}
  hull_pixels_by_pix: Dict[float, List[int]] = {p: [] for p in pixel_sizes}

  occl_by_pix_delta: Dict[float, Dict[float, List[float]]] = {p: {d: [] for d in deltas} for p in pixel_sizes}
  diff_by_pix_delta: Dict[float, Dict[float, List[float]]] = {p: {d: [] for d in deltas if d > 0} for p in pixel_sizes}

  dz_by_pix: Dict[float, List[float]] = {p: [] for p in pixel_sizes}  # depth margins for occluding pixels (delta=0)

  # Bookkeeping
  num_scene_used = 0
  num_fruits_used = 0
  num_fruit_dirs = 0

  # Semantics: from dataset we observed {stem=1, leaf=2, ped=3, fruit=4, shoot=5}
  SEM_LEAF = 2
  SEM_FRUIT = 4

  for si, pc_path in enumerate(pc_files):
    scene_id = pc_path.name.split("_pc")[0]  # scene_XXXXX
    json_path = data_dir / f"{scene_id}_graph.json"
    if not json_path.exists():
      continue

    if (si + 1) % 25 == 0 or si == 0 or (si + 1) == len(pc_files):
      print(f"[{si+1:4d}/{len(pc_files)}] {scene_id}", flush=True)

    g = _load_json(json_path)
    occl = g.get("occlusion", {})
    max_dist = float(args.max_dist) if args.max_dist is not None else float(occl.get("max_dist", 0.2))

    nodes = g.get("nodes", [])
    fruits = [n for n in nodes if n.get("type") == "fruit"]
    leaves = [n for n in nodes if n.get("type") == "leaf"]
    if not fruits or not leaves:
      continue

    # attached_to mapping: child -> parent
    attached_to = {e["source"]: e["target"] for e in g.get("edges", []) if e.get("label") == "attached_to" and "source" in e and "target" in e}
    node_by_id = {n.get("id"): n for n in nodes}

    # Load pointcloud
    npz = np.load(pc_path)
    pts = npz["points"].astype(np.float32, copy=False)
    sem = npz["semantics"].astype(np.int32, copy=False)
    inst = npz["instances"].astype(np.int32, copy=False)

    # Build instance -> slice mapping for leaf+fruit points only (faster than per-instance np.where).
    mask_relevant = (sem == SEM_LEAF) | (sem == SEM_FRUIT)
    if not mask_relevant.any():
      continue
    idx_all = np.nonzero(mask_relevant)[0]
    inst_all = inst[idx_all]
    order = np.argsort(inst_all)
    inst_sorted = inst_all[order]
    idx_sorted = idx_all[order]
    uniq_inst, starts, counts = np.unique(inst_sorted, return_index=True, return_counts=True)
    inst_to_pos = {int(i): (int(s), int(c)) for i, s, c in zip(uniq_inst.tolist(), starts.tolist(), counts.tolist())}

    # Pre-extract leaf and fruit instance IDs from JSON (and centroids)
    leaf_iids = []
    leaf_centroids = []
    for n in leaves:
      iid = int(n.get("instance_id"))
      leaf_iids.append(iid)
      leaf_centroids.append(n.get("centroid"))
    leaf_centroids = np.asarray(leaf_centroids, dtype=np.float32)

    fruit_entries = []
    for n in fruits:
      fid = str(n.get("id"))
      iid = int(n.get("instance_id"))
      # Need stem for local frame
      ped_id = attached_to.get(fid)
      stem_id = attached_to.get(ped_id) if ped_id else None
      stem_node = node_by_id.get(stem_id) if stem_id else None
      if stem_node is None:
        continue
      fruit_entries.append((fid, iid, np.asarray(n.get("centroid"), dtype=np.float32), np.asarray(stem_node.get("centroid"), dtype=np.float32)))

    if not fruit_entries:
      continue

    # Sample fruits
    if args.fruits_per_scene > 0 and len(fruit_entries) > args.fruits_per_scene:
      fruit_entries = rng.sample(fruit_entries, k=args.fruits_per_scene)

    num_scene_used += 1

    for (fid, fruit_iid, fruit_center, stem_center) in fruit_entries:
      # Gather fruit points
      if fruit_iid not in inst_to_pos:
        continue
      s0, c0 = inst_to_pos[fruit_iid]
      fruit_idx = idx_sorted[s0 : s0 + c0]
      fruit_pts = pts[fruit_idx]
      if args.max_fruit_points and fruit_pts.shape[0] > args.max_fruit_points:
        take = np_rng.choice(fruit_pts.shape[0], size=args.max_fruit_points, replace=False)
        fruit_pts = fruit_pts[take]

      # Candidate leaves by centroid distance (world).
      dists = np.linalg.norm(leaf_centroids - fruit_center[None, :], axis=1)
      cand_mask = dists <= max_dist
      cand_leaf_iids = [leaf_iids[i] for i in np.nonzero(cand_mask)[0].tolist()]
      if not cand_leaf_iids:
        continue

      # Gather candidate leaf points (union)
      leaf_indices: List[np.ndarray] = []
      for liid in cand_leaf_iids:
        pos = inst_to_pos.get(liid)
        if pos is None:
          continue
        s1, c1 = pos
        leaf_indices.append(idx_sorted[s1 : s1 + c1])
      if not leaf_indices:
        continue
      leaf_idx = np.concatenate(leaf_indices, axis=0)
      leaf_pts = pts[leaf_idx]
      if args.max_leaf_points and leaf_pts.shape[0] > args.max_leaf_points:
        take = np_rng.choice(leaf_pts.shape[0], size=args.max_leaf_points, replace=False)
        leaf_pts = leaf_pts[take]

      # Fruit-local transform
      R = fruit_local_frame(fruit_center, stem_center)  # rows
      fruit_local = (R @ (fruit_pts - fruit_center).T).T
      leaf_local = (R @ (leaf_pts - fruit_center).T).T

      num_fruits_used += 1

      for (d, u, v) in dir_bases:
        # Project to plane
        x_f = fruit_local @ u
        y_f = fruit_local @ v
        z_f = fruit_local @ d
        x_l = leaf_local @ u
        y_l = leaf_local @ v
        z_l = leaf_local @ d

        # Build convex hull polygon from fruit points in (x,y)
        pts2 = np.stack([x_f, y_f], axis=1)
        if pts2.shape[0] > 5000:
          sel = np_rng.choice(pts2.shape[0], size=5000, replace=False)
          pts2 = pts2[sel]
        poly = convex_hull(pts2)
        if len(poly) < 3:
          continue

        # Use combined min as origin so fruit+leaf share key space (no negative indices).
        min_x = float(min(x_f.min(initial=0.0), x_l.min(initial=0.0)))
        min_y = float(min(y_f.min(initial=0.0), y_l.min(initial=0.0)))
        max_x = float(max(x_f.max(initial=0.0), x_l.max(initial=0.0)))
        max_y = float(max(y_f.max(initial=0.0), y_l.max(initial=0.0)))

        # Slight margin so boundary points aren't dropped due to float jitter.
        margin = 1e-6
        x0 = min_x - margin
        y0 = min_y - margin

        for pix in pixel_sizes:
          W = int(math.floor((max_x - x0) / pix)) + 1
          # H = int(math.floor((max_y - y0) / pix)) + 1  # not needed for key

          ix_f = np.floor((x_f - x0) / pix).astype(np.int32)
          iy_f = np.floor((y_f - y0) / pix).astype(np.int32)
          keys_f = (ix_f.astype(np.int64) + np.int64(W) * iy_f.astype(np.int64))
          kf, zf = zbuffer_min(keys_f, z_f.astype(np.float32, copy=False))
          if kf.size == 0:
            continue

          ix_l = np.floor((x_l - x0) / pix).astype(np.int32)
          iy_l = np.floor((y_l - y0) / pix).astype(np.int32)
          keys_l = (ix_l.astype(np.int64) + np.int64(W) * iy_l.astype(np.int64))
          kl, zl = zbuffer_min(keys_l, z_l.astype(np.float32, copy=False))

          # Coverage: z-buffer fruit pixels vs hull pixels.
          hull_px = rasterized_hull_pixel_count(poly, x0, y0, pix)
          if hull_px <= 0:
            continue
          fruit_px = int(kf.size)
          cov = float(fruit_px) / float(hull_px)
          coverage_by_pix[pix].append(cov)
          fruit_pixels_by_pix[pix].append(fruit_px)
          hull_pixels_by_pix[pix].append(int(hull_px))

          # Match leaf pixels to fruit pixels via searchsorted.
          if kl.size == 0:
            for delta in deltas:
              occl_by_pix_delta[pix][delta].append(0.0)
              if delta > 0:
                diff_by_pix_delta[pix][delta].append(0.0)
            continue

          order_l = np.argsort(kl)
          kl_s = kl[order_l]
          zl_s = zl[order_l]
          pos = np.searchsorted(kl_s, kf)
          match = (pos < kl_s.size) & (kl_s[np.minimum(pos, kl_s.size - 1)] == kf)
          leaf_z = np.full_like(zf, np.inf, dtype=np.float32)
          leaf_z[match] = zl_s[pos[match]]

          # occlusion: leaf closer than fruit by delta
          base_occl = (leaf_z < zf)  # delta=0
          dz = (zf - leaf_z)[base_occl]
          # Keep a capped sample of dz for this pix to avoid huge memory.
          if dz.size:
            if dz.size > 2000:
              take = np_rng.choice(dz.size, size=2000, replace=False)
              dz = dz[take]
            dz_by_pix[pix].extend(dz.astype(np.float32).tolist())

          occl0 = float(base_occl.mean())
          occl_by_pix_delta[pix][0.0].append(occl0)
          for delta in deltas:
            if delta == 0.0:
              continue
            occl = float((leaf_z < (zf - float(delta))).mean())
            occl_by_pix_delta[pix][delta].append(occl)
            diff_by_pix_delta[pix][delta].append(abs(occl - occl0))

          num_fruit_dirs += 1

  # Aggregate results
  summary: Dict[str, Any] = {
    "input": {
      "data_root": str(data_root),
      "data_dir": str(data_dir),
      "num_scenes_scanned": len(pc_files),
      "max_scenes": args.max_scenes,
      "fruits_per_scene": args.fruits_per_scene,
      "pixel_sizes": pixel_sizes,
      "deltas": deltas,
      "seed": args.seed,
      "dirs_mode": args.dirs_mode,
      "max_leaf_points": args.max_leaf_points,
      "max_fruit_points": args.max_fruit_points,
      "max_dist_override": args.max_dist,
    },
    "counts": {
      "num_scenes_used": num_scene_used,
      "num_fruits_used": num_fruits_used,
      "num_fruit_dirs": num_fruit_dirs,
    },
    "coverage": {},
    "occlusion": {},
    "delta_sensitivity": {},
    "depth_margin_dz(delta0)": {},
  }

  for pix in pixel_sizes:
    cov = np.asarray(coverage_by_pix[pix], dtype=np.float32)
    summary["coverage"][str(pix)] = {
      "coverage": _summary(cov),
      "fruit_pixels": _summary(np.asarray(fruit_pixels_by_pix[pix], dtype=np.float32)),
      "hull_pixels": _summary(np.asarray(hull_pixels_by_pix[pix], dtype=np.float32)),
    }

    summary["occlusion"][str(pix)] = {str(delta): _summary(np.asarray(occl_by_pix_delta[pix][delta], dtype=np.float32)) for delta in deltas}
    summary["delta_sensitivity"][str(pix)] = {str(delta): _summary(np.asarray(diff_by_pix_delta[pix][delta], dtype=np.float32)) for delta in deltas if delta > 0}
    summary["depth_margin_dz(delta0)"][str(pix)] = _summary(np.asarray(dz_by_pix[pix], dtype=np.float32))

  _save_json(out_dir / "summary.json", summary)

  # Plots
  if plt is not None:
    for pix in pixel_sizes:
      cov = np.asarray(coverage_by_pix[pix], dtype=np.float32)
      _plot_hist(cov, plots_dir / f"coverage_pix_{pix:.4f}.png", f"Fruit depth-map coverage @ pix={pix:.4f}m", "coverage", bins=200, logy=True)

      dz = np.asarray(dz_by_pix[pix], dtype=np.float32)
      if dz.size:
        _plot_hist(dz, plots_dir / f"dz_occluding_pix_{pix:.4f}.png", f"Depth margin dz=z_fruit-z_leaf for occluding pixels (delta=0) @ pix={pix:.4f}m", "dz (m)", bins=200, logy=True)

    # Box plots across pix sizes for quick comparison
    cov_groups = {f"{p:.4f}": np.asarray(coverage_by_pix[p], dtype=np.float32) for p in pixel_sizes}
    _plot_box_by_group(cov_groups, plots_dir / "coverage_box_by_pix.png", "Coverage distribution by pixel size", "coverage")

    for delta in deltas:
      groups = {f"{p:.4f}": np.asarray(occl_by_pix_delta[p][delta], dtype=np.float32) for p in pixel_sizes}
      _plot_box_by_group(groups, plots_dir / f"occlusion_box_delta_{delta:.3f}.png", f"Occlusion (union z-buffer) by pixel size (delta={delta:.3f}m)", "occlusion fraction")

    for delta in deltas:
      if delta <= 0:
        continue
      groups = {f"{p:.4f}": np.asarray(diff_by_pix_delta[p][delta], dtype=np.float32) for p in pixel_sizes}
      _plot_box_by_group(groups, plots_dir / f"delta_diff_box_{delta:.3f}.png", f"|occl(delta)-occl(0)| by pixel size (delta={delta:.3f}m)", "abs diff")

  print(f"Wrote: {out_dir / 'summary.json'}", flush=True)
  print(f"Plots: {plots_dir}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

