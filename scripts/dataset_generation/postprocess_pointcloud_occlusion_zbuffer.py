#!/usr/bin/env python3
"""Postprocess pointcloud datasets with z-buffer occlusion targets (binary sidecar).

This computes pointcloud-based occlusion signals for each fruit and its nearby leaves
using a 2.5D z-buffer in a fruit-local coordinate frame (same semantics as the OBB
occlusion dirs stored in the JSON graph):
  - Fruit-local frame: +X points toward the fruit's stem (XY plane), +Z is world up.
  - Directions: uses graph["occlusion"]["dirs"] (18) when present to preserve ordering.

Metrics (per fruit, per direction):
  - union_occlusion: fraction of fruit pixels occluded by any candidate leaf.

Metrics (per leaf-fruit pair, per direction):
  1) potential_occlusion: fraction of fruit pixels where *this leaf* is in front of the fruit
     (does NOT "steal" credit from other leaves).
  2) exclusive_occlusion: fraction of fruit pixels where this leaf is the *frontmost* occluder.
  3) k-buffer occlusion (k=3): fraction of fruit pixels where this leaf is the 1st/2nd/3rd
     closest occluding leaf (depth-rank layers).

Storage:
  - Writes a per-scene `.npz` sidecar next to the existing `scene_XXXXX_graph.json` and
    `scene_XXXXX_pc_4mm.npz` files. This avoids bloating JSON.

Defaults are grounded in the 4mm voxelized pointclouds:
  - pixel_size=0.004m (4mm)
  - delta=0.004m (4mm) depth tolerance for the in-front test
"""

from __future__ import annotations

import argparse
import math
import random
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
  from tqdm import tqdm  # type: ignore
except Exception:
  tqdm = None


def _load_json(path: Path) -> Dict[str, Any]:
  try:
    import orjson  # type: ignore

    return orjson.loads(path.read_bytes())
  except Exception:
    pass
  import json

  with path.open("r") as f:
    return json.load(f)


def fruit_local_frame(fruit_center: np.ndarray, stem_center: np.ndarray) -> np.ndarray:
  """Return 3x3 rotation matrix (rows are x,y,z) for fruit-local frame."""
  x_vec = np.asarray([stem_center[0] - fruit_center[0], stem_center[1] - fruit_center[1], 0.0], dtype=np.float32)
  if float(np.linalg.norm(x_vec)) < 1e-6:
    x_vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  x_unit = x_vec / (np.linalg.norm(x_vec) + 1e-12)

  z_unit = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
  y_vec = np.cross(z_unit, x_unit)
  if float(np.linalg.norm(y_vec)) < 1e-6:
    y_vec = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
  y_unit = y_vec / (np.linalg.norm(y_vec) + 1e-12)

  # Re-orthogonalize x to avoid drift.
  x_unit = np.cross(y_unit, z_unit)
  x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-12)
  return np.vstack([x_unit, y_unit, z_unit]).astype(np.float32)  # rows


def build_dir_bases(dirs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """For each direction d, build an orthonormal basis (d, u, v) with u,v spanning plane orthogonal to d."""
  bases: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
  x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  for d in dirs:
    d = d.astype(np.float32, copy=False)
    d = d / (np.linalg.norm(d) + 1e-12)
    tmp = z if abs(float(np.dot(d, z))) < 0.9 else x
    u = np.cross(tmp, d)
    if float(np.linalg.norm(u)) < 1e-6:
      tmp = x
      u = np.cross(tmp, d)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(d, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    bases.append((d.astype(np.float32), u.astype(np.float32), v.astype(np.float32)))
  return bases


def build_occlusion_dirs() -> np.ndarray:
  """Return the 18 canonical directions (axes +/- + diagonals)."""
  base = [
    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
  ]
  dirs: List[np.ndarray] = []
  for v in base:
    dirs.append(v)
    dirs.append(-v)
  for i, j in [(0, 1), (0, 2), (1, 2)]:
    for s1 in (-1.0, 1.0):
      for s2 in (-1.0, 1.0):
        v = base[i] * float(s1) + base[j] * float(s2)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs.append(v.astype(np.float32))
  return np.stack(dirs, axis=0).astype(np.float32, copy=False)


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


def _parse_scene_ids(s: Optional[str]) -> Optional[List[str]]:
  if not s:
    return None
  out: List[str] = []
  for part in s.split(","):
    part = part.strip()
    if not part:
      continue
    out.append(part)
  return out or None


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Compute pointcloud z-buffer occlusion targets and write NPZ sidecars.")
  p.add_argument("--data_root", required=True, help="Root folder containing data/scene_XXXXX_pc_4mm.npz and *_graph.json")
  p.add_argument("--out_dir", default=None, help="Output directory for sidecars (default: <data_root>/data)")
  p.add_argument("--use_4mm", action="store_true", help="Use *_pc_4mm.npz (default)")
  p.add_argument("--pixel_size", type=float, default=0.004, help="Raster pixel size in meters (default: 0.004)")
  p.add_argument("--delta", type=float, default=0.004, help="Depth tolerance in meters for in-front test (default: 0.004)")
  p.add_argument("--k", type=int, default=3, help="K for k-buffer ranks (default: 3; only 3 is supported)")
  p.add_argument("--max_dist", type=float, default=None, help="Leaf candidate radius (defaults to graph['occlusion']['max_dist'] or 0.2)")
  p.add_argument("--max_scenes", type=int, default=None, help="Max scenes to process")
  p.add_argument("--sample_random", action="store_true", help="Randomly sample scenes when max_scenes is set")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--scene_ids", default=None, help="Comma-separated list like scene_00000,scene_00012 to restrict processing")
  p.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecars")
  p.add_argument("--drop_zero_pairs", action="store_true", help="Drop leaf-fruit rows where all metrics are 0 (default)")
  p.add_argument("--keep_zero_pairs", action="store_true", help="Keep leaf-fruit rows even when all metrics are 0")
  p.add_argument("--max_leaf_points", type=int, default=0, help="Optional cap: randomly sample this many leaf points per fruit (0=disable)")
  p.add_argument("--max_fruit_points", type=int, default=0, help="Optional cap: randomly sample this many fruit points per fruit (0=disable)")
  return p.parse_args()


def main() -> int:
  args = parse_args()
  print("Starting pointcloud occlusion z-buffer postprocess...", flush=True)
  if args.k != 3:
    raise SystemExit("--k other than 3 is not supported yet (requested k=%d)" % int(args.k))
  pix = float(args.pixel_size)
  delta = float(args.delta)
  if pix <= 0:
    raise SystemExit("--pixel_size must be > 0")
  if delta < 0:
    raise SystemExit("--delta must be >= 0")

  drop_zero_pairs = True
  if args.keep_zero_pairs:
    drop_zero_pairs = False
  if args.drop_zero_pairs:
    drop_zero_pairs = True

  rng = random.Random(args.seed)
  np_rng = np.random.default_rng(args.seed)

  data_root = Path(args.data_root).expanduser().resolve()
  data_dir = data_root / "data"
  if not data_dir.exists():
    data_dir = data_root

  out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else data_dir
  # Some datasets live in shared/read-only folders (e.g. /home/data/...). If the caller
  # didn't specify an explicit output folder, fall back to a local writable analysis dir.
  if args.out_dir is None and (not out_dir.exists() or not os.access(out_dir, os.W_OK)):
    out_dir = (Path("analysis") / f"occl_zbuf_sidecars_{data_root.name}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  scene_ids = _parse_scene_ids(args.scene_ids)

  suffix = "_pc_4mm.npz" if args.use_4mm else "_pc.npz"
  print(f"Scanning for pointcloud files under {data_dir} (suffix={suffix!r})...", flush=True)
  pc_files = sorted(data_dir.glob(f"scene_*{suffix}"))
  if scene_ids is not None:
    want = set(scene_ids)
    pc_files = [p for p in pc_files if p.name.split("_pc")[0] in want]
  if not pc_files:
    raise SystemExit(f"No pointcloud npz files found under {data_dir} with suffix {suffix!r}")

  if args.max_scenes is not None and args.max_scenes < len(pc_files):
    if args.sample_random:
      pc_files = rng.sample(pc_files, k=args.max_scenes)
    else:
      pc_files = pc_files[: args.max_scenes]

  ps_mm = int(round(pix * 1000.0))
  d_mm = int(round(delta * 1000.0))

  print(
    "Pointcloud occlusion z-buffer postprocess\n"
    f"  data_dir: {data_dir}\n"
    f"  out_dir:  {out_dir}\n"
    f"  scenes:   {len(pc_files)}\n"
    f"  pc:       {suffix}\n"
    f"  pixel:    {pix:.6f} m ({ps_mm} mm)\n"
    f"  delta:    {delta:.6f} m ({d_mm} mm)\n"
    f"  k:        {args.k}\n"
    f"  max_dist: {'(from graph)' if args.max_dist is None else f'{args.max_dist:.3f} m (override)'}\n"
    f"  drop_zero_pairs: {drop_zero_pairs}\n",
    flush=True,
  )

  # Semantics: industrial dataset uses {stem=1, leaf=2, ped=3, fruit=4, shoot=5}
  SEM_LEAF = 2
  SEM_FRUIT = 4

  skipped_missing_json = 0
  skipped_exists = 0
  total_fruits = 0
  total_pairs = 0
  printed_first_scene_stats = False

  num_scenes_written = 0
  pbar = (
    tqdm(
      pc_files,
      desc="Scenes",
      unit="scene",
      smoothing=0.05,
      dynamic_ncols=True,
      file=sys.stdout,
      mininterval=0.25,
    )
    if tqdm is not None
    else None
  )
  for si, pc_path in enumerate(pbar or pc_files):
    scene_id = pc_path.name.split("_pc")[0]  # scene_XXXXX
    if pbar is not None:
      # Show the current scene immediately (even while it's still processing).
      pbar.set_postfix(
        scene=scene_id,
        written=num_scenes_written,
        skip_exists=skipped_exists,
        skip_nojson=skipped_missing_json,
        fruits=total_fruits,
        pairs=total_pairs,
      )
      pbar.refresh()
    json_path = data_dir / f"{scene_id}_graph.json"
    if not json_path.exists():
      skipped_missing_json += 1
      continue

    out_path = out_dir / f"{scene_id}_occl_zbuf_k3_ps{ps_mm}mm_d{d_mm}mm.npz"
    if out_path.exists() and not args.overwrite:
      skipped_exists += 1
      continue

    if pbar is None and ((si + 1) % 25 == 0 or si == 0 or (si + 1) == len(pc_files)):
      print(f"[{si+1:4d}/{len(pc_files)}] {scene_id}", flush=True)

    t0_scene = time.time()
    g = _load_json(json_path)
    occl = g.get("occlusion", {}) or {}
    max_dist = float(args.max_dist) if args.max_dist is not None else float(occl.get("max_dist", 0.2))

    # Direction ordering: prefer graph-provided dirs to preserve index mapping.
    dirs_raw = occl.get("dirs")
    if not dirs_raw:
      # Real-data graphs may not carry an occlusion block; fall back to canonical ordering.
      dirs = build_occlusion_dirs()
    else:
      dirs = np.asarray(dirs_raw, dtype=np.float32)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
      raise SystemExit(f"{json_path}: invalid dirs array shape {dirs.shape}")
    if int(dirs.shape[0]) != 18:
      raise SystemExit(f"{json_path}: expected 18 dirs, got {dirs.shape}")
    dir_bases = build_dir_bases(dirs)
    num_dirs = int(dirs.shape[0])

    nodes = g.get("nodes", [])
    if not nodes:
      continue
    fruits = [n for n in nodes if n.get("type") == "fruit"]
    leaves = [n for n in nodes if n.get("type") == "leaf"]
    if not fruits or not leaves:
      continue

    # attached_to mapping: child -> parent
    attached_to = {e["source"]: e["target"] for e in g.get("edges", []) if e.get("label") == "attached_to" and "source" in e and "target" in e}
    node_by_id = {n.get("id"): n for n in nodes}

    leaf_iids_all: List[int] = []
    leaf_centroids_all: List[List[float]] = []
    for n in leaves:
      iid = n.get("instance_id")
      cen = n.get("centroid")
      if iid is None or cen is None:
        continue
      leaf_iids_all.append(int(iid))
      leaf_centroids_all.append(cen)
    if not leaf_iids_all:
      continue
    leaf_centroids_all_arr = np.asarray(leaf_centroids_all, dtype=np.float32)

    # Load pointcloud.
    npz = np.load(pc_path)
    pts = npz["points"].astype(np.float32, copy=False)
    sem = npz["semantics"].astype(np.int32, copy=False)
    inst = npz["instances"].astype(np.int32, copy=False)

    # Build instance -> slice mapping for leaf+fruit points only.
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

    fruit_iids_scene: List[int] = []
    fruit_union_scene: List[np.ndarray] = []
    fruit_pix_scene: List[np.ndarray] = []

    pair_fruit_iid: List[int] = []
    pair_leaf_iid: List[int] = []
    pair_dist: List[float] = []
    pair_potential: List[np.ndarray] = []
    pair_exclusive: List[np.ndarray] = []
    pair_k_rank: List[np.ndarray] = []  # [D,3]

    fruit_iter = fruits
    fruit_pbar = None
    if tqdm is not None and pbar is not None and fruits:
      fruit_pbar = tqdm(
        fruits,
        desc=f"{scene_id} fruits",
        unit="fruit",
        leave=False,
        position=1,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
      )
      fruit_iter = fruit_pbar

    for fruit_node in fruit_iter:
      fid = str(fruit_node.get("id"))
      fruit_iid = fruit_node.get("instance_id")
      fruit_center = fruit_node.get("centroid")
      if fruit_iid is None or fruit_center is None:
        continue
      fruit_iid = int(fruit_iid)
      fruit_center = np.asarray(fruit_center, dtype=np.float32)

      # Need stem for local frame: fruit -> peduncle -> stem.
      ped_id = attached_to.get(fid)
      stem_id = attached_to.get(ped_id) if ped_id else None
      stem_node = node_by_id.get(stem_id) if stem_id else None
      if stem_node is None or stem_node.get("centroid") is None:
        continue
      stem_center = np.asarray(stem_node.get("centroid"), dtype=np.float32)

      # Fruit points.
      pos0 = inst_to_pos.get(fruit_iid)
      if pos0 is None:
        continue
      s0, c0 = pos0
      fruit_idx = idx_sorted[s0 : s0 + c0]
      fruit_pts = pts[fruit_idx]
      if args.max_fruit_points and fruit_pts.shape[0] > args.max_fruit_points:
        take = np_rng.choice(fruit_pts.shape[0], size=args.max_fruit_points, replace=False)
        fruit_pts = fruit_pts[take]

      # Candidate leaves by centroid distance (world).
      dists = np.linalg.norm(leaf_centroids_all_arr - fruit_center[None, :], axis=1)
      cand_idx = np.nonzero(dists <= max_dist)[0]
      if cand_idx.size == 0:
        continue

      # Gather candidate leaf points (union) and per-point leaf index.
      cand_leaf_iids: List[int] = []
      cand_leaf_centroids: List[np.ndarray] = []
      leaf_pts_list: List[np.ndarray] = []
      leaf_leafidx_list: List[np.ndarray] = []
      for i in cand_idx.tolist():
        liid = int(leaf_iids_all[i])
        pos1 = inst_to_pos.get(liid)
        if pos1 is None:
          continue
        s1, c1 = pos1
        idx = idx_sorted[s1 : s1 + c1]
        if idx.size == 0:
          continue
        j = len(cand_leaf_iids)
        cand_leaf_iids.append(liid)
        cand_leaf_centroids.append(leaf_centroids_all_arr[i])
        leaf_pts_list.append(pts[idx])
        leaf_leafidx_list.append(np.full((idx.size,), j, dtype=np.int32))
      if not cand_leaf_iids:
        continue

      leaf_pts = np.concatenate(leaf_pts_list, axis=0)
      leaf_leafidx = np.concatenate(leaf_leafidx_list, axis=0)
      if args.max_leaf_points and leaf_pts.shape[0] > args.max_leaf_points:
        take = np_rng.choice(leaf_pts.shape[0], size=args.max_leaf_points, replace=False)
        leaf_pts = leaf_pts[take]
        leaf_leafidx = leaf_leafidx[take]

      cand_leaf_centroids_arr = np.stack(cand_leaf_centroids, axis=0).astype(np.float32, copy=False)
      cand_leaf_dists = np.linalg.norm(cand_leaf_centroids_arr - fruit_center[None, :], axis=1).astype(np.float32)

      # Fruit-local transform.
      R = fruit_local_frame(fruit_center, stem_center)  # rows
      fruit_local = (R @ (fruit_pts - fruit_center).T).T  # [Nf,3]
      leaf_local = (R @ (leaf_pts - fruit_center).T).T    # [Nl,3]

      M = int(len(cand_leaf_iids))
      if M <= 0:
        continue

      union = np.zeros((num_dirs,), dtype=np.float32)
      pix_counts = np.zeros((num_dirs,), dtype=np.int32)
      pot = np.zeros((M, num_dirs), dtype=np.float32)
      excl = np.zeros((M, num_dirs), dtype=np.float32)
      krank = np.zeros((M, num_dirs, 3), dtype=np.float32)

      for di, (d, u, v) in enumerate(dir_bases):
        # Project to plane + depth.
        x_f = fruit_local @ u
        y_f = fruit_local @ v
        z_f = fruit_local @ d
        x_l = leaf_local @ u
        y_l = leaf_local @ v
        z_l = leaf_local @ d

        min_x = float(min(x_f.min(initial=0.0), x_l.min(initial=0.0)))
        min_y = float(min(y_f.min(initial=0.0), y_l.min(initial=0.0)))
        max_x = float(max(x_f.max(initial=0.0), x_l.max(initial=0.0)))
        max_y = float(max(y_f.max(initial=0.0), y_l.max(initial=0.0)))

        margin = 1e-6
        x0 = min_x - margin
        y0 = min_y - margin
        W = int(math.floor((max_x - x0) / pix)) + 1

        ix_f = np.floor((x_f - x0) / pix).astype(np.int32)
        iy_f = np.floor((y_f - y0) / pix).astype(np.int32)
        keys_f = (ix_f.astype(np.int64) + np.int64(W) * iy_f.astype(np.int64))
        fruit_keys, zf = zbuffer_min(keys_f, z_f.astype(np.float32, copy=False))
        P = int(fruit_keys.size)
        if P <= 0:
          continue
        pix_counts[di] = P

        ix_l = np.floor((x_l - x0) / pix).astype(np.int32)
        iy_l = np.floor((y_l - y0) / pix).astype(np.int32)
        keys_l = (ix_l.astype(np.int64) + np.int64(W) * iy_l.astype(np.int64))

        # Map leaf points onto the fruit pixel set (we only care about overlap w/ fruit silhouette pixels).
        pos = np.searchsorted(fruit_keys, keys_l)
        valid = (pos < P) & (fruit_keys[np.minimum(pos, P - 1)] == keys_l)
        if not np.any(valid):
          continue

        pix_idx = pos[valid].astype(np.int32, copy=False)  # [Nv]
        leaf_idx = leaf_leafidx[valid].astype(np.int32, copy=False)
        zlv = z_l[valid].astype(np.float32, copy=False)

        # Per-(leaf, fruit_pixel) minimum depth (potential occlusion).
        flat = leaf_idx.astype(np.int64) * np.int64(P) + pix_idx.astype(np.int64)
        leaf_min = np.full((M * P,), np.inf, dtype=np.float32)
        np.minimum.at(leaf_min, flat, zlv)
        leaf_min = leaf_min.reshape(M, P)

        thresh = zf - float(delta)
        occl = leaf_min < thresh[None, :]
        pot[:, di] = occl.sum(axis=1, dtype=np.int32) / float(P)

        # K-buffer (k=3) based on per-leaf depth minima.
        if M >= 3:
          top_idx = np.argpartition(leaf_min, kth=2, axis=0)[:3, :]  # [3,P]
        elif M == 2:
          top_idx = np.argpartition(leaf_min, kth=1, axis=0)[:2, :]  # [2,P]
        else:  # M == 1
          top_idx = np.zeros((1, P), dtype=np.int64)

        top_depth = np.take_along_axis(leaf_min, top_idx, axis=0)  # [k_eff,P]
        ordk = np.argsort(top_depth, axis=0)
        top_idx = np.take_along_axis(top_idx, ordk, axis=0)
        top_depth = np.take_along_axis(top_depth, ordk, axis=0)

        # Rank 1.
        m1 = top_depth[0] < thresh
        union[di] = float(m1.mean())
        if np.any(m1):
          c1 = np.bincount(top_idx[0, m1].astype(np.int64), minlength=M)
          excl[:, di] = c1.astype(np.float32) / float(P)
          krank[:, di, 0] = excl[:, di]

        # Rank 2/3.
        if top_depth.shape[0] >= 2:
          m2 = top_depth[1] < thresh
          if np.any(m2):
            c2 = np.bincount(top_idx[1, m2].astype(np.int64), minlength=M)
            krank[:, di, 1] = c2.astype(np.float32) / float(P)
        if top_depth.shape[0] >= 3:
          m3 = top_depth[2] < thresh
          if np.any(m3):
            c3 = np.bincount(top_idx[2, m3].astype(np.int64), minlength=M)
            krank[:, di, 2] = c3.astype(np.float32) / float(P)

      # Record fruit-level metrics.
      fruit_iids_scene.append(fruit_iid)
      fruit_union_scene.append(union.astype(np.float16))
      fruit_pix_scene.append(pix_counts.astype(np.int32))

      # Record leaf-fruit pair metrics (sparse rows).
      for j, liid in enumerate(cand_leaf_iids):
        row_pot = pot[j]
        row_excl = excl[j]
        row_k = krank[j]
        if drop_zero_pairs and (np.all(row_pot == 0) and np.all(row_k == 0) and np.all(row_excl == 0)):
          continue
        pair_fruit_iid.append(fruit_iid)
        pair_leaf_iid.append(int(liid))
        pair_dist.append(float(cand_leaf_dists[j]))
        pair_potential.append(row_pot.astype(np.float16, copy=False))
        pair_exclusive.append(row_excl.astype(np.float16, copy=False))
        pair_k_rank.append(row_k.astype(np.float16, copy=False))

    if fruit_pbar is not None:
      fruit_pbar.close()

    # Write sidecar for this scene.
    fruit_iid_arr = np.asarray(fruit_iids_scene, dtype=np.int32)
    if fruit_union_scene:
      fruit_union_arr = np.stack(fruit_union_scene, axis=0).astype(np.float16, copy=False)
      fruit_pix_arr = np.stack(fruit_pix_scene, axis=0).astype(np.int32, copy=False)
    else:
      fruit_union_arr = np.zeros((0, num_dirs), dtype=np.float16)
      fruit_pix_arr = np.zeros((0, num_dirs), dtype=np.int32)

    pair_fruit_iid_arr = np.asarray(pair_fruit_iid, dtype=np.int32)
    pair_leaf_iid_arr = np.asarray(pair_leaf_iid, dtype=np.int32)
    pair_dist_arr = np.asarray(pair_dist, dtype=np.float32)
    if pair_potential:
      pair_potential_arr = np.stack(pair_potential, axis=0).astype(np.float16, copy=False)
      pair_exclusive_arr = np.stack(pair_exclusive, axis=0).astype(np.float16, copy=False)
      pair_k_rank_arr = np.stack(pair_k_rank, axis=0).astype(np.float16, copy=False)
    else:
      pair_potential_arr = np.zeros((0, num_dirs), dtype=np.float16)
      pair_exclusive_arr = np.zeros((0, num_dirs), dtype=np.float16)
      pair_k_rank_arr = np.zeros((0, num_dirs, 3), dtype=np.float16)

    np.savez_compressed(
      out_path,
      schema_version=np.asarray(1, dtype=np.int32),
      scene_id=np.asarray(scene_id),
      pixel_size_m=np.asarray(pix, dtype=np.float32),
      delta_m=np.asarray(delta, dtype=np.float32),
      max_dist_m=np.asarray(max_dist, dtype=np.float32),
      dirs=dirs.astype(np.float32, copy=False),
      fruit_iid=fruit_iid_arr,
      fruit_union=fruit_union_arr,
      fruit_pixel_count=fruit_pix_arr,
      pair_fruit_iid=pair_fruit_iid_arr,
      pair_leaf_iid=pair_leaf_iid_arr,
      pair_dist=pair_dist_arr,
      pair_potential=pair_potential_arr,
      pair_exclusive=pair_exclusive_arr,
      pair_k_rank=pair_k_rank_arr,
    )

    if not printed_first_scene_stats:
      # One-time sanity print.
      printed_first_scene_stats = True
      mean_union = float(fruit_union_arr.mean()) if fruit_union_arr.size else 0.0
      max_union = float(fruit_union_arr.max()) if fruit_union_arr.size else 0.0
      msg = (
        f"[sanity] {scene_id}: fruits={int(fruit_iid_arr.size)} pairs={int(pair_fruit_iid_arr.size)} "
        f"mean_union={mean_union:.3f} max_union={max_union:.3f} scene_time={time.time()-t0_scene:.1f}s"
      )
      if pbar is not None:
        pbar.write(msg)
      else:
        print(msg, flush=True)

    total_fruits += int(fruit_iid_arr.size)
    total_pairs += int(pair_fruit_iid_arr.size)
    num_scenes_written += 1
    if pbar is not None:
      pbar.set_postfix(
        written=num_scenes_written,
        skip_exists=skipped_exists,
        skip_nojson=skipped_missing_json,
        fruits=total_fruits,
        pairs=total_pairs,
      )

  if pbar is not None:
    pbar.close()

  print(
    "Done\n"
    f"  written:  {num_scenes_written}\n"
    f"  skipped (exists): {skipped_exists}\n"
    f"  skipped (missing json): {skipped_missing_json}\n"
    f"  total fruits: {total_fruits}\n"
    f"  total pairs:  {total_pairs}\n"
    f"  out_dir: {out_dir}\n",
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
