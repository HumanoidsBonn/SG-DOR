#!/usr/bin/env python3
"""
Visualize pointcloud z-buffer occlusion metrics for a few fruits/leaves, and optionally
compare against OBB-based occlusion proxies (silhouette overlap).

Outputs (per selected fruit):
- A 3D scatter plot showing fruit + a top-occluding leaf + parent stem points.
- A 6-panel pixel-grid visualization for ±X/±Y/±Z fruit-local view directions.
- If enabled, two additional 6-panel OBB visualizations:
  - pixel-grid filled OBB silhouettes,
  - non-pixel polygon hull overlays.
If enabled, also writes dataset-level OBB-vs-pointcloud correlation plots into the
same output folder that contains the per-example subfolders.

This is intended as a sanity/inspection tool to verify that the z-buffer sidecars
match what we think we're computing.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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

    x_unit = np.cross(y_unit, z_unit)
    x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-12)
    return np.vstack([x_unit, y_unit, z_unit]).astype(np.float32)  # rows


def build_dir_bases(dirs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each direction d, build an orthonormal basis (d, u, v)."""
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


def pca_obb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA OBB for visualization.
    Returns (center[3], axes[3,3] columns, extents[3]).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        c = pts.mean(axis=0) if pts.size else np.zeros((3,), dtype=np.float32)
        return c, np.eye(3, dtype=np.float32), np.zeros((3,), dtype=np.float32)
    c = pts.mean(axis=0)
    x = pts - c[None, :]
    cov = (x.T @ x) / float(max(pts.shape[0] - 1, 1))
    w, v = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(-w)
    v = v[:, order].astype(np.float32, copy=False)
    # ensure right-handed
    if float(np.linalg.det(v)) < 0:
        v[:, 2] *= -1.0
    proj = x @ v  # [N,3]
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    ext = (hi - lo).astype(np.float32)
    center_local = 0.5 * (lo + hi)
    center = (c + v @ center_local).astype(np.float32)
    return center, v, ext


def obb_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """8 corners for OBB with axes columns."""
    c = np.asarray(center, dtype=np.float32)
    a = np.asarray(axes, dtype=np.float32)
    e = np.asarray(extents, dtype=np.float32)
    half = 0.5 * e
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(c + sx * half[0] * a[:, 0] + sy * half[1] * a[:, 1] + sz * half[2] * a[:, 2])
    return np.asarray(corners, dtype=np.float32)


def draw_obb(ax, center: np.ndarray, axes: np.ndarray, extents: np.ndarray, color: str, lw: float = 1.5, alpha: float = 1.0) -> None:
    corners = obb_corners(center, axes, extents)
    # edges between corners that differ by 1 bit
    idx = list(range(8))
    for i in idx:
        for j in idx:
            if bin(i ^ j).count("1") == 1:
                p = corners[i]
                q = corners[j]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, lw=lw, alpha=alpha)


def find_dir_indices_xyz(dirs: np.ndarray, tol: float = 1e-4) -> Dict[str, int]:
    want = {
        "+X": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "-X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        "+Y": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "-Y": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "+Z": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "-Z": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    }
    out: Dict[str, int] = {}
    dn = dirs.astype(np.float32, copy=False)
    dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-12)
    for k, v in want.items():
        dots = dn @ v
        i = int(np.argmax(dots))
        if float(dots[i]) >= (1.0 - float(tol)):
            out[k] = i
    return out


def compute_rasters_for_direction(
    fruit_local: np.ndarray,
    leaf_local_all: np.ndarray,
    leaf_leafidx_all: np.ndarray,
    leaf_local_sel: np.ndarray,
    pix: float,
    delta: float,
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """
    Compute fruit pixel set + depth, selected leaf silhouette, and selected leaf occlusion masks.

    For visualization we keep the raster origin consistent with sidecar generation (bounds based
    on fruit + all candidate leaves), but return a crop bbox around fruit+selected leaf pixels
    for readability.
    """
    d, u, v = basis
    x_f = fruit_local @ u
    y_f = fruit_local @ v
    z_f = fruit_local @ d

    x_l = leaf_local_all @ u
    y_l = leaf_local_all @ v
    z_l = leaf_local_all @ d

    x_ls = leaf_local_sel @ u
    y_ls = leaf_local_sel @ v
    z_ls = leaf_local_sel @ d

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
        return {"P": 0}

    thresh = zf - float(delta)

    # All leaves -> map to fruit pixels (to build leaf_min on fruit silhouette pixels, as in sidecar).
    ix_l = np.floor((x_l - x0) / pix).astype(np.int32)
    iy_l = np.floor((y_l - y0) / pix).astype(np.int32)
    keys_l = (ix_l.astype(np.int64) + np.int64(W) * iy_l.astype(np.int64))
    pos = np.searchsorted(fruit_keys, keys_l)
    valid = (pos < P) & (fruit_keys[np.minimum(pos, P - 1)] == keys_l)

    pix_idx = pos[valid].astype(np.int32, copy=False)
    leaf_idx = leaf_leafidx_all[valid].astype(np.int32, copy=False)
    zlv = z_l[valid].astype(np.float32, copy=False)

    if leaf_leafidx_all.size == 0:
        return {"P": 0}
    M = int(leaf_leafidx_all.max()) + 1
    leaf_min = np.full((M * P,), np.inf, dtype=np.float32)
    flat = leaf_idx.astype(np.int64) * np.int64(P) + pix_idx.astype(np.int64)
    np.minimum.at(leaf_min, flat, zlv)
    leaf_min = leaf_min.reshape(M, P)

    # Selected leaf: compute its own min depths on fruit pixels (potential).
    ix_s = np.floor((x_ls - x0) / pix).astype(np.int32)
    iy_s = np.floor((y_ls - y0) / pix).astype(np.int32)
    keys_s = (ix_s.astype(np.int64) + np.int64(W) * iy_s.astype(np.int64))
    pos_s = np.searchsorted(fruit_keys, keys_s)
    valid_s = (pos_s < P) & (fruit_keys[np.minimum(pos_s, P - 1)] == keys_s)
    pix_s = pos_s[valid_s].astype(np.int32, copy=False)
    zsv = z_ls[valid_s].astype(np.float32, copy=False)

    sel_min = np.full((P,), np.inf, dtype=np.float32)
    np.minimum.at(sel_min, pix_s.astype(np.int64), zsv)

    pot_mask_1d = (sel_min < thresh)
    leaf_over_fruit_1d = np.isfinite(sel_min)

    # Any leaf in front of fruit.
    min_z = np.min(leaf_min, axis=0)  # [P]
    union_mask_1d = (min_z < thresh)

    # Selected leaf is remapped to index 0 by the caller.
    argmin = np.argmin(leaf_min, axis=0)  # [P]
    excl_mask_1d = union_mask_1d & (argmin == 0)

    # Build dense grids for plotting.
    ix = (fruit_keys % np.int64(W)).astype(np.int32)
    iy = (fruit_keys // np.int64(W)).astype(np.int32)
    H = int(iy.max(initial=0)) + 1

    fruit_mask = np.zeros((H, W), dtype=bool)
    fruit_depth = np.full((H, W), np.nan, dtype=np.float32)
    fruit_mask[iy, ix] = True
    fruit_depth[iy, ix] = zf.astype(np.float32, copy=False)

    pot_mask = np.zeros((H, W), dtype=bool)
    pot_mask[iy, ix] = pot_mask_1d

    union_mask = np.zeros((H, W), dtype=bool)
    union_mask[iy, ix] = union_mask_1d

    excl_mask = np.zeros((H, W), dtype=bool)
    excl_mask[iy, ix] = excl_mask_1d

    leaf_over_fruit_mask = np.zeros((H, W), dtype=bool)
    leaf_over_fruit_mask[iy, ix] = leaf_over_fruit_1d

    # Selected leaf silhouette pixels (in this full grid), including those outside fruit.
    leaf_keys, _ = zbuffer_min(keys_s.astype(np.int64, copy=False), z_ls.astype(np.float32, copy=False))
    ix_lf = (leaf_keys % np.int64(W)).astype(np.int32, copy=False)
    iy_lf = (leaf_keys // np.int64(W)).astype(np.int32, copy=False)
    # Grid height may need to expand if the leaf projects beyond fruit pixel extents.
    H2 = int(max(H - 1, int(iy_lf.max(initial=0)))) + 1 if iy_lf.size else H
    if H2 != H:
        fruit_mask2 = np.zeros((H2, W), dtype=bool)
        fruit_depth2 = np.full((H2, W), np.nan, dtype=np.float32)
        union_mask2 = np.zeros((H2, W), dtype=bool)
        pot_mask2 = np.zeros((H2, W), dtype=bool)
        excl_mask2 = np.zeros((H2, W), dtype=bool)
        leaf_over_fruit_mask2 = np.zeros((H2, W), dtype=bool)
        fruit_mask2[:H, :] = fruit_mask
        fruit_depth2[:H, :] = fruit_depth
        union_mask2[:H, :] = union_mask
        pot_mask2[:H, :] = pot_mask
        excl_mask2[:H, :] = excl_mask
        leaf_over_fruit_mask2[:H, :] = leaf_over_fruit_mask
        fruit_mask, fruit_depth, union_mask, pot_mask, excl_mask, leaf_over_fruit_mask = (
            fruit_mask2,
            fruit_depth2,
            union_mask2,
            pot_mask2,
            excl_mask2,
            leaf_over_fruit_mask2,
        )
        H = H2

    leaf_mask = np.zeros((H, W), dtype=bool)
    if ix_lf.size:
        leaf_mask[iy_lf, ix_lf] = True

    # Crop bbox around fruit+leaf pixels for readability.
    any_mask = fruit_mask | leaf_mask
    if bool(any_mask.any()):
        ys, xs = np.nonzero(any_mask)
        y0c = max(int(ys.min()) - 2, 0)
        y1c = min(int(ys.max()) + 3, H)
        x0c = max(int(xs.min()) - 2, 0)
        x1c = min(int(xs.max()) + 3, W)
        crop = (slice(y0c, y1c), slice(x0c, x1c))
    else:
        crop = (slice(0, H), slice(0, W))

    return {
        "P": P,
        "H": H,
        "W": W,
        "union_mask": union_mask,
        "fruit_mask": fruit_mask,
        "fruit_depth": fruit_depth,
        "leaf_mask": leaf_mask,
        "leaf_over_fruit_mask": leaf_over_fruit_mask,
        "pot_mask": pot_mask,
        "excl_mask": excl_mask,
        "potential": float(pot_mask_1d.mean()),
        "leaf_over_fruit": float(leaf_over_fruit_1d.mean()),
        "crop": crop,
    }


def cross_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """2D cross product (scalar)."""
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(points_2d: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Compute the convex hull of 2D points (monotonic chain)."""
    if len(points_2d) <= 1:
        return [(float(p[0]), float(p[1])) for p in points_2d]
    pts = sorted(set((float(p[0]), float(p[1])) for p in points_2d))
    if len(pts) <= 1:
        return pts

    def build_half(seq):
        half: List[Tuple[float, float]] = []
        for p in seq:
            while len(half) >= 2 and cross_2d(
                (half[-1][0] - half[-2][0], half[-1][1] - half[-2][1]),
                (p[0] - half[-1][0], p[1] - half[-1][1]),
            ) <= 0:
                half.pop()
            half.append(p)
        return half

    lower = build_half(pts)
    upper = build_half(reversed(pts))
    return lower[:-1] + upper[:-1]


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    """Compute polygon area via the shoelace formula."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % len(poly)][0]), float(poly[(i + 1) % len(poly)][1])
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def clip_polygon(subject: Sequence[Sequence[float]], clipper: Sequence[Sequence[float]]) -> List[List[float]]:
    """Clip a convex polygon by another convex polygon (Sutherland--Hodgman)."""
    if not subject or not clipper:
        return []

    def inside(p, a, b):
        return cross_2d((b[0] - a[0], b[1] - a[1]), (p[0] - a[0], p[1] - a[1])) >= 0

    def intersection(p1, p2, a, b):
        s = np.array(p1, dtype=float)
        e = np.array(p2, dtype=float)
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        d1 = e - s
        d2 = b - a
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(float(denom)) < 1e-8:
            return e.tolist()
        t = ((a[0] - s[0]) * d2[1] - (a[1] - s[1]) * d2[0]) / denom
        return (s + t * d1).tolist()

    output = list([list(p) for p in subject])
    for i in range(len(clipper)):
        input_list = output
        if not input_list:
            break
        output = []
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        prev = input_list[-1]
        for curr in input_list:
            if inside(curr, a, b):
                if not inside(prev, a, b):
                    output.append(intersection(prev, curr, a, b))
                output.append(curr)
            elif inside(prev, a, b):
                output.append(intersection(prev, curr, a, b))
            prev = curr
    return output


def project_obb_hull_uv(
    *,
    node: Dict[str, Any],
    fruit_center: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> List[Tuple[float, float]]:
    """Project a node's OBB corners into fruit-local (u,v) and return convex hull."""
    obb = node.get("obb") or {}
    c = np.asarray(obb.get("center", node.get("centroid", [0.0, 0.0, 0.0])), dtype=np.float32)
    axes = obb.get("axes", np.eye(3, dtype=np.float32))
    ext = obb.get("extents", [0.0, 0.0, 0.0])
    corners = obb_corners(c, axes, ext)  # world coords
    local = (corners - fruit_center[None, :]) @ R.T  # fruit-local coords
    uu = local @ u
    vv = local @ v
    return convex_hull(list(zip(uu.tolist(), vv.tolist())))


def rasterize_obb_polys(
    *,
    pix: float,
    fruit_poly: Sequence[Sequence[float]],
    leaf_poly: Sequence[Sequence[float]],
    union_leaf_polys: Sequence[Sequence[Sequence[float]]],
    union_use: Sequence[bool],
) -> Dict[str, Any]:
    """Rasterize filled OBB silhouettes for qualitative comparison (pixel grid)."""
    from matplotlib.path import Path

    fruit_poly = np.asarray(fruit_poly, dtype=float)
    leaf_poly = np.asarray(leaf_poly, dtype=float)
    if fruit_poly.size == 0:
        return {"H": 0, "W": 0}

    xs = fruit_poly[:, 0]
    ys = fruit_poly[:, 1]
    if leaf_poly.size:
        xs = np.concatenate([xs, leaf_poly[:, 0]], axis=0)
        ys = np.concatenate([ys, leaf_poly[:, 1]], axis=0)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    margin = 1e-6
    x0 = min_x - margin
    y0 = min_y - margin
    W = int(math.floor((max_x - x0) / pix)) + 1
    H = int(math.floor((max_y - y0) / pix)) + 1

    cx = x0 + (np.arange(W, dtype=np.float32) + 0.5) * float(pix)
    cy = y0 + (np.arange(H, dtype=np.float32) + 0.5) * float(pix)
    gx, gy = np.meshgrid(cx, cy)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    fruit_path = Path(fruit_poly)
    fruit_mask = fruit_path.contains_points(pts).reshape(H, W)

    leaf_mask = np.zeros((H, W), dtype=bool)
    if leaf_poly.size:
        leaf_path = Path(leaf_poly)
        leaf_mask = leaf_path.contains_points(pts).reshape(H, W)

    union_leaf_mask = np.zeros((H, W), dtype=bool)
    for poly, use in zip(union_leaf_polys, union_use):
        if not use:
            continue
        poly = np.asarray(poly, dtype=float)
        if poly.size == 0:
            continue
        path = Path(poly)
        union_leaf_mask |= path.contains_points(pts).reshape(H, W)

    return {
        "H": H,
        "W": W,
        "fruit_mask": fruit_mask,
        "leaf_mask": leaf_mask,
        "union_leaf_mask": union_leaf_mask,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize pointcloud z-buffer occlusion sidecar metrics.")
    p.add_argument("--data_root", required=True, help="Dataset root containing data/*.npz and *_graph.json")
    p.add_argument("--sidecar_dir", default=None, help="Optional explicit sidecar dir (defaults: data/ then analysis/occl_zbuf_sidecars_*)")
    p.add_argument("--use_4mm", action="store_true", default=True, help="Use *_pc_4mm.npz (default)")
    p.add_argument("--drop_shoots", action="store_true", help="Drop shoot points for visualization (semantics==5)")
    p.add_argument("--scene_ids", default=None, help="Comma-separated list like scene_00013,scene_00042 to restrict")
    p.add_argument("--fruit_iid", type=int, default=None, help="Optional: force a specific fruit instance_id")
    p.add_argument("--leaf_iid", type=int, default=None, help="Optional: force a specific leaf instance_id (for the chosen fruit)")
    p.add_argument("--max_scenes", type=int, default=8, help="Max scenes to sample")
    p.add_argument("--fruits_per_scene", type=int, default=1, help="How many fruits to visualize per scene")
    p.add_argument(
        "--pick_mode",
        type=str,
        default="mixed",
        choices=["random", "high", "low", "mixed", "mismatch_high", "mismatch_low", "mismatch_mixed"],
        help="How to pick fruits (by PC union, or by PC-vs-OBB union mismatch when available)",
    )
    p.add_argument(
        "--panel_mode",
        type=str,
        default="potential",
        choices=["potential", "full"],
        help=(
            "Per-direction panel visualization mode. "
            "'potential' shows only fruit+leaf potential occlusion (cleaner); "
            "'full' shows union + potential + exclusive overlays."
        ),
    )
    p.add_argument("--compare_obb", action="store_true", help="Also render OBB-based qualitative panels (pixel + polygon)")
    p.add_argument("--num_examples", type=int, default=None, help="Stop after writing this many fruit examples (across scenes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="analysis/vis_pc_occl_zbuf", help="Output directory")
    p.add_argument("--max_points_vis", type=int, default=0, help="Optional cap for points per object in 3D plots (0=disable)")
    return p.parse_args()


def _default_sidecar_path(data_root: Path, data_dir: Path, scene_id: str, sidecar_dir: Optional[str]) -> Optional[Path]:
    prefer = f"{scene_id}_occl_zbuf_k3_ps4mm_d4mm.npz"
    glob_fallback = f"{scene_id}_occl_zbuf_k3_*.npz"
    dirs: List[Path] = []
    if sidecar_dir:
        dirs.append(Path(sidecar_dir))
    dirs.append(data_dir)
    dirs.append(Path("analysis") / f"occl_zbuf_sidecars_{data_root.name}")
    for d in dirs:
        p = d / prefer
        if p.exists():
            return p
        ms = sorted(d.glob(glob_fallback))
        if ms:
            return ms[0]
    return None


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    data_root = Path(args.data_root).expanduser().resolve()
    data_dir = data_root / "data"
    if not data_dir.exists():
        data_dir = data_root

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_pc_4mm.npz" if args.use_4mm else "_pc.npz"
    pc_files = sorted(glob.glob(str(data_dir / f"scene_*{suffix}")))
    if not pc_files:
        raise SystemExit(f"No pointcloud files matching *{suffix} under {data_dir}")

    if args.scene_ids:
        want = set([s.strip() for s in str(args.scene_ids).split(",") if s.strip()])
        pc_files = [p for p in pc_files if Path(p).name.split("_pc")[0] in want]
        if not pc_files:
            raise SystemExit(f"No scenes matched --scene_ids under {data_dir}")

    max_scenes = min(int(args.max_scenes), len(pc_files))
    if args.scene_ids:
        # Keep stable order when user explicitly selects scenes.
        pc_files = pc_files[:max_scenes]
    else:
        picks = rng.choice(len(pc_files), size=max_scenes, replace=False).tolist()
        pc_files = [pc_files[i] for i in picks]

    # Avoid matplotlib cache warnings in environments where ~/.config is not writable (common on shared servers).
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"mplconfig_{os.getuid()}"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa
    except Exception as e:
        raise SystemExit(f"matplotlib is required: {e}")

    # Accumulate per-(fruit,dir) union pairs for dataset-level correlation plots.
    corr_pc: List[np.ndarray] = []
    corr_obb: List[np.ndarray] = []

    total_written = 0
    examples_written = 0
    for si, pc_path_str in enumerate(pc_files):
        # Keep scanning scenes for correlation plots even after enough examples were written,
        # but avoid doing any expensive per-example pointcloud work when not needed.
        need_examples = args.num_examples is None or examples_written < int(args.num_examples)
        if not need_examples and (not bool(args.compare_obb)):
            break

        pc_path = Path(pc_path_str)
        scene_id = pc_path.name.split("_pc")[0]
        json_path = data_dir / f"{scene_id}_graph.json"
        if not json_path.exists():
            continue
        sidecar_path = _default_sidecar_path(data_root, data_dir, scene_id, args.sidecar_dir)
        if sidecar_path is None or (not sidecar_path.exists()):
            continue

        g = _load_json(json_path)
        occl = g.get("occlusion", {}) or {}
        dirs_raw = occl.get("dirs")
        if not dirs_raw:
            continue
        dirs = np.asarray(dirs_raw, dtype=np.float32)
        bases = build_dir_bases(dirs)
        dir_idx = find_dir_indices_xyz(dirs)
        if len(dir_idx) < 6:
            # If ordering is non-canonical, still proceed with whatever matches.
            pass

        with np.load(sidecar_path) as z:
            pix = float(z.get("pixel_size_m", 0.004))
            delta = float(z.get("delta_m", 0.004))
            fruit_iid = z["fruit_iid"].astype(np.int32, copy=False)
            fruit_union = z["fruit_union"].astype(np.float32, copy=False)  # [F,D]
            pair_fruit = z["pair_fruit_iid"].astype(np.int32, copy=False)
            pair_leaf = z["pair_leaf_iid"].astype(np.int32, copy=False)
            pair_pot = z["pair_potential"].astype(np.float32, copy=False)  # [R,D]
            pair_excl = z["pair_exclusive"].astype(np.float32, copy=False) if "pair_exclusive" in z else None
            pair_k = z["pair_k_rank"].astype(np.float32, copy=False)  # [R,D,3]

        nodes = g.get("nodes", []) or []
        fruits = [n for n in nodes if n.get("type") == "fruit" and n.get("instance_id") is not None]
        if not fruits:
            continue

        fruit_nodes_by_iid = {int(n["instance_id"]): n for n in fruits}
        leaf_nodes_by_iid = {int(n["instance_id"]): n for n in nodes if n.get("type") == "leaf" and n.get("instance_id") is not None}

        # Optional OBB union targets (per fruit, 18 dirs) stored in the JSON nodes.
        fruit_obb_union_by_iid: Dict[int, np.ndarray] = {}
        for n in fruits:
            ot = n.get("occlusion_targets") or {}
            ur = ot.get("union_raw")
            if ur is None:
                continue
            try:
                uv = np.asarray(ur, dtype=np.float32)
                if uv.ndim == 1 and uv.size == int(fruit_union.shape[1]):
                    fruit_obb_union_by_iid[int(n["instance_id"])] = uv
            except Exception:
                continue

        # Pick fruits present in sidecar+json.
        union_max = fruit_union.max(axis=1) if fruit_union.size else np.zeros((0,), dtype=np.float32)
        valid_rows = [i for i, iid in enumerate(fruit_iid.tolist()) if int(iid) in fruit_nodes_by_iid]
        if not valid_rows:
            continue

        # (Optional) Dataset-level OBB-vs-pointcloud union comparison.
        # We compare per fruit-direction union from the pointcloud sidecar against the OBB union_raw stored
        # in the JSON nodes. This is used only for diagnostics/figures; it is not part of training.
        if bool(args.compare_obb) and fruit_obb_union_by_iid:
            for r in valid_rows:
                fiid0 = int(fruit_iid[int(r)])
                obb_u = fruit_obb_union_by_iid.get(fiid0)
                if obb_u is None:
                    continue
                pc_u = fruit_union[int(r)].astype(np.float32, copy=False)
                if int(pc_u.size) != int(obb_u.size):
                    continue
                m = np.isfinite(pc_u) & np.isfinite(obb_u)
                if not bool(m.any()):
                    continue
                # Copy the small slices to avoid holding on to full-scene arrays via views.
                corr_pc.append(pc_u[m].astype(np.float32, copy=True))
                corr_obb.append(obb_u[m].astype(np.float32, copy=True))

        # Choose fruit rows according to pick_mode (unless fruit_iid is forced).
        rows = np.asarray(valid_rows, dtype=np.int64)
        scores_union = union_max[rows]
        order_union = rows[np.argsort(scores_union)]

        # Mismatch score: max_d |pc_union - obb_union_raw| for this fruit (if OBB union is available).
        mismatch = np.full((rows.size,), np.nan, dtype=np.float32)
        for ii, r in enumerate(rows.tolist()):
            fiid = int(fruit_iid[int(r)])
            obb_u = fruit_obb_union_by_iid.get(fiid)
            if obb_u is None:
                continue
            pc_u = fruit_union[int(r)].astype(np.float32, copy=False)
            mismatch[ii] = float(np.max(np.abs(pc_u - obb_u)))
        if bool(np.isfinite(mismatch).any()):
            order_mis = rows[np.argsort(mismatch, kind="mergesort")]
        else:
            order_mis = order_union

        picked_rows: List[int] = []
        if args.fruit_iid is not None:
            fr0 = np.where(fruit_iid == int(args.fruit_iid))[0]
            if fr0.size and int(fr0[0]) in valid_rows:
                picked_rows = [int(fr0[0])]
            else:
                continue
        for k in range(int(args.fruits_per_scene) if args.fruit_iid is None else 0):
            if args.pick_mode == "random":
                picked_rows.append(int(rng.choice(rows)))
            elif args.pick_mode == "high":
                picked_rows.append(int(order_union[-1 - (k % len(order_union))]))
            elif args.pick_mode == "low":
                picked_rows.append(int(order_union[(k % len(order_union))]))
            elif args.pick_mode == "mismatch_high":
                picked_rows.append(int(order_mis[-1 - (k % len(order_mis))]))
            elif args.pick_mode == "mismatch_low":
                picked_rows.append(int(order_mis[(k % len(order_mis))]))
            else:  # mixed / mismatch_mixed
                if args.pick_mode == "mismatch_mixed":
                    if (si + k) % 2 == 0:
                        picked_rows.append(int(order_mis[-1 - (k % len(order_mis))]))
                    else:
                        picked_rows.append(int(order_mis[(k % len(order_mis))]))
                else:
                    if (si + k) % 2 == 0:
                        picked_rows.append(int(order_union[-1 - (k % len(order_union))]))
                    else:
                        picked_rows.append(int(order_union[(k % len(order_union))]))

        # If we're only collecting correlation plots, we can skip the expensive pointcloud load + plots.
        if not need_examples:
            continue

        # Load full pointcloud.
        npz = np.load(pc_path)
        pts = npz["points"].astype(np.float32, copy=False)
        sem = npz["semantics"].astype(np.int32, copy=False)
        inst = npz["instances"].astype(np.int32, copy=False)

        m = inst > 0
        if args.drop_shoots:
            m = m & (sem != 5)
        pts = pts[m]
        sem = sem[m]
        inst = inst[m]

        # Build instance -> indices mapping (all points).
        orderp = np.argsort(inst)
        inst_sorted = inst[orderp]
        idx_sorted = orderp
        uniq_inst, starts, counts = np.unique(inst_sorted, return_index=True, return_counts=True)
        inst_to_pos = {int(i): (int(s), int(c)) for i, s, c in zip(uniq_inst.tolist(), starts.tolist(), counts.tolist())}

        # attached_to mapping: child -> parent
        attached_to = {e["source"]: e["target"] for e in g.get("edges", []) if e.get("label") == "attached_to"}
        node_by_id = {str(n.get("id")): n for n in nodes if n.get("id") is not None}

        for fr in picked_rows:
            if args.num_examples is not None and examples_written >= int(args.num_examples):
                break
            fiid = int(fruit_iid[fr])
            fnode = fruit_nodes_by_iid.get(fiid)
            if fnode is None:
                continue

            # Fruit -> ped -> stem
            fid = str(fnode.get("id"))
            ped_id = attached_to.get(fid)
            stem_id = attached_to.get(ped_id) if ped_id else None
            stem_node = node_by_id.get(str(stem_id)) if stem_id else None
            if stem_node is None:
                continue
            fruit_center = np.asarray(fnode.get("centroid"), dtype=np.float32)
            stem_center = np.asarray(stem_node.get("centroid"), dtype=np.float32)
            R = fruit_local_frame(fruit_center, stem_center)

            # Pick a leaf for this fruit: max potential across dirs.
            pm = (pair_fruit == fiid)
            leaf_iids = pair_leaf[pm]
            if leaf_iids.size == 0:
                continue
            pot_rows = pair_pot[pm]  # [L,D]
            if args.leaf_iid is not None:
                sel = np.where(leaf_iids == int(args.leaf_iid))[0]
                if sel.size == 0:
                    continue
                jbest = int(sel[0])
                liid = int(args.leaf_iid)
            else:
                score = pot_rows.max(axis=1)
                jbest = int(np.argmax(score))
                liid = int(leaf_iids[jbest])

            # Per-example output folder (keeps each fruit example self-contained).
            ex_dir = out_dir / f"example_{examples_written:03d}_{scene_id}_fruit_{fiid}_leaf_{liid}"
            ex_dir.mkdir(parents=True, exist_ok=True)

            # Points for fruit/stem/leaf
            def _pts_for_iid(iid: int) -> Optional[np.ndarray]:
                pos = inst_to_pos.get(int(iid))
                if pos is None:
                    return None
                s, c = pos
                idx = idx_sorted[s : s + c]
                return pts[idx]

            fruit_pts = _pts_for_iid(fiid)
            leaf_pts = _pts_for_iid(liid)
            stem_pts = _pts_for_iid(int(stem_node.get("instance_id", -1)))
            if fruit_pts is None or leaf_pts is None or stem_pts is None:
                continue

            # Optional cap for faster 3D plotting.
            def _cap(p: np.ndarray) -> np.ndarray:
                if int(args.max_points_vis) <= 0 or p.shape[0] <= int(args.max_points_vis):
                    return p
                take = rng.choice(p.shape[0], size=int(args.max_points_vis), replace=False)
                return p[take]

            fruit_pts_vis = _cap(fruit_pts)
            leaf_pts_vis = _cap(leaf_pts)
            stem_pts_vis = _cap(stem_pts)

            # Candidate leaves for union recompute: those in sidecar rows for this fruit.
            # (This matches what was stored; we don't re-run max_dist selection here.)
            leaf_iids_all = leaf_iids.astype(np.int32, copy=False)
            leaf_pts_all: Optional[np.ndarray] = None
            leaf_idx_all: Optional[np.ndarray] = None
            if str(args.panel_mode) == "full":
                pts_all: List[np.ndarray] = []
                idx_all: List[np.ndarray] = []
                for j, iid in enumerate(leaf_iids_all.tolist()):
                    p = _pts_for_iid(int(iid))
                    if p is None:
                        continue
                    pts_all.append(p)
                    idx_all.append(np.full((p.shape[0],), j, dtype=np.int32))
                if not pts_all:
                    continue
                leaf_pts_all = np.concatenate(pts_all, axis=0)
                leaf_idx_all = np.concatenate(idx_all, axis=0)

            # Remap so selected leaf is index 0 (for exclusive/k-rank masks).
            if str(args.panel_mode) == "full":
                assert leaf_idx_all is not None and leaf_pts_all is not None
                try:
                    sel_j = int(np.where(leaf_iids_all == liid)[0][0])
                except Exception:
                    sel_j = jbest
                if sel_j != 0:
                    # swap indices 0 <-> sel_j
                    li0 = int(leaf_iids_all[0])
                    leaf_iids_all = leaf_iids_all.copy()
                    leaf_iids_all[0], leaf_iids_all[sel_j] = leaf_iids_all[sel_j], leaf_iids_all[0]
                    leaf_idx_all = leaf_idx_all.copy()
                    leaf_idx_all[leaf_idx_all == 0] = -999
                    leaf_idx_all[leaf_idx_all == sel_j] = 0
                    leaf_idx_all[leaf_idx_all == -999] = sel_j
                    # also swap leaf_pts_vis if it was leaf 0
                    if li0 == liid:
                        pass

            # Fruit/local transforms
            fruit_local = (fruit_pts - fruit_center[None, :]) @ R.T
            leaf_local_sel = (leaf_pts - fruit_center[None, :]) @ R.T
            if str(args.panel_mode) == "full":
                assert leaf_pts_all is not None and leaf_idx_all is not None
                leaf_local_all = (leaf_pts_all - fruit_center[None, :]) @ R.T
                leaf_leafidx_all = leaf_idx_all
            else:
                # Potential-only mode: treat the selected leaf as the only candidate leaf.
                # This matches the "in vacuum" interpretation and keeps the plot simple.
                leaf_local_all = leaf_local_sel
                leaf_leafidx_all = np.zeros((leaf_local_sel.shape[0],), dtype=np.int32)

            # 3D plot
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(stem_pts_vis[:, 0], stem_pts_vis[:, 1], stem_pts_vis[:, 2], s=1, c="#8c564b", alpha=0.25, label="stem")
            ax.scatter(leaf_pts_vis[:, 0], leaf_pts_vis[:, 1], leaf_pts_vis[:, 2], s=1, c="#2ca02c", alpha=0.35, label="leaf")
            ax.scatter(fruit_pts_vis[:, 0], fruit_pts_vis[:, 1], fruit_pts_vis[:, 2], s=1, c="#d62728", alpha=0.45, label="fruit")

            for pts0, col in [(stem_pts, "#8c564b"), (leaf_pts, "#2ca02c"), (fruit_pts, "#d62728")]:
                c0, a0, e0 = pca_obb(pts0)
                draw_obb(ax, c0, a0, e0, color=col, lw=1.5, alpha=0.9)

            ax.set_title(f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (pix={pix*1000:.0f}mm delta={delta*1000:.0f}mm)")
            ax.legend(loc="upper right")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            fig.tight_layout()
            out3d = ex_dir / "scene.png"
            fig.savefig(out3d, dpi=160)
            plt.close(fig)

            # Direction raster plots: ±X/±Y/±Z if available.
            panels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
            fig = plt.figure(figsize=(16, 9))
            for pi, key in enumerate(panels):
                ax = fig.add_subplot(2, 3, pi + 1)
                di = dir_idx.get(key, None)
                if di is None:
                    ax.axis("off")
                    ax.set_title(f"{key} (missing dir)")
                    continue

                ras = compute_rasters_for_direction(
                    fruit_local=fruit_local,
                    leaf_local_all=leaf_local_all,
                    leaf_leafidx_all=leaf_leafidx_all,
                    leaf_local_sel=leaf_local_sel,
                    pix=pix,
                    delta=delta,
                    basis=bases[di],
                )
                if ras.get("P", 0) <= 0:
                    ax.axis("off")
                    ax.set_title(f"{key} (no pixels)")
                    continue

                # Visualization overlays (pixel grid).
                # 'potential' mode: show only fruit+leaf and the selected leaf's potential occlusion.
                # 'full' mode: additionally show union + exclusive attribution.
                fruit = ras["fruit_mask"]
                leaf = ras["leaf_mask"]
                leaf_over = ras["leaf_over_fruit_mask"]
                potm = ras["pot_mask"]
                unionm = ras["union_mask"]
                exclm = ras["excl_mask"]
                crop = ras["crop"]

                img = np.zeros((*fruit.shape, 3), dtype=np.float32)
                img[fruit] = np.array([0.45, 0.45, 0.45], dtype=np.float32)

                # Leaf pixels not overlapping fruit silhouette.
                leaf_out = leaf & (~fruit)
                if bool(leaf_out.any()):
                    img[leaf_out] = np.maximum(img[leaf_out], np.array([0.0, 0.35, 0.0], dtype=np.float32))

                # Leaf pixels overlapping fruit silhouette (regardless of depth).
                leaf_on = leaf_over
                if bool(leaf_on.any()):
                    img[leaf_on] = np.maximum(img[leaf_on], np.array([0.0, 0.65, 0.0], dtype=np.float32))

                if str(args.panel_mode) == "full":
                    # Union occlusion pixels.
                    if bool(unionm.any()):
                        img[unionm] = np.maximum(img[unionm], np.array([0.95, 0.0, 0.0], dtype=np.float32))

                # Potential occlusion pixels.
                if bool(potm.any()):
                    img[potm] = np.array([0.0, 0.95, 0.0], dtype=np.float32)

                if str(args.panel_mode) == "full":
                    # Exclusive/frontmost pixels.
                    if bool(exclm.any()):
                        img[exclm] = np.array([0.2, 0.2, 0.95], dtype=np.float32)

                imgc = img[crop]
                ax.imshow(imgc, origin="lower", interpolation="nearest")

                # Sidecar values (for this fruit + selected leaf).
                pot_sc = float(pot_rows[jbest, di])
                pot_re = float(ras["potential"])
                leaf_on = float(ras["leaf_over_fruit"])

                if str(args.panel_mode) == "full":
                    union_sc = float(fruit_union[fr, di])
                    title = (
                        f"{key} di={di}\n"
                        f"union sidecar={union_sc:.3f}\n"
                        f"potential sidecar={pot_sc:.3f} recomputed={pot_re:.3f}\n"
                        f"leaf_on_fruit (proj overlap)={leaf_on:.3f}"
                    )
                else:
                    title = (
                        f"{key} di={di}\n"
                        f"potential sidecar={pot_sc:.3f} recomputed={pot_re:.3f}\n"
                        f"leaf_on_fruit (proj overlap)={leaf_on:.3f}"
                    )
                ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

            if str(args.panel_mode) == "full":
                supt = f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (pixel grid: red=union, green=potential, blue=exclusive)"
            else:
                supt = f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (pixel grid: green=potential (in vacuum))"
            fig.suptitle(supt, fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out2d = ex_dir / "pc_pixel.png"
            fig.savefig(out2d, dpi=160)
            plt.close(fig)

            total_written += 2

            # Optional: OBB comparison panels (pixel-grid silhouettes + polygon hull overlays).
            if bool(getattr(args, "compare_obb", False)):
                leaf_node = leaf_nodes_by_iid.get(int(liid))
                if leaf_node is not None:
                    occl_edges = (occl.get("edges") or []) if isinstance(occl, dict) else []
                    fruit_node_id = str(fnode.get("id"))

                    # Candidate leaves for OBB union proxy: all occlusion edges for this fruit.
                    cand_leaf_nodes: List[Dict[str, Any]] = []
                    cand_leaf_cent_local: List[np.ndarray] = []
                    seen_leaf_ids: set[str] = set()
                    for e in occl_edges:
                        if str(e.get("fruit_id")) != fruit_node_id:
                            continue
                        leaf_id = str(e.get("leaf_id"))
                        if leaf_id in seen_leaf_ids:
                            continue
                        seen_leaf_ids.add(leaf_id)
                        n = node_by_id.get(leaf_id)
                        if n is None or n.get("type") != "leaf":
                            continue
                        lc = np.asarray(n.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
                        lc_local = (lc - fruit_center) @ R.T
                        cand_leaf_nodes.append(n)
                        cand_leaf_cent_local.append(lc_local.astype(np.float32, copy=False))

                    # Stored OBB leaf->fruit overlaps for the selected pair (if present).
                    raw_overlap_vec: Optional[List[float]] = None
                    sel_leaf_node_id = str(leaf_node.get("id"))
                    for e in occl_edges:
                        if str(e.get("fruit_id")) == fruit_node_id and str(e.get("leaf_id")) == sel_leaf_node_id:
                            ro = e.get("raw_overlap")
                            if isinstance(ro, list):
                                raw_overlap_vec = [float(x) for x in ro]
                            break

                    obb_union_vec = fruit_obb_union_by_iid.get(int(fiid))

                    # (A) OBB pixel-grid panels.
                    fig = plt.figure(figsize=(16, 9))
                    for pi, key in enumerate(panels):
                        ax = fig.add_subplot(2, 3, pi + 1)
                        di = dir_idx.get(key, None)
                        if di is None:
                            ax.axis("off")
                            ax.set_title(f"{key} (missing dir)")
                            continue

                        d, u, v = bases[int(di)]
                        fruit_poly = project_obb_hull_uv(node=fnode, fruit_center=fruit_center, R=R, u=u, v=v)
                        leaf_poly = project_obb_hull_uv(node=leaf_node, fruit_center=fruit_center, R=R, u=u, v=v)

                        union_polys: List[List[Tuple[float, float]]] = []
                        union_use: List[bool] = []
                        if str(args.panel_mode) == "full":
                            for n, lc_local in zip(cand_leaf_nodes, cand_leaf_cent_local):
                                union_polys.append(project_obb_hull_uv(node=n, fruit_center=fruit_center, R=R, u=u, v=v))
                                union_use.append(float(np.dot(lc_local, d)) > 0.0)

                        ras_obb = rasterize_obb_polys(
                            pix=float(pix),
                            fruit_poly=fruit_poly,
                            leaf_poly=leaf_poly,
                            union_leaf_polys=union_polys,
                            union_use=union_use,
                        )
                        if int(ras_obb.get("H", 0)) <= 0:
                            ax.axis("off")
                            ax.set_title(f"{key} (no pixels)")
                            continue

                        fruit_m = ras_obb["fruit_mask"]
                        leaf_m = ras_obb["leaf_mask"]
                        union_m = ras_obb["union_leaf_mask"] & fruit_m

                        img = np.zeros((*fruit_m.shape, 3), dtype=np.float32)
                        img[fruit_m] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
                        if str(args.panel_mode) == "full":
                            if bool(union_m.any()):
                                img[union_m] = np.maximum(img[union_m], np.array([0.95, 0.0, 0.0], dtype=np.float32))

                        # Selected leaf silhouette (OBB).
                        leaf_out = leaf_m & (~fruit_m)
                        if bool(leaf_out.any()):
                            img[leaf_out] = np.maximum(img[leaf_out], np.array([0.0, 0.35, 0.0], dtype=np.float32))
                        leaf_on = leaf_m & fruit_m
                        if bool(leaf_on.any()):
                            img[leaf_on] = np.maximum(img[leaf_on], np.array([0.0, 0.95, 0.0], dtype=np.float32))

                        ax.imshow(img, origin="lower", interpolation="nearest")

                        raw_sc = float(raw_overlap_vec[int(di)]) if raw_overlap_vec is not None and int(di) < len(raw_overlap_vec) else float("nan")
                        pot_sc = float(pot_rows[jbest, int(di)])
                        if str(args.panel_mode) == "full":
                            pc_u = float(fruit_union[int(fr), int(di)])
                            obb_u = float(obb_union_vec[int(di)]) if obb_union_vec is not None and int(di) < int(obb_union_vec.size) else float("nan")
                            ax.set_title(
                                f"{key} di={di}\nPC union={pc_u:.3f}  OBB union_raw={obb_u:.3f}\nOBB raw_overlap(sel)={raw_sc:.3f}",
                                fontsize=9,
                            )
                        else:
                            ax.set_title(
                                f"{key} di={di}\nPC potential={pot_sc:.3f}\nOBB raw_overlap(sel)={raw_sc:.3f}",
                                fontsize=9,
                            )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    if str(args.panel_mode) == "full":
                        supt = f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (OBB pixel grid: red=union proxy, green=selected leaf silhouette)"
                    else:
                        supt = f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (OBB pixel grid: green=selected leaf overlap (in vacuum))"
                    fig.suptitle(supt, fontsize=11)
                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    out_obb_pixel = ex_dir / "obb_pixel.png"
                    fig.savefig(out_obb_pixel, dpi=160)
                    plt.close(fig)

                    # (B) OBB polygon panels (non-pixel).
                    fig = plt.figure(figsize=(16, 9))
                    leaf_center_local = (np.asarray(leaf_node.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32) - fruit_center) @ R.T
                    for pi, key in enumerate(panels):
                        ax = fig.add_subplot(2, 3, pi + 1)
                        di = dir_idx.get(key, None)
                        if di is None:
                            ax.axis("off")
                            ax.set_title(f"{key} (missing dir)")
                            continue

                        d, u, v = bases[int(di)]
                        fruit_poly = project_obb_hull_uv(node=fnode, fruit_center=fruit_center, R=R, u=u, v=v)
                        leaf_poly = project_obb_hull_uv(node=leaf_node, fruit_center=fruit_center, R=R, u=u, v=v)
                        inter: List[List[float]] = []
                        if float(np.dot(leaf_center_local, d)) > 0.0 and fruit_poly and leaf_poly:
                            inter = clip_polygon(leaf_poly, fruit_poly)

                        if fruit_poly:
                            fx = [p[0] for p in fruit_poly] + [fruit_poly[0][0]]
                            fy = [p[1] for p in fruit_poly] + [fruit_poly[0][1]]
                            ax.plot(fx, fy, color="tab:orange", lw=1.5)
                        if leaf_poly:
                            lx = [p[0] for p in leaf_poly] + [leaf_poly[0][0]]
                            ly = [p[1] for p in leaf_poly] + [leaf_poly[0][1]]
                            ax.plot(lx, ly, color="tab:green", lw=1.5)
                        if inter:
                            ix = [p[0] for p in inter] + [inter[0][0]]
                            iy = [p[1] for p in inter] + [inter[0][1]]
                            fill_col = "tab:red" if str(args.panel_mode) == "full" else "tab:green"
                            ax.fill(ix, iy, color=fill_col, alpha=0.25, linewidth=0.0)

                        fruit_area = polygon_area(fruit_poly)
                        inter_area = polygon_area(inter) if inter else 0.0
                        raw_computed = (inter_area / fruit_area) if fruit_area > 0 else 0.0
                        raw_sc = float(raw_overlap_vec[int(di)]) if raw_overlap_vec is not None and int(di) < len(raw_overlap_vec) else float("nan")
                        ax.set_title(f"{key} di={di}\nraw_overlap stored={raw_sc:.3f} computed={raw_computed:.3f}", fontsize=9)
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_xticks([])
                        ax.set_yticks([])

                    fig.suptitle(f"{scene_id} fruit_iid={fiid} leaf_iid={liid} (OBB polygons: outlines + intersection)", fontsize=11)
                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    out_obb_poly = ex_dir / "obb_poly.png"
                    fig.savefig(out_obb_poly, dpi=160)
                    plt.close(fig)

                    total_written += 2

            examples_written += 1
            print(f"Wrote {ex_dir}", flush=True)

    # Write dataset-level correlation plots at the top-level (same folder that contains example subfolders).
    if bool(args.compare_obb):
        if corr_pc and corr_obb:
            pc_all = np.concatenate(corr_pc, axis=0).astype(np.float64, copy=False)
            obb_all = np.concatenate(corr_obb, axis=0).astype(np.float64, copy=False)
            n = int(pc_all.size)

            # Pearson correlation (computed directly to avoid scipy dependency).
            x = obb_all
            y = pc_all
            xm = float(x.mean()) if n else 0.0
            ym = float(y.mean()) if n else 0.0
            xv = x - xm
            yv = y - ym
            cov = float(np.mean(xv * yv)) if n else 0.0
            xs = float(np.sqrt(np.mean(xv * xv))) if n else 0.0
            ys = float(np.sqrt(np.mean(yv * yv))) if n else 0.0
            r = cov / (xs * ys + 1e-12)

            # (1) Scatter (subsampled for speed).
            max_scatter = 200_000
            if n > max_scatter:
                take = rng.choice(n, size=max_scatter, replace=False)
                xs_sc = x[take]
                ys_sc = y[take]
            else:
                xs_sc = x
                ys_sc = y
            fig = plt.figure(figsize=(6.0, 6.0))
            ax = fig.add_subplot(111)
            ax.scatter(xs_sc, ys_sc, s=2, alpha=0.15, edgecolors="none")
            ax.plot([0, 1], [0, 1], "--", color="black", lw=1.0, alpha=0.7)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("OBB union (union_raw)")
            ax.set_ylabel("Pointcloud union (z-buffer)")
            ax.set_title(f"OBB vs pointcloud union (n={n:,}, r={r:.3f})")
            fig.tight_layout()
            out_scatter = out_dir / "01_obb_vs_pc_union_scatter.png"
            fig.savefig(out_scatter, dpi=200)
            plt.close(fig)

            # (2) Hexbin density plot (log counts).
            fig = plt.figure(figsize=(6.0, 6.0))
            ax = fig.add_subplot(111)
            hb = ax.hexbin(
                x,
                y,
                gridsize=70,
                extent=(0.0, 1.0, 0.0, 1.0),
                bins="log",
                mincnt=1,
                cmap="viridis",
            )
            ax.plot([0, 1], [0, 1], "--", color="white", lw=1.0, alpha=0.8)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("OBB union (union_raw)")
            ax.set_ylabel("Pointcloud union (z-buffer)")
            ax.set_title(f"OBB vs pointcloud union (hexbin, n={n:,}, r={r:.3f})")
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("log10(count)")
            fig.tight_layout()
            out_hex = out_dir / "02_obb_vs_pc_union_hexbin.png"
            fig.savefig(out_hex, dpi=200)
            plt.close(fig)

            # (3) Residual plot: (PC - OBB) vs OBB (density + mean trend).
            resid = (y - x).astype(np.float64, copy=False)
            # Robust y-limits (99% interval) for readability.
            try:
                lo, hi = np.quantile(resid, [0.01, 0.99]).tolist()
            except Exception:
                lo, hi = float(resid.min(initial=-1.0)), float(resid.max(initial=1.0))
            lo = max(-1.0, float(lo))
            hi = min(1.0, float(hi))
            if hi - lo < 1e-6:
                lo, hi = -0.05, 0.05
            pad = 0.05 * (hi - lo)
            ylo = max(-1.0, lo - pad)
            yhi = min(1.0, hi + pad)

            fig = plt.figure(figsize=(7.0, 5.0))
            ax = fig.add_subplot(111)
            hb = ax.hexbin(
                x,
                resid,
                gridsize=70,
                extent=(0.0, 1.0, ylo, yhi),
                bins="log",
                mincnt=1,
                cmap="magma",
            )
            ax.axhline(0.0, color="white", lw=1.0, alpha=0.8)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel("OBB union (union_raw)")
            ax.set_ylabel("Residual: pointcloud − OBB")
            ax.set_title("Residuals show OBB over-/under-estimation vs union_raw")

            # Mean residual trend over OBB bins.
            nb = 25
            bins = np.linspace(0.0, 1.0, nb + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            mean_r = np.full((nb,), np.nan, dtype=np.float64)
            for bi in range(nb):
                m = (x >= bins[bi]) & (x < bins[bi + 1])
                if bool(m.any()):
                    mean_r[bi] = float(np.mean(resid[m]))
            ax.plot(centers, mean_r, color="cyan", lw=2.0, alpha=0.9, label="mean residual")
            ax.legend(loc="lower left")

            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("log10(count)")
            fig.tight_layout()
            out_res = out_dir / "03_pc_minus_obb_vs_obb_residual_hexbin.png"
            fig.savefig(out_res, dpi=200)
            plt.close(fig)

            total_written += 3
            print(f"Wrote correlation plots: {out_scatter.name}, {out_hex.name}, {out_res.name}", flush=True)
        else:
            print("[warn] --compare_obb set but no JSON union_raw targets were found; skipping correlation plots.", flush=True)

    print(f"Done. Wrote {total_written} images into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
