#!/usr/bin/env python3
"""
Convert a textured OBJ mesh (OBJ+MTL+texture) into a colored point cloud saved as .npz.

This is intended for quick ingestion of "real" reconstructions into the same *point cloud* conventions
used by this repo (XYZ + normals + RGB), so we can later add instance segmentation / scene-graph steps.

Output .npz keys:
  - points:  (N, 3) float32, world units (whatever the OBJ uses)
  - normals: (N, 3) float32, unit-length (approx)
  - colors:  (N, 3) uint8, RGB in [0,255]

Notes:
  - OBJ vertex colors are rare; this script samples colors from the diffuse texture (map_Kd) via UVs.
  - We sample points uniformly over *surface area* by picking triangles proportional to area and
    sampling barycentric coordinates.
  - Optional voxel downsampling keeps one point per voxel (fast, preserves sharp colors).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple


def _load_texture_from_obj(obj_path: Path) -> Path:
    """Resolve the first `map_Kd` referenced by the OBJ's mtllib (very small parser)."""
    mtllib = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
                break
    if not mtllib:
        raise SystemExit(f"[abort] OBJ has no mtllib: {obj_path}")

    mtl_path = (obj_path.parent / mtllib).resolve()
    if not mtl_path.exists():
        raise SystemExit(f"[abort] MTL not found: {mtl_path}")

    tex_rel = None
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("map_Kd "):
                tex_rel = line.split(maxsplit=1)[1].strip()
                break
    if not tex_rel:
        raise SystemExit(f"[abort] MTL has no map_Kd texture: {mtl_path}")

    tex_path = (mtl_path.parent / tex_rel).resolve()
    if tex_path.exists():
        return tex_path
    # Fallback: relative to OBJ directory (some exporters do this).
    tex_path = (obj_path.parent / tex_rel).resolve()
    if tex_path.exists():
        return tex_path
    raise SystemExit(f"[abort] texture not found (map_Kd={tex_rel!r}) relative to {mtl_path} or {obj_path}")


def _normalize(v):
    import numpy as np

    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)


def _sample_barycentric(rng, n: int):
    """Return barycentric weights (w0,w1,w2) for uniform sampling on a triangle."""
    import numpy as np

    r1 = rng.random(n, dtype=np.float64)
    r2 = rng.random(n, dtype=np.float64)
    s = np.sqrt(r1)
    w0 = 1.0 - s
    w1 = s * (1.0 - r2)
    w2 = s * r2
    return w0, w1, w2


def _triangle_areas(vertices, triangles, *, chunk: int = 200_000):
    import numpy as np

    F = int(triangles.shape[0])
    areas = np.empty((F,), dtype=np.float64)
    for start in range(0, F, int(chunk)):
        end = min(F, start + int(chunk))
        tri = triangles[start:end]
        a = vertices[tri[:, 0]]
        b = vertices[tri[:, 1]]
        c = vertices[tri[:, 2]]
        ab = b - a
        ac = c - a
        cr = np.cross(ab, ac)
        areas[start:end] = 0.5 * np.linalg.norm(cr, axis=1)
    return areas


def _uv_to_pixel(uv, w: int, h: int, *, flip_v: bool, wrap: bool):
    import numpy as np

    u = uv[..., 0].astype(np.float64, copy=False)
    v = uv[..., 1].astype(np.float64, copy=False)
    if wrap:
        u = np.mod(u, 1.0)
        v = np.mod(v, 1.0)
    else:
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
    if flip_v:
        v = 1.0 - v
    x = np.clip((u * (w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip((v * (h - 1)).astype(np.int64), 0, h - 1)
    return x, y


def _voxel_downsample_first(points, normals, colors, voxel_size: float):
    """Keep the first point per voxel (fast, deterministic-ish)."""
    import numpy as np

    if voxel_size <= 0:
        return points, normals, colors
    vox = np.floor(points / float(voxel_size)).astype(np.int32, copy=False)
    # unique rows with first index
    _, first = np.unique(vox, axis=0, return_index=True)
    first = np.sort(first)
    return points[first], normals[first], colors[first]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample a textured OBJ mesh into a colored point cloud .npz")
    ap.add_argument("obj", type=str, help="Input OBJ path (expects mtllib + map_Kd texture).")
    ap.add_argument("--out", type=str, default=None, help="Output .npz path (default: <obj>.npz)")
    ap.add_argument("--texture", type=str, default=None, help="Override texture path (default: parse mtllib/map_Kd).")
    ap.add_argument("--num_samples", type=int, default=600_000, help="Number of surface samples before downsampling.")
    ap.add_argument("--voxel", type=float, default=0.004, help="Voxel size (meters) for downsampling; 0 disables.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    ap.add_argument("--flip_v", action="store_true", help="Flip V coordinate (OBJ uses V-up; image uses top-down).")
    ap.add_argument("--wrap_uv", action="store_true", help="Wrap UVs with mod 1.0 instead of clamping.")
    args = ap.parse_args()

    import numpy as np
    import open3d as o3d
    from PIL import Image

    obj_path = Path(args.obj)
    if not obj_path.exists():
        raise SystemExit(f"[abort] input does not exist: {obj_path}")

    out_path = Path(args.out) if args.out else obj_path.with_suffix(".npz")
    tex_path = Path(args.texture) if args.texture else _load_texture_from_obj(obj_path)
    if not tex_path.exists():
        raise SystemExit(f"[abort] texture does not exist: {tex_path}")

    print(f"[info] reading mesh: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise SystemExit(f"[abort] failed to read mesh vertices/triangles from: {obj_path}")
    if not mesh.has_vertex_normals():
        print("[info] mesh has no vertex normals; computing...")
        mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices, dtype=np.float64)
    VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    T = np.asarray(mesh.triangles, dtype=np.int64)

    tri_uvs = getattr(mesh, "triangle_uvs", None)
    if tri_uvs is None:
        raise SystemExit("[abort] mesh has no triangle_uvs; cannot sample textured colors.")
    UV = np.asarray(tri_uvs, dtype=np.float64)
    if UV.ndim != 2 or UV.shape[1] != 2 or UV.shape[0] != 3 * T.shape[0]:
        raise SystemExit(f"[abort] unexpected triangle_uvs shape: {UV.shape} (expected {(3*T.shape[0], 2)})")
    UV = UV.reshape((-1, 3, 2))  # [F,3,2]

    print(f"[info] mesh: verts={V.shape[0]} tris={T.shape[0]} uv_tris={UV.shape[0]}")

    print(f"[info] reading texture: {tex_path}")
    img = Image.open(tex_path)
    img = img.convert("RGBA")
    tex = np.asarray(img, dtype=np.uint8)  # [H,W,4]
    H, W = int(tex.shape[0]), int(tex.shape[1])
    print(f"[info] texture: {W}x{H}")

    n = int(args.num_samples)
    if n <= 0:
        raise SystemExit("[abort] --num_samples must be >0")
    rng = np.random.default_rng(int(args.seed))

    print("[info] computing triangle areas...")
    areas = _triangle_areas(V, T)
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise SystemExit(f"[abort] invalid total area: {total_area}")
    cdf = np.cumsum(areas)

    print(f"[info] sampling {n} surface points (area-weighted)...")
    r = rng.random(n, dtype=np.float64) * cdf[-1]
    tri_idx = np.searchsorted(cdf, r, side="right").astype(np.int64, copy=False)  # [n]
    tri = T[tri_idx]  # [n,3]

    v0 = V[tri[:, 0]]
    v1 = V[tri[:, 1]]
    v2 = V[tri[:, 2]]
    n0 = VN[tri[:, 0]]
    n1 = VN[tri[:, 1]]
    n2 = VN[tri[:, 2]]

    uv = UV[tri_idx]  # [n,3,2]
    uv0 = uv[:, 0, :]
    uv1 = uv[:, 1, :]
    uv2 = uv[:, 2, :]

    w0, w1, w2 = _sample_barycentric(rng, n)
    w0 = w0[:, None]
    w1 = w1[:, None]
    w2 = w2[:, None]

    pts = (w0 * v0 + w1 * v1 + w2 * v2).astype(np.float32, copy=False)
    norms = _normalize(w0 * n0 + w1 * n1 + w2 * n2).astype(np.float32, copy=False)
    uvp = (w0 * uv0 + w1 * uv1 + w2 * uv2)

    x, y = _uv_to_pixel(uvp, W, H, flip_v=bool(args.flip_v), wrap=bool(args.wrap_uv))
    rgb = tex[y, x, :3].astype(np.uint8, copy=False)

    if float(args.voxel) > 0.0:
        before = int(pts.shape[0])
        pts, norms, rgb = _voxel_downsample_first(pts, norms, rgb, float(args.voxel))
        after = int(pts.shape[0])
        print(f"[info] voxel downsample {before} -> {after} (voxel={float(args.voxel):g}m)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, points=pts, normals=norms, colors=rgb)
    print(f"[ok] wrote {int(pts.shape[0])} points -> {out_path}")


if __name__ == "__main__":
    main()

