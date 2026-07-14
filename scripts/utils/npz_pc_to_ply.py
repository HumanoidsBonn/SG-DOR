#!/usr/bin/env python3
"""
Convert a pepper dataset point cloud saved as .npz into a .ply file that Blender can import.

Expected .npz keys:
  - points:   (N, 3) float32/float64
Optional keys:
  - colors:   (N, 3) uint8 or float in [0, 1]
  - normals:  (N, 3) float32/float64

By default we write a binary little-endian PLY with XYZ + RGB (if available) and normals (if available).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "[abort] numpy is required for this script. "
            "Activate the environment you used for dataset generation (e.g. conda env 'bproc').\n"
            f"Import error: {e}"
        )
    return np


def _as_uint8_rgb(np, colors) -> Optional["np.ndarray"]:
    if colors is None:
        return None
    c = np.asarray(colors)
    if c.size == 0:
        return None
    if c.dtype == np.uint8:
        return c
    # Common case: float colors in [0, 1]
    c = c.astype(np.float32, copy=False)
    if float(c.max(initial=0.0)) <= 1.0 + 1e-6:
        c = np.clip(c, 0.0, 1.0) * 255.0
    c = np.clip(c, 0.0, 255.0).astype(np.uint8)
    return c


def _subsample_indices(np, n: int, max_points: int, mode: str, seed: int) -> "np.ndarray":
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    max_points = int(max_points)
    if mode == "stride":
        return np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    # default: random
    rng = np.random.default_rng(int(seed))
    return rng.choice(n, size=max_points, replace=False).astype(np.int64, copy=False)


def _build_ply_header(
    n: int,
    *,
    ascii: bool,
    has_normals: bool,
    has_colors: bool,
) -> str:
    fmt = "ascii 1.0" if ascii else "binary_little_endian 1.0"
    lines = [
        "ply",
        f"format {fmt}",
        f"element vertex {int(n)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        lines += [
            "property float nx",
            "property float ny",
            "property float nz",
        ]
    if has_colors:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _write_ply_binary(np, out_path: Path, points, normals, colors) -> None:
    has_normals = normals is not None and int(normals.shape[0]) == int(points.shape[0])
    has_colors = colors is not None and int(colors.shape[0]) == int(points.shape[0])

    header = _build_ply_header(
        int(points.shape[0]),
        ascii=False,
        has_normals=has_normals,
        has_colors=has_colors,
    )
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_normals:
        dtype += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if has_colors:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    arr = np.empty((int(points.shape[0]),), dtype=np.dtype(dtype))
    p = np.asarray(points, dtype=np.float32)
    arr["x"] = p[:, 0]
    arr["y"] = p[:, 1]
    arr["z"] = p[:, 2]
    if has_normals:
        n = np.asarray(normals, dtype=np.float32)
        arr["nx"] = n[:, 0]
        arr["ny"] = n[:, 1]
        arr["nz"] = n[:, 2]
    if has_colors:
        c = _as_uint8_rgb(np, colors)
        assert c is not None
        arr["red"] = c[:, 0]
        arr["green"] = c[:, 1]
        arr["blue"] = c[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def _write_ply_ascii(np, out_path: Path, points, normals, colors) -> None:
    has_normals = normals is not None and int(normals.shape[0]) == int(points.shape[0])
    has_colors = colors is not None and int(colors.shape[0]) == int(points.shape[0])
    header = _build_ply_header(
        int(points.shape[0]),
        ascii=True,
        has_normals=has_normals,
        has_colors=has_colors,
    )
    c = _as_uint8_rgb(np, colors) if has_colors else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        p = np.asarray(points, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32) if has_normals else None
        for i in range(int(p.shape[0])):
            row = [f"{p[i,0]:.6f}", f"{p[i,1]:.6f}", f"{p[i,2]:.6f}"]
            if n is not None:
                row += [f"{n[i,0]:.6f}", f"{n[i,1]:.6f}", f"{n[i,2]:.6f}"]
            if c is not None:
                row += [str(int(c[i, 0])), str(int(c[i, 1])), str(int(c[i, 2]))]
            f.write(" ".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert dataset .npz pointclouds to Blender-importable .ply")
    ap.add_argument("npz", type=str, help="Input .npz point cloud (keys: points[, colors][, normals])")
    ap.add_argument("--out", type=str, default=None, help="Output .ply path (default: <npz>.ply)")
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="If >0, subsample to at most this many points (useful for Blender).",
    )
    ap.add_argument(
        "--subsample_mode",
        type=str,
        default="random",
        choices=["random", "stride"],
        help="Subsampling strategy when --max_points is used.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for random subsampling.")
    ap.add_argument("--ascii", action="store_true", help="Write ASCII PLY (larger/slower). Default: binary.")
    args = ap.parse_args()

    np = _try_import_numpy()

    in_path = Path(args.npz)
    if not in_path.exists():
        raise SystemExit(f"[abort] input does not exist: {in_path}")
    out_path = Path(args.out) if args.out else in_path.with_suffix(".ply")

    with np.load(in_path) as z:
        if "points" not in z:
            raise SystemExit(f"[abort] missing 'points' in {in_path}")
        points = z["points"]
        colors = z["colors"] if "colors" in z else None
        normals = z["normals"] if "normals" in z else None

    points = np.asarray(points)
    if points.ndim != 2 or int(points.shape[1]) != 3:
        raise SystemExit(f"[abort] points must be (N,3), got {points.shape}")
    n = int(points.shape[0])

    idx = _subsample_indices(np, n, int(args.max_points), str(args.subsample_mode), int(args.seed))
    points = points[idx]
    if colors is not None:
        colors = np.asarray(colors)[idx]
    if normals is not None:
        normals = np.asarray(normals)[idx]

    if args.ascii:
        _write_ply_ascii(np, out_path, points, normals, colors)
    else:
        _write_ply_binary(np, out_path, points, normals, colors)

    print(f"[ok] wrote {int(points.shape[0])} points -> {out_path}")


if __name__ == "__main__":
    main()

