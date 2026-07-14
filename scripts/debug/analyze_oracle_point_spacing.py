#!/usr/bin/env python3
"""
Estimate point spacing statistics for the Oracle pointcloud dataset.

Primary use: pick a reasonable `--xyz_jitter` scale grounded in the data,
instead of arbitrary values.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _percentiles(x: np.ndarray, ps: Tuple[float, ...]) -> List[float]:
    return [float(np.percentile(x, p)) for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root (contains data/*.npz)")
    ap.add_argument("--use_4mm", action="store_true", default=True, help="Use *_4mm.npz files (default)")
    ap.add_argument("--no_4mm", action="store_false", dest="use_4mm")
    ap.add_argument("--drop_shoots", action="store_true", help="Drop shoot points (semantics==5)")
    ap.add_argument("--max_scenes", type=int, default=25, help="Number of scenes to sample")
    ap.add_argument("--points_query", type=int, default=5000, help="Points per scene to query NN distances for")
    ap.add_argument("--points_fit_cap", type=int, default=200000, help="Cap points per scene used to fit NN structure")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--out_dir", type=str, default="analysis", help="Output directory for optional artifacts")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    data_dir = Path(args.data_root) / "data"
    if not data_dir.exists():
        data_dir = Path(args.data_root)

    suffix = "_4mm.npz" if args.use_4mm else "_pc.npz"
    files = sorted(glob.glob(str(data_dir / f"*{suffix}")))
    if not files:
        raise SystemExit(f"No pointcloud files matching *{suffix} under {data_dir}")

    max_scenes = min(int(args.max_scenes), len(files))
    picks = rng.choice(len(files), size=max_scenes, replace=False).tolist()
    picked_files = [files[i] for i in picks]

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise SystemExit(f"scikit-learn is required for this script: {e}")

    all_nn: List[np.ndarray] = []
    scene_sizes: List[int] = []

    for fp in picked_files:
        z = np.load(fp)
        pts = z["points"].astype(np.float32, copy=False)
        sem = z["semantics"].astype(np.int32, copy=False)
        inst = z["instances"].astype(np.int32, copy=False)

        m = inst > 0
        if args.drop_shoots:
            m = m & (sem != 5)
        if not bool(m.any()):
            continue
        pts = pts[m]

        if pts.shape[0] < 10:
            continue
        scene_sizes.append(int(pts.shape[0]))

        # Fit on a capped subset for runtime.
        fit_pts = pts
        if fit_pts.shape[0] > int(args.points_fit_cap):
            idx = rng.choice(fit_pts.shape[0], size=int(args.points_fit_cap), replace=False)
            fit_pts = fit_pts[idx]

        # Query on a random subset (or all).
        q_pts = pts
        if q_pts.shape[0] > int(args.points_query):
            idx = rng.choice(q_pts.shape[0], size=int(args.points_query), replace=False)
            q_pts = q_pts[idx]

        nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
        nn.fit(fit_pts)
        dists, _ = nn.kneighbors(q_pts, n_neighbors=2, return_distance=True)
        # dists[:,0] is self-distance ~0; dists[:,1] is NN.
        all_nn.append(dists[:, 1].astype(np.float32, copy=False))

    if not all_nn:
        raise SystemExit("No valid scenes processed.")

    nn_all = np.concatenate(all_nn, axis=0)
    ps = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    q = _percentiles(nn_all, ps)

    med = float(np.median(nn_all))
    mean = float(np.mean(nn_all))

    print("\nOracle point spacing (nearest-neighbor distance) stats:")
    print(f"  scenes: {len(all_nn)} (requested {max_scenes})")
    print(f"  points queried total: {nn_all.size}")
    print(f"  mean nn dist:   {mean:.6f} m ({mean*1000:.2f} mm)")
    print(f"  median nn dist: {med:.6f} m ({med*1000:.2f} mm)")
    for p, v in zip(ps, q):
        print(f"  p{p:02d}: {v:.6f} m ({v*1000:.2f} mm)")

    # Suggest jitter scales as fractions of median NN distance.
    print("\nSuggested xyz_jitter std (rule-of-thumb from median NN distance):")
    for frac in (0.10, 0.25, 0.50, 0.75):
        j = frac * med
        print(f"  {frac:>4.2f} * median: {j:.6f} m ({j*1000:.2f} mm)")

    # Optional save (for quick offline plotting).
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"oracle_nn_dists_{Path(args.data_root).name}_{'4mm' if args.use_4mm else 'pc'}_n{len(all_nn)}.npz"
    np.savez_compressed(out_npz, nn=nn_all, scene_sizes=np.asarray(scene_sizes, dtype=np.int32))
    print(f"\nWrote: {out_npz}")

    # Optional plot if matplotlib exists.
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 4))
        plt.hist(nn_all * 1000.0, bins=80)
        plt.xlabel("Nearest-neighbor distance (mm)")
        plt.ylabel("Count")
        plt.title("Oracle point spacing (NN distance)")
        plt.tight_layout()
        out_png = out_dir / (out_npz.stem + ".png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Wrote: {out_png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

