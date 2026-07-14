#!/usr/bin/env python3
"""Sanity check OBB axes layout against corner projections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pepper_scene_graphs.configs.config import Config
from pepper_scene_graphs.data.dataset import load_graph_dataset
from pepper_scene_graphs.data.transforms import feature_slices


def obb_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> np.ndarray:
  """Compute 8 OBB corners from center/axes/extents."""
  half = extents * 0.5
  corners = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        offset = (
          axes[:, 0] * (sx * half[0])
          + axes[:, 1] * (sy * half[1])
          + axes[:, 2] * (sz * half[2])
        )
        corners.append(center + offset)
  return np.asarray(corners, dtype=np.float32)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Check OBB axes layout.")
  parser.add_argument("--dataset-path", required=True, help="Path to dataset (.json or .pt)")
  parser.add_argument("--num-nodes", type=int, default=5, help="Number of nodes to sample")
  parser.add_argument("--seed", type=int, default=0, help="RNG seed")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg = Config()
  data_list = load_graph_dataset(args.dataset_path, cfg)
  if not data_list:
    raise SystemExit("Dataset is empty.")

  rng = np.random.default_rng(args.seed)
  data = data_list[0]
  if data.x.numel() == 0:
    raise SystemExit("No nodes found.")

  sl = feature_slices(cfg.features)
  if "axes" not in sl or "extents" not in sl:
    raise SystemExit("Axes/extents features are disabled in config.")

  a0, a1 = sl["axes"]
  e0, e1 = sl["extents"]
  num_nodes = data.x.size(0)
  sample = rng.choice(num_nodes, size=min(args.num_nodes, num_nodes), replace=False)
  dirs = np.array(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [1.0, 1.0, 0.0],
      [1.0, 0.0, 1.0],
      [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
  )
  dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

  max_err = 0.0
  for idx in sample:
    center = data.pos[idx].cpu().numpy()
    axes = data.x[idx, a0:a1].view(3, 3).cpu().numpy()
    extents = data.x[idx, e0:e1].cpu().numpy()
    corners = obb_corners(center, axes, extents)

    for d in dirs:
      proj = corners @ d
      c = center @ d
      radius_corner = float(np.max(proj) - c)
      # Compute radius via axis dot method.
      dots = np.abs(axes.T @ d)
      radius_axis = float(np.sum(dots * (extents * 0.5)))
      err = abs(radius_corner - radius_axis)
      max_err = max(max_err, err)

  print(f"Checked {len(sample)} nodes. Max radius error: {max_err:.6f}")


if __name__ == "__main__":
  main()
