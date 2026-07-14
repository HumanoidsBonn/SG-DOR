#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for segmented dataset visualization."""
  parser = argparse.ArgumentParser(description="Compare a segmented scene against the original dataset.")
  parser.add_argument("--unsegmented", required=True, help="Path to unsegmented dataset JSON")
  parser.add_argument("--segmented", required=True, help="Path to segmented dataset JSON")
  parser.add_argument("--scene-index", type=int, default=0, help="Scene index to visualize")
  return parser.parse_args()


def load_scene(path: str, idx: int) -> Dict[str, Any]:
  """Load a single scene graph from a dataset JSON file."""
  dataset_path = Path(path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)
  with open(dataset_path, "r", encoding="utf-8") as f:
    graphs = json.load(f)
  if not isinstance(graphs, list):
    raise ValueError("Expected a list of graphs in JSON.")
  if idx < 0 or idx >= len(graphs):
    raise IndexError(f"Scene index {idx} out of range for {len(graphs)} scenes.")
  return graphs[idx]


def obb_corners(center: Any, axes: Any, extents: Any) -> Optional[np.ndarray]:
  """Compute the 8 OBB corners given center, axes (columns), and extents."""
  axes = np.asarray(axes, dtype=np.float32)
  center = np.asarray(center, dtype=np.float32)
  extents = np.asarray(extents, dtype=np.float32)
  if axes.shape != (3, 3) or extents.size != 3:
    return None
  half = 0.5 * extents
  # axes are stored as columns
  a0, a1, a2 = axes[:, 0], axes[:, 1], axes[:, 2]
  corners = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        corners.append(center + sx * half[0] * a0 + sy * half[1] * a1 + sz * half[2] * a2)
  return np.asarray(corners, dtype=np.float32)


def draw_obb(
  ax: Any,
  center: Any,
  axes: Any,
  extents: Any,
  color: str,
  alpha: float = 0.8,
  lw: float = 1.0,
) -> None:
  """Render a single OBB as wireframe lines."""
  corners = obb_corners(center, axes, extents)
  if corners is None:
    return
  edges = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
  ]
  for i, j in edges:
    xs = [corners[i, 0], corners[j, 0]]
    ys = [corners[i, 1], corners[j, 1]]
    zs = [corners[i, 2], corners[j, 2]]
    ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=lw)


def draw_scene(ax: Any, graph: Dict[str, Any], title: str) -> None:
  """Draw a single scene with stem/fruit OBBs and leaf/peduncle centroids."""
  ax.set_title(title)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.grid(True, alpha=0.2)

  # Render nodes by type with minimal clutter (OBBs for stems/fruits, dots for others).
  for node in graph.get("nodes", []):
    ntype = node.get("type")
    if ntype in {"stem", "fruit"}:
      obb = node.get("obb", {})
      draw_obb(
        ax,
        obb.get("center", node.get("centroid", [0, 0, 0])),
        obb.get("axes", np.eye(3)),
        obb.get("extents", [0, 0, 0]),
        color="tab:red" if ntype == "stem" else "tab:orange",
        alpha=0.7,
        lw=0.8,
      )
    elif ntype in {"leaf", "peduncle"}:
      c = np.asarray(node.get("centroid", [0, 0, 0]), dtype=np.float32)
      size = 12.0
      ext = node.get("obb", {}).get("extents", [0.0, 0.0, 0.0])
      if ext:
        size = max(6.0, 30.0 * float(np.mean(ext)))
      color = "tab:green" if ntype == "leaf" else "tab:blue"
      ax.scatter([c[0]], [c[1]], [c[2]], s=size, color=color, alpha=0.8)

  # Equal aspect ratio based on node centroids.
  pts = [n.get("centroid") for n in graph.get("nodes", []) if n.get("centroid") is not None]
  if pts:
    pts = np.asarray(pts, dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges)) if float(np.max(ranges)) > 0 else 1.0
    mids = 0.5 * (mins + maxs)
    ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
    ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
    ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)


def main() -> None:
  """Load a paired segmented/unsegmented scene and display them side-by-side."""
  args = parse_args()
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    raise RuntimeError("matplotlib is required for this script.") from exc

  g_unseg = load_scene(args.unsegmented, args.scene_index)
  g_seg = load_scene(args.segmented, args.scene_index)

  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1, 2, 1, projection="3d")
  ax2 = fig.add_subplot(1, 2, 2, projection="3d")
  draw_scene(ax1, g_unseg, "Unsegmented")
  draw_scene(ax2, g_seg, "Segmented")
  plt.tight_layout()

  out_dir = Path(args.unsegmented).resolve().parent / "debug"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"segmented_compare_scene{args.scene_index}.png"
  fig.savefig(out_path, dpi=200)
  print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
  main()
