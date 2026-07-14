#!/usr/bin/env python3

"""Visualize occlusion projections for a single leaf-fruit pair.

This script loads a dataset JSON, picks one leaf-fruit occlusion edge from a
chosen scene (filtered by how many nonzero raw overlap directions it has),
and saves a multi-panel PNG with:
  - 3D OBBs (fruit, leaf, parent stem) and fruit-local frame axes
  - Six 2D projection panels for ±X/±Y/±Z directions

The output is saved into a `debug/` subfolder next to the dataset JSON.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(description="Visualize occlusion overlap for a leaf-fruit pair.")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
  parser.add_argument("--scene-index", type=int, default=0, help="Scene index to visualize")
  parser.add_argument("--num-pairs", type=int, default=5, help="Number of pairs to visualize")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for pair selection")
  return parser.parse_args()


def obb_corners(center: Sequence[float], axes: Sequence[Sequence[float]], extents: Sequence[float]) -> np.ndarray:
  """Compute the 8 OBB corners for a node."""
  axes = np.asarray(axes, dtype=np.float32)
  center = np.asarray(center, dtype=np.float32)
  extents = np.asarray(extents, dtype=np.float32)
  half = 0.5 * extents
  a0, a1, a2 = axes[:, 0], axes[:, 1], axes[:, 2]
  corners = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        corners.append(center + sx * half[0] * a0 + sy * half[1] * a1 + sz * half[2] * a2)
  return np.asarray(corners, dtype=np.float32)


def build_occlusion_dirs() -> List[np.ndarray]:
  """Build the 18 canonical occlusion directions (axes + diagonals)."""
  base = [
    np.array([1.0, 0.0, 0.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
    np.array([0.0, 0.0, 1.0], dtype=float),
  ]
  dirs = []
  for v in base:
    dirs.append(v)
    dirs.append(-v)
  for i, j in [(0, 1), (0, 2), (1, 2)]:
    for s1 in (-1.0, 1.0):
      for s2 in (-1.0, 1.0):
        v = base[i] * s1 + base[j] * s2
        v = v / np.linalg.norm(v)
        dirs.append(v)
  return dirs


def build_dir_bases(dirs: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Build per-direction orthonormal bases for projection."""
  bases = []
  z = np.array([0.0, 0.0, 1.0], dtype=float)
  x = np.array([1.0, 0.0, 0.0], dtype=float)
  for d in dirs:
    d = d / np.linalg.norm(d)
    tmp = z if abs(np.dot(d, z)) < 0.9 else x
    u = np.cross(tmp, d)
    if np.linalg.norm(u) < 1e-6:
      tmp = x
      u = np.cross(tmp, d)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    v = v / np.linalg.norm(v)
    bases.append((d, u, v))
  return bases


def fruit_local_frame(fruit_center: np.ndarray, stem_center: np.ndarray) -> np.ndarray:
  """Build fruit-local frame with +X pointing toward the stem in the XY plane."""
  x_vec = np.array(
    [stem_center[0] - fruit_center[0], stem_center[1] - fruit_center[1], 0.0],
    dtype=float,
  )
  if np.linalg.norm(x_vec) < 1e-6:
    x_vec = np.array([1.0, 0.0, 0.0], dtype=float)
  x_unit = x_vec / np.linalg.norm(x_vec)
  z_unit = np.array([0.0, 0.0, 1.0], dtype=float)
  y_vec = np.cross(z_unit, x_unit)
  if np.linalg.norm(y_vec) < 1e-6:
    y_vec = np.array([0.0, 1.0, 0.0], dtype=float)
  y_unit = y_vec / np.linalg.norm(y_vec)
  x_unit = np.cross(y_unit, z_unit)
  x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-8)
  return np.vstack([x_unit, y_unit, z_unit])


def cross_2d(a: Sequence[float], b: Sequence[float]) -> float:
  """2D cross product (scalar)."""
  return a[0] * b[1] - a[1] * b[0]


def convex_hull(points_2d: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
  """Compute the convex hull of 2D points (monotonic chain)."""
  if len(points_2d) <= 1:
    return [(float(p[0]), float(p[1])) for p in points_2d]
  pts = sorted(set((float(p[0]), float(p[1])) for p in points_2d))
  if len(pts) <= 1:
    return pts

  def build_half(seq):
    half = []
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
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % len(poly)]
    area += x1 * y2 - x2 * y1
  return abs(area) * 0.5


def clip_polygon(subject: Sequence[Sequence[float]], clipper: Sequence[Sequence[float]]) -> List[List[float]]:
  """Clip a convex polygon by another convex polygon."""
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
    if abs(denom) < 1e-8:
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


def main() -> None:
  """Entry point to render a leaf-fruit occlusion visualization."""
  args = parse_args()
  dataset_path = Path(args.dataset_path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)

  with dataset_path.open("r", encoding="utf-8") as f:
    graphs = json.load(f)
  if not isinstance(graphs, list):
    raise ValueError("Expected a list of graphs in JSON.")
  if args.scene_index < 0 or args.scene_index >= len(graphs):
    raise IndexError(f"Scene index {args.scene_index} out of range (0..{len(graphs)-1}).")

  graph = graphs[args.scene_index]
  occl = graph.get("occlusion")
  if not occl or "edges" not in occl:
    raise ValueError("No occlusion block found in this scene.")

  random.seed(args.seed)
  edges = occl.get("edges", [])
  filtered = []
  for e in edges:
    raw = e.get("raw_overlap", [])
    raw_xyz = raw[:6]
    if any(float(v) > 0.0 for v in raw_xyz):
      filtered.append(e)
  if not filtered:
    raise ValueError("No occlusion edges with nonzero overlap among ±XYZ.")

  rng = random.Random(args.seed)
  rng.shuffle(filtered)

  node_by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in graph.get("nodes", [])}
  attached_to = {
    e["source"]: e["target"]
    for e in graph.get("edges", [])
    if e.get("label") == "attached_to"
  }

  pair_infos = []
  for e in filtered:
    if len(pair_infos) >= args.num_pairs:
      break
    leaf_id = e["leaf_id"]
    fruit_id = e["fruit_id"]
    leaf_node = node_by_id.get(leaf_id)
    fruit_node = node_by_id.get(fruit_id)
    if leaf_node is None or fruit_node is None:
      continue
    ped_id = attached_to.get(fruit_id)
    stem_id = attached_to.get(ped_id) if ped_id else None
    stem_node = node_by_id.get(stem_id) if stem_id else None
    if stem_node is None:
      continue
    pair_infos.append((leaf_node, fruit_node, stem_node))

  if not pair_infos:
    raise ValueError("No valid leaf-fruit-stem triplets found for visualization.")

  # Precompute direction bases.
  dirs = build_occlusion_dirs()
  dir_bases = build_dir_bases(dirs)

  weight = float(occl.get("weight", 0.0))

  try:
    import matplotlib.pyplot as plt
    # matplotlib only needed inside the CLI to allow import without GUI deps.
  except Exception as exc:
    raise RuntimeError("matplotlib is required for this script.") from exc

  def draw_obb(ax, node, color):
    corners = obb_corners(node["centroid"], node["obb"]["axes"], node["obb"]["extents"])
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
      ax.plot(xs, ys, zs, color=color, alpha=0.7, linewidth=1.0)

  dir_labels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
  out_dir = dataset_path.parent / "debug"
  out_dir.mkdir(parents=True, exist_ok=True)

  for pair_idx, (leaf_node, fruit_node, stem_node) in enumerate(pair_infos):
    fruit_center = np.asarray(fruit_node["centroid"], dtype=float)
    stem_center = np.asarray(stem_node["centroid"], dtype=float)
    leaf_center = np.asarray(leaf_node["centroid"], dtype=float)
    dist = float(np.linalg.norm(leaf_center - fruit_center))

    # Build local frame and corner sets.
    R = fruit_local_frame(fruit_center, stem_center)
    fruit_corners = obb_corners(fruit_center, fruit_node["obb"]["axes"], fruit_node["obb"]["extents"])
    leaf_corners = obb_corners(leaf_center, leaf_node["obb"]["axes"], leaf_node["obb"]["extents"])
    fruit_local = (R @ (fruit_corners - fruit_center).T).T
    leaf_local = (R @ (leaf_corners - fruit_center).T).T
    leaf_center_local = R @ (leaf_center - fruit_center)

    fruit_extents = np.asarray(fruit_node["obb"]["extents"], dtype=float)
    fruit_radius = 0.5 * float(np.linalg.norm(fruit_extents))
    if fruit_radius <= 1e-6:
      fruit_radius = 1e-6
    dist_ratio = dist / fruit_radius
    weight_fac = 1.0 / (1.0 + weight * dist_ratio * dist_ratio)

    # Precompute fruit projection polygons for all directions.
    fruit_proj = []
    for (_d, u, v) in dir_bases:
      u_vals = fruit_local @ u
      v_vals = fruit_local @ v
      fruit_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
      fruit_hull = convex_hull(fruit_pts)
      fruit_area = polygon_area(fruit_hull)
      fruit_proj.append((fruit_hull, fruit_area))

    # Layout for a single pair: 3 rows x 3 cols (3D + six 2D projections).
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    draw_obb(ax3d, fruit_node, "tab:orange")
    draw_obb(ax3d, leaf_node, "tab:green")
    draw_obb(ax3d, stem_node, "tab:red")

    axis_len = 0.3 * float(fruit_radius)
    ax3d.quiver(
      fruit_center[0], fruit_center[1], fruit_center[2],
      R[0, 0] * axis_len, R[0, 1] * axis_len, R[0, 2] * axis_len,
      color="r",
    )
    ax3d.quiver(
      fruit_center[0], fruit_center[1], fruit_center[2],
      R[1, 0] * axis_len, R[1, 1] * axis_len, R[1, 2] * axis_len,
      color="g",
    )
    ax3d.quiver(
      fruit_center[0], fruit_center[1], fruit_center[2],
      R[2, 0] * axis_len, R[2, 1] * axis_len, R[2, 2] * axis_len,
      color="b",
    )
    ax3d.set_title(f"Pair {pair_idx + 1}: dist={dist:.3f} wfac={weight_fac:.3f}")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    for idx in range(6):
      row = idx // 2
      col = 1 + (idx % 2)
      ax = fig.add_subplot(gs[row, col])
      d, u, v = dir_bases[idx]
      fruit_hull, fruit_area = fruit_proj[idx]
      u_vals = leaf_local @ u
      v_vals = leaf_local @ v
      leaf_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
      leaf_hull = convex_hull(leaf_pts)
      raw = 0.0
      if float(np.dot(leaf_center_local, d)) > 0.0 and fruit_area > 0.0 and leaf_hull:
        inter = clip_polygon(leaf_hull, fruit_hull)
        inter_area = polygon_area(inter)
        raw = inter_area / fruit_area if fruit_area > 0.0 else 0.0
      weighted = raw * weight_fac
      if fruit_hull:
        fx = [p[0] for p in fruit_hull] + [fruit_hull[0][0]]
        fy = [p[1] for p in fruit_hull] + [fruit_hull[0][1]]
        ax.plot(fx, fy, color="tab:orange", linewidth=1.5)
      if leaf_hull:
        lx = [p[0] for p in leaf_hull] + [leaf_hull[0][0]]
        ly = [p[1] for p in leaf_hull] + [leaf_hull[0][1]]
        ax.plot(lx, ly, color="tab:green", linewidth=1.5)
      # Scale the view to projected polygon extents so shapes are visible.
      all_x = [p[0] for p in fruit_hull + leaf_hull] if (fruit_hull or leaf_hull) else [0.0, 1.0]
      all_y = [p[1] for p in fruit_hull + leaf_hull] if (fruit_hull or leaf_hull) else [0.0, 1.0]
      umin = min(all_x)
      umax = max(all_x)
      vmin = min(all_y)
      vmax = max(all_y)
      pad_u = 0.05 * max(1e-6, umax - umin)
      pad_v = 0.05 * max(1e-6, vmax - vmin)
      ax.set_xlim(umin - pad_u, umax + pad_u)
      ax.set_ylim(vmin - pad_v, vmax + pad_v)
      ax.set_aspect("equal", adjustable="box")
      ax.set_title(f"{dir_labels[idx]} raw={raw:.3f} w={weighted:.3f}")
      ax.set_xlabel("u")
      ax.set_ylabel("v")
      ax.grid(True, alpha=0.2)

    fig.suptitle(
      f"Scene {args.scene_index} Pair {pair_idx + 1} (dist={dist:.3f}, wfac={weight_fac:.3f})",
      fontsize=12,
    )
    out_path = out_dir / f"occlusion_pair_scene{args.scene_index}_idx{pair_idx + 1}.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    print(f"Saved occlusion visualization to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
  main()
