#!/usr/bin/env python3

"""Recompute occlusion overlaps using silhouette (convex hull) projections.

This script loads an existing scene-graph JSON dataset and recomputes the
leaf-fruit occlusion overlaps using polygon overlap in the projected u/v plane.
It replaces the occlusion block in each scene and writes a new JSON file
with a `_silhouettes` suffix by default.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MAX_DIST = 0.15
DEFAULT_WEIGHT = 0.2
DEFAULT_GRID_CELL_SIZE = 0.02


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for occlusion postprocessing."""
  parser = argparse.ArgumentParser(description="Postprocess occlusion overlaps with silhouette projections.")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
  parser.add_argument("--out", default=None, help="Output path (default: *_silhouettes.json)")
  parser.add_argument("--max-dist", type=float, default=None, help="Override occlusion max distance")
  parser.add_argument("--weight", type=float, default=None, help="Override occlusion weight")
  parser.add_argument("--grid-cell-size", type=float, default=None, help="Override occlusion grid cell size")
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
    # Build one half of the hull (lower or upper) using a stack.
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

  output = [list(p) for p in subject]
  for i in range(len(clipper)):
    # Clip the subject polygon by each edge of the clipper.
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


def build_leaf_grid(leaf_entries: Sequence[Dict[str, Any]], cell_size: float) -> Dict[Tuple[int, int, int], List[Dict[str, Any]]]:
  """Bucket leaves into a 3D grid for proximity queries."""
  grid: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
  inv = 1.0 / max(cell_size, 1e-6)
  for entry in leaf_entries:
    x, y, z = entry["center"]
    key = (int(x * inv), int(y * inv), int(z * inv))
    grid.setdefault(key, []).append(entry)
  return grid


def leaf_candidates_from_grid(
  fruit_center: np.ndarray,
  leaf_grid: Dict[Tuple[int, int, int], List[Dict[str, Any]]],
  cell_size: float,
  max_dist: float,
) -> List[Dict[str, Any]]:
  """Collect leaf candidates within a radius using the grid buckets."""
  if cell_size <= 0:
    return []
  inv = 1.0 / cell_size
  fx, fy, fz = fruit_center.tolist()
  cx, cy, cz = int(fx * inv), int(fy * inv), int(fz * inv)
  r = int(np.ceil(max_dist * inv))
  candidates: List[Dict[str, Any]] = []
  seen = set()
  # Scan neighboring grid cells and keep leaves within max_dist.
  for dx in range(-r, r + 1):
    for dy in range(-r, r + 1):
      for dz in range(-r, r + 1):
        key = (cx + dx, cy + dy, cz + dz)
        for entry in leaf_grid.get(key, []):
          if entry["id"] in seen:
            continue
          seen.add(entry["id"])
          dist = np.linalg.norm(entry["center"] - fruit_center)
          if dist <= max_dist:
            candidates.append(entry)
  return candidates


def resolve_param(scene_occl: Optional[Dict[str, Any]], override: Optional[float], default: float, key: str) -> float:
  """Resolve occlusion parameter using override, scene metadata, or default."""
  if override is not None:
    return float(override)
  if scene_occl and key in scene_occl:
    return float(scene_occl[key])
  return float(default)


def main() -> None:
  """Entry point: recompute occlusion overlaps and write a new dataset JSON."""
  args = parse_args()
  dataset_path = Path(args.dataset_path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)

  with dataset_path.open("r", encoding="utf-8") as f:
    graphs = json.load(f)
  if not isinstance(graphs, list):
    raise ValueError("Expected a list of graphs in JSON.")

  out_path = Path(args.out) if args.out else dataset_path.with_name(f"{dataset_path.stem}_silhouettes.json")

  occlusion_dirs = build_occlusion_dirs()
  occlusion_dir_bases = build_dir_bases(occlusion_dirs)

  updated_graphs = []
  for graph in graphs:
    # Resolve per-dataset occlusion params (override -> scene -> defaults).
    scene_occl = graph.get("occlusion")
    max_dist = resolve_param(scene_occl, args.max_dist, DEFAULT_MAX_DIST, "max_dist")
    weight = resolve_param(scene_occl, args.weight, DEFAULT_WEIGHT, "weight")
    grid_cell_size = resolve_param(scene_occl, args.grid_cell_size, DEFAULT_GRID_CELL_SIZE, "grid_cell_size")

    # Build node lookups and leaf grid once per scene.
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}
    attached_to = {
      e["source"]: e["target"]
      for e in edges
      if e.get("label") == "attached_to"
    }
    leaf_entries = []
    for node in nodes:
      if node.get("type") != "leaf":
        continue
      center = np.array(node["centroid"], dtype=float)
      axes = np.array(node["obb"]["axes"], dtype=float)
      extents = np.array(node["obb"]["extents"], dtype=float)
      corners = obb_corners(center, axes, extents)
      leaf_entries.append({"id": node["id"], "center": center, "corners": corners})
    leaf_grid = build_leaf_grid(leaf_entries, grid_cell_size)

    occlusion_edges: List[Dict[str, Any]] = []

    # Iterate fruits and compute silhouette overlaps against nearby leaves.
    for node in nodes:
      if node.get("type") != "fruit":
        continue
      fruit_id = node["id"]
      ped_id = attached_to.get(fruit_id)
      stem_id = attached_to.get(ped_id) if ped_id else None
      stem_node = node_by_id.get(stem_id) if stem_id else None
      if stem_node is None:
        continue

      fruit_center = np.array(node["centroid"], dtype=float)
      stem_center = np.array(stem_node["centroid"], dtype=float)
      R = fruit_local_frame(fruit_center, stem_center)
      fruit_axes = np.array(node["obb"]["axes"], dtype=float)
      fruit_extents = np.array(node["obb"]["extents"], dtype=float)
      fruit_radius = 0.5 * float(np.linalg.norm(fruit_extents))
      if fruit_radius <= 1e-6:
        fruit_radius = 1e-6
      fruit_corners = obb_corners(fruit_center, fruit_axes, fruit_extents)
      fruit_local = (R @ (fruit_corners - fruit_center).T).T

      # Precompute fruit silhouette projections for all directions.
      fruit_proj = []
      for (_d, u, v) in occlusion_dir_bases:
        u_vals = fruit_local @ u
        v_vals = fruit_local @ v
        fruit_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
        fruit_hull = convex_hull(fruit_pts)
        fruit_area = polygon_area(fruit_hull)
        fruit_proj.append((fruit_hull, fruit_area))

      candidates = leaf_candidates_from_grid(fruit_center, leaf_grid, grid_cell_size, max_dist)
      for leaf in candidates:
        # Compute per-leaf distance weight and project into fruit-local frame.
        dist = float(np.linalg.norm(leaf["center"] - fruit_center))
        dist_ratio = dist / fruit_radius
        weight_fac = 1.0 / (1.0 + weight * dist_ratio * dist_ratio)
        leaf_center_local = R @ (leaf["center"] - fruit_center)
        leaf_local = (R @ (leaf["corners"] - fruit_center).T).T

        raw_vals = []
        weighted_vals = []
        has_overlap = False

        for (d, u, v), fp in zip(occlusion_dir_bases, fruit_proj):
          fruit_hull, fruit_area = fp
          if float(np.dot(leaf_center_local, d)) <= 0.0 or fruit_area <= 0.0:
            raw = 0.0
          else:
            u_vals = leaf_local @ u
            v_vals = leaf_local @ v
            leaf_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
            leaf_hull = convex_hull(leaf_pts)
            if not leaf_hull:
              raw = 0.0
            else:
              inter = clip_polygon(leaf_hull, fruit_hull)
              inter_area = polygon_area(inter)
              raw = inter_area / fruit_area if fruit_area > 0.0 else 0.0
          raw = float(raw)
          weighted = raw * weight_fac
          raw_vals.append(raw)
          weighted_vals.append(float(weighted))
          if raw > 0.0:
            has_overlap = True

        if has_overlap:
          occlusion_edges.append(
            {
              "leaf_id": leaf["id"],
              "fruit_id": fruit_id,
              "dist": dist,
              "raw_overlap": raw_vals,
              "weighted_overlap": weighted_vals,
            }
          )

    graph = dict(graph)
    graph["occlusion"] = {
      "dirs": [d.tolist() for d in occlusion_dirs],
      "edges": occlusion_edges,
      "max_dist": max_dist,
      "weight": weight,
      "weight_mode": "inv_sq_fruit_radius",
      "grid_cell_size": grid_cell_size,
    }
    updated_graphs.append(graph)

  with out_path.open("w", encoding="utf-8") as f:
    json.dump(updated_graphs, f, indent=2)
  print(f"Wrote silhouette occlusion dataset to {out_path}")


if __name__ == "__main__":
  main()
