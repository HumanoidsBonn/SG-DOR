#!/usr/bin/env python3
"""Visualize occlusion candidate sampling stages for a random fruit per scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from pepper_scene_graphs.configs.config import Config
from pepper_scene_graphs.constants import NODE_CLASSES


CATEGORY_STYLES: Dict[str, Dict[str, Any]] = {
  "neutral": {"color": "#b0b0b0", "alpha": 0.25, "size": 6},
  "base_kept": {"color": "#1f77b4", "alpha": 0.9, "size": 20},
  "base_thin_drop": {"color": "#4c78a8", "alpha": 0.5, "size": 12},
  "base_none_drop": {"color": "#6baed6", "alpha": 0.5, "size": 12},
  "ring_kept": {"color": "#ff7f0e", "alpha": 0.9, "size": 20},
  "ring_thin_drop": {"color": "#f28e2b", "alpha": 0.5, "size": 12},
  "ring_none_drop": {"color": "#ffbe7d", "alpha": 0.5, "size": 12},
  "far_kept": {"color": "#2ca02c", "alpha": 0.9, "size": 20},
  "far_thin_drop": {"color": "#59a14f", "alpha": 0.5, "size": 12},
  "far_none_drop": {"color": "#8cd17d", "alpha": 0.5, "size": 12},
}


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for candidate visualization."""
  parser = argparse.ArgumentParser(description="Visualize occlusion candidate sampling stages.")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
  parser.add_argument("--num-scenes", type=int, default=5, help="Number of scenes to visualize")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for fruit sampling")
  parser.add_argument("--scene-offset", type=int, default=0, help="Starting scene index")
  parser.add_argument(
    "--sphere-res",
    type=int,
    default=24,
    help="Resolution for distance spheres (lower is faster).",
  )
  return parser.parse_args()


def load_graphs(path: str) -> List[Dict[str, Any]]:
  """Load raw scene graphs from a JSON file."""
  dataset_path = Path(path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)
  with dataset_path.open("r", encoding="utf-8") as f:
    graphs = json.load(f)
  if not isinstance(graphs, list):
    raise ValueError("Expected a list of graphs in JSON.")
  return graphs


def occlusion_axis_pairs(mode: str) -> List[Tuple[int, int]]:
  """Return index pairs (pos, neg) for each axis in the occlusion encoding."""
  if mode == "axes3":
    return [(0, 1), (2, 3), (4, 5)]
  if mode == "axes9":
    return [
      (0, 1),    # X
      (2, 3),    # Y
      (4, 5),    # Z
      (9, 6),    # X+Y vs -X-Y
      (8, 7),    # X-Y vs -X+Y
      (13, 10),  # X+Z vs -X-Z
      (12, 11),  # X-Z vs -X+Z
      (17, 14),  # Y+Z vs -Y-Z
      (16, 15),  # Y-Z vs -Y+Z
    ]
  raise ValueError(f"Unknown dir_mode: {mode}")


def occlusion_dir_indices(mode: str) -> List[int]:
  """Flatten axis pairs into a list of indices."""
  pairs = occlusion_axis_pairs(mode)
  return [idx for pair in pairs for idx in pair]


def bin_occlusion_values(values: np.ndarray, none_thresh: float, high_thresh: float) -> np.ndarray:
  """Bin occlusion overlaps into 5 classes per axis.

  Classes: 0=none, 1=+low, 2=+high, 3=-low, 4=-high.

  Args:
      values: Array of shape (N, 2 * num_axes) with pos/neg pairs.
      none_thresh: Threshold below which overlaps are treated as none.
      high_thresh: Threshold above which overlaps are treated as high.

  Returns:
      Integer array of shape (N, num_axes).
  """
  if values.size == 0:
    return np.zeros((0, 0), dtype=np.int64)
  pos = values[:, 0::2]
  neg = values[:, 1::2]
  max_vals = np.maximum(pos, neg)
  none_mask = max_vals <= none_thresh
  high_mask = max_vals >= high_thresh
  sign_pos = pos >= neg

  bins = np.zeros_like(pos, dtype=np.int64)
  bins = np.where(sign_pos & ~none_mask & ~high_mask, 1, bins)
  bins = np.where(sign_pos & high_mask, 2, bins)
  bins = np.where(~sign_pos & ~none_mask & ~high_mask, 3, bins)
  bins = np.where(~sign_pos & high_mask, 4, bins)
  return bins


def bin_counts_by_axis(bins: np.ndarray) -> Dict[str, Dict[str, int]]:
  """Count bin occurrences per axis."""
  if bins.size == 0:
    return {}
  counts: Dict[str, Dict[str, int]] = {}
  labels = ["none", "+low", "+high", "-low", "-high"]
  for axis_idx in range(bins.shape[1]):
    axis_bins = bins[:, axis_idx]
    axis_counts = {labels[i]: int(np.sum(axis_bins == i)) for i in range(len(labels))}
    counts[f"axis_{axis_idx}"] = axis_counts
  return counts


def build_occlusion_map(graph: Dict[str, Any], target_mode: str) -> Dict[Tuple[int, int], np.ndarray]:
  """Build a map from (leaf_id, fruit_id) to occlusion overlap values."""
  occl = graph.get("occlusion") or {}
  edges = occl.get("edges", [])
  mapping: Dict[Tuple[int, int], np.ndarray] = {}
  for e in edges:
    leaf_id = e.get("leaf_id")
    fruit_id = e.get("fruit_id")
    if target_mode == "weighted":
      vals = e.get("weighted_overlap")
    else:
      vals = e.get("raw_overlap")
    if leaf_id is None or fruit_id is None or vals is None:
      continue
    mapping[(leaf_id, fruit_id)] = np.asarray(vals, dtype=float)
  return mapping


def _sample_without_replacement(indices: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
  """Sample up to num_samples indices without replacement."""
  if num_samples <= 0 or indices.size == 0:
    return indices[:0]
  if indices.size <= num_samples:
    return indices
  perm = rng.permutation(indices.size)
  return indices[perm[:num_samples]]


def _sphere_mesh(radius: float, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generate a sphere mesh for visualization."""
  u = np.linspace(0.0, 2.0 * np.pi, resolution)
  v = np.linspace(0.0, np.pi, resolution)
  x = radius * np.outer(np.cos(u), np.sin(v))
  y = radius * np.outer(np.sin(u), np.sin(v))
  z = radius * np.outer(np.ones_like(u), np.cos(v))
  return x, y, z


def _draw_stem_axes(ax: Any, center: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> None:
  """Draw stem axes as line segments."""
  for i in range(3):
    axis = axes[:, i]
    half = 0.5 * extents[i]
    p0 = center - axis * half
    p1 = center + axis * half
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="tab:red", linewidth=1.0, alpha=0.7)


def _draw_obb(ax: Any, center: np.ndarray, axes: np.ndarray, extents: np.ndarray, color: str) -> None:
  """Draw an OBB wireframe."""
  half = 0.5 * extents
  a0, a1, a2 = axes[:, 0], axes[:, 1], axes[:, 2]
  corners = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        corners.append(center + sx * half[0] * a0 + sy * half[1] * a1 + sz * half[2] * a2)
  corners = np.asarray(corners, dtype=float)
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
    ax.plot(
      [corners[i, 0], corners[j, 0]],
      [corners[i, 1], corners[j, 1]],
      [corners[i, 2], corners[j, 2]],
      color=color,
      linewidth=0.8,
      alpha=0.8,
    )


def _assign_category(categories: Dict[int, str], ids: List[int], label: str) -> None:
  """Assign a category label to a list of leaf ids."""
  for leaf_id in ids:
    categories[leaf_id] = label


def visualize_scene(
  graph: Dict[str, Any],
  cfg: Config,
  scene_index: int,
  rng: np.random.Generator,
  out_dir: Path,
  sphere_res: int,
) -> None:
  """Render a single scene with candidate stages for one fruit."""
  nodes = graph.get("nodes", [])
  if not nodes:
    return

  # Collect node ids and centroids.
  leaf_nodes = [n for n in nodes if n.get("type") == "leaf"]
  fruit_nodes = [n for n in nodes if n.get("type") == "fruit"]
  stem_nodes = [n for n in nodes if n.get("type") == "stem"]
  if not leaf_nodes or not fruit_nodes:
    return

  leaf_ids = [n.get("id") for n in leaf_nodes]
  fruit_ids = [n.get("id") for n in fruit_nodes]
  leaf_pos = np.asarray([n.get("centroid", [0.0, 0.0, 0.0]) for n in leaf_nodes], dtype=float)
  fruit_pos = np.asarray([n.get("centroid", [0.0, 0.0, 0.0]) for n in fruit_nodes], dtype=float)

  cand_cfg = cfg.occlusion.candidates
  target_cfg = cfg.occlusion.targets
  dir_idx = occlusion_dir_indices(str(target_cfg.dir_mode).lower())

  occl_map = build_occlusion_map(graph, target_cfg.target_mode)

  # Choose a fruit with at least one base candidate.
  max_dist = float(cand_cfg.max_dist)
  base_counts = []
  for j in range(len(fruit_ids)):
    dist = np.linalg.norm(leaf_pos - fruit_pos[j], axis=1)
    base_counts.append(int(np.sum(dist <= max_dist)))
  valid_fruits = [i for i, c in enumerate(base_counts) if c > 0]
  if not valid_fruits:
    return
  fruit_idx_local = int(rng.choice(valid_fruits))
  fruit_id = fruit_ids[fruit_idx_local]

  # Compute distance-based masks.
  dists = np.linalg.norm(leaf_pos - fruit_pos[fruit_idx_local], axis=1)
  base_mask = dists <= max_dist
  ring_mask = (dists > max_dist) & (dists <= float(cand_cfg.ring_max_dist))
  far_mask = dists > float(cand_cfg.ring_max_dist)

  base_idx = np.where(base_mask)[0]
  ring_idx = np.where(ring_mask)[0]
  far_idx = np.where(far_mask)[0]

  # Extra negatives from ring/far.
  n_extra = int(np.ceil(base_idx.size * float(cand_cfg.extra_neg_ratio)))
  n_ring = int(round(n_extra * float(cand_cfg.ring_frac)))
  n_far = max(0, n_extra - n_ring)
  ring_sample = _sample_without_replacement(ring_idx, n_ring, rng)
  far_sample = _sample_without_replacement(far_idx, n_far, rng)
  extra_idx = np.concatenate([ring_sample, far_sample], axis=0) if ring_sample.size + far_sample.size > 0 else np.array([], dtype=int)

  # Candidate list before thinning.
  cand_idx = np.concatenate([base_idx, extra_idx], axis=0) if extra_idx.size > 0 else base_idx.copy()
  cand_origin = {}
  for idx in base_idx.tolist():
    cand_origin[idx] = "base"
  for idx in ring_sample.tolist():
    cand_origin[idx] = "ring"
  for idx in far_sample.tolist():
    cand_origin[idx] = "far"

  # Prepare stats tracking per stage.
  stats: Dict[str, Any] = {
    "scene_index": scene_index,
    "fruit_id": fruit_id,
    "counts": {},
    "bin_counts": {},
  }

  def _values_for(indices: List[int]) -> np.ndarray:
    if not indices:
      return np.zeros((0, len(dir_idx)), dtype=float)
    vals = []
    for idx in indices:
      leaf_id = leaf_ids[idx]
      raw_vals = occl_map.get((leaf_id, fruit_id))
      if raw_vals is None:
        raw_vals = np.zeros((18,), dtype=float)
      vals.append(raw_vals[dir_idx])
    return np.asarray(vals, dtype=float)

  def _record_stage(stage: str, indices: List[int]) -> None:
    stats["counts"][stage] = int(len(indices))
    vals = _values_for(indices)
    bins = bin_occlusion_values(vals, float(target_cfg.none_thresh), float(target_cfg.high_thresh))
    stats["bin_counts"][stage] = bin_counts_by_axis(bins)

  # Stage: base candidates only.
  _record_stage("base_all", base_idx.tolist())

  # Stage: base + extra (pre-thinning).
  _record_stage("pre_thin_all", cand_idx.tolist())

  # Thinning stage.
  thin_keep_prob = float(cand_cfg.thin_keep_prob)
  thin_keep = rng.random(cand_idx.size) < thin_keep_prob if cand_idx.size > 0 else np.array([], dtype=bool)
  thin_keep_idx = cand_idx[thin_keep]
  thin_drop_idx = cand_idx[~thin_keep]

  if thin_keep_idx.size == 0 and base_idx.size > 0:
    thin_keep_idx = base_idx[:1]
    thin_drop_idx = np.setdiff1d(cand_idx, thin_keep_idx)

  _record_stage("post_thin", thin_keep_idx.tolist())

  # None-keep filtering.
  none_keep_prob = float(cand_cfg.none_keep_prob)
  none_thresh = float(target_cfg.none_thresh)
  kept_idx = []
  none_drop_idx = []

  for idx in thin_keep_idx.tolist():
    leaf_id = leaf_ids[idx]
    vals = occl_map.get((leaf_id, fruit_id))
    if vals is None:
      vals = np.zeros((18,), dtype=float)
    dir_vals = vals[dir_idx]
    all_none = bool(np.all(dir_vals <= none_thresh))
    if all_none and rng.random() >= none_keep_prob:
      none_drop_idx.append(idx)
      continue
    kept_idx.append(idx)

  if not kept_idx and thin_keep_idx.size > 0:
    kept_idx = [int(thin_keep_idx[0])]
    none_drop_idx = [idx for idx in none_drop_idx if idx != kept_idx[0]]

  _record_stage("post_none_keep", kept_idx)

  # Assign categories for visualization.
  categories = {leaf_id: "neutral" for leaf_id in leaf_ids}

  base_thin_drop = [leaf_ids[i] for i in thin_drop_idx.tolist() if cand_origin.get(i) == "base"]
  ring_thin_drop = [leaf_ids[i] for i in thin_drop_idx.tolist() if cand_origin.get(i) == "ring"]
  far_thin_drop = [leaf_ids[i] for i in thin_drop_idx.tolist() if cand_origin.get(i) == "far"]

  base_none_drop = [leaf_ids[i] for i in none_drop_idx if cand_origin.get(i) == "base"]
  ring_none_drop = [leaf_ids[i] for i in none_drop_idx if cand_origin.get(i) == "ring"]
  far_none_drop = [leaf_ids[i] for i in none_drop_idx if cand_origin.get(i) == "far"]

  base_kept = [leaf_ids[i] for i in kept_idx if cand_origin.get(i) == "base"]
  ring_kept = [leaf_ids[i] for i in kept_idx if cand_origin.get(i) == "ring"]
  far_kept = [leaf_ids[i] for i in kept_idx if cand_origin.get(i) == "far"]

  _assign_category(categories, base_thin_drop, "base_thin_drop")
  _assign_category(categories, ring_thin_drop, "ring_thin_drop")
  _assign_category(categories, far_thin_drop, "far_thin_drop")
  _assign_category(categories, base_none_drop, "base_none_drop")
  _assign_category(categories, ring_none_drop, "ring_none_drop")
  _assign_category(categories, far_none_drop, "far_none_drop")
  _assign_category(categories, base_kept, "base_kept")
  _assign_category(categories, ring_kept, "ring_kept")
  _assign_category(categories, far_kept, "far_kept")

  # Build plot.
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    raise RuntimeError("matplotlib is required for this script.") from exc

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")
  ax.set_title(f"Scene {scene_index} fruit {fruit_id}")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  # Draw stems as axes.
  for stem in stem_nodes:
    obb = stem.get("obb", {})
    center = np.asarray(obb.get("center", stem.get("centroid", [0.0, 0.0, 0.0])), dtype=float)
    axes = np.asarray(obb.get("axes", np.eye(3)), dtype=float)
    extents = np.asarray(obb.get("extents", [0.0, 0.0, 0.0]), dtype=float)
    _draw_stem_axes(ax, center, axes, extents)

  # Draw fruit as OBB.
  fruit_node = fruit_nodes[fruit_idx_local]
  fruit_obb = fruit_node.get("obb", {})
  fruit_center = np.asarray(fruit_obb.get("center", fruit_node.get("centroid", [0.0, 0.0, 0.0])), dtype=float)
  fruit_axes = np.asarray(fruit_obb.get("axes", np.eye(3)), dtype=float)
  fruit_extents = np.asarray(fruit_obb.get("extents", [0.0, 0.0, 0.0]), dtype=float)
  _draw_obb(ax, fruit_center, fruit_axes, fruit_extents, color="tab:orange")
  ax.scatter([fruit_center[0]], [fruit_center[1]], [fruit_center[2]], color="tab:orange", s=40)

  # Draw candidate leaves by category.
  for label, style in CATEGORY_STYLES.items():
    ids = [leaf_id for leaf_id, cat in categories.items() if cat == label]
    if not ids:
      continue
    idxs = [leaf_ids.index(leaf_id) for leaf_id in ids]
    pts = leaf_pos[idxs]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=style["size"], color=style["color"], alpha=style["alpha"], label=label)

  # Draw distance spheres.
  for radius, color in [(max_dist, "#999999"), (float(cand_cfg.ring_max_dist), "#bbbbbb")]:
    xs, ys, zs = _sphere_mesh(radius, sphere_res)
    ax.plot_surface(
      xs + fruit_center[0],
      ys + fruit_center[1],
      zs + fruit_center[2],
      color=color,
      alpha=0.08,
      linewidth=0,
    )

  ax.legend(loc="upper right", fontsize=7)

  # Equalize axes.
  pts = np.asarray([n.get("centroid") for n in nodes if n.get("centroid") is not None], dtype=float)
  if pts.size > 0:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges)) if float(np.max(ranges)) > 0 else 1.0
    mids = 0.5 * (mins + maxs)
    ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
    ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
    ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)

  out_path = out_dir / f"occlusion_candidates_scene{scene_index}_fruit{fruit_id}.png"
  fig.savefig(out_path, dpi=200)
  plt.close(fig)
  print(f"Saved candidate visualization to {out_path}")

  # Write per-fruit stats alongside the visualization.
  stats_path = out_dir / f"occlusion_candidates_scene{scene_index}_fruit{fruit_id}.json"
  stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
  print(f"Saved candidate stats to {stats_path}")


def main() -> None:
  """Entry point for candidate visualization script."""
  args = parse_args()
  cfg = Config()
  graphs = load_graphs(args.dataset_path)
  start = int(args.scene_offset)
  end = min(start + int(args.num_scenes), len(graphs))
  if start >= len(graphs):
    raise IndexError("Scene offset exceeds dataset length.")

  out_dir = Path(args.dataset_path).resolve().parent / "debug"
  out_dir.mkdir(parents=True, exist_ok=True)

  for idx in range(start, end):
    rng = np.random.default_rng(args.seed + idx)
    visualize_scene(graphs[idx], cfg, idx, rng, out_dir, args.sphere_res)


if __name__ == "__main__":
  main()
