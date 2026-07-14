"""Validate occlusion candidate cache consistency and alignment.

Checks cached base/ring/far pools against recomputed distances and validates
that cached occlusion values match the occlusion edges stored in the dataset.
Writes a JSON report next to the dataset under the debug/ folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))

from pepper_scene_graphs.configs.config import Config
from pepper_scene_graphs.data.dataset import load_graph_dataset
from pepper_scene_graphs.constants import NODE_CLASSES


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Check occlusion candidate cache consistency.")
  parser.add_argument("--dataset-path", required=True, help="Path to dataset JSON.")
  parser.add_argument("--num-scenes", type=int, default=5, help="Number of scenes to check.")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
  parser.add_argument("--pair-samples", type=int, default=500, help="Number of cached pairs to validate.")
  return parser.parse_args()


def _map_occl_edges(
  graph: Dict[str, Any]
) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[Tuple[int, int], List[float]]]:
  raw = {}
  weighted = {}
  occl = graph.get("occlusion", {})
  for e in occl.get("edges", []):
    leaf_id = e.get("leaf_id")
    fruit_id = e.get("fruit_id")
    if leaf_id is None or fruit_id is None:
      continue
    key = (leaf_id, fruit_id)
    raw[key] = e.get("raw_overlap", [])
    weighted[key] = e.get("weighted_overlap", [])
  return raw, weighted


def _node_id_map(graph: Dict[str, Any]) -> Dict[Any, int]:
  return {n["id"]: idx for idx, n in enumerate(graph["nodes"])}


def _collect_pairs(graph: Dict[str, Any], max_dist: float, ring_max: float) -> Dict[str, Any]:
  node_ids = _node_id_map(graph)
  nodes = graph["nodes"]
  leaf_idx = [i for i, n in enumerate(nodes) if n["type"] == "leaf"]
  fruit_idx = [i for i, n in enumerate(nodes) if n["type"] == "fruit"]
  if not leaf_idx or not fruit_idx:
    return {
      "base_pairs": [],
      "ring_pairs": [],
      "far_pairs": [],
    }
  leaf_pos = torch.tensor([nodes[i]["centroid"] for i in leaf_idx], dtype=torch.float32)
  fruit_pos = torch.tensor([nodes[i]["centroid"] for i in fruit_idx], dtype=torch.float32)
  dists = torch.cdist(leaf_pos, fruit_pos)
  base_pairs: List[Tuple[int, int]] = []
  ring_pairs: List[Tuple[int, int]] = []
  far_pairs: List[Tuple[int, int]] = []
  for j, f in enumerate(fruit_idx):
    dist_j = dists[:, j]
    base_mask = dist_j <= max_dist
    ring_mask = (dist_j > max_dist) & (dist_j <= ring_max)
    far_mask = dist_j > ring_max
    for l in np.asarray(leaf_idx)[base_mask.cpu().numpy()]:
      base_pairs.append((l, f))
    for l in np.asarray(leaf_idx)[ring_mask.cpu().numpy()]:
      ring_pairs.append((l, f))
    for l in np.asarray(leaf_idx)[far_mask.cpu().numpy()]:
      far_pairs.append((l, f))
  return {
    "base_pairs": base_pairs,
    "ring_pairs": ring_pairs,
    "far_pairs": far_pairs,
  }


def _check_scene(
  graph: Dict[str, Any],
  data: Any,
  cfg: Config,
  rng: np.random.Generator,
  pair_samples: int,
) -> Dict[str, Any]:
  max_dist = float(cfg.occlusion.candidates.max_dist)
  ring_max = float(cfg.occlusion.candidates.ring_max_dist)
  raw_map, weighted_map = _map_occl_edges(graph)
  recomputed = _collect_pairs(graph, max_dist, ring_max)

  cached = {
    "base_pairs": data.occl_base_pairs.cpu().tolist() if hasattr(data, "occl_base_pairs") else [],
    "ring_pairs": data.occl_ring_pairs.cpu().tolist() if hasattr(data, "occl_ring_pairs") else [],
    "far_pairs": data.occl_far_pairs.cpu().tolist() if hasattr(data, "occl_far_pairs") else [],
  }

  pair_stats = {}
  for key in ("base_pairs", "ring_pairs", "far_pairs"):
    cached_set = set(tuple(p) for p in cached[key])
    recomputed_set = set(recomputed[key])
    pair_stats[key] = {
      "cached_count": len(cached_set),
      "recomputed_count": len(recomputed_set),
      "missing_in_cache": len(recomputed_set - cached_set),
      "extra_in_cache": len(cached_set - recomputed_set),
    }

  # Validate cached values against data.occl_raw/weighted (index-aligned).
  sample_results = []
  if hasattr(data, "occl_pairs") and data.occl_pairs.numel() > 0:
    pair_to_raw = {}
    pair_to_weighted = {}
    for pair, raw_vals, weighted_vals in zip(
      data.occl_pairs.tolist(),
      data.occl_raw.tolist(),
      data.occl_weighted.tolist(),
    ):
      key = (int(pair[0]), int(pair[1]))
      pair_to_raw[key] = raw_vals
      pair_to_weighted[key] = weighted_vals
  else:
    pair_to_raw = {}
    pair_to_weighted = {}

  all_pairs = cached["base_pairs"] + cached["ring_pairs"] + cached["far_pairs"]
  if all_pairs:
    sample_n = min(pair_samples, len(all_pairs))
    sample_idx = rng.choice(len(all_pairs), size=sample_n, replace=False)
    for idx in sample_idx:
      l_idx, f_idx = all_pairs[idx]
      key = (int(l_idx), int(f_idx))
      raw_gt = pair_to_raw.get(key, [0.0] * 18)
      weighted_gt = pair_to_weighted.get(key, [0.0] * 18)
      raw_cached = None
      weighted_cached = None
      if hasattr(data, "occl_base_raw"):
        def _lookup(pool_pairs, pool_raw, pool_weighted):
          if pool_pairs.numel() == 0:
            return None, None
          matches = (pool_pairs[:, 0] == l_idx) & (pool_pairs[:, 1] == f_idx)
          if matches.any():
            idx0 = int(matches.nonzero(as_tuple=False)[0].item())
            return pool_raw[idx0].tolist(), pool_weighted[idx0].tolist()
          return None, None

        raw_cached, weighted_cached = _lookup(data.occl_base_pairs, data.occl_base_raw, data.occl_base_weighted)
        if raw_cached is None:
          raw_cached, weighted_cached = _lookup(data.occl_ring_pairs, data.occl_ring_raw, data.occl_ring_weighted)
        if raw_cached is None:
          raw_cached, weighted_cached = _lookup(data.occl_far_pairs, data.occl_far_raw, data.occl_far_weighted)
      sample_results.append(
        {
          "leaf_idx": int(l_idx),
          "fruit_idx": int(f_idx),
          "raw_match": raw_cached == raw_gt,
          "weighted_match": weighted_cached == weighted_gt,
        }
      )

  return {
    "pair_stats": pair_stats,
    "value_sample_checks": sample_results,
  }


def main() -> None:
  args = _parse_args()
  cfg = Config()
  dataset_path = Path(args.dataset_path)
  out_dir = dataset_path.parent / "debug"
  out_dir.mkdir(parents=True, exist_ok=True)

  with dataset_path.open("r", encoding="utf-8") as f:
    graphs = json.load(f)
  data_list = load_graph_dataset(str(dataset_path), cfg)

  rng = np.random.default_rng(args.seed)
  num_scenes = min(args.num_scenes, len(graphs))
  scene_indices = list(range(num_scenes))

  report = {
    "dataset_path": str(dataset_path),
    "num_scenes_checked": num_scenes,
    "max_dist": float(cfg.occlusion.candidates.max_dist),
    "ring_max_dist": float(cfg.occlusion.candidates.ring_max_dist),
    "scenes": {},
  }

  for idx in scene_indices:
    report["scenes"][str(idx)] = _check_scene(
      graphs[idx], data_list[idx], cfg, rng, args.pair_samples
    )

  out_path = out_dir / "occlusion_cache_check.json"
  with out_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  print(f"Saved occlusion cache check to {out_path}")


if __name__ == "__main__":
  main()
