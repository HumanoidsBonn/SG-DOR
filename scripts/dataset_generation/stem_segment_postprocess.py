#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for stem segmentation."""
  parser = argparse.ArgumentParser(description="Segment stems in a scene-graph JSON dataset")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
  parser.add_argument("--out", default=None, help="Output JSON path (default: *_segmented.json)")
  parser.add_argument("--num-segments", type=int, required=True, help="Number of stem segments per stem")
  parser.add_argument("--cutoff-height", type=float, required=True, help="Total stem height to keep from base")
  parser.add_argument("--base-z", type=float, required=True, help="Fixed stem base z coordinate")
  return parser.parse_args()


def summary_stats(values: Sequence[float]) -> Dict[str, float]:
  """Compute summary statistics for a list of numeric values."""
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return {
      "count": 0,
      "mean": float("nan"),
      "median": float("nan"),
      "p90": float("nan"),
      "p95": float("nan"),
      "min": float("nan"),
      "max": float("nan"),
    }
  return {
    "count": int(arr.size),
    "mean": float(arr.mean()),
    "median": float(np.median(arr)),
    "p90": float(np.percentile(arr, 90)),
    "p95": float(np.percentile(arr, 95)),
    "min": float(arr.min()),
    "max": float(arr.max()),
  }


def build_segment_node(
  stem_node: Dict[str, Any],
  seg_idx: int,
  seg_center: np.ndarray,
  axes: np.ndarray,
  seg_extents: np.ndarray,
  orientation: np.ndarray,
) -> Dict[str, Any]:
  """Create a segmented stem node entry."""
  seg_id = f"{stem_node['id']}_seg{seg_idx}"
  color = stem_node.get("color", [0.8, 0.8, 0.8])
  return {
    "id": seg_id,
    "type": stem_node["type"],
    "prototype": stem_node.get("prototype"),
    "obb": {
      "center": seg_center.tolist(),
      "axes": axes.tolist(),
      "extents": seg_extents.tolist(),
    },
    "centroid": seg_center.tolist(),
    "orientation": orientation.tolist(),
    "color": color,
  }


def remap_stem_edge(stem_id: str, nonstem_centroid: np.ndarray, seg_centers: List[np.ndarray]) -> str:
  """Assign a non-stem to the closest stem segment (by |Δz|)."""
  if not seg_centers:
    return stem_id
  centers = np.asarray(seg_centers, dtype=np.float32)
  d = np.abs(centers[:, 2] - float(nonstem_centroid[2]))
  seg_idx = int(np.argmin(d))
  return f"{stem_id}_seg{seg_idx}"


def main() -> None:
  """Entry point: load dataset, segment stems, remap edges, save output + sanity stats."""
  args = parse_args()
  dataset_path = Path(args.dataset_path)
  if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)

  with open(dataset_path, "r", encoding="utf-8") as f:
    graphs = json.load(f)
  if not isinstance(graphs, list):
    raise ValueError("Expected a list of graphs in JSON.")

  num_segments = int(args.num_segments)
  if num_segments <= 0:
    raise ValueError("--num-segments must be > 0")
  cutoff_height = float(args.cutoff_height)
  if cutoff_height <= 0:
    raise ValueError("--cutoff-height must be > 0")
  base_z = float(args.base_z)

  seg_len = cutoff_height / float(num_segments)

  stats: Dict[str, Any] = {
    "scene_count": len(graphs),
    "num_segments": num_segments,
    "cutoff_height": cutoff_height,
    "base_z": base_z,
    "segment_length": seg_len,
    "stems_before": 0,
    "stems_after": 0,
    "nodes_before": 0,
    "nodes_after": 0,
    "edges_before": 0,
    "edges_after": 0,
    "segments_per_stem": [],
    "cut_heights": [],
    "stem_heights": [],
    "remapped_edges": 0,
    "missing_edge_refs": 0,
    "occlusion_missing_refs": 0,
  }

  out_graphs = []
  for graph in graphs:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    stats["nodes_before"] += len(nodes)
    stats["edges_before"] += len(edges)

    stem_nodes = [n for n in nodes if n.get("type") == "stem"]
    nonstem_nodes = [n for n in nodes if n.get("type") != "stem"]
    stats["stems_before"] += len(stem_nodes)

    # 1) Build new node list (non-stems preserved, stems segmented)
    new_nodes: List[Dict[str, Any]] = []
    stem_seg_centers: Dict[str, List[np.ndarray]] = {}
    new_nodes.extend(nonstem_nodes)

    for stem in stem_nodes:
      centroid = np.asarray(stem["centroid"], dtype=np.float32)
      base_xy = centroid[:2].astype(np.float32)

      # Use original stem axes/extents/orientation, but replace the main extent with segment length
      axes = np.asarray(stem["obb"]["axes"], dtype=np.float32)
      extents = np.asarray(stem["obb"]["extents"], dtype=np.float32)
      orientation = np.asarray(stem["orientation"], dtype=np.float32)
      main_idx = int(np.argmax(extents))
      seg_extents = extents.astype(np.float32).copy()
      seg_extents[main_idx] = float(seg_len)

      stats["stem_heights"].append(cutoff_height)
      stats["cut_heights"].append(cutoff_height)
      stats["segments_per_stem"].append(num_segments)

      seg_centers: List[np.ndarray] = []
      for i in range(num_segments):
        # center z = base_z + (i + 0.5) * seg_len
        center_z = base_z + (i + 0.5) * seg_len
        seg_center = np.array([base_xy[0], base_xy[1], center_z], dtype=np.float32)
        seg_centers.append(seg_center)
        new_nodes.append(build_segment_node(stem, i, seg_center, axes, seg_extents, orientation))
      stem_seg_centers[stem["id"]] = seg_centers

    stats["stems_after"] += len([n for n in new_nodes if n.get("type") == "stem"])

    # 2) Build ID map for sanity checks
    new_ids = {n["id"] for n in new_nodes}

    # 3) Remap edges to closest segment (by |Δz|)
    new_edges = []
    for e in edges:
      src = e["source"]
      dst = e["target"]
      if src in stem_seg_centers and dst not in stem_seg_centers:
        non = next(n for n in new_nodes if n["id"] == dst)
        new_src = remap_stem_edge(src, np.asarray(non["centroid"], dtype=np.float32), stem_seg_centers[src])
        new_edges.append({**e, "source": new_src, "target": dst})
        stats["remapped_edges"] += 1
      elif dst in stem_seg_centers and src not in stem_seg_centers:
        non = next(n for n in new_nodes if n["id"] == src)
        new_dst = remap_stem_edge(dst, np.asarray(non["centroid"], dtype=np.float32), stem_seg_centers[dst])
        new_edges.append({**e, "source": src, "target": new_dst})
        stats["remapped_edges"] += 1
      else:
        new_edges.append(e)

    # 4) Sanity: missing refs in edges
    for e in new_edges:
      if e["source"] not in new_ids or e["target"] not in new_ids:
        stats["missing_edge_refs"] += 1

    # 5) Occlusion check (leaf-fruit only): ensure ids exist
    occl = graph.get("occlusion")
    if occl:
      for oe in occl.get("edges", []):
        if oe.get("leaf_id") not in new_ids or oe.get("fruit_id") not in new_ids:
          stats["occlusion_missing_refs"] += 1

    stats["nodes_after"] += len(new_nodes)
    stats["edges_after"] += len(new_edges)
    out_graphs.append({**graph, "nodes": new_nodes, "edges": new_edges})

  stats["segments_per_stem_stats"] = summary_stats(stats["segments_per_stem"])
  stats["cut_height_stats"] = summary_stats(stats["cut_heights"])
  stats["stem_height_stats"] = summary_stats(stats["stem_heights"])

  out_path = Path(args.out) if args.out else dataset_path.with_name(dataset_path.stem + "_segmented.json")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_graphs, f, indent=2)

  stats_path = out_path.with_name(out_path.stem + "_sanity.json")
  with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
  print(f"Saved segmented dataset to {out_path}")
  print(f"Saved sanity stats to {stats_path}")


if __name__ == "__main__":
  main()
