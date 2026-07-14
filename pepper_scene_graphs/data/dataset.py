# pepper_scene_graphs/data/dataset.py

try:
  import ujson as json
except ImportError:
  import json

from pathlib import Path
import re
import numpy as np
import torch
from torch_geometric.data import Data

from typing import Any, Dict, List, Union, Tuple

from pepper_scene_graphs.constants import NODE_CLASSES, EDGE_LABELS
from pepper_scene_graphs.data.transforms import node_feature_vector


def _stem_group_key(node_id: str) -> str:
  """Return a stem-level key by stripping a trailing segment suffix."""
  return re.sub(r"_seg\d+$", "", str(node_id))


class SceneGraphData(Data):
  """Custom PyG Data object for scene graphs."""

  def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
    if key in {"edge_index", "gt_edge_pairs", "occl_pairs", "occl_base_pairs", "occl_ring_pairs", "occl_far_pairs"}:
      return self.num_nodes
    return super().__inc__(key, value, *args, **kwargs)


def build_graph_data(graph: Dict[str, Any], cfg: Any) -> SceneGraphData:
  """Convert a raw graph dictionary into a PyG SceneGraphData object.

  Args:
      graph: Raw dictionary containing 'nodes' and 'edges'.
      cfg: Configuration object for feature extraction.

  Returns:
      A SceneGraphData object ready for PyG loaders.
  """
  nodes = graph["nodes"]
  node_indices = {node["id"]: idx for idx, node in enumerate(nodes)}

  feats = []
  pos = []
  y = []
  stem_group_map: Dict[str, int] = {}
  stem_group_ids: List[int] = []
  for node in nodes:
    f, p = node_feature_vector(node, cfg)
    feats.append(f)
    pos.append(p)
    y.append(NODE_CLASSES.index(node["type"]))
    if node["type"] == "stem":
      key = _stem_group_key(node["id"])
      if key not in stem_group_map:
        stem_group_map[key] = len(stem_group_map)
      stem_group_ids.append(stem_group_map[key])
    else:
      stem_group_ids.append(-1)

  edges = graph.get("edges", [])
  pairs = []
  pair_labels = []
  edge_index_list = []
  for e in edges:
    u = node_indices[e["source"]]
    v = node_indices[e["target"]]
    if u == v:
      continue
    lab = EDGE_LABELS.index(e["label"])
    pairs.append([min(u, v), max(u, v)])
    pair_labels.append(lab)
    edge_index_list.append([u, v])
    edge_index_list.append([v, u])

  if edge_index_list:
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
  else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

  data = SceneGraphData(
    x=torch.tensor(np.stack(feats), dtype=torch.float32),
    pos=torch.tensor(np.stack(pos), dtype=torch.float32),
    y_node=torch.tensor(y, dtype=torch.long),
    edge_index=edge_index,
  )
  data.node_stem_group_id = torch.tensor(stem_group_ids, dtype=torch.long)

  if pairs:
    data.gt_edge_pairs = torch.tensor(pairs, dtype=torch.long)
    data.gt_edge_labels = torch.tensor(pair_labels, dtype=torch.long)
  else:
    data.gt_edge_pairs = torch.empty((0, 2), dtype=torch.long)
    data.gt_edge_labels = torch.empty((0,), dtype=torch.long)

  # Optional occlusion block (leaf-fruit overlaps)
  occl = graph.get("occlusion")
  if occl and occl.get("edges"):
    occl_pairs = []
    occl_raw = []
    occl_weighted = []
    for e in occl.get("edges", []):
      leaf_id = e.get("leaf_id")
      fruit_id = e.get("fruit_id")
      if leaf_id not in node_indices or fruit_id not in node_indices:
        continue
      occl_pairs.append([node_indices[leaf_id], node_indices[fruit_id]])
      occl_raw.append(e.get("raw_overlap", []))
      occl_weighted.append(e.get("weighted_overlap", []))
    if occl_pairs:
      data.occl_pairs = torch.tensor(occl_pairs, dtype=torch.long)
      data.occl_raw = torch.tensor(np.asarray(occl_raw, dtype=np.float32), dtype=torch.float32)
      data.occl_weighted = torch.tensor(np.asarray(occl_weighted, dtype=np.float32), dtype=torch.float32)
    else:
      data.occl_pairs = torch.empty((0, 2), dtype=torch.long)
      data.occl_raw = torch.empty((0, 0), dtype=torch.float32)
      data.occl_weighted = torch.empty((0, 0), dtype=torch.float32)
  else:
    data.occl_pairs = torch.empty((0, 2), dtype=torch.long)
    data.occl_raw = torch.empty((0, 0), dtype=torch.float32)
    data.occl_weighted = torch.empty((0, 0), dtype=torch.float32)

  # Optional occlusion candidate pools (precompute leaf-fruit distance buckets).
  if cfg.occlusion.enabled:
    leaf_idx = (data.y_node == NODE_CLASSES.index("leaf")).nonzero(as_tuple=False).flatten()
    fruit_idx = (data.y_node == NODE_CLASSES.index("fruit")).nonzero(as_tuple=False).flatten()
    max_dist = float(cfg.occlusion.candidates.max_dist)
    ring_max = float(cfg.occlusion.candidates.ring_max_dist)
    occl_map_raw: Dict[Tuple[int, int], List[float]] = {}
    occl_map_weighted: Dict[Tuple[int, int], List[float]] = {}
    if data.occl_pairs.numel() > 0:
      for pair, raw_vals, weighted_vals in zip(
        data.occl_pairs.tolist(),
        data.occl_raw.tolist(),
        data.occl_weighted.tolist(),
      ):
        key = (int(pair[0]), int(pair[1]))
        occl_map_raw[key] = raw_vals
        occl_map_weighted[key] = weighted_vals
    zero_vals = [0.0] * 18

    base_pairs: List[List[int]] = []
    ring_pairs: List[List[int]] = []
    far_pairs: List[List[int]] = []
    base_raw: List[List[float]] = []
    ring_raw: List[List[float]] = []
    far_raw: List[List[float]] = []
    base_weighted: List[List[float]] = []
    ring_weighted: List[List[float]] = []
    far_weighted: List[List[float]] = []
    if leaf_idx.numel() > 0 and fruit_idx.numel() > 0:
      dists = torch.cdist(data.pos[leaf_idx], data.pos[fruit_idx])
      for j, f_idx in enumerate(fruit_idx.tolist()):
        dist_j = dists[:, j]
        base_mask = dist_j <= max_dist
        ring_mask = (dist_j > max_dist) & (dist_j <= ring_max)
        far_mask = dist_j > ring_max
        for l_idx in leaf_idx[base_mask].tolist():
          key = (int(l_idx), int(f_idx))
          base_pairs.append([l_idx, f_idx])
          base_raw.append(occl_map_raw.get(key, zero_vals))
          base_weighted.append(occl_map_weighted.get(key, zero_vals))
        for l_idx in leaf_idx[ring_mask].tolist():
          key = (int(l_idx), int(f_idx))
          ring_pairs.append([l_idx, f_idx])
          ring_raw.append(occl_map_raw.get(key, zero_vals))
          ring_weighted.append(occl_map_weighted.get(key, zero_vals))
        for l_idx in leaf_idx[far_mask].tolist():
          key = (int(l_idx), int(f_idx))
          far_pairs.append([l_idx, f_idx])
          far_raw.append(occl_map_raw.get(key, zero_vals))
          far_weighted.append(occl_map_weighted.get(key, zero_vals))
    data.occl_base_pairs = torch.tensor(base_pairs, dtype=torch.long) if base_pairs else torch.empty((0, 2), dtype=torch.long)
    data.occl_ring_pairs = torch.tensor(ring_pairs, dtype=torch.long) if ring_pairs else torch.empty((0, 2), dtype=torch.long)
    data.occl_far_pairs = torch.tensor(far_pairs, dtype=torch.long) if far_pairs else torch.empty((0, 2), dtype=torch.long)
    data.occl_base_raw = torch.tensor(np.asarray(base_raw, dtype=np.float32), dtype=torch.float32) if base_raw else torch.empty((0, 18), dtype=torch.float32)
    data.occl_ring_raw = torch.tensor(np.asarray(ring_raw, dtype=np.float32), dtype=torch.float32) if ring_raw else torch.empty((0, 18), dtype=torch.float32)
    data.occl_far_raw = torch.tensor(np.asarray(far_raw, dtype=np.float32), dtype=torch.float32) if far_raw else torch.empty((0, 18), dtype=torch.float32)
    data.occl_base_weighted = torch.tensor(np.asarray(base_weighted, dtype=np.float32), dtype=torch.float32) if base_weighted else torch.empty((0, 18), dtype=torch.float32)
    data.occl_ring_weighted = torch.tensor(np.asarray(ring_weighted, dtype=np.float32), dtype=torch.float32) if ring_weighted else torch.empty((0, 18), dtype=torch.float32)
    data.occl_far_weighted = torch.tensor(np.asarray(far_weighted, dtype=np.float32), dtype=torch.float32) if far_weighted else torch.empty((0, 18), dtype=torch.float32)
    if data.occl_base_pairs.numel() > 0:
      order = data.occl_base_pairs[:, 1].argsort()
      data.occl_base_pairs = data.occl_base_pairs[order]
      data.occl_base_raw = data.occl_base_raw[order]
      data.occl_base_weighted = data.occl_base_weighted[order]
    if data.occl_ring_pairs.numel() > 0:
      order = data.occl_ring_pairs[:, 1].argsort()
      data.occl_ring_pairs = data.occl_ring_pairs[order]
      data.occl_ring_raw = data.occl_ring_raw[order]
      data.occl_ring_weighted = data.occl_ring_weighted[order]
    if data.occl_far_pairs.numel() > 0:
      order = data.occl_far_pairs[:, 1].argsort()
      data.occl_far_pairs = data.occl_far_pairs[order]
      data.occl_far_raw = data.occl_far_raw[order]
      data.occl_far_weighted = data.occl_far_weighted[order]
    data.occl_cand_cfg = torch.tensor([max_dist, ring_max], dtype=torch.float32)

  return data


def load_graph_dataset(path: str, cfg: Any) -> List[SceneGraphData]:
  """Load a dataset from a file path (.pt or .json).
  
  Automatically manages a .pt binary cache for .json files to speed up subsequent loads.

  Args:
      path: Path to the dataset file (.json or .pt).
      cfg: Configuration object.

  Returns:
      A list of SceneGraphData objects.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(p)

  # 1. Directly handle .pt files
  if p.suffix == ".pt":
    data_list = torch.load(p)
    if not isinstance(data_list, list):
      raise ValueError("Expected list of Data in .pt")
    return data_list

  # 2. Handle .json files with binary caching
  if p.suffix == ".json":
    cache_path = p.with_suffix(".pt")
    
    # Load from cache if it exists and is newer than the source JSON
    if cache_path.exists() and cache_path.stat().st_mtime > p.stat().st_mtime:
      print(f"Loading dataset from binary cache: {cache_path}")
      try:
        data_list = torch.load(cache_path)
        if isinstance(data_list, list):
          need_stem_equiv = bool(getattr(getattr(cfg, "edge_exist", None), "stem_segment_equiv", False))
          if need_stem_equiv and len(data_list) > 0 and not hasattr(data_list[0], "node_stem_group_id"):
            print("Binary cache missing node_stem_group_id; rebuilding from JSON...")
          else:
            return data_list
      except Exception:
        print("Warning: Failed to load cache, falling back to JSON.")

    print(f"Loading dataset from JSON: {p} (this may take a while)...")
    with open(p, "r", encoding="utf-8") as f:
      graphs = json.load(f)
    
    data_list = [build_graph_data(g, cfg) for g in graphs]
    
    # Save to cache for next time
    try:
      print(f"Saving binary cache to: {cache_path}")
      torch.save(data_list, cache_path)
    except Exception as e:
      print(f"Warning: Failed to save cache: {e}")
      
    return data_list

  raise ValueError(f"Unsupported dataset format: {p.suffix}")

  raise ValueError(f"Unsupported dataset format: {p.suffix}")


def load_raw_graphs(path: str) -> List[Dict[str, Any]]:
  """Load raw scene graph dicts (only supports JSON).

  Args:
      path: Path to the JSON file.

  Returns:
      A list of raw graph dictionaries.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(p)
  if p.suffix != ".json":
    raise ValueError("Raw graph stats currently only support JSON datasets")
  with p.open("r", encoding="utf-8") as f:
    return json.load(f)
