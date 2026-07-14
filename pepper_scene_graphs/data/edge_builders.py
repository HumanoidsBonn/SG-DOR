# pepper_scene_graphs/data/edge_builders.py

import torch
from torch_geometric.nn import knn_graph, radius_graph

from pepper_scene_graphs.constants import NODE_CLASSES


from typing import Any, List, Set, Tuple


def build_allowed_type_matrix(cfg: Any, node_classes: List[str]) -> torch.Tensor:
  """Build a boolean matrix defining allowed edge connections based on node types.

  Args:
      cfg: Configuration object containing allowed_pairs.
      node_classes: List of node class names.

  Returns:
      A boolean tensor of shape (num_classes, num_classes).
  """
  mat = torch.zeros((len(node_classes), len(node_classes)), dtype=torch.bool)
  for a, b in cfg.allowed_pairs:
    i = node_classes.index(a)
    j = node_classes.index(b)
    mat[i, j] = True
    mat[j, i] = True
  return mat


def scheduled_tau(type_cfg: Any, epoch: int) -> float:
  """Compute the scheduled tau threshold for the current epoch.

  Args:
      type_cfg: Type constraint configuration.
      epoch: Current training epoch.

  Returns:
      The tau value to use.
  """
  if not type_cfg.tau_schedule_enabled:
    return float(type_cfg.tau)
  values = type_cfg.tau_values
  if not values:
    return float(type_cfg.tau)
  interval = max(1, int(type_cfg.tau_interval_epochs))
  idx = min(len(values) - 1, max(0, (epoch - 1) // interval))
  return float(values[idx])


def build_candidate_edges(
  data: Any,
  k: int,
  extra_stems_per_node: int,
  dist_threshold: float,
  radius_max_neighbors: int,
) -> torch.Tensor:
  """Build candidate edges with Oracle-parity logic: radius ∪ kNN ∪ extra stems.

  Args:
      data: PyG data object containing `pos`, `batch`, `y_node`.
      k: Number of nearest neighbors.
      extra_stems_per_node: Number of stems to connect to every non-stem node.
      dist_threshold: Radius threshold for radius graph candidate edges.
      radius_max_neighbors: Max neighbors per node in radius graph.

  Returns:
      Directed edge index tensor of shape (2, num_edges), deduplicated.
  """
  pos, batch = data.pos, data.batch

  # 1) Base candidates: radius + kNN (same spirit as Oracle)
  e_radius = radius_graph(
    pos,
    r=float(dist_threshold),
    batch=batch,
    loop=False,
    max_num_neighbors=int(radius_max_neighbors),
  )
  e_knn = knn_graph(pos, k=k, batch=batch, loop=False)
  all_edges = [e_radius, e_knn]

  # 2) Extra stem candidates (XY-only nearest stems per non-stem)
  if extra_stems_per_node > 0:
    stem_idx = NODE_CLASSES.index("stem")
    is_stem = data.y_node == stem_idx
    is_non_stem = ~is_stem

    if is_stem.any() and is_non_stem.any():
      pos_xy = pos[:, :2]
      for b in torch.unique(batch):
        b_mask = batch == b
        b_stems = b_mask & is_stem
        b_non_stems = b_mask & is_non_stem

        if not b_stems.any() or not b_non_stems.any():
          continue

        s_idx = torch.nonzero(b_stems, as_tuple=False).squeeze(-1)
        n_idx = torch.nonzero(b_non_stems, as_tuple=False).squeeze(-1)
        dists = torch.cdist(pos_xy[n_idx], pos_xy[s_idx])
        k_use = min(int(extra_stems_per_node), s_idx.size(0))
        nn_topk = torch.topk(dists, k_use, dim=1, largest=False).indices

        src_nodes = n_idx.repeat_interleave(k_use)
        dst_nodes = s_idx[nn_topk.flatten()]
        all_edges.append(torch.stack([src_nodes, dst_nodes], dim=0))

  # 3) Canonical undirected dedupe, then restore both directions
  combined = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long, device=pos.device)
  if combined.numel() == 0:
    return torch.empty((2, 0), dtype=torch.long, device=pos.device)

  combined_sorted = torch.sort(combined, dim=0).values
  unique_undir = torch.unique(combined_sorted, dim=1)
  rev = torch.stack([unique_undir[1], unique_undir[0]], dim=0)
  final = torch.cat([unique_undir, rev], dim=1)
  return torch.unique(final, dim=1)


def undirected_unique(edge_index: torch.Tensor) -> torch.Tensor:
  """Convert directed edge index to unique undirected pairs (canonical ordering).

  Args:
      edge_index: Edge index tensor of shape (2, num_edges).

  Returns:
      Unique undirected pairs of shape (2, num_unique_edges), where src < dst.
  """
  if edge_index.numel() == 0:
    return edge_index
  src, dst = edge_index
  mask = src < dst
  src = src[mask]
  dst = dst[mask]
  if src.numel() == 0:
    return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
  pairs = torch.stack([src, dst], dim=1)
  pairs = torch.unique(pairs, dim=0)
  return pairs.t().contiguous()


def stem_aware_edge_distance(
  pos: torch.Tensor,
  edge_index: torch.Tensor,
  node_logits: torch.Tensor,
  y_node: torch.Tensor,
  mode: str = "pred",
) -> torch.Tensor:
  """Compute edge distances, using 2D distance for edges connected to stems.

  Args:
      pos: Node positions of shape (num_nodes, 3).
      edge_index: Edge indices of shape (2, num_edges).
      node_logits: Predicted node logits (used if mode="pred").
      y_node: Ground truth node labels (used if mode="gt").
      mode: 'pred' or 'gt' to determine stem identity.

  Returns:
      Tensor of distances of shape (num_edges,).
  """
  if edge_index.numel() == 0:
    return pos.new_empty((0,))
  src = edge_index[0]
  dst = edge_index[1]
  d = pos[src] - pos[dst]
  dist_xyz = torch.norm(d, dim=-1)
  dist_xy = torch.norm(d[:, :2], dim=-1)
  stem_idx = NODE_CLASSES.index("stem")

  if str(mode).lower() == "gt":
    is_stem_edge = (y_node[src] == stem_idx) | (y_node[dst] == stem_idx)
  else:
    pred_src = node_logits[src].argmax(dim=-1)
    pred_dst = node_logits[dst].argmax(dim=-1)
    is_stem_edge = (pred_src == stem_idx) | (pred_dst == stem_idx)

  return torch.where(is_stem_edge, dist_xy, dist_xyz)
