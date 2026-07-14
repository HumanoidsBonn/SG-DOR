# pepper_scene_graphs/data/build_physical_adjacency.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class AdjacencyResult:
  edge_index: np.ndarray  # [2, E] int64
  edge_min_dist: np.ndarray  # [E] float32
  contact_count: np.ndarray  # [E] int32


def _voxel_key(xyz: np.ndarray, voxel: float) -> np.ndarray:
  return np.floor(xyz / voxel).astype(np.int32)


def build_instance_adjacency_from_voxels(
  xyz: np.ndarray,
  instance_id: np.ndarray,
  semantic_id: np.ndarray | None = None,
  *,
  eps: float = 0.003,
  voxel: float = 0.002,
  max_pairs_per_voxel: int = 2000
) -> AdjacencyResult:
  """
  Instance-level physical adjacency using voxel co-occupancy and neighbor voxels.

  Inputs:
    xyz: [N, 3] float32/float64 world coordinates
    instance_id: [N] int
    semantic_id: optional [N] int (unused here but kept for convenience)
  Parameters:
    eps: contact threshold in meters (min dist < eps => edge)
    voxel: voxel size in meters; should be <= eps
    max_pairs_per_voxel: safety cap to avoid pathological dense voxels
  Returns:
    edge_index: [2, E] undirected edges represented as both directions handled outside if needed
    edge_min_dist: [E]
    contact_count: [E] how many voxel-level contacts supported the edge (rough proxy)
  """
  xyz = np.asarray(xyz)
  inst = np.asarray(instance_id)

  if xyz.ndim != 2 or xyz.shape[1] != 3:
    raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
  if inst.ndim != 1 or inst.shape[0] != xyz.shape[0]:
    raise ValueError("instance_id must be [N] matching xyz")

  keys = _voxel_key(xyz, voxel)
  # Pack voxel keys to tuples for dict
  vox_tuples = [tuple(k) for k in keys]

  # Map voxel -> list of point indices (or instance ids directly)
  vox_map: Dict[Tuple[int, int, int], List[int]] = {}
  for i, v in enumerate(vox_tuples):
    if v not in vox_map:
      vox_map[v] = []
    vox_map[v].append(i)

  # Neighbor offsets (26-neighborhood + self)
  offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

  # Accumulate pair stats at instance-pair level
  pair_min_dist: Dict[Tuple[int, int], float] = {}
  pair_count: Dict[Tuple[int, int], int] = {}

  # Iterate voxels and their neighbors
  for v, idxs in vox_map.items():
    pts_i = idxs
    inst_i = inst[pts_i]

    # Skip empty or single-instance-only voxels early
    if len(pts_i) == 0:
      continue

    for dx, dy, dz in offs:
      vn = (v[0] + dx, v[1] + dy, v[2] + dz)
      if vn not in vox_map:
        continue

      pts_j = vox_map[vn]
      if len(pts_j) == 0:
        continue

      # Safety cap (pathological density)
      if len(pts_i) * len(pts_j) > max_pairs_per_voxel:
        continue

      inst_j = inst[pts_j]

      # Compute distances between points in voxel v and neighbor voxel vn
      # We keep it brute-force within voxel neighborhood; voxel size keeps it small.
      xi = xyz[pts_i]  # [Mi,3]
      xj = xyz[pts_j]  # [Mj,3]

      # (Mi, Mj) squared distances
      d2 = np.sum((xi[:, None, :] - xj[None, :, :]) ** 2, axis=-1)
      # Candidate contacts below eps
      mask = d2 < (eps * eps)
      if not np.any(mask):
        continue

      ii, jj = np.where(mask)
      for a, b in zip(ii.tolist(), jj.tolist()):
        ia = int(inst_i[a])
        ib = int(inst_j[b])
        if ia == ib:
          continue
        # Canonical ordering for undirected pair
        p = (ia, ib) if ia < ib else (ib, ia)

        dist = float(np.sqrt(d2[a, b]))
        if p not in pair_min_dist:
          pair_min_dist[p] = dist
          pair_count[p] = 1
        else:
          if dist < pair_min_dist[p]:
            pair_min_dist[p] = dist
          pair_count[p] += 1

  # Build edge list
  pairs = list(pair_min_dist.keys())
  if len(pairs) == 0:
    return AdjacencyResult(
      edge_index=np.zeros((2, 0), dtype=np.int64),
      edge_min_dist=np.zeros((0,), dtype=np.float32),
      contact_count=np.zeros((0,), dtype=np.int32)
    )

  src = np.array([p[0] for p in pairs], dtype=np.int64)
  dst = np.array([p[1] for p in pairs], dtype=np.int64)
  edge_index = np.stack([src, dst], axis=0)  # [2,E]

  edge_min_dist = np.array([pair_min_dist[p] for p in pairs], dtype=np.float32)
  contact_count = np.array([pair_count[p] for p in pairs], dtype=np.int32)

  return AdjacencyResult(edge_index=edge_index, edge_min_dist=edge_min_dist, contact_count=contact_count)
