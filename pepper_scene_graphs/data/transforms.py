# pepper_scene_graphs/data/transforms.py

import colorsys
import math
import random
import numpy as np
import torch

from typing import Any, Dict, List, Tuple

from pepper_scene_graphs.constants import NODE_CLASSES


def feature_slices(flags: Any) -> Dict[str, Tuple[int, int]]:
  """Compute feature vector slice indices based on enabled flags.

  Args:
      flags: Configuration object with feature flags (use_centroid, etc.).

  Returns:
      Dictionary mapping feature names to (start, end) index tuples.
  """
  offset = 0
  sl = {}
  if flags.use_centroid:
    sl["centroid"] = (offset, offset + 3)
    offset += 3
  if flags.use_axes:
    sl["axes"] = (offset, offset + 9)
    offset += 9
  if flags.use_extents:
    sl["extents"] = (offset, offset + 3)
    offset += 3
  if flags.use_orientation:
    sl["orientation"] = (offset, offset + 3)
    offset += 3
  if flags.use_color:
    sl["color"] = (offset, offset + 3)
    offset += 3
  if flags.use_base_xy:
    sl["base_xy"] = (offset, offset + 2)
    offset += 2
  if flags.use_stem_size:
    sl["stem_size"] = (offset, offset + 2)
    offset += 2
  sl["total_dim"] = offset
  return sl


def sample_rotation(rot_cfg: Any) -> np.ndarray:
  """Sample a random 3D rotation matrix based on configuration ranges.

  Args:
      rot_cfg: Configuration object with yaw/pitch/roll degree ranges.

  Returns:
      3x3 rotation matrix as numpy array (float32).
  """
  yaw = math.radians(random.uniform(*rot_cfg.yaw_deg))
  pitch = math.radians(random.uniform(*rot_cfg.pitch_deg))
  roll = math.radians(random.uniform(*rot_cfg.roll_deg))
  cy, sy = math.cos(yaw), math.sin(yaw)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cr, sr = math.cos(roll), math.sin(roll)
  R = np.array(
    [
      [cy * cr + sy * sp * sr, sr * cp, -sy * cr + cy * sp * sr],
      [-cy * sr + sy * sp * cr, cr * cp, sy * sr + cy * sp * cr],
      [sy * cp, -sp, cy * cp],
    ],
    dtype=np.float32,
  )
  return R


def center_data(data: Any, cfg: Any) -> Any:
  """Center the scene graph data around the mean stem position.

  Args:
      data: PyG data object.
      cfg: Configuration object.

  Returns:
      Modified data object with centered positions and features.
  """
  sl = feature_slices(cfg.features)
  stem_idx = NODE_CLASSES.index("stem")
  stem_mask = data.y_node == stem_idx
  x = data.x

  if not hasattr(data, "batch") or data.batch is None:
    stem_xy = data.pos[stem_mask][:, :2]
    if stem_xy.numel() == 0:
      return data
    stem_mean_xy = stem_xy.mean(dim=0)
    t = torch.tensor([stem_mean_xy[0], stem_mean_xy[1], 0.0], device=data.pos.device, dtype=data.pos.dtype)
    data.pos = data.pos - t
    if "centroid" in sl:
      c0, c1 = sl["centroid"]
      x[:, c0:c1] = x[:, c0:c1] - t
    if "base_xy" in sl:
      b0, b1 = sl["base_xy"]
      x[:, b0:b1] = x[:, b0:b1] - t[:2]
    data.x = x
    return data

  for g in data.batch.unique().tolist():
    gm = data.batch == g
    gs = gm & stem_mask
    if not gs.any():
      continue
    stem_xy = data.pos[gs][:, :2]
    stem_mean_xy = stem_xy.mean(dim=0)
    t = torch.tensor([stem_mean_xy[0], stem_mean_xy[1], 0.0], device=data.pos.device, dtype=data.pos.dtype)
    data.pos[gm] = data.pos[gm] - t
    if "centroid" in sl:
      c0, c1 = sl["centroid"]
      x[gm, c0:c1] = x[gm, c0:c1] - t
    if "base_xy" in sl:
      b0, b1 = sl["base_xy"]
      x[gm, b0:b1] = x[gm, b0:b1] - t[:2]

  data.x = x
  return data


def rotate_data(data: Any, R: np.ndarray, cfg: Any) -> Any:
  """Apply a 3D rotation to the scene graph data.

  Args:
      data: PyG data object.
      R: 3x3 rotation matrix.
      cfg: Configuration object.

  Returns:
      Modified data object with rotated positions and features.
  """
  sl = feature_slices(cfg.features)
  x = data.x
  R_t = torch.from_numpy(R).to(x.device)

  data.pos = R_t.mm(data.pos.T).T

  if "centroid" in sl:
    c0, c1 = sl["centroid"]
    x[:, c0:c1] = R_t.mm(x[:, c0:c1].T).T

  if "axes" in sl:
    a0, a1 = sl["axes"]
    axes = x[:, a0:a1].reshape(-1, 3, 3)
    axes = torch.matmul(R_t, axes)
    x[:, a0:a1] = axes.reshape(-1, 9)

  if "orientation" in sl:
    o0, o1 = sl["orientation"]
    orient = x[:, o0:o1]
    orient = torch.matmul(R_t, orient.unsqueeze(-1)).squeeze(-1)
    x[:, o0:o1] = orient

  if "base_xy" in sl:
    b0, b1 = sl["base_xy"]
    base_xy = x[:, b0:b1]
    base_xyz = torch.cat(
      [base_xy, torch.zeros((base_xy.size(0), 1), device=base_xy.device, dtype=base_xy.dtype)],
      dim=1,
    )
    base_xyz = R_t.mm(base_xyz.T).T
    x[:, b0:b1] = base_xyz[:, :2]

  data.x = x
  return data


def apply_dropout(data: Any, cfg: Any) -> Any:
  """Apply random node and edge dropout for augmentation.

  Args:
      data: PyG data object.
      cfg: Configuration object.

  Returns:
      Modified data object (or original if dropout disabled).
  """
  if not cfg.drop_aug.apply_drop:
    return data

  keep = torch.rand(data.num_nodes, device=data.x.device) > cfg.drop_aug.node_drop_prob

  if cfg.drop_aug.protect_stems:
    stem_idx = NODE_CLASSES.index("stem")
    stems = (data.y_node == stem_idx).nonzero(as_tuple=False).flatten()
    keep[stems] = True

  if keep.sum() == 0:
    keep[random.randrange(data.num_nodes)] = True

  idx_map = torch.full((data.num_nodes,), -1, device=data.x.device, dtype=torch.long)
  idx_map[keep] = torch.arange(int(keep.sum().item()), device=data.x.device)

  data.x = data.x[keep]
  data.pos = data.pos[keep]
  data.y_node = data.y_node[keep]
  if hasattr(data, "batch"):
    data.batch = data.batch[keep]

  if hasattr(data, "edge_index"):
    ei = data.edge_index
    me = keep[ei[0]] & keep[ei[1]]
    ei = ei[:, me]
    data.edge_index = idx_map[ei]

  if hasattr(data, "gt_edge_pairs"):
    gp = data.gt_edge_pairs
    mg = keep[gp[:, 0]] & keep[gp[:, 1]]
    gp = gp[mg]
    data.gt_edge_pairs = torch.stack([idx_map[gp[:, 0]], idx_map[gp[:, 1]]], dim=1)
    if hasattr(data, "gt_edge_labels"):
      data.gt_edge_labels = data.gt_edge_labels[mg]

  if hasattr(data, "gt_edge_pairs") and data.gt_edge_pairs.numel() > 0:
    keep_e = torch.rand(data.gt_edge_pairs.size(0), device=data.x.device) > cfg.drop_aug.edge_drop_prob
    data.gt_edge_pairs = data.gt_edge_pairs[keep_e]
    if hasattr(data, "gt_edge_labels"):
      data.gt_edge_labels = data.gt_edge_labels[keep_e]

  if hasattr(data, "occl_pairs") and data.occl_pairs.numel() > 0:
    # Drop occlusion pairs if either endpoint was removed.
    occl = data.occl_pairs
    keep_o = keep[occl[:, 0]] & keep[occl[:, 1]]
    occl = occl[keep_o]
    data.occl_pairs = torch.stack([idx_map[occl[:, 0]], idx_map[occl[:, 1]]], dim=1)
    if hasattr(data, "occl_raw"):
      data.occl_raw = data.occl_raw[keep_o]
    if hasattr(data, "occl_weighted"):
      data.occl_weighted = data.occl_weighted[keep_o]

  # Drop occlusion candidate pools if either endpoint was removed.
  for attr in ("occl_base_pairs", "occl_ring_pairs", "occl_far_pairs"):
    if hasattr(data, attr):
      pairs = getattr(data, attr)
      if pairs is None or pairs.numel() == 0:
        continue
      keep_p = keep[pairs[:, 0]] & keep[pairs[:, 1]]
      pairs = pairs[keep_p]
      setattr(data, attr, torch.stack([idx_map[pairs[:, 0]], idx_map[pairs[:, 1]]], dim=1))
      if attr == "occl_base_pairs":
        if hasattr(data, "occl_base_raw"):
          data.occl_base_raw = data.occl_base_raw[keep_p]
        if hasattr(data, "occl_base_weighted"):
          data.occl_base_weighted = data.occl_base_weighted[keep_p]
      if attr == "occl_ring_pairs":
        if hasattr(data, "occl_ring_raw"):
          data.occl_ring_raw = data.occl_ring_raw[keep_p]
        if hasattr(data, "occl_ring_weighted"):
          data.occl_ring_weighted = data.occl_ring_weighted[keep_p]
      if attr == "occl_far_pairs":
        if hasattr(data, "occl_far_raw"):
          data.occl_far_raw = data.occl_far_raw[keep_p]
        if hasattr(data, "occl_far_weighted"):
          data.occl_far_weighted = data.occl_far_weighted[keep_p]

  return data


def perturb_vector(vec: List[float], std: float, normalize: bool = False) -> np.ndarray:
  """Add Gaussian noise to a vector.

  Args:
      vec: Input vector (list or array).
      std: Standard deviation of noise.
      normalize: Whether to re-normalize the result.

  Returns:
      Perturbed vector as numpy array.
  """
  arr = np.asarray(vec, dtype=np.float32)
  if std > 0:
    arr = arr + np.random.normal(0.0, std, size=arr.shape)
  if normalize:
    n = np.linalg.norm(arr) + 1e-8
    arr = arr / n
  return arr


def perturb_axes(axes: List[float], std: float) -> np.ndarray:
  """Add noise to OBB axes and re-orthogonalize (approx).

  Args:
      axes: Flattened axes (9 element list/array).
      std: Standard deviation of noise.

  Returns:
      Perturbed axes matrix (3x3).
  """
  arr = np.asarray(axes, dtype=np.float32)
  if std > 0:
    arr = arr + np.random.normal(0.0, std, size=arr.shape)
  for i in range(arr.shape[1]):
    n = np.linalg.norm(arr[:, i]) + 1e-8
    arr[:, i] = arr[:, i] / n
  return arr


def perturb_extents(extents: List[float], std: float) -> np.ndarray:
  """Add noise to OBB extents.

  Args:
      extents: OBB extents vector.
      std: Standard deviation of noise.

  Returns:
      Perturbed extents array.
  """
  arr = np.asarray(extents, dtype=np.float32)
  if std > 0:
    arr = arr + np.random.normal(0.0, std, size=arr.shape)
  return np.clip(arr, 1e-4, None)


def perturb_color(rgb: List[float], noise_cfg: Any) -> np.ndarray:
  """Add HSV noise to an RGB color.

  Args:
      rgb: Input RGB color (0-1).
      noise_cfg: configuration with color noise parameters.

  Returns:
      Perturbed RGB color array (float32).
  """
  r, g, b = [float(np.clip(c, 0.0, 1.0)) for c in rgb]
  h, s, v = colorsys.rgb_to_hsv(r, g, b)
  h = (h + np.random.normal(0.0, noise_cfg.color_hue_std)) % 1.0
  s = np.clip(s + np.random.normal(0.0, noise_cfg.color_sat_std), 0.0, 1.0)
  v = np.clip(
    v + np.random.normal(0.0, noise_cfg.color_val_std) + random.uniform(*noise_cfg.shadow_shift_range),
    0.0,
    1.0,
  )
  r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
  return np.array([r2, g2, b2], dtype=np.float32)


def node_feature_vector(node: Dict[str, Any], cfg: Any) -> Tuple[np.ndarray, np.ndarray]:
  """Construct the input feature vector for a node.

  Args:
      node: Node dictionary from raw graph.
      cfg: Configuration object.

  Returns:
      Tuple of (feature_vector, centroid_position).
  """
  feats = []

  centroid = perturb_vector(node["centroid"], cfg.noise.centroid_std)
  centroid_raw = centroid.copy()

  axes_mat = perturb_axes(node["obb"]["axes"], cfg.noise.axis_std)
  axes = axes_mat.reshape(-1)

  extents = perturb_extents(node["obb"]["extents"], cfg.noise.extent_std)
  orientation = perturb_vector(node["orientation"], cfg.noise.orientation_std, normalize=True)
  color = perturb_color(node.get("color", [0.8, 0.8, 0.8]), cfg.noise)

  stem_size = np.zeros(2, dtype=np.float32)
  if extents.size:
    main_idx = int(np.argmax(extents))
    main_axis = axes_mat[:, main_idx]
    base = centroid - main_axis * (extents[main_idx] * 0.5)
    base_xy = base[:2].astype(np.float32)
    minor = [extents[i] for i in range(3) if i != main_idx]
    if len(minor) == 2:
      radius = 0.25 * float(minor[0] + minor[1])
    elif len(minor) == 1:
      radius = 0.5 * float(minor[0])
    else:
      radius = 0.0
    height = float(extents[main_idx])
    stem_size = np.array([radius, height], dtype=np.float32)
  else:
    base_xy = centroid[:2].astype(np.float32)

  if node["type"] == "stem":
    if cfg.features.use_centroid:
      centroid = np.zeros(3, dtype=np.float32)
  else:
    if cfg.features.use_stem_size:
      stem_size = np.zeros(2, dtype=np.float32)

  if cfg.features.use_centroid:
    feats.append(centroid)
  if cfg.features.use_axes:
    feats.append(axes)
  if cfg.features.use_extents:
    feats.append(extents)
  if cfg.features.use_orientation:
    feats.append(orientation)
  if cfg.features.use_color:
    feats.append(color)
  if cfg.features.use_base_xy:
    feats.append(base_xy)
  if cfg.features.use_stem_size:
    feats.append(stem_size)

  if not feats:
    feats.append(np.zeros(3, dtype=np.float32))

  x = np.concatenate(feats, axis=0).astype(np.float32)
  # Critical Fix: Always use raw centroid for graph structure (pos), regardless of whether it's in the features (x)
  pos = centroid_raw.astype(np.float32)
  return x, pos
