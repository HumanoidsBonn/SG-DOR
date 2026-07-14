# pepper_scene_graphs/models/oracle_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

from pepper_scene_graphs.models.pointnet2_utils import SetAbstractionMSG


def _ensure_bnc(x: torch.Tensor) -> torch.Tensor:
  """
  Ensure point tensors are [B, N, C].
  Accepts [B, N, C] or [B, C, N].
  """
  if x.dim() != 3:
    raise ValueError(f"Expected 3D tensor, got shape={tuple(x.shape)}")
  b, d1, d2 = x.shape
  # Heuristic: if last dim is 3 (xyz) or 3/6/9 feature, assume [B, N, C].
  if d2 in (3, 6, 9, 12):
    return x
  # Otherwise if middle dim looks like channel count, transpose.
  return x.transpose(1, 2).contiguous()


def _pool_over_samples(feats: torch.Tensor) -> torch.Tensor:
  """
  Max pool over the SA sample dimension S.
  Supports:
    feats: [B, C, S] -> returns [B, C]
    feats: [B, S, C] -> returns [B, C]
  """
  if feats.dim() != 3:
    raise ValueError(f"Expected 3D feats, got shape={tuple(feats.shape)}")

  # Common PointNet++ impl returns [B, C, S]
  if feats.shape[1] > feats.shape[2]:
    # [B, C, S]
    return feats.max(dim=2).values

  # [B, S, C]
  return feats.max(dim=1).values


class SharedMSGEncoder(nn.Module):
  """
  Per-organ PointNet++ MSG encoder producing a single identity embedding per instance.
  Inputs:
    organ_points:  [B, N, 3]
    organ_normals: [B, N, 3]
  Output:
    node_identity: [B, feat_dim]
  """
  def __init__(
    self,
    feat_dim: int = 128,
    npoint: int = 256,
    radius_list: Optional[list[float]] = None,
    nsample_list: Optional[list[int]] = None,
    use_normals: bool = True,
    use_color: bool = True
  ):
    super().__init__()

    self.use_normals = use_normals
    self.use_color = use_color

    if radius_list is None:
      radius_list = [0.01, 0.02, 0.04]
    if nsample_list is None:
      nsample_list = [16, 32, 64]

    # Dynamic input channels based on features
    in_ch = 0
    if use_normals: in_ch += 3
    if use_color: in_ch += 3

    # MSG: features are concatenated across scales.
    self.sa1 = SetAbstractionMSG(
      npoint=npoint,
      radius_list=radius_list,
      nsample_list=nsample_list,
      in_ch=max(in_ch, 0), 
      mlp_list=[
        [32, 32, 64],
        [64, 64, 128],
        [64, 96, 128]
      ]
    )

    # We’ll infer the concat feature dim at runtime once and initialize post_msg lazily.
    self._post_msg: Optional[nn.Sequential] = None
    self._feat_dim = feat_dim

  def _init_post(self, in_dim: int) -> None:
    self._post_msg = nn.Sequential(
      nn.Linear(in_dim, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, self._feat_dim)
    )

  def forward(self, organ_points: torch.Tensor, organ_normals: torch.Tensor, organ_colors: torch.Tensor = None, chunk_size: int = 512) -> torch.Tensor:
    """
    Args:
        organ_points:  [B, N, 3]
        organ_normals: [B, N, 3]
        organ_colors:  [B, N, 3] (optional, zeroed if None)
        chunk_size: number of instances to process in one pass to save VRAM.
    """
    organ_points = _ensure_bnc(organ_points)
    organ_normals = _ensure_bnc(organ_normals) if organ_normals is not None else torch.zeros_like(organ_points)
    
    if organ_colors is None:
        organ_colors = torch.zeros_like(organ_points)
    else:
        organ_colors = _ensure_bnc(organ_colors)

    num_instances = organ_points.shape[0]

    # Ensure post_msg is initialized
    if self._post_msg is None:
      # Temporary dummy forward to get input dim
      with torch.no_grad():
        dummy_pts = organ_points[:1]
        dummy_norms = organ_normals[:1]
        dummy_cols = organ_colors[:1]
        
        # Build dummy in_feats
        feats_list = []
        if self.use_normals: feats_list.append(dummy_norms)
        if self.use_color: feats_list.append(dummy_cols)
        
        dummy_in_feats = torch.cat(feats_list, dim=-1) if feats_list else None
        _, dummy_feats = self.sa1(dummy_pts, dummy_in_feats)
        dummy_pooled = _pool_over_samples(dummy_feats)
        self._init_post(dummy_pooled.shape[-1])
      self._post_msg = self._post_msg.to(organ_points.device)

    dummy = torch.tensor([1.0], device=organ_points.device, requires_grad=True)

    def run_chunk(pts, norms, cols, dummy_val):
      # Build chunk in_feats
      feats_list = []
      if self.use_normals: feats_list.append(norms)
      if self.use_color: feats_list.append(cols)
      
      in_feats = torch.cat(feats_list, dim=-1) if feats_list else None
      l1_xyz, l1_feats = self.sa1(pts, in_feats)
      pooled = _pool_over_samples(l1_feats)
      return self._post_msg(pooled)

    results = []
    for i in range(0, num_instances, chunk_size):
      pts_chunk = organ_points[i:i + chunk_size]
      norms_chunk = organ_normals[i:i + chunk_size]
      cols_chunk = organ_colors[i:i + chunk_size]

      if self.training:
        # Torch 2.4+ requires use_reentrant to be explicit; False is recommended.
        # Older torch versions may not accept this kwarg, so fall back gracefully.
        try:
          chunk_out = torch.utils.checkpoint.checkpoint(
            run_chunk,
            pts_chunk,
            norms_chunk,
            cols_chunk,
            dummy,
            use_reentrant=False,
          )
        except TypeError:
          chunk_out = torch.utils.checkpoint.checkpoint(run_chunk, pts_chunk, norms_chunk, cols_chunk, dummy)
      else:
        chunk_out = run_chunk(pts_chunk, norms_chunk, cols_chunk, None)
      results.append(chunk_out)

    return torch.cat(results, dim=0)
