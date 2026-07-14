import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


from pepper_scene_graphs.models.pointnet2_utils import (
    SetAbstractionSSG, 
    FeaturePropagation, 
    square_distance, 
    index_points, 
    farthest_point_sample, 
    ball_query
)


class PointNet2SemanticSeg(nn.Module):
  """
  Full-scene semantic segmentation.
  Expects xyz+normals: x in (B, N, 6) or (xyz, normals) separately.
  """
  def __init__(self, num_classes: int, in_ch: int = 9):
    super().__init__()
    # SA layers (SSG)
    # in_ch represents the total features per point (e.g., 9 if xyz + normals + RGB)
    # We strip xyz (3) internally, so features provided to sa1 should be (in_ch - 3)
    feature_ch = in_ch - 3
    
    self.sa1 = SetAbstractionSSG(npoint=1024, radius=0.02, nsample=32, in_ch=feature_ch, mlp=[64, 64, 128])
    self.sa2 = SetAbstractionSSG(npoint=256, radius=0.05, nsample=32, in_ch=128, mlp=[128, 128, 256])
    self.sa3 = SetAbstractionSSG(npoint=64, radius=0.10, nsample=32, in_ch=256, mlp=[256, 256, 512])
    self.sa4 = SetAbstractionSSG(npoint=16, radius=0.20, nsample=32, in_ch=512, mlp=[512, 512, 1024])

    # FP layers
    self.fp4 = FeaturePropagation(in_ch=1024 + 512, mlp=[512, 512])
    self.fp3 = FeaturePropagation(in_ch=512 + 256, mlp=[512, 256])
    self.fp2 = FeaturePropagation(in_ch=256 + 128, mlp=[256, 128])
    self.fp1 = FeaturePropagation(in_ch=128 + feature_ch, mlp=[128, 128, 128])

    # Per-point classifier
    self.classifier = nn.Sequential(
      nn.Conv1d(128, 128, 1, bias=False),
      nn.BatchNorm1d(128),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Conv1d(128, num_classes, 1)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, N, in_ch)
    xyz = x[:, :, 0:3].contiguous()
    l0_points = x[:, :, 3:].contiguous()

    l0_xyz = xyz
    # l0_points: (B, N, feature_ch)

    l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
    l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
    l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
    l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

    # Feature propagation back up to N points
    l3_points_up = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
    l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_up)
    l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
    l0_points_up = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_up)

    # logits: (B, num_classes, N) -> (B, N, num_classes)
    logits = self.classifier(l0_points_up.permute(0, 2, 1).contiguous())
    return logits.permute(0, 2, 1).contiguous()


if __name__ == "__main__":
  # Quick shape check
  B, N = 2, 20000
  num_classes = 6
  model = PointNet2SemanticSeg(num_classes=num_classes)
  x = torch.randn(B, N, 9)
  with torch.no_grad():
    y = model(x)
  print("out:", y.shape)  # (B, N, num_classes)
