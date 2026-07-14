import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from pepper_scene_graphs.models.pointnet2_utils import SetAbstractionSSG, SetAbstractionMSG, FeaturePropagation

class PointNet2MultiHead(nn.Module):
    """
    Multi-Head PointNet++ for Simultaneous Semantic and Instance Segmentation.
    Heads:
    1. Semantic: (B, N, Classes) - Softmax classification
    2. Offset: (B, N, 3) - Centroid voting (regression)
    3. Extent: (B, N, 3) - AABB dimension regression
    """
    def __init__(self, num_classes: int, in_ch: int = 9, use_msg: bool = False):
        super().__init__()
        feature_ch = in_ch - 3
        
        self.use_msg = use_msg

        if self.use_msg:
             # MSG Configuration (2 scales per level, summing to SSG channels)
             # SA1: 64+64=128
             self.sa1 = SetAbstractionMSG(1024, [0.02, 0.04], [16, 32], feature_ch, [[32, 32, 64], [32, 32, 64]])
             # SA2: 128+128=256
             self.sa2 = SetAbstractionMSG(256, [0.05, 0.10], [16, 32], 128, [[64, 64, 128], [64, 64, 128]])
             # SA3: 256+256=512
             self.sa3 = SetAbstractionMSG(64, [0.10, 0.20], [16, 32], 256, [[128, 128, 256], [128, 128, 256]])
             # SA4: 512+512=1024
             self.sa4 = SetAbstractionMSG(16, [0.20, 0.40], [16, 32], 512, [[256, 256, 512], [256, 256, 512]])
        else:
             # SSG Configuration (Legacy v6/v12)
             self.sa1 = SetAbstractionSSG(npoint=1024, radius=0.02, nsample=32, in_ch=feature_ch, mlp=[64, 64, 128])
             self.sa2 = SetAbstractionSSG(npoint=256, radius=0.05, nsample=32, in_ch=128, mlp=[128, 128, 256])
             self.sa3 = SetAbstractionSSG(npoint=64, radius=0.10, nsample=32, in_ch=256, mlp=[256, 256, 512])
             self.sa4 = SetAbstractionSSG(npoint=16, radius=0.20, nsample=32, in_ch=512, mlp=[512, 512, 1024])

        # --- Decoder (FP) ---
        # Dimensions are identical for MSG/SSG due to channel summation
        self.fp4 = FeaturePropagation(in_ch=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_ch=512 + 256, mlp=[512, 256])
        self.fp2 = FeaturePropagation(in_ch=256 + 128, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_ch=128 + feature_ch, mlp=[128, 128, 128])

        # --- Shared Final Feature MLP ---
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # --- Task Heads ---
        # 1. Semantic Head (Standard classification)
        self.semantic_head = nn.Conv1d(128, num_classes, 1)
        
        # 2. Offset Branch (Decoupled Polar Coordinates)
        # Direction Head (Compass): Unit vector to centroid
        self.offset_dir = nn.Conv1d(128, 3, 1)
        # Magnitude Head (Ruler): Distance to centroid
        self.offset_mag = nn.Conv1d(128, 1, 1)
        
        # 3. Extent Head (Regression: Log-Instance Dimensions L,W,H)
        self.extent_head = nn.Conv1d(128, 3, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, N, in_ch)
        xyz = x[:, :, 0:3].contiguous()
        l0_points = x[:, :, 3:].contiguous()

        l0_xyz = xyz

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder
        l3_points_up = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_up)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
        l0_points_up = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_up)

        # (B, N, 128) -> (B, 128, N) for Conv1d
        feat_raw = l0_points_up.permute(0, 2, 1).contiguous()
        feat = self.shared_mlp(feat_raw)

        # Branches
        # (B, num_classes, N) -> (B, N, num_classes)
        sem_logits = self.semantic_head(feat).permute(0, 2, 1).contiguous()
        
        # --- Decoupled Offset Logic ---
        # 1. Direction (B, 3, N) -> (B, N, 3) + L2 Normalization
        raw_dir = self.offset_dir(feat).permute(0, 2, 1).contiguous()
        unit_dir = F.normalize(raw_dir, p=2, dim=-1)
        
        # 2. Magnitude (B, 1, N) -> (B, N, 1) + Softplus (Distance)
        mag = F.softplus(self.offset_mag(feat)).permute(0, 2, 1).contiguous()
        
        # 3. Extent (B, 3, N) -> (B, N, 3) (Log-Meters: Unconstrained)
        extents = self.extent_head(feat).permute(0, 2, 1).contiguous()
        
        # Return features as (B, N, 128) for downstream pooling
        feat_out = feat.permute(0, 2, 1).contiguous()

        return sem_logits, unit_dir, mag, extents, feat_out

if __name__ == "__main__":
    B, N = 2, 8192
    num_classes = 5
    model = PointNet2MultiHead(num_classes=num_classes)
    x = torch.randn(B, N, 9)
    with torch.no_grad():
        sem, unit_dir, mag, ext = model(x)
    print(f"Sem: {sem.shape}, UnitDir: {unit_dir.shape}, Mag: {mag.shape}, Ext: {ext.shape}")
