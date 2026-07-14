
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn*xn + yn*yn + zn*zn) + (xm*xm + ym*ym + zm*zm) - 2 * (xn*xm + yn*ym + zn*zm)
    
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, K, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=1)
    return centroids

def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    
    # Use topk instead of sort for better memory/speed
    # Mask out points beyond radius
    sqrdists[sqrdists > radius ** 2] = 1e10
    _, group_idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False)
    
    # Optional: ensure we don't pick the 1e10 points if they are among topk
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = torch.gather(sqrdists, 2, group_idx) > 1e9
    group_idx[mask] = group_first[mask]
    
    return group_idx

class SharedMLP(nn.Module):
    def __init__(self, in_ch: int, mlp: List[int], bn: bool = True):
        super().__init__()
        layers = []
        last = in_ch
        for out_ch in mlp:
            layers.append(nn.Conv2d(last, out_ch, kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SetAbstractionSSG(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_ch: int, mlp: List[int], use_xyz: bool = True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.mlp = SharedMLP(in_ch + (3 if use_xyz else 0), mlp)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        B, N, _ = xyz.shape
        S = self.npoint
        
        # Fast FPS for large N
        if N > 16384:
            sub_idx = torch.randperm(N, device=xyz.device)[:16384]
            xyz_sub = xyz[:, sub_idx, :]
            fps_idx = farthest_point_sample(xyz_sub, S)
            fps_idx = sub_idx[fps_idx]
        else:
            fps_idx = farthest_point_sample(xyz, S)
            
        new_xyz = index_points(xyz, fps_idx)
        group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

        if points is not None:
            grouped_points = index_points(points, group_idx)
            if self.use_xyz:
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        
        return new_xyz, new_points

class SetAbstractionMSG(nn.Module):
    def __init__(self, npoint: int, radius_list: List[float], nsample_list: List[int], in_ch: int, mlp_list: List[List[int]], use_xyz: bool = True):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.use_xyz = use_xyz
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            layers = []
            last = in_ch + (3 if use_xyz else 0)
            for out_ch in mlp_list[i]:
                layers.append(nn.Conv2d(last, out_ch, kernel_size=1, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                last = out_ch
            self.conv_blocks.append(nn.Sequential(*layers))

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """
        xyz: [B, N, 3]
        points: [B, N, C]
        """
        B, N, C = xyz.shape
        S = self.npoint
        
        # Fast FPS for large N
        if N > 16384:
            sub_idx = torch.randperm(N, device=xyz.device)[:16384]
            xyz_sub = xyz[:, sub_idx, :]
            fps_idx = farthest_point_sample(xyz_sub, S)
            fps_idx = sub_idx[fps_idx]
        else:
            fps_idx = farthest_point_sample(xyz, S)
            
        new_xyz = index_points(xyz, fps_idx)
        
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            nsample = self.nsample_list[i]
            group_idx = ball_query(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, 3)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                if self.use_xyz:
                    grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous() # [B, D, S, K]
            
            for layer in self.conv_blocks[i]: # Applying manual MLP layers because shared MLP class expects slight diff
                 if isinstance(layer, nn.Conv2d):
                     grouped_points = layer(grouped_points)
                 elif isinstance(layer, nn.BatchNorm2d):
                     grouped_points = layer(grouped_points)
                 elif isinstance(layer, nn.ReLU):
                     grouped_points = layer(grouped_points)

            new_points = torch.max(grouped_points, 3)[0] # Max pooling over K -> [B, D, S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1) # [B, sum(D), S]
        new_points_concat = new_points_concat.permute(0, 2, 1).contiguous()
        
        return new_xyz, new_points_concat

class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, mlp: List[int], bn: bool = True):
        super().__init__()
        layers = []
        last = in_ch
        for out_ch in mlp:
            layers.append(nn.Conv1d(last, out_ch, kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: Optional[torch.Tensor], points2: torch.Tensor):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=False)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm

            grouped_points = index_points(points2, idx)
            interpolated = torch.sum(grouped_points * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.net(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_points
