import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, radius, PointNetConv


class ProposalModule(nn.Module):
  """
  VoteNet-lite proposal head.
  Produces K object proposals from voted points.
  """
  def __init__(self, feat_dim=128):
    super().__init__()

    self.group_aggregator = PointNetConv(
      local_nn=nn.Sequential(
        nn.Linear(feat_dim + 3, feat_dim),
        nn.ReLU(),
        nn.Linear(feat_dim, feat_dim),
      )
    )

    # [MODIFY] v10: 2 evidence (alpha/beta) + 3 center + 3 extent = 8
    # v10.6: Decoupled Heads
    # Discovery branch (Alpha/Beta)
    self.discovery_head = nn.Sequential(
      nn.Linear(feat_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 2)
    )
    # Localization branch (Center/Extent)
    self.localization_head = nn.Sequential(
      nn.Linear(feat_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 6)
    )

    # [v10.6 BIAS] Decoupled initialization
    with torch.no_grad():
      # Discovery: Start with Pro-Object Bias
      nn.init.xavier_uniform_(self.discovery_head[-1].weight, gain=0.01)
      self.discovery_head[-1].bias.fill_(0.0)
      self.discovery_head[-1].bias[0] = 1.0 # Objectness Bias
      
      # Localization: Neutral
      nn.init.xavier_uniform_(self.localization_head[-1].weight, gain=0.01)
      self.localization_head[-1].bias.fill_(0.0)

  def forward(
      self,
      xyz: torch.Tensor,       # (N,3) voted points
      feat: torch.Tensor,      # (N,C) backbone features
      batch: torch.Tensor,     # (N,)
      ratio: float = 0.1,
      radius_r: float = 0.05,
      max_nn: int = 32,
      gt_centers: torch.Tensor = None, # (B, M, 3)
      gt_mask: torch.Tensor = None     # (B, M)
  ):
    """
    Returns:
      Dict containing:
        "obj_prob": (K,) Expected objectness probability
        "alpha": (K,) Beta alpha parameter
        "beta": (K,) Beta beta parameter
        "uncertainty": (K,) Vacuity/Uncertainty
        "center": (K,3) Predicted proposal centers
        "extent": (K,3) Predicted proposal extents
        "seed_batch": (K,) Batch index for each seed
    """

    # 1) FPS seeds
    seed_idx = fps(xyz, batch=batch, ratio=ratio)
    seed_xyz = xyz[seed_idx]
    seed_feat = feat[seed_idx]
    seed_batch = batch[seed_idx]
    
    # 1b) Hybrid Sampling (GT Injection) during training
    if self.training and gt_centers is not None and gt_mask is not None:
        batch_size = gt_centers.shape[0]
        gt_seeds_xyz = []
        gt_seeds_feat = []
        gt_seeds_batch = []
        
        for b in range(batch_size):
            b_valid = gt_mask[b] > 0.5
            if not b_valid.any(): continue
            
            # Select GT centers for this batch item
            g_cen = gt_centers[b][b_valid] # (M_b, 3)
            
            # Add small noise to avoid trivial solutions (jittering)
            g_cen = g_cen + torch.randn_like(g_cen) * 0.005 
            
            # Borrow features from nearest backbone points in this batch item
            b_points_mask = (batch == b)
            if not b_points_mask.any(): continue
            
            b_xyz = xyz[b_points_mask]
            b_feat = feat[b_points_mask]
            
            # Find nearest neighbor manually for feature assignment
            # dists: (M_b, N_b)
            dists = torch.cdist(g_cen, b_xyz)
            nn_idx = dists.argmin(dim=1)
            g_feat = b_feat[nn_idx]
            
            gt_seeds_xyz.append(g_cen)
            gt_seeds_feat.append(g_feat)
            gt_seeds_batch.append(torch.full((g_cen.shape[0],), b, dtype=torch.long, device=xyz.device))
        
        if gt_seeds_xyz:
            gt_seeds_xyz = torch.cat(gt_seeds_xyz, dim=0)
            gt_seeds_feat = torch.cat(gt_seeds_feat, dim=0)
            gt_seeds_batch = torch.cat(gt_seeds_batch, dim=0)
            
            # Concatenate FPS + GT seeds
            seed_xyz = torch.cat([seed_xyz, gt_seeds_xyz], dim=0)
            seed_feat = torch.cat([seed_feat, gt_seeds_feat], dim=0)
            seed_batch = torch.cat([seed_batch, gt_seeds_batch], dim=0)

    # 2) Radius grouping (points -> seeds)
    row, col = radius(
      x=xyz,
      y=seed_xyz,
      r=radius_r,
      batch_x=batch,
      batch_y=seed_batch,
      max_num_neighbors=max_nn
    )
    edge_index = torch.stack([col, row], dim=0)

    # 3) Bipartite PointNetConv (aggregate point features to seeds)
    seed_agg = self.group_aggregator(
      x=(feat, seed_feat),
      pos=(xyz, seed_xyz),
      edge_index=edge_index
    )  # (K,C)

    # 3. Decoupled Discovery & Localization (v10.6)
    disc_logits = self.discovery_head(seed_agg)
    loc_preds = self.localization_head(seed_agg)
    
    # Beta-Bernoulli Evidence
    evidence = torch.clamp(F.softplus(disc_logits), max=100.0)
    alpha = evidence[:, 0] + 1.0
    beta = evidence[:, 1] + 1.0
    
    # Expected prob for metrics/NMS
    S = alpha + beta
    obj_prob = alpha / S
    uncertainty = 2.0 / S
    ranking_score = obj_prob * (1.0 - uncertainty)


    dcenter = loc_preds[:, :3]
    raw_extent = loc_preds[:, 3:6]
    dextent = F.softplus(raw_extent) # [FIX] Strictly positive extents
    prop_center = seed_xyz + dcenter

    # Return dict for v10 loss processing
    return {
        "obj_prob": obj_prob,
        "ranking_score": ranking_score, # For v11 NMS
        "alpha": alpha,
        "beta": beta,
        "uncertainty": uncertainty,
        "center": prop_center,
        "extent": dextent,
        "seed_batch": seed_batch
    }
