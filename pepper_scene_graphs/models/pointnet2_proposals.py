import torch
import torch.nn as nn
from .pointnet2_multihead import PointNet2MultiHead
from .proposal_module import ProposalModule
from pepper_scene_graphs.training.proposals.refinement_module import LocalRefiner

class PointNet2Proposals(nn.Module):
    """
    Wrapper around PointNet2MultiHead that adds a VoteNet-lite Proposal Module.
    Now includes v11 Sparse Refinement (Precision Zoom).
    """
    def __init__(self, num_classes: int, in_ch: int = 9, feat_dim: int = 128, use_msg: bool = False):
        super().__init__()
        self.backbone = PointNet2MultiHead(num_classes=num_classes, in_ch=in_ch, use_msg=use_msg)
        self.feat_dim = feat_dim
        self.proposal_module = ProposalModule(feat_dim=feat_dim)
        
        # v11 Refinement: Zoom into Top-K proposals using high-res geometry
        self.refiner = LocalRefiner(feat_dim=feat_dim)

        # Class-specific grouping hyperparams from Handoff (v8)
        self.class_cfg = {
            0: {'ratio': 0.12, 'radius_r': 0.06, 'max_nn': 48}, # Stem: 0.08 -> 0.12
            1: {'ratio': 0.15, 'radius_r': 0.03, 'max_nn': 32}, # Leaf: 0.10 -> 0.15
            2: {'ratio': 0.15, 'radius_r': 0.025, 'max_nn': 32},# Peduncle
            3: {'ratio': 0.10, 'radius_r': 0.05, 'max_nn': 32}  # Fruit
        }

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None, 
                gt_centers: torch.Tensor = None, gt_mask: torch.Tensor = None,
                gt_sem: torch.Tensor = None, gt_classes: torch.Tensor = None,
                use_gt_masks: bool = True, top_k: int = 128):
        """
        Args:
            x: (B, N, C) Input Point Cloud
            batch: (B*N) or (B, N)
            gt_centers: (B, M, 3) Optional GT centers for training injection
            gt_mask: (B, M) Optional GT mask indicating valid centers
            gt_sem: (B, N) Optional GT semantics for point masking
            gt_classes: (B, M) Optional GT class labels for centers
            use_gt_masks: If True, uses gt_sem for point filtering.
            top_k: Max proposals to refine per class/batch.
        """
        if use_gt_masks:
            assert gt_sem is not None, "use_gt_masks=True requires gt_sem input"

        # 1. Run Backbone
        sem_logits, unit_dir, mag, extents, feat_out = self.backbone(x)

        # 2. Construct Voted Points
        xyz = x[:, :, 0:3].contiguous() # (B, N, 3)
        offsets = unit_dir * mag # (B, N, 3)
        xyz_voted = xyz + offsets

        # 3. Preparation for Proposal Module
        B, N, _ = xyz.shape
        C = feat_out.size(-1)
        assert C == self.feat_dim, f"Backbone feat_out dim {C} != Proposal feat_dim {self.feat_dim}"

        xyz_flat = xyz.reshape(B * N, 3)
        feat_flat = feat_out.reshape(B * N, C)
        
        if batch is None:
            batch_flat = torch.arange(B, device=x.device).repeat_interleave(N)
        else:
            batch_flat = batch.view(-1)

        # Semantics source: GT or Predicted
        sem_src = gt_sem.view(-1) if use_gt_masks else sem_logits.argmax(-1).view(-1)
        
        proposals = {}
        refinements = {}

        # 4. Iterate over classes
        for c, cfg in self.class_cfg.items():
            # A) Mask points for this class
            mask_c = (sem_src == c)
            if mask_c.sum() < 5: # Skip if too few points
                proposals[c] = None
                continue
            
            # B) Prepare Class-Specific GT Centers (for Hybrid Sampling)
            gt_mask_c = None
            if gt_centers is not None and gt_mask is not None and gt_classes is not None:
                # Filter mask to only include centers of class c
                gt_mask_c = gt_mask.bool() & (gt_classes == c)
            
            # C) Run Proposal Module
            # [FIX] Use voted points (xyz_voted) for clustering centers
            # We must flatten xyz_voted to match (B*N, 3)
            xyz_voted_flat = xyz_voted.reshape(B * N, 3)
            
            # Logging: Check for starvation
            num_pts_c = mask_c.sum().item()
            
            out = self.proposal_module(
                xyz=xyz_voted_flat[mask_c], # [FIX] Vote-space clustering
                feat=feat_flat[mask_c],
                batch=batch_flat[mask_c],
                gt_centers=gt_centers, 
                gt_mask=gt_mask_c,     
                **cfg
            )
            
            proposals[c] = {
                "obj_prob": out["obj_prob"],
                "ranking_score": out["ranking_score"],
                "alpha": out["alpha"],
                "beta": out["beta"],
                "uncertainty": out["uncertainty"],
                "center": out["center"],
                "extent": out["extent"], 
                "seed_batch": out["seed_batch"],
                "num_pts": num_pts_c # [FIX] logging
            }

            # D) v11 Refinement (Sparse Zoom)
            # Zoom into Top-K using High-Res geometry. Detach feat to save VRAM.
            # Refinement still uses raw geometry (xyz_flat) to align with original surface
            refine_out = self.refiner(
                proposals=proposals[c],
                raw_xyz=xyz_flat,
                raw_feat=feat_flat,
                batch_map=batch_flat,
                top_k=top_k
            )
            refinements[c] = refine_out

        return {
            "sem_logits": sem_logits,
            "unit_dir": unit_dir,
            "mag": mag,
            "extents": extents,
            "feat_out": feat_out,
            "proposals": proposals,
            "refinements": refinements, # Indexed by class
            "xyz_voted": xyz_voted
        }
