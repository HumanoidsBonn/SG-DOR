# pepper_scene_graphs/models/oracle_joint_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from pepper_scene_graphs.models.backbones import GINEBackbone, ResGINEBackbone
from pepper_scene_graphs.models.oracle_encoder import SharedMSGEncoder
from torch_geometric.nn import radius_graph, knn_graph


class OracleGTBaseline(nn.Module):
  """
  Oracle baseline:
    - Node identity from GT per-instance point clusters via MSG encoder
    - Structural reasoning via GINE backbone
    - Heads: node class, edge exist, edge label, centroid offset, extent delta
  """
  def __init__(self, cfg, node_classes: int, edge_labels: int):
    super().__init__()
    self.cfg = cfg
    hdim = cfg.model.hidden_dim
    self.occl_enabled = bool(getattr(cfg.model, "occl_enabled", False))
    self.occl_num_dirs = int(getattr(cfg.model, "occl_num_dirs", 18))
    # Canonical occlusion direction vectors (in the fruit-local frame) in the same order
    # as `scripts/debug/visualize_occlusion_pair.py:build_occlusion_dirs()`.
    self.register_buffer("occl_dirs", self._build_canonical_occl_dirs(self.occl_num_dirs), persistent=False)

    self.shared_encoder = SharedMSGEncoder(
      feat_dim=hdim,
      npoint=getattr(cfg.model, "oracle_msg_npoint", 256),
      radius_list=getattr(cfg.model, "oracle_msg_radius_list", [0.01, 0.02, 0.04]),
      nsample_list=getattr(cfg.model, "oracle_msg_nsample_list", [16, 32, 64]),
      use_normals=getattr(cfg.model, "use_normals", True),
      use_color=getattr(cfg.model, "use_rgb", True)
    )

    # Node features: identity (hdim) + json_geom (6: centroid(3)+extent(3))
    self.node_feat_mode = getattr(cfg.model, "node_feat_mode", None) # None, 'gt', or 'local'
    self.no_geom_injection = getattr(cfg.model, "no_geom_injection", False)
    
    in_dim = hdim
    if not self.no_geom_injection:
        in_dim += 6 # centroid(3) + extent(3)

    if self.node_feat_mode in ['gt', 'local']:
        in_dim += node_classes

    # Edge features for GINE: rel(3) + dist(1) => 4 dims
    edge_dim = 4

    self.backbone = GINEBackbone(
      in_dim=in_dim,
      hidden_dim=hdim,
      num_layers=cfg.model.num_layers,
      dropout=cfg.model.dropout
    )

    # Heads
    self.local_node_head = nn.Linear(hdim, node_classes)  # Predict from encoder alone
    self.node_head = nn.Linear(hdim, node_classes)        # Predict from GNN refined feats

    # Edge heads use [h_src, h_dst, edge_attr(4)]
    self.edge_exist_head = nn.Sequential(
      nn.Linear(hdim * 2 + edge_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, 1)
    )
    self.edge_label_head = nn.Sequential(
      nn.Linear(hdim * 2 + edge_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, edge_labels)
    )

    # Regression heads (per node)
    self.centroid_offset_head = nn.Linear(hdim, 3)  # predict delta in world frame
    self.extent_delta_head = nn.Linear(hdim, 3)     # predict delta extents (AABB/OBB extents)

    # Occlusion heads (optional; used by pointcloud z-buffer occlusion supervision).
    # The key issue for occlusion is *direction-conditioned pooling* over leaf neighborhoods.
    # We therefore use a small cross-attention style scorer:
    #   - query: (fruit embedding, dir embedding)
    #   - key/value: (leaf embedding, fruit-local pair geometry, dir geometry)
    # This avoids relying on the structural candidate graph to route information fruit<->leaf.
    self.occl_dir_embed: Optional[nn.Embedding] = None
    self.occl_q_proj: Optional[nn.Module] = None
    self.occl_k_proj: Optional[nn.Module] = None
    self.occl_v_proj: Optional[nn.Module] = None
    self.occl_union_out: Optional[nn.Module] = None
    # Optional leaf-set self-attention (Transformer encoder) over candidate leaves.
    # When enabled, we condition it on view direction so leaves can interact
    # direction-specifically (depth ordering, silhouette competition, etc.).
    self.occl_leaf_tok_proj: Optional[nn.Module] = None
    self.occl_leaf_dir_geom_proj: Optional[nn.Module] = None
    self.occl_leaf_dir_emb_proj: Optional[nn.Module] = None
    self.occl_leaf_sa: Optional[nn.Module] = None
    self.occl_leaf_sa_mode: str = str(getattr(cfg.model, "occl_leaf_sa_mode", "per_fruit_dir"))
    if self.occl_enabled:
      self._build_occl_heads(node_dim=hdim)

  @staticmethod
  def _build_canonical_occl_dirs(num_dirs: int) -> torch.Tensor:
    """
    Build the canonical direction vectors used by our occlusion convention.

    These are defined in the fruit-local coordinate frame (+X toward stem in XY, +Z world up).
    Ordering matters because direction-indexed targets/logits assume a fixed index mapping.
    """
    if int(num_dirs) != 18:
      # Fallback: slice the canonical 18 if the caller requests fewer (keeps index stability).
      num_dirs = int(num_dirs)
    base = torch.tensor(
      [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      dtype=torch.float32,
    )
    dirs = []
    for v in base:
      dirs.append(v)
      dirs.append(-v)
    for i, j in ((0, 1), (0, 2), (1, 2)):
      for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
          v = base[i] * float(s1) + base[j] * float(s2)
          v = v / (torch.norm(v) + 1e-12)
          dirs.append(v)
    out = torch.stack(dirs, dim=0)  # [18,3]
    if out.size(0) != 18:
      return out
    if int(num_dirs) < 18:
      return out[: int(num_dirs)]
    if int(num_dirs) > 18:
      # Repeat to reach the requested size (should not happen in practice).
      reps = (int(num_dirs) + 17) // 18
      out = out.repeat(reps, 1)[: int(num_dirs)]
    return out

  def _build_occl_heads(self, node_dim: int) -> None:
    hdim = int(self.cfg.model.hidden_dim)
    num_dirs = int(self.occl_num_dirs)
    # Pair geometry (fruit-local, normalized by fruit radius):
    #   rel_local_normed(3) + dist_normed(1) + leaf_rad_ratio(1) + fruit_extent_normed(3) + leaf_extent_normed(3)
    # => 11 dims.
    pair_geom_dim = 11
    dir_embed_dim = int(getattr(self.cfg.model, "occl_dir_embed_dim", 16))
    self.occl_dir_embed = nn.Embedding(num_dirs, dir_embed_dim)
    attn_dim = int(getattr(self.cfg.model, "occl_attn_dim", 64))

    # Optional: leaf-set self-attention over candidate leaves per fruit and direction.
    # Token design (additive direction conditioning):
    #   tok = proj_base([h_leaf, pair_geom]) + proj_dir(dir_geom) + proj_emb(dir_emb)
    sa_layers = int(getattr(self.cfg.model, "occl_leaf_sa_layers", 0))
    self.occl_leaf_sa_mode = str(getattr(self.cfg.model, "occl_leaf_sa_mode", "per_fruit_dir"))
    if self.occl_leaf_sa_mode not in ("per_fruit", "per_fruit_dir"):
      raise ValueError(f"Unknown occl_leaf_sa_mode: {self.occl_leaf_sa_mode!r} (expected 'per_fruit' or 'per_fruit_dir')")
    sa_dim = int(getattr(self.cfg.model, "occl_leaf_sa_dim", hdim))
    sa_heads = int(getattr(self.cfg.model, "occl_leaf_sa_heads", 4))
    sa_ff_mult = int(getattr(self.cfg.model, "occl_leaf_sa_ff_mult", 4))
    sa_dropout = float(getattr(self.cfg.model, "occl_leaf_sa_dropout", getattr(self.cfg.model, "dropout", 0.1)))
    if sa_layers > 0:
      if sa_dim <= 0:
        raise ValueError(f"occl_leaf_sa_dim must be >0, got {sa_dim}")
      if sa_heads <= 0:
        raise ValueError(f"occl_leaf_sa_heads must be >0, got {sa_heads}")
      if sa_dim % sa_heads != 0:
        raise ValueError(f"occl_leaf_sa_dim ({sa_dim}) must be divisible by occl_leaf_sa_heads ({sa_heads})")

      # Leaf token: [leaf node embedding, fruit-local pair geometry] -> d_model
      tok_in_dim = int(node_dim + pair_geom_dim)
      self.occl_leaf_tok_proj = nn.Sequential(
        nn.LayerNorm(tok_in_dim),
        nn.Linear(tok_in_dim, hdim),
        nn.ReLU(inplace=True),
        nn.Linear(hdim, sa_dim),
      )
      # Direction bias projections.
      # dir_geom is [depth_norm, lateral_norm] (2 dims) computed in fruit-local frame.
      self.occl_leaf_dir_geom_proj = nn.Sequential(
        nn.LayerNorm(2),
        nn.Linear(2, hdim),
        nn.ReLU(inplace=True),
        nn.Linear(hdim, sa_dim),
      )
      # Direction id embedding -> d_model.
      self.occl_leaf_dir_emb_proj = nn.Sequential(
        nn.LayerNorm(dir_embed_dim),
        nn.Linear(dir_embed_dim, hdim),
        nn.ReLU(inplace=True),
        nn.Linear(hdim, sa_dim),
      )
      layer = nn.TransformerEncoderLayer(
        d_model=sa_dim,
        nhead=sa_heads,
        dim_feedforward=int(sa_ff_mult * sa_dim),
        dropout=float(sa_dropout),
        batch_first=True,
        activation="gelu",
      )
      self.occl_leaf_sa = nn.TransformerEncoder(layer, num_layers=sa_layers)
    else:
      self.occl_leaf_tok_proj = None
      self.occl_leaf_dir_geom_proj = None
      self.occl_leaf_dir_emb_proj = None
      self.occl_leaf_sa = None
    # Query: (fruit, dir)
    self.occl_q_proj = nn.Sequential(
      nn.LayerNorm(node_dim + dir_embed_dim),
      nn.Linear(node_dim + dir_embed_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, attn_dim),
    )
    # Separate rank query/key projections (avoid conflicting supervision between
    # continuous potential regression logits and evidential listwise ranking).
    self.occl_q_rank_proj = nn.Sequential(
      nn.LayerNorm(node_dim + dir_embed_dim),
      nn.Linear(node_dim + dir_embed_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, attn_dim),
    )
    # Key/value: (leaf, pair_geom, dir_geom[depth,lateral])
    # If leaf-set self-attention is enabled, the "leaf" input becomes the contextualized
    # leaf token of dim `sa_dim` (rather than the raw node embedding dim).
    leaf_kv_dim = int(sa_dim) if (sa_layers > 0) else int(node_dim)
    kv_in_dim = int(leaf_kv_dim + pair_geom_dim + 2)
    self.occl_k_proj = nn.Sequential(
      nn.LayerNorm(kv_in_dim),
      nn.Linear(kv_in_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, attn_dim),
    )
    self.occl_k_rank_proj = nn.Sequential(
      nn.LayerNorm(kv_in_dim),
      nn.Linear(kv_in_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, attn_dim),
    )
    self.occl_v_proj = nn.Sequential(
      nn.LayerNorm(kv_in_dim),
      nn.Linear(kv_in_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, attn_dim),
    )
    # Union: (fruit, dir, pooled leaf context)
    self.occl_union_out = nn.Sequential(
      nn.LayerNorm(node_dim + dir_embed_dim + attn_dim),
      nn.Linear(node_dim + dir_embed_dim + attn_dim, hdim),
      nn.ReLU(inplace=True),
      nn.Linear(hdim, 1),
    )

  def _occl_leaf_set_encode(
    self,
    *,
    leaf_tok: torch.Tensor,  # [M, d_model]
    pair_row: torch.Tensor,  # [M] in [0..F-1]
    F: int,
  ) -> torch.Tensor:
    """
    Run leaf-set self-attention once per group using padding+mask.

    Args:
      leaf_tok: per-token leaf embeddings for a fixed batch of groups.
      pair_row: group id for each token in [0..F-1].
      F: number of groups in this batch.

    Returns:
      leaf_tok_ctx: [M, d_model] aligned with the input ordering.
    """
    if self.occl_leaf_sa is None:
      return leaf_tok
    if leaf_tok.numel() == 0:
      return leaf_tok
    if F <= 0:
      return leaf_tok

    device = leaf_tok.device
    dtype = leaf_tok.dtype

    # Pack variable-length leaf sets into [F, Lmax, d_model].
    counts = torch.bincount(pair_row, minlength=F)  # [F]
    Lmax = int(counts.max().item())
    if Lmax <= 0:
      return leaf_tok

    # Sort by group id so each group's leaves become contiguous.
    order = torch.argsort(pair_row)
    row_s = pair_row[order]
    tok_s = leaf_tok[order]

    # Position within each group block in the sorted list.
    idx = torch.arange(row_s.numel(), device=device, dtype=torch.long)
    starts = counts.cumsum(0) - counts  # [F]
    pos_s = idx - starts[row_s]  # [M] in [0..count(row)-1]

    # [F, Lmax] padding mask: True means "ignore"/padding.
    tok_pad = torch.zeros((F, Lmax, tok_s.size(-1)), device=device, dtype=dtype)
    pad_mask = torch.ones((F, Lmax), device=device, dtype=torch.bool)
    tok_pad[row_s, pos_s] = tok_s
    pad_mask[row_s, pos_s] = False

    # Avoid all-masked rows (can lead to NaNs in attention); safe because we never
    # gather outputs for fruits with count==0 (no pairs).
    empty_rows = counts == 0
    if bool(empty_rows.any()):
      pad_mask[empty_rows, 0] = False

    tok_pad_ctx = self.occl_leaf_sa(tok_pad, src_key_padding_mask=pad_mask)  # [F, Lmax, d_model]
    tok_ctx_s = tok_pad_ctx[row_s, pos_s]  # [M, d_model]

    # Unsort back to original per-pair order.
    tok_ctx = torch.empty_like(leaf_tok)
    tok_ctx[order] = tok_ctx_s
    return tok_ctx

  def _occl_score_pairs_and_union_attn(
    self,
    *,
    h_node: torch.Tensor,          # [N, node_dim]
    fruit_nodes: torch.Tensor,     # [F] node indices (order matches union targets)
    pair_index: torch.Tensor,      # [2, M] (fruit_node_idx, leaf_node_idx)
    pair_geom: torch.Tensor,       # [M, 11] fruit-local normalized geometry (no dir)
    dir_geom: torch.Tensor,        # [M, D, 2] = [depth_n, lateral_n]
    attn_temp: float,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direction-conditioned cross-attention style scoring.

    Returns:
      pair_potential_logits: [M, D]
      pair_rank_logits: [M, D]
      union_logits: [F, D]
    """
    device = h_node.device
    dtype = h_node.dtype
    D = int(self.occl_num_dirs)
    if fruit_nodes.numel() == 0:
      pair_pot_logits = torch.empty((pair_index.size(1), D), device=device, dtype=dtype)
      pair_rank_logits = torch.empty((pair_index.size(1), D), device=device, dtype=dtype)
      union_logits = torch.empty((0, D), device=device, dtype=dtype)
      return pair_pot_logits, pair_rank_logits, union_logits

    if (
      self.occl_dir_embed is None
      or self.occl_q_proj is None
      or self.occl_q_rank_proj is None
      or self.occl_k_proj is None
      or self.occl_k_rank_proj is None
      or self.occl_v_proj is None
      or self.occl_union_out is None
    ):
      pair_pot_logits = torch.empty((pair_index.size(1), D), device=device, dtype=dtype)
      pair_rank_logits = torch.empty((pair_index.size(1), D), device=device, dtype=dtype)
      union_logits = torch.empty((fruit_nodes.size(0), D), device=device, dtype=dtype)
      return pair_pot_logits, pair_rank_logits, union_logits

    F = int(fruit_nodes.numel())

    # Map fruit node indices -> rows in [0..F-1].
    row_of_node = torch.full((h_node.size(0),), -1, device=device, dtype=torch.long)
    row_of_node[fruit_nodes] = torch.arange(F, device=device, dtype=torch.long)
    pair_fruit = pair_index[0] if pair_index.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
    pair_leaf = pair_index[1] if pair_index.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
    pair_row = row_of_node[pair_fruit]  # [M]
    valid = pair_row >= 0

    # Direction embeddings.
    dir_ids = torch.arange(D, device=device, dtype=torch.long)
    dir_emb = self.occl_dir_embed(dir_ids).to(device=device, dtype=dtype)  # [D, E]

    # Queries (fruit, dir).
    h_fruit = h_node[fruit_nodes]  # [F, node_dim]
    q_in = torch.cat(
      [
        h_fruit.unsqueeze(1).expand(F, D, h_fruit.size(-1)),
        dir_emb.unsqueeze(0).expand(F, D, dir_emb.size(-1)),
      ],
      dim=-1,
    )
    q = self.occl_q_proj(q_in)  # [F, D, A]
    q_rank = self.occl_q_rank_proj(q_in)  # [F, D, A]

    # If there are no candidate leaf pairs, fall back to fruit+dir only (context=0).
    if pair_index.numel() == 0:
      A = int(q.size(-1))
      context = torch.zeros((F, D, A), device=device, dtype=dtype)
      union_in = torch.cat(
        [
          h_fruit.unsqueeze(1).expand(F, D, h_fruit.size(-1)),
          dir_emb.unsqueeze(0).expand(F, D, dir_emb.size(-1)),
          context,
        ],
        dim=-1,
      )
      union_logits = self.occl_union_out(union_in).squeeze(-1)  # [F,D]
      pair_pot_logits = torch.empty((0, D), device=device, dtype=dtype)
      pair_rank_logits = torch.empty((0, D), device=device, dtype=dtype)
      return pair_pot_logits, pair_rank_logits, union_logits

    M = int(pair_index.size(1))
    if not bool(valid.any()):
      pair_pot_logits = torch.empty((M, D), device=device, dtype=dtype)
      pair_rank_logits = torch.empty((M, D), device=device, dtype=dtype)
      union_logits = torch.empty((F, D), device=device, dtype=dtype)
      return pair_pot_logits, pair_rank_logits, union_logits

    # Keys/values (leaf, pair_geom, dir_geom).
    h_leaf = h_node[pair_leaf]  # [M, node_dim]
    # Optional leaf-set self-attention: contextualize leaf tokens within each fruit's candidate set.
    # Direction-aware, vectorized over (fruit,dir):
    #   tok = proj_base([h_leaf, pair_geom]) + proj_dir(dir_geom) + proj_emb(dir_emb)
    if (
      self.occl_leaf_tok_proj is not None
      and self.occl_leaf_dir_geom_proj is not None
      and self.occl_leaf_dir_emb_proj is not None
      and self.occl_leaf_sa is not None
    ):
      v_idx = torch.nonzero(valid).squeeze(-1)
      if int(v_idx.numel()) > 0:
        Mv = int(v_idx.numel())
        tok_in = torch.cat([h_leaf[v_idx], pair_geom[v_idx]], dim=-1)
        tok_base = self.occl_leaf_tok_proj(tok_in)  # [Mv, d_model]
        d_model = int(tok_base.size(-1))

        mode = str(getattr(self, "occl_leaf_sa_mode", "per_fruit_dir"))
        if mode == "per_fruit":
          # Direction-independent SA: run once per fruit, then share across directions.
          tok_ctx = self._occl_leaf_set_encode(leaf_tok=tok_base, pair_row=pair_row[v_idx], F=F)  # [Mv, d_model]
          leaf_feat_d = h_leaf.new_zeros((M, D, d_model))
          leaf_feat_d[v_idx] = tok_ctx.unsqueeze(1).expand(Mv, D, d_model).to(dtype=dtype)
        else:
          # Direction-aware SA: additive conditioning and group by (fruit, dir) => F*D groups.
          # dir_geom: [Mv, D, 2] -> [Mv, D, d_model]
          dir_bias = self.occl_leaf_dir_geom_proj(dir_geom[v_idx].reshape(Mv * D, -1)).reshape(Mv, D, d_model)
          # dir_emb: [D, E] -> [D, d_model]
          emb_bias = self.occl_leaf_dir_emb_proj(dir_emb).unsqueeze(0)  # [1, D, d_model]

          tok_dir = tok_base.unsqueeze(1) + dir_bias + emb_bias  # [Mv, D, d_model]
          # Group id for each token: row_dir in [0..F*D-1].
          row_dir = pair_row[v_idx].unsqueeze(1) * D + dir_ids.view(1, D)  # [Mv, D]

          tok_dir_flat = tok_dir.reshape(Mv * D, d_model)    # [Mv*D, d_model]
          row_dir_flat = row_dir.reshape(Mv * D)             # [Mv*D]
          tok_ctx_flat = self._occl_leaf_set_encode(leaf_tok=tok_dir_flat, pair_row=row_dir_flat, F=F * D)
          tok_ctx = tok_ctx_flat.reshape(Mv, D, d_model)     # [Mv, D, d_model]

          leaf_feat_d = h_leaf.new_zeros((M, D, d_model))
          leaf_feat_d[v_idx] = tok_ctx.to(dtype=dtype)
      else:
        # Should not happen when `valid.any()` is True, but keep dims consistent if it does.
        d_model = int(self.occl_leaf_dir_emb_proj[-1].out_features)
        leaf_feat_d = h_leaf.new_zeros((M, D, d_model))
    else:
      leaf_feat_d = h_leaf.unsqueeze(1).expand(M, D, h_leaf.size(-1))

    pair_geom_d = pair_geom.unsqueeze(1).expand(M, D, pair_geom.size(-1))
    kv_in = torch.cat([leaf_feat_d, pair_geom_d, dir_geom.to(dtype=dtype)], dim=-1)  # [M, D, kv_in_dim]
    k = self.occl_k_proj(kv_in)  # [M, D, A]
    v = self.occl_v_proj(kv_in)  # [M, D, A]
    k_rank = self.occl_k_rank_proj(kv_in)  # [M, D, A]

    # Pair logits via scaled dot product.
    A = int(q.size(-1))
    scale = float(A) ** -0.5
    q_pair = q[pair_row.clamp(min=0)]  # [M, D, A] (safe even if invalid; masked out below)
    pair_pot_logits = (q_pair * k).sum(dim=-1) * float(scale)  # [M, D]
    q_pair_rank = q_rank[pair_row.clamp(min=0)]  # [M, D, A]
    pair_rank_logits = (q_pair_rank * k_rank).sum(dim=-1) * float(scale)  # [M, D]
    if bool((~valid).any()):
      pair_pot_logits = pair_pot_logits.masked_fill((~valid).unsqueeze(-1), 0.0)
      pair_rank_logits = pair_rank_logits.masked_fill((~valid).unsqueeze(-1), 0.0)

    # Union logits via attention pooling over leaves per fruit and direction.
    union_logits = torch.empty((F, D), device=device, dtype=dtype)
    temp = float(max(attn_temp, 1e-6))
    for r in range(F):
      m = valid & (pair_row == r)
      if not bool(m.any()):
        context = torch.zeros((D, A), device=device, dtype=dtype)
      else:
        logits_r = pair_pot_logits[m]  # [L, D]
        v_r = v[m]                 # [L, D, A]
        w = torch.softmax(logits_r / temp, dim=0)  # [L, D]
        context = (w.unsqueeze(-1) * v_r).sum(dim=0)  # [D, A]

      union_in = torch.cat(
        [
          h_fruit[r].unsqueeze(0).expand(D, h_fruit.size(-1)),
          dir_emb,
          context,
        ],
        dim=-1,
      )
      union_logits[r] = self.occl_union_out(union_in).squeeze(-1)

    return pair_pot_logits, pair_rank_logits, union_logits

  def _compute_fruit_local_frames(
    self,
    pred_c: torch.Tensor,
    batch: torch.Tensor,
    node_logits: torch.Tensor,
    fruit_nodes: torch.Tensor,
  ) -> torch.Tensor:
    """
    Compute fruit-local frames for each fruit node index in `fruit_nodes`.

    Convention matches the z-buffer target generation:
      - +X points toward the nearest stem in the XY plane (z component forced to 0)
      - +Z is world up
      - +Y completes a right-handed frame

    Returns:
      R: [F,3,3] rotation matrices with rows (x,y,z), mapping world->local: p_local = R @ p_world.
    """
    device = pred_c.device
    dtype = pred_c.dtype
    F = int(fruit_nodes.numel())
    if F == 0:
      return torch.empty((0, 3, 3), device=device, dtype=dtype)

    # Identify stems from predicted node types (near-perfect in our runs; avoids GT leak).
    with torch.no_grad():
      pred_lbl = node_logits.argmax(dim=-1)
    STEM = 0
    stem_nodes = torch.nonzero(pred_lbl == STEM).squeeze(-1)

    # Default to identity frame if no stems are available in this subgraph.
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(F, 1, 1)
    if stem_nodes.numel() == 0:
      return R

    fruit_pos = pred_c[fruit_nodes]      # [F,3]
    fruit_batch = batch[fruit_nodes]     # [F]
    stem_pos = pred_c[stem_nodes]        # [S,3]
    stem_batch = batch[stem_nodes]       # [S]

    z_unit = pred_c.new_tensor([0.0, 0.0, 1.0]).view(1, 3)
    x_fallback = pred_c.new_tensor([1.0, 0.0, 0.0]).view(1, 3)
    y_fallback = pred_c.new_tensor([0.0, 1.0, 0.0]).view(1, 3)

    for b in torch.unique(fruit_batch):
      f_mask = (fruit_batch == b)
      if not bool(f_mask.any()):
        continue
      s_mask = (stem_batch == b)
      if not bool(s_mask.any()):
        continue

      fp_xy = fruit_pos[f_mask][:, :2]  # [Fb,2]
      sp_xy = stem_pos[s_mask][:, :2]   # [Sb,2]
      d_xy = torch.cdist(fp_xy, sp_xy)  # [Fb,Sb]
      nn = d_xy.argmin(dim=1)           # [Fb]
      nn_stem = stem_nodes[s_mask][nn]  # [Fb]

      x_vec = pred_c[nn_stem] - fruit_pos[f_mask]  # [Fb,3]
      x_vec = torch.stack([x_vec[:, 0], x_vec[:, 1], torch.zeros_like(x_vec[:, 0])], dim=-1)  # z=0
      nx = torch.norm(x_vec, dim=-1, keepdim=True)
      x_unit = x_vec / (nx + 1e-12)
      x_unit = torch.where(nx < 1e-6, x_fallback.expand_as(x_unit), x_unit)

      z = z_unit.expand_as(x_unit)
      y_vec = torch.cross(z, x_unit, dim=-1)
      ny = torch.norm(y_vec, dim=-1, keepdim=True)
      y_unit = y_vec / (ny + 1e-12)
      y_unit = torch.where(ny < 1e-6, y_fallback.expand_as(y_unit), y_unit)

      # Re-orthogonalize x to avoid drift.
      x_unit = torch.cross(y_unit, z, dim=-1)
      x_unit = x_unit / (torch.norm(x_unit, dim=-1, keepdim=True) + 1e-12)

      R[f_mask] = torch.stack([x_unit, y_unit, z], dim=1)  # rows
    return R

  @staticmethod
  def _compute_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    pos: [N, 3]
    edge_index: [2, E]
    returns edge_attr: [E, 4] = [rel(3), dist(1)]
    """
    if edge_index.numel() == 0:
        return torch.empty((0, 4), device=pos.device)
    src = edge_index[0]
    dst = edge_index[1]
    rel = pos[src] - pos[dst]
    dist = torch.norm(rel, dim=-1, keepdim=True)
    return torch.cat([rel, dist], dim=-1)

  def _forward_occl(
    self,
    data,
    *,
    h_node: torch.Tensor,          # [N, node_dim]
    node_logits: torch.Tensor,     # [N, C]
    centroid_offset: torch.Tensor, # [N, 3]
    extent_raw: torch.Tensor,      # [N, 3] (point-derived)
    radius_raw: torch.Tensor,      # [N] (point-derived RMS radius)
    cluster_mean: torch.Tensor,    # [N, 3] (point-derived or json_geom centroid)
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute occlusion logits (fruit union per dir, leaf-fruit potential per dir) for this batch.

    Returns:
      occl_fruit_union_logits: [F, D] (F matches data.occl_fruit_idx)
      occl_pair_potential_logits: [M, D] (M matches data.occl_pair_index)
      occl_pair_rank_logits: [M, D] (M matches data.occl_pair_index; used for listwise rank loss)
    """
    device = h_node.device
    D = int(self.occl_num_dirs)

    fruit_nodes = getattr(data, "occl_fruit_idx", None)
    if fruit_nodes is None or int(fruit_nodes.numel()) == 0:
      return (
        torch.empty((0, D), device=device),
        torch.empty((0, D), device=device),
        torch.empty((0, D), device=device),
      )

    pair_index = getattr(data, "occl_pair_index", None)
    if pair_index is None:
      pair_index = torch.empty((2, 0), device=device, dtype=torch.long)

    # If the attention modules are not present (or occl is disabled), emit empty tensors.
    if not self.occl_enabled or self.occl_dir_embed is None:
      return (
        torch.empty((int(fruit_nodes.numel()), D), device=device),
        torch.empty((int(pair_index.size(1)), D), device=device),
        torch.empty((int(pair_index.size(1)), D), device=device),
      )

    # Build fruit-local relative features for each (fruit, leaf) candidate pair.
    pair_geom = torch.empty((int(pair_index.size(1)), 11), device=device, dtype=h_node.dtype)
    dir_geom = torch.empty((int(pair_index.size(1)), D, 2), device=device, dtype=h_node.dtype)
    if pair_index.numel() > 0:
      fruit_idx = pair_index[0]
      leaf_idx = pair_index[1]

      # In no-geom-injection mode, cluster_mean is derived from points; otherwise fall back to json_geom.
      base_c = cluster_mean if self.no_geom_injection else data.json_geom[:, :3]
      pred_c = base_c + centroid_offset  # [N,3]
      rel_world = pred_c[leaf_idx] - pred_c[fruit_idx]  # [M,3]

      # Fruit-local frame (+X toward nearest stem in XY, +Z up).
      fruit_nodes_unique = torch.unique(fruit_idx)
      R_fruit = self._compute_fruit_local_frames(pred_c, data.batch, node_logits, fruit_nodes_unique)  # [Fu,3,3]
      row_of_node = torch.full((pred_c.size(0),), -1, device=device, dtype=torch.long)
      row_of_node[fruit_nodes_unique] = torch.arange(fruit_nodes_unique.numel(), device=device, dtype=torch.long)
      R_pairs = R_fruit[row_of_node[fruit_idx]]  # [M,3,3]

      rel_local = torch.bmm(R_pairs, rel_world.unsqueeze(-1)).squeeze(-1)  # [M,3]

      # Normalize by fruit size (RMS radius) to reduce scale sensitivity.
      fruit_rad = radius_raw[fruit_idx].clamp(min=1e-6)
      leaf_rad = radius_raw[leaf_idx].clamp(min=1e-6)
      scale = fruit_rad.unsqueeze(-1)

      rel_local_n = rel_local / (scale + 1e-12)
      dist_n = torch.norm(rel_local_n, dim=-1, keepdim=True)
      leaf_rad_ratio = (leaf_rad / (fruit_rad + 1e-12)).unsqueeze(-1)
      fruit_ext_n = extent_raw[fruit_idx] / (scale + 1e-12)
      leaf_ext_n = extent_raw[leaf_idx] / (scale + 1e-12)

      pair_geom = torch.cat([rel_local_n, dist_n, leaf_rad_ratio, fruit_ext_n, leaf_ext_n], dim=-1)  # [M,11]

      # Direction-conditioned depth/lateral cues (fruit-local).
      dirs = self.occl_dirs.to(device=device, dtype=h_node.dtype)  # [D,3]
      depth_n = rel_local_n @ dirs.T  # [M,D]
      rel_n2 = (rel_local_n.pow(2).sum(dim=-1, keepdim=True))  # [M,1]
      lateral2_n = rel_n2 - depth_n.pow(2)
      lateral_n = torch.sqrt(torch.clamp(lateral2_n, min=0.0) + 1e-12)  # [M,D]
      dir_geom = torch.stack([depth_n, lateral_n], dim=-1)  # [M,D,2]

    # Optional ablation: drop fruit-local pair geometry features (keep shapes stable).
    pair_geom_mode = str(getattr(self.cfg.model, "occl_pair_geom_mode", "full")).strip().lower()
    if pair_geom_mode == "none":
      pair_geom = torch.zeros_like(pair_geom)
    elif pair_geom_mode != "full":
      raise ValueError(
        f"Unknown occl_pair_geom_mode: {pair_geom_mode!r} (expected 'full' or 'none')"
      )

    attn_temp = float(getattr(self.cfg.model, "occl_attn_temp", 1.0))
    occl_pair_pot_logits, occl_pair_rank_logits, occl_union_logits = self._occl_score_pairs_and_union_attn(
      h_node=h_node,
      fruit_nodes=fruit_nodes.to(device=device),
      pair_index=pair_index.to(device=device),
      pair_geom=pair_geom,
      dir_geom=dir_geom,
      attn_temp=attn_temp,
    )

    return occl_union_logits, occl_pair_pot_logits, occl_pair_rank_logits

  def build_candidates(self, pos: torch.Tensor, batch: torch.Tensor, node_logits: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated candidate edge generation.
    pos: [N, 3] (Use cluster_mean for Blind Oracle)
    node_logits: [N, node_classes] (Use predicted logits to avoid GT label leak)
    """
    k = getattr(self.cfg.model, 'k_candidates', 8)
    r = getattr(self.cfg.model, 'dist_threshold', 0.3)
    extra_stems = getattr(self.cfg.model, 'extra_stems_per_node', 2)
    
    # Deriving labels from logits to ensure zero GT-label cheat in blind mode
    node_labels = node_logits.argmax(dim=-1)
    
    # 1. Radius and k-NN Graphs
    e_radius = radius_graph(pos, r=r, batch=batch, loop=False, max_num_neighbors=64)
    e_knn = knn_graph(pos, k=k, batch=batch, loop=False)
    
    all_edges = [e_radius, e_knn]
    
    # 2. Extra Stems (XY-only search) using PREDICTED labels
    if extra_stems > 0:
        stem_idx = 0 # Stem(1) -> 0 mapping in Oracle dataset
        is_stem = (node_labels == stem_idx)
        is_non_stem = ~is_stem
        
        if is_stem.any() and is_non_stem.any():
            pos_xy = pos[:, :2]
            for b in torch.unique(batch):
                b_mask = (batch == b)
                b_stems = b_mask & is_stem
                b_non_stems = b_mask & is_non_stem
                
                if b_stems.any() and b_non_stems.any():
                    s_idx = torch.nonzero(b_stems).squeeze(-1)
                    n_idx = torch.nonzero(b_non_stems).squeeze(-1)
                    
                    dists = torch.cdist(pos_xy[n_idx], pos_xy[s_idx])
                    num_k = min(extra_stems, s_idx.size(0))
                    nn_topk = torch.topk(dists, num_k, dim=1, largest=False).indices
                    
                    src_nodes = n_idx.repeat_interleave(num_k)
                    dst_nodes = s_idx[nn_topk.flatten()]
                    all_edges.append(torch.stack([src_nodes, dst_nodes], dim=0))
    
    combined = torch.cat(all_edges, dim=1)
    combined_sorted = torch.sort(combined, dim=0).values
    unique_undir = torch.unique(combined_sorted, dim=1)
    
    rev = torch.stack([unique_undir[1], unique_undir[0]], dim=0)
    final = torch.cat([unique_undir, rev], dim=1)
    return torch.unique(final, dim=1)

  def _extract_node_samples(self, pos, norm, color, node_id, num_nodes, node_classes, npoint=512, class_dropout_rates=None):
    """
    Vectorized GPU-only cluster extraction.
    Extracts 'npoint' from each instance in the subgraph.
    """
    # 1. Sort to group points by node for vectorized indexing
    perm = torch.argsort(node_id)
    pos = pos[perm]
    norm = norm[perm]
    color = color[perm]
    node_id = node_id[perm]
    
    # 2. Count points per node and find offsets
    counts = torch.bincount(node_id, minlength=num_nodes)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=pos.device), counts.cumsum(0)[:-1]])
    
    # 3. Build vectorized indices: [num_nodes * npoint]
    # We take the first 'npoint' or wrap around if not enough points
    sample_idx_local = torch.arange(npoint, device=pos.device).repeat(num_nodes)
    # Ensure wrap-around for small instances
    valid_counts = counts.repeat_interleave(npoint).clamp(min=1)
    sample_idx_local = sample_idx_local % valid_counts
    
    # Add base offsets
    global_idx = sample_idx_local + offsets.repeat_interleave(npoint)
    # [num_nodes, npoint] indices
    node_pt_idx = global_idx.view(num_nodes, npoint)

    # 4. Targetted Post-Sampling Dropout (drops and heavily duplicates remaining)
    if self.training and class_dropout_rates is not None:
        if isinstance(class_dropout_rates, list):
            class_dropout_rates = torch.tensor(class_dropout_rates, device=pos.device, dtype=torch.float32)
            
        drop_probs = class_dropout_rates[node_classes] # [num_nodes]
        mask_drop = drop_probs > 0.0
        
        if mask_drop.any():
            drop_nodes = torch.nonzero(mask_drop).squeeze(-1)
            for n_idx in drop_nodes:
                p = drop_probs[n_idx].item()
                num_keep = max(1, int(npoint * (1.0 - p)))
                num_drop = npoint - num_keep
                
                if num_drop <= 0:
                    continue
                    
                # The points we currently have for this node
                pts = node_pt_idx[n_idx].clone()
                
                # Shuffle the points
                shuf = torch.randperm(npoint, device=pos.device)
                pts = pts[shuf]
                
                kept_pts = pts[:num_keep]
                
                # duplicate the kept points to fill the gap
                dup_idx = torch.randint(0, num_keep, (num_drop,), device=pos.device)
                dup_pts = kept_pts[dup_idx]
                
                # Replace the original row
                node_pt_idx[n_idx] = torch.cat([kept_pts, dup_pts])
                
    global_idx_final = node_pt_idx.view(-1)

    return pos[global_idx_final].view(num_nodes, npoint, 3), \
           norm[global_idx_final].view(num_nodes, npoint, 3), \
           color[global_idx_final].view(num_nodes, npoint, 3)

  def forward(self, data):
    """
    data must contain:
      pos_subgraph: [TotalPoints, 3]
      norm_subgraph: [TotalPoints, 3]
      node_id_per_point: [TotalPoints]
      json_geom:  [N, 6] centroid(3), extent(3) in world coords
      num_nodes:  int (total nodes in batch)
    """
    npoint = getattr(self.cfg.model, "oracle_msg_npoint", 512)
    class_dropout_rates = getattr(self.cfg.model, "class_dropout_rates", None)
    
    # 1. GPU-Accelerated Extraction
    x_organ, n_organ, c_organ = self._extract_node_samples(
        data.pos_subgraph, data.norm_subgraph, data.color_subgraph,
        data.node_id_per_point, data.y_node.size(0), 
        node_classes=data.y_node,
        npoint=npoint, class_dropout_rates=class_dropout_rates
    )
    
    # 2. Local Centering (GPU)
    # If no_geom_injection is False, we use GT centroids (The "Cheat" Centering).
    # If True, we use the mean of the cluster points (The "Zero-Cheat" Centering).
    if not self.no_geom_injection:
        cluster_mean = data.json_geom[:, :3]
    else:
        cluster_mean = x_organ.mean(dim=1) # [N, 3]

    x_organ = x_organ - cluster_mean[:, None, :3]

    # Size cues derived from the centered point clusters (available in both cheat and blind modes).
    # These are used for occlusion scoring (leaf/fruit scale matters a lot for visibility).
    extent_raw = x_organ.max(dim=1).values - x_organ.min(dim=1).values  # [N,3]
    radius_raw = torch.sqrt((x_organ.pow(2).sum(dim=-1)).mean(dim=1).clamp(min=0.0) + 1e-12)  # [N]
    
    identity = self.shared_encoder(x_organ, n_organ, c_organ)  # [N, hdim]
    local_node_logits = self.local_node_head(identity)  # [N, node_classes]
    
    # --- Semantic Feature Injection ---
    sem_feats = None
    if self.node_feat_mode == 'gt':
        # One-hot GT
        sem_feats = F.one_hot(data.y_node, num_classes=local_node_logits.shape[-1]).float()
    elif self.node_feat_mode == 'local':
        # Probability distribution from the shared encoder (local head)
        # We can stop gradient on these if we want the GNN to learn to correct them without affecting the head
        sem_feats = torch.softmax(local_node_logits, dim=-1)

    # Build node features
    feats_to_cat = [identity]
    if not self.no_geom_injection:
        feats_to_cat.append(data.json_geom) # centroid(3) + extent(3)
        
    if sem_feats is not None:
        feats_to_cat.append(sem_feats)
        
    node_feats = torch.cat(feats_to_cat, dim=-1)
    
    # 3. Backbone & Candidates (Truly Blind path)
    if self.no_geom_injection:
        pos = cluster_mean
        graph_logits = local_node_logits
    else:
        pos = data.json_geom[:, :3]
        # One-hot GT as logits to maintain legacy GT connectivity behavior
        graph_logits = F.one_hot(data.y_node, num_classes=local_node_logits.shape[-1]).float()

    edge_index = self.build_candidates(pos, data.batch, graph_logits)
    edge_attr = self._compute_edge_attr(pos, edge_index)   # [E, 4]
    h = self.backbone(node_feats, edge_index, edge_attr=edge_attr)  # [N, hdim]

    node_logits = self.node_head(h)

    src = edge_index[0]
    dst = edge_index[1]
    h_edge = torch.cat([h[src], h[dst], edge_attr], dim=-1)

    edge_exist_logits = self.edge_exist_head(h_edge)  # [E, 1]
    edge_label_logits = self.edge_label_head(h_edge)  # [E, edge_labels]

    centroid_offset = self.centroid_offset_head(h)    # [N, 3]
    extent_delta = self.extent_delta_head(h)          # [N, 3]

    occl_fruit_union_logits, occl_pair_potential_logits, occl_pair_rank_logits = self._forward_occl(
      data,
      h_node=h,
      node_logits=node_logits,
      centroid_offset=centroid_offset,
      extent_raw=extent_raw,
      radius_raw=radius_raw,
      cluster_mean=cluster_mean,
    )

    return {
      "edge_index": edge_index,
      "local_node_logits": local_node_logits,
      "node_logits": node_logits,
      "edge_exist_logits": edge_exist_logits,
      "edge_label_logits": edge_label_logits,
      "centroid_offset": centroid_offset,
      "extent_delta": extent_delta,
      "cluster_mean": cluster_mean if self.no_geom_injection else None,
      "occl_fruit_union_logits": occl_fruit_union_logits,
      "occl_pair_potential_logits": occl_pair_potential_logits,
      "occl_pair_rank_logits": occl_pair_rank_logits,
    }


class OracleResBaseline(OracleGTBaseline):
  """
  [V12 Elite] Oracle baseline with:
    - ResGINEBackbone (Unique layer parameters + Skip connections)
    - Identity Concatenation (Heads see both GNN context and original PointNet features)
  """
  def __init__(self, cfg, node_classes: int, edge_labels: int):
    super().__init__(cfg, node_classes, edge_labels)
    hdim = cfg.model.hidden_dim
    edge_dim = 4

    # Override Backbone with Residual version
    in_dim = hdim
    if not self.no_geom_injection:
        in_dim += 6 # centroid(3) + extent(3)
        
    if self.node_feat_mode in ['gt', 'local']:
        in_dim += node_classes
        
    self.backbone = ResGINEBackbone(
      in_dim=in_dim,
      hidden_dim=hdim,
      num_layers=cfg.model.num_layers,
      dropout=cfg.model.dropout
    )

    # Override heads to look at [h_refined, identity]
    self.node_head = nn.Linear(hdim * 2, node_classes)
    self.centroid_offset_head = nn.Linear(hdim * 2, 3)
    self.extent_delta_head = nn.Linear(hdim * 2, 3)

    # Rebuild occlusion heads for the new node embedding dim (hdim*2).
    if self.occl_enabled:
      self._build_occl_heads(node_dim=hdim * 2)

  def forward(self, data):
    npoint = getattr(self.cfg.model, "oracle_msg_npoint", 512)
    class_dropout_rates = getattr(self.cfg.model, "class_dropout_rates", None)
    
    # 1. GPU-Accelerated Extraction & Local Head
    x_organ, n_organ, c_organ = self._extract_node_samples(
        data.pos_subgraph, data.norm_subgraph, data.color_subgraph,
        data.node_id_per_point, data.y_node.size(0), 
        node_classes=data.y_node,
        npoint=npoint, class_dropout_rates=class_dropout_rates
    )
    
    if not self.no_geom_injection:
        cluster_mean = data.json_geom[:, :3]
    else:
        cluster_mean = x_organ.mean(dim=1)

    x_organ = x_organ - cluster_mean[:, None, :3]

    # Size cues from point clusters (see base class forward for details).
    extent_raw = x_organ.max(dim=1).values - x_organ.min(dim=1).values  # [N,3]
    radius_raw = torch.sqrt((x_organ.pow(2).sum(dim=-1)).mean(dim=1).clamp(min=0.0) + 1e-12)  # [N]
    
    identity = self.shared_encoder(x_organ, n_organ, c_organ)
    local_node_logits = self.local_node_head(identity)
    
    # 2. Semantic Injection
    sem_feats = None
    if self.node_feat_mode == 'gt':
        sem_feats = F.one_hot(data.y_node, num_classes=local_node_logits.shape[-1]).float()
    elif self.node_feat_mode == 'local':
        sem_feats = torch.softmax(local_node_logits, dim=-1)

    # Build node features
    feats_to_cat = [identity]
    if not self.no_geom_injection:
        feats_to_cat.append(data.json_geom) # centroid(3) + extent(3)
        
    if sem_feats is not None:
        feats_to_cat.append(sem_feats)
        
    node_feats = torch.cat(feats_to_cat, dim=-1)
    
    # 3. Backbone & Candidates (Truly Blind path)
    if self.no_geom_injection:
        pos = cluster_mean
        graph_logits = local_node_logits
    else:
        pos = data.json_geom[:, :3]
        graph_logits = F.one_hot(data.y_node, num_classes=local_node_logits.shape[-1]).float()

    edge_index = self.build_candidates(pos, data.batch, graph_logits)
    edge_attr = self._compute_edge_attr(pos, edge_index)
    
    h = self.backbone(node_feats, edge_index, edge_attr=edge_attr)
    
    # [V12 Elite] Identity Concatenation
    h_combined = torch.cat([h, identity], dim=-1)

    node_logits = self.node_head(h_combined)
    
    # 4. Final Predictions
    src, dst = edge_index[0], edge_index[1]
    h_edge = torch.cat([h[src], h[dst], edge_attr], dim=-1)

    centroid_offset = self.centroid_offset_head(h_combined)
    extent_delta = self.extent_delta_head(h_combined)

    # Occlusion feature-source ablation:
    # - combined: [h_gnn, identity] (default)
    # - gnn_only: [h_gnn, 0]
    # - identity_only: [0, identity]
    occl_src = str(getattr(self.cfg.model, "occl_node_feat_source", "combined")).strip().lower()
    if occl_src == "combined":
      h_occl = h_combined
    elif occl_src == "gnn_only":
      h_occl = torch.cat([h, torch.zeros_like(identity)], dim=-1)
    elif occl_src == "identity_only":
      h_occl = torch.cat([torch.zeros_like(h), identity], dim=-1)
    else:
      raise ValueError(
        f"Unknown occl_node_feat_source: {occl_src!r} "
        "(expected 'combined', 'gnn_only', or 'identity_only')"
      )

    occl_fruit_union_logits, occl_pair_potential_logits, occl_pair_rank_logits = self._forward_occl(
      data,
      h_node=h_occl,
      node_logits=node_logits,
      centroid_offset=centroid_offset,
      extent_raw=extent_raw,
      radius_raw=radius_raw,
      cluster_mean=cluster_mean,
    )

    return {
      "edge_index": edge_index,
      "local_node_logits": local_node_logits,
      "node_logits": node_logits,
      "edge_exist_logits": self.edge_exist_head(h_edge),
      "edge_label_logits": self.edge_label_head(h_edge),
      "centroid_offset": centroid_offset,
      "extent_delta": extent_delta,
      "cluster_mean": cluster_mean if self.no_geom_injection else None,
      "occl_fruit_union_logits": occl_fruit_union_logits,
      "occl_pair_potential_logits": occl_pair_potential_logits,
      "occl_pair_rank_logits": occl_pair_rank_logits,
    }
