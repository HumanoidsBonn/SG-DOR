# pepper_scene_graphs/models/joint_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple

from pepper_scene_graphs.constants import NODE_CLASSES, EDGE_LABELS
from pepper_scene_graphs.utils.evidence import get_evidence_fn
from pepper_scene_graphs.data.transforms import feature_slices
from pepper_scene_graphs.data.edge_builders import (
  build_candidate_edges,
  undirected_unique,
  build_allowed_type_matrix,
  stem_aware_edge_distance,
)
from pepper_scene_graphs.models.registry import make_backbone

import pepper_scene_graphs.models.backbones  # ensure registration


class JointSceneGraphModel(nn.Module):
  """Joint model for scene graph generation, predicting nodes, edges, and relations."""

  def __init__(self, in_dim: int, cfg: Any):
    """Initialize the joint model.

    Args:
        in_dim: Input feature dimension.
        cfg: Configuration object containing model and training parameters.
    """
    super().__init__()
    self.cfg = cfg
    self.ev_nodes = cfg.evidence.use_nodes
    self.ev_edge_exist = cfg.evidence.use_edge_exist
    self.ev_edge_rel = cfg.evidence.use_edge_rel
    self.evidence_fn = get_evidence_fn(cfg.evidence.fn)
    self.hidden_dim = cfg.model.hidden_dim
    self.node_feat_mode = str(getattr(cfg.model, "node_feat_mode", "local")).lower()
    self.detach_local_semantics = bool(getattr(cfg.model, "detach_local_semantics", False))
    self.node_identity_concat = bool(getattr(cfg.model, "node_identity_concat", True))

    # Oracle-style local node head from raw per-node OBB features.
    self.local_node_head = nn.Sequential(
      nn.Linear(in_dim, cfg.model.hidden_dim),
      nn.ReLU(),
      nn.Linear(cfg.model.hidden_dim, len(NODE_CLASSES)),
    )

    backbone_in_dim = in_dim + (len(NODE_CLASSES) if self.node_feat_mode in {"local", "gt"} else 0)
    self.backbone = make_backbone(
      cfg.model.backbone,
      in_dim=backbone_in_dim,
      hidden_dim=cfg.model.hidden_dim,
      num_layers=cfg.model.num_layers,
      dropout=cfg.model.dropout,
    )

    hdim = cfg.model.hidden_dim
    node_head_in_dim = hdim + (in_dim if self.node_identity_concat else 0)
    self.node_head = nn.Sequential(
      nn.Linear(node_head_in_dim, hdim),
      nn.ReLU(),
      nn.Linear(hdim, len(NODE_CLASSES)),
    )
    
    # Pre-register allowed type matrix as a persistent buffer
    # This avoids rebuilding it on CPU every forward pass
    allowed_mat = build_allowed_type_matrix(cfg.edge_candidates.type_constraint, NODE_CLASSES)
    self.register_buffer("allowed_type_mat", allowed_mat)
    
    # Decouple edge_emb_dim from hardcoded node assumption.
    # We define node_feat_dim explicitly based on what we append:
    # 1. log-probabilities (len(NODE_CLASSES)) if evidential/softmax
    # 2. Or potentially just raw logits (len(NODE_CLASSES))
    self.node_feat_dim = len(NODE_CLASSES) 
    
    edge_emb_dim = hdim * 2 + 2 * self.node_feat_dim + 1
    # Edge existence:
    # - Non-evidential: a single logit (BCEWithLogits)
    # - Evidential: still predict a single logit z, then form a tied Beta/Dirichlet
    #   alpha = [e(-z)+1, e(z)+1]. This avoids the failure mode where both class
    #   logits become very negative -> alpha≈[1,1] and gradients collapse.
    edge_exist_out_dim = 1

    self.edge_exist_head = nn.Sequential(
      nn.Linear(edge_emb_dim, hdim),
      nn.ReLU(),
      nn.Linear(hdim, hdim),
      nn.ReLU(),
      nn.Linear(hdim, edge_exist_out_dim),
    )
    self.edge_label_head = nn.Sequential(
      nn.Linear(edge_emb_dim, hdim),
      nn.ReLU(),
      nn.Linear(hdim, hdim),
      nn.ReLU(),
      nn.Linear(hdim, len(EDGE_LABELS)),
    )

    # Initialize occlusion heads (if enabled).
    self.build_occlusion_heads()

  def forward(self, data: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass of the joint model.

    Args:
        data: Input PyG data object containing `x`, `pos`, `batch`, etc.

    Returns:
        A tuple containing:
        - local_node_out: Local pre-GNN node logits.
        - node_out: Node classification logits/evidence.
        - edge_index: Filtered edge indices used for prediction.
        - edge_exist_out: Edge existence scores/evidence.
        - edge_label_out: Edge relation label scores/evidence.
        - h: Backbone node embeddings.
    """
    # 1. Build initial k-NN graph and augment with extra stem edges
    mp_edge_index = build_candidate_edges(
      data,
      self.cfg.train.k,
      self.cfg.stem_knn.extra_stems_per_node,
      self.cfg.edge_candidates.dist_threshold,
      self.cfg.edge_candidates.radius_max_neighbors,
    )
    if mp_edge_index.numel() == 0:
      mp_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)

    # 2. Compute geometric edge features (relative pos + distance) for the backbone
    # These are used by GINE to capture local structural context
    backbone_edge_attr = self._compute_backbone_edge_features(data.pos, mp_edge_index)

    # 3. Oracle-style local node prediction and semantic injection.
    local_node_out = self.local_node_head(data.x)
    sem_feats = None
    if self.node_feat_mode == "gt":
      sem_feats = F.one_hot(data.y_node, num_classes=len(NODE_CLASSES)).float()
    elif self.node_feat_mode == "local":
      sem_feats = torch.softmax(local_node_out, dim=-1)
      if self.detach_local_semantics:
        sem_feats = sem_feats.detach()

    backbone_in = torch.cat([data.x, sem_feats], dim=-1) if sem_feats is not None else data.x

    # 4. Propagate messages through the GNN backbone.
    h = self.backbone(backbone_in, mp_edge_index, edge_attr=backbone_edge_attr)

    # 5. Predict global/refined node semantics.
    node_head_in = torch.cat([h, data.x], dim=-1) if self.node_identity_concat else h
    node_out = self.node_head(node_head_in)
    if self.ev_nodes and self.evidence_fn is not None:
      # Convert logits to Dirichlet evidence parameters
      node_out = torch.clamp(self.evidence_fn(node_out), min=0.0) + 1.0

    # 5. Extract unique undirected edges for downstream relation prediction
    edge_index = undirected_unique(mp_edge_index)
    if edge_index.numel() == 0:
      return local_node_out, node_out, edge_index, torch.empty(0, device=data.x.device), torch.empty((0, len(EDGE_LABELS)), device=data.x.device), h

    # 6. Apply biological gating/type constraints to prune nonsensical candidates
    node_probs, node_edge_feat = self._node_probs_and_edge_feat(node_out)
    edge_index = self._type_gate(edge_index, data, node_probs)

    if edge_index.numel() == 0:
      return local_node_out, node_out, edge_index, torch.empty(0, device=data.x.device), torch.empty((0, len(EDGE_LABELS)), device=data.x.device), h

    # 7. Concatenate node and pairwise geometric features for edge heads
    h_src = h[edge_index[0]]
    h_dst = h[edge_index[1]]
    p_src = node_edge_feat[edge_index[0]]
    p_dst = node_edge_feat[edge_index[1]]

    # Compute stem-aware distance (2D for stems, 3D otherwise)
    dist = stem_aware_edge_distance(
      data.pos,
      edge_index,
      node_edge_feat,
      data.y_node,
      mode=self.cfg.edge_features.stem_distance_mode,
    ).unsqueeze(-1)

    h_edge = torch.cat([h_src, h_dst, p_src, p_dst, dist], dim=-1)

    # 8. Predict edge existence and relation type
    edge_exist_logits = self.edge_exist_head(h_edge).squeeze(-1)
    if self.ev_edge_exist and self.evidence_fn is not None:
      # Tied Beta/Dirichlet parameterization for binary existence:
      # class 0 (no-edge): alpha0 = e(-z) + 1
      # class 1 (edge):    alpha1 = e( z) + 1
      a0 = self.evidence_fn(-edge_exist_logits) + 1.0
      a1 = self.evidence_fn(edge_exist_logits) + 1.0
      edge_exist_out = torch.stack([a0, a1], dim=-1)
    else:
      edge_exist_out = edge_exist_logits

    edge_label_out = self.edge_label_head(h_edge)
    if self.ev_edge_rel and self.evidence_fn is not None:
      edge_label_out = torch.clamp(self.evidence_fn(edge_label_out), min=0.0) + 1.0

    return local_node_out, node_out, edge_index, edge_exist_out, edge_label_out, h

  def _node_probs_and_edge_feat(self, node_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized node probabilities and features for edge prediction.
    
    Handles both standard softmax and evidential (Dirichlet) outputs.

    Args:
        node_out: Raw output from node head.

    Returns:
        Tuple of (probabilities, feature_representation).
    """
    if self.ev_nodes and self.evidence_fn is not None:
      alpha = node_out
      probs = alpha / alpha.sum(dim=-1, keepdim=True)
      edge_feat = torch.log(probs + 1e-8)
    else:
      probs = node_out.softmax(dim=-1)
      edge_feat = node_out
    tc = self.cfg.edge_candidates.type_constraint
    if tc.detach_node_probs:
      probs = probs.detach()
      edge_feat = edge_feat.detach()
    return probs, edge_feat

  def _type_gate(self, edge_index: torch.Tensor, data: Any, node_probs: torch.Tensor) -> torch.Tensor:
    """Filter candidate edges based on biological type constraints.

    Uses `allowed_type_matrix` to prune edges between incompatible node types.

    Args:
        edge_index: Candidate edge indices.
        data: PyG data object (for GT labels if mode='gt').
        node_probs: Predicted node probabilities (for mode='pred').

    Returns:
        Filtered edge_index.
    """
    tc = self.cfg.edge_candidates.type_constraint
    if not tc.enabled or edge_index.numel() == 0:
      return edge_index

    if not hasattr(self, "allowed_type_mat"):
       # Fallback for legacy loading if buffer missed
       A = build_allowed_type_matrix(tc, NODE_CLASSES).to(edge_index.device)
    else:
       A = self.allowed_type_mat
    
    mode = str(tc.mode).lower()

    if mode == "gt":
      t_u = data.y_node[edge_index[0]]
      t_v = data.y_node[edge_index[1]]
      mask = A[t_u, t_v]
      return edge_index[:, mask]

    if mode == "pred_hard":
      pred_t = node_probs.argmax(dim=-1)
      t_u = pred_t[edge_index[0]]
      t_v = pred_t[edge_index[1]]
      mask = A[t_u, t_v]
      return edge_index[:, mask]

    if mode == "pred_soft":
      pu = node_probs[edge_index[0]]
      pv = node_probs[edge_index[1]]
      p_allow = ((pu @ A.float()) * pv).sum(dim=-1)
      mask = p_allow >= float(tc.tau)
      return edge_index[:, mask]

    raise ValueError(f"Unknown type_constraint.mode: {tc.mode}")

  def _compute_backbone_edge_features(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute geometric features for backbone edges.

    Features: [relative_x, relative_y, relative_z, euclidean_distance]

    Args:
        pos: Node positions (num_nodes, 3).
        edge_index: Edge indices (2, num_edges).

    Returns:
        Edge attributes tensor of shape (num_edges, 4).
    """
    if edge_index.numel() == 0:
      return torch.empty((0, 4), device=pos.device, dtype=pos.dtype)

    src, dst = edge_index[0], edge_index[1]
    rel_pos = pos[src] - pos[dst]  # (num_edges, 3)
    dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (num_edges, 1)

    return torch.cat([rel_pos, dist], dim=-1)

  def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int) -> nn.Sequential:
    """Construct a simple MLP with ReLU activations."""
    if layers <= 1:
      return nn.Sequential(nn.Linear(in_dim, out_dim))
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(layers - 2):
      mods.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    mods.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*mods)

  def _occlusion_dir_count(self) -> int:
    """Return the number of occlusion directions (explicit, not collapsed)."""
    mode = str(self.cfg.occlusion.targets.dir_mode).lower()
    if mode in ("axes3", "dirs6"):
      return 6
    if mode in ("axes9", "dirs18"):
      return 18
    raise ValueError(f"Unknown occlusion dir_mode: {self.cfg.occlusion.targets.dir_mode}")

  def _fruit_local_frame(self, fruit_center: torch.Tensor, stem_center: torch.Tensor) -> torch.Tensor:
    """Compute fruit-local frames for a batch of fruits.

    +X points from fruit to stem in XY, +Z is world up, +Y completes RHS.
    """
    x_vec = stem_center - fruit_center
    x_vec = torch.stack([x_vec[:, 0], x_vec[:, 1], torch.zeros_like(x_vec[:, 0])], dim=1)
    x_norm = torch.norm(x_vec, dim=1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=x_vec.device, dtype=x_vec.dtype).unsqueeze(0)
    x_unit = torch.where(x_norm > 1e-6, x_vec / (x_norm + 1e-8), fallback)
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=x_vec.device, dtype=x_vec.dtype).unsqueeze(0).repeat(x_vec.size(0), 1)
    y_vec = torch.cross(z_unit, x_unit, dim=1)
    y_norm = torch.norm(y_vec, dim=1, keepdim=True)
    y_fallback = torch.tensor([0.0, 1.0, 0.0], device=x_vec.device, dtype=x_vec.dtype).unsqueeze(0)
    y_unit = torch.where(y_norm > 1e-6, y_vec / (y_norm + 1e-8), y_fallback)
    x_unit = torch.cross(y_unit, z_unit, dim=1)
    x_unit = x_unit / (torch.norm(x_unit, dim=1, keepdim=True) + 1e-8)
    return torch.stack([x_unit, y_unit, z_unit], dim=1)

  @staticmethod
  def _build_occlusion_dir_bases() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 18 canonical directions and their orthonormal (u, v) bases."""
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
          v = base[i] * s1 + base[j] * s2
          v = v / (torch.norm(v) + 1e-8)
          dirs.append(v)
    dirs_t = torch.stack(dirs, dim=0)

    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    us = []
    vs = []
    for d in dirs_t:
      d = d / (torch.norm(d) + 1e-8)
      tmp = z if abs(torch.dot(d, z)) < 0.9 else x
      u = torch.cross(tmp, d, dim=0)
      if torch.norm(u) < 1e-6:
        tmp = x
        u = torch.cross(tmp, d, dim=0)
      u = u / (torch.norm(u) + 1e-8)
      v = torch.cross(d, u, dim=0)
      v = v / (torch.norm(v) + 1e-8)
      us.append(u)
      vs.append(v)
    return dirs_t, torch.stack(us, dim=0), torch.stack(vs, dim=0)

  @staticmethod
  def _obb_radii(
    axes_local: torch.Tensor,
    half_extents: torch.Tensor,
    dirs: torch.Tensor,
  ) -> torch.Tensor:
    """Compute OBB radius along each direction in dirs.

    Args:
        axes_local: (N, 3, 3) OBB axes in local frame (columns are axis vectors).
        half_extents: (N, 3) half-lengths along each axis.
        dirs: (D, 3) unit directions.

    Returns:
        Radii tensor of shape (N, D).
    """
    axes_t = axes_local.transpose(1, 2)  # (N, 3, 3) axis vectors
    dots = torch.einsum("nij,dj->nid", axes_t, dirs)  # (N, 3, D)
    dots = dots.permute(0, 2, 1).abs()  # (N, D, 3)
    return (dots * half_extents.unsqueeze(1)).sum(dim=-1)

  def _occlusion_geom_features(
    self,
    data: Any,
    leaf_idx: torch.Tensor,
    fruit_idx: torch.Tensor,
    stem_idx: torch.Tensor,
  ) -> torch.Tensor:
    """Compute direction-wise geometry features for occlusion prediction."""
    sl = feature_slices(self.cfg.features)
    if "axes" not in sl or "extents" not in sl:
      raise ValueError("Occlusion features require axes and extents to be enabled in config.")
    a0, a1 = sl["axes"]
    e0, e1 = sl["extents"]

    leaf_center = data.pos[leaf_idx]
    fruit_center = data.pos[fruit_idx]
    stem_center = data.pos[stem_idx]
    rel = leaf_center - fruit_center

    R = self._fruit_local_frame(fruit_center, stem_center)  # (N, 3, 3)
    rel_local = torch.bmm(R, rel.unsqueeze(-1)).squeeze(-1)

    leaf_axes = data.x[leaf_idx, a0:a1].view(-1, 3, 3)
    fruit_axes = data.x[fruit_idx, a0:a1].view(-1, 3, 3)
    leaf_axes_local = torch.bmm(R, leaf_axes)
    fruit_axes_local = torch.bmm(R, fruit_axes)

    leaf_extents = 0.5 * data.x[leaf_idx, e0:e1]
    fruit_extents = 0.5 * data.x[fruit_idx, e0:e1]

    num_dirs = self._occlusion_dir_count()
    dirs = self.occl_dirs[:num_dirs]
    u_dirs = self.occl_us[:num_dirs]
    v_dirs = self.occl_vs[:num_dirs]

    c_d = rel_local @ dirs.t()  # (N, D)
    o_u = rel_local @ u_dirs.t()
    o_v = rel_local @ v_dirs.t()

    leaf_r_d = self._obb_radii(leaf_axes_local, leaf_extents, dirs)
    leaf_r_u = self._obb_radii(leaf_axes_local, leaf_extents, u_dirs)
    leaf_r_v = self._obb_radii(leaf_axes_local, leaf_extents, v_dirs)
    fruit_r_d = self._obb_radii(fruit_axes_local, fruit_extents, dirs)
    fruit_r_u = self._obb_radii(fruit_axes_local, fruit_extents, u_dirs)
    fruit_r_v = self._obb_radii(fruit_axes_local, fruit_extents, v_dirs)

    gap = (c_d - leaf_r_d) - fruit_r_d
    eps = 1e-6
    gap_norm = gap / (fruit_r_d + eps)
    o_u_norm = o_u / (fruit_r_u + eps)
    o_v_norm = o_v / (fruit_r_v + eps)
    leaf_u_norm = leaf_r_u / (fruit_r_u + eps)
    leaf_v_norm = leaf_r_v / (fruit_r_v + eps)

    feat_norm = torch.stack([gap_norm, o_u_norm, o_v_norm, leaf_u_norm, leaf_v_norm], dim=-1)
    feat_raw = torch.stack([fruit_r_u, fruit_r_v, fruit_r_d, leaf_r_u, leaf_r_v, leaf_r_d], dim=-1)
    return torch.cat([feat_norm, feat_raw], dim=-1)

  def build_occlusion_heads(self) -> None:
    """Initialize occlusion geometry embedder and head."""
    if not self.cfg.occlusion.enabled:
      self.occl_geom_mlp = None
      self.occl_head = None
      return

    if not hasattr(self, "occl_dirs"):
      dirs, us, vs = self._build_occlusion_dir_bases()
      self.register_buffer("occl_dirs", dirs)
      self.register_buffer("occl_us", us)
      self.register_buffer("occl_vs", vs)

    geom_in_dim = 11  # gap/offset/size norms + raw radii (per direction)
    geom_hidden = int(self.cfg.occlusion.head.geom_hidden_dim)
    geom_layers = int(self.cfg.occlusion.head.geom_layers)
    head_hidden = int(self.cfg.occlusion.head.head_hidden_dim)
    head_layers = int(self.cfg.occlusion.head.head_layers)
    num_bins = 3

    self.occl_geom_mlp = self._build_mlp(geom_in_dim, geom_hidden, geom_hidden, geom_layers)
    head_in_dim = geom_hidden
    self.occl_head = self._build_mlp(head_in_dim, head_hidden, num_bins, head_layers)

  def occlusion_forward(
    self,
    data: Any,
    h: torch.Tensor,
    occl_pairs: torch.Tensor,
    stem_indices: torch.Tensor,
  ) -> torch.Tensor:
    """Compute occlusion logits/evidence for leaf-fruit candidate pairs."""
    if not self.cfg.occlusion.enabled or occl_pairs.numel() == 0:
      return torch.empty((0, 0, 0), device=h.device)
    if self.occl_geom_mlp is None or self.occl_head is None:
      self.build_occlusion_heads()

    leaf_idx = occl_pairs[:, 0]
    fruit_idx = occl_pairs[:, 1]
    geom = self._occlusion_geom_features(data, leaf_idx, fruit_idx, stem_indices)
    num_dirs = self._occlusion_dir_count()
    num_pairs = geom.size(0)
    geom_embed = self.occl_geom_mlp(geom.view(-1, geom.size(-1))).view(num_pairs, num_dirs, -1)

    logits = self.occl_head(geom_embed.view(-1, geom_embed.size(-1)))
    logits = logits.view(-1, num_dirs, 3)
    if self.cfg.evidence.use_occlusion and self.evidence_fn is not None:
      logits = self.evidence_fn(logits) + 1.0
    return logits
