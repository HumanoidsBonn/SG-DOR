# pepper_scene_graphs/models/backbones.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINEConv

from pepper_scene_graphs.models.registry import register_backbone


class BackboneBase(nn.Module):
  """Abstract base class for GNN backbones."""

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
    """Forward pass of the backbone.

    Args:
        x: Node feature matrix of shape (num_nodes, in_dim).
        edge_index: Edge indices of shape (2, num_edges).
        edge_attr: Optional edge attributes/features.

    Returns:
        Encoded node features of shape (num_nodes, hidden_dim).
    """
    raise NotImplementedError


@register_backbone("graphsage")
class GraphSAGEBackbone(BackboneBase):
  """GraphSAGE backbone utilizing SAGEConv layers."""

  def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    """Initialize the GraphSAGE backbone.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_layers: Number of GraphSAGE layers.
        dropout: Dropout probability.
    """
    super().__init__()
    self.convs = nn.ModuleList()
    d = in_dim
    for _ in range(num_layers):
      self.convs.append(SAGEConv(d, hidden_dim))
      d = hidden_dim
    self.dropout = float(dropout)

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
    h = x
    for conv in self.convs:
      h = conv(h, edge_index)
      h = F.relu(h)
      if self.dropout > 0:
        h = F.dropout(h, p=self.dropout, training=self.training)
    return h


@register_backbone("gine")
class GINEBackbone(BackboneBase):
  """GINE backbone utilizing GINEConv (Graph Isomorphism Network with Edge features)."""

  def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    """Initialize the GINE backbone.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_layers: Number of GINE layers.
        dropout: Dropout probability.
    """
    super().__init__()
    self.convs = nn.ModuleList()
    self.mlp = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
    )
    self.input_proj = nn.Linear(in_dim, hidden_dim)
    # Project 4-dim geometric edge features to hidden_dim
    self.edge_proj = nn.Linear(4, hidden_dim)

    for _ in range(num_layers):
      self.convs.append(GINEConv(nn=self.mlp))
    self.dropout = float(dropout)

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
    h = self.input_proj(x)

    if edge_attr is not None:
      # Project edge features to match node hidden dim for GINE summation
      edge_emb = self.edge_proj(edge_attr)
    else:
      # If no edge features provided, use zeros (though this defeats the purpose of GINE)
      edge_emb = torch.zeros((edge_index.size(1), h.size(1)), device=h.device, dtype=h.dtype)

    for conv in self.convs:
      h = conv(h, edge_index, edge_attr=edge_emb)
      h = F.relu(h)
      if self.dropout > 0:
        h = F.dropout(h, p=self.dropout, training=self.training)
    return h


@register_backbone("res_gine")
class ResGINEBackbone(BackboneBase):
  """[V12 Elite] ResGINE backbone with residual connections and layer-unique MLPs."""

  def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    super().__init__()
    self.convs = nn.ModuleList()
    self.input_proj = nn.Linear(in_dim, hidden_dim)
    self.edge_proj = nn.Linear(4, hidden_dim)

    for _ in range(num_layers):
      # Unique MLP per layer for increased capacity
      layer_mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
      )
      self.convs.append(GINEConv(nn=layer_mlp))
    self.dropout = float(dropout)

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
    h = self.input_proj(x)
    edge_emb = self.edge_proj(edge_attr) if edge_attr is not None else \
               torch.zeros((edge_index.size(1), h.size(1)), device=h.device, dtype=h.dtype)

    for conv in self.convs:
      # Residual update path
      h_next = conv(h, edge_index, edge_attr=edge_emb)
      h_next = F.relu(h_next)
      if self.dropout > 0:
        h_next = F.dropout(h_next, p=self.dropout, training=self.training)
      h = h + h_next # Skip connection
      
    return h
