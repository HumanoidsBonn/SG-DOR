# pepper_scene_graphs/config.py

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class TrainCfg:
  """Training hyperparameters."""
  epochs: int = 100  #: Number of training epochs
  batch_size: int = 32  #: Batch size for training/eval
  lr: float = 4e-3  #: Adam optimizer learning rate
  k: int = 8  #: Number of neighbors for initial k-NN graph
  device: str = "cuda"  #: Computation device (cuda or cpu)
  hidden_dim: int = 128  #: Hidden dimension for GNN and MLPs
  num_layers: int = 4  #: Number of GNN layers
  dropout: float = 0.1  #: Dropout probability
  val_ratio: float = 0.2  #: Fraction of data used for validation
  seed: int = 0  #: Random seed for reproducibility


@dataclass
class FeatureCfg:
  """Flags to enable/disable specific node features."""
  use_centroid: bool = True  #: 3D centroid of the organ
  use_axes: bool = True  #: Principal axes of the OBB
  use_extents: bool = True  #: Spatial extents of the OBB
  use_orientation: bool = True  #: Main growth orientation vector
  use_color: bool = True  #: Mean HSV color data
  use_base_xy: bool = False  #: XY projection of the organ base
  use_stem_size: bool = False  #: Stem-specific radius and height


@dataclass
class NoiseCfg:
  """Standard deviations for Gaussian noise augmentation."""
  centroid_std: float = 0.025  #: Noise on centroid positions
  orientation_std: float = 0.08  #: Noise on orientation vectors
  axis_std: float = 0.05  #: Noise on OBB principal axes
  extent_std: float = 0.005  #: Noise on OBB extents
  color_hue_std: float = 0.05  #: Hue noise (0-1)
  color_sat_std: float = 0.1  #: Saturation noise (0-1)
  color_val_std: float = 0.15  #: Value/Brightness noise (0-1)
  shadow_shift_range: Tuple[float, float] = (-0.1, 0.1)  #: Global brightness shift


@dataclass
class RotAugCfg:
  """Rotation augmentation configuration."""
  apply_rot: bool = True  #: Enable random rotation augmentation
  yaw_deg: Tuple[float, float] = (0.0, 360.0)  #: Random yaw range (degrees)
  pitch_deg: Tuple[float, float] = (-3.0, 3.0)  #: Random pitch range (degrees)
  roll_deg: Tuple[float, float] = (-3.0, 3.0)  #: Random roll range (degrees)


@dataclass
class DropAugCfg:
  """Node/edge dropout augmentation configuration."""
  apply_drop: bool = True  #: Enable stochastic node/edge dropout
  node_drop_prob: float = 0.08  #: Probability to drop a node
  edge_drop_prob: float = 0.15  #: Probability to drop a GT edge
  protect_stems: bool = True  #: Prevent stems from being dropped


@dataclass
class StemKnnCfg:
  """Extra stem-neighbor connections added to the k-NN graph."""
  extra_stems_per_node: int = 2  #: Number of nearest stems to attach per non-stem


@dataclass
class TypeConstraintCfg:
  """Biological type constraint gating for candidate edges."""
  enabled: bool = False  #: Enable type-based gating
  mode: str = "pred_soft"  # gt, pred_hard, pred_soft
  tau: float = 0.25  #: Threshold for pred_soft gating
  tau_schedule_enabled: bool = True  #: Enable scheduled tau values
  tau_values: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25])  #: Tau schedule values
  tau_interval_epochs: int = 5  #: Epochs per tau value when scheduling
  detach_node_probs: bool = True  #: Stop gradients through node-prob gating
  allowed_pairs: List[List[str]] = field(default_factory=lambda: [
    ["leaf", "stem"],
    ["peduncle", "stem"],
    ["fruit", "peduncle"],
  ])


@dataclass
class EdgeCandidateCfg:
  """Candidate edge construction configuration."""
  dist_threshold: float = 0.15  #: Radius threshold (meters) for candidate edges
  radius_max_neighbors: int = 64  #: Max neighbors in radius graph construction
  type_constraint: TypeConstraintCfg = field(default_factory=TypeConstraintCfg)


@dataclass
class EdgeFeatureCfg:
  """Feature options for edge heads."""
  stem_distance_mode: str = "pred"  # pred, gt


@dataclass
class HeadsCfg:
  """Enable/disable prediction heads (loss and metrics)."""
  node: bool = True  #: Node classification head
  edge_exist: bool = True  #: Edge existence head
  edge_label: bool = True  #: Edge relation head
  occlusion: bool = False  #: Leaf-fruit occlusion head


@dataclass
class LossWeightsCfg:
  """Relative weights for each task loss."""
  node: float = 0.4  #: Weight on global/refined node classification loss
  node_local: float = 1.0  #: Weight on local pre-GNN node classification loss
  edge_exist: float = 1.5  #: Weight on edge existence loss
  edge_label: float = 0.05  #: Weight on edge relation loss
  occlusion: float = 2.0  #: Weight on occlusion classification loss


@dataclass
class CurriculumCfg:
  """Oracle-style staged curriculum for multi-task optimization."""
  enabled: bool = True  #: Enable staged task weighting
  stage_gnn_epoch: int = 6  #: Epoch when global node + edge existence turn on
  stage_edge_label_epoch: int = 6  #: Epoch when edge label task turns on
  stage_occlusion_epoch: int = 6  #: Epoch when occlusion task turns on


@dataclass
class NodeLossCfg:
  """Class weighting for node classification."""
  weight_mode: str = "fixed"  # none, fixed
  class_weights: List[float] = field(default_factory=lambda: [3.0, 1.0, 1.0, 1.0])  #: Per-class weights


@dataclass
class EdgeExistCfg:
  """Configuration for edge existence loss and thresholding."""
  loss: str = "bce"  # bce, focal; OVERWRITTEN if evidential enabled
  weight_mode: str = "fixed"  # fixed, neg_pos
  weight_value: float = 4.0  #: Fixed positive weight (used when weight_mode=fixed)
  weight_max: float = 50.0  #: Cap on neg/pos ratio (weight_mode=neg_pos)
  pred_thresh: float = 0.5  #: Threshold for edge existence metrics
  focal_alpha: float = 0.75  #: Focal loss alpha (if enabled)
  focal_gamma: float = 1.0  #: Focal loss gamma (if enabled)
  thresh_sweep_enabled: bool = True  #: Enable eval-time threshold sweep
  sweep_min: float = 0.05  #: Sweep min threshold
  sweep_max: float = 0.95  #: Sweep max threshold
  sweep_steps: int = 19  #: Number of sweep thresholds
  sweep_metric: str = "f1"  #: Metric for picking best threshold
  neg_sampling_enabled: bool = False  #: Enable negative sampling for edge existence
  neg_to_pos: float = 11.0  #: Negatives per positive when sampling
  stem_segment_equiv: bool = True  #: Use representative segment supervision; loss ignores sibling segments of same stem


@dataclass
class EvidenceCfg:
  """Evidential deep learning configuration."""
  use_nodes: bool = True  #: Use evidential node classification
  use_edge_exist: bool = True  #: Use evidential edge existence
  use_edge_rel: bool = True  #: Use evidential edge relations
  use_occlusion: bool = True  #: Use evidential occlusion classification
  fn: str = "softplus"  # relu, exp, softplus, none
  kl_coeff: float = 0.1  #: KL regularization coefficient
  kl_warmup_enabled: bool = True  #: Enable linear KL warm-up schedule
  kl_warmup_epochs: int = 20  #: Number of warm-up epochs (0 disables warm-up)
  kl_warmup_start: float = 0.0  #: Starting KL coefficient for warm-up


@dataclass
class LoggingCfg:
  """Logging and visualization intervals."""
  edge_score_plot_interval: int = 5  #: Epoch interval for edge score plots (if enabled)


@dataclass
class CheckpointCfg:
  """Checkpoint selection and retention policy."""
  monitor: str = "val_edge_exist_best_f1"  #: Validation metric to monitor (common: val_edge_exist_f1, val_edge_exist_best_f1, val_loss, val_node_acc)
  mode: str = "max"  #: Optimization direction for monitor ("max" for F1/acc/precision/recall, "min" for loss/ece/brier)
  start_epoch: int = 1  #: Earliest epoch considered for best-checkpoint selection
  save_top_k: int = 1  #: Number of best checkpoints to keep (set 1 to keep only current best)
  save_last: bool = True  #: Whether to always save the latest checkpoint
  tie_breaker: str = "val_edge_exist_ece"  #: Secondary metric for ties (typical: val_loss or val_edge_exist_f1)


@dataclass
class ModelCfg:
  """Model architecture settings."""
  backbone: str = "res_gine"  #: GNN backbone type (graphsage, gine, res_gine)
  hidden_dim: int = 128  #: Common hidden dimension for all layers
  num_layers: int = 4  #: Number of backbone messaging layers
  dropout: float = 0.1  #: Dropout rate in model heads
  node_feat_mode: str = "local"  #: Semantic injection mode: none, local, gt
  detach_local_semantics: bool = False  #: Detach local semantic probs before injection
  node_identity_concat: bool = True  #: Concatenate raw node features with GNN output for global node head


@dataclass
class OcclusionCandidateCfg:
  """Candidate sampling configuration for occlusion learning."""
  max_dist: float = 0.15  #: Leaf-fruit max distance for base candidates
  thin_keep_prob: float = 1.0  #: Per-epoch keep prob for thinning candidates
  none_keep_prob: float = 0.25  #: Per-epoch keep prob for non-candidates (ie probability for loss to *not* be masked for directions where occlusion is below "none" threshold)
  extra_neg_ratio: float = 0.0  #: Extra negatives per base candidate
  ring_max_dist: float = 0.3  #: Upper distance for ring negatives
  ring_frac: float = 0.5  #: Fraction of extra negatives sampled from the ring


@dataclass
class OcclusionTargetCfg:
  """Occlusion target binning configuration."""
  target_mode: str = "raw"  # raw, weighted
  dir_mode: str = "dirs6"  # axes3/dirs6 or axes9/dirs18 (explicit directions)
  none_thresh: float = 0.0  #: Upper bound for "none" bin
  high_thresh: float = 0.15  #: Lower bound for "high" bin
  loss: str = "ce"  # ce, focal; ignored if evidential occlusion enabled
  weight_mode: str = "fixed"  #: Class weighting for occlusion loss (none, fixed)
  class_weights: list[float] = field(default_factory=lambda: [1, 1.0, 1.2])  #: Class weights for 3-bin targets
  focal_alpha: float = 0.25  #: Focal loss alpha (if enabled)
  focal_gamma: float = 2.0  #: Focal loss gamma (if enabled)


@dataclass
class OcclusionHeadCfg:
  """Occlusion head and geometry embedding configuration."""
  geom_hidden_dim: int = 256  #: Hidden dimension for geometry MLP
  geom_layers: int = 4  #: Number of layers in geometry MLP
  head_hidden_dim: int = 256  #: Hidden dimension for occlusion head MLP
  head_layers: int = 4  #: Number of layers in occlusion head MLP


@dataclass
class OcclusionCfg:
  """Configuration for occlusion prediction."""
  enabled: bool = False  #: Enable occlusion prediction head
  candidates: OcclusionCandidateCfg = field(default_factory=OcclusionCandidateCfg)
  targets: OcclusionTargetCfg = field(default_factory=OcclusionTargetCfg)
  head: OcclusionHeadCfg = field(default_factory=OcclusionHeadCfg)


@dataclass
class Config:
  """Top-level configuration container."""
  train: TrainCfg = field(default_factory=TrainCfg)
  features: FeatureCfg = field(default_factory=FeatureCfg)
  noise: NoiseCfg = field(default_factory=NoiseCfg)
  rot_aug: RotAugCfg = field(default_factory=RotAugCfg)
  drop_aug: DropAugCfg = field(default_factory=DropAugCfg)
  stem_knn: StemKnnCfg = field(default_factory=StemKnnCfg)
  edge_candidates: EdgeCandidateCfg = field(default_factory=EdgeCandidateCfg)
  edge_features: EdgeFeatureCfg = field(default_factory=EdgeFeatureCfg)
  heads: HeadsCfg = field(default_factory=HeadsCfg)
  loss_weights: LossWeightsCfg = field(default_factory=LossWeightsCfg)
  curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
  node_loss: NodeLossCfg = field(default_factory=NodeLossCfg)
  edge_exist: EdgeExistCfg = field(default_factory=EdgeExistCfg)
  evidence: EvidenceCfg = field(default_factory=EvidenceCfg)
  logging: LoggingCfg = field(default_factory=LoggingCfg)
  checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
  model: ModelCfg = field(default_factory=ModelCfg)
  occlusion: OcclusionCfg = field(default_factory=OcclusionCfg)
