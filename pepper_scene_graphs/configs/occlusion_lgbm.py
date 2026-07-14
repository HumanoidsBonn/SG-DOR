"""LightGBM baseline configuration for occlusion prediction."""

from dataclasses import dataclass, field
from typing import Optional

from pepper_scene_graphs.configs.config import Config


@dataclass
class LGBMCandidateCfg:
  """Candidate selection for LightGBM baseline."""
  max_dist: float = 0.15  #: Max leaf-fruit distance for candidates


@dataclass
class LGBMTargetCfg:
  """Target binning for LightGBM baseline."""
  target_mode: str = "raw"  #: raw or weighted occlusion overlaps
  dir_mode: str = "dirs6"  #: dirs6 or dirs18 (explicit direction bins)
  none_thresh: float = 0.015  #: Upper bound for "none" bin
  high_thresh: float = 0.15  #: Lower bound for "high" bin


@dataclass
class LGBMFeatureCfg:
  """Feature flags for LightGBM baseline."""
  include_dist_ratio: bool = True  #: Append dist_ratio feature (fruit-leaf distance / fruit radius; radius is a mean half-extent proxy)
  include_corner_uv: bool = True  #: Append projected corner (u,v) features per direction
  corner_uv_norm: str = "fruit_radius"  #: corner uv normalization (fruit_radius)


@dataclass
class LGBMSplitCfg:
  """Train/val split options for LightGBM baseline."""
  val_ratio: float = 0.2  #: Fraction of scenes held out for validation
  seed: int = 0  #: RNG seed for scene split
  max_scenes: int = 0  #: Limit number of scenes (0 = all)


@dataclass
class LGBMWeightCfg:
  """Class weighting for LightGBM baseline."""
  """ONLY used for single-stage models."""
  weight_mode: str = "fixed"  #: inverse or fixed
  fixed_class_weights: list[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])  #: Used when weight_mode=fixed
  weight_cap: float = 50.0  #: Maximum class weight after inverse-frequency scaling


@dataclass
class LGBMParamsCfg:
  """LightGBM training hyperparameters."""
  learning_rate: float = 0.05
  num_leaves: int = 255
  min_data_in_leaf: int = 200
  feature_fraction: float = 0.8
  bagging_fraction: float = 0.8
  bagging_freq: int = 1
  max_bin: int = 255
  num_boost_round: int = 8000
  early_stopping_rounds: int = 200


@dataclass
class LGBMEvalCfg:
  """Optional periodic eval metrics on validation subset."""
  eval_interval: int = 0  #: Iteration interval for extra metrics (0 disables)
  eval_subsample: int = 100000  #: Subsample size for eval metrics (0 = full val set)
  eval_seed: int = 0  #: RNG seed for subsample selection


@dataclass
class LGBMTuningCfg:
  """Threshold tuning configuration."""
  enabled: bool = True  #: Enable threshold tuning sweep
  t0_min: float = 0.2  #: Min threshold for none class (single-stage)
  t0_max: float = 0.95  #: Max threshold for none class (single-stage)
  t2_min: float = 0.05  #: Min threshold for high class (single-stage)
  t2_max: float = 0.95  #: Max threshold for high class (single-stage)
  t_any_min: Optional[float] = 0.1  #: Min threshold for any-vs-none (two-stage)
  t_any_max: Optional[float] = 0.9  #: Max threshold for any-vs-none (two-stage)
  t_hi_min: Optional[float] = 0.1  #: Min threshold for high-vs-low (two-stage)
  t_hi_max: Optional[float] = 0.9  #: Max threshold for high-vs-low (two-stage)
  step: float = 0.03  #: Step size for threshold sweep
  p_min: float = 0.5  #: Minimum precision for recall-constrained objective
  refine_enabled: bool = True  #: Run a refinement sweep around the coarse best thresholds
  refine_window: float = 0.12  #: +/- window around coarse best for refinement
  refine_step: float = 0.01  #: Step size for refinement sweep


@dataclass
class LGBMDeadZoneCfg:
  """Dead-zone filtering for Model A (any vs none)."""
  enabled: bool = True  #: Enable dead-zone filtering for Model A only
  dead_lo: float = 0.0  #: Lower bound for dead zone
  dead_hi: float = 0.05  #: Upper bound for dead zone
  mode: str = "downweight"  #: drop or downweight
  downweight_factor: float = 0.05  #: Weight multiplier for downweight mode


@dataclass
class LGBMTwoStageCfg:
  """Two-stage (any -> high/low) configuration."""
  enabled: bool = True  #: Enable two-stage training
  any_weight_cap: float = 10.0  #: Cap for pos_weight in any-vs-none classifier
  high_weight_cap: float = 10.0  #: Cap for pos_weight in high-vs-low classifier
  any_min_data_in_leaf: Optional[int] = 50  #: Override min_data_in_leaf for Model A
  high_min_data_in_leaf: Optional[int] = 100  #: Override min_data_in_leaf for Model B


@dataclass
class OcclusionLGBMCfg:
  """Top-level LightGBM baseline config."""
  run_name: str = "lgbm_occlusion"
  output_root: str = "training"
  wandb: bool = False
  dataset_path: Optional[str] = None
  candidates: LGBMCandidateCfg = field(default_factory=LGBMCandidateCfg)
  targets: LGBMTargetCfg = field(default_factory=LGBMTargetCfg)
  features: LGBMFeatureCfg = field(default_factory=LGBMFeatureCfg)
  split: LGBMSplitCfg = field(default_factory=LGBMSplitCfg)
  weights: LGBMWeightCfg = field(default_factory=LGBMWeightCfg)
  params: LGBMParamsCfg = field(default_factory=LGBMParamsCfg)
  eval: LGBMEvalCfg = field(default_factory=LGBMEvalCfg)
  tuning: LGBMTuningCfg = field(default_factory=LGBMTuningCfg)
  two_stage: LGBMTwoStageCfg = field(default_factory=LGBMTwoStageCfg)
  dead_zone: LGBMDeadZoneCfg = field(default_factory=LGBMDeadZoneCfg)


def build_data_cfg(cfg: OcclusionLGBMCfg) -> Config:
  """Build a Config instance for dataset loading."""
  data_cfg = Config()
  data_cfg.occlusion.enabled = True
  data_cfg.occlusion.candidates.max_dist = float(cfg.candidates.max_dist)
  # Avoid building ring/far pools for this baseline.
  data_cfg.occlusion.candidates.ring_max_dist = float(cfg.candidates.max_dist)
  data_cfg.occlusion.targets.dir_mode = str(cfg.targets.dir_mode)
  data_cfg.occlusion.targets.target_mode = str(cfg.targets.target_mode)
  return data_cfg
