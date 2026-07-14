#!/usr/bin/env python3
"""Train a LightGBM baseline for leaf-fruit occlusion prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
  import lightgbm as lgb
except ImportError as exc:
  raise SystemExit("LightGBM is not installed. Add it to environment.yml and reinstall.") from exc

try:
  import wandb
except ImportError:
  wandb = None
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

from pepper_scene_graphs.configs.occlusion_lgbm import OcclusionLGBMCfg, build_data_cfg
from pepper_scene_graphs.data.dataset import load_graph_dataset
from pepper_scene_graphs.data.transforms import feature_slices
from pepper_scene_graphs.training.occlusion import occlusion_dir_indices, occlusion_dir_labels, _fruit_stem_indices


def parse_args() -> argparse.Namespace:
  """Parse CLI args for LightGBM baseline."""
  parser = argparse.ArgumentParser(description="Train LightGBM occlusion baseline.")
  parser.add_argument("--dataset-path", required=True, help="Path to dataset (.json or .pt)")
  parser.add_argument("--run-name", default="lgbm_occlusion", help="Run name suffix")
  parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
  return parser.parse_args()


def _build_occlusion_dir_bases() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Build 18 canonical directions and (u, v) bases (dataset order)."""
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


def _fruit_local_frame(fruit_center: torch.Tensor, stem_center: torch.Tensor) -> torch.Tensor:
  """Compute fruit-local frames (same convention as model)."""
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


def _obb_radii(axes_local: torch.Tensor, half_extents: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
  """Compute OBB radius along each direction in dirs."""
  axes_t = axes_local.transpose(1, 2)
  dots = torch.einsum("nij,dj->nid", axes_t, dirs)
  dots = dots.permute(0, 2, 1).abs()
  return (dots * half_extents.unsqueeze(1)).sum(dim=-1)


def _occlusion_geom_features(
  data: Any,
  leaf_idx: torch.Tensor,
  fruit_idx: torch.Tensor,
  stem_idx: torch.Tensor,
  dir_idx: List[int],
  include_dist_ratio: bool,
  include_corner_uv: bool,
  corner_uv_norm: str,
  feature_cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute per-direction geometry features for occlusion baseline."""
  sl = feature_slices(feature_cfg)
  if "axes" not in sl or "extents" not in sl:
    raise ValueError("Occlusion features require axes and extents to be enabled.")
  a0, a1 = sl["axes"]
  e0, e1 = sl["extents"]

  leaf_center = data.pos[leaf_idx]
  fruit_center = data.pos[fruit_idx]
  stem_center = data.pos[stem_idx]
  rel = leaf_center - fruit_center

  R = _fruit_local_frame(fruit_center, stem_center)
  rel_local = torch.bmm(R, rel.unsqueeze(-1)).squeeze(-1)

  leaf_axes = data.x[leaf_idx, a0:a1].view(-1, 3, 3)
  fruit_axes = data.x[fruit_idx, a0:a1].view(-1, 3, 3)
  leaf_axes_local = torch.bmm(R, leaf_axes)
  fruit_axes_local = torch.bmm(R, fruit_axes)

  leaf_extents = 0.5 * data.x[leaf_idx, e0:e1]
  fruit_extents = 0.5 * data.x[fruit_idx, e0:e1]

  dirs = _OCCL_DIRS[dir_idx]
  u_dirs = _OCCL_US[dir_idx]
  v_dirs = _OCCL_VS[dir_idx]

  c_d = rel_local @ dirs.t()
  o_u = rel_local @ u_dirs.t()
  o_v = rel_local @ v_dirs.t()

  leaf_r_d = _obb_radii(leaf_axes_local, leaf_extents, dirs)
  leaf_r_u = _obb_radii(leaf_axes_local, leaf_extents, u_dirs)
  leaf_r_v = _obb_radii(leaf_axes_local, leaf_extents, v_dirs)
  fruit_r_d = _obb_radii(fruit_axes_local, fruit_extents, dirs)
  fruit_r_u = _obb_radii(fruit_axes_local, fruit_extents, u_dirs)
  fruit_r_v = _obb_radii(fruit_axes_local, fruit_extents, v_dirs)

  gap = (c_d - leaf_r_d) - fruit_r_d
  eps = 1e-6
  gap_norm = gap / (fruit_r_d + eps)
  o_u_norm = o_u / (fruit_r_u + eps)
  o_v_norm = o_v / (fruit_r_v + eps)
  leaf_u_norm = leaf_r_u / (fruit_r_u + eps)
  leaf_v_norm = leaf_r_v / (fruit_r_v + eps)

  feat_norm = torch.stack([gap_norm, o_u_norm, o_v_norm, leaf_u_norm, leaf_v_norm], dim=-1)
  feat_raw = torch.stack([fruit_r_u, fruit_r_v, fruit_r_d, leaf_r_u, leaf_r_v, leaf_r_d], dim=-1)
  feats = torch.cat([feat_norm, feat_raw], dim=-1)

  # Approximate 2D silhouette overlap (axis-aligned in u,v plane).
  left = torch.maximum(-fruit_r_u, o_u - leaf_r_u)
  right = torch.minimum(fruit_r_u, o_u + leaf_r_u)
  bottom = torch.maximum(-fruit_r_v, o_v - leaf_r_v)
  top = torch.minimum(fruit_r_v, o_v + leaf_r_v)
  w = torch.clamp(right - left, min=0.0)
  h = torch.clamp(top - bottom, min=0.0)
  inter = w * h
  fruit_area = (2.0 * fruit_r_u) * (2.0 * fruit_r_v)
  leaf_area = (2.0 * leaf_r_u) * (2.0 * leaf_r_v)
  inter_norm = inter / (fruit_area + eps)
  overlap_flag = (inter > 0).to(inter.dtype)
  iou = inter / (fruit_area + leaf_area - inter + eps)

  overlap_feats = torch.stack([inter_norm, overlap_flag, iou], dim=-1)
  feats = torch.cat([feats, overlap_feats], dim=-1)

  if include_corner_uv:
    # Project OBB corners into (u,v) plane and normalize by fruit scale.
    signs = torch.tensor(
      [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0,  1.0,  1.0],
      ],
      device=feats.device,
      dtype=feats.dtype,
    )
    leaf_half = leaf_extents
    fruit_half = fruit_extents
    leaf_offsets = signs[None, :, :] * leaf_half[:, None, :]
    fruit_offsets = signs[None, :, :] * fruit_half[:, None, :]
    # Corners in fruit-local frame.
    leaf_corners = rel_local[:, None, :] + torch.einsum("nij,nkj->nki", leaf_axes_local, leaf_offsets)
    fruit_corners = torch.einsum("nij,nkj->nki", fruit_axes_local, fruit_offsets)
    # Project to u,v for each direction: N x 8 x D
    leaf_u = torch.einsum("nkd,md->nkm", leaf_corners, u_dirs)
    leaf_v = torch.einsum("nkd,md->nkm", leaf_corners, v_dirs)
    fruit_u = torch.einsum("nkd,md->nkm", fruit_corners, u_dirs)
    fruit_v = torch.einsum("nkd,md->nkm", fruit_corners, v_dirs)
    # Reorder to N x D x 8 x 2, then flatten to N x D x 16
    leaf_uv = torch.stack([leaf_u, leaf_v], dim=-1).permute(0, 2, 1, 3).reshape(leaf_u.size(0), leaf_u.size(2), -1)
    fruit_uv = torch.stack([fruit_u, fruit_v], dim=-1).permute(0, 2, 1, 3).reshape(fruit_u.size(0), fruit_u.size(2), -1)
    if corner_uv_norm.lower() == "fruit_radius":
      scale = torch.norm(fruit_extents, dim=-1, keepdim=True)
      leaf_uv = leaf_uv / (scale[:, None, :] + eps)
      fruit_uv = fruit_uv / (scale[:, None, :] + eps)
    else:
      raise ValueError(f"Unknown corner_uv_norm: {corner_uv_norm}")

    feats = torch.cat([feats, fruit_uv, leaf_uv], dim=-1)

  if include_dist_ratio:
    dist = torch.norm(rel, dim=-1)
    fruit_radius = fruit_extents.mean(dim=-1)
    dist_ratio = dist / (fruit_radius + eps)
    dist_ratio = dist_ratio[:, None].repeat(1, feats.size(1))
    feats = torch.cat([feats, dist_ratio.unsqueeze(-1)], dim=-1)
  return feats, inter_norm


_OCCL_DIRS, _OCCL_US, _OCCL_VS = _build_occlusion_dir_bases()


def _bin_targets(values: torch.Tensor, none_t: float, high_t: float) -> torch.Tensor:
  """Bin raw overlap values into 3 classes."""
  bins = torch.zeros_like(values, dtype=torch.long)
  bins = torch.where(values > none_t, torch.ones_like(bins), bins)
  bins = torch.where(values >= high_t, torch.full_like(bins, 2), bins)
  return bins


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
  """Compute macro precision/recall/F1 and accuracy."""
  conf = np.zeros((3, 3), dtype=np.int64)
  np.add.at(conf, (y_true, y_pred), 1)
  tp = np.diag(conf)
  fp = conf.sum(axis=0) - tp
  fn = conf.sum(axis=1) - tp
  prec = np.divide(tp, tp + fp + 1e-8)
  rec = np.divide(tp, tp + fn + 1e-8)
  f1 = np.divide(2 * prec * rec, prec + rec + 1e-8)
  return {
    "accuracy": float(tp.sum() / max(1, conf.sum())),
    "precision_macro": float(np.mean(prec)),
    "recall_macro": float(np.mean(rec)),
    "f1_macro": float(np.mean(f1)),
  }


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
  """Compute binary metrics for non-none vs none."""
  y_true_bin = (y_true > 0).astype(np.int64)
  y_pred_bin = (y_pred > 0).astype(np.int64)
  tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
  fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
  fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
  prec = tp / (tp + fp + 1e-8)
  rec = tp / (tp + fn + 1e-8)
  f1 = 2 * prec * rec / (prec + rec + 1e-8)
  return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
  """Compute a 3x3 confusion matrix."""
  conf = np.zeros((3, 3), dtype=np.int64)
  np.add.at(conf, (y_true, y_pred), 1)
  return conf


def _plot_eval_curve(curve_path: Path, out_path: Path) -> None:
  """Plot eval-curve metrics from JSONL."""
  if plt is None or not curve_path.exists():
    return
  records = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
  if not records:
    return
  iters = [r["iter"] for r in records]
  macro = [r["macro"]["f1_macro"] for r in records]
  binary = [r["binary_non_none"]["f1"] for r in records]
  plt.figure(figsize=(8, 4))
  plt.plot(iters, macro, label="macro_f1")
  plt.plot(iters, binary, label="non_none_f1")
  plt.xlabel("iteration")
  plt.ylabel("F1")
  plt.title("Eval Curve")
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def _plot_confusion(conf: np.ndarray, out_path: Path) -> None:
  """Plot a 3x3 confusion matrix heatmap."""
  if plt is None:
    return
  plt.figure(figsize=(4, 4))
  plt.imshow(conf, cmap="Blues")
  plt.colorbar()
  plt.xticks([0, 1, 2], ["none", "low", "high"])
  plt.yticks([0, 1, 2], ["none", "low", "high"])
  plt.xlabel("pred")
  plt.ylabel("true")
  plt.title("Confusion Matrix")
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def _plot_class_metrics(conf: np.ndarray, out_path: Path) -> None:
  """Plot per-class precision/recall bars."""
  if plt is None:
    return
  tp = np.diag(conf)
  fp = conf.sum(axis=0) - tp
  fn = conf.sum(axis=1) - tp
  prec = tp / (tp + fp + 1e-8)
  rec = tp / (tp + fn + 1e-8)
  labels = ["none", "low", "high"]
  x = np.arange(len(labels))
  width = 0.35
  plt.figure(figsize=(5, 3))
  plt.bar(x - width / 2, prec, width, label="precision")
  plt.bar(x + width / 2, rec, width, label="recall")
  plt.xticks(x, labels)
  plt.ylim(0, 1)
  plt.title("Per-class Precision/Recall")
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def _plot_threshold_summary(tuning_path: Path, out_path: Path) -> None:
  """Plot a small table-like summary of tuned thresholds."""
  if plt is None or not tuning_path.exists():
    return
  data = json.loads(tuning_path.read_text())
  rows = [
    ("baseline", data["macro_metrics"]["baseline"]["f1_macro"], data["binary_metrics"]["baseline"]["f1"]),
    ("macro_f1", data["macro_metrics"]["tuned_macro_f1"]["f1_macro"], data["binary_metrics"]["tuned_macro_f1"]["f1"]),
    ("non_none_f1", data["macro_metrics"]["tuned_non_none_f1"]["f1_macro"], data["binary_metrics"]["tuned_non_none_f1"]["f1"]),
  ]
  headers = ["setting", "macro_f1", "non_none_f1"]
  plt.figure(figsize=(5, 2))
  plt.axis("off")
  table = plt.table(cellText=rows, colLabels=headers, loc="center")
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1.0, 1.2)
  plt.title("Threshold Tuning Summary")
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def _plot_per_dir(metrics_per_dir: dict[str, dict[str, float]], out_path: Path) -> None:
  """Plot per-direction macro F1."""
  if plt is None or not metrics_per_dir:
    return
  labels = list(metrics_per_dir.keys())
  vals = [metrics_per_dir[k]["f1_macro"] for k in labels]
  plt.figure(figsize=(6, 3))
  plt.bar(labels, vals)
  plt.ylim(0, 1)
  plt.title("Per-direction Macro F1")
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def main() -> None:
  """Train a LightGBM baseline for occlusion prediction."""
  args = parse_args()
  cfg = OcclusionLGBMCfg()
  cfg.dataset_path = args.dataset_path
  cfg.run_name = args.run_name
  cfg.wandb = args.wandb

  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  run_dir = Path(cfg.output_root) / f"{timestamp}_lgbm_{cfg.run_name}"
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "wandb").mkdir(parents=True, exist_ok=True)

  with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, indent=2)

  if cfg.wandb and wandb is not None:
    os.environ["WANDB_DIR"] = str(run_dir / "wandb")
    wandb.init(
      project="pepper_scene_graphs",
      name=f"lgbm_{cfg.run_name}_{timestamp}",
      config=asdict(cfg),
      dir=str(run_dir / "wandb"),
    )

  data_cfg = build_data_cfg(cfg)
  data_list = load_graph_dataset(cfg.dataset_path, data_cfg)
  if cfg.split.max_scenes and cfg.split.max_scenes > 0:
    data_list = data_list[: cfg.split.max_scenes]

  dir_idx = occlusion_dir_indices(data_cfg)
  dir_labels = occlusion_dir_labels(data_cfg)
  num_dirs = len(dir_idx)

  feat_list: List[np.ndarray] = []
  y_list: List[np.ndarray] = []
  dir_list: List[np.ndarray] = []
  scene_list: List[np.ndarray] = []
  overlap_list: List[np.ndarray] = []
  overlap_none_vals: List[np.ndarray] = []
  overlap_non_none_vals: List[np.ndarray] = []

  for scene_id, data in enumerate(data_list):
    if not hasattr(data, "occl_base_pairs") or data.occl_base_pairs.numel() == 0:
      continue
    if cfg.targets.target_mode == "weighted":
      values = data.occl_base_weighted
    else:
      values = data.occl_base_raw
    if values.numel() == 0:
      continue

    occl_pairs = data.occl_base_pairs
    leaf_idx = occl_pairs[:, 0]
    fruit_idx = occl_pairs[:, 1]
    fruit_to_stem = _fruit_stem_indices(data)
    stem_idx = fruit_to_stem[fruit_idx]
    valid = stem_idx >= 0
    if not valid.any():
      continue
    leaf_idx = leaf_idx[valid]
    fruit_idx = fruit_idx[valid]
    stem_idx = stem_idx[valid]
    values = values[valid]

    geom, inter_norm = _occlusion_geom_features(
      data,
      leaf_idx,
      fruit_idx,
      stem_idx,
      dir_idx,
      include_dist_ratio=cfg.features.include_dist_ratio,
      include_corner_uv=cfg.features.include_corner_uv,
      corner_uv_norm=str(cfg.features.corner_uv_norm),
      feature_cfg=data_cfg.features,
    )
    geom = geom.view(-1, geom.size(-1)).cpu().numpy().astype(np.float32)
    inter_norm = inter_norm.reshape(-1).cpu().numpy().astype(np.float32)
    overlap_flat = values[:, dir_idx].reshape(-1).cpu().numpy().astype(np.float32)
    overlap_list.append(overlap_flat)

    # NOTE: dir_idx is currently contiguous (0..num_dirs-1) for dirs6/dirs18.
    # If a future patch uses a subset of directions, remap indices before one-hot.
    dir_ids = np.array(dir_idx, dtype=np.int32)
    dir_ids = np.tile(dir_ids[None, :], (geom.shape[0] // num_dirs, 1)).reshape(-1)
    dir_onehot = np.zeros((dir_ids.shape[0], num_dirs), dtype=np.float32)
    dir_onehot[np.arange(dir_ids.shape[0]), dir_ids] = 1.0

    X = np.concatenate([geom, dir_onehot], axis=1)
    y = _bin_targets(values[:, dir_idx], cfg.targets.none_thresh, cfg.targets.high_thresh)
    y = y.reshape(-1).cpu().numpy()

    # Track overlap stats for GT none vs non-none.
    none_mask = y == 0
    non_none_mask = ~none_mask
    if none_mask.any():
      overlap_none_vals.append(inter_norm[none_mask])
    if non_none_mask.any():
      overlap_non_none_vals.append(inter_norm[non_none_mask])

    feat_list.append(X)
    y_list.append(y)
    dir_list.append(dir_ids.reshape(-1))
    scene_list.append(np.full((y.shape[0],), scene_id, dtype=np.int64))

  if not feat_list:
    raise RuntimeError("No occlusion samples found for LightGBM training.")

  X = np.concatenate(feat_list, axis=0)
  y = np.concatenate(y_list, axis=0)
  dir_ids_all = np.concatenate(dir_list, axis=0)
  scene_ids = np.concatenate(scene_list, axis=0)
  overlap_all = np.concatenate(overlap_list, axis=0) if overlap_list else None

  if overlap_none_vals or overlap_non_none_vals:
    overlap_stats = {}
    if overlap_none_vals:
      vals = np.concatenate(overlap_none_vals, axis=0)
      overlap_stats["none"] = {
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
      }
    if overlap_non_none_vals:
      vals = np.concatenate(overlap_non_none_vals, axis=0)
      overlap_stats["non_none"] = {
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
      }
    with open(run_dir / "overlap_feature_stats.json", "w", encoding="utf-8") as f:
      json.dump(overlap_stats, f, indent=2)
    print("Overlap feature stats:", overlap_stats)

  unique_scenes = np.unique(scene_ids)
  rng = np.random.default_rng(cfg.split.seed)
  rng.shuffle(unique_scenes)
  val_count = max(1, int(len(unique_scenes) * cfg.split.val_ratio))
  val_scenes = set(unique_scenes[:val_count])
  train_mask = np.array([sid not in val_scenes for sid in scene_ids])
  val_mask = ~train_mask

  X_train = X[train_mask]
  y_train = y[train_mask]
  X_val = X[val_mask]
  y_val = y[val_mask]
  dir_val = dir_ids_all[val_mask]
  overlap_train = overlap_all[train_mask] if overlap_all is not None else None

  counts = np.bincount(y_train, minlength=3).astype(np.float32)
  p_any_val = None
  p_hi_val = None
  if cfg.two_stage.enabled:
    class_weights = np.ones(3, dtype=np.float32)
  else:
    if str(cfg.weights.weight_mode).lower() == "fixed":
      class_weights = np.asarray(cfg.weights.fixed_class_weights, dtype=np.float32)
    else:
      inv = counts.sum() / (counts + 1e-8)
      class_weights = np.minimum(inv, float(cfg.weights.weight_cap))

  if cfg.two_stage.enabled:
    # Stage A: any vs none.
    y_any_train = (y_train > 0).astype(np.int32)
    y_any_val = (y_val > 0).astype(np.int32)
    X_train_any = X_train
    if overlap_train is not None:
      pos_mask = y_any_train == 1
      pos_count = int(pos_mask.sum())
      print("Dead-zone band stats (dead_lo=0.0):")
      for hi in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        band = (overlap_train > 0.0) & (overlap_train < hi)
        band_count = int(band.sum())
        frac_all = band_count / max(1, overlap_train.size)
        frac_pos = int((band & pos_mask).sum()) / max(1, pos_count)
        print(f"  dead_hi={hi:.2f} | count={band_count} | frac_all={frac_all:.4f} | frac_pos={frac_pos:.4f}")
    neg = float((y_any_train == 0).sum())
    pos = float((y_any_train == 1).sum())
    any_pos_weight = min(neg / max(pos, 1.0), float(cfg.two_stage.any_weight_cap))
    any_weights = np.where(y_any_train == 1, any_pos_weight, 1.0)
    dead_cfg = cfg.dead_zone
    dead_removed = 0
    dead_removed_frac = 0.0
    if dead_cfg.enabled and overlap_train is not None:
      dead_lo = float(dead_cfg.dead_lo)
      dead_hi = float(dead_cfg.dead_hi)
      if dead_hi > dead_lo:
        mask_dead = (overlap_train > dead_lo) & (overlap_train < dead_hi)
        dead_removed = int(mask_dead.sum())
        dead_removed_frac = float(dead_removed / max(1, mask_dead.size))
        mode = str(dead_cfg.mode).lower()
        if mode == "drop":
          keep = ~mask_dead
          X_train_any = X_train_any[keep]
          y_any_train = y_any_train[keep]
          any_weights = any_weights[keep]
          overlap_train = overlap_train[keep]
        elif mode == "downweight":
          any_weights = any_weights.astype(np.float32)
          any_weights[mask_dead] *= float(dead_cfg.downweight_factor)
        else:
          raise ValueError(f"Unknown dead_zone.mode: {dead_cfg.mode}")
        # Recompute pos/neg weight after dead-zone filtering.
        neg = float((y_any_train == 0).sum())
        pos = float((y_any_train == 1).sum())
        any_pos_weight = min(neg / max(pos, 1.0), float(cfg.two_stage.any_weight_cap))
        any_weights = np.where(y_any_train == 1, any_pos_weight, any_weights)

    train_any = lgb.Dataset(
      X_train_any,
      label=y_any_train,
      weight=any_weights,
      free_raw_data=False,
    )
    val_any = lgb.Dataset(
      X_val,
      label=y_any_val,
      free_raw_data=False,
    )

    min_leaf_any = cfg.two_stage.any_min_data_in_leaf
    params_any = {
      "objective": "binary",
      "learning_rate": cfg.params.learning_rate,
      "num_leaves": cfg.params.num_leaves,
      "min_data_in_leaf": cfg.params.min_data_in_leaf if min_leaf_any is None else int(min_leaf_any),
      "feature_fraction": cfg.params.feature_fraction,
      "bagging_fraction": cfg.params.bagging_fraction,
      "bagging_freq": cfg.params.bagging_freq,
      "max_bin": cfg.params.max_bin,
      "metric": "binary_logloss",
      "verbosity": -1,
    }

    model_any = lgb.train(
      params_any,
      train_any,
      num_boost_round=cfg.params.num_boost_round,
      valid_sets=[val_any],
      valid_names=["val"],
      callbacks=[
        lgb.early_stopping(cfg.params.early_stopping_rounds),
        lgb.log_evaluation(period=100),
      ],
    )

    p_any_train = model_any.predict(X_train)
    p_any_val = model_any.predict(X_val)

    # Stage B: high vs low on positives only.
    pos_mask_train = y_train > 0
    pos_mask_val = y_val > 0
    if not np.any(pos_mask_train):
      raise RuntimeError("No positive (low/high) samples available for stage B training.")

    X_hi_train = np.concatenate([X_train[pos_mask_train], p_any_train[pos_mask_train, None]], axis=1)
    X_hi_val = np.concatenate([X_val[pos_mask_val], p_any_val[pos_mask_val, None]], axis=1)
    y_hi_train = (y_train[pos_mask_train] == 2).astype(np.int32)
    y_hi_val = (y_val[pos_mask_val] == 2).astype(np.int32)
    neg_hi = float((y_hi_train == 0).sum())
    pos_hi = float((y_hi_train == 1).sum())
    hi_pos_weight = min(neg_hi / max(pos_hi, 1.0), float(cfg.two_stage.high_weight_cap))
    hi_weights = np.where(y_hi_train == 1, hi_pos_weight, 1.0)

    train_hi = lgb.Dataset(
      X_hi_train,
      label=y_hi_train,
      weight=hi_weights,
      free_raw_data=False,
    )
    val_hi = lgb.Dataset(
      X_hi_val,
      label=y_hi_val,
      free_raw_data=False,
    )

    min_leaf_hi = cfg.two_stage.high_min_data_in_leaf
    params_hi = {
      "objective": "binary",
      "learning_rate": cfg.params.learning_rate,
      "num_leaves": cfg.params.num_leaves,
      "min_data_in_leaf": cfg.params.min_data_in_leaf if min_leaf_hi is None else int(min_leaf_hi),
      "feature_fraction": cfg.params.feature_fraction,
      "bagging_fraction": cfg.params.bagging_fraction,
      "bagging_freq": cfg.params.bagging_freq,
      "max_bin": cfg.params.max_bin,
      "metric": "binary_logloss",
      "verbosity": -1,
    }

    model_hi = lgb.train(
      params_hi,
      train_hi,
      num_boost_round=cfg.params.num_boost_round,
      valid_sets=[val_hi],
      valid_names=["val"],
      callbacks=[
        lgb.early_stopping(cfg.params.early_stopping_rounds),
        lgb.log_evaluation(period=100),
      ],
    )

    model_any.save_model(str(run_dir / "model_any.txt"))
    model_hi.save_model(str(run_dir / "model_hi.txt"))

    p_hi_val = model_hi.predict(np.concatenate([X_val, p_any_val[:, None]], axis=1))
    p_none = 1.0 - p_any_val
    p_high = p_any_val * p_hi_val
    p_low = p_any_val * (1.0 - p_hi_val)
    val_pred = np.stack([p_none, p_low, p_high], axis=1)
    y_pred = np.argmax(val_pred, axis=1)
  else:
    weights_train = class_weights[y_train]
    train_data = lgb.Dataset(
      X_train,
      label=y_train,
      weight=weights_train,
      free_raw_data=False,
    )
    val_data = lgb.Dataset(
      X_val,
      label=y_val,
      free_raw_data=False,
    )

    params = {
      "objective": "multiclass",
      "num_class": 3,
      "learning_rate": cfg.params.learning_rate,
      "num_leaves": cfg.params.num_leaves,
      "min_data_in_leaf": cfg.params.min_data_in_leaf,
      "feature_fraction": cfg.params.feature_fraction,
      "bagging_fraction": cfg.params.bagging_fraction,
      "bagging_freq": cfg.params.bagging_freq,
      "max_bin": cfg.params.max_bin,
      "metric": "multi_logloss",
      "verbosity": -1,
    }

    model = lgb.train(
      params,
      train_data,
      num_boost_round=cfg.params.num_boost_round,
      valid_sets=[val_data],
      valid_names=["val"],
      callbacks=[
        lgb.early_stopping(cfg.params.early_stopping_rounds),
        lgb.log_evaluation(period=100),
      ],
    )

    model.save_model(str(run_dir / "model.txt"))

    val_pred = model.predict(X_val)
    y_pred = np.argmax(val_pred, axis=1)

  metrics = _compute_metrics(y_val, y_pred)
  metrics["binary_non_none"] = _binary_metrics(y_val, y_pred)
  metrics["class_weights"] = class_weights.tolist()
  metrics["train_class_counts"] = counts.tolist()
  metrics["num_train"] = int(X_train.shape[0])
  metrics["num_val"] = int(X_val.shape[0])
  metrics["val_dir_counts"] = {label: int((dir_val == d_idx).sum()) for d_idx, label in zip(dir_idx, dir_labels)}
  if cfg.two_stage.enabled:
    metrics["two_stage"] = {
      "any_pos_weight": float(any_pos_weight),
      "high_pos_weight": float(hi_pos_weight),
    }
    if p_hi_val is not None:
      pos_mask_val = y_val > 0
      if np.any(pos_mask_val):
        y_hi_val = (y_val[pos_mask_val] == 2).astype(np.int32)
        pred_hi = (p_hi_val[pos_mask_val] >= 0.5).astype(np.int32)
        metrics["two_stage"]["high_vs_low"] = _binary_metrics(y_hi_val, pred_hi)
    if cfg.dead_zone.enabled:
      metrics["two_stage"]["dead_zone"] = {
        "removed": dead_removed,
        "removed_frac": dead_removed_frac,
        "dead_lo": float(cfg.dead_zone.dead_lo),
        "dead_hi": float(cfg.dead_zone.dead_hi),
        "mode": str(cfg.dead_zone.mode),
        "downweight_factor": float(cfg.dead_zone.downweight_factor),
      }

  per_dir = {}
  per_dir_conf = {}
  for d_idx, label in zip(dir_idx, dir_labels):
    mask = dir_val == d_idx
    if not np.any(mask):
      continue
    per_dir[label] = _compute_metrics(y_val[mask], y_pred[mask])
    per_dir[label]["binary_non_none"] = _binary_metrics(y_val[mask], y_pred[mask])
    per_dir_conf[label] = _confusion_matrix(y_val[mask], y_pred[mask]).tolist()

  conf_overall = _confusion_matrix(y_val, y_pred)

  # Threshold tuning over validation probabilities.
  tuning_cfg = cfg.tuning
  t_any_min = tuning_cfg.t_any_min if tuning_cfg.t_any_min is not None else tuning_cfg.t0_min
  t_any_max = tuning_cfg.t_any_max if tuning_cfg.t_any_max is not None else tuning_cfg.t0_max
  t_hi_min = tuning_cfg.t_hi_min if tuning_cfg.t_hi_min is not None else tuning_cfg.t2_min
  t_hi_max = tuning_cfg.t_hi_max if tuning_cfg.t_hi_max is not None else tuning_cfg.t2_max
  t0_vals = np.arange(t_any_min, t_any_max + 1e-9, tuning_cfg.step)
  t2_vals = np.arange(t_hi_min, t_hi_max + 1e-9, tuning_cfg.step)
  best_macro = {"score": -1.0}
  best_non_none = {"score": -1.0}
  best_rec = {"score": -1.0}
  coarse_macro = None
  coarse_non_none = None
  coarse_rec = None
  refined_macro = None
  refined_non_none = None
  refined_rec = None
  p_min = float(tuning_cfg.p_min)

  def _sweep_objective(t0_vals: np.ndarray, t2_vals: np.ndarray, pred_fn, objective: str) -> dict[str, object]:
    best = {"score": -1.0}
    for t0 in t0_vals:
      for t2 in t2_vals:
        pred = pred_fn(float(t0), float(t2))
        m = _compute_metrics(y_val, pred)
        b = _binary_metrics(y_val, pred)
        if objective == "macro":
          score = m["f1_macro"]
        elif objective == "non_none":
          score = b["f1"]
        else:
          if b["precision"] < p_min:
            continue
          score = b["recall"]
        if score > best["score"]:
          best = {
            "score": float(score),
            "t0": float(t0),
            "t2": float(t2),
            "metrics": m,
            "binary": b,
          }
    return best

  def _refine_bounds(best: dict[str, object], key0: str, key2: str) -> tuple[np.ndarray, np.ndarray]:
    window = float(tuning_cfg.refine_window)
    step = float(tuning_cfg.refine_step)
    t0 = best.get(key0)
    t2 = best.get(key2)
    if t0 is None or t2 is None:
      return np.array([]), np.array([])
    lo0 = max(0.0, float(t0) - window)
    hi0 = min(1.0, float(t0) + window)
    lo2 = max(0.0, float(t2) - window)
    hi2 = min(1.0, float(t2) + window)
    return (
      np.arange(lo0, hi0 + 1e-9, step),
      np.arange(lo2, hi2 + 1e-9, step),
    )

  if cfg.two_stage.enabled:
    p_any = p_any_val
    p_hi = p_hi_val
    if tuning_cfg.enabled:
      def _pred_any(t_any: float, t_hi: float) -> np.ndarray:
        pred_any = p_any >= t_any
        pred = np.ones_like(y_pred)
        pred[p_hi >= t_hi] = 2
        pred[~pred_any] = 0
        return pred

      coarse_macro = _sweep_objective(t0_vals, t2_vals, _pred_any, "macro")
      coarse_non_none = _sweep_objective(t0_vals, t2_vals, _pred_any, "non_none")
      coarse_rec = _sweep_objective(t0_vals, t2_vals, _pred_any, "rec_at_prec")

      best_macro = coarse_macro
      best_non_none = coarse_non_none
      best_rec = coarse_rec

      if tuning_cfg.refine_enabled and coarse_macro and coarse_macro.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_macro, "t0", "t2")
        if r0.size and r2.size:
          refined_macro = _sweep_objective(r0, r2, _pred_any, "macro")
          best_macro = refined_macro
      if tuning_cfg.refine_enabled and coarse_non_none and coarse_non_none.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_non_none, "t0", "t2")
        if r0.size and r2.size:
          refined_non_none = _sweep_objective(r0, r2, _pred_any, "non_none")
          best_non_none = refined_non_none
      if tuning_cfg.refine_enabled and coarse_rec and coarse_rec.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_rec, "t0", "t2")
        if r0.size and r2.size:
          refined_rec = _sweep_objective(r0, r2, _pred_any, "rec_at_prec")
          best_rec = refined_rec
  else:
    p_none = val_pred[:, 0]
    p_high = val_pred[:, 2]

    if tuning_cfg.enabled:
      def _pred_single(t0: float, t2: float) -> np.ndarray:
        none_mask = p_none >= t0
        pred = np.ones_like(y_pred)
        pred[p_high >= t2] = 2
        pred[none_mask] = 0
        return pred

      coarse_macro = _sweep_objective(t0_vals, t2_vals, _pred_single, "macro")
      coarse_non_none = _sweep_objective(t0_vals, t2_vals, _pred_single, "non_none")
      coarse_rec = _sweep_objective(t0_vals, t2_vals, _pred_single, "rec_at_prec")

      best_macro = coarse_macro
      best_non_none = coarse_non_none
      best_rec = coarse_rec

      if tuning_cfg.refine_enabled and coarse_macro and coarse_macro.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_macro, "t0", "t2")
        if r0.size and r2.size:
          refined_macro = _sweep_objective(r0, r2, _pred_single, "macro")
          best_macro = refined_macro
      if tuning_cfg.refine_enabled and coarse_non_none and coarse_non_none.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_non_none, "t0", "t2")
        if r0.size and r2.size:
          refined_non_none = _sweep_objective(r0, r2, _pred_single, "non_none")
          best_non_none = refined_non_none
      if tuning_cfg.refine_enabled and coarse_rec and coarse_rec.get("score", -1.0) >= 0:
        r0, r2 = _refine_bounds(best_rec, "t0", "t2")
        if r0.size and r2.size:
          refined_rec = _sweep_objective(r0, r2, _pred_single, "rec_at_prec")
          best_rec = refined_rec

  with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  with open(run_dir / "metrics_per_dir.json", "w", encoding="utf-8") as f:
    json.dump(per_dir, f, indent=2)
  with open(run_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
    json.dump(conf_overall.tolist(), f, indent=2)
  with open(run_dir / "confusion_matrix_per_dir.json", "w", encoding="utf-8") as f:
    json.dump(per_dir_conf, f, indent=2)
  if tuning_cfg.enabled:
    def _strip_metrics(d: dict[str, object] | None) -> dict[str, object] | None:
      if d is None:
        return None
      return {k: v for k, v in d.items() if k in {"score", "t0", "t2"}}
    def _alias_thresholds(d: dict[str, object] | None) -> dict[str, object] | None:
      if d is None:
        return None
      if not cfg.two_stage.enabled:
        return d
      out = dict(d)
      if "t0" in out:
        out["t_any"] = out.pop("t0")
      if "t2" in out:
        out["t_hi"] = out.pop("t2")
      return out

    with open(run_dir / "threshold_tuning.json", "w", encoding="utf-8") as f:
      json.dump(
        {
          "p_min": p_min,
          "coarse": {
            "thresholds_macro_f1": _alias_thresholds(_strip_metrics(coarse_macro)),
            "thresholds_non_none_f1": _alias_thresholds(_strip_metrics(coarse_non_none)),
            "thresholds_rec_at_prec": _alias_thresholds(_strip_metrics(coarse_rec)),
          },
          "refined": {
            "thresholds_macro_f1": _alias_thresholds(_strip_metrics(refined_macro)),
            "thresholds_non_none_f1": _alias_thresholds(_strip_metrics(refined_non_none)),
            "thresholds_rec_at_prec": _alias_thresholds(_strip_metrics(refined_rec)),
          },
          "thresholds_macro_f1": _alias_thresholds(best_macro),
          "thresholds_non_none_f1": _alias_thresholds(best_non_none),
          "thresholds_rec_at_prec": _alias_thresholds({**best_rec, "p_min": p_min}),
          "macro_metrics": {
            "baseline": _compute_metrics(y_val, y_pred),
            "tuned_macro_f1": best_macro.get("metrics"),
            "tuned_non_none_f1": best_non_none.get("metrics"),
            "tuned_rec_at_prec": best_rec.get("metrics"),
          },
          "binary_metrics": {
            "baseline": _binary_metrics(y_val, y_pred),
            "tuned_macro_f1": best_macro.get("binary"),
            "tuned_non_none_f1": best_non_none.get("binary"),
            "tuned_rec_at_prec": best_rec.get("binary"),
          },
        },
        f,
        indent=2,
      )

  # Optional summary plots.
  if plt is not None:
    _plot_eval_curve(run_dir / "eval_curve.jsonl", run_dir / "report_eval_curve.png")
    _plot_confusion(conf_overall, run_dir / "report_confusion.png")
    _plot_class_metrics(conf_overall, run_dir / "report_class_metrics.png")
    if tuning_cfg.enabled:
      _plot_threshold_summary(run_dir / "threshold_tuning.json", run_dir / "report_thresholds.png")
    _plot_per_dir(per_dir, run_dir / "report_per_dir.png")

  feat_names = [f"feat_{i}" for i in range(X.shape[1] - num_dirs)]
  feat_names += [f"dir_{label}" for label in dir_labels]
  if cfg.two_stage.enabled:
    fi_any = model_any.feature_importance(importance_type="gain")
    fi_hi = model_hi.feature_importance(importance_type="gain")
    fi_any_path = run_dir / "feature_importance_any.csv"
    fi_hi_path = run_dir / "feature_importance_hi.csv"
    with open(fi_any_path, "w", encoding="utf-8") as f:
      f.write("feature,importance\n")
      for name, val in zip(feat_names, fi_any):
        f.write(f"{name},{val}\n")
    with open(fi_hi_path, "w", encoding="utf-8") as f:
      f.write("feature,importance\n")
      hi_names = feat_names + ["p_any"]
      for name, val in zip(hi_names, fi_hi):
        f.write(f"{name},{val}\n")
  else:
    fi = model.feature_importance(importance_type="gain")
    fi_path = run_dir / "feature_importance.csv"
    with open(fi_path, "w", encoding="utf-8") as f:
      f.write("feature,importance\n")
      for name, val in zip(feat_names, fi):
        f.write(f"{name},{val}\n")

  # Optional periodic evaluation (expensive).
  eval_interval = int(cfg.eval.eval_interval)
  if eval_interval > 0:
    rng = np.random.default_rng(cfg.eval.eval_seed)
    if cfg.eval.eval_subsample and cfg.eval.eval_subsample > 0 and cfg.eval.eval_subsample < X_val.shape[0]:
      eval_idx = rng.choice(X_val.shape[0], size=cfg.eval.eval_subsample, replace=False)
      X_eval = X_val[eval_idx]
      y_eval = y_val[eval_idx]
    else:
      X_eval = X_val
      y_eval = y_val
    if cfg.two_stage.enabled:
      max_iter = max(
        model_any.best_iteration or cfg.params.num_boost_round,
        model_hi.best_iteration or cfg.params.num_boost_round,
      )
    else:
      max_iter = model.best_iteration or cfg.params.num_boost_round
    curve_path = run_dir / "eval_curve.jsonl"
    with open(curve_path, "w", encoding="utf-8") as f:
      for it in range(eval_interval, max_iter + 1, eval_interval):
        if cfg.two_stage.enabled:
          p_any_it = model_any.predict(X_eval, num_iteration=it)
          p_hi_it = model_hi.predict(
            np.concatenate([X_eval, p_any_it[:, None]], axis=1),
            num_iteration=it,
          )
          p_none = 1.0 - p_any_it
          p_high = p_any_it * p_hi_it
          p_low = p_any_it * (1.0 - p_hi_it)
          probs = np.stack([p_none, p_low, p_high], axis=1)
        else:
          probs = model.predict(X_eval, num_iteration=it)
        pred_it = np.argmax(probs, axis=1)
        m = _compute_metrics(y_eval, pred_it)
        b = _binary_metrics(y_eval, pred_it)
        record = {
          "iter": it,
          "macro": m,
          "binary_non_none": b,
        }
        f.write(json.dumps(record) + "\n")
    if cfg.wandb and wandb is not None:
      wandb.log({"eval_curve_path": str(curve_path)})

  if cfg.wandb and wandb is not None:
    wandb.log(metrics)
    wandb.log({f"val/binary_non_none/{k}": v for k, v in metrics["binary_non_none"].items()})
    for label, md in per_dir.items():
      wandb.log({
        f"val/{label}/precision_macro": md["precision_macro"],
        f"val/{label}/recall_macro": md["recall_macro"],
        f"val/{label}/f1_macro": md["f1_macro"],
      })
    wandb.log({"val/confusion_matrix": wandb.Table(
      data=conf_overall.tolist(),
      columns=["pred_0", "pred_1", "pred_2"],
    )})
    if per_dir_conf:
      wandb.log({
        f"val/{label}/confusion_matrix": wandb.Table(
          data=mat,
          columns=["pred_0", "pred_1", "pred_2"],
        )
        for label, mat in per_dir_conf.items()
      })
    if cfg.two_stage.enabled:
      wandb.log({
        "feature_importance_any": wandb.Table(
          data=list(zip(feat_names, fi_any)),
          columns=["feature", "importance"],
        ),
        "feature_importance_hi": wandb.Table(
          data=list(zip(feat_names + ["p_any"], fi_hi)),
          columns=["feature", "importance"],
        ),
      })
    else:
      wandb.log({"feature_importance": wandb.Table(
        data=list(zip(feat_names, fi)),
        columns=["feature", "importance"],
      )})
    wandb.finish()


if __name__ == "__main__":
  main()
