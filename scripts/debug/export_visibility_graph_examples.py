#!/usr/bin/env python3

"""Export "visibility graph" occlusion examples for thesis figures.

This script loads a trained oracle checkpoint, runs inference on a dataset split,
and exports a small set of *interesting* fruit-centered occlusion examples:

- a 3D local view (fruit + selected leaves + nearest stem) with OBB wireframes
  and leaf->fruit occlusion edges for one chosen direction
- an 18-direction union plot (GT vs predicted)
- a ranked-leaf plot for the chosen direction showing:
    - GT graded relevance (k-buffer mass, stacked by rank-1/2/3 when available)
    - predicted rank probability (EDL rank expectation if enabled, else softmax)
    - predicted potential (sigmoid)

Outputs are written into `--out_dir`:
- `examples.json` summary
- `example_XXX_*.png` figures per example

Notes:
- This is an *analysis/visualization* helper. It is fine to use GT/sidecar labels
  to *select* interesting cases for plotting.
- The exported "visibility graph" is local by construction: we only include the
  leaves present in `batch.occl_pair_index` (the model's candidate set).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _softplus_np(x: np.ndarray) -> np.ndarray:
  # Stable softplus for numpy.
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


def _ndcg_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
  """NDCG@k with linear gain and log2 discount (matches train_oracle helper)."""
  if scores.size == 0:
    return 0.0
  order_pred = np.argsort(-scores)
  order_ideal = np.argsort(-rel)
  kk = int(min(int(k), int(scores.size)))
  if kk <= 0:
    return 0.0
  discount = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
  dcg = float((rel[order_pred[:kk]] * discount).sum())
  idcg = float((rel[order_ideal[:kk]] * discount).sum())
  return float(dcg / (idcg + 1e-12))


def _dir_local_to_world(
  v_local: np.ndarray,
  *,
  fruit_center_xyz: np.ndarray,
  stem_center_xyz: Optional[np.ndarray],
) -> np.ndarray:
  """Map a fruit-local direction vector into world coords.

  Fruit-local convention:
    +X points toward the nearest stem in the XY plane
    +Z is world up
    +Y completes a right-handed frame.
  """
  v_local = np.asarray(v_local, dtype=np.float32).reshape(3)
  z_hat = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

  if stem_center_xyz is None:
    x_hat = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  else:
    d = np.asarray(stem_center_xyz, dtype=np.float32).reshape(3) - np.asarray(fruit_center_xyz, dtype=np.float32).reshape(3)
    d[2] = 0.0
    n = float(np.linalg.norm(d))
    if n < 1e-6:
      x_hat = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
      x_hat = d / n

  y_hat = np.cross(z_hat, x_hat)
  ny = float(np.linalg.norm(y_hat))
  if ny < 1e-6:
    # Degenerate: fall back to a fixed Y axis.
    y_hat = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
  else:
    y_hat = y_hat / ny
  # Re-orthogonalize X to be safe.
  x_hat = np.cross(y_hat, z_hat)

  v_world = x_hat * float(v_local[0]) + y_hat * float(v_local[1]) + z_hat * float(v_local[2])
  nw = float(np.linalg.norm(v_world))
  if nw < 1e-6:
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
  return v_world / nw


def _build_dir_labels(num_dirs: int) -> List[str]:
  """
  Human-friendly labels for canonical occlusion directions.

  Must match `Oracle*Baseline._build_canonical_occl_dirs()` ordering:
    x, -x, y, -y, z, -z,
    (-x-y), (-x+y), (x-y), (x+y),
    (-x-z), (-x+z), (x-z), (x+z),
    (-y-z), (-y+z), (y-z), (y+z)
  """
  labels = ["x", "-x", "y", "-y", "z", "-z"]

  def _pair(a: str, b: str, s1: float, s2: float) -> str:
    p1 = ("" if s1 > 0 else "-") + a
    p2 = ("+" if s2 > 0 else "-") + b
    return p1 + p2

  for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
    for s1 in (-1.0, 1.0):
      for s2 in (-1.0, 1.0):
        labels.append(_pair(a, b, s1, s2))

  if int(num_dirs) <= len(labels):
    return labels[: int(num_dirs)]
  # Fallback: repeat if caller requests more than 18 (shouldn't happen in practice).
  reps = (int(num_dirs) + len(labels) - 1) // len(labels)
  out = (labels * reps)[: int(num_dirs)]
  return out


def _obb_corners(center: Sequence[float], axes: Sequence[Sequence[float]], extents: Sequence[float]) -> np.ndarray:
  """Compute OBB corners (8,3)."""
  axes = np.asarray(axes, dtype=np.float32)  # [3,3] columns or rows? JSON uses columns in prior scripts.
  center = np.asarray(center, dtype=np.float32)
  extents = np.asarray(extents, dtype=np.float32)
  half = 0.5 * extents
  a0, a1, a2 = axes[:, 0], axes[:, 1], axes[:, 2]
  corners = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        corners.append(center + sx * half[0] * a0 + sy * half[1] * a1 + sz * half[2] * a2)
  return np.stack(corners, axis=0)


def _draw_obb(ax, corners: np.ndarray, *, color: str, lw: float = 1.0, alpha: float = 1.0) -> None:
  # 12 edges between corners (standard cube wiring for our corner ordering).
  edges = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
  ]
  for i, j in edges:
    p = corners[i]
    q = corners[j]
    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, lw=lw, alpha=alpha)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Export fruit-centric visibility-graph occlusion examples.")
  p.add_argument("--data_root", required=True, help="Dataset root (same as training --data_root).")
  p.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Dataset split.")
  p.add_argument("--run_dir", type=str, default=None, help="Oracle run directory (training/oracle/...).")
  p.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint (.pt). Overrides --run_dir.")
  p.add_argument("--ckpt_kind", type=str, default="best", choices=["best", "last"], help="When using --run_dir, which checkpoint to load.")
  p.add_argument("--device", type=str, default="cuda", help="torch device for inference (e.g. cuda or cpu).")
  p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (keep small; default=1).")
  p.add_argument("--max_scenes", type=int, default=200, help="Max scenes to scan for candidates.")
  p.add_argument("--num_examples", type=int, default=10, help="How many examples to export.")
  p.add_argument("--seed", type=int, default=0, help="Random seed (selection tie-breaks only).")
  p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: analysis/vis_visibility_graph_<ts>).")

  # "Interestingness" filters.
  p.add_argument("--min_explain", type=float, default=0.85, help="Min mean explainability on dominant dirs (present-leaf exclusive mass / union).")
  p.add_argument("--min_cand_leaves", type=int, default=12, help="Min candidate leaves for the fruit.")
  p.add_argument("--min_dom_dirs", type=int, default=1, help="Min number of dominant (union>=tau) directions.")
  p.add_argument("--max_top1_share", type=float, default=0.9, help="Require at least one dominant dir with top1_share <= this (competition).")

  # Example picking.
  p.add_argument(
    "--pick_mode",
    type=str,
    default="spread_perf",
    choices=["top", "spread_perf"],
    help="How to pick examples after filtering. 'spread_perf' keeps cases interesting but spreads ranking performance.",
  )
  p.add_argument(
    "--pick_pool",
    type=int,
    default=300,
    help="Pool size (top by interestingness) used when pick_mode=spread_perf.",
  )

  # 3D view options.
  p.add_argument(
    "--cam_align",
    type=str,
    default="dir",
    choices=["dir", "fixed"],
    help="3D camera alignment: 'dir' looks along the chosen direction; 'fixed' uses a constant elev/azim.",
  )
  p.add_argument("--cam_elev", type=float, default=18.0, help="Matplotlib 3D elevation when cam_align=fixed.")
  p.add_argument("--cam_azim", type=float, default=-60.0, help="Matplotlib 3D azimuth when cam_align=fixed.")
  p.add_argument("--triad_scale", type=float, default=0.8, help="Scale factor for the XYZ triad length (relative to fruit radius).")
  p.add_argument("--ctx_leaves", type=int, default=30, help="How many extra nearby leaves to show as faint context in 3D (0 disables).")
  p.add_argument("--ctx_max_pts", type=int, default=600, help="Max points per context leaf for 3D plotting (keeps file size manageable).")
  p.add_argument("--ctx_opacity", type=float, default=0.06, help="Opacity for context leaves in 3D plots.")
  p.add_argument("--plot_plane", type=int, default=1, choices=[0, 1], help="Whether to draw a translucent occlusion plane for the chosen direction.")

  # Plot choices.
  p.add_argument("--topk_plot", type=int, default=8, help="How many leaves to show in the ranking plot.")
  p.add_argument(
    "--include_worst_leaf",
    type=int,
    default=0,
    choices=[0, 1],
    help="Whether to add one worst-ranked leaf (by predicted rank) for contrast.",
  )
  return p.parse_args()


@dataclass
class _Example:
  score: float
  perf: float
  union_mae_dom: float
  scene_idx: int
  scene_id: str
  fruit_row: int
  fruit_node: int
  fruit_iid: int
  dir_hi: int
  explain_mean_dom: float
  ndcg3_dom: float
  ndcg3_hi: float
  perf_tag: str = ""


def _load_json_graph(data_root: str, scene_id: str) -> Dict[int, dict]:
  """Load scene graph JSON and return iid->node dict."""
  data_dir = Path(data_root) / "data"
  if not data_dir.exists():
    data_dir = Path(data_root)
  p = data_dir / f"{scene_id}_graph.json"
  with open(p, "r", encoding="utf-8") as f:
    j = json.load(f)
  nodes = j.get("nodes") or j.get("Nodes") or []
  out: Dict[int, dict] = {}
  for n in nodes:
    try:
      iid = int(n.get("instance_id", -1))
    except Exception:
      continue
    if iid >= 0:
      out[iid] = n
  return out


def main() -> None:
  args = parse_args()

  # Lazy imports (torch is only available in your training env).
  import random
  import torch
  from torch.utils.data import DataLoader

  # Dataset class name in this repo is OraclePepperDataset (legacy code prints "OracleDataset: ...").
  from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn
  from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline, OracleResBaseline

  random.seed(int(args.seed))
  np.random.seed(int(args.seed))
  torch.manual_seed(int(args.seed))

  run_dir = Path(args.run_dir) if args.run_dir else None
  ckpt_path: Optional[Path] = Path(args.ckpt) if args.ckpt else None
  run_cfg = {}
  if ckpt_path is None:
    if run_dir is None:
      raise SystemExit("Provide either --ckpt or --run_dir.")
    if not run_dir.exists():
      raise SystemExit(f"--run_dir does not exist: {run_dir}")
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
      run_cfg = json.load(f)
    ckpt_name = "best_model.pt" if args.ckpt_kind == "best" else "last_model.pt"
    ckpt_path = run_dir / ckpt_name
  if ckpt_path is None or (not ckpt_path.exists()):
    raise SystemExit(f"Checkpoint not found: {ckpt_path}")

  # Rebuild model config (matches scripts/train_oracle.py).
  def _cfg_get(key: str, default):
    return run_cfg.get(key, default)

  # Dataset sampling config: prefer run config if present.
  split = str(args.split)
  ds_base = OraclePepperDataset(
    data_root=str(args.data_root),
    split=split,
    npoint_per_instance=int(_cfg_get("npoint", 256)),
    augment=False,
    rotate_aug=False,
    rotate_mode="none",
    merge_shoots=not bool(_cfg_get("drop_shoots", False)),
    drop_shoots=bool(_cfg_get("drop_shoots", False)),
    drop_json=False,
    subgraph_enabled=bool(_cfg_get("subgraph_enabled", True)),
    subgraph_size=int(_cfg_get("subgraph_size", 288)),
    subgraph_size_min=_cfg_get("subgraph_size_min", None),
    use_4mm=bool(_cfg_get("use_4mm", True)),
    hard_pair_constraint=bool(_cfg_get("hard_pair_constraint", True)),
    occl_aug_enabled=bool(_cfg_get("occl_aug_enabled", True)),
    occl_sidecar_dir=_cfg_get("occl_sidecar_dir", None),
    occl_aug_num_fruits=int(_cfg_get("occl_aug_num_fruits", 4)),
    occl_aug_union_tau=float(_cfg_get("occl_aug_union_tau", 0.5)),
    occl_aug_resample_tries=int(_cfg_get("occl_aug_resample_tries", 5)),
    occl_aug_max_dist=float(_cfg_get("occl_aug_max_dist", 0.2)),
    occl_aug_max_pos_leaves=int(_cfg_get("occl_aug_max_pos_leaves", 16)),
    occl_aug_max_neg_leaves=int(_cfg_get("occl_aug_max_neg_leaves", 8)),
    occl_aug_max_total_leaves=int(_cfg_get("occl_aug_max_total_leaves", 96)),
    occl_aug_reserve_nodes=int(_cfg_get("occl_aug_reserve_nodes", 64)),
    occl_targets_enabled=True,
    occl_targets_union_tau=float(_cfg_get("occl_union_tau", 0.5)),
    occl_targets_max_leaves_per_fruit=int(_cfg_get("occl_targets_max_leaves_per_fruit", 64)),
    occl_leafset_policy=str(_cfg_get("occl_leafset_policy", "geom_stratified")),
    occl_supervise_max_fruits_per_scene=int(_cfg_get("occl_supervise_max_fruits_per_scene", 12)),
    occl_supervise_remain_rate_min=float(_cfg_get("occl_supervise_remain_rate_min", 0.9)),
    occl_supervise_min_dom_dirs=int(_cfg_get("occl_supervise_min_dom_dirs", 1)),
  )

  # IMPORTANT: OraclePepperDataset uses python's global RNG (`random`) for subgraph and occlusion-aug focus fruit selection.
  # If we later re-fetch ds[idx] to plot an example, the supervised fruit set can differ due to RNG state drift, which can
  # break fruit indexing (and can silently change which fruit/leafset we're plotting).
  #
  # To keep example extraction reproducible, we wrap the dataset to re-seed per item index.
  class _SeededDataset(torch.utils.data.Dataset):
    def __init__(self, base, base_seed: int):
      self.base = base
      self.base_seed = int(base_seed)

    def __len__(self):
      return len(self.base)

    def __getitem__(self, idx: int):
      s = (self.base_seed * 1000003 + int(idx)) & 0xFFFFFFFF
      random.seed(int(s))
      np.random.seed(int(s))
      torch.manual_seed(int(s))
      return self.base[int(idx)]

  ds = _SeededDataset(ds_base, base_seed=int(args.seed))
  loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=oracle_collate_fn, num_workers=0)

  # Model.
  from types import SimpleNamespace

  cfg = SimpleNamespace(
    model=SimpleNamespace(
      hidden_dim=128,
      num_layers=4,
      dropout=0.1,
      oracle_msg_npoint=int(_cfg_get("npoint", 256)),
      oracle_msg_radius_list=[0.01, 0.02, 0.04],
      oracle_msg_nsample_list=[16, 32, 64],
      node_feat_mode=str(_cfg_get("node_feat_mode", "local")),
      k_candidates=int(_cfg_get("k_candidates", 8)),
      dist_threshold=float(_cfg_get("dist_threshold", 0.15)),
      extra_stems_per_node=int(_cfg_get("extra_stems_per_node", 2)),
      use_normals=bool(_cfg_get("use_normals", True)),
      use_rgb=bool(_cfg_get("use_rgb", True)),
      no_geom_injection=bool(_cfg_get("no_geom_injection", True)),
      occl_enabled=True,
      occl_num_dirs=18,
      occl_dir_embed_dim=int(_cfg_get("occl_dir_embed_dim", 16)),
      occl_attn_dim=int(_cfg_get("occl_attn_dim", 64)),
      occl_attn_temp=float(_cfg_get("occl_attn_temp", 1.0)),
      occl_leaf_sa_layers=int(_cfg_get("occl_leaf_sa_layers", 2)),
      occl_leaf_sa_mode=str(_cfg_get("occl_leaf_sa_mode", "per_fruit_dir")),
      occl_leaf_sa_dim=int(_cfg_get("occl_leaf_sa_dim", 128)),
      occl_leaf_sa_heads=int(_cfg_get("occl_leaf_sa_heads", 4)),
      occl_leaf_sa_ff_mult=int(_cfg_get("occl_leaf_sa_ff_mult", 4)),
      occl_leaf_sa_dropout=float(_cfg_get("occl_leaf_sa_dropout", 0.1)),
    )
  )

  model_type = str(_cfg_get("model_type", "res")).lower()
  if model_type == "res":
    model = OracleResBaseline(cfg, node_classes=4, edge_labels=3)
  else:
    model = OracleGTBaseline(cfg, node_classes=4, edge_labels=3)

  device = torch.device(str(args.device))
  model = model.to(device)

  ckpt = torch.load(str(ckpt_path), map_location=device)
  # Backwards compatibility: older checkpoints (pre 2.1.9) do not have the split rank projections.
  # For visualization purposes we can:
  #  - load with strict=False
  #  - copy potential q/k proj weights into the rank q/k proj modules so rank==potential (legacy behavior).
  state = ckpt.get("model_state", ckpt)

  # SharedMSGEncoder initializes its post-MLP lazily on first forward, but our checkpoints contain its weights.
  # If we load a checkpoint before the module exists, those keys become "unexpected" and the encoder stays random.
  # Fix: materialize `_post_msg` from the checkpoint weight shape prior to load_state_dict.
  def _maybe_init_shared_encoder_post_msg_from_state(sd: dict) -> None:
    try:
      se = getattr(model, "shared_encoder", None)
      if se is None:
        return
      if getattr(se, "_post_msg", None) is not None:
        return

      # Handle possible prefixes (e.g. 'module.').
      key = None
      for k in sd.keys():
        if str(k).endswith("shared_encoder._post_msg.0.weight"):
          key = str(k)
          break
      if key is None:
        return
      w = sd[key]
      in_dim = int(w.shape[1])
      if hasattr(se, "_init_post"):
        se._init_post(in_dim)
        se._post_msg = se._post_msg.to(device)
        print(f"[info] initialized shared_encoder._post_msg for checkpoint load (in_dim={in_dim}).", flush=True)
    except Exception as e:
      print(f"[warn] failed to init shared_encoder._post_msg from checkpoint: {e}", flush=True)

  if isinstance(state, dict):
    _maybe_init_shared_encoder_post_msg_from_state(state)

  try:
    model.load_state_dict(state, strict=True)
  except RuntimeError as e:
    incompat = model.load_state_dict(state, strict=False)
    missing = set(getattr(incompat, "missing_keys", []) or [])
    unexpected = set(getattr(incompat, "unexpected_keys", []) or [])
    print(f"[warn] partial checkpoint load (strict=False) for visualization: {e}", flush=True)
    if unexpected:
      print(f"[warn] unexpected keys: {sorted(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}", flush=True)

    # If the only missing parts are the split rank projections, restore legacy behavior.
    if any(k.startswith("occl_q_rank_proj.") for k in missing) or any(k.startswith("occl_k_rank_proj.") for k in missing):
      try:
        if hasattr(model, "occl_q_rank_proj") and hasattr(model, "occl_q_proj"):
          model.occl_q_rank_proj.load_state_dict(model.occl_q_proj.state_dict())
        if hasattr(model, "occl_k_rank_proj") and hasattr(model, "occl_k_proj"):
          model.occl_k_rank_proj.load_state_dict(model.occl_k_proj.state_dict())
        print("[warn] initialized rank projections from potential projections (legacy rank==potential).", flush=True)
      except Exception as ee:
        print(f"[warn] failed to initialize rank projections from potential projections: {ee}", flush=True)

  model.eval()
  dirs_local = None
  try:
    dirs_local = model.occl_dirs.detach().cpu().numpy().astype(np.float32)  # [D,3] in fruit-local coords
  except Exception:
    dirs_local = None

  # Export paths.
  if args.out_dir:
    out_dir = Path(args.out_dir)
  else:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("analysis") / f"vis_visibility_graph_{ts}"
  out_dir.mkdir(parents=True, exist_ok=True)

  # Matplotlib (lazy).
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

  union_tau = float(_cfg_get("occl_union_tau", 0.5))
  pot_eps = float(_cfg_get("occl_potential_eps", 0.05))
  edl_rank = bool(_cfg_get("edl_occl_rank", False))
  dir_labels = _build_dir_labels(int(getattr(cfg.model, "occl_num_dirs", 18)))

  examples: List[_Example] = []

  def _batch_scene_id(batch) -> str:
    sid = getattr(batch, "scene_id", None)
    if sid is None:
      return ""
    # PyG Batch may store strings as a list of python strings.
    if isinstance(sid, (list, tuple)):
      return str(sid[0])
    return str(sid)

  # Scan scenes until we have enough candidates.
  with torch.no_grad():
    for si, batch in enumerate(loader):
      if si >= int(args.max_scenes):
        break
      batch = batch.to(device)
      out = model(batch)

      fruit_idx = getattr(batch, "occl_fruit_idx", None)
      pair_index = getattr(batch, "occl_pair_index", None)
      y_union = getattr(batch, "occl_fruit_union", None)
      y_pot = getattr(batch, "occl_pair_potential", None)
      y_exc = getattr(batch, "occl_pair_exclusive", None)
      y_k = getattr(batch, "occl_pair_k_rank", None)
      if fruit_idx is None or pair_index is None or y_union is None or y_pot is None:
        continue
      if int(fruit_idx.numel()) == 0 or int(pair_index.numel()) == 0:
        continue

      # Predictions.
      union_logits = out.get("occl_fruit_union_logits", None)
      pot_logits = out.get("occl_pair_potential_logits", None)
      rank_logits = out.get("occl_pair_rank_logits", None)
      if union_logits is None or pot_logits is None or rank_logits is None:
        continue

      node_iid = batch.node_iid.detach().cpu().numpy().astype(np.int64)
      fruit_nodes = fruit_idx.detach().cpu().numpy().astype(np.int64)
      pair_f = pair_index[0].detach().cpu().numpy().astype(np.int64)
      pair_l = pair_index[1].detach().cpu().numpy().astype(np.int64)

      union_gt = y_union.detach().cpu().numpy().astype(np.float32)
      pot_gt = y_pot.detach().cpu().numpy().astype(np.float32)
      exc_gt = y_exc.detach().cpu().numpy().astype(np.float32) if (y_exc is not None and y_exc.numel() > 0) else None
      k_gt = y_k.detach().cpu().numpy().astype(np.float32) if (y_k is not None and y_k.numel() > 0) else None
      union_pr_all = _sigmoid_np(union_logits.detach().cpu().numpy().astype(np.float32))

      # Score candidates fruit-by-fruit.
      scene_id = _batch_scene_id(batch)
      for r, fnode in enumerate(fruit_nodes.tolist()):
        m = (pair_f == int(fnode))
        L = int(m.sum())
        if L < int(args.min_cand_leaves):
          continue

        u = union_gt[r]  # [D]
        u_pr = union_pr_all[r]  # [D]
        dom = (u >= union_tau)
        if int(dom.sum()) < int(args.min_dom_dirs):
          continue

        # Explainability (dominant dirs only): sum exclusive (rank1) mass over present leaves / union.
        if exc_gt is not None:
          union_expl = exc_gt[m].sum(axis=0).clip(0.0, 1.0)
        elif k_gt is not None:
          union_expl = k_gt[m, :, 0].sum(axis=0).clip(0.0, 1.0)
        else:
          union_expl = np.zeros_like(u)
        frac = np.ones_like(u)
        pos = (u > 1e-6)
        frac[pos] = (union_expl[pos] / np.maximum(u[pos], 1e-6)).clip(0.0, 1.0)
        explain_mean_dom = float(frac[dom].mean()) if bool(dom.any()) else 0.0
        if explain_mean_dom < float(args.min_explain):
          continue

        # GT relevance for ranking: top-3 mass (k-buffer sum) if available else potential.
        if k_gt is not None:
          rel = k_gt[m].sum(axis=-1)  # [L,D]
        else:
          rel = pot_gt[m]  # [L,D]
        # Drop dust.
        rel = np.where(pot_gt[m] >= pot_eps, rel, 0.0)

        # Competition: at least one dominant dir where top1 share is not ~1.
        top1_share = 1.0
        for d in np.nonzero(dom)[0].tolist():
          s = float(rel[:, d].sum())
          if s <= 0.0:
            continue
          share = float(rel[:, d].max() / (s + 1e-12))
          top1_share = min(top1_share, share)
        if top1_share > float(args.max_top1_share):
          continue

        # Pred ranking score for NDCG (dominant dirs): use rank logits.
        rank_logits_np = rank_logits.detach().cpu().numpy().astype(np.float32)[m]  # [L,D]
        ndcgs: List[float] = []
        for d in np.nonzero(dom)[0].tolist():
          if float(rel[:, d].sum()) <= 0.0:
            continue
          ndcgs.append(_ndcg_at_k(rank_logits_np[:, d], rel[:, d], k=3))
        ndcg3_dom = float(np.mean(ndcgs)) if ndcgs else 0.0

        # Choose a "main" direction for visualization.
        # We want a direction with meaningful *competition* among leaves (ranking is interesting).
        # So among dominant directions (union>=tau) we pick the one with the lowest top1_share.
        # Fallback: strongest GT union.
        dir_hi = int(np.argmax(u))
        best = None  # (top1_share, -union, dir)
        for d in np.nonzero(dom)[0].tolist():
          s = float(rel[:, d].sum())
          if s <= 0.0:
            continue
          share = float(rel[:, d].max() / (s + 1e-12))
          cand = (share, -float(u[d]), int(d))
          if best is None or cand < best:
            best = cand
        if best is not None:
          dir_hi = int(best[2])

        # Ranking quality on the chosen direction (this should match what the leaf-ranking plot shows).
        if float(rel[:, dir_hi].sum()) > 0.0:
          ndcg3_hi = float(_ndcg_at_k(rank_logits_np[:, dir_hi], rel[:, dir_hi], k=3))
        else:
          ndcg3_hi = 0.0

        # Union quality on dominant dirs (mostly for diagnosing union head behavior).
        union_mae_dom = float(np.abs(u_pr[dom] - u[dom]).mean()) if bool(dom.any()) else float("nan")

        # Interestingness score: emphasize explainability + competition + ranking quality.
        score = float(1.5 * ndcg3_dom + 1.0 * explain_mean_dom + 0.5 * (1.0 - top1_share))
        # Performance score used for spread picking: chosen-dir ranking quality.
        # This prevents "perf_good" examples from looking terrible on the plotted direction.
        perf = float(ndcg3_hi)

        examples.append(
          _Example(
            score=score,
            perf=perf,
            union_mae_dom=union_mae_dom,
            scene_idx=int(si),
            scene_id=scene_id or f"scene_{si:05d}",
            fruit_row=int(r),
            fruit_node=int(fnode),
            fruit_iid=int(node_iid[int(fnode)]),
            dir_hi=dir_hi,
            explain_mean_dom=explain_mean_dom,
            ndcg3_dom=ndcg3_dom,
            ndcg3_hi=ndcg3_hi,
          )
        )

      # Early stop if we already have many candidates (keep runtime bounded).
      if len(examples) >= max(50, int(args.num_examples) * 5):
        break

  def _prefer_unique_scenes_fill(
    cand: List[_Example],
    n: int,
    *,
    sort_key,
  ) -> List[_Example]:
    cand = sorted(cand, key=sort_key, reverse=True)
    picked: List[_Example] = []
    used_scenes: set = set()
    used_keys: set = set()
    for ex in cand:
      k = (ex.scene_id, ex.fruit_iid)
      if k in used_keys:
        continue
      if ex.scene_id in used_scenes:
        continue
      picked.append(ex)
      used_scenes.add(ex.scene_id)
      used_keys.add(k)
      if len(picked) >= int(n):
        return picked
    # If we don't have enough unique scenes, allow duplicates but keep (scene,fruit) unique.
    for ex in cand:
      k = (ex.scene_id, ex.fruit_iid)
      if k in used_keys:
        continue
      picked.append(ex)
      used_keys.add(k)
      if len(picked) >= int(n):
        return picked
    return picked

  def _pick_spread_perf(examples_sorted_by_interest: List[_Example], n: int, pool_size: int) -> List[_Example]:
    pool = examples_sorted_by_interest[: int(min(len(examples_sorted_by_interest), max(1, int(pool_size))))]
    if not pool:
      return []
    pool_sorted = sorted(pool, key=lambda e: (e.perf, -e.score))

    used_scenes: set = set()
    used_keys: set = set()
    picked: List[_Example] = []

    def _take_in_order(cand: List[_Example], n_take: int, tag: str) -> None:
      nonlocal picked
      if n_take <= 0:
        return
      for ex in cand:
        if len(picked) >= int(args.num_examples):
          return
        k = (ex.scene_id, ex.fruit_iid)
        if k in used_keys:
          continue
        if ex.scene_id in used_scenes:
          continue
        ex.perf_tag = str(tag)
        picked.append(ex)
        used_keys.add(k)
        used_scenes.add(ex.scene_id)
        if len([p for p in picked if p.perf_tag == tag]) >= int(n_take):
          return

    if n <= 1:
      _take_in_order(list(reversed(pool_sorted)), 1, "mid")
      return picked

    # Split into bad/mid/good groups by perf rank (not by absolute thresholds).
    # This avoids "good" being assigned arbitrarily when there are many perf ties:
    # "good" always means "picked from the best-performing tail we could find under uniqueness constraints".
    n_bad = max(1, int(math.ceil(float(n) / 3.0)))
    n_good = max(1, int(math.floor(float(n) / 3.0)))
    n_mid = int(n) - int(n_bad) - int(n_good)
    if n_mid < 0:
      n_mid = 0

    # Bad: lowest perf.
    _take_in_order(pool_sorted, n_bad, "bad")

    # Good: highest perf.
    _take_in_order(list(reversed(pool_sorted)), n_good, "good")

    # Mid: around median (walk outward).
    if n_mid > 0:
      mid = int(len(pool_sorted) // 2)
      order_mid: List[int] = []
      for off in range(0, len(pool_sorted)):
        for j in (mid - off, mid + off):
          if 0 <= j < len(pool_sorted):
            order_mid.append(int(j))
        if len(order_mid) >= len(pool_sorted):
          break
      cand_mid = [pool_sorted[j] for j in order_mid]
      _take_in_order(cand_mid, n_mid, "mid")

    # Fill any remaining slots (uniqueness may have prevented hitting counts).
    if len(picked) < int(n):
      for ex in list(reversed(pool_sorted)):
        k = (ex.scene_id, ex.fruit_iid)
        if k in used_keys:
          continue
        ex.perf_tag = ex.perf_tag or "mid"
        picked.append(ex)
        used_keys.add(k)
        if len(picked) >= int(n):
          break
    return picked

  # Example selection:
  #  1) Filtered examples are scored by interestingness.
  #  2) pick_mode='top': take the top interesting examples.
  #  3) pick_mode='spread_perf': among the top `pick_pool` interesting ones, pick a perf-spread set.
  examples.sort(key=lambda e: e.score, reverse=True)
  if str(args.pick_mode) == "top":
    picked = _prefer_unique_scenes_fill(examples, int(args.num_examples), sort_key=lambda e: e.score)
  else:
    picked = _pick_spread_perf(examples, int(args.num_examples), int(args.pick_pool))

  if not picked:
    raise SystemExit("No examples matched the filters. Try lowering --min_explain/--min_cand_leaves or increasing --max_scenes.")

  # Helper to get one dataset item by scene_idx.
  # We use ds.indices mapping, so we can reconstruct the item index deterministically.
  def _dataset_item_for_scene_idx(scene_idx: int) -> int:
    # ds.indices maps dataset index -> scene index in pc_files.
    # Here `scene_idx` is the iterator index in the DataLoader over ds, so it is the dataset index already.
    return int(scene_idx)

  exported: List[dict] = []
  for i, ex in enumerate(picked):
    item_idx = _dataset_item_for_scene_idx(ex.scene_idx)
    batch = ds[item_idx]

    # Re-run model on the selected batch (batch_size=1) for plotting.
    from pepper_scene_graphs.data.oracle_dataset import oracle_collate_fn as _coll
    import torch

    b = _coll([batch]).to(device)
    with torch.no_grad():
      out = model(b)

    # Pull tensors (CPU numpy).
    fruit_idx = b.occl_fruit_idx.detach().cpu().numpy().astype(np.int64)
    pair_index = b.occl_pair_index.detach().cpu().numpy().astype(np.int64)
    y_union = b.occl_fruit_union.detach().cpu().numpy().astype(np.float32)
    y_pot = b.occl_pair_potential.detach().cpu().numpy().astype(np.float32)
    y_exc = b.occl_pair_exclusive.detach().cpu().numpy().astype(np.float32) if b.occl_pair_exclusive.numel() > 0 else None
    y_k = b.occl_pair_k_rank.detach().cpu().numpy().astype(np.float32) if b.occl_pair_k_rank.numel() > 0 else None
    node_iid = b.node_iid.detach().cpu().numpy().astype(np.int64)

    union_logits = out["occl_fruit_union_logits"].detach().cpu().numpy().astype(np.float32)
    pot_logits = out["occl_pair_potential_logits"].detach().cpu().numpy().astype(np.float32)
    rank_logits = out["occl_pair_rank_logits"].detach().cpu().numpy().astype(np.float32)

    # Fruit row r in this batch is stable by iid matching.
    fruit_iids = [int(node_iid[int(n)]) for n in fruit_idx.tolist()]
    try:
      r = fruit_iids.index(int(ex.fruit_iid))
    except ValueError:
      # Fallback: clamp to valid range (should be rare if dataset seeding is stable).
      r = int(ex.fruit_row)
      r = max(0, min(r, int(fruit_idx.shape[0]) - 1))
    fruit_node = int(fruit_idx[r])
    fruit_iid = int(node_iid[int(fruit_node)])

    pair_f = pair_index[0]
    pair_l = pair_index[1]
    m = (pair_f == fruit_node)
    leaf_nodes = pair_l[m].astype(np.int64)
    leaf_iids = [int(node_iid[int(n)]) for n in leaf_nodes.tolist()]

    D = int(y_union.shape[1])
    dir_hi = int(ex.dir_hi)

    # Union plot data.
    union_gt = y_union[r]
    union_pr = _sigmoid_np(union_logits[r])

    # Ranking data for chosen direction.
    pot_pr = _sigmoid_np(pot_logits[m, dir_hi])  # [L]
    rank_sc = rank_logits[m, dir_hi]             # [L]

    # Optional EDL rank uncertainty:
    #   alpha_l = softplus(logit_l) + 1
    #   S = sum_l alpha_l
    #   u = L / S  (higher u => lower evidence / higher uncertainty)
    u_rank = float("nan")
    S_rank = float("nan")
    u_rank_per_dir = None  # [D] or None
    S_rank_per_dir = None  # [D] or None
    if edl_rank:
      alpha = _softplus_np(rank_sc) + 1.0
      S_rank = float(alpha.sum() + 1e-12)
      p_rank = alpha / S_rank
      u_rank = float(len(alpha) / S_rank)

      # Per-direction uncertainty (same candidate leaf set, varying direction).
      # Shape: [L,D] -> sum over leaves -> [D]
      rank_sc_all = rank_logits[m, :D]  # [L,D]
      alpha_all = _softplus_np(rank_sc_all) + 1.0
      S_rank_per_dir = alpha_all.sum(axis=0) + 1e-12
      u_rank_per_dir = float(alpha_all.shape[0]) / S_rank_per_dir
    else:
      exp_sc = np.exp(rank_sc - float(rank_sc.max()))
      p_rank = exp_sc / (exp_sc.sum() + 1e-12)

    # GT relevance stacked (k1/k2/k3) if available.
    if y_k is not None:
      rel_k = y_k[m, dir_hi, :]  # [L,3]
      rel = rel_k.sum(axis=-1)
    else:
      rel_k = None
      rel = y_pot[m, dir_hi]
    rel = np.where(y_pot[m, dir_hi] >= pot_eps, rel, 0.0)

    # Select leaves to show:
    #   - top-K by predicted rank prob (model-important)
    #   - union with top-K by GT relevance (GT-important)
    # This makes TP/FP/FN visible in both 3D and the ranking plot.
    n_leaves = int(len(leaf_nodes))
    k_plot = int(min(int(args.topk_plot), n_leaves))
    order_pred = np.argsort(-p_rank)
    top_pred_idx = order_pred[:k_plot].tolist()

    pos_gt = np.nonzero(rel > 0.0)[0]
    if pos_gt.size > 0:
      order_gt = pos_gt[np.argsort(-rel[pos_gt])]
      top_gt_idx = order_gt[:k_plot].tolist()
    else:
      top_gt_idx = []

    plot_idx = list(dict.fromkeys(top_pred_idx + top_gt_idx))
    # Optionally add one low-score leaf for contrast (if available).
    if int(args.include_worst_leaf) == 1 and n_leaves > 0 and len(order_pred) > 0:
      plot_idx.append(int(order_pred[-1]))
    plot_idx = list(dict.fromkeys(plot_idx))

    leaf_nodes_sel = [int(leaf_nodes[j]) for j in plot_idx]
    leaf_iids_sel = [int(leaf_iids[j]) for j in plot_idx]
    pot_pr_sel = pot_pr[plot_idx]
    p_rank_sel = p_rank[plot_idx]
    rel_sel = rel[plot_idx]
    rel_k_sel = rel_k[plot_idx] if rel_k is not None else None

    pred_top_set = set(int(j) for j in top_pred_idx)
    gt_top_set = set(int(j) for j in top_gt_idx)
    leaf_cat = []
    for j in plot_idx:
      j = int(j)
      in_pred = j in pred_top_set
      in_gt = j in gt_top_set
      if in_pred and in_gt:
        leaf_cat.append("TP")
      elif in_gt:
        leaf_cat.append("FN")
      elif in_pred:
        leaf_cat.append("FP")
      else:
        leaf_cat.append("TN")

    # Load JSON node metadata for OBB axes.
    json_nodes = _load_json_graph(str(args.data_root), str(ex.scene_id))

    # Extract points for selected nodes (subsample per node for plotting).
    pts = b.pos_subgraph.detach().cpu().numpy().astype(np.float32)
    nid = b.node_id_per_point.detach().cpu().numpy().astype(np.int64)

    def _sample_points_for_node(node_idx: int, max_pts: int = 2500, rng: Optional[np.random.Generator] = None) -> np.ndarray:
      mask = (nid == int(node_idx))
      idxs = np.nonzero(mask)[0]
      if idxs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
      if idxs.size <= max_pts:
        return pts[idxs]
      if rng is None:
        sel = np.random.choice(idxs, size=max_pts, replace=False)
      else:
        sel = rng.choice(idxs, size=max_pts, replace=False)
      return pts[sel]

    fruit_pts = _sample_points_for_node(fruit_node, max_pts=4000)
    leaf_pts_list = [_sample_points_for_node(n, max_pts=2500) for n in leaf_nodes_sel]

    # Extra faint "context" leaves (ignored by the top-K plots) to show local clutter.
    ctx_leaf_nodes: List[int] = []
    ctx_leaf_iids: List[int] = []
    ctx_pts = np.zeros((0, 3), dtype=np.float32)
    if int(args.ctx_leaves) > 0 and n_leaves > 0:
      exclude = set(int(j) for j in plot_idx)
      remain = [int(j) for j in range(int(n_leaves)) if int(j) not in exclude]
      if remain:
        k_ctx = int(min(int(args.ctx_leaves), len(remain)))
        # Use a dedicated RNG so context sampling doesn't perturb other randomness.
        scene_h = int(zlib.crc32(str(ex.scene_id).encode("utf-8")))
        seed_ctx = (int(args.seed) * 1000003 + scene_h * 97 + int(fruit_iid) * 31 + int(dir_hi)) & 0xFFFFFFFF
        rng_ctx = np.random.default_rng(int(seed_ctx))
        ctx_idx = rng_ctx.choice(np.asarray(remain, dtype=np.int64), size=k_ctx, replace=False)
        ctx_idx = [int(j) for j in np.asarray(ctx_idx).tolist()]
        ctx_leaf_nodes = [int(leaf_nodes[j]) for j in ctx_idx]
        ctx_leaf_iids = [int(leaf_iids[j]) for j in ctx_idx]
        ctx_pts_list = [_sample_points_for_node(n, max_pts=int(args.ctx_max_pts), rng=rng_ctx) for n in ctx_leaf_nodes]
        ctx_pts_list = [p for p in ctx_pts_list if p.size]
        if ctx_pts_list:
          ctx_pts = np.concatenate(ctx_pts_list, axis=0)

    # Pick a stem: nearest stem by JSON type if available (fallback: none).
    stem_node_idx: Optional[int] = None
    # Use point-derived centroids for plotting (keeps the viz consistent with what the model actually "sees").
    fruit_center = fruit_pts.mean(axis=0).astype(np.float32) if fruit_pts.size else b.json_geom[fruit_node, :3].detach().cpu().numpy().astype(np.float32)
    stem_cands: List[Tuple[float, int]] = []
    for nidx in range(int(b.y_node.numel())):
      iid = int(node_iid[int(nidx)])
      meta = json_nodes.get(iid)
      if meta is None:
        continue
      if str(meta.get("type", "")).lower() != "stem":
        continue
      c = b.json_geom[nidx, :3].detach().cpu().numpy().astype(np.float32)
      d = float(np.linalg.norm((c[:2] - fruit_center[:2])))
      stem_cands.append((d, int(nidx)))
    if stem_cands:
      stem_cands.sort(key=lambda t: t[0])
      stem_node_idx = int(stem_cands[0][1])
    stem_pts = _sample_points_for_node(stem_node_idx, max_pts=4000) if stem_node_idx is not None else np.zeros((0, 3), dtype=np.float32)
    stem_center = stem_pts.mean(axis=0).astype(np.float32) if stem_pts.size else None

    # ---- Output folder for this example ----
    tag = str(ex.perf_tag or "unk")
    ex_dir = out_dir / f"example_{i:03d}_perf_{tag}_{ex.scene_id}_fruit_{fruit_iid}"
    ex_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plot (A): 3D view (separate, larger) ----
    fig3d = plt.figure(figsize=(8.5, 7.5))
    ax3d = fig3d.add_subplot(1, 1, 1, projection="3d")
    dir_lbl = dir_labels[int(dir_hi)] if (dir_labels and int(dir_hi) < len(dir_labels)) else str(dir_hi)
    ax3d.set_title(f"{ex.scene_id}  fruit={fruit_iid}  dir={dir_lbl} (idx={dir_hi})", pad=14)
    if stem_pts.size:
      ax3d.scatter(stem_pts[:, 0], stem_pts[:, 1], stem_pts[:, 2], s=1, c="#7a5230", alpha=0.15, label="stem")
    if fruit_pts.size:
      ax3d.scatter(fruit_pts[:, 0], fruit_pts[:, 1], fruit_pts[:, 2], s=1, c="#d62728", alpha=0.25, label="fruit")
    if ctx_pts.size:
      ax3d.scatter(ctx_pts[:, 0], ctx_pts[:, 1], ctx_pts[:, 2], s=1, c="#9e9e9e", alpha=float(args.ctx_opacity))
    # Leaf colors in 3D indicate GT vs pred importance for the chosen direction:
    #   TP (in top-K GT and top-K pred): green
    #   FN (top-K GT only): orange
    #   FP (top-K pred only): purple
    #   TN (neither; contrast leaf): gray
    base_cols = {
      "TP": (0.2, 0.8, 0.2),
      "FN": (1.0, 0.55, 0.0),
      "FP": (0.6, 0.25, 0.85),
      "TN": (0.55, 0.55, 0.55),
    }
    rel_max = float(rel_sel.max()) if rel_sel.size else 0.0
    pmax = float(p_rank_sel.max()) if p_rank_sel.size else 0.0

    leaf_centers_sel: List[Optional[np.ndarray]] = []
    leaf_iid_to_color: Dict[int, Tuple[float, float, float]] = {}

    for lp, liid, psc, prk, rgt, cat in zip(leaf_pts_list, leaf_iids_sel, pot_pr_sel, p_rank_sel, rel_sel, leaf_cat):
      liid = int(liid)
      if lp.size:
        lc = lp.mean(axis=0).astype(np.float32)
      else:
        meta = json_nodes.get(int(liid))
        if meta is not None and meta.get("center") is not None:
          lc = np.asarray(meta["center"], dtype=np.float32)
        else:
          lc = None
      leaf_centers_sel.append(lc)

      if lp.size == 0:
        # Nothing to scatter; still keep center for edge drawing if available.
        leaf_iid_to_color[liid] = base_cols.get(str(cat), (0.55, 0.55, 0.55))
        continue
      base = base_cols.get(str(cat), (0.55, 0.55, 0.55))
      gt_scale = float(rgt / (rel_max + 1e-12)) if rel_max > 0 else 0.0
      pr_scale = float(prk / (pmax + 1e-12)) if pmax > 0 else 0.0
      if cat == "TP":
        scale = max(gt_scale, pr_scale)
      elif cat == "FN":
        # For FN, scale intensity by *model* importance (how close it was to picking it),
        # rather than by GT relevance (which would always look "important").
        scale = pr_scale
      elif cat == "FP":
        scale = pr_scale
      else:
        scale = 0.0
      mul = 0.3 + 0.7 * float(np.clip(scale, 0.0, 1.0))
      col = (base[0] * mul, base[1] * mul, base[2] * mul)
      ax3d.scatter(lp[:, 0], lp[:, 1], lp[:, 2], s=1, c=[col], alpha=0.22)
      leaf_iid_to_color[liid] = base

    # OBB wireframes (from JSON when available).
    def _maybe_draw_obb(iid: int, *, color: str, lw: float) -> None:
      meta = json_nodes.get(int(iid))
      if meta is None:
        return
      c = meta.get("center")
      axes = meta.get("axes")
      ext = meta.get("extent") or meta.get("extents")
      if c is None or axes is None or ext is None:
        return
      corners = _obb_corners(c, axes, ext)
      _draw_obb(ax3d, corners, color=color, lw=lw, alpha=0.9)

    _maybe_draw_obb(fruit_iid, color="#d62728", lw=2.0)
    if stem_node_idx is not None:
      _maybe_draw_obb(int(node_iid[int(stem_node_idx)]), color="#7a5230", lw=1.5)
    for liid in leaf_iids_sel:
      col = leaf_iid_to_color.get(int(liid), (0.2, 0.8, 0.2))
      _maybe_draw_obb(liid, color=col, lw=1.2)

    # Precompute a fruit-local frame in world coords (for triad + direction arrow + camera).
    z_hat = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if stem_center is None:
      x_hat = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
      d = (np.asarray(stem_center, dtype=np.float32) - np.asarray(fruit_center, dtype=np.float32))
      d[2] = 0.0
      n = float(np.linalg.norm(d))
      x_hat = (d / n) if n > 1e-6 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    y_hat = np.cross(z_hat, x_hat)
    ny = float(np.linalg.norm(y_hat))
    y_hat = (y_hat / ny) if ny > 1e-6 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    x_hat = np.cross(y_hat, z_hat)

    # Fruit size proxy for scaling arrows.
    if fruit_pts.size:
      rr = float(np.sqrt(((fruit_pts - fruit_center[None, :]) ** 2).sum(axis=1)).max())
    else:
      rr = 0.05

    # XYZ triad anchored near fruit (slightly above fruit to avoid occlusion).
    triad_len = float(max(0.03, float(args.triad_scale) * rr))
    triad_o = fruit_center + z_hat * float(0.8 * rr + 0.01)
    triad = [
      ("x", x_hat, "#d62728"),
      ("y", y_hat, "#2ca02c"),
      ("z", z_hat, "#1f77b4"),
    ]
    for name, v, col in triad:
      e = triad_o + triad_len * v
      ax3d.plot([triad_o[0], e[0]], [triad_o[1], e[1]], [triad_o[2], e[2]], c=col, lw=2.5, alpha=0.95)
      ax3d.text(e[0], e[1], e[2], name, color=col, fontsize=10)

    # Direction arrow (chosen direction) in world coords.
    # We draw it as a "view direction" arrow coming *into* the fruit (ending just
    # in front of the fruit surface), which reads better than an outward ray.
    v_world = None
    if dirs_local is not None and int(dir_hi) < int(dirs_local.shape[0]):
      v_world = _dir_local_to_world(dirs_local[int(dir_hi)], fruit_center_xyz=fruit_center, stem_center_xyz=stem_center)
      arrow_len = float(np.clip(3.0 * rr, 0.10, 0.25))
      tip_off = float(np.clip(1.2 * rr, 0.02, 0.06))
      tip = fruit_center - tip_off * v_world
      start = tip - arrow_len * v_world
      try:
        ax3d.quiver(
          float(start[0]),
          float(start[1]),
          float(start[2]),
          float(v_world[0]),
          float(v_world[1]),
          float(v_world[2]),
          length=float(arrow_len),
          normalize=True,
          color="#111111",
          linewidth=1.6,
          arrow_length_ratio=0.25,
          alpha=0.9,
        )
      except Exception:
        ax3d.plot([start[0], tip[0]], [start[1], tip[1]], [start[2], tip[2]], c="#111111", lw=1.8, alpha=0.9)

    # Optional: occlusion plane orthogonal to the chosen direction, through the fruit center.
    if int(args.plot_plane) == 1 and v_world is not None:
      nrm = np.asarray(v_world, dtype=np.float32)
      # Choose a stable basis for the plane.
      u = np.cross(nrm, z_hat)
      if float(np.linalg.norm(u)) < 1e-6:
        u = np.cross(nrm, x_hat)
      u = u / (float(np.linalg.norm(u)) + 1e-12)
      v = np.cross(nrm, u)
      plane_size = float(np.clip(4.0 * rr, 0.15, 0.6))
      grid = np.linspace(-plane_size, plane_size, 6, dtype=np.float32)
      S, T = np.meshgrid(grid, grid)
      X = fruit_center[0] + S * u[0] + T * v[0]
      Y = fruit_center[1] + S * u[1] + T * v[1]
      Z = fruit_center[2] + S * u[2] + T * v[2]
      ax3d.plot_surface(X, Y, Z, color="#bdbdbd", alpha=0.08, linewidth=0, antialiased=True, shade=False)

    # Leaf->fruit edges for chosen direction.
    # Solid edges: model top-K predicted leaves; width scales with predicted potential.
    # Dashed edges: GT top-K leaves that the model *missed* (FN); width scales with GT relevance.
    fruit_c = fruit_center
    rel_norm = rel_sel / (rel_max + 1e-12) if rel_max > 0 else np.zeros_like(rel_sel)
    for liid, cat, psc, rn, lc in zip(leaf_iids_sel, leaf_cat, pot_pr_sel, rel_norm, leaf_centers_sel):
      if lc is None:
        continue
      col = base_cols.get(str(cat), (0.55, 0.55, 0.55))
      if cat in ("TP", "FP"):
        w = 0.5 + 4.0 * float(psc)
        ax3d.plot([lc[0], fruit_c[0]], [lc[1], fruit_c[1]], [lc[2], fruit_c[2]], c=col, lw=w, alpha=0.35)
      elif cat == "FN":
        w = 0.5 + 4.0 * float(rn)
        ax3d.plot([lc[0], fruit_c[0]], [lc[1], fruit_c[1]], [lc[2], fruit_c[2]], c=col, lw=w, alpha=0.25, linestyle="--")

    # Make the 3D plot look a bit more "paper-like".
    ax3d.grid(False)
    try:
      ax3d.set_box_aspect((1, 1, 1))
    except Exception:
      pass
    # Hide numeric axes/ticks; triad provides orientation.
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_xlabel("")
    ax3d.set_ylabel("")
    ax3d.set_zlabel("")
    try:
      ax3d.xaxis.pane.set_visible(False)
      ax3d.yaxis.pane.set_visible(False)
      ax3d.zaxis.pane.set_visible(False)
    except Exception:
      pass
    ax3d.legend(loc="upper left", fontsize=9, frameon=False)

    # Equal-ish axis scaling for better depth perception.
    try:
      to_cat = [fruit_pts, stem_pts] + [lp for lp in leaf_pts_list if lp.size]
      if ctx_pts.size:
        to_cat.append(ctx_pts)
      xs = np.concatenate([a[:, 0] for a in to_cat]) if fruit_pts.size else None
      ys = np.concatenate([a[:, 1] for a in to_cat]) if fruit_pts.size else None
      zs = np.concatenate([a[:, 2] for a in to_cat]) if fruit_pts.size else None
      if xs is not None and xs.size:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        z0, z1 = float(zs.min()), float(zs.max())
        cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
        r = max(x1 - x0, y1 - y0, z1 - z0) / 2.0
        ax3d.set_xlim(cx - r, cx + r)
        ax3d.set_ylim(cy - r, cy + r)
        ax3d.set_zlim(cz - r, cz + r)
    except Exception:
      pass

    # Viewpoint.
    try:
      if str(args.cam_align) == "dir" and v_world is not None:
        cam = -np.asarray(v_world, dtype=np.float32)
        az = math.degrees(math.atan2(float(cam[1]), float(cam[0])))
        el = math.degrees(math.atan2(float(cam[2]), float(np.sqrt(float(cam[0] ** 2 + cam[1] ** 2)) + 1e-12)))
        ax3d.view_init(elev=float(el), azim=float(az))
      else:
        ax3d.view_init(elev=float(args.cam_elev), azim=float(args.cam_azim))
    except Exception:
      pass

    png_3d = ex_dir / "3d.png"
    fig3d.tight_layout()
    fig3d.savefig(png_3d, dpi=200, bbox_inches="tight")
    plt.close(fig3d)

    # Optional: interactive Plotly 3D view (precise orbit/pan/zoom control).
    html_3d = None
    try:
      import plotly.graph_objects as go  # type: ignore

      def _rgb_str(col: Tuple[float, float, float]) -> str:
        r = int(np.clip(col[0], 0.0, 1.0) * 255.0)
        g = int(np.clip(col[1], 0.0, 1.0) * 255.0)
        b = int(np.clip(col[2], 0.0, 1.0) * 255.0)
        return f"rgb({r},{g},{b})"

      def _center_pts(p: np.ndarray) -> np.ndarray:
        if p.size == 0:
          return p
        return p - fruit_center[None, :]

      figp = go.Figure()
      r_vis = float(np.clip(4.0 * rr, 0.15, 0.6))

      stem_c = _center_pts(stem_pts)
      fruit_cpts = _center_pts(fruit_pts)
      if stem_c.size:
        figp.add_trace(go.Scatter3d(
          x=stem_c[:, 0], y=stem_c[:, 1], z=stem_c[:, 2],
          mode="markers",
          marker=dict(size=2, color="#7a5230", opacity=0.15),
          name="stem",
        ))
      if fruit_cpts.size:
        figp.add_trace(go.Scatter3d(
          x=fruit_cpts[:, 0], y=fruit_cpts[:, 1], z=fruit_cpts[:, 2],
          mode="markers",
          marker=dict(size=2, color="#d62728", opacity=0.25),
          name="fruit",
        ))

      # Occlusion plane (toggleable) in centered coordinates.
      # Draw early so it tends to sit "behind" points/edges.
      if int(args.plot_plane) == 1 and v_world is not None:
        nrm = np.asarray(v_world, dtype=np.float32)
        u = np.cross(nrm, z_hat)
        if float(np.linalg.norm(u)) < 1e-6:
          u = np.cross(nrm, x_hat)
        u = u / (float(np.linalg.norm(u)) + 1e-12)
        v = np.cross(nrm, u)
        s = float(r_vis)
        c00 = (-s * u - s * v).astype(np.float32)
        c01 = (-s * u + s * v).astype(np.float32)
        c10 = (s * u - s * v).astype(np.float32)
        c11 = (s * u + s * v).astype(np.float32)
        V = np.stack([c00, c01, c10, c11], axis=0)
        figp.add_trace(go.Mesh3d(
          x=V[:, 0], y=V[:, 1], z=V[:, 2],
          i=[0, 1], j=[1, 3], k=[2, 2],
          color="rgb(190,190,190)",
          opacity=0.12,
          name="occlusion plane",
          legendgroup="plane",
          showlegend=True,
          flatshading=True,
          lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        ))

      # Extra nearby context leaves (faint gray, toggleable).
      if ctx_pts.size:
        ctx_c = _center_pts(ctx_pts)
        figp.add_trace(go.Scatter3d(
          x=ctx_c[:, 0], y=ctx_c[:, 1], z=ctx_c[:, 2],
          mode="markers",
          marker=dict(size=2, color="#9e9e9e", opacity=float(args.ctx_opacity)),
          name=f"context leaves (n={len(ctx_leaf_iids)})",
          legendgroup="context",
          showlegend=True,
        ))

      # Leaves (selected set), grouped by TP/FP/FN/TN for clean toggles.
      cat_order = ["TP", "FN", "FP", "TN"]
      cat_names = {"TP": "leaves TP", "FN": "leaves FN (missed)", "FP": "leaves FP", "TN": "leaves TN"}
      cat_pts: Dict[str, List[np.ndarray]] = {c: [] for c in cat_order}
      cat_cols: Dict[str, List[np.ndarray]] = {c: [] for c in cat_order}
      for lp, cat, prk in zip(leaf_pts_list, leaf_cat, p_rank_sel):
        if lp.size == 0:
          continue
        cat = str(cat)
        if cat not in cat_pts:
          cat = "TN"
        lpc = _center_pts(lp)
        cat_pts[cat].append(lpc)
        if cat == "FN":
          # Scale FN intensity by model rank-prob mass: brighter means "model almost picked it".
          scale = float(prk / (pmax + 1e-12)) if float(pmax) > 0 else 0.0
          mul = 0.3 + 0.7 * float(np.clip(scale, 0.0, 1.0))
          base = base_cols.get("FN", (1.0, 0.55, 0.0))
          col = (base[0] * mul, base[1] * mul, base[2] * mul)
          col_str = _rgb_str(col)
          cat_cols[cat].append(np.full((int(lpc.shape[0]),), col_str, dtype=object))
        else:
          base = base_cols.get(str(cat), (0.55, 0.55, 0.55))
          col_str = _rgb_str(base)
          cat_cols[cat].append(np.full((int(lpc.shape[0]),), col_str, dtype=object))
      for cat in cat_order:
        if not cat_pts[cat]:
          continue
        pcat = np.concatenate(cat_pts[cat], axis=0)
        ccat = np.concatenate(cat_cols[cat], axis=0)
        base = base_cols.get(str(cat), (0.55, 0.55, 0.55))
        figp.add_trace(go.Scatter3d(
          x=pcat[:, 0], y=pcat[:, 1], z=pcat[:, 2],
          mode="markers",
          marker=dict(size=2, color=ccat, opacity=0.22),
          name=cat_names.get(cat, f"leaves {cat}"),
          legendgroup=f"leaves_{cat}",
          showlegend=True,
        ))

      # Edges (centered).
      pred_edge_legend = True
      miss_edge_legend = True
      for liid, cat, psc, rn, lc in zip(leaf_iids_sel, leaf_cat, pot_pr_sel, rel_norm, leaf_centers_sel):
        if lc is None:
          continue
        lc_c = np.asarray(lc, dtype=np.float32) - fruit_center
        col = base_cols.get(str(cat), (0.55, 0.55, 0.55))
        if cat in ("TP", "FP"):
          w = 1.0 + 6.0 * float(psc)
          dash = None
          alpha = 0.35
          showleg = bool(pred_edge_legend)
          pred_edge_legend = False
          name = "pred edges"
          group = "pred_edges"
        elif cat == "FN":
          w = 1.0 + 6.0 * float(rn)
          dash = "dash"
          alpha = 0.25
          showleg = bool(miss_edge_legend)
          miss_edge_legend = False
          name = "missed GT edges"
          group = "missed_edges"
        else:
          continue
        figp.add_trace(go.Scatter3d(
          x=[float(lc_c[0]), 0.0],
          y=[float(lc_c[1]), 0.0],
          z=[float(lc_c[2]), 0.0],
          mode="lines",
          line=dict(color=_rgb_str(col), width=float(w), dash=dash),
          opacity=float(alpha),
          name=name,
          legendgroup=group,
          showlegend=showleg,
        ))

      # Triad + direction arrow (centered).
      triad_o_c = triad_o - fruit_center
      triad_legend = True
      for name, v, col in triad:
        e = triad_o_c + triad_len * v
        figp.add_trace(go.Scatter3d(
          x=[float(triad_o_c[0]), float(e[0])],
          y=[float(triad_o_c[1]), float(e[1])],
          z=[float(triad_o_c[2]), float(e[2])],
          mode="lines+markers",
          line=dict(color=col, width=6),
          marker=dict(size=2, color=col),
          name="xyz triad" if triad_legend else "xyz triad",
          legendgroup="triad",
          showlegend=bool(triad_legend),
        ))
        triad_legend = False
      if v_world is not None:
        arrow_len = float(np.clip(3.0 * rr, 0.10, 0.25))
        tip_off = float(np.clip(1.2 * rr, 0.02, 0.06))
        tip_c = (-tip_off * np.asarray(v_world, dtype=np.float32)).reshape(3)
        start_c = (tip_c - arrow_len * np.asarray(v_world, dtype=np.float32)).reshape(3)
        figp.add_trace(go.Scatter3d(
          x=[float(start_c[0]), float(tip_c[0])],
          y=[float(start_c[1]), float(tip_c[1])],
          z=[float(start_c[2]), float(tip_c[2])],
          mode="lines",
          line=dict(color="#111111", width=4),
          name="view dir",
          legendgroup="dir",
          showlegend=True,
        ))
        # Arrowhead at the tip (toggleable via legendgroup="dir").
        try:
          figp.add_trace(go.Cone(
            x=[float(tip_c[0])],
            y=[float(tip_c[1])],
            z=[float(tip_c[2])],
            u=[float(v_world[0])],
            v=[float(v_world[1])],
            w=[float(v_world[2])],
            anchor="tip",
            sizemode="absolute",
            sizeref=float(np.clip(0.9 * tip_off, 0.015, 0.05)),
            showscale=False,
            colorscale=[[0.0, "#111111"], [1.0, "#111111"]],
            opacity=1.0,
            name="view dir head",
            legendgroup="dir",
            showlegend=False,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
          ))
        except Exception:
          pass

      # Camera aligned with the chosen direction.
      if v_world is not None:
        cam = -np.asarray(v_world, dtype=np.float32)
        eye = cam * r_vis + z_hat * float(0.35 * rr)
      else:
        eye = np.asarray([r_vis, r_vis, r_vis], dtype=np.float32)
      camera = dict(
        up=dict(x=0.0, y=0.0, z=1.0),
        eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
      )

      figp.update_layout(
        title=f"{ex.scene_id}  fruit={fruit_iid}  dir={dir_lbl} (idx={dir_hi})",
        showlegend=True,
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
        scene=dict(
          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
          zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
          aspectmode="data",
          camera=camera,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
      )

      html_3d = ex_dir / "3d.html"
      figp.write_html(
        str(html_3d),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
      )
    except Exception:
      html_3d = None

    # ---- Plot (B): Union per direction (separate, wide) ----
    figu, axu = plt.subplots(figsize=(13.5, 3.7))
    axu.set_title("Union per direction (GT vs pred)")
    xs = np.arange(D)
    # Make the bar chart visually compact:
    # - tighter x-limits (remove whitespace before first/after last bar)
    # - slightly thinner bars than default without making inter-direction gaps huge
    bar_w = 0.30
    off = bar_w / 2.0
    axu.bar(xs - off, union_gt, width=bar_w, color="#6e6e6e", alpha=0.55, label="GT")
    axu.bar(xs + off, union_pr, width=bar_w, color="#ff7f0e", alpha=0.55, label="pred")
    axu.set_xlabel("direction")
    axu.set_ylabel("union")
    axu.set_ylim(0.0, 1.05)
    # Range matches bar width so we don't waste space at the ends.
    axu.set_xlim(-bar_w - 0.05, (D - 1) + bar_w + 0.05)
    axu.margins(x=0.0)

    # Direction tick labels (human-friendly).
    labels = dir_labels[:D] if dir_labels else [str(i) for i in xs.tolist()]
    axu.set_xticks(xs)
    axu.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Bold label for the visualized direction (bottom).
    vis_lbl = labels[int(dir_hi)] if int(dir_hi) < len(labels) else str(dir_hi)
    figu.text(0.5, 0.01, f"visualized direction: {vis_lbl} (idx={dir_hi})", ha="center", va="bottom", fontsize=12, fontweight="bold")

    axu.legend(fontsize=9, frameon=False, loc="upper right")
    figu.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    png_union = ex_dir / "union.png"
    figu.savefig(png_union, dpi=200, bbox_inches="tight")
    plt.close(figu)

    # ---- Plot (C): Leaf ranking for chosen direction (separate) ----
    figr, axr = plt.subplots(figsize=(7.2, 6.0))
    if edl_rank:
      axr.set_title(
        f"Leaf ranking (dir={vis_lbl}, idx={dir_hi})\n"
        f"EDL rank uncertainty: u={u_rank:.3f}, S={S_rank:.1f}"
      )
    else:
      axr.set_title(f"Leaf ranking (dir={vis_lbl}, idx={dir_hi})")
    y = np.arange(len(leaf_iids_sel))
    # GT relevance (stacked k1/k2/k3 if available).
    if rel_k_sel is not None:
      k1 = rel_k_sel[:, 0]
      k2 = rel_k_sel[:, 1]
      k3 = rel_k_sel[:, 2]
      axr.barh(y, k1, color="#2ca02c", alpha=0.35, label="GT k1")
      axr.barh(y, k2, left=k1, color="#2ca02c", alpha=0.20, label="GT k2")
      axr.barh(y, k3, left=(k1 + k2), color="#2ca02c", alpha=0.10, label="GT k3")
    else:
      axr.barh(y, rel_sel, color="#2ca02c", alpha=0.25, label="GT rel")
    # Pred rank prob (marker) and pred potential (marker).
    axr.plot(p_rank_sel, y, "o", color="#1f77b4", label="pred rank p")
    axr.plot(pot_pr_sel, y, "x", color="#d62728", label="pred potential")
    axr.set_yticks(y)
    axr.set_yticklabels([str(liid) for liid in leaf_iids_sel], fontsize=8)
    axr.set_xlabel("score / mass")
    axr.invert_yaxis()
    axr.set_xlim(0.0, max(1.0, float(rel_sel.max()) + 0.05))
    axr.legend(fontsize=9, loc="lower right", frameon=False)
    figr.tight_layout()
    png_rank = ex_dir / "ranking.png"
    figr.savefig(png_rank, dpi=200, bbox_inches="tight")
    plt.close(figr)

    # ---- Plot (D): Rank uncertainty per direction (EDL only) ----
    # Keep this plot small and consistent with the union-per-direction plot.
    if edl_rank and u_rank_per_dir is not None:
      figq, axq = plt.subplots(figsize=(13.5, 3.2))
      axq.set_title("Rank uncertainty per direction (EDL)")
      xsq = np.arange(D)
      labels = dir_labels[:D] if dir_labels else [str(i) for i in xsq.tolist()]

      # Bars, with the chosen direction highlighted.
      base_c = "#1f77b4"
      hi_c = "#d62728"
      cols = [base_c] * D
      if 0 <= int(dir_hi) < D:
        cols[int(dir_hi)] = hi_c
      axq.bar(xsq, u_rank_per_dir[:D], color=cols, alpha=0.75, width=0.55)

      axq.set_xlabel("direction")
      axq.set_ylabel(r"$u = L / S$")
      axq.set_ylim(0.0, 1.05)
      axq.set_xlim(-0.5, (D - 1) + 0.5)
      axq.margins(x=0.0)
      axq.set_xticks(xsq)
      axq.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

      # Add a compact annotation for the chosen direction.
      if 0 <= int(dir_hi) < D:
        axq.text(
          0.01,
          0.98,
          f"chosen: {labels[int(dir_hi)]} (idx={dir_hi})",
          transform=axq.transAxes,
          ha="left",
          va="top",
          fontsize=10,
          color="#111111",
        )

      figq.tight_layout()
      png_u = ex_dir / "rank_uncertainty.png"
      figq.savefig(png_u, dpi=200, bbox_inches="tight")
      plt.close(figq)
    else:
      png_u = None

    exported.append(
      {
        "rank": i,
        "score": ex.score,
        "perf": ex.perf,
        "union_mae_dom": ex.union_mae_dom,
        "perf_tag": ex.perf_tag,
        "scene_id": ex.scene_id,
        "fruit_iid": fruit_iid,
        "dir_hi": dir_hi,
        "explain_mean_dom": ex.explain_mean_dom,
        "ndcg3_dom": ex.ndcg3_dom,
        "ndcg3_hi": ex.ndcg3_hi,
        "n_cand_leaves": int(len(leaf_iids)),
        "n_ctx_leaves": int(len(ctx_leaf_iids)),
        "out_dir": str(ex_dir),
        "png_3d": str(png_3d),
        "png_union": str(png_union),
        "png_rank": str(png_rank),
        "png_rank_uncertainty": str(png_u) if png_u is not None else None,
        "html_3d": str(html_3d) if html_3d is not None else None,
      }
    )

  # Write summary.
  with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
    json.dump(
      {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt_path),
        "split": split,
        "union_tau": union_tau,
        "pot_eps": pot_eps,
        "edl_rank": edl_rank,
        "filters": {
          "min_explain": float(args.min_explain),
          "min_cand_leaves": int(args.min_cand_leaves),
          "min_dom_dirs": int(args.min_dom_dirs),
          "max_top1_share": float(args.max_top1_share),
        },
        "pick": {
          "mode": str(args.pick_mode),
          "pool": int(args.pick_pool),
        },
        "viz": {
          "topk_plot": int(args.topk_plot),
          "include_worst_leaf": int(args.include_worst_leaf),
          "ctx_leaves": int(args.ctx_leaves),
          "ctx_max_pts": int(args.ctx_max_pts),
          "ctx_opacity": float(args.ctx_opacity),
          "plot_plane": int(args.plot_plane),
        },
        "examples": exported,
      },
      f,
      indent=2,
      sort_keys=True,
    )
  print(f"Wrote {len(exported)} examples to {out_dir}", flush=True)


if __name__ == "__main__":
  main()
