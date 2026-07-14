#!/usr/bin/env python3
"""
Profile oracle model forward runtime breakdown.

Goal: quantify relative cost of
  - PointNet++ MSG encoder (shared_encoder)
  - candidate edge generation (build_candidates)
  - GNN backbone (GINE / ResGINE)
  - occlusion forward (total) and leaf-set self-attention (Transformer encoder)

Notes:
  - Run as a module (from repo root) so `pepper_scene_graphs` is importable:
      python -u -m scripts.debug.profile_oracle_runtime_breakdown ...
  - Uses CUDA events when running on GPU; falls back to CPU timers otherwise.
  - Measures forward only (inference-like) by default; `--mode train` measures forward in train mode.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn
from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline, OracleResBaseline


class _Timer:
  def __init__(self, device: torch.device):
    self.device = device
    self.use_cuda = (device.type == "cuda")
    self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    self._durations_ms: Dict[str, List[float]] = defaultdict(list)  # for CPU

  def hook_module(self, module: torch.nn.Module, label: str) -> None:
    if self.use_cuda:
      def _pre(m: torch.nn.Module, _inp: Tuple[Any, ...]) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        setattr(m, f"_prof_start_{label}", start)

      def _post(m: torch.nn.Module, _inp: Tuple[Any, ...], _out: Any) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        start = getattr(m, f"_prof_start_{label}")
        self._events[label].append((start, end))

    else:
      def _pre(m: torch.nn.Module, _inp: Tuple[Any, ...]) -> None:
        setattr(m, f"_prof_t0_{label}", time.perf_counter())

      def _post(m: torch.nn.Module, _inp: Tuple[Any, ...], _out: Any) -> None:
        t0 = getattr(m, f"_prof_t0_{label}")
        self._durations_ms[label].append((time.perf_counter() - t0) * 1e3)

    module.register_forward_pre_hook(_pre)
    module.register_forward_hook(_post)

  def wrap_method(self, obj: Any, method_name: str, label: str) -> None:
    """Monkeypatch `obj.method_name` to time it under `label`."""
    orig = getattr(obj, method_name)

    if self.use_cuda:
      def _wrapped(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig(*args, **kwargs)
        end.record()
        self._events[label].append((start, end))
        return out
    else:
      def _wrapped(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        self._durations_ms[label].append((time.perf_counter() - t0) * 1e3)
        return out

    # Bind to instance.
    import types
    setattr(obj, method_name, types.MethodType(_wrapped, obj))

  def clear(self) -> None:
    self._events.clear()
    self._durations_ms.clear()

  def synchronize(self) -> None:
    if self.use_cuda:
      torch.cuda.synchronize(self.device)

  def get_ms(self, label: str) -> List[float]:
    if self.use_cuda:
      evs = self._events.get(label, [])
      return [s.elapsed_time(e) for (s, e) in evs]
    return self._durations_ms.get(label, [])


def _mean_std(xs: List[float]) -> Tuple[float, float]:
  if not xs:
    return 0.0, 0.0
  if len(xs) == 1:
    return float(xs[0]), 0.0
  return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _make_cfg(args: argparse.Namespace) -> SimpleNamespace:
  # Keep aligned with scripts/train_oracle.py defaults (only the model fields we use).
  return SimpleNamespace(
    model=SimpleNamespace(
      hidden_dim=128,
      num_layers=4,
      dropout=0.1,
      oracle_msg_npoint=args.npoint,
      oracle_msg_radius_list=[0.01, 0.02, 0.04],
      oracle_msg_nsample_list=[16, 32, 64],
      node_feat_mode=args.node_feat_mode,
      k_candidates=args.k_candidates,
      dist_threshold=args.dist_threshold,
      extra_stems_per_node=args.extra_stems_per_node,
      use_normals=args.use_normals,
      use_rgb=args.use_rgb,
      no_geom_injection=args.no_geom_injection,
      occl_enabled=args.occl_enabled,
      occl_num_dirs=18,
      occl_dir_embed_dim=args.occl_dir_embed_dim,
      occl_attn_dim=args.occl_attn_dim,
      occl_attn_temp=args.occl_attn_temp,
      occl_leaf_sa_layers=args.occl_leaf_sa_layers,
      occl_leaf_sa_dim=args.occl_leaf_sa_dim,
      occl_leaf_sa_heads=args.occl_leaf_sa_heads,
      occl_leaf_sa_ff_mult=args.occl_leaf_sa_ff_mult,
      occl_leaf_sa_dropout=args.occl_leaf_sa_dropout,
    )
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--data_root", type=str, required=True)
  p.add_argument("--split", type=str, default="val", choices=["train", "val"])
  p.add_argument("--use_4mm", action="store_true", default=True)
  p.add_argument("--no_4mm", action="store_false", dest="use_4mm")

  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--subgraph_enabled", action="store_true", default=True)
  p.add_argument("--subgraph_size", type=int, default=256)
  p.add_argument("--subgraph_size_min", type=int, default=256)
  p.add_argument("--hard_pair_constraint", action="store_true", default=True)
  p.add_argument("--no_hard_pair_constraint", action="store_false", dest="hard_pair_constraint")

  p.add_argument("--model_type", type=str, default="res", choices=["baseline", "res"])
  p.add_argument("--node_feat_mode", type=str, default="local", choices=["gt", "local", "none"])
  p.add_argument("--no_geom_injection", action="store_true", default=True)
  p.add_argument("--drop_shoots", action="store_true", default=True)

  p.add_argument("--npoint", type=int, default=256)
  p.add_argument("--k_candidates", type=int, default=8)
  p.add_argument("--extra_stems_per_node", type=int, default=2)
  p.add_argument("--dist_threshold", type=float, default=0.15)

  p.add_argument("--use_rgb", action="store_true", default=True)
  p.add_argument("--no_rgb", action="store_false", dest="use_rgb")
  p.add_argument("--use_normals", action="store_true", default=True)
  p.add_argument("--no_normals", action="store_false", dest="use_normals")

  # Occlusion toggles (to make sure we profile the occl path that your runs use).
  p.add_argument("--occl_aug_enabled", action="store_true", default=True)
  p.add_argument("--occl_sidecar_dir", type=str, default=None)
  p.add_argument("--occl_aug_num_fruits", type=int, default=4)
  p.add_argument("--occl_aug_union_tau", type=float, default=0.5)
  p.add_argument("--occl_aug_resample_tries", type=int, default=5)
  p.add_argument("--occl_aug_max_dist", type=float, default=0.2)
  p.add_argument("--occl_aug_max_pos_leaves", type=int, default=16)
  p.add_argument("--occl_aug_max_neg_leaves", type=int, default=8)
  p.add_argument("--occl_aug_max_total_leaves", type=int, default=96)

  p.add_argument("--occl_enabled", action="store_true", default=True)
  p.add_argument("--occl_max_leaves_per_fruit", type=int, default=64)
  p.add_argument("--occl_dir_embed_dim", type=int, default=16)
  p.add_argument("--occl_attn_dim", type=int, default=64)
  p.add_argument("--occl_attn_temp", type=float, default=1.0)
  p.add_argument("--occl_leaf_sa_layers", type=int, default=1)
  p.add_argument("--occl_leaf_sa_dim", type=int, default=128)
  p.add_argument("--occl_leaf_sa_heads", type=int, default=4)
  p.add_argument("--occl_leaf_sa_ff_mult", type=int, default=4)
  p.add_argument("--occl_leaf_sa_dropout", type=float, default=0.1)

  # Profiling.
  p.add_argument("--device", type=str, default="cuda")
  p.add_argument("--mode", type=str, default="eval", choices=["eval", "train"])
  p.add_argument("--warmup", type=int, default=10)
  p.add_argument("--iters", type=int, default=50)
  p.add_argument("--seed", type=int, default=0)

  args = p.parse_args()

  if args.node_feat_mode == "none":
    args.node_feat_mode = None

  torch.manual_seed(int(args.seed))

  device = torch.device(args.device)
  if device.type == "cuda" and (not torch.cuda.is_available()):
    print("[warn] CUDA requested but not available; falling back to CPU.", flush=True)
    device = torch.device("cpu")

  ds = OraclePepperDataset(
    data_root=args.data_root,
    split=args.split,
    npoint_per_instance=args.npoint,
    augment=False,
    rotate_aug=False,
    drop_shoots=args.drop_shoots,
    k_candidates=args.k_candidates,
    extra_stems_per_node=args.extra_stems_per_node,
    dist_threshold=args.dist_threshold,
    use_rgb=args.use_rgb,
    use_4mm=args.use_4mm,
    subgraph_enabled=args.subgraph_enabled,
    subgraph_size=args.subgraph_size,
    subgraph_size_min=args.subgraph_size_min,
    hard_pair_constraint=args.hard_pair_constraint,
    occl_aug_enabled=args.occl_aug_enabled,
    occl_sidecar_dir=args.occl_sidecar_dir,
    occl_aug_num_fruits=args.occl_aug_num_fruits,
    occl_aug_union_tau=args.occl_aug_union_tau,
    occl_aug_resample_tries=args.occl_aug_resample_tries,
    occl_aug_max_dist=args.occl_aug_max_dist,
    occl_aug_max_pos_leaves=args.occl_aug_max_pos_leaves,
    occl_aug_max_neg_leaves=args.occl_aug_max_neg_leaves,
    occl_aug_max_total_leaves=args.occl_aug_max_total_leaves,
    occl_targets_enabled=args.occl_enabled,
    occl_targets_max_leaves_per_fruit=args.occl_max_leaves_per_fruit,
  )

  loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=oracle_collate_fn)
  batch = next(iter(loader))

  cfg = _make_cfg(args)
  if args.model_type == "res":
    model = OracleResBaseline(cfg, node_classes=4, edge_labels=3)
  else:
    model = OracleGTBaseline(cfg, node_classes=4, edge_labels=3)
  model = model.to(device)

  if args.mode == "eval":
    model.eval()
  else:
    model.train()

  batch = batch.to(device)

  # Print batch stats so the timings are interpretable.
  n_nodes = int(batch.y_node.size(0))
  n_pts = int(batch.pos_subgraph.size(0))
  occl_f = int(getattr(batch, "occl_fruit_idx", torch.empty((0,))).numel())
  occl_m = int(getattr(batch, "occl_pair_index", torch.empty((2, 0))).size(1))
  print(
    f"[batch] split={args.split} nodes={n_nodes} points={n_pts} occl_fruits={occl_f} occl_pairs={occl_m} device={device}",
    flush=True,
  )

  # Instrumentation.
  timer = _Timer(device)
  timer.hook_module(model.shared_encoder, "pointnet_shared_encoder")
  timer.hook_module(model.backbone, "gnn_backbone")
  if getattr(model, "occl_leaf_sa", None) is not None:
    timer.hook_module(model.occl_leaf_sa, "occl_leaf_sa")

  # Useful method-level timings (not modules).
  timer.wrap_method(model, "build_candidates", "build_candidates")
  timer.wrap_method(model, "_forward_occl", "occl_forward_total")

  # Total forward.
  total_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
  total_ms_cpu: List[float] = []

  def _run_forward() -> None:
    _ = model(batch)

  # Warmup.
  with torch.no_grad() if args.mode == "eval" else torch.enable_grad():
    for _ in range(max(int(args.warmup), 0)):
      _run_forward()

  # Timed iterations.
  with torch.no_grad() if args.mode == "eval" else torch.enable_grad():
    for _ in range(max(int(args.iters), 1)):
      if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_forward()
        end.record()
        total_events.append((start, end))
      else:
        t0 = time.perf_counter()
        _run_forward()
        total_ms_cpu.append((time.perf_counter() - t0) * 1e3)

  timer.synchronize()

  if device.type == "cuda":
    total_ms = [s.elapsed_time(e) for (s, e) in total_events]
  else:
    total_ms = total_ms_cpu

  labels = [
    "pointnet_shared_encoder",
    "build_candidates",
    "gnn_backbone",
    "occl_forward_total",
    "occl_leaf_sa",
  ]

  print("\n[forward runtime breakdown] (mean ± std, ms)", flush=True)
  mean_total, std_total = _mean_std(total_ms)
  print(f"  total_forward:           {mean_total:8.2f} ± {std_total:6.2f}", flush=True)

  for lab in labels:
    xs = timer.get_ms(lab)
    m, s = _mean_std(xs)
    frac = (m / mean_total * 100.0) if mean_total > 0 else 0.0
    if xs:
      print(f"  {lab:22s} {m:8.2f} ± {s:6.2f}   ({frac:5.1f}%)", flush=True)

  print("\n[notes]", flush=True)
  print("  - Times are inclusive for each label (but these blocks are mostly sequential).", flush=True)
  print("  - occl_leaf_sa is a subset of occl_forward_total.", flush=True)
  print("  - build_candidates cost can be significant on GPU due to kNN/radius graph ops.", flush=True)


if __name__ == "__main__":
  main()

