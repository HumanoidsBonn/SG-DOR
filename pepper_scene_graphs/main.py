# pepper_scene_graphs/main.py

import argparse
import json
import os
import math
import shutil

from typing import Any, List, Tuple

import numpy as np
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import torch
import wandb
from torch_geometric.loader import DataLoader

from pepper_scene_graphs.configs.config import Config
from pepper_scene_graphs.utils.seed import set_seed
from pepper_scene_graphs.data.dataset import load_graph_dataset
from pepper_scene_graphs.models.joint_model import JointSceneGraphModel
from pepper_scene_graphs.training.trainer import train_one_epoch, evaluate
from pepper_scene_graphs.data.edge_builders import scheduled_tau
from pepper_scene_graphs.constants import NODE_CLASSES, EDGE_LABELS


HYPERPARAM_ALIASES = {
    "lr": ("train", "lr"),
    "bs": ("train", "batch_size"),
    "epochs": ("train", "epochs"),
    "drop": ("train", "dropout"),
    "hd": ("model", "hidden_dim"),
    "layers": ("model", "num_layers"),
    "k": ("train", "k"),
    "backbone": ("model", "backbone"),
}


def _apply_config_overrides(cfg: Config, param_entries: List[str]) -> List[Tuple[str, Any]]:
  """Apply --param overrides to cfg. Returns list of applied (alias, value)."""
  overrides: List[Tuple[str, Any]] = []
  for entry in param_entries:
    if "=" not in entry:
      raise ValueError(f"Invalid --param entry: {entry}")
    alias, raw_value = entry.split("=", 1)
    alias = alias.strip()
    if alias not in HYPERPARAM_ALIASES:
      raise ValueError(f"Unknown param alias: {alias}")
    target_path = HYPERPARAM_ALIASES[alias]
    try:
      if "." in raw_value or "e" in raw_value.lower():
        value: Any = float(raw_value)
      else:
        value = int(raw_value)
    except ValueError:
      value = raw_value
    target_obj: Any = cfg
    for attr in target_path[:-1]:
      target_obj = getattr(target_obj, attr)
    setattr(target_obj, target_path[-1], value)
    overrides.append((alias, value))
  return overrides


def _apply_cfg_dict(dst: Any, src: Any) -> None:
  """Recursively apply a dict (as written in run_dir/config.json) onto a Config object.

  This is intentionally permissive: unknown keys are ignored so older/newer configs
  can still be loaded.
  """
  if not isinstance(src, dict):
    return
  for k, v in src.items():
    if not hasattr(dst, k):
      continue
    cur = getattr(dst, k)
    if isinstance(cur, (int, float, str, bool)) or cur is None:
      setattr(dst, k, v)
    elif isinstance(cur, (list, tuple, dict)):
      setattr(dst, k, v)
    else:
      # Assume nested config object (dataclass-like).
      if isinstance(v, dict):
        _apply_cfg_dict(cur, v)
      else:
        setattr(dst, k, v)


def _find_run_dir_from_checkpoint(ckpt_path: Path, max_up: int = 5) -> Path:
  """Find the run directory containing config.json, starting from ckpt parent."""
  cur = ckpt_path.parent
  for _ in range(max_up + 1):
    if (cur / "config.json").exists():
      return cur
    if cur.parent == cur:
      break
    cur = cur.parent
  raise FileNotFoundError(f"Could not find config.json near checkpoint: {ckpt_path}")


def format_param_value(value: Any) -> str:
    """Format parameter value for run naming.
    
    Replaces decimals with 'p' to avoid filesystem issues.
    """
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def _json_safe(obj: Any) -> Any:
    """Recursively convert tensors/arrays into JSON-serializable types."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _add_prefixed_metrics(dest: dict, prefix: str, metrics: dict, skip_suffixes: Tuple[str, ...] = ("_conf",)) -> None:
  """Add metrics to a dict with a prefix, skipping non-scalar tensors."""
  for key, val in metrics.items():
    if any(key.endswith(sfx) for sfx in skip_suffixes):
      continue
    dest[f"{prefix}{key}"] = val


def _metric_candidates(name: str) -> List[str]:
  """Return possible metric-key variants for lookup."""
  candidates = [name]
  if name.startswith("val/"):
    candidates.append(name[len("val/"):])
  if name.startswith("val_"):
    candidates.append(name[len("val_"):])
  if not name.startswith("val/") and not name.startswith("val_"):
    candidates.extend([f"val/{name}", f"val_{name}"])
  # preserve order while removing duplicates
  out = []
  seen = set()
  for c in candidates:
    if c not in seen:
      out.append(c)
      seen.add(c)
  return out


def _lookup_metric(metrics: dict, name: str) -> float:
  """Look up a metric by key, tolerating val/ and val_ naming variants."""
  for key in _metric_candidates(name):
    if key in metrics:
      try:
        return float(metrics[key])
      except (TypeError, ValueError):
        return float("nan")
  return float("nan")


def _metric_prefers_lower(name: str) -> bool:
  """Heuristic for deciding whether smaller metric values are better."""
  low_tokens = ("loss", "rmse", "mae", "error", "ece", "brier")
  lname = name.lower()
  return any(tok in lname for tok in low_tokens)


def _metric_score(value: float, mode: str) -> float:
  """Convert metric value to a higher-is-better score for ranking."""
  if not math.isfinite(value):
    return float("-inf")
  return value if mode == "max" else -value

def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for training and evaluation."""
  p = argparse.ArgumentParser()
  p.add_argument("--dataset-path", required=True)
  p.add_argument("--runs-root", default="training")
  p.add_argument("--run-name", default=None)
  p.add_argument("--eval-only", action="store_true")
  p.add_argument("--checkpoint-path", default=None)
  p.add_argument(
    "--eval-config",
    choices=["default", "run"],
    default=None,
    help=(
      "Eval-only config source. Use 'default' for pepper_scene_graphs/configs/config.py defaults, "
      "or 'run' to auto-load config.json from the checkpoint's run folder."
    ),
  )
  p.add_argument("--wandb", action="store_true")
  p.add_argument("--wandb-project", default="pepper_scene_graphs")
  p.add_argument("--wandb-run-name", default=None)
  p.add_argument(
    "--param",
    action="append",
    default=[],
    help="Override shorthand config param (e.g. lr=0.001). Known aliases: " + ", ".join(HYPERPARAM_ALIASES),
  )
  return p.parse_args()


def split_dataset(data_list: List[Any], val_ratio: float, seed: int) -> Tuple[List[Any], List[Any]]:
  """Split dataset into training and validation sets.
  
  Args:
      data_list: List of PyG Data objects.
      val_ratio: Fraction of data to use for validation.
      seed: Random seed for shuffling.
      
  Returns:
      Tuple of (train_list, val_list).
  """
  import random
  idx = list(range(len(data_list)))
  random.Random(seed).shuffle(idx)
  val_n = max(1, int(len(idx) * val_ratio))
  val_idx = idx[:val_n]
  tr_idx = idx[val_n:]
  train = [data_list[i] for i in tr_idx]
  val = [data_list[i] for i in val_idx]
  if len(train) == 0:
    train, val = val, []
  return train, val


def main():
  """Main entry point for training and evaluation.
  
  Parses arguments, loads config, sets up model/data, and runs the training loop.
  """
  args = parse_args()
  if args.eval_only:
    if args.checkpoint_path is None:
      raise ValueError("--eval-only requires --checkpoint-path")
    if args.eval_config is None:
      raise ValueError("--eval-only requires --eval-config (default|run)")

    ckpt_path = Path(args.checkpoint_path)
    run_dir = _find_run_dir_from_checkpoint(ckpt_path) if args.eval_config == "run" else None
    cfg = Config()
    if args.eval_config == "run":
      with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
      _apply_cfg_dict(cfg, cfg_dict)

    _apply_config_overrides(cfg, args.param)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    data_list = load_graph_dataset(args.dataset_path, cfg)
    in_dim = data_list[0].x.size(-1)
    model = JointSceneGraphModel(in_dim=in_dim, cfg=cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    eval_loader = DataLoader(data_list, batch_size=cfg.train.batch_size, shuffle=False)
    do_sweep = bool(cfg.edge_exist.thresh_sweep_enabled)
    metrics = evaluate(model, eval_loader, device, cfg, threshold_sweep=do_sweep, epoch=None, collect_detailed=True)

    eval_dir = ckpt_path.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    node_conf = metrics.get("node_conf")
    edge_conf = metrics.get("edge_conf")
    if node_conf is not None and edge_conf is not None:
      np.savez(
        eval_dir / "confusion_eval.npz",
        node_conf=node_conf.numpy(),
        edge_conf=edge_conf.numpy(),
        node_classes=np.array(NODE_CLASSES),
        edge_labels=np.array(EDGE_LABELS),
      )
    metrics_eval = {
      "loss": metrics.get("loss", float("nan")),
      "node_loss": metrics.get("node_loss", float("nan")),
      "local_node_loss": metrics.get("local_node_loss", float("nan")),
      "edge_exist_loss": metrics.get("edge_exist_loss", float("nan")),
      "edge_label_loss": metrics.get("edge_label_loss", float("nan")),
      "occl_loss": metrics.get("occl_loss", float("nan")),
      "node_acc": metrics.get("node_acc", float("nan")),
      "local_node_acc": metrics.get("local_node_acc", float("nan")),
      "edge_exist_precision": metrics.get("edge_exist_precision", float("nan")),
      "edge_exist_recall": metrics.get("edge_exist_recall", float("nan")),
      "edge_exist_f1": metrics.get("edge_exist_f1", float("nan")),
      "edge_exist_05_prec": metrics.get("edge_exist_05_prec", float("nan")),
      "edge_exist_05_rec": metrics.get("edge_exist_05_rec", float("nan")),
      "edge_exist_05_f1": metrics.get("edge_exist_05_f1", float("nan")),
      "edge_exist_auc": metrics.get("edge_exist_auc", float("nan")),
      "edge_exist_ap": metrics.get("edge_exist_ap", float("nan")),
      "edge_exist_ece": metrics.get("edge_exist_ece", float("nan")),
      "edge_exist_brier": metrics.get("edge_exist_brier", float("nan")),
      "edge_exist_mean_prob_pos": metrics.get("edge_exist_mean_prob_pos", float("nan")),
      "edge_exist_mean_prob_neg": metrics.get("edge_exist_mean_prob_neg", float("nan")),
      "edge_exist_best_thresh": metrics.get("edge_exist_best_thresh", float("nan")),
      "edge_exist_best_f1": metrics.get("edge_exist_best_f1", float("nan")),
      "edge_exist_best_precision": metrics.get("edge_exist_best_precision", float("nan")),
      "edge_exist_best_recall": metrics.get("edge_exist_best_recall", float("nan")),
      "edge_rel_acc": metrics.get("edge_rel_acc", float("nan")),
      "knn_hit_rate": metrics.get("knn_hit_rate", float("nan")),
      "edge_exist_pos_rate": metrics.get("edge_exist_pos_rate", float("nan")),
      "edge_exist_pred_pos_rate": metrics.get("edge_exist_pred_pos_rate", float("nan")),
      "node_ece": metrics.get("node_ece", float("nan")),
      "node_brier": metrics.get("node_brier", float("nan")),
      "node_evidence_mean": metrics.get("node_evidence_mean", float("nan")),
      "edge_evidence_mean": metrics.get("edge_evidence_mean", float("nan")),
      "edge_exist_group_overall": metrics.get("edge_exist_group_overall", {}),
      "edge_exist_group_overall_at_best_thresh": metrics.get("edge_exist_group_overall_at_best_thresh", {}),
    }
    with open(eval_dir / "metrics_eval.json", "w", encoding="utf-8") as f:
      json.dump(_json_safe(metrics_eval), f, indent=2)

    with open(eval_dir / "per_type_eval.json", "w", encoding="utf-8") as f:
      json.dump(
        _json_safe(
          {
            "edge_exist_per_type": metrics.get("edge_exist_per_type", {}),
            "edge_exist_group_per_type": metrics.get("edge_exist_group_per_type", {}),
            "edge_exist_per_type_at_best_thresh": metrics.get("edge_exist_per_type_at_best_thresh", {}),
            "edge_exist_per_type_at_own_best_thresh": metrics.get("edge_exist_per_type_at_own_best_thresh", {}),
            "edge_exist_group_per_type_at_best_thresh": metrics.get("edge_exist_group_per_type_at_best_thresh", {}),
            "edge_exist_group_per_type_at_own_best_thresh": metrics.get("edge_exist_group_per_type_at_own_best_thresh", {}),
            "edge_exist_per_type_best": metrics.get("edge_exist_per_type_best", {}),
          }
        ),
        f,
        indent=2,
      )

    if "edge_exist_threshold_curve" in metrics:
      with open(eval_dir / "threshold_sweep_eval.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics["edge_exist_threshold_curve"]), f, indent=2)

    if args.wandb:
      wandb_run_name = args.wandb_run_name or f"{ckpt_path.parent.name}_eval"
      wandb_run_dir = eval_dir / "wandb"
      wandb_run_dir.mkdir(parents=True, exist_ok=True)
      os.environ["WANDB_DIR"] = str(wandb_run_dir)
      wandb.init(
        project=args.wandb_project,
        dir=str(wandb_run_dir),
        name=wandb_run_name,
        config=asdict(cfg),
      )
      eval_log = {
        "eval/loss": metrics["loss"],
        "eval/node_acc": metrics["node_acc"],
        "eval/edge_exist_precision": metrics["edge_exist_precision"],
        "eval/edge_exist_recall": metrics["edge_exist_recall"],
        "eval/edge_exist_f1": metrics["edge_exist_f1"],
        "eval/edge_rel_acc": metrics["edge_rel_acc"],
        "eval/knn_hit_rate": metrics["knn_hit_rate"],
        "eval/edge_exist_05_prec": metrics.get("edge_exist_05_prec", float("nan")),
        "eval/edge_exist_05_rec": metrics.get("edge_exist_05_rec", float("nan")),
        "eval/edge_exist_05_f1": metrics.get("edge_exist_05_f1", float("nan")),
      }
      if metrics["node_hist"]:
        eval_log["eval/node_hist"] = wandb.Histogram(metrics["node_hist"])
      if metrics["edge_hist"]:
        eval_log["eval/edge_hist"] = wandb.Histogram(metrics["edge_hist"])
      wandb.log(eval_log)
      wandb.finish()
    return

  cfg = Config()
  overrides = _apply_config_overrides(cfg, args.param)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_suffix = "".join(f"_{alias}{format_param_value(value)}" for alias, value in overrides)
  run_name_base = args.run_name or f"{timestamp}{run_suffix}"
  wandb_run_name = args.wandb_run_name or run_name_base
  run_dir_name = run_name_base
  if not run_dir_name.startswith(timestamp):
    run_dir_name = f"{timestamp}_{run_dir_name}"
  run_dir = Path(args.runs_root) / run_dir_name
  run_dir.mkdir(parents=True, exist_ok=True)

  set_seed(cfg.train.seed)

  # Track per-epoch metrics in a local JSONL history file for easy offline analysis.
  history_path = run_dir / "metrics_history.jsonl"
  history_f = history_path.open("w", encoding="utf-8")
  wandb_dir = None

  if args.wandb:
    wandb_run_dir = run_dir / "wandb"
    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_run_dir)
    wandb.init(
      project=args.wandb_project,
      dir=str(wandb_run_dir),
      name=wandb_run_name,
      config=asdict(cfg),
    )
    if wandb.run is not None and wandb.run.dir is not None:
      wandb_dir = Path(wandb.run.dir)

  data_list = load_graph_dataset(args.dataset_path, cfg)
  train_set, val_set = split_dataset(data_list, cfg.train.val_ratio, cfg.train.seed)

  train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False) if val_set else None

  device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
  in_dim = data_list[0].x.size(-1)

  model = JointSceneGraphModel(in_dim=in_dim, cfg=cfg).to(device)

  with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(cfg.__dict__, f, default=lambda o: o.__dict__, indent=2)

  optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
  ckpt_path = run_dir / "best_model.pt"
  last_ckpt_path = run_dir / "last_model.pt"
  best_pred_dump_path = run_dir / "preds_best.json"
  final_pred_dump_path = run_dir / "preds_final.json"
  last_val_metrics = None
  ckpt_cfg = cfg.checkpoint

  monitor_name = ckpt_cfg.monitor
  monitor_mode = ckpt_cfg.mode.lower()
  if monitor_mode not in ("min", "max"):
    raise ValueError(f"cfg.checkpoint.mode must be 'min' or 'max', got: {ckpt_cfg.mode}")
  start_epoch = max(1, int(ckpt_cfg.start_epoch))
  save_top_k = max(0, int(ckpt_cfg.save_top_k))
  save_last = bool(ckpt_cfg.save_last)
  tie_name = ckpt_cfg.tie_breaker
  tie_mode = "min" if _metric_prefers_lower(tie_name) else "max"

  topk_records = []
  warned_missing_monitor = False

  def _rank_tuple(rec: dict) -> Tuple[float, float, int]:
    return (
      _metric_score(rec["monitor_value"], monitor_mode),
      _metric_score(rec["tie_value"], tie_mode),
      int(rec["epoch"]),
    )

  def _save_checkpoint(path: Path, epoch: int, val_metrics: dict, monitor_value: float, tie_value: float) -> None:
    torch.save(
      {
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "epoch": epoch,
        "val_loss": float(val_metrics.get("loss", float("nan"))),
        "monitor_name": monitor_name,
        "monitor_mode": monitor_mode,
        "monitor_value": float(monitor_value),
        "tie_breaker": tie_name,
        "tie_value": float(tie_value),
      },
      path,
    )

  def _save_best_artifacts(val_metrics: dict) -> None:
    if val_metrics.get("pred_dump") is not None:
      with open(best_pred_dump_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics["pred_dump"], f, indent=2)
    node_conf = val_metrics.get("node_conf")
    edge_conf = val_metrics.get("edge_conf")
    if node_conf is not None and edge_conf is not None:
      np.savez(
        run_dir / "confusion_best.npz",
        node_conf=node_conf.numpy(),
        edge_conf=edge_conf.numpy(),
        node_classes=np.array(NODE_CLASSES),
        edge_labels=np.array(EDGE_LABELS),
      )

  for epoch in range(1, cfg.train.epochs + 1):
    tc = cfg.edge_candidates.type_constraint
    tc.tau = scheduled_tau(tc, epoch)

    tr = train_one_epoch(model, train_loader, optim, device, cfg, epoch)
    if val_loader is not None:
      do_sweep = bool(cfg.edge_exist.thresh_sweep_enabled)
      va = evaluate(model, val_loader, device, cfg, threshold_sweep=do_sweep, epoch=epoch)
    else:
      va = {"loss": float("nan")}
    last_val_metrics = va

    # Persist full metrics history (train + val) as JSONL.
    history_record = {
      "epoch": epoch,
      "train": _json_safe(tr),
      "val": _json_safe(va) if val_loader is not None else None,
    }
    history_f.write(json.dumps(history_record) + "\n")
    history_f.flush()

    print(
      f"E{epoch:03d} lr={optim.param_groups[0]['lr']:.2e} | "
      f"L tr/va {tr['loss']:.3f}/{va['loss']:.3f} | "
      f"NodeAcc {tr['node_acc']:.3f}/{va['node_acc']:.3f} | "
      f"Exist: P/R/F1 {tr['edge_exist_precision']:.2f}/{tr['edge_exist_recall']:.2f}/{tr['edge_exist_f1']:.2f} -> "
      f"{va['edge_exist_precision']:.2f}/{va['edge_exist_recall']:.2f}/{va['edge_exist_f1']:.2f} | "
      f"Exist 0.5: P/R/F1 {tr['edge_exist_05_prec']:.2f}/{tr['edge_exist_05_rec']:.2f}/{tr['edge_exist_05_f1']:.2f} -> "
      f"{va['edge_exist_05_prec']:.2f}/{va['edge_exist_05_rec']:.2f}/{va['edge_exist_05_f1']:.2f} | "
      f"AUC {tr.get('edge_exist_auc', 0):.3f}/{va.get('edge_exist_auc', 0):.3f} | "
      f"Prob+/- {tr.get('edge_exist_mean_prob_pos', 0):.2f}/{tr.get('edge_exist_mean_prob_neg', 0):.2f} -> "
      f"{va.get('edge_exist_mean_prob_pos', 0):.2f}/{va.get('edge_exist_mean_prob_neg', 0):.2f} | "
      f"Cand hit {tr.get('knn_hit_rate', 0):.2f}/{va.get('knn_hit_rate', 0):.2f} "
      f"posrate {tr.get('edge_exist_pos_rate', 0):.3f}/{va.get('edge_exist_pos_rate', 0):.3f} "
      f"pred+ {tr.get('edge_exist_pred_pos_rate', 0):.3f}/{va.get('edge_exist_pred_pos_rate', 0):.3f} | "
      f"bestF1 {va.get('edge_exist_best_f1', 0):.2f}@{va.get('edge_exist_best_thresh', 0):.2f} | "
      f"ev {va.get('node_evidence_mean', 0):.2f}/{va.get('edge_evidence_mean', 0):.2f}"
    )
    if args.wandb:
      log_payload = {
        "epoch": epoch,
        "train/loss": tr["loss"],
        "train/node_loss": tr["node_loss"],
        "train/edge_exist_loss": tr["edge_exist_loss"],
        "train/edge_label_loss": tr["edge_label_loss"],
        "train/node_acc": tr["node_acc"],
        "train/edge_exist_precision": tr["edge_exist_precision"],
        "train/edge_exist_recall": tr["edge_exist_recall"],
        "train/edge_exist_f1": tr["edge_exist_f1"],
        "train/edge_exist_05_prec": tr["edge_exist_05_prec"],
        "train/edge_exist_05_rec": tr["edge_exist_05_rec"],
        "train/edge_exist_05_f1": tr["edge_exist_05_f1"],

      }
      _add_prefixed_metrics(log_payload, "train/", tr)
      if val_loader is not None:
        val_log = {
          "val/loss": va["loss"],
          "val/node_loss": va["node_loss"],
          "val/edge_exist_loss": va["edge_exist_loss"],
          "val/edge_label_loss": va["edge_label_loss"],
          "val/node_acc": va["node_acc"],
          "val/edge_exist_precision": va["edge_exist_precision"],
          "val/edge_exist_recall": va["edge_exist_recall"],
          "val/edge_exist_f1": va["edge_exist_f1"],
          "val/edge_exist_05_prec": va["edge_exist_05_prec"],
          "val/edge_exist_05_rec": va["edge_exist_05_rec"],
          "val/edge_exist_05_f1": va["edge_exist_05_f1"],
          "val/edge_pos_count": va["edge_pos_count"],
          "val/edge_rel_acc": va["edge_rel_acc"],
          "val/edge_exist_pos_rate": va["edge_exist_pos_rate"],
          "val/knn_hit_rate": va["knn_hit_rate"],
          "val/node_ece": va.get("node_ece", float("nan")),
          "val/edge_exist_ece": va.get("edge_exist_ece", float("nan")),
          "val/node_brier": va.get("node_brier", float("nan")),
          "val/edge_exist_brier": va.get("edge_exist_brier", float("nan")),
          "val/node_evidence_mean": va.get("node_evidence_mean", float("nan")),
          "val/edge_evidence_mean": va.get("edge_evidence_mean", float("nan")),
          "val/edge_exist_best_thresh": va.get("edge_exist_best_thresh", float("nan")),
          "val/edge_exist_best_f1": va.get("edge_exist_best_f1", float("nan")),
          "val/edge_exist_best_precision": va.get("edge_exist_best_precision", float("nan")),
          "val/edge_exist_best_recall": va.get("edge_exist_best_recall", float("nan")),
        }
        _add_prefixed_metrics(val_log, "val/", va)
        if va["node_hist"]:
          val_log["val/node_hist"] = wandb.Histogram(va["node_hist"])
        if va["edge_hist"]:
          val_log["val/edge_hist"] = wandb.Histogram(va["edge_hist"])
        log_payload.update(val_log)
      wandb.log(log_payload)

    monitor_val = _lookup_metric(va, monitor_name) if val_loader is not None else float("nan")
    tie_val = _lookup_metric(va, tie_name) if (val_loader is not None and tie_name) else float("nan")

    if save_last:
      _save_checkpoint(last_ckpt_path, epoch, va, monitor_val, tie_val)

    if val_loader is not None and save_top_k > 0 and epoch >= start_epoch:
      if math.isfinite(monitor_val):
        rec_path = run_dir / f"checkpoint_epoch{epoch:03d}.pt"
        rec = {
          "epoch": epoch,
          "path": rec_path,
          "monitor_value": monitor_val,
          "tie_value": tie_val,
        }
        should_add = len(topk_records) < save_top_k
        if not should_add and topk_records:
          worst = topk_records[-1]
          should_add = _rank_tuple(rec) > _rank_tuple(worst)
        if should_add:
          _save_checkpoint(rec_path, epoch, va, monitor_val, tie_val)
          topk_records.append(rec)
          topk_records.sort(key=_rank_tuple, reverse=True)
          while len(topk_records) > save_top_k:
            drop = topk_records.pop(-1)
            try:
              drop["path"].unlink(missing_ok=True)
            except TypeError:
              if drop["path"].exists():
                drop["path"].unlink()
          best_rec = topk_records[0]
          if best_rec["epoch"] == epoch:
            shutil.copy2(best_rec["path"], ckpt_path)
            _save_best_artifacts(va)
      elif not warned_missing_monitor:
        print(
          f"Warning: checkpoint monitor '{monitor_name}' not found in validation metrics. "
          "No best-checkpoint updates will happen until this metric is available."
        )
        warned_missing_monitor = True

  if val_loader is not None and last_val_metrics and last_val_metrics.get("pred_dump") is not None:
    with open(final_pred_dump_path, "w", encoding="utf-8") as f:
      json.dump(last_val_metrics["pred_dump"], f, indent=2)
    node_conf = last_val_metrics.get("node_conf")
    edge_conf = last_val_metrics.get("edge_conf")
    if node_conf is not None and edge_conf is not None:
      np.savez(
        run_dir / "confusion_final.npz",
        node_conf=node_conf.numpy(),
        edge_conf=edge_conf.numpy(),
        node_classes=np.array(NODE_CLASSES),
        edge_labels=np.array(EDGE_LABELS),
      )

  if not ckpt_path.exists():
    if last_ckpt_path.exists():
      shutil.copy2(last_ckpt_path, ckpt_path)
      print("Warning: no best checkpoint selected by policy; fell back to last checkpoint.")
    else:
      print("Warning: no checkpoint was saved.")

  print(
    f"Done. Best ckpt: {ckpt_path} | "
    f"monitor={monitor_name} mode={monitor_mode} start_epoch={start_epoch} top_k={save_top_k}"
  )
  history_f.close()

  # Copy metrics history into the wandb run folder (if enabled) for easy access.
  if wandb_dir is not None:
    try:
      shutil.copy(history_path, wandb_dir / "metrics_history.jsonl")
    except Exception as exc:
      print(f"Warning: failed to copy metrics history into wandb folder: {exc}")

  if args.wandb:
    wandb.finish()


if __name__ == "__main__":
  main()
