#!/usr/bin/env python3
"""
Run Oracle (S2) inference on a single real-world scene exported in synthetic-compatible format.

This script:
  - Loads a trained oracle checkpoint (default: S2 ablation).
  - Loads a single scene from `datasets/real_data/` (or a provided --data_root).
  - Uses occlusion GT from sidecars (default: z-buffer) to compute metrics.
  - Writes a compact, Blender-friendly set of artifacts into:
      <data_root>/inference/<run_name>/

Outputs:
  - summary.json: overall metrics + counts
  - fruit_dir_table.csv: per-(fruit,dir) summary table
  - cases_good.json / cases_bad.json: ranked case summaries
  - cases/good/*.json and cases/bad/*.json: full per-case payloads for Blender viz

Recommended run (use your training env with torch+torch_geometric installed):
  python scripts/real_data/infer_oracle_real_scene.py \
    --data_root datasets/real_data \
    --scene_id scene_00000 \
    --ckpt_run_dir training/oracle/20260219_021539_oracle_ablate_S2_leaf_sa_per_fruit \
    --ckpt_kind best \
    --occl_gt_source zbuf
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repository root is in path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn  # noqa: E402
from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline, OracleResBaseline  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _as_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{str(k): v for k, v in (d or {}).items()})


def build_cfg_from_args(args: SimpleNamespace) -> SimpleNamespace:
    # Mirrors `scripts/train_oracle.py` cfg construction.
    occl_node_feat_source = getattr(args, "occl_node_feat_source", None)
    if occl_node_feat_source is None:
        occl_node_feat_source = "combined"
    m = SimpleNamespace(
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
        oracle_msg_npoint=int(getattr(args, "npoint", 512)),
        oracle_msg_radius_list=[0.01, 0.02, 0.04],
        oracle_msg_nsample_list=[16, 32, 64],
        node_feat_mode=getattr(args, "node_feat_mode", None),
        k_candidates=int(getattr(args, "k_candidates", 8)),
        dist_threshold=float(getattr(args, "dist_threshold", 0.3)),
        extra_stems_per_node=int(getattr(args, "extra_stems_per_node", 2)),
        use_normals=bool(getattr(args, "use_normals", True)),
        use_rgb=bool(getattr(args, "use_rgb", True)),
        no_geom_injection=bool(getattr(args, "no_geom_injection", False)),
        class_dropout_rates=[
            float(getattr(args, "pd_stem", 0.0)),
            float(getattr(args, "pd_leaf", 0.0)),
            float(getattr(args, "pd_fruit", 0.0)),
            float(getattr(args, "pd_peduncle", 0.0)),
        ],
        occl_enabled=bool(getattr(args, "occl_enabled", True)),
        occl_num_dirs=18,
        occl_dir_embed_dim=int(getattr(args, "occl_dir_embed_dim", 16)),
        occl_attn_dim=int(getattr(args, "occl_attn_dim", 64)),
        occl_attn_temp=float(getattr(args, "occl_attn_temp", 1.0)),
        occl_node_feat_source=str(occl_node_feat_source).strip(),
        occl_pair_geom_mode=str(getattr(args, "occl_pair_geom_mode", "full")).strip(),
        occl_leaf_sa_layers=int(getattr(args, "occl_leaf_sa_layers", 0)),
        occl_leaf_sa_mode=str(getattr(args, "occl_leaf_sa_mode", "per_fruit_dir")).strip(),
        occl_leaf_sa_dim=int(getattr(args, "occl_leaf_sa_dim", 128)),
        occl_leaf_sa_heads=int(getattr(args, "occl_leaf_sa_heads", 4)),
        occl_leaf_sa_ff_mult=int(getattr(args, "occl_leaf_sa_ff_mult", 4)),
        occl_leaf_sa_dropout=float(getattr(args, "occl_leaf_sa_dropout", 0.1)),
    )
    return SimpleNamespace(model=m)


def _load_checkpoint(
    ckpt_path: Path, model: torch.nn.Module, device: torch.device, strict: bool = False
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model_state" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'model_state': {ckpt_path}")
    incompatible = model.load_state_dict(ckpt["model_state"], strict=bool(strict))
    if not strict:
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            def _short(xs: List[str]) -> List[str]:
                return xs[:8] + (["..."] if len(xs) > 8 else [])
            print(
                f"[checkpoint] strict=False: missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(f"[checkpoint] missing_keys sample: {_short(missing)}", flush=True)
            if unexpected:
                print(f"[checkpoint] unexpected_keys sample: {_short(unexpected)}", flush=True)
    return ckpt


def sweep_best_f1_binary(probs: np.ndarray, labels: np.ndarray, num_thresholds: int = 200) -> Dict[str, float]:
    """
    Sweep thresholds in [0,1] to find the best F1 for a binary classifier.
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    thresholds = np.linspace(0.0, 1.0, num=int(num_thresholds), dtype=np.float32)

    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thresh = 0.5

    y_pos = (labels == 1)
    if int(y_pos.sum()) == 0:
        return {"best_f1": 0.0, "best_prec": 0.0, "best_rec": 0.0, "best_thresh": best_thresh}

    for t in thresholds.tolist():
        pred = (probs >= float(t))
        tp = int(np.logical_and(pred, y_pos).sum())
        fp = int(np.logical_and(pred, ~y_pos).sum())
        fn = int(np.logical_and(~pred, y_pos).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if (f1 > best_f1) or (f1 == best_f1 and float(t) > best_thresh):
            best_f1 = float(f1)
            best_prec = float(prec)
            best_rec = float(rec)
            best_thresh = float(t)

    return {"best_f1": best_f1, "best_prec": best_prec, "best_rec": best_rec, "best_thresh": best_thresh}


def compute_occl_dir_stats(occl_union_prob: torch.Tensor, occl_union: torch.Tensor, union_tau: float) -> Dict[str, float]:
    """
    Fruit-direction stats (dominant direction classification derived from union>=tau).
    occl_union_prob: [F, D] in [0,1]
    occl_union: [F, D] in [0,1]
    """
    if occl_union_prob is None or occl_union is None:
        return {}
    if occl_union_prob.numel() == 0 or occl_union.numel() == 0:
        return {}
    probs = occl_union_prob.detach().float()
    y_dom = (occl_union.detach().float() >= float(union_tau)).float()
    pred = (probs >= 0.5).float()

    # Flatten across fruits*dirs.
    p = pred.flatten().cpu().numpy().astype(np.int32)
    y = y_dom.flatten().cpu().numpy().astype(np.int32)
    # Manual PRF (avoid sklearn dependency in this script).
    tp = int(np.logical_and(p == 1, y == 1).sum())
    fp = int(np.logical_and(p == 1, y == 0).sum())
    fn = int(np.logical_and(p == 0, y == 1).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    brier_dom = float(torch.mean((probs - y_dom) ** 2).item())
    brier_union = float(torch.mean((probs - occl_union.detach().float()) ** 2).item())
    pos_rate = float(y_dom.mean().item())
    pred_pos_rate = float(pred.mean().item())
    return {
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "brier_dom": brier_dom,
        "brier_union": brier_union,
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
    }


def compute_union_mae_bins(
    occl_union_prob: torch.Tensor,
    occl_union: torch.Tensor,
    *,
    tau_low: float,
    tau_dom: float,
) -> Dict[str, float]:
    if occl_union_prob.numel() == 0 or occl_union.numel() == 0:
        return {}
    gt = occl_union.detach().float()
    err = torch.abs(occl_union_prob.detach().float() - gt)
    low = gt <= float(tau_low)
    mid = (gt > float(tau_low)) & (gt < float(tau_dom))
    high = gt >= float(tau_dom)
    out: Dict[str, float] = {}
    if bool(low.any()):
        out["mae_low"] = float(err[low].mean().item())
    if bool(mid.any()):
        out["mae_mid"] = float(err[mid].mean().item())
    if bool(high.any()):
        out["mae_high"] = float(err[high].mean().item())
    out["mae_all"] = float(err.mean().item())
    out["gt_mean"] = float(gt.mean().item())
    out["pred_mean"] = float(occl_union_prob.detach().float().mean().item())
    out["gt_max"] = float(gt.max().item())
    out["pred_max"] = float(occl_union_prob.detach().float().max().item())
    return out


def _ndcg_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    k = int(k)
    if k <= 0:
        return 0.0
    if scores.size == 0 or rel.size == 0:
        return 0.0
    order_pred = np.argsort(-scores)
    order_ideal = np.argsort(-rel)
    kk = min(k, int(order_pred.shape[0]))
    if kk <= 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
    top_pred = order_pred[:kk]
    top_ideal = order_ideal[:kk]
    dcg = float((rel[top_pred] * discount).sum())
    idcg = float((rel[top_ideal] * discount).sum())
    return float(dcg / (idcg + 1e-12))


def _mass_at_k(scores: np.ndarray, mass: np.ndarray, k: int) -> float:
    k = int(k)
    if k <= 0:
        return 0.0
    if scores.size == 0 or mass.size == 0:
        return 0.0
    total = float(mass.sum())
    if total <= 0.0:
        return 0.0
    order = np.argsort(-scores)
    kk = min(k, int(order.shape[0]))
    return float(mass[order[:kk]].sum()) / (total + 1e-12)


def _recall_at_k(scores: np.ndarray, y_pos: np.ndarray, k: int) -> float:
    k = int(k)
    if k <= 0:
        return 0.0
    n_pos = int(y_pos.sum())
    if n_pos <= 0:
        return 0.0
    order = np.argsort(-scores)
    kk = min(k, int(order.shape[0]))
    return float(y_pos[order[:kk]].sum()) / float(n_pos)


def fruit_local_frame(fruit_center: np.ndarray, stem_center: np.ndarray) -> np.ndarray:
    """Return 3x3 matrix R (rows are x,y,z in world coords) for the fruit-local frame."""
    x_vec = np.asarray([stem_center[0] - fruit_center[0], stem_center[1] - fruit_center[1], 0.0], dtype=np.float32)
    if float(np.linalg.norm(x_vec)) < 1e-6:
        x_vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    x_unit = x_vec / (np.linalg.norm(x_vec) + 1e-12)
    z_unit = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    y_vec = np.cross(z_unit, x_unit)
    if float(np.linalg.norm(y_vec)) < 1e-6:
        y_vec = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    y_unit = y_vec / (np.linalg.norm(y_vec) + 1e-12)
    x_unit = np.cross(y_unit, z_unit)
    x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-12)
    return np.vstack([x_unit, y_unit, z_unit]).astype(np.float32)


def build_occlusion_dirs() -> np.ndarray:
    base = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    dirs: List[np.ndarray] = []
    for v in base:
        dirs.append(v)
        dirs.append(-v)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v = base[i] * float(s1) + base[j] * float(s2)
                v = v / (np.linalg.norm(v) + 1e-12)
                dirs.append(v.astype(np.float32))
    return np.stack(dirs, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class NodeInfo:
    instance_id: int
    node_id: str
    typ: str
    semantic_id: int
    centroid: np.ndarray


def _index_graph_nodes(graph: Dict[str, Any]) -> Tuple[Dict[int, NodeInfo], Dict[str, NodeInfo]]:
    by_iid: Dict[int, NodeInfo] = {}
    by_node_id: Dict[str, NodeInfo] = {}
    for n in graph.get("nodes", []) or []:
        try:
            iid = int(n.get("instance_id"))
        except Exception:
            continue
        node_id = str(n.get("id"))
        typ = str(n.get("type"))
        sem = int(n.get("semantic_id"))
        c = np.asarray(n.get("centroid", [0, 0, 0]), dtype=np.float32).reshape(3)
        info = NodeInfo(instance_id=iid, node_id=node_id, typ=typ, semantic_id=sem, centroid=c)
        by_iid[iid] = info
        by_node_id[node_id] = info
    return by_iid, by_node_id


def _build_parent_map(graph: Dict[str, Any]) -> Dict[str, str]:
    parent: Dict[str, str] = {}
    for e in graph.get("edges", []) or []:
        if str(e.get("label")) != "attached_to":
            continue
        s = str(e.get("source"))
        t = str(e.get("target"))
        if s and t:
            parent[s] = t
    return parent


def _resolve_stem_for_fruit(
    fruit_node_id: str, parent: Dict[str, str], by_node_id: Dict[str, NodeInfo]
) -> Optional[NodeInfo]:
    cur = str(fruit_node_id)
    for _ in range(3):  # fruit -> ped -> stem (max)
        nxt = parent.get(cur)
        if not nxt:
            return None
        info = by_node_id.get(str(nxt))
        if info and info.typ == "stem":
            return info
        cur = str(nxt)
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run oracle inference on a single real scene and write viz-friendly outputs.")
    ap.add_argument("--data_root", type=str, default="datasets/real_data", help="Folder containing scene_XXXXX_pc_4mm.npz and scene_XXXXX_graph.json.")
    ap.add_argument("--scene_id", type=str, default="scene_00000", help="Scene id like scene_00000.")
    ap.add_argument(
        "--ckpt_run_dir",
        type=str,
        default="training/oracle/20260219_021539_oracle_ablate_S2_leaf_sa_per_fruit",
        help="Training run directory containing best_model.pt / last_model.pt.",
    )
    ap.add_argument("--ckpt_kind", type=str, default="best", choices=["best", "last"], help="Which checkpoint to load.")
    ap.add_argument("--occl_gt_source", type=str, default="zbuf", choices=["zbuf", "cam_ortho", "cam_persp"], help="Occlusion GT source.")
    ap.add_argument("--occl_cam_jitter_level", type=int, default=0, choices=[0, 1, 2], help="Cam GT only: 0=base,1=small,2=large jitter.")
    ap.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cuda:0, cpu). Default: cuda if available.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (for deterministic fruit selection).")
    ap.add_argument("--num_fruits", type=int, default=999, help="How many fruits to supervise/eval (>=#fruits => all).")
    ap.add_argument("--top_n", type=int, default=20, help="How many good/bad cases to write as full JSONs.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <data_root>/inference/<timestamp>_S2_<scene_id>_<gt>).")
    return ap.parse_args()


def main() -> int:
    args_cli = parse_args()

    random.seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))

    data_root = Path(args_cli.data_root).expanduser().resolve()
    scene_id = str(args_cli.scene_id).strip()
    ckpt_run_dir = Path(args_cli.ckpt_run_dir).expanduser().resolve()
    ckpt_path = ckpt_run_dir / ("best_model.pt" if args_cli.ckpt_kind == "best" else "last_model.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"[abort] checkpoint not found: {ckpt_path}")

    device_str = args_cli.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(device_str))

    out_root = Path(args_cli.out_dir) if args_cli.out_dir else None
    if out_root is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = data_root / "inference" / f"{ts}_S2_{scene_id}_{str(args_cli.occl_gt_source)}"
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cases" / "good").mkdir(parents=True, exist_ok=True)
    (out_root / "cases" / "bad").mkdir(parents=True, exist_ok=True)

    # Load checkpoint args (for exact S2 config).
    ckpt_cpu = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args_dict = ckpt_cpu.get("args", None)
    if not isinstance(ckpt_args_dict, dict):
        # Fallback to config.json.
        cfg_path = ckpt_run_dir / "config.json"
        if cfg_path.exists():
            ckpt_args_dict = _load_json(cfg_path)
        else:
            raise SystemExit("[abort] checkpoint has no args and config.json missing")
    ckpt_args = _as_namespace(ckpt_args_dict)

    cfg = build_cfg_from_args(ckpt_args)
    model_type = str(getattr(ckpt_args, "model_type", "res")).strip().lower()
    if model_type == "res":
        model: torch.nn.Module = OracleResBaseline(cfg, node_classes=4, edge_labels=3)
    else:
        model = OracleGTBaseline(cfg, node_classes=4, edge_labels=3)
    model.to(device)
    _load_checkpoint(ckpt_path, model=model, device=device, strict=False)
    model.eval()

    # Build dataset (val split) restricted to a single scene id.
    try:
        scene_num = int(scene_id.split("_")[1])
    except Exception:
        scene_num = 0

    # Override occl-aug "num fruits" so targets cover all fruits in this one scene.
    occl_union_tau = float(getattr(ckpt_args, "occl_union_tau", 0.5))
    occl_pot_eps = float(getattr(ckpt_args, "occl_potential_eps", 0.05))
    occl_tau_low = float(getattr(ckpt_args, "occl_potential_union_tau_low", 0.25))
    occl_max_dist = float(getattr(ckpt_args, "occl_aug_max_dist", 0.2))
    occl_max_leaves = int(getattr(ckpt_args, "occl_max_leaves_per_fruit", 64))
    occl_leafset_policy = str(getattr(ckpt_args, "occl_leafset_policy", "geom_stratified")).strip().lower()

    ds = OraclePepperDataset(
        data_root=str(data_root),
        split="val",
        augment=False,
        rotate_aug=False,
        rotate_mode="none",
        drop_shoots=bool(getattr(ckpt_args, "drop_shoots", False)),
        k_candidates=int(getattr(ckpt_args, "k_candidates", 8)),
        extra_stems_per_node=int(getattr(ckpt_args, "extra_stems_per_node", 2)),
        dist_threshold=float(getattr(ckpt_args, "dist_threshold", 0.3)),
        use_rgb=bool(getattr(ckpt_args, "use_rgb", True)),
        use_normals=bool(getattr(ckpt_args, "use_normals", True)),
        subgraph_enabled=False,
        hard_pair_constraint=bool(getattr(ckpt_args, "hard_pair_constraint", True)),
        occl_aug_enabled=False,
        occl_sidecar_dir=str(data_root),
        occl_aug_num_fruits=int(args_cli.num_fruits),
        occl_aug_union_tau=float(getattr(ckpt_args, "occl_aug_union_tau", 0.5)),
        occl_aug_resample_tries=int(getattr(ckpt_args, "occl_aug_resample_tries", 5)),
        occl_aug_max_dist=float(occl_max_dist),
        occl_aug_max_pos_leaves=int(getattr(ckpt_args, "occl_aug_max_pos_leaves", 16)),
        occl_aug_max_neg_leaves=int(getattr(ckpt_args, "occl_aug_max_neg_leaves", 8)),
        occl_aug_max_total_leaves=int(getattr(ckpt_args, "occl_aug_max_total_leaves", 96)),
        occl_aug_reserve_nodes=int(getattr(ckpt_args, "occl_aug_reserve_nodes", 0)),
        occl_targets_enabled=True,
        occl_gt_source=str(args_cli.occl_gt_source),
        occl_cam_jitter_level=int(args_cli.occl_cam_jitter_level),
        occl_targets_union_tau=float(occl_union_tau),
        occl_targets_max_leaves_per_fruit=int(occl_max_leaves),
        occl_leafset_policy=str(occl_leafset_policy),
        occl_supervise_max_fruits_per_scene=0,
        scene_id_allowlist={int(scene_num)},
        use_4mm=True,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=oracle_collate_fn)
    batch = next(iter(loader))
    batch = batch.to(device)

    # Load graph + instance map for Blender-friendly naming.
    graph_path = data_root / f"{scene_id}_graph.json"
    inst_map_path = data_root / f"{scene_id}_instance_map.json"
    if not graph_path.exists():
        raise SystemExit(f"[abort] graph not found: {graph_path}")
    if not inst_map_path.exists():
        raise SystemExit(f"[abort] instance map not found: {inst_map_path}")
    graph = _load_json(graph_path)
    inst_map_raw = _load_json(inst_map_path)
    inst_name: Dict[int, str] = {}
    inst_type: Dict[int, str] = {}
    for k, v in (inst_map_raw or {}).items():
        try:
            iid = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            inst_name[iid] = str(v.get("blender_object_name", f"iid_{iid}"))
            inst_type[iid] = str(v.get("type", ""))

    by_iid, by_node_id = _index_graph_nodes(graph)
    parent = _build_parent_map(graph)

    # Direction vectors (fruit-local) from sidecar if present.
    sidecar_path = None
    try:
        # Use dataset's resolver indirectly by relying on our known naming.
        suffix = "occl_zbuf_k3_ps4mm_d4mm.npz" if args_cli.occl_gt_source == "zbuf" else "occl_cam_k3_ps4mm_d4mm_both.npz"
        cand = data_root / f"{scene_id}_{suffix}"
        if cand.exists():
            sidecar_path = cand
        else:
            # Fallback: any matching file in data_root.
            pat = f"{scene_id}_occl_{'zbuf' if args_cli.occl_gt_source=='zbuf' else 'cam'}_*.npz"
            matches = sorted(data_root.glob(pat))
            sidecar_path = matches[0] if matches else None
    except Exception:
        sidecar_path = None
    dirs_local = None
    if sidecar_path is not None:
        try:
            with np.load(sidecar_path) as z:
                if "dirs" in z:
                    dirs_local = z["dirs"].astype(np.float32, copy=False)
                elif "dirs_base" in z:
                    dirs_local = z["dirs_base"].astype(np.float32, copy=False)
        except Exception:
            dirs_local = None
    if dirs_local is None or int(dirs_local.shape[0]) < 18:
        dirs_local = build_occlusion_dirs()
    dirs_local = np.asarray(dirs_local, dtype=np.float32)[:18]

    with torch.no_grad():
        out = model(batch)

    # Pull occlusion tensors.
    y_union = getattr(batch, "occl_fruit_union", None)
    y_pot = getattr(batch, "occl_pair_potential", None)
    y_excl = getattr(batch, "occl_pair_exclusive", None)
    y_k = getattr(batch, "occl_pair_k_rank", None)
    fruit_nodes = getattr(batch, "occl_fruit_idx", None)
    pair_index = getattr(batch, "occl_pair_index", None)

    if y_union is None or fruit_nodes is None or pair_index is None:
        raise SystemExit("[abort] batch missing occlusion tensors; did you generate the zbuf sidecar?")

    union_logits = out.get("occl_fruit_union_logits", None)
    pair_logits = out.get("occl_pair_potential_logits", None)
    pair_rank_logits = out.get("occl_pair_rank_logits", None)
    if union_logits is None or pair_logits is None:
        raise SystemExit("[abort] model did not emit occlusion logits (occl_enabled?)")

    union_prob = torch.sigmoid(union_logits.detach())
    pair_prob = torch.sigmoid(pair_logits.detach())

    # Aggregate metrics.
    metrics: Dict[str, Any] = {}
    metrics["occl_dir"] = compute_occl_dir_stats(union_prob, y_union, union_tau=occl_union_tau)
    metrics["occl_dir_best_f1"] = sweep_best_f1_binary(
        union_prob.flatten().cpu().numpy(), (y_union.detach().cpu().numpy().flatten() >= occl_union_tau).astype(np.int32)
    )
    metrics["occl_union_mae_bins"] = compute_union_mae_bins(union_prob, y_union, tau_low=occl_tau_low, tau_dom=occl_union_tau)

    # Explainability (present-leaf union fraction).
    explain = {"w_mean": 0.0, "w_pos_mean": 0.0}
    try:
        if y_union.numel() > 0 and y_excl is not None and y_excl.numel() > 0 and pair_index.numel() > 0:
            device_ = y_union.device
            y_union_dev = y_union.to(device_)
            D = int(y_union_dev.size(1))
            row_of_node = torch.full((batch.y_node.size(0),), -1, device=device_, dtype=torch.long)
            fruit_nodes_dev = fruit_nodes.to(device_)
            row_of_node[fruit_nodes_dev] = torch.arange(fruit_nodes_dev.numel(), device=device_, dtype=torch.long)
            pair_fruit_row = row_of_node[pair_index[0].to(device_)]
            valid = pair_fruit_row >= 0
            union_expl = torch.zeros((fruit_nodes_dev.numel(), D), device=device_, dtype=torch.float32)
            if bool(valid.any()):
                union_expl.index_add_(0, pair_fruit_row[valid], y_excl.to(device_)[valid].float())
            union_expl = union_expl.clamp(min=0.0, max=1.0)
            denom = torch.clamp(y_union_dev.float(), min=1e-6)
            frac = torch.clamp(union_expl / denom, min=0.0, max=1.0)
            w = torch.where(y_union_dev.float() > 1e-6, frac, torch.ones_like(frac))
            explain["w_mean"] = float(w.mean().item())
            pos = (y_union_dev.float() > 1e-6)
            explain["w_pos_mean"] = float(w[pos].mean().item()) if bool(pos.any()) else float(w.mean().item())
    except Exception:
        pass
    metrics["occl_explainability"] = explain

    # Leaf ranking metrics (global aggregates, matching training/val behavior).
    leaf_metrics: Dict[str, float] = {}
    if y_pot is not None and y_pot.numel() > 0 and pair_index.numel() > 0:
        pot_np = y_pot.detach().cpu().numpy()
        prob_np = pair_prob.detach().cpu().numpy()
        pair_fruit_node = pair_index[0].detach().cpu().numpy().astype(np.int64)
        fruit_nodes_np = fruit_nodes.detach().cpu().numpy().astype(np.int64)
        union_np = y_union.detach().cpu().numpy()
        fruit_to_row = {int(n): int(r) for r, n in enumerate(fruit_nodes_np.tolist())}

        # Recall@K on positives defined by potential>=eps, over all directions.
        for k in (1, 3, 5):
            vals = []
            for fnode in fruit_nodes_np.tolist():
                row = fruit_to_row.get(int(fnode))
                if row is None:
                    continue
                m = (pair_fruit_node == int(fnode))
                if not np.any(m):
                    continue
                probs_f = prob_np[m]  # [L,D]
                pot_f = pot_np[m]     # [L,D]
                for d in range(pot_f.shape[1]):
                    y_pos = (pot_f[:, d] >= occl_pot_eps)
                    if int(y_pos.sum()) == 0:
                        continue
                    vals.append(_recall_at_k(probs_f[:, d], y_pos, k=k))
            leaf_metrics[f"recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0

        # Mass@K and NDCG@K using k-buffer mass (sum over ranks), with dust filtered by pot>=eps.
        if y_k is not None and y_k.numel() > 0:
            mass_np = y_k.detach().cpu().numpy().sum(axis=-1)  # [M,D]
            mass_np = np.where(pot_np >= occl_pot_eps, mass_np, 0.0)
            for k in (1, 3, 5):
                vals_mass = []
                vals_ndcg = []
                for fnode in fruit_nodes_np.tolist():
                    m = (pair_fruit_node == int(fnode))
                    if not np.any(m):
                        continue
                    probs_f = prob_np[m]  # [L,D]
                    mass_f = mass_np[m]   # [L,D]
                    for d in range(mass_f.shape[1]):
                        if float(mass_f[:, d].sum()) <= 0.0:
                            continue
                        vals_mass.append(_mass_at_k(probs_f[:, d], mass_f[:, d], k=k))
                        vals_ndcg.append(_ndcg_at_k(probs_f[:, d], mass_f[:, d], k=k))
                leaf_metrics[f"mass_at_{k}"] = float(np.mean(vals_mass)) if vals_mass else 0.0
                leaf_metrics[f"ndcg_at_{k}"] = float(np.mean(np.clip(vals_ndcg, 0.0, 1.0))) if vals_ndcg else 0.0
    metrics["occl_leaf"] = leaf_metrics

    # Build per-(fruit,dir) table and rank cases.
    fruit_iids = batch.node_iid[fruit_nodes].detach().cpu().numpy().astype(np.int32)  # [F]
    fruit_nodes_np = fruit_nodes.detach().cpu().numpy().astype(np.int64)
    pair_fruit_node = pair_index[0].detach().cpu().numpy().astype(np.int64)
    pair_leaf_node = pair_index[1].detach().cpu().numpy().astype(np.int64)
    leaf_iids = batch.node_iid[pair_index[1]].detach().cpu().numpy().astype(np.int32)  # [M]

    y_union_np = y_union.detach().cpu().numpy()
    union_prob_np = union_prob.detach().cpu().numpy()
    y_pot_np = y_pot.detach().cpu().numpy() if (y_pot is not None and y_pot.numel() > 0) else None
    y_mass_np = (y_k.detach().cpu().numpy().sum(axis=-1) if (y_k is not None and y_k.numel() > 0) else None)
    if y_mass_np is not None and y_pot_np is not None:
        y_mass_np = np.where(y_pot_np >= occl_pot_eps, y_mass_np, 0.0)
    pair_prob_np = pair_prob.detach().cpu().numpy()
    pair_rank_np = pair_rank_logits.detach().cpu().numpy() if (pair_rank_logits is not None and pair_rank_logits.numel() > 0) else None

    fruit_to_row = {int(n): int(r) for r, n in enumerate(fruit_nodes_np.tolist())}

    # Pre-group pair rows by fruit row for faster per-dir stats.
    rows_by_fruit_row: Dict[int, np.ndarray] = {}
    for r, fnode in enumerate(fruit_nodes_np.tolist()):
        m = np.nonzero(pair_fruit_node == int(fnode))[0]
        rows_by_fruit_row[int(r)] = m.astype(np.int64, copy=False)

    fruit_dir_rows: List[Dict[str, Any]] = []
    case_summaries_good: List[Dict[str, Any]] = []
    case_summaries_bad: List[Dict[str, Any]] = []

    def _interest_score(u: float, n_pos: int, mass_total: float) -> float:
        # Encourage large occlusion and multiple contributing leaves.
        return float(u) * (1.0 + min(int(n_pos), 6) / 6.0) + 0.15 * float(math.log1p(max(0.0, mass_total)))

    for r in range(int(fruit_iids.shape[0])):
        fiid = int(fruit_iids[r])
        fnode = int(fruit_nodes_np[r])
        f_name = inst_name.get(fiid, f"fruit_iid_{fiid}")
        f_node_info = by_iid.get(fiid)
        fruit_center = (f_node_info.centroid if f_node_info is not None else None)

        # Resolve stem for local->world direction mapping.
        stem_info = None
        if f_node_info is not None:
            stem_info = _resolve_stem_for_fruit(f_node_info.node_id, parent, by_node_id)
        stem_center = stem_info.centroid if stem_info is not None else None
        R_rows = fruit_local_frame(fruit_center, stem_center) if (fruit_center is not None and stem_center is not None) else None

        pair_rows = rows_by_fruit_row.get(int(r), np.asarray([], dtype=np.int64))
        leaf_iids_r = leaf_iids[pair_rows] if pair_rows.size else np.asarray([], dtype=np.int32)

        for d in range(18):
            u_gt = float(y_union_np[r, d])
            u_pred = float(union_prob_np[r, d])
            u_err = float(abs(u_pred - u_gt))

            n_pos = 0
            mass_total = 0.0
            ndcg3 = 0.0
            mass3 = 0.0
            rec3 = 0.0
            if pair_rows.size and y_pot_np is not None:
                pot_d = y_pot_np[pair_rows, d]
                n_pos = int((pot_d >= occl_pot_eps).sum())
                if y_mass_np is not None:
                    mass_d = y_mass_np[pair_rows, d]
                    mass_total = float(mass_d.sum())
                    ndcg3 = _ndcg_at_k(pair_prob_np[pair_rows, d], mass_d, k=3)
                    mass3 = _mass_at_k(pair_prob_np[pair_rows, d], mass_d, k=3)
                rec3 = _recall_at_k(pair_prob_np[pair_rows, d], (pot_d >= occl_pot_eps), k=3)

            d_local = dirs_local[d].astype(np.float32, copy=False)
            d_world = None
            if R_rows is not None:
                d_world = (R_rows.T @ d_local.reshape(3, 1)).reshape(3).astype(np.float32, copy=False)

            row = {
                "scene_id": scene_id,
                "fruit_iid": fiid,
                "fruit_node": fnode,
                "fruit_name": f_name,
                "dir_idx": int(d),
                "union_gt": u_gt,
                "union_pred": u_pred,
                "union_abs_err": u_err,
                "n_candidate_leaves": int(leaf_iids_r.size),
                "n_pos_leaves": int(n_pos),
                "mass_total": float(mass_total),
                "recall_at_3": float(rec3),
                "mass_at_3": float(mass3),
                "ndcg_at_3": float(ndcg3),
                "interest": _interest_score(u_gt, n_pos, mass_total),
            }
            fruit_dir_rows.append(row)

            # Case buckets: "good interesting" vs "bad interesting"
            if u_gt >= 0.15 and int(n_pos) >= 2:
                if u_err <= 0.10 and ndcg3 >= 0.75:
                    case_summaries_good.append(row)
                if u_err >= 0.20 or (ndcg3 > 0.0 and ndcg3 <= 0.35):
                    case_summaries_bad.append(row)

    fruit_dir_rows.sort(key=lambda r: float(r.get("interest", 0.0)), reverse=True)
    case_summaries_good.sort(key=lambda r: float(r.get("interest", 0.0)), reverse=True)
    case_summaries_bad.sort(key=lambda r: float(r.get("interest", 0.0)), reverse=True)

    # Write per-(fruit,dir) table CSV.
    csv_path = out_root / "fruit_dir_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "fruit_iid",
                "fruit_name",
                "dir_idx",
                "union_gt",
                "union_pred",
                "union_abs_err",
                "n_candidate_leaves",
                "n_pos_leaves",
                "mass_total",
                "recall_at_3",
                "mass_at_3",
                "ndcg_at_3",
                "interest",
            ],
        )
        w.writeheader()
        for r in fruit_dir_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    _dump_json(out_root / "cases_good.json", case_summaries_good)
    _dump_json(out_root / "cases_bad.json", case_summaries_bad)
    _dump_json(out_root / "cases_ranked.json", fruit_dir_rows[:500])

    # Write full case JSONs for Blender viz.
    def _write_cases(kind: str, rows: List[Dict[str, Any]], top_n: int) -> None:
        out_dir = out_root / "cases" / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows[: int(top_n)]):
            fiid = int(r["fruit_iid"])
            d = int(r["dir_idx"])
            # Locate fruit row and pair rows.
            frow = None
            for rr, iid in enumerate(fruit_iids.tolist()):
                if int(iid) == fiid:
                    frow = int(rr)
                    break
            if frow is None:
                continue
            pair_rows = rows_by_fruit_row.get(int(frow), np.asarray([], dtype=np.int64))
            rank_src = "pred_potential"
            leaves: List[Dict[str, Any]] = []
            if pair_rows.size:
                # Per-leaf fields for this (fruit,dir).
                # We write two rank views:
                #   - pred_pot_rank: rank by predicted potential probability (sigmoid occl_pair_potential_logits)
                #   - pred_rank:     rank by the dedicated listwise rank head logits (occl_pair_rank_logits)
                # The two heads can disagree; keeping both makes debugging and viz easier.
                scores_pot = pair_prob_np[pair_rows, d].astype(np.float32, copy=False)  # [L]
                scores_rank = None
                if pair_rank_np is not None:
                    try:
                        scores_rank = pair_rank_np[pair_rows, d].astype(np.float32, copy=False)  # [L]
                        if not np.isfinite(scores_rank).any():
                            scores_rank = None
                    except Exception:
                        scores_rank = None

                order_pot = np.argsort(-scores_pot, kind="stable")
                pot_rank_of_local = np.empty((int(order_pot.size),), dtype=np.int32)
                pot_rank_of_local[order_pot] = np.arange(1, int(order_pot.size) + 1, dtype=np.int32)

                if scores_rank is not None:
                    # Break ties stably, secondary by potential to keep deterministic ordering.
                    # (np.lexsort uses last key as primary.)
                    order_rank = np.lexsort(((-scores_pot).astype(np.float32), (-scores_rank).astype(np.float32)))
                    order_rank = order_rank.astype(np.int64, copy=False)
                    order_rank = order_rank  # ascending on -score => descending score
                    rank_src = "pred_rank_logit"
                else:
                    order_rank = order_pot
                    rank_src = "pred_potential"

                rank_of_local = np.empty((int(order_rank.size),), dtype=np.int32)
                rank_of_local[order_rank] = np.arange(1, int(order_rank.size) + 1, dtype=np.int32)

                # Emit entries, sorted by the primary rank (rank head when present).
                for local_j in order_rank.tolist():
                    prow = int(pair_rows[int(local_j)])
                    liid = int(leaf_iids[prow])
                    entry = {
                        "leaf_iid": liid,
                        "leaf_name": inst_name.get(liid, f"leaf_iid_{liid}"),
                        "leaf_type": inst_type.get(liid, ""),
                        "pred_potential": float(pair_prob_np[prow, d]),
                        "pred_rank": int(rank_of_local[int(local_j)]),
                        "pred_pot_rank": int(pot_rank_of_local[int(local_j)]),
                        "gt_potential": float(y_pot_np[prow, d]) if y_pot_np is not None else None,
                        "gt_mass": float(y_mass_np[prow, d]) if y_mass_np is not None else None,
                    }
                    if scores_rank is not None:
                        entry["pred_rank_logit"] = float(pair_rank_np[prow, d])
                    # Centroid for quick inspection.
                    ni = by_iid.get(liid)
                    if ni is not None:
                        entry["leaf_centroid_world"] = ni.centroid.astype(np.float32).tolist()
                    leaves.append(entry)

            # Fruit/stem geometry for arrows.
            f_info = by_iid.get(fiid)
            stem_info = None
            if f_info is not None:
                stem_info = _resolve_stem_for_fruit(f_info.node_id, parent, by_node_id)
            fruit_center = f_info.centroid.astype(np.float32).tolist() if f_info is not None else None
            stem_center = stem_info.centroid.astype(np.float32).tolist() if stem_info is not None else None
            R_rows = None
            d_world = None
            d_local = dirs_local[d].astype(np.float32, copy=False).tolist()
            if fruit_center is not None and stem_center is not None:
                R_rows_np = fruit_local_frame(np.asarray(fruit_center, dtype=np.float32), np.asarray(stem_center, dtype=np.float32))
                R_rows = R_rows_np.astype(np.float32).tolist()
                d_world_np = (R_rows_np.T @ np.asarray(d_local, dtype=np.float32).reshape(3, 1)).reshape(3)
                d_world = d_world_np.astype(np.float32).tolist()

            payload = {
                "scene_id": scene_id,
                "gt_source": str(args_cli.occl_gt_source),
                "ckpt": {"run_dir": str(ckpt_run_dir), "ckpt_kind": str(args_cli.ckpt_kind), "ckpt_path": str(ckpt_path)},
                "fruit": {
                    "instance_id": fiid,
                    "blender_object_name": inst_name.get(fiid, f"fruit_iid_{fiid}"),
                    "centroid_world": fruit_center,
                },
                "stem": {
                    "instance_id": int(stem_info.instance_id) if stem_info is not None else None,
                    "blender_object_name": inst_name.get(int(stem_info.instance_id), "") if stem_info is not None else None,
                    "centroid_world": stem_center,
                },
                "direction": {
                    "dir_idx": int(d),
                    "d_local": d_local,
                    "d_world": d_world,
                    "fruit_local_R_rows_world": R_rows,
                },
                "union": {"gt": float(r["union_gt"]), "pred": float(r["union_pred"]), "abs_err": float(r["union_abs_err"])},
                "leaves": leaves,
                "notes": {
                    "pot_eps": float(occl_pot_eps),
                    "union_tau": float(occl_union_tau),
                    "leaf_rank_source": str(rank_src),
                    "leaf_rank_fields": {
                        "pred_rank": "primary rank used for visualization (rank head if available, else potential)",
                        "pred_pot_rank": "rank by pred_potential (sigmoid potential head)",
                        "pred_rank_logit": "raw rank head logit (higher => stronger occluder prediction)",
                    },
                },
            }
            _dump_json(out_dir / f"case_{i:03d}_fruit{fiid}_dir{d:02d}.json", payload)

    _write_cases("good", case_summaries_good, top_n=int(args_cli.top_n))
    _write_cases("bad", case_summaries_bad, top_n=int(args_cli.top_n))

    summary = {
        "scene_id": scene_id,
        "data_root": str(data_root),
        "gt_source": str(args_cli.occl_gt_source),
        "ckpt": {
            "run_dir": str(ckpt_run_dir),
            "ckpt_kind": str(args_cli.ckpt_kind),
            "ckpt_path": str(ckpt_path),
            "epoch": int(ckpt_cpu.get("epoch", -1)) if isinstance(ckpt_cpu, dict) else -1,
        },
        "counts": {
            "num_nodes": int(batch.y_node.size(0)),
            "num_fruits_supervised": int(fruit_nodes.numel()),
            "num_pairs": int(pair_index.size(1)),
        },
        "occl_params": {
            "union_tau": float(occl_union_tau),
            "pot_eps": float(occl_pot_eps),
            "tau_low": float(occl_tau_low),
            "max_dist": float(occl_max_dist),
            "max_leaves_per_fruit": int(occl_max_leaves),
            "leafset_policy": str(occl_leafset_policy),
        },
        "metrics": metrics,
        "outputs": {
            "fruit_dir_table_csv": str(csv_path),
            "cases_good_json": str(out_root / "cases_good.json"),
            "cases_bad_json": str(out_root / "cases_bad.json"),
            "cases_dir": str(out_root / "cases"),
        },
    }
    _dump_json(out_root / "summary.json", summary)

    print(f"[ok] wrote inference outputs -> {out_root}", flush=True)
    print(f"[ok] summary: {out_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
