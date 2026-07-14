
import os
import json
import shlex
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import datetime
import numpy as np
import sys
from types import SimpleNamespace
from typing import Dict, Tuple, Optional, List, Set
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import warnings

# Reduce noisy PyTorch warnings that can spam per-iteration logs.
# (We still fix the root cause where possible, but this keeps progress bars readable.)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)

# Ensure repository root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn
from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline, OracleResBaseline
from pepper_scene_graphs.utils.evidence import get_evidence_fn

def compute_metrics(logits, targets, mask=None, labels=None):
    if logits.numel() == 0:
        return {}
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
        if targets.numel() == 0:
            return {}
            
    preds = logits.argmax(dim=-1).cpu().numpy()
    target_np = targets.cpu().numpy()
    
    acc = (preds == target_np).mean()
    macro_f1 = f1_score(target_np, preds, average='macro', zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        target_np, preds, labels=labels, zero_division=0
    )
    
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        target_np, preds, average='macro', zero_division=0
    )
    
    metrics = {
        "acc": acc,
        "f1_macro": macro_f1,
        "prec_macro": p_macro,
        "rec_macro": r_macro,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }
    return metrics

def binary_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    """
    Expected Calibration Error for binary classification.
    probs: [N] in [0,1]
    labels: [N] in {0,1}
    """
    if probs.numel() == 0:
        return 0.0
    probs = probs.detach().float().flatten()
    labels = labels.detach().float().flatten()

    # Bin edges [0,1]. We include right edge in last bin.
    edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    n = float(probs.numel())
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == num_bins - 1:
            m = (probs >= lo) & (probs <= hi)
        else:
            m = (probs >= lo) & (probs < hi)
        if not bool(m.any()):
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece = ece + (m.float().mean() * torch.abs(acc - conf))
    return float(ece.item())

def multiclass_ece(probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
    """
    Expected Calibration Error for multiclass classification.

    Uses max-probability confidence and top-1 correctness (standard ECE).
    probs: [N, C], row-stochastic
    targets: [N] int64
    """
    if probs.numel() == 0:
        return 0.0
    probs = probs.detach().float()
    targets = targets.detach().long().to(probs.device)
    conf, pred = probs.max(dim=-1)  # [N]
    acc = (pred == targets).float()
    edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == num_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not bool(m.any()):
            continue
        ece = ece + (m.float().mean() * torch.abs(acc[m].mean() - conf[m].mean()))
    return float(ece.item())

def multiclass_brier(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Multiclass Brier score: mean ||p - one_hot(y)||^2."""
    if probs.numel() == 0:
        return 0.0
    probs = probs.detach().float()
    targets = targets.detach().long().to(probs.device)
    one_hot = F.one_hot(targets, num_classes=int(probs.size(-1))).to(dtype=probs.dtype, device=probs.device)
    return float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item())

def _get_edl_evidence_fn(args) -> Optional[callable]:
    """
    Return the evidence activation fn for EDL heads (cached on args).

    We keep this in the training script (instead of the model) so we can toggle
    evidential losses per-head without changing model outputs.
    """
    fn = getattr(args, "_edl_evidence_fn_impl", None)
    if fn is not None:
        return fn
    name = str(getattr(args, "edl_fn", "softplus")).lower()
    fn = get_evidence_fn(name)
    # EDL requires a non-negative evidence map; fall back to softplus if "none" was given.
    if fn is None:
        fn = get_evidence_fn("softplus")
    setattr(args, "_edl_evidence_fn_impl", fn)
    return fn

def _get_edl_kl_coeff(args, epoch: int) -> float:
    """Linear KL warmup (start -> kl_coeff over kl_warmup_epochs)."""
    target = float(getattr(args, "edl_kl_coeff", 0.1))
    warmup_epochs = int(getattr(args, "edl_kl_warmup_epochs", 20))
    start = float(getattr(args, "edl_kl_warmup_start", 0.0))
    if warmup_epochs <= 0:
        return target
    if warmup_epochs <= 1:
        return target
    progress = (int(epoch) - 1) / float(warmup_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    return start + (target - start) * progress


def _get_edl_kl_coeff_edge_exist(args, epoch: int) -> float:
    """
    Edge-existence-specific EDL KL coefficient (optional override).

    Edge existence can be substantially more sensitive to KL regularization than
    other EDL heads. This lets you tune it without perturbing occlusion/node EDL.
    """
    override = getattr(args, "edl_kl_coeff_edge_exist", None)
    if override is None:
        return _get_edl_kl_coeff(args, epoch)
    target = float(override)
    warmup_epochs = int(getattr(args, "edl_kl_warmup_epochs", 20))
    start = float(getattr(args, "edl_kl_warmup_start", 0.0))
    if warmup_epochs <= 0:
        return target
    if warmup_epochs <= 1:
        return target
    progress = (int(epoch) - 1) / float(warmup_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    return start + (target - start) * progress

def _edl_dirichlet_ce_per_sample(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample EDL cross-entropy data term (log form used in this repo)."""
    if alpha.numel() == 0:
        return torch.empty((0,), device=alpha.device, dtype=alpha.dtype)
    S = alpha.sum(dim=-1, keepdim=True)
    one_hot = F.one_hot(target.long(), num_classes=int(alpha.size(-1))).to(dtype=alpha.dtype, device=alpha.device)
    return (one_hot * (torch.log(S) - torch.log(alpha))).sum(dim=-1)

def _edl_dirichlet_kld_per_sample(alpha: torch.Tensor) -> torch.Tensor:
    """Per-sample KL( Dir(alpha) || Dir(1) )."""
    if alpha.numel() == 0:
        return torch.empty((0,), device=alpha.device, dtype=alpha.dtype)
    K = int(alpha.size(-1))
    beta = torch.ones((K,), device=alpha.device, dtype=alpha.dtype)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum()
    lnB_alpha = torch.lgamma(S_alpha).squeeze(-1) - torch.lgamma(alpha).sum(dim=-1)
    lnB_beta = torch.lgamma(beta).sum() - torch.lgamma(S_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    kld = lnB_alpha + lnB_beta + ((alpha - beta) * digamma_diff).sum(dim=-1)
    return kld

def _edl_kl_alpha_tilde_per_sample(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample KL regularizer with alpha-tilde trick (matches OBB pipeline)."""
    if alpha.numel() == 0:
        return torch.empty((0,), device=alpha.device, dtype=alpha.dtype)
    K = int(alpha.size(-1))
    one_hot = F.one_hot(target.long(), num_classes=K).to(dtype=alpha.dtype, device=alpha.device)
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    return _edl_dirichlet_kld_per_sample(alpha_tilde)

def _edl_tied_beta_alpha(logits: torch.Tensor, evidence_fn: callable) -> torch.Tensor:
    """
    Tied-Beta / 2-class Dirichlet for binary tasks from a single logit z.

    alpha0 = e(-z) + 1
    alpha1 = e( z) + 1
    """
    a0 = evidence_fn(-logits) + 1.0
    a1 = evidence_fn(logits) + 1.0
    return torch.stack([a0, a1], dim=-1)

def sweep_best_f1_binary(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Sweep thresholds and return the best-F1 operating point for binary classification.

    Intended for validation: it gives you `best_f1`/`best_thresh` without having to commit
    to a fixed 0.5 threshold while the model is still calibrating.

    Tie-break: among equal best F1, pick the *highest* threshold (more conservative).
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if probs.size == 0 or labels.size == 0:
        return {}
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, num=101, dtype=np.float32)

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

def compute_occl_dir_stats(
    occl_union_logits: torch.Tensor,
    occl_union: torch.Tensor,
    union_tau: float,
) -> Dict[str, float]:
    """
    Fruit-direction stats (dominant direction classification derived from union>=tau).
    occl_union_logits: [F, D]
    occl_union: [F, D] in [0,1]
    """
    if occl_union_logits is None or occl_union is None:
        return {}
    if occl_union_logits.numel() == 0 or occl_union.numel() == 0:
        return {}
    probs = torch.sigmoid(occl_union_logits.detach())
    y_dom = (occl_union.detach() >= float(union_tau)).float()
    pred = (probs >= 0.5).float()

    # Flatten across fruits*dirs.
    p = pred.flatten().cpu().numpy().astype(np.int32)
    y = y_dom.flatten().cpu().numpy().astype(np.int32)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)

    brier_dom = float(torch.mean((probs - y_dom) ** 2).item())
    brier_union = float(torch.mean((probs - occl_union.detach().float()) ** 2).item())
    ece_dom = binary_ece(probs.flatten(), y_dom.flatten(), num_bins=15)
    pos_rate = float(y_dom.mean().item())
    pred_pos_rate = float(pred.mean().item())
    return {
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "brier_dom": brier_dom,
        "brier_union": brier_union,
        "ece_dom": ece_dom,
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
    }

def compute_occl_leaf_retrieval_stats(
    pair_logits: torch.Tensor,
    pair_potential: torch.Tensor,
    pair_probs: Optional[torch.Tensor],
    pair_index: torch.Tensor,
    fruit_idx: torch.Tensor,
    fruit_union: torch.Tensor,
    union_tau: Optional[float],
    pot_eps: float,
    *,
    union_lo: Optional[float] = None,
    union_hi: Optional[float] = None,
    union_lo_inclusive: bool = True,
    union_hi_inclusive: bool = True,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Leaf retrieval stats for a selected set of union directions.
    Uses GT positives defined by potential>=pot_eps.

    pair_logits: [M, D]
    pair_potential: [M, D] in [0,1]
    pair_index: [2, M] (fruit_node_idx, leaf_node_idx)
    fruit_idx: [F] node indices corresponding to fruit_union rows
    fruit_union: [F, D]
    union selection:
      - if (union_lo, union_hi) are both None:
          - if union_tau is None: evaluate all directions
          - else: evaluate directions with union>=union_tau
      - else: evaluate directions in the (lo/hi)-bounded range with configurable inclusivity
    """
    if pair_logits is None or pair_potential is None:
        return {}
    if pair_logits.numel() == 0 or pair_potential.numel() == 0:
        return {}
    if pair_index is None or pair_index.numel() == 0:
        return {}
    if fruit_idx is None or fruit_union is None or fruit_idx.numel() == 0:
        return {}

    if pair_probs is not None:
        probs = pair_probs.detach().cpu().numpy()              # [M,D]
    else:
        probs = torch.sigmoid(pair_logits.detach()).cpu().numpy()  # [M,D]
    pot = pair_potential.detach().cpu().numpy()               # [M,D]
    pair_fruit = pair_index[0].detach().cpu().numpy().astype(np.int64)  # [M]

    fruit_idx_np = fruit_idx.detach().cpu().numpy().astype(np.int64)
    union_np = fruit_union.detach().cpu().numpy()
    fruit_to_row = {int(i): int(r) for r, i in enumerate(fruit_idx_np.tolist())}

    rec_lists = {k: [] for k in ks}

    def _dir_selected(u: float) -> bool:
        if union_lo is None and union_hi is None:
            if union_tau is None:
                return True
            return u >= float(union_tau)
        if union_lo is not None:
            if union_lo_inclusive:
                if u < float(union_lo):
                    return False
            else:
                if u <= float(union_lo):
                    return False
        if union_hi is not None:
            if union_hi_inclusive:
                if u > float(union_hi):
                    return False
            else:
                if u >= float(union_hi):
                    return False
        return True
    for fnode in fruit_idx_np.tolist():
        row = fruit_to_row.get(int(fnode))
        if row is None:
            continue
        union_f = union_np[row]  # [D]
        # pairs belonging to this fruit
        m = (pair_fruit == int(fnode))
        if not np.any(m):
            continue
        probs_f = probs[m]  # [L,D]
        pot_f = pot[m]      # [L,D]

        for d in range(pot_f.shape[1]):
            if not _dir_selected(float(union_f[d])):
                continue
            y = (pot_f[:, d] >= float(pot_eps))
            n_pos = int(y.sum())
            if n_pos == 0:
                continue
            order = np.argsort(-probs_f[:, d])
            for k in ks:
                kk = min(int(k), int(order.shape[0]))
                topk = order[:kk]
                rec = float(y[topk].sum()) / float(n_pos)
                rec_lists[k].append(rec)

    out: Dict[str, float] = {}
    for k in ks:
        vals = rec_lists[k]
        out[f"recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0
    return out

def compute_occl_leaf_mass_at_k_stats(
    pair_logits: torch.Tensor,
    pair_mass: torch.Tensor,
    pair_probs: Optional[torch.Tensor],
    pair_index: torch.Tensor,
    fruit_idx: torch.Tensor,
    fruit_union: torch.Tensor,
    union_tau: Optional[float],
    *,
    union_lo: Optional[float] = None,
    union_hi: Optional[float] = None,
    union_lo_inclusive: bool = True,
    union_hi_inclusive: bool = True,
    pot: Optional[torch.Tensor] = None,
    pot_eps: Optional[float] = None,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Leaf retrieval stats for a selected set of union directions, but weighted by GT occlusion mass.

    This better matches the "top occluders" goal than Recall@K (which only counts leaves).

    pair_logits: [M, D]
    pair_mass: [M, D] in [0,1] (e.g. k-buffer mass: sum(pair_k_rank, axis=-1))
    pot: optional [M, D] in [0,1]; if provided with pot_eps, we zero-out pair_mass where pot<pot_eps
         to drop tiny/pixel-noise contributions.
    """
    if pair_logits is None or pair_mass is None:
        return {}
    if pair_logits.numel() == 0 or pair_mass.numel() == 0:
        return {}
    if pair_index is None or pair_index.numel() == 0:
        return {}
    if fruit_idx is None or fruit_union is None or fruit_idx.numel() == 0:
        return {}

    if pair_probs is not None:
        probs = pair_probs.detach().cpu().numpy()              # [M,D]
    else:
        probs = torch.sigmoid(pair_logits.detach()).cpu().numpy()  # [M,D]
    mass = pair_mass.detach().cpu().numpy()                   # [M,D]
    if pot is not None and pot_eps is not None:
        pot_np = pot.detach().cpu().numpy()
        mass = np.where(pot_np >= float(pot_eps), mass, 0.0)

    pair_fruit = pair_index[0].detach().cpu().numpy().astype(np.int64)  # [M]
    fruit_idx_np = fruit_idx.detach().cpu().numpy().astype(np.int64)
    union_np = fruit_union.detach().cpu().numpy()
    fruit_to_row = {int(i): int(r) for r, i in enumerate(fruit_idx_np.tolist())}

    mass_lists = {k: [] for k in ks}

    def _dir_selected(u: float) -> bool:
        if union_lo is None and union_hi is None:
            if union_tau is None:
                return True
            return u >= float(union_tau)
        if union_lo is not None:
            if union_lo_inclusive:
                if u < float(union_lo):
                    return False
            else:
                if u <= float(union_lo):
                    return False
        if union_hi is not None:
            if union_hi_inclusive:
                if u > float(union_hi):
                    return False
            else:
                if u >= float(union_hi):
                    return False
        return True
    for fnode in fruit_idx_np.tolist():
        row = fruit_to_row.get(int(fnode))
        if row is None:
            continue
        union_f = union_np[row]  # [D]
        m = (pair_fruit == int(fnode))
        if not np.any(m):
            continue
        probs_f = probs[m]  # [L,D]
        mass_f = mass[m]    # [L,D]

        for d in range(mass_f.shape[1]):
            if not _dir_selected(float(union_f[d])):
                continue
            gt_mass = mass_f[:, d]
            total = float(gt_mass.sum())
            if total <= 0.0:
                continue
            order = np.argsort(-probs_f[:, d])
            for k in ks:
                kk = min(int(k), int(order.shape[0]))
                topk = order[:kk]
                frac = float(gt_mass[topk].sum()) / (total + 1e-12)
                mass_lists[k].append(frac)

    out: Dict[str, float] = {}
    for k in ks:
        vals = mass_lists[k]
        out[f"mass_at_{k}"] = float(np.mean(vals)) if vals else 0.0
    return out

def compute_occl_leaf_ndcg_at_k_stats(
    pair_logits: torch.Tensor,
    pair_rel: torch.Tensor,
    pair_probs: Optional[torch.Tensor],
    pair_index: torch.Tensor,
    fruit_idx: torch.Tensor,
    fruit_union: torch.Tensor,
    union_tau: Optional[float],
    *,
    union_lo: Optional[float] = None,
    union_hi: Optional[float] = None,
    union_lo_inclusive: bool = True,
    union_hi_inclusive: bool = True,
    pot: Optional[torch.Tensor] = None,
    pot_eps: Optional[float] = None,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    NDCG@K for leaf ranking on dominant directions only, using graded relevance.

    This complements Mass@K:
      - Mass@K measures "how much GT mass is captured by top-K predicted leaves",
        but is insensitive to the ordering *within* the top-K.
      - NDCG@K discounts by rank and therefore rewards putting the strongest occluders earlier.

    pair_rel: [M, D] graded relevance (e.g. k-buffer mass = sum(pair_k_rank, axis=-1)).
    pot/pot_eps (optional): if provided, we zero-out relevance where potential < pot_eps to
    drop tiny/pixel-noise contributions (aligns with other occlusion metrics).
    """
    if pair_logits is None or pair_rel is None:
        return {}
    if pair_logits.numel() == 0 or pair_rel.numel() == 0:
        return {}
    if pair_index is None or pair_index.numel() == 0:
        return {}
    if fruit_idx is None or fruit_union is None or fruit_idx.numel() == 0:
        return {}

    if pair_probs is not None:
        probs = pair_probs.detach().cpu().numpy()              # [M,D]
    else:
        probs = torch.sigmoid(pair_logits.detach()).cpu().numpy()  # [M,D]
    rel = pair_rel.detach().cpu().numpy()                     # [M,D]
    if pot is not None and pot_eps is not None:
        pot_np = pot.detach().cpu().numpy()
        rel = np.where(pot_np >= float(pot_eps), rel, 0.0)

    pair_fruit = pair_index[0].detach().cpu().numpy().astype(np.int64)  # [M]
    fruit_idx_np = fruit_idx.detach().cpu().numpy().astype(np.int64)
    union_np = fruit_union.detach().cpu().numpy()
    fruit_to_row = {int(i): int(r) for r, i in enumerate(fruit_idx_np.tolist())}

    ndcg_lists = {k: [] for k in ks}

    def _dir_selected(u: float) -> bool:
        if union_lo is None and union_hi is None:
            if union_tau is None:
                return True
            return u >= float(union_tau)
        if union_lo is not None:
            if union_lo_inclusive:
                if u < float(union_lo):
                    return False
            else:
                if u <= float(union_lo):
                    return False
        if union_hi is not None:
            if union_hi_inclusive:
                if u > float(union_hi):
                    return False
            else:
                if u >= float(union_hi):
                    return False
        return True
    for fnode in fruit_idx_np.tolist():
        row = fruit_to_row.get(int(fnode))
        if row is None:
            continue
        union_f = union_np[row]  # [D]
        m = (pair_fruit == int(fnode))
        if not np.any(m):
            continue
        probs_f = probs[m]  # [L,D]
        rel_f = rel[m]      # [L,D]

        for d in range(rel_f.shape[1]):
            if not _dir_selected(float(union_f[d])):
                continue
            gt_rel = rel_f[:, d]
            if float(gt_rel.sum()) <= 0.0:
                continue

            order_pred = np.argsort(-probs_f[:, d])
            order_ideal = np.argsort(-gt_rel)
            for k in ks:
                kk = min(int(k), int(order_pred.shape[0]))
                if kk <= 0:
                    continue
                discount = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
                top_pred = order_pred[:kk]
                top_ideal = order_ideal[:kk]
                dcg = float((gt_rel[top_pred] * discount).sum())
                idcg = float((gt_rel[top_ideal] * discount).sum())
                ndcg = dcg / (idcg + 1e-12)
                ndcg_lists[k].append(ndcg)

    out: Dict[str, float] = {}
    for k in ks:
        vals = ndcg_lists[k]
        out[f"ndcg_at_{k}"] = float(np.mean(np.clip(vals, 0.0, 1.0))) if vals else 0.0
    return out

def _append_stats(dst: Dict[str, List[float]], src: Dict[str, float]) -> None:
    """
    Append scalar stats into a dict-of-lists without requiring pre-declared keys.

    This prevents fragile KeyErrors when we add new occlusion metric fields (e.g. oracle_*/_*_norm).
    """
    for k, v in src.items():
        dst.setdefault(k, []).append(float(v))

def print_text_confusion_matrix(matrix, class_names, title="Confusion Matrix"):
    # matrix: [C, C] tensor or ndarray
    # Rows = Ground Truth, Cols = Prediction
    print(f"\n--- {title} ---", flush=True)
    header = "GT\\Pred".center(10) + " | " + " | ".join([f"{name[:8]:>8}" for name in class_names])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, name in enumerate(class_names):
        row_vals = " | ".join([f"{int(matrix[i, j]):>8}" for j in range(len(class_names))])
        print(f"{name[:9]:>10} | {row_vals}", flush=True)
    print("-" * len(header), flush=True)

def aggregate_per_class_and_return(metrics_list, class_names, prefix):
    if not metrics_list: return {}
    
    agg_dict = {}
    # Average overall metrics
    agg_dict[f"{prefix}_acc"] = np.mean([m['acc'] for m in metrics_list])
    agg_dict[f"{prefix}_f1_macro"] = np.mean([m['f1_macro'] for m in metrics_list])
    agg_dict[f"{prefix}_prec_macro"] = np.mean([m['prec_macro'] for m in metrics_list])
    agg_dict[f"{prefix}_rec_macro"] = np.mean([m['rec_macro'] for m in metrics_list])
    
    # Average per-class metrics
    avg_prec = np.mean([m['precision'] for m in metrics_list], axis=0)
    avg_rec = np.mean([m['recall'] for m in metrics_list], axis=0)
    avg_f1 = np.mean([m['f1'] for m in metrics_list], axis=0)
    
    for i, name in enumerate(class_names):
        agg_dict[f"{prefix}_{name}_precision"] = avg_prec[i]
        agg_dict[f"{prefix}_{name}_recall"] = avg_rec[i]
        agg_dict[f"{prefix}_{name}_f1"] = avg_f1[i]
    return agg_dict

def compute_weights(dataset, num_classes):
    print("Computing class weights (node)...", flush=True)
    # Use 100 random samples to estimate distribution
    indices = np.random.choice(len(dataset), min(len(dataset), 100), replace=False)
    counts = np.zeros(num_classes)
    for i in tqdm(indices, desc="Weight Estimation"):
        batch = dataset[i]
        lbl = batch.y_node.cpu().numpy()
        for c in range(num_classes):
             counts[c] += np.sum(lbl == c)
             
    counts[counts == 0] = 1 # avoid inf
    total = np.sum(counts)
    weights = total / (num_classes * counts)
    weights = weights / np.mean(weights)
    print(f"Node Weights: {weights}", flush=True)
    return torch.from_numpy(weights).float()

def build_oracle_targets(edge_index, batch):
    """
    On-GPU target builder for dynamically generated candidates.
    Matches edge_index [2, E_pred] against batch.gt_edge_index [2, E_gt]
    """
    if edge_index.numel() == 0:
        return torch.zeros(0, device=edge_index.device), \
               torch.full((0,), -1, dtype=torch.long, device=edge_index.device), \
               0
               
    # Hashing: (u, v) -> u * N + v
    # Use a large enough multiplier to avoid collisions
    num_nodes = batch.y_node.size(0)
    
    # Sort pred edges to ensure canonical undirected key for matching
    # Note: model's edge_index is already bidirectional, so we match both (u,v) and (v,u) 
    # if we want to supervise every message. Or we can just sort and hash.
    # To keep it simple: we want y_exist to be 1 for BOTH directions if the edge is GT.
    
    # Pred Hashing
    pred_u, pred_v = edge_index[0], edge_index[1]
    # Canonical undirected key
    pred_min = torch.min(pred_u, pred_v)
    pred_max = torch.max(pred_u, pred_v)
    pred_hash = pred_min * num_nodes + pred_max
    
    # GT Hashing (batch.gt_edge_index is undirected)
    gt_u, gt_v = batch.gt_edge_index[0], batch.gt_edge_index[1]
    gt_min = torch.min(gt_u, gt_v)
    gt_max = torch.max(gt_u, gt_v)
    gt_hash = gt_min * num_nodes + gt_max
    
    # 1. Existence Target
    is_gt = torch.isin(pred_hash, gt_hash)
    y_exist = is_gt.float() # [E_pred]
    
    # 2. Label Target
    y_edge = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=edge_index.device)
    if gt_hash.numel() > 0:
        # Sort GT hash to allow indexing
        gt_perm = torch.argsort(gt_hash)
        gt_hash_sorted = gt_hash[gt_perm]
        gt_labels_sorted = batch.gt_edge_label[gt_perm]
        
        # Find where each pred hash sits in GT sorted list
        match_idx = torch.searchsorted(gt_hash_sorted, pred_hash)
        # Verify valid matches (torch.isin already confirmed existence, but we need exact index)
        valid_match = (match_idx < gt_hash_sorted.size(0))
        # Ensure the index found actually matches the hash
        actual_match = valid_match & (gt_hash_sorted[match_idx.clamp(max=gt_hash_sorted.size(0)-1)] == pred_hash)
        
        y_edge[actual_match] = gt_labels_sorted[match_idx[actual_match].clamp(max=gt_labels_sorted.size(0)-1)]
        
    # num_gt_in_cand: unique pairs found
    num_gt_in_cand = torch.unique(pred_hash[is_gt]).numel()
    return y_exist, y_edge, num_gt_in_cand

def compute_oracle_loss(preds, batch, args, epoch, batch_idx=0, node_weights=None):
    # 0. Local Node Loss (Always trained)
    y_node = batch.y_node
    loss_node_local = F.cross_entropy(preds['local_node_logits'], y_node, weight=node_weights)

    # Evidential head toggles (all off by default).
    edl_node = bool(getattr(args, "edl_node", False))
    edl_edge_exist = bool(getattr(args, "edl_edge_exist", False))
    edl_edge_label = bool(getattr(args, "edl_edge_label", False))
    edl_occl_potential = bool(getattr(args, "edl_occl_potential", False))
    edl_occl_rank = bool(getattr(args, "edl_occl_rank", False))
    edl_any = edl_node or edl_edge_exist or edl_edge_label or edl_occl_potential or edl_occl_rank
    edl_fn = _get_edl_evidence_fn(args) if edl_any else None
    kl_coeff = _get_edl_kl_coeff(args, int(epoch)) if edl_any else 0.0
    kl_coeff_edge_exist = _get_edl_kl_coeff_edge_exist(args, int(epoch)) if edl_edge_exist else 0.0
    
    # Define Curriculum Weights
    # Phase 1: 1-5  -> Only Local Node
    # Phase 2: 6-10 -> + Regression
    # Phase 3: 11+  -> + GNN (Global Node, Edge Exist, Edge Label)
    
    w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
    w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0
    
    # 1. Node Loss (Global/Refined)
    node_logits = preds['node_logits']
    if edl_node and (edl_fn is not None):
        alpha = torch.clamp(edl_fn(node_logits), min=0.0) + 1.0  # [N,C]
        y = y_node.long()
        data_term = _edl_dirichlet_ce_per_sample(alpha, y)  # [N]
        if node_weights is not None:
            w = node_weights.to(device=data_term.device, dtype=data_term.dtype)[y]
            data_term = data_term * w
        loss_node = data_term.mean() + float(kl_coeff) * _edl_kl_alpha_tilde_per_sample(alpha, y).mean()
    else:
        loss_node = F.cross_entropy(node_logits, y_node, weight=node_weights)
    
    # 2. Edge Targets & Existence
    edge_index = preds['edge_index']
    y_exist, y_edge, num_gt_in_cand = build_oracle_targets(edge_index, batch)

    edge_exist_logits = preds['edge_exist_logits'].squeeze(-1) # [E]
    if edge_exist_logits.numel() > 0:
        if getattr(args, 'dynamic_edge_weight', False):
            with torch.no_grad():
                num_pos = (y_exist == 1).sum().float()
                num_neg = (y_exist == 0).sum().float()
                # Dynamic weight for positive class: num_neg / num_pos
                # Clamp to avoid extreme values
                pw_val = (num_neg / num_pos.clamp(min=1.0)).clamp(min=1.0, max=100.0)
                pw = torch.tensor([pw_val], device=node_logits.device)
        else:
            pw_edge = getattr(args, "pos_weight_edge_exist", None)
            if pw_edge is None:
                pw_edge = args.pos_weight
            pw = torch.tensor([float(pw_edge)], device=node_logits.device)

        if edl_edge_exist and (edl_fn is not None):
            y = y_exist.long()
            alpha = _edl_tied_beta_alpha(edge_exist_logits, edl_fn)  # [E,2]
            data_term = _edl_dirichlet_ce_per_sample(alpha, y)  # [E]
            # Match BCEWithLogits(pos_weight=...) semantics: weight positives only.
            w = torch.where(y > 0, pw.to(dtype=data_term.dtype), torch.ones_like(data_term))
            loss_edge_exist = (data_term * w).mean() + float(kl_coeff_edge_exist) * _edl_kl_alpha_tilde_per_sample(alpha, y).mean()
        else:
            loss_edge_exist = F.binary_cross_entropy_with_logits(edge_exist_logits, y_exist, pos_weight=pw)
    else:
        loss_edge_exist = torch.tensor(0.0).to(node_logits.device)
        
    # 3. Edge Labeling
    edge_label_logits = preds['edge_label_logits']
    edge_mask = (y_edge != -1)
    if edge_mask.any():
        if edl_edge_label and (edl_fn is not None):
            pos_logits = edge_label_logits[edge_mask]
            y = y_edge[edge_mask].long()
            alpha = torch.clamp(edl_fn(pos_logits), min=0.0) + 1.0
            data_term = _edl_dirichlet_ce_per_sample(alpha, y)
            loss_edge_label = data_term.mean() + float(kl_coeff) * _edl_kl_alpha_tilde_per_sample(alpha, y).mean()
        else:
            loss_edge_label = F.cross_entropy(edge_label_logits[edge_mask], y_edge[edge_mask])
    else:
        loss_edge_label = torch.tensor(0.0).to(node_logits.device)
        
    # 4. Regression
    centroid_offset = preds['centroid_offset']
    extent_pred = preds['extent_delta'] 
    
    if getattr(args, 'no_geom_injection', False):
        # Target is the vector FROM cluster_mean TO GT_centroid
        # cluster_mean is where the model centered the points
        target_offset = batch.json_geom[:, :3] - preds['cluster_mean']
        target_extent = batch.json_geom[:, 3:]
    else:
        # Predict delta (forced to 0 because input is already centered at GT)
        target_offset = torch.zeros_like(centroid_offset)
        target_extent = torch.zeros_like(extent_pred)

    loss_centroid = F.smooth_l1_loss(centroid_offset, target_offset)
    loss_extent = F.smooth_l1_loss(extent_pred, target_extent)
    
    # [V12 Audit] Calculate L1 MAE for better monitoring
    with torch.no_grad():
        mae_centroid = torch.abs(centroid_offset - target_offset).mean()
        mae_extent = torch.abs(extent_pred - target_extent).mean()

    # 5. Occlusion (optional; pointcloud z-buffer supervision)
    w_occl_curric = 1.0 if (getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0)) else 0.0
    loss_occl_union = torch.tensor(0.0, device=node_logits.device)
    loss_occl_potential = torch.tensor(0.0, device=node_logits.device)
    loss_occl_rank = torch.tensor(0.0, device=node_logits.device)
    loss_occl_consistency = torch.tensor(0.0, device=node_logits.device)
    occl_union_explain_w_mean = 0.0
    occl_union_explain_w_pos_mean = 0.0
    if getattr(args, "occl_enabled", False) and w_occl_curric > 0:
        occl_union_logits = preds.get("occl_fruit_union_logits", None)
        occl_pair_pot_logits = preds.get("occl_pair_potential_logits", None)
        # Rank logits are kept separate from potential logits to avoid conflicting supervision
        # (continuous potential regression vs evidential listwise rank).
        occl_pair_rank_logits = preds.get("occl_pair_rank_logits", occl_pair_pot_logits)

        y_union = getattr(batch, "occl_fruit_union", None)
        y_pot = getattr(batch, "occl_pair_potential", None)
        y_excl = getattr(batch, "occl_pair_exclusive", None)
        fruit_idx = getattr(batch, "occl_fruit_idx", None)
        pair_index = getattr(batch, "occl_pair_index", None)

        device = node_logits.device
        # Union explainability weight (how much of union is attributable to leaves present in this subgraph).
        # Used both for union supervision and (optionally) for the leaf->union consistency term.
        w_union_explain = None
        if y_union is not None and y_union.numel() > 0:
            w_union_explain = torch.ones_like(y_union.to(device), dtype=torch.float32, device=device)

        if occl_union_logits is not None and y_union is not None and occl_union_logits.numel() > 0 and y_union.numel() > 0:
            # Soft target in [0,1]: probability a random fruit pixel is occluded from that direction.
            #
            # Important: our occlusion supervision may include occluders (leaves) that are *not present* in the sampled
            # subgraph / candidate pair set. Without care, this creates an "impossible target" for a leaf-pooled union head.
            #
            # Fix: weight each (fruit,dir) union BCE term by how much of the full union is explainable by the *present*
            # candidate leaves. We estimate explainable union as the sum of exclusive (rank-1) mass over present leaves.
            # For union==0 directions we keep weight=1 (always explainable).
            y_union_dev = y_union.to(device)
            if w_union_explain is None:
                w_union_explain = torch.ones_like(y_union_dev, dtype=torch.float32, device=device)
            if (
                y_excl is not None
                and pair_index is not None
                and fruit_idx is not None
                and y_excl.numel() > 0
                and pair_index.numel() > 0
                and fruit_idx.numel() > 0
            ):
                D = int(y_union_dev.size(1))
                # Map node indices -> row indices in y_union (focus fruits only).
                row_of_node = torch.full((batch.y_node.size(0),), -1, device=device, dtype=torch.long)
                fruit_idx_dev = fruit_idx.to(device)
                row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device, dtype=torch.long)
                pair_fruit_row = row_of_node[pair_index[0].to(device)]  # [M]
                valid = pair_fruit_row >= 0
                if bool(valid.any()):
                    # union_expl[r,d] = sum exclusive mass over present leaves for this fruit.
                    union_expl = torch.zeros((fruit_idx_dev.numel(), D), device=device, dtype=torch.float32)
                    union_expl.index_add_(0, pair_fruit_row[valid], y_excl.to(device)[valid].float())
                    union_expl = union_expl.clamp(min=0.0, max=1.0)
                    denom = torch.clamp(y_union_dev.float(), min=1e-6)
                    frac = torch.clamp(union_expl / denom, min=0.0, max=1.0)
                    # Keep union==0 directions fully supervised (negatives are always "explainable").
                    w_union_explain = torch.where(y_union_dev.float() > 1e-6, frac, torch.ones_like(frac))

                    # Debug: how much of the union is explainable by present leaves (pos dirs only).
                    with torch.no_grad():
                        occl_union_explain_w_mean = float(w_union_explain.mean().item())
                        pos = (y_union_dev.float() > 1e-6)
                        if bool(pos.any()):
                            occl_union_explain_w_pos_mean = float(w_union_explain[pos].mean().item())

            bce = F.binary_cross_entropy_with_logits(occl_union_logits, y_union_dev, reduction="none")  # [F,D]
            w_sum = w_union_explain.sum()
            if float(w_sum.item()) > 0.0:
                loss_occl_union = (bce * w_union_explain).sum() / (w_sum + 1e-12)
            else:
                loss_occl_union = torch.zeros((), device=device)

        if (
            occl_pair_pot_logits is not None and y_pot is not None and pair_index is not None and fruit_idx is not None and y_union is not None
            and occl_pair_pot_logits.numel() > 0 and y_pot.numel() > 0 and pair_index.numel() > 0 and fruit_idx.numel() > 0 and y_union.numel() > 0
        ):
            # For the potential BCE, we supervise all dominant directions (union>=tau) and also a small sample of
            # mid/low-occlusion directions by default (stratified), to avoid biasing the model to only heavily-occluded
            # viewpoints while still not flooding it with trivial negatives.
            union_tau = float(getattr(args, "occl_union_tau", 0.5))
            pot_eps = float(getattr(args, "occl_potential_eps", 0.05))
            pot_dir_policy = str(getattr(args, "occl_potential_dir_sampling", "stratified")).strip().lower()
            union = y_union.to(device)
            cons_mode = str(getattr(args, "occl_consistency_mode", "auto")).strip().lower()
            if cons_mode == "auto":
                cons_mode = "bce_dom" if (edl_occl_potential and (edl_fn is not None)) else "mse"

            # Build a per-(fruit,dir) selection mask for the potential BCE.
            if pot_dir_policy == "stratified":
                tau_lo = float(getattr(args, "occl_potential_union_tau_low", 0.25))
                n_mid = int(getattr(args, "occl_potential_n_mid_dirs", 3))
                n_low = int(getattr(args, "occl_potential_n_low_dirs", 1))
                Ff, D = int(union.size(0)), int(union.size(1))
                sel_dirs = torch.zeros((Ff, D), dtype=torch.bool, device=device)
                high = (union >= union_tau)
                mid = (union > tau_lo) & (union < union_tau)
                low = (union <= tau_lo)
                sel_dirs |= high
                for r in range(Ff):
                    if n_mid > 0:
                        mid_idx = torch.nonzero(mid[r], as_tuple=False).flatten()
                        if int(mid_idx.numel()) > 0:
                            k = min(n_mid, int(mid_idx.numel()))
                            perm = torch.randperm(int(mid_idx.numel()), device=device)[:k]
                            sel_dirs[r, mid_idx[perm]] = True
                    if n_low > 0:
                        low_idx = torch.nonzero(low[r], as_tuple=False).flatten()
                        if int(low_idx.numel()) > 0:
                            k = min(n_low, int(low_idx.numel()))
                            perm = torch.randperm(int(low_idx.numel()), device=device)[:k]
                            sel_dirs[r, low_idx[perm]] = True
            else:
                # Backwards-compatible: dominant-only.
                sel_dirs = (union >= union_tau)

            # Map pair fruit nodes -> supervised fruit rows, then broadcast to [M,D].
            fruit_idx_dev = fruit_idx.to(device)
            row_of_node = torch.full((batch.y_node.size(0),), -1, device=device, dtype=torch.long)
            row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device, dtype=torch.long)
            pair_fruit_row = row_of_node[pair_index[0].to(device)]  # [M]
            valid = pair_fruit_row >= 0
            if bool(valid.any()):
                pair_fruit_row_safe = pair_fruit_row.clamp(min=0)
                sel_mask = sel_dirs[pair_fruit_row_safe]  # [M,D]
                if not bool(valid.all()):
                    sel_mask = sel_mask & valid.unsqueeze(1)

                if bool(sel_mask.any()):
                    logits_flat = occl_pair_pot_logits.to(device)[sel_mask]
                    y_flat = y_pot.to(device)[sel_mask]
                    # Drop "dust" supervision: treat very small potential occlusions as negatives.
                    # This matches how retrieval positives are defined (potential>=eps) and reduces label noise.
                    y_flat = torch.where(y_flat >= pot_eps, y_flat, torch.zeros_like(y_flat))
                    if edl_occl_potential and (edl_fn is not None):
                        # Evidential (binary) variant: supervise "is an occluder" (potential>=eps).
                        # We intentionally binarize here because EDL-Dirichlet CE is a classification loss.
                        y_bin = (y_flat >= pot_eps).long()
                        alpha = _edl_tied_beta_alpha(logits_flat, edl_fn)  # [K,2]
                        data_term = _edl_dirichlet_ce_per_sample(alpha, y_bin)
                        # Use the same pos_weight knob as edge existence for now (simple + stable).
                        pw = torch.tensor(float(getattr(args, "pos_weight", 10.0)), device=device, dtype=data_term.dtype)
                        w = torch.where(y_bin > 0, pw, torch.ones_like(data_term))
                        loss_occl_potential = (data_term * w).mean() + float(kl_coeff) * _edl_kl_alpha_tilde_per_sample(alpha, y_bin).mean()
                    else:
                        # Default: BCE regression to the soft target in [0,1].
                        loss_occl_potential = F.binary_cross_entropy_with_logits(logits_flat, y_flat)

            # (2.2) Listwise ranking loss: for each (fruit, dir), rank leaves by potential occlusion.
            # We form a soft target distribution t_l ∝ potential_l (after eps gating) and optimize
            # -sum t_l log softmax(score_l) across the candidate leaves.
            # (2.3) Consistency regularizer: aggregate predicted per-leaf occlusion into a union estimate
            # and regress it towards the GT union, using the same dominant-direction gating.
            try:
                D = int(y_union.size(1))
                # Map node indices -> row indices in y_union (focus fruits only).
                row_of_node = torch.full((batch.y_node.size(0),), -1, device=device, dtype=torch.long)
                fruit_idx_dev = fruit_idx.to(device)
                row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device, dtype=torch.long)
                pair_fruit_row = row_of_node[pair_index[0].to(device)]  # [M]
                valid = pair_fruit_row >= 0

                if bool(valid.any()):
                    pair_fruit_row = pair_fruit_row[valid]
                    pair_logits_pot = occl_pair_pot_logits[valid]
                    pair_logits_rank = occl_pair_rank_logits.to(device)[valid] if (occl_pair_rank_logits is not None) else pair_logits_pot
                    pot = y_pot.to(device)[valid]
                    union = y_union.to(device)
                    k_rank = getattr(batch, "occl_pair_k_rank", None)
                    k_sum = None
                    if k_rank is not None and k_rank.numel() > 0:
                        # K-buffer "top-3 mass": how often a leaf is among the top-K occluders per pixel.
                        # This is aligned with the goal of finding leaves that remain problematic even if the
                        # closest occluder is removed.
                        k_sum = k_rank.to(device)[valid].sum(dim=-1)  # [M, D]
                        # Drop "dust" contributions (pixel noise / near-zero overlaps).
                        k_sum = torch.where(pot >= pot_eps, k_sum, torch.zeros_like(k_sum))
                        gamma = float(getattr(args, "occl_rank_gamma", 1.0))
                        if gamma != 1.0:
                            k_sum = torch.clamp(k_sum, min=0.0).pow(gamma)

                    # --- Listwise ranking (per fruit) ---
                    rank_loss_sum = torch.zeros((), device=device)
                    rank_kl_sum = torch.zeros((), device=device)
                    rank_count = torch.zeros((), device=device)
                    for r in range(int(fruit_idx_dev.numel())):
                        m = (pair_fruit_row == r)
                        if not bool(m.any()):
                            continue
                        logits_r = pair_logits_rank[m]  # [L, D]
                        pot_r = pot[m]            # [L, D]
                        if k_sum is not None:
                            mass_r = k_sum[m]     # [L, D]
                        else:
                            mass_r = pot_r

                        mass_r = torch.where(pot_r >= pot_eps, mass_r, torch.zeros_like(mass_r))
                        sum_mass = mass_r.sum(dim=0)  # [D]
                        use = (union[r] >= union_tau) & (sum_mass > 0)
                        if not bool(use.any()):
                            continue

                        t = mass_r / (sum_mass.unsqueeze(0) + 1e-12)  # [L,D]
                        if edl_occl_rank and (edl_fn is not None):
                            # Evidential listwise: treat leaf scores as evidence for a Dirichlet distribution
                            # over leaves, and match the GT mass distribution via expected log-likelihood.
                            alpha = torch.clamp(edl_fn(logits_r), min=0.0) + 1.0  # [L,D]
                            S = alpha.sum(dim=0, keepdim=True)  # [1,D]
                            e_log_p = torch.digamma(alpha) - torch.digamma(S)  # [L,D]
                            loss_dir = -(t * e_log_p).sum(dim=0)  # [D]

                            # KL regularizer to uniform Dirichlet. Coefficient is separate because K (#leaves)
                            # varies and KL can be large for large candidate sets.
                            klc_rank = float(getattr(args, "edl_kl_coeff_rank", 0.01))
                            if klc_rank > 0.0:
                                # Treat each direction as a separate Dirichlet sample.
                                kld_dir = _edl_dirichlet_kld_per_sample(alpha.transpose(0, 1))  # [D]
                                rank_kl_sum = rank_kl_sum + kld_dir[use].sum()
                        else:
                            # Default listwise: softmax over leaves (dim=0), independently per direction.
                            log_p = F.log_softmax(logits_r, dim=0)  # [L, D]
                            loss_dir = -(t * log_p).sum(dim=0)  # [D]

                        rank_loss_sum = rank_loss_sum + loss_dir[use].sum()
                        rank_count = rank_count + use.float().sum()

                    if float(rank_count.item()) > 0.0:
                        loss_occl_rank = rank_loss_sum / rank_count
                        if edl_occl_rank and float(getattr(args, "edl_kl_coeff_rank", 0.01)) > 0.0:
                            loss_occl_rank = loss_occl_rank + float(getattr(args, "edl_kl_coeff_rank", 0.01)) * (rank_kl_sum / rank_count)

                    # --- Leaf->union consistency (dominant dirs with at least one positive leaf) ---
                    if cons_mode != "off":
                        if cons_mode == "bce_dom":
                            # EDL potential is a binary "is an occluder" objective (potential>=eps), so the per-leaf
                            # p_mean is not an occlusion *amount*. Matching continuous union via MSE is ill-posed and
                            # tends to plateau. Instead, encourage leaf predictions to be consistent with *dominance*
                            # (union>=tau) using BCE on an aggregated union probability.
                            if edl_occl_potential and (edl_fn is not None):
                                alpha_p = _edl_tied_beta_alpha(pair_logits_pot, edl_fn)  # [M,D,2]
                                p = alpha_p[..., 1] / (alpha_p.sum(dim=-1) + 1e-12)  # [M,D]
                            else:
                                p = torch.sigmoid(pair_logits_pot)  # [M,D]

                            log_not = torch.log(torch.clamp(1.0 - p, min=1e-6))  # [M,D]
                            sum_log_not = torch.zeros((fruit_idx_dev.numel(), D), device=device, dtype=torch.float32)
                            sum_log_not.index_add_(0, pair_fruit_row, log_not)
                            u_pred = 1.0 - torch.exp(sum_log_not)  # [F,D] in [0,1]

                            y_dom = (union >= union_tau).float()  # [F,D]
                            # Use the same direction selection as the potential BCE, so we get both positives and
                            # informative negatives without flooding training with trivial directions.
                            use = sel_dirs

                            # Weight positives by explainability (avoid punishing missing-leaf cases); keep negatives full weight.
                            if w_union_explain is not None:
                                w_eff = torch.where(y_dom > 0.5, w_union_explain, torch.ones_like(y_dom))
                            else:
                                w_eff = torch.ones_like(y_dom)
                            w_eff = w_eff * use.float()
                            w_sum = w_eff.sum()
                            if float(w_sum.item()) > 0.0:
                                u_clamped = u_pred.clamp(min=1e-6, max=1.0 - 1e-6)
                                bce_u = F.binary_cross_entropy(u_clamped, y_dom, reduction="none")
                                loss_occl_consistency = (bce_u * w_eff).sum() / (w_sum + 1e-12)
                        else:
                            # Default (non-EDL): regress aggregated leaf probabilities to continuous union (dominant dirs only).
                            gt_pos = (pot >= pot_eps).float()  # [M, D]
                            pos_count = torch.zeros((fruit_idx_dev.numel(), D), device=device, dtype=torch.float32)
                            pos_count.index_add_(0, pair_fruit_row, gt_pos)
                            has_pos = pos_count > 0

                            q = torch.sigmoid(pair_logits_pot) * gt_pos  # only consider "significant" GT occluders
                            log_not = torch.log(torch.clamp(1.0 - q, min=1e-6))  # [M, D]
                            sum_log_not = torch.zeros((fruit_idx_dev.numel(), D), device=device, dtype=torch.float32)
                            sum_log_not.index_add_(0, pair_fruit_row, log_not)
                            u_pred = 1.0 - torch.exp(sum_log_not)  # [F, D]

                            use_cons = (union >= union_tau) & has_pos
                            if bool(use_cons.any()):
                                loss_occl_consistency = F.mse_loss(u_pred[use_cons], union[use_cons])
            except Exception:
                # Keep training robust if a corner case slips through; losses stay at 0.
                pass

    # Curriculum Logic
    if getattr(args, 'reversed_curriculum', False):
        # Reversed: Local -> Reg -> GNN
        w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
        w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0
        # Node local is always 1.0
    else:
        # Original: GNN -> Reg
        w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
        w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0

    total_loss = (
        args.w_node * loss_node_local +
        w_gnn * (args.w_node * loss_node + 
                 args.w_edge_exist * loss_edge_exist + 
                 args.w_edge_label * loss_edge_label) +
        w_reg_curric * args.w_reg * (loss_centroid + loss_extent) +
        w_occl_curric * (
            getattr(args, "w_occl_union", 0.0) * loss_occl_union +
            getattr(args, "w_occl_potential", 0.0) * loss_occl_potential +
            getattr(args, "w_occl_rank", 0.0) * loss_occl_rank +
            getattr(args, "w_occl_consistency", 0.0) * loss_occl_consistency
        )
    )
    
    losses = {
        "loss_total": total_loss,
        "node_local": loss_node_local,
        "node": loss_node,
        "edge_exist": loss_edge_exist,
        "edge_label": loss_edge_label,
        "centroid": loss_centroid,
        "extent": loss_extent,
        "mae_centroid": mae_centroid,
        "mae_extent": mae_extent,
        "occl_union": loss_occl_union,
        "occl_potential": loss_occl_potential,
        "occl_rank": loss_occl_rank,
        "occl_consistency": loss_occl_consistency,
        "occl_union_explain_w_mean": occl_union_explain_w_mean,
        "occl_union_explain_w_pos_mean": occl_union_explain_w_pos_mean,
    }
    
    targets = {
        "y_exist": y_exist,
        "y_edge": y_edge,
        "num_gt_in_cand": num_gt_in_cand
    }
    
    return losses, targets

def validate(model, loader, device, args, epoch):
    model.eval()
    
    # Track raw lists for averaging
    node_local_metrics_list = []
    node_metrics_list = []
    edge_label_metrics_list = []

    # --- Generic calibration (for both EDL and non-EDL runs) ---
    # We want A0 vs A1 to be comparable on calibration metrics (ECE/Brier), so we log
    # these for the standard heads regardless of whether evidential losses are enabled.
    # For EDL runs we use the Dirichlet/Beta *mean* probabilities; otherwise softmax/sigmoid.
    node_calib_probs: List[torch.Tensor] = []
    node_calib_targets: List[torch.Tensor] = []
    edge_label_calib_probs: List[torch.Tensor] = []
    edge_label_calib_targets: List[torch.Tensor] = []

    # --- EDL (evidential) diagnostics (validation) ---
    # We keep this minimal: calibration (ECE/Brier) + evidence strength/uncertainty.
    edl_node_probs: List[torch.Tensor] = []
    edl_node_targets: List[torch.Tensor] = []
    edl_node_S_sum = 0.0
    edl_node_u_sum = 0.0
    edl_node_n = 0
    edl_node_S_correct_sum = 0.0
    edl_node_u_correct_sum = 0.0
    edl_node_n_correct = 0
    edl_node_S_wrong_sum = 0.0
    edl_node_u_wrong_sum = 0.0
    edl_node_n_wrong = 0

    edl_edge_label_probs: List[torch.Tensor] = []
    edl_edge_label_targets: List[torch.Tensor] = []
    edl_edge_label_S_sum = 0.0
    edl_edge_label_u_sum = 0.0
    edl_edge_label_n = 0
    edl_edge_label_S_correct_sum = 0.0
    edl_edge_label_u_correct_sum = 0.0
    edl_edge_label_n_correct = 0
    edl_edge_label_S_wrong_sum = 0.0
    edl_edge_label_u_wrong_sum = 0.0
    edl_edge_label_n_wrong = 0

    edl_edge_exist_S_sum = 0.0
    edl_edge_exist_u_sum = 0.0
    edl_edge_exist_n = 0
    edl_edge_exist_S_correct_sum = 0.0
    edl_edge_exist_u_correct_sum = 0.0
    edl_edge_exist_n_correct = 0
    edl_edge_exist_S_wrong_sum = 0.0
    edl_edge_exist_u_wrong_sum = 0.0
    edl_edge_exist_n_wrong = 0

    edl_occl_potential_probs: List[np.ndarray] = []
    edl_occl_potential_labels: List[np.ndarray] = []
    edl_occl_potential_S_sum = 0.0
    edl_occl_potential_u_sum = 0.0
    edl_occl_potential_n = 0
    edl_occl_potential_S_correct_sum = 0.0
    edl_occl_potential_u_correct_sum = 0.0
    edl_occl_potential_n_correct = 0
    edl_occl_potential_S_wrong_sum = 0.0
    edl_occl_potential_u_wrong_sum = 0.0
    edl_occl_potential_n_wrong = 0

    edl_occl_rank_S_sum = 0.0
    edl_occl_rank_u_sum = 0.0
    edl_occl_rank_n = 0
    
    # Store all preds/targets for Confusion Matrix
    all_node_local_preds = []
    all_node_global_preds = []
    all_node_targets = []
    
    # Special tracking for edge existence (binary)
    exist_stats = {"prec": [], "rec": [], "f1": [], "pos_rate": [], "recall_cand": []}
    all_exist_probs: List[np.ndarray] = []
    all_exist_labels: List[np.ndarray] = []

    # Occlusion stats (pointcloud z-buffer supervision).
    occl_dir_stats = {
        "prec": [],
        "rec": [],
        "f1": [],
        "brier_dom": [],
        "brier_union": [],
        "ece_dom": [],
        "pos_rate": [],
        "pred_pos_rate": [],
    }
    # Union regression error per bin (useful to see where the union head struggles).
    # Bins follow the same thresholds as stratified potential sampling:
    #   low:  union <= --occl_potential_union_tau_low
    #   mid:  (tau_low, --occl_union_tau)
    #   high: union >= --occl_union_tau
    occl_union_mae_low_sum = 0.0
    occl_union_mae_low_n = 0
    occl_union_mae_mid_sum = 0.0
    occl_union_mae_mid_n = 0
    occl_union_mae_high_sum = 0.0
    occl_union_mae_high_n = 0
    occl_leaf_stats = {
        "recall_at_1": [],
        "recall_at_3": [],
        "recall_at_5": [],
    }
    # GT-mass-weighted retrieval metrics using the k-buffer sidecar targets (top-3 "backup occluders").
    occl_leaf_mass_stats = {
        "mass_at_1": [],
        "mass_at_3": [],
        "mass_at_5": [],
    }
    # Ordering-aware graded ranking metric (k-buffer mass as relevance).
    occl_leaf_ndcg_stats = {
        "ndcg_at_1": [],
        "ndcg_at_3": [],
        "ndcg_at_5": [],
    }
    
    node_classes = ["Stem", "Leaf", "Ped", "Fruit"]
    edge_classes = ["Stem-Leaf", "Stem-Ped", "Ped-Fruit"]

    centroid_maes = []
    extent_maes = []
    cls_centroid_maes = {c: [] for c in node_classes}
    cls_extent_maes = {c: [] for c in node_classes}

    # For best-threshold sweeps of dominant-direction classification (binary derived task).
    all_occl_dir_probs: List[np.ndarray] = []
    all_occl_dir_labels: List[np.ndarray] = []

    # Union explainability (how much of GT union is attributable to leaves present in the sampled subgraph).
    occl_union_explain_sum = 0.0
    occl_union_explain_n = 0
    occl_union_explain_pos_sum = 0.0
    occl_union_explain_pos_n = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            if args.zero_inputs:
                batch.x_organ.zero_()
                batch.n_organ.zero_()
                # batch.json_geom.zero_() # Legacy, we handle it below for blind mode

            # [Zero-Cheat Stress Test]
            gt_geom_clone = None
            if getattr(args, 'no_geom_injection', False):
                gt_geom_clone = batch.json_geom.clone()
                batch.json_geom = torch.zeros_like(batch.json_geom) # Absolute Zero-Cheat
            
            out = model(batch)
            
            # Restore for metric calculation
            if gt_geom_clone is not None:
                batch.json_geom = gt_geom_clone
            
            # 1. Node Metrics
            n_mets_local = compute_metrics(out['local_node_logits'], batch.y_node, labels=list(range(len(node_classes))))
            if n_mets_local: node_local_metrics_list.append(n_mets_local)
            
            n_mets = compute_metrics(out['node_logits'], batch.y_node, labels=list(range(len(node_classes))))
            if n_mets: node_metrics_list.append(n_mets)

            # Store for Confusion Matrix
            all_node_local_preds.append(out['local_node_logits'].argmax(dim=-1).cpu().numpy())
            all_node_global_preds.append(out['node_logits'].argmax(dim=-1).cpu().numpy())
            all_node_targets.append(batch.y_node.cpu().numpy())

            # --- Generic node calibration (global head) ---
            # For EDL runs we append in the EDL block below to avoid duplicate work.
            if (not bool(getattr(args, "edl_node", False))) and epoch >= getattr(args, "stage_gnn_epoch", 0):
                try:
                    logits = out.get("node_logits", None)
                    if logits is not None and logits.numel() > 0:
                        probs = F.softmax(logits.detach(), dim=-1)
                        y = batch.y_node.to(device=probs.device).long()
                        node_calib_probs.append(probs.cpu())
                        node_calib_targets.append(y.detach().cpu())
                except Exception:
                    pass

            # --- EDL node diagnostics (global head) ---
            if bool(getattr(args, "edl_node", False)) and epoch >= getattr(args, "stage_gnn_epoch", 0):
                ev_fn = _get_edl_evidence_fn(args)
                if ev_fn is not None and out.get("node_logits", None) is not None and out["node_logits"].numel() > 0:
                    logits = out["node_logits"]
                    alpha = torch.clamp(ev_fn(logits), min=0.0) + 1.0  # [N,C]
                    S = alpha.sum(dim=-1)  # [N]
                    K = int(alpha.size(-1))
                    u = float(K) / (S + 1e-12)
                    probs = alpha / (S.unsqueeze(-1) + 1e-12)
                    y = batch.y_node.to(device=probs.device).long()

                    edl_node_S_sum += float(S.sum().item())
                    edl_node_u_sum += float(u.sum().item())
                    edl_node_n += int(S.numel())

                    pred = probs.argmax(dim=-1)
                    correct = (pred == y)
                    if bool(correct.any()):
                        edl_node_S_correct_sum += float(S[correct].sum().item())
                        edl_node_u_correct_sum += float(u[correct].sum().item())
                        edl_node_n_correct += int(correct.sum().item())
                    wrong = ~correct
                    if bool(wrong.any()):
                        edl_node_S_wrong_sum += float(S[wrong].sum().item())
                        edl_node_u_wrong_sum += float(u[wrong].sum().item())
                        edl_node_n_wrong += int(wrong.sum().item())

                    # For multiclass ECE/Brier (using EDL mean probs).
                    edl_node_probs.append(probs.detach().cpu())
                    edl_node_targets.append(y.detach().cpu())
                    node_calib_probs.append(probs.detach().cpu())
                    node_calib_targets.append(y.detach().cpu())
            
            # 2. Edge Existence
            edge_index = out['edge_index']
            y_exist, y_edge, num_gt_found = build_oracle_targets(edge_index, batch)

            if y_exist.numel() > 0:
                edge_logits = out['edge_exist_logits'].squeeze(-1)
                alpha_exist = None
                if bool(getattr(args, "edl_edge_exist", False)):
                    ev_fn = _get_edl_evidence_fn(args)
                    if ev_fn is None:
                        probs = torch.sigmoid(edge_logits)
                    else:
                        alpha_exist = _edl_tied_beta_alpha(edge_logits, ev_fn)  # [E,2]
                        probs = alpha_exist[..., 1] / (alpha_exist.sum(dim=-1) + 1e-12)
                else:
                    probs = torch.sigmoid(edge_logits)
                preds_val = (probs > 0.5).long().cpu().numpy()
                labels = y_exist.cpu().numpy()
                
                p, r, f, _ = precision_recall_fscore_support(labels, preds_val, average='binary', zero_division=0)
                exist_stats["prec"].append(p)
                exist_stats["rec"].append(r)
                exist_stats["f1"].append(f)
                
                # Candidate Stats
                total_gt = batch.num_gt_total.sum().item()
                caught_gt = num_gt_found
                num_cand = len(y_exist) / 2 # Undirected
                if total_gt > 0:
                    exist_stats["recall_cand"].append(caught_gt / total_gt)
                if num_cand > 0:
                    exist_stats["pos_rate"].append(caught_gt / num_cand)

                all_exist_probs.append(probs.detach().cpu().numpy().astype(np.float32, copy=False))
                all_exist_labels.append(labels.astype(np.int32, copy=False))

                # --- EDL edge existence diagnostics ---
                if alpha_exist is not None and epoch >= getattr(args, "stage_gnn_epoch", 0):
                    S = alpha_exist.sum(dim=-1)  # [E]
                    u = 2.0 / (S + 1e-12)
                    y = y_exist.to(device=S.device).long()
                    pred = (probs > 0.5).long()
                    correct = (pred == y)

                    edl_edge_exist_S_sum += float(S.sum().item())
                    edl_edge_exist_u_sum += float(u.sum().item())
                    edl_edge_exist_n += int(S.numel())
                    if bool(correct.any()):
                        edl_edge_exist_S_correct_sum += float(S[correct].sum().item())
                        edl_edge_exist_u_correct_sum += float(u[correct].sum().item())
                        edl_edge_exist_n_correct += int(correct.sum().item())
                    wrong = ~correct
                    if bool(wrong.any()):
                        edl_edge_exist_S_wrong_sum += float(S[wrong].sum().item())
                        edl_edge_exist_u_wrong_sum += float(u[wrong].sum().item())
                        edl_edge_exist_n_wrong += int(wrong.sum().item())
                
                # 3. Edge Labeling (only on positives)
                edge_mask = (y_edge != -1)
                if edge_mask.any():
                    e_mets = compute_metrics(out['edge_label_logits'], y_edge, mask=edge_mask, labels=list(range(len(edge_classes))))
                    if e_mets: edge_label_metrics_list.append(e_mets)

                    # --- Generic edge-label calibration (positives only) ---
                    # For EDL runs we append in the EDL block below to avoid duplicate work.
                    if (not bool(getattr(args, "edl_edge_label", False))) and epoch >= getattr(args, "stage_gnn_epoch", 0):
                        try:
                            logits = out.get("edge_label_logits", None)
                            if logits is not None and logits.numel() > 0:
                                probs_lbl = F.softmax(logits[edge_mask].detach(), dim=-1)
                                y_lbl = y_edge[edge_mask].to(device=probs_lbl.device).long()
                                edge_label_calib_probs.append(probs_lbl.cpu())
                                edge_label_calib_targets.append(y_lbl.detach().cpu())
                        except Exception:
                            pass

                    # --- EDL edge label diagnostics (positives only) ---
                    if bool(getattr(args, "edl_edge_label", False)) and epoch >= getattr(args, "stage_gnn_epoch", 0):
                        ev_fn = _get_edl_evidence_fn(args)
                        if ev_fn is not None:
                            logits = out["edge_label_logits"][edge_mask]
                            y = y_edge[edge_mask].to(device=logits.device).long()
                            alpha = torch.clamp(ev_fn(logits), min=0.0) + 1.0  # [Epos,C]
                            S = alpha.sum(dim=-1)
                            K = int(alpha.size(-1))
                            u = float(K) / (S + 1e-12)
                            probs_lbl = alpha / (S.unsqueeze(-1) + 1e-12)
                            pred = probs_lbl.argmax(dim=-1)
                            correct = (pred == y)

                            edl_edge_label_S_sum += float(S.sum().item())
                            edl_edge_label_u_sum += float(u.sum().item())
                            edl_edge_label_n += int(S.numel())
                            if bool(correct.any()):
                                edl_edge_label_S_correct_sum += float(S[correct].sum().item())
                                edl_edge_label_u_correct_sum += float(u[correct].sum().item())
                                edl_edge_label_n_correct += int(correct.sum().item())
                            wrong = ~correct
                            if bool(wrong.any()):
                                edl_edge_label_S_wrong_sum += float(S[wrong].sum().item())
                                edl_edge_label_u_wrong_sum += float(u[wrong].sum().item())
                                edl_edge_label_n_wrong += int(wrong.sum().item())

                            edl_edge_label_probs.append(probs_lbl.detach().cpu())
                            edl_edge_label_targets.append(y.detach().cpu())
                            edge_label_calib_probs.append(probs_lbl.detach().cpu())
                            edge_label_calib_targets.append(y.detach().cpu())
            
            # 4. Regression Metrics
            for c_idx, name in enumerate(node_classes):
                mask = (batch.y_node == c_idx)
                if mask.any():
                    if getattr(args, 'no_geom_injection', False):
                        target_off = batch.json_geom[mask, :3] - out['cluster_mean'][mask]
                        target_ext = batch.json_geom[mask, 3:]
                    else:
                        target_off = torch.zeros_like(out['centroid_offset'][mask])
                        target_ext = torch.zeros_like(out['extent_delta'][mask])

                    c_mae = torch.abs(out['centroid_offset'][mask] - target_off).mean().item()
                    e_mae = torch.abs(out['extent_delta'][mask] - target_ext).mean().item()
                    
                    cls_centroid_maes[name].append(c_mae)
                    cls_extent_maes[name].append(e_mae)
            
            if getattr(args, 'no_geom_injection', False):
                t_off = batch.json_geom[:, :3] - out['cluster_mean']
                t_ext = batch.json_geom[:, 3:]
            else:
                t_off = torch.zeros_like(out['centroid_offset'])
                t_ext = torch.zeros_like(out['extent_delta'])

            centroid_maes.append(torch.abs(out['centroid_offset'] - t_off).mean().item())
            extent_maes.append(torch.abs(out['extent_delta'] - t_ext).mean().item())

            # 5. Occlusion metrics (only after stage, if enabled).
            if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
                ds = compute_occl_dir_stats(
                    out.get("occl_fruit_union_logits", None),
                    getattr(batch, "occl_fruit_union", None),
                    union_tau=float(getattr(args, "occl_union_tau", 0.5)),
                )
                _append_stats(occl_dir_stats, ds)

                # Explainability: fraction of union explainable by present leaves (sum exclusive mass over present leaves).
                try:
                    y_union = getattr(batch, "occl_fruit_union", None)
                    y_excl = getattr(batch, "occl_pair_exclusive", None)
                    fruit_idx = getattr(batch, "occl_fruit_idx", None)
                    pair_index = getattr(batch, "occl_pair_index", None)
                    if (
                        y_union is not None and y_excl is not None and fruit_idx is not None and pair_index is not None
                        and y_union.numel() > 0 and y_excl.numel() > 0 and fruit_idx.numel() > 0 and pair_index.numel() > 0
                    ):
                        device_ = y_union.device
                        y_union_dev = y_union.to(device_)
                        D = int(y_union_dev.size(1))
                        # Map node indices -> row indices in y_union (focus fruits only).
                        row_of_node = torch.full((batch.y_node.size(0),), -1, device=device_, dtype=torch.long)
                        fruit_idx_dev = fruit_idx.to(device_)
                        row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device_, dtype=torch.long)
                        pair_fruit_row = row_of_node[pair_index[0].to(device_)]
                        valid = pair_fruit_row >= 0
                        if bool(valid.any()):
                            union_expl = torch.zeros((fruit_idx_dev.numel(), D), device=device_, dtype=torch.float32)
                            union_expl.index_add_(0, pair_fruit_row[valid], y_excl.to(device_)[valid].float())
                            union_expl = union_expl.clamp(min=0.0, max=1.0)
                            denom = torch.clamp(y_union_dev.float(), min=1e-6)
                            frac = torch.clamp(union_expl / denom, min=0.0, max=1.0)
                            w_union_explain = torch.where(
                                y_union_dev.float() > 1e-6, frac, torch.ones_like(frac)
                            )
                            occl_union_explain_sum += float(w_union_explain.sum().item())
                            occl_union_explain_n += int(w_union_explain.numel())
                            pos = (y_union_dev.float() > 1e-6)
                            if bool(pos.any()):
                                occl_union_explain_pos_sum += float(w_union_explain[pos].sum().item())
                                occl_union_explain_pos_n += int(pos.sum().item())
                except Exception:
                    pass

                # Collect flat dominant-direction probs/labels for best-F1 threshold sweep.
                oul = out.get("occl_fruit_union_logits", None)
                you = getattr(batch, "occl_fruit_union", None)
                if oul is not None and you is not None and oul.numel() > 0 and you.numel() > 0:
                    probs_dom = torch.sigmoid(oul.detach())
                    y_dom = (you.detach() >= float(getattr(args, "occl_union_tau", 0.5))).float()
                    all_occl_dir_probs.append(probs_dom.flatten().cpu().numpy().astype(np.float32, copy=False))
                    all_occl_dir_labels.append(y_dom.flatten().cpu().numpy().astype(np.int32, copy=False))

                    # Union MAE per bin (low/mid/high), based on GT union (not pred).
                    try:
                        union_tau = float(getattr(args, "occl_union_tau", 0.5))
                        tau_lo = float(getattr(args, "occl_potential_union_tau_low", 0.25))
                        gt = you.detach().to(device=device, dtype=probs_dom.dtype)
                        err = torch.abs(probs_dom - gt)
                        low = gt <= tau_lo
                        mid = (gt > tau_lo) & (gt < union_tau)
                        high = gt >= union_tau
                        if bool(low.any()):
                            occl_union_mae_low_sum += float(err[low].sum().item())
                            occl_union_mae_low_n += int(low.sum().item())
                        if bool(mid.any()):
                            occl_union_mae_mid_sum += float(err[mid].sum().item())
                            occl_union_mae_mid_n += int(mid.sum().item())
                        if bool(high.any()):
                            occl_union_mae_high_sum += float(err[high].sum().item())
                            occl_union_mae_high_n += int(high.sum().item())
                    except Exception:
                        pass

                    # Use evidential mean probability for leaf scoring when enabled; otherwise sigmoid(logit).
                    pair_logits = out.get("occl_pair_potential_logits", None)
                    pair_probs = None
                    if bool(getattr(args, "edl_occl_potential", False)) and pair_logits is not None and pair_logits.numel() > 0:
                        ev_fn = _get_edl_evidence_fn(args)
                        if ev_fn is not None:
                            alpha_p = _edl_tied_beta_alpha(pair_logits, ev_fn)  # [M,D,2]
                            pair_probs = alpha_p[..., 1] / (alpha_p.sum(dim=-1) + 1e-12)  # [M,D]

                    ls = compute_occl_leaf_retrieval_stats(
                        pair_logits,
                        getattr(batch, "occl_pair_potential", None),
                        pair_probs,
                        getattr(batch, "occl_pair_index", None),
                        getattr(batch, "occl_fruit_idx", None),
                        getattr(batch, "occl_fruit_union", None),
                        union_tau=None,
                        pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                        ks=(1, 3, 5),
                    )
                    _append_stats(occl_leaf_stats, ls)

                    # Mass@K using k-buffer supervision.
                    pk = getattr(batch, "occl_pair_k_rank", None)
                    if pk is not None and pk.numel() > 0:
                        mass = pk.sum(dim=-1)  # [M,D]
                        ms = compute_occl_leaf_mass_at_k_stats(
                            pair_logits,
                            mass,
                            pair_probs,
                            getattr(batch, "occl_pair_index", None),
                            getattr(batch, "occl_fruit_idx", None),
                            getattr(batch, "occl_fruit_union", None),
                            union_tau=None,
                            pot=getattr(batch, "occl_pair_potential", None),
                            pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                            ks=(1, 3, 5),
                        )
                        _append_stats(occl_leaf_mass_stats, ms)

                        nd = compute_occl_leaf_ndcg_at_k_stats(
                            pair_logits,
                            mass,
                            pair_probs,
                            getattr(batch, "occl_pair_index", None),
                            getattr(batch, "occl_fruit_idx", None),
                            getattr(batch, "occl_fruit_union", None),
                            union_tau=None,
                            pot=getattr(batch, "occl_pair_potential", None),
                            pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                            ks=(1, 3, 5),
                        )
                        _append_stats(occl_leaf_ndcg_stats, nd)

                    # --- EDL occlusion diagnostics ---
                    # (1) Potential calibration/uncertainty (binary: potential>=eps) on the same (fruit,dir)
                    # selection mask used for supervision (dominant + stratified mid/low by default).
                    if bool(getattr(args, "edl_occl_potential", False)):
                        try:
                            ev_fn = _get_edl_evidence_fn(args)
                            y_pot = getattr(batch, "occl_pair_potential", None)
                            pair_index = getattr(batch, "occl_pair_index", None)
                            fruit_idx = getattr(batch, "occl_fruit_idx", None)
                            y_union = getattr(batch, "occl_fruit_union", None)
                            if (
                                ev_fn is not None and pair_logits is not None and y_pot is not None and pair_index is not None and fruit_idx is not None and y_union is not None
                                and pair_logits.numel() > 0 and y_pot.numel() > 0 and pair_index.numel() > 0 and fruit_idx.numel() > 0 and y_union.numel() > 0
                            ):
                                union_tau = float(getattr(args, "occl_union_tau", 0.5))
                                pot_eps = float(getattr(args, "occl_potential_eps", 0.05))
                                pot_dir_policy = str(getattr(args, "occl_potential_dir_sampling", "stratified")).strip().lower()
                                union = y_union.to(device=device)
                                Ff, D = int(union.size(0)), int(union.size(1))

                                # Deterministic stratified direction selection for stable val metrics.
                                if pot_dir_policy == "stratified":
                                    tau_lo = float(getattr(args, "occl_potential_union_tau_low", 0.25))
                                    n_mid = int(getattr(args, "occl_potential_n_mid_dirs", 3))
                                    n_low = int(getattr(args, "occl_potential_n_low_dirs", 1))
                                    sel_dirs = torch.zeros((Ff, D), dtype=torch.bool, device=device)
                                    high = (union >= union_tau)
                                    mid = (union > tau_lo) & (union < union_tau)
                                    low = (union <= tau_lo)
                                    sel_dirs |= high
                                    for r in range(Ff):
                                        if n_mid > 0:
                                            mid_idx = torch.nonzero(mid[r], as_tuple=False).flatten()
                                            if int(mid_idx.numel()) > 0:
                                                sel_dirs[r, mid_idx[: min(n_mid, int(mid_idx.numel()))]] = True
                                        if n_low > 0:
                                            low_idx = torch.nonzero(low[r], as_tuple=False).flatten()
                                            if int(low_idx.numel()) > 0:
                                                sel_dirs[r, low_idx[: min(n_low, int(low_idx.numel()))]] = True
                                else:
                                    sel_dirs = (union >= union_tau)

                                # Map (fruit_node_idx, leaf_node_idx) pairs -> fruit rows.
                                fruit_idx_dev = fruit_idx.to(device)
                                row_of_node = torch.full((batch.y_node.size(0),), -1, device=device, dtype=torch.long)
                                row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device, dtype=torch.long)
                                pair_fruit_row = row_of_node[pair_index[0].to(device)]  # [M]
                                valid = pair_fruit_row >= 0
                                if bool(valid.any()):
                                    pair_fruit_row_safe = pair_fruit_row.clamp(min=0)
                                    sel_mask = sel_dirs[pair_fruit_row_safe]
                                    if not bool(valid.all()):
                                        sel_mask = sel_mask & valid.unsqueeze(1)

                                    if bool(sel_mask.any()):
                                        logits_flat = pair_logits.to(device)[sel_mask]
                                        y_flat = y_pot.to(device)[sel_mask]
                                        y_bin = (y_flat >= pot_eps).long()
                                        alpha = _edl_tied_beta_alpha(logits_flat, ev_fn)  # [K,2]
                                        S = alpha.sum(dim=-1)
                                        u = 2.0 / (S + 1e-12)
                                        p = alpha[..., 1] / (S + 1e-12)

                                        pred = (p > 0.5).long()
                                        correct = (pred == y_bin)

                                        edl_occl_potential_S_sum += float(S.sum().item())
                                        edl_occl_potential_u_sum += float(u.sum().item())
                                        edl_occl_potential_n += int(S.numel())
                                        if bool(correct.any()):
                                            edl_occl_potential_S_correct_sum += float(S[correct].sum().item())
                                            edl_occl_potential_u_correct_sum += float(u[correct].sum().item())
                                            edl_occl_potential_n_correct += int(correct.sum().item())
                                        wrong = ~correct
                                        if bool(wrong.any()):
                                            edl_occl_potential_S_wrong_sum += float(S[wrong].sum().item())
                                            edl_occl_potential_u_wrong_sum += float(u[wrong].sum().item())
                                            edl_occl_potential_n_wrong += int(wrong.sum().item())

                                        edl_occl_potential_probs.append(p.detach().cpu().numpy().astype(np.float32, copy=False))
                                        edl_occl_potential_labels.append(y_bin.detach().cpu().numpy().astype(np.int32, copy=False))
                        except Exception:
                            pass

                    # (2) Rank distribution uncertainty (Dirichlet over leaves) on dominant directions.
                    if bool(getattr(args, "edl_occl_rank", False)):
                        try:
                            ev_fn = _get_edl_evidence_fn(args)
                            # Use the dedicated rank logits when available; fall back to potential logits for
                            # backwards compatibility with older checkpoints.
                            pair_rank_logits = out.get("occl_pair_rank_logits", pair_logits)
                            y_pot = getattr(batch, "occl_pair_potential", None)
                            pair_index = getattr(batch, "occl_pair_index", None)
                            fruit_idx = getattr(batch, "occl_fruit_idx", None)
                            y_union = getattr(batch, "occl_fruit_union", None)
                            if (
                                ev_fn is not None and pair_rank_logits is not None and y_pot is not None and pair_index is not None and fruit_idx is not None and y_union is not None
                                and pair_rank_logits.numel() > 0 and y_pot.numel() > 0 and pair_index.numel() > 0 and fruit_idx.numel() > 0 and y_union.numel() > 0
                            ):
                                union_tau = float(getattr(args, "occl_union_tau", 0.5))
                                pot_eps = float(getattr(args, "occl_potential_eps", 0.05))
                                union = y_union.to(device=device)
                                pk = getattr(batch, "occl_pair_k_rank", None)
                                if pk is not None and pk.numel() > 0:
                                    mass = pk.to(device=device).sum(dim=-1)  # [M,D]
                                else:
                                    mass = y_pot.to(device=device)
                                pot = y_pot.to(device=device)
                                mass = torch.where(pot >= pot_eps, mass, torch.zeros_like(mass))
                                gamma = float(getattr(args, "occl_rank_gamma", 1.0))
                                if gamma != 1.0:
                                    mass = torch.clamp(mass, min=0.0).pow(gamma)

                                fruit_idx_dev = fruit_idx.to(device)
                                row_of_node = torch.full((batch.y_node.size(0),), -1, device=device, dtype=torch.long)
                                row_of_node[fruit_idx_dev] = torch.arange(fruit_idx_dev.numel(), device=device, dtype=torch.long)
                                pair_fruit_row = row_of_node[pair_index[0].to(device)]  # [M]
                                valid = pair_fruit_row >= 0
                                if bool(valid.any()):
                                    pair_fruit_row = pair_fruit_row[valid]
                                    logits_v = pair_rank_logits.to(device)[valid]
                                    mass_v = mass[valid]
                                    for r in range(int(fruit_idx_dev.numel())):
                                        m = (pair_fruit_row == r)
                                        if not bool(m.any()):
                                            continue
                                        logits_r = logits_v[m]  # [L,D]
                                        mass_r = mass_v[m]     # [L,D]
                                        sum_mass = mass_r.sum(dim=0)  # [D]
                                        use = (union[r] >= union_tau) & (sum_mass > 0)
                                        if not bool(use.any()):
                                            continue
                                        alpha = torch.clamp(ev_fn(logits_r), min=0.0) + 1.0  # [L,D]
                                        S_dir = alpha.sum(dim=0)  # [D]
                                        L = float(alpha.size(0))
                                        u_dir = L / (S_dir + 1e-12)
                                        edl_occl_rank_S_sum += float(S_dir[use].sum().item())
                                        edl_occl_rank_u_sum += float(u_dir[use].sum().item())
                                        edl_occl_rank_n += int(use.sum().item())
                        except Exception:
                            pass
            
    # Aggregation
    log_dict = {"epoch": epoch}

    # Node Local
    if node_local_metrics_list:
        node_local_agg = aggregate_per_class_and_return(node_local_metrics_list, node_classes, "val/node_local")
        log_dict.update(node_local_agg)
        
    # Node Global
    if node_metrics_list:
        node_global_agg = aggregate_per_class_and_return(node_metrics_list, node_classes, "val/node")
        log_dict.update(node_global_agg)
    
    # Edge Label
    if edge_label_metrics_list:
        edge_label_agg = aggregate_per_class_and_return(edge_label_metrics_list, edge_classes, "val/edge_label")
        log_dict.update(edge_label_agg)
    
    # Edge Existence
    for k, v in exist_stats.items():
        if v: log_dict[f"val/edge_exist_{k}"] = np.mean(v)

    # Best-threshold sweep + calibration for edge existence (global over the whole val set).
    if all_exist_probs and all_exist_labels:
        probs_all = np.concatenate(all_exist_probs, axis=0)
        y_all = np.concatenate(all_exist_labels, axis=0)
        best = sweep_best_f1_binary(probs_all, y_all)
        log_dict["val/edge_exist_best_f1"] = best.get("best_f1", 0.0)
        log_dict["val/edge_exist_best_prec"] = best.get("best_prec", 0.0)
        log_dict["val/edge_exist_best_rec"] = best.get("best_rec", 0.0)
        log_dict["val/edge_exist_best_thresh"] = best.get("best_thresh", 0.5)

        log_dict["val/edge_exist_brier"] = float(np.mean((probs_all - y_all.astype(np.float32)) ** 2))
        log_dict["val/edge_exist_ece"] = binary_ece(torch.from_numpy(probs_all), torch.from_numpy(y_all), num_bins=15)

    # Generic multiclass calibration (A0 vs A1 comparable).
    if epoch >= getattr(args, "stage_gnn_epoch", 0):
        if node_calib_probs and node_calib_targets:
            probs = torch.cat(node_calib_probs, dim=0)
            t = torch.cat(node_calib_targets, dim=0)
            log_dict["val/node_ece"] = multiclass_ece(probs, t, num_bins=15)
            log_dict["val/node_brier"] = multiclass_brier(probs, t)
        if edge_label_calib_probs and edge_label_calib_targets:
            probs = torch.cat(edge_label_calib_probs, dim=0)
            t = torch.cat(edge_label_calib_targets, dim=0)
            log_dict["val/edge_label_ece"] = multiclass_ece(probs, t, num_bins=15)
            log_dict["val/edge_label_brier"] = multiclass_brier(probs, t)

    # --- EDL (evidential) diagnostics ---
    # Only log these when the corresponding EDL mode is enabled, to avoid metric bloat.
    if bool(getattr(args, "edl_node", False)) and epoch >= getattr(args, "stage_gnn_epoch", 0) and edl_node_n > 0:
        log_dict["val/edl_node_S_mean"] = float(edl_node_S_sum / max(1, edl_node_n))
        log_dict["val/edl_node_u_mean"] = float(edl_node_u_sum / max(1, edl_node_n))
        if edl_node_n_correct > 0:
            log_dict["val/edl_node_S_correct"] = float(edl_node_S_correct_sum / max(1, edl_node_n_correct))
            log_dict["val/edl_node_u_correct"] = float(edl_node_u_correct_sum / max(1, edl_node_n_correct))
        if edl_node_n_wrong > 0:
            log_dict["val/edl_node_S_wrong"] = float(edl_node_S_wrong_sum / max(1, edl_node_n_wrong))
            log_dict["val/edl_node_u_wrong"] = float(edl_node_u_wrong_sum / max(1, edl_node_n_wrong))
        if edl_node_probs and edl_node_targets:
            probs = torch.cat(edl_node_probs, dim=0)
            t = torch.cat(edl_node_targets, dim=0)
            log_dict["val/edl_node_ece"] = multiclass_ece(probs, t, num_bins=15)
            log_dict["val/edl_node_brier"] = multiclass_brier(probs, t)

    if bool(getattr(args, "edl_edge_label", False)) and epoch >= getattr(args, "stage_gnn_epoch", 0) and edl_edge_label_n > 0:
        log_dict["val/edl_edge_label_S_mean"] = float(edl_edge_label_S_sum / max(1, edl_edge_label_n))
        log_dict["val/edl_edge_label_u_mean"] = float(edl_edge_label_u_sum / max(1, edl_edge_label_n))
        if edl_edge_label_n_correct > 0:
            log_dict["val/edl_edge_label_S_correct"] = float(edl_edge_label_S_correct_sum / max(1, edl_edge_label_n_correct))
            log_dict["val/edl_edge_label_u_correct"] = float(edl_edge_label_u_correct_sum / max(1, edl_edge_label_n_correct))
        if edl_edge_label_n_wrong > 0:
            log_dict["val/edl_edge_label_S_wrong"] = float(edl_edge_label_S_wrong_sum / max(1, edl_edge_label_n_wrong))
            log_dict["val/edl_edge_label_u_wrong"] = float(edl_edge_label_u_wrong_sum / max(1, edl_edge_label_n_wrong))
        if edl_edge_label_probs and edl_edge_label_targets:
            probs = torch.cat(edl_edge_label_probs, dim=0)
            t = torch.cat(edl_edge_label_targets, dim=0)
            log_dict["val/edl_edge_label_ece"] = multiclass_ece(probs, t, num_bins=15)
            log_dict["val/edl_edge_label_brier"] = multiclass_brier(probs, t)

    if bool(getattr(args, "edl_edge_exist", False)) and epoch >= getattr(args, "stage_gnn_epoch", 0) and edl_edge_exist_n > 0:
        log_dict["val/edl_edge_exist_S_mean"] = float(edl_edge_exist_S_sum / max(1, edl_edge_exist_n))
        log_dict["val/edl_edge_exist_u_mean"] = float(edl_edge_exist_u_sum / max(1, edl_edge_exist_n))
        if edl_edge_exist_n_correct > 0:
            log_dict["val/edl_edge_exist_S_correct"] = float(edl_edge_exist_S_correct_sum / max(1, edl_edge_exist_n_correct))
            log_dict["val/edl_edge_exist_u_correct"] = float(edl_edge_exist_u_correct_sum / max(1, edl_edge_exist_n_correct))
        if edl_edge_exist_n_wrong > 0:
            log_dict["val/edl_edge_exist_S_wrong"] = float(edl_edge_exist_S_wrong_sum / max(1, edl_edge_exist_n_wrong))
            log_dict["val/edl_edge_exist_u_wrong"] = float(edl_edge_exist_u_wrong_sum / max(1, edl_edge_exist_n_wrong))

    # Occlusion stats
    if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
        for k, v in occl_dir_stats.items():
            if v:
                log_dict[f"val/occl_dir_{k}"] = float(np.mean(v))
        for k, v in occl_leaf_stats.items():
            if v:
                log_dict[f"val/occl_leaf_{k}"] = float(np.mean(v))
        for k, v in occl_leaf_mass_stats.items():
            if v:
                log_dict[f"val/occl_leaf_{k}"] = float(np.mean(v))
        for k, v in occl_leaf_ndcg_stats.items():
            if v:
                log_dict[f"val/occl_leaf_{k}"] = float(np.mean(v))

        # Union MAE per bin (low/mid/high).
        if occl_union_mae_low_n > 0:
            log_dict["val/occl_union_mae_low"] = float(occl_union_mae_low_sum / max(1, occl_union_mae_low_n))
        if occl_union_mae_mid_n > 0:
            log_dict["val/occl_union_mae_mid"] = float(occl_union_mae_mid_sum / max(1, occl_union_mae_mid_n))
        if occl_union_mae_high_n > 0:
            log_dict["val/occl_union_mae_high"] = float(occl_union_mae_high_sum / max(1, occl_union_mae_high_n))

        if all_occl_dir_probs and all_occl_dir_labels:
            probs_all = np.concatenate(all_occl_dir_probs, axis=0)
            y_all = np.concatenate(all_occl_dir_labels, axis=0)
            best = sweep_best_f1_binary(probs_all, y_all)
            log_dict["val/occl_dir_best_f1"] = best.get("best_f1", 0.0)
            log_dict["val/occl_dir_best_prec"] = best.get("best_prec", 0.0)
            log_dict["val/occl_dir_best_rec"] = best.get("best_rec", 0.0)
            log_dict["val/occl_dir_best_thresh"] = best.get("best_thresh", 0.5)

        if occl_union_explain_n > 0:
            log_dict["val/occl_union_explain_w_mean"] = float(occl_union_explain_sum / max(1, occl_union_explain_n))
        if occl_union_explain_pos_n > 0:
            log_dict["val/occl_union_explain_w_pos_mean"] = float(occl_union_explain_pos_sum / max(1, occl_union_explain_pos_n))

        # EDL occlusion metrics (optional; only when enabled).
        if bool(getattr(args, "edl_occl_potential", False)) and edl_occl_potential_n > 0:
            log_dict["val/edl_occl_potential_S_mean"] = float(edl_occl_potential_S_sum / max(1, edl_occl_potential_n))
            log_dict["val/edl_occl_potential_u_mean"] = float(edl_occl_potential_u_sum / max(1, edl_occl_potential_n))
            if edl_occl_potential_n_correct > 0:
                log_dict["val/edl_occl_potential_S_correct"] = float(edl_occl_potential_S_correct_sum / max(1, edl_occl_potential_n_correct))
                log_dict["val/edl_occl_potential_u_correct"] = float(edl_occl_potential_u_correct_sum / max(1, edl_occl_potential_n_correct))
            if edl_occl_potential_n_wrong > 0:
                log_dict["val/edl_occl_potential_S_wrong"] = float(edl_occl_potential_S_wrong_sum / max(1, edl_occl_potential_n_wrong))
                log_dict["val/edl_occl_potential_u_wrong"] = float(edl_occl_potential_u_wrong_sum / max(1, edl_occl_potential_n_wrong))
            if edl_occl_potential_probs and edl_occl_potential_labels:
                p_all = np.concatenate(edl_occl_potential_probs, axis=0)
                y_all = np.concatenate(edl_occl_potential_labels, axis=0)
                log_dict["val/edl_occl_potential_brier"] = float(np.mean((p_all - y_all.astype(np.float32)) ** 2))
                log_dict["val/edl_occl_potential_ece"] = binary_ece(torch.from_numpy(p_all), torch.from_numpy(y_all), num_bins=15)

        if bool(getattr(args, "edl_occl_rank", False)) and edl_occl_rank_n > 0:
            log_dict["val/edl_occl_rank_S_mean"] = float(edl_occl_rank_S_sum / max(1, edl_occl_rank_n))
            log_dict["val/edl_occl_rank_u_mean"] = float(edl_occl_rank_u_sum / max(1, edl_occl_rank_n))

    log_dict["val/centroid_mae"] = np.mean(centroid_maes) if centroid_maes else 0.0
    log_dict["val/extent_mae"] = np.mean(extent_maes) if extent_maes else 0.0
    
    # Class-wise Regression MAE
    for name in node_classes:
        if name in cls_centroid_maes and cls_centroid_maes[name]:
            c_mae = np.mean(cls_centroid_maes[name])
            e_mae = np.mean(cls_extent_maes[name])
            log_dict[f"val/centroid_mae_{name}"] = c_mae
            log_dict[f"val/extent_mae_{name}"] = e_mae
    
    w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
    w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0

    print(f"\nValidation Epoch {epoch}:")
    # 1. Node Local (Always)
    print(f"  Node Local: Acc={log_dict.get('val/node_local_acc', 0):.4f}, "
          f"P(Macro)={log_dict.get('val/node_local_prec_macro', 0):.4f}, "
          f"R(Macro)={log_dict.get('val/node_local_rec_macro', 0):.4f}")
    if node_local_metrics_list:
        # Per-class detailed breakdown
        for i, name in enumerate(node_classes):
            p = log_dict.get(f"val/node_local_{name}_precision", 0)
            r = log_dict.get(f"val/node_local_{name}_recall", 0)
            f = log_dict.get(f"val/node_local_{name}_f1", 0)
            print(f"        {name:7}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")
        
        # Confusion Matrix
        cm_local = confusion_matrix(np.concatenate(all_node_targets), np.concatenate(all_node_local_preds), labels=range(len(node_classes)))
        print_text_confusion_matrix(cm_local, node_classes, "Node Local Confusion Matrix (Val)")
        
    # 2. GNN (Gated)
    if w_gnn > 0:
        print(f"\n  Node Global: Acc={log_dict.get('val/node_acc', 0):.4f}, "
              f"P(Macro)={log_dict.get('val/node_prec_macro', 0):.4f}, "
              f"R(Macro)={log_dict.get('val/node_rec_macro', 0):.4f}")
        if node_metrics_list:
            for i, name in enumerate(node_classes):
                p = log_dict.get(f"val/node_{name}_precision", 0)
                r = log_dict.get(f"val/node_{name}_recall", 0)
                f = log_dict.get(f"val/node_{name}_f1", 0)
                print(f"        {name:7}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")
            
            # Confusion Matrix
            cm_glob = confusion_matrix(np.concatenate(all_node_targets), np.concatenate(all_node_global_preds), labels=range(len(node_classes)))
            print_text_confusion_matrix(cm_glob, node_classes, "Node Global Confusion Matrix (Val)")
            
        print(f"\n  Edge Exist: F1={log_dict.get('val/edge_exist_f1', 0):.4f}, "
              f"P={log_dict.get('val/edge_exist_prec', 0):.4f}, "
              f"R={log_dict.get('val/edge_exist_rec', 0):.4f}")
        print(f"              CandRecall={log_dict.get('val/edge_exist_recall_cand', 0):.4f}, "
              f"PosRate={log_dict.get('val/edge_exist_pos_rate', 0):.4f}")
        if "val/node_ece" in log_dict and "val/edge_label_ece" in log_dict:
            node_u = f", u={log_dict.get('val/edl_node_u_mean'):.4f}" if "val/edl_node_u_mean" in log_dict else ""
            ee_u = f", u={log_dict.get('val/edl_edge_exist_u_mean'):.4f}" if "val/edl_edge_exist_u_mean" in log_dict else ""
            el_u = f", u={log_dict.get('val/edl_edge_label_u_mean'):.4f}" if "val/edl_edge_label_u_mean" in log_dict else ""
            or_u = f", u={log_dict.get('val/edl_occl_rank_u_mean'):.4f}" if "val/edl_occl_rank_u_mean" in log_dict else ""
            print(
                f"  Calib: node ECE/Brier={log_dict.get('val/node_ece', 0):.4f}/{log_dict.get('val/node_brier', 0):.4f}{node_u} | "
                f"edge-exist ECE/Brier={log_dict.get('val/edge_exist_ece', 0):.4f}/{log_dict.get('val/edge_exist_brier', 0):.4f}{ee_u} | "
                f"edge-label ECE/Brier={log_dict.get('val/edge_label_ece', 0):.4f}/{log_dict.get('val/edge_label_brier', 0):.4f}{el_u} | "
                f"occl-rank{or_u}",
                flush=True,
            )
        
        print(f"\n  Edge Label: Acc={log_dict.get('val/edge_label_acc', 0):.4f}, "
              f"P(Macro)={log_dict.get('val/edge_label_prec_macro', 0):.4f}, "
              f"R(Macro)={log_dict.get('val/edge_label_rec_macro', 0):.4f}")
        if edge_label_metrics_list:
            for i, name in enumerate(edge_classes):
                p = log_dict.get(f"val/edge_label_{name}_precision", 0)
                r = log_dict.get(f"val/edge_label_{name}_recall", 0)
                f = log_dict.get(f"val/edge_label_{name}_f1", 0)
                print(f"        {name:9}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")
        
    if w_reg_curric > 0:
         print(f"\n  MAE: Centroid={log_dict.get('val/centroid_mae', 0):.4f}, "
               f"Extent={log_dict.get('val/extent_mae', 0):.4f}")
         for name in node_classes:
             c_mae = log_dict.get(f"val/centroid_mae_{name}", 0)
             e_mae = log_dict.get(f"val/extent_mae_{name}", 0)
             if c_mae > 0:
                print(f"        {name:7}: Centroid={c_mae:.4f}, Extent={e_mae:.4f}")

    if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
        print(
            f"\n  Occl Dir: F1={log_dict.get('val/occl_dir_f1', 0):.4f}, "
            f"P={log_dict.get('val/occl_dir_prec', 0):.4f}, "
            f"R={log_dict.get('val/occl_dir_rec', 0):.4f}, "
            f"ECE={log_dict.get('val/occl_dir_ece_dom', 0):.4f}, "
            f"Brier(dom)={log_dict.get('val/occl_dir_brier_dom', 0):.4f}",
            flush=True,
        )
        if "val/occl_union_explain_w_pos_mean" in log_dict:
            print(
                f"  Occl Expl: w_mean={log_dict.get('val/occl_union_explain_w_mean', 0):.3f}, "
                f"w_pos_mean={log_dict.get('val/occl_union_explain_w_pos_mean', 0):.3f}",
                flush=True,
            )
        print(
            f"  Occl Leaf: Recall@1/3/5={log_dict.get('val/occl_leaf_recall_at_1', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_recall_at_3', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_recall_at_5', 0):.3f}",
            flush=True,
        )
        print(
            f"            Mass@1/3/5={log_dict.get('val/occl_leaf_mass_at_1', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_mass_at_3', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_mass_at_5', 0):.3f}",
            flush=True,
        )
        print(
            f"            NDCG@1/3/5={log_dict.get('val/occl_leaf_ndcg_at_1', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_ndcg_at_3', 0):.3f}/"
            f"{log_dict.get('val/occl_leaf_ndcg_at_5', 0):.3f}",
            flush=True,
        )

    if wandb.run:

        # Confusion Matrices
        if all_node_targets:
            y_true = np.concatenate(all_node_targets)
            y_local = np.concatenate(all_node_local_preds)
            y_global = np.concatenate(all_node_global_preds)
            
            log_dict["val/conf_mat_local"] = wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_local, class_names=node_classes
            )
            log_dict["val/conf_mat_global"] = wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_global, class_names=node_classes
            )

        wandb.log(log_dict, step=epoch)
    return log_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "--name", type=str, default="oracle_gnn_v12")
    parser.add_argument("--npoint", type=int, default=256, help="Number of points per instance")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for validation (full scenes)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pos_weight", type=float, default=10.0, help="Weight for positive edges in existence loss")
    parser.add_argument(
        "--pos_weight_edge_exist",
        type=float,
        default=None,
        help="Optional override for edge existence pos_weight only (default: use --pos_weight)",
    )
    # Evidential Deep Learning (EDL) toggles (all OFF by default; enable per-head explicitly).
    # These apply to heads *after* the GNN (scene-graph + occlusion), and only change the loss/probability mapping.
    parser.add_argument("--edl_node", action="store_true", help="Use evidential (Dirichlet) loss for global node head (default: off)")
    parser.add_argument("--edl_edge_exist", action="store_true", help="Use evidential (tied-Beta/Dirichlet) loss for edge existence (default: off)")
    parser.add_argument("--edl_edge_label", action="store_true", help="Use evidential (Dirichlet) loss for edge label head (default: off)")
    parser.add_argument("--edl_occl_potential", action="store_true", help="Use evidential (tied-Beta/Dirichlet) loss for occlusion potential head (default: off)")
    parser.add_argument("--edl_occl_rank", action="store_true", help="Use evidential (Dirichlet) listwise loss for occlusion rank distribution (default: off)")
    parser.add_argument(
        "--edl_fn",
        type=str,
        default="softplus",
        choices=["relu", "exp", "softplus"],
        help="Evidence activation for EDL heads",
    )
    parser.add_argument("--edl_kl_coeff", type=float, default=0.1, help="EDL KL regularization coefficient")
    parser.add_argument(
        "--edl_kl_coeff_edge_exist",
        type=float,
        default=None,
        help="Optional override for edge existence EDL KL coefficient only (default: use --edl_kl_coeff)",
    )
    parser.add_argument("--edl_kl_warmup_epochs", type=int, default=20, help="EDL KL warmup epochs (0 disables warmup)")
    parser.add_argument("--edl_kl_warmup_start", type=float, default=0.0, help="EDL KL warmup starting coefficient")
    parser.add_argument(
        "--edl_kl_coeff_rank",
        type=float,
        default=0.01,
        help="EDL KL coefficient for occlusion rank Dirichlet (separate due to variable leafset size)",
    )
    parser.add_argument("--w_node", type=float, default=1.0)
    parser.add_argument("--w_edge_exist", type=float, default=0.1)
    parser.add_argument("--w_edge_label", type=float, default=0.5)
    parser.add_argument("--w_reg", type=float, default=1.0)
    parser.add_argument("--data_root", type=str, default="/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors")
    parser.add_argument("--drop_shoots", action="store_true", help="Drop shoot points instead of merging them")
    parser.add_argument("--k_candidates", type=int, default=8, help="Number of kNN candidates per node")
    parser.add_argument("--extra_stems_per_node", type=int, default=2, help="Number of extra stem candidates per non-stem node")
    parser.add_argument("--dist_threshold", type=float, default=0.15, help="Distance threshold for radius-based candidate edges")
    parser.add_argument("--use_rgb", action="store_true", default=True, help="Use RGB features in encoder")
    parser.add_argument("--no_rgb", action="store_false", dest="use_rgb", help="Disable RGB features")
    parser.add_argument("--use_normals", action="store_true", default=True, help="Use normals in encoder")
    parser.add_argument("--no_normals", action="store_false", dest="use_normals", help="Disable normals in encoder")
    parser.add_argument("--no_geom_injection", action="store_true", help="Do not inject GT centroids/extents into GNN")
    # Train-time augmentations.
    # - Point-level augs are enabled by non-zero rates (set to 0 to disable).
    # - Z-rotation is enabled by default via `--z_rot_aug_mode` (set to `none` to disable).
    parser.add_argument("--color_dropout", type=float, default=0.25, help="Color dropout rate (set 0 to disable)")
    parser.add_argument("--xyz_jitter", type=float, default=0.001, help="Point coordinate jitter std in meters (set 0 to disable)")
    
    # Random drop before sampling
    parser.add_argument("--point_dropout", type=float, default=0.1, help="Point dropout rate (set 0 to disable)")
    
    # Targetted Post-sampling drop (drops and duplicates to maintain exact points per cluster)
    parser.add_argument("--pd_stem", type=float, default=0.0, help="Post-sampling dropout rate for stem (0)")
    parser.add_argument("--pd_leaf", type=float, default=0.0, help="Post-sampling dropout rate for leaves (1)")
    parser.add_argument("--pd_fruit", type=float, default=0.0, help="Post-sampling dropout rate for peppers/fruits (2)")
    parser.add_argument("--pd_peduncle", type=float, default=0.0, help="Post-sampling dropout rate for peduncles (3)")
    parser.add_argument(
        "--z_rot_aug_mode",
        type=str,
        default="continuous",
        choices=["none", "continuous", "discrete90"],
        help="Z-rotation augmentation mode (set to 'none' to disable rotation)",
    )
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval between full validations")
    # W&B control. Default is online unless explicitly disabled.
    parser.add_argument("--offline", action="store_true", help="Run W&B in offline mode (legacy; use --wandb_mode offline)")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode override. If omitted, uses --offline to decide between online/offline.",
    )
    parser.add_argument("--wandb_project", type=str, default="pepper_oracle", help="W&B project name")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B group (for ablation collections)")
    parser.add_argument(
        "--wandb_tag",
        action="append",
        default=None,
        help="Add a W&B tag (repeatable). Example: --wandb_tag L1 --wandb_tag loss_ablation",
    )
    parser.add_argument("--zero_inputs", action="store_true", help="Zero out all inputs for sanity check")
    parser.add_argument("--shuffle_labels", action="store_true", help="Shuffle labels for sanity check")
    parser.add_argument("--val_only", action="store_true", help="Run only validation and exit")
    parser.add_argument(
        "--runs_root",
        type=str,
        default="training/oracle",
        help="Root directory where oracle run folders (configs/checkpoints) are saved",
    )
    parser.add_argument(
        "--ckpt_monitor",
        type=str,
        default="val/occl_dir_best_f1",
        help="Validation metric key to monitor for `best_model.pt`",
    )
    parser.add_argument(
        "--ckpt_mode",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Whether lower or higher is better for --ckpt_monitor",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume from (loads model+optimizer+scheduler and continues)",
    )
    parser.add_argument("--ablation_key", type=str, default=None, help="Optional short ablation key (A0, L1, ...)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed (for reproducibility). Omit to keep current behavior.")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="If >0, cap the number of training batches per epoch (useful for smoke tests).",
    )
    parser.add_argument('--dynamic_edge_weight', action='store_true', help='Enable dynamic pos_weight for edge existence')
    parser.add_argument('--subgraph_enabled', action='store_true', help='Enable v12 Subgraph Sampling (Train only)')
    parser.add_argument('--subgraph_size', type=int, default=64, help='Maximum nodes to sample per subgraph')
    parser.add_argument('--subgraph_size_min', type=int, default=None, help='Minimum nodes for variable-size sampling')
    # Occlusion-augmented sampling: keep stem-biased BFS, then inject fruit-centric leaf neighborhoods.
    parser.add_argument('--occl_aug_enabled', action='store_true', help='Augment BFS subgraphs with occlusion-relevant leaves around a few focus fruits')
    parser.add_argument(
        '--occl_sidecar_dir',
        type=str,
        default=None,
        help=(
            "Optional directory containing occlusion sidecars. "
            "For z-buffer: scene_XXXXX_occl_zbuf_k3_*.npz. "
            "For camera ray: scene_XXXXX_occl_cam_*.npz. "
            "(auto-detect only supported for z-buffer; pass explicitly for camera eval)"
        ),
    )
    parser.add_argument(
        "--occl_gt_source",
        type=str,
        default="zbuf",
        choices=["zbuf", "cam_ortho", "cam_persp"],
        help="Occlusion GT source for occl supervision/metrics: zbuf (default), cam_ortho, cam_persp.",
    )
    parser.add_argument(
        "--occl_cam_jitter_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "Camera GT only: which direction set to evaluate from cam sidecars. "
            "0=base (canonical), 1=small jitter, 2=large jitter. "
            "We still map jittered dirs back to the 18 canonical indices via dir_base_index."
        ),
    )
    parser.add_argument(
        "--occl_eval_first_n_scenes",
        type=int,
        default=0,
        help=(
            "Eval-only helper: if >0, restrict VAL dataset to the first N scene ids present in --occl_sidecar_dir "
            "(sorted by scene id). Useful for camera GT evaluation on a fixed subset."
        ),
    )
    parser.add_argument(
        "--occl_eval_scene_ids_file",
        type=str,
        default=None,
        help=(
            "Eval-only helper: path to a .json (list of ints) or .txt (one int per line) file specifying the exact "
            "scene ids to evaluate on. Overrides --occl_eval_first_n_scenes."
        ),
    )
    parser.add_argument('--occl_aug_num_fruits', type=int, default=4, help='Number of focus fruits per subgraph (random)')
    parser.add_argument('--occl_aug_union_tau', type=float, default=0.5, help='High-union guardrail threshold for focus fruit resampling')
    parser.add_argument('--occl_aug_resample_tries', type=int, default=5, help='Resample tries to include at least one high-union fruit when available')
    parser.add_argument('--occl_aug_max_dist', type=float, default=0.2, help='Leaf candidate radius around each focus fruit (meters)')
    parser.add_argument('--occl_aug_max_pos_leaves', type=int, default=16, help='Max occluding leaves to add per focus fruit')
    parser.add_argument('--occl_aug_max_neg_leaves', type=int, default=8, help='Max random nearby (negative) leaves to add per focus fruit')
    parser.add_argument('--occl_aug_max_total_leaves', type=int, default=96, help='Global cap on added leaf nodes across all focus fruits')
    parser.add_argument('--occl_aug_reserve_nodes', type=int, default=0, help='Reserve this many nodes in the subgraph for occlusion augmentation (BFS core uses subgraph_size - reserve)')

    # Occlusion supervision (pointcloud z-buffer sidecars).
    parser.add_argument('--occl_enabled', action='store_true', help='Enable occlusion supervision (union + potential from z-buffer sidecars)')
    parser.add_argument('--w_occl_union', type=float, default=0.1, help='Loss weight for fruit union occlusion head')
    parser.add_argument('--w_occl_potential', type=float, default=0.1, help='Loss weight for leaf potential occlusion head')
    parser.add_argument('--w_occl_rank', type=float, default=0.1, help='Loss weight for listwise leaf ranking loss (dominant dirs only)')
    parser.add_argument('--w_occl_consistency', type=float, default=0.05, help='Loss weight for leaf->union consistency regularizer')
    parser.add_argument(
        '--occl_consistency_mode',
        type=str,
        default="auto",
        choices=["auto", "mse", "bce_dom", "off"],
        help=(
            'Consistency loss mode. `auto` (default) uses MSE-to-continuous-union when occlusion potential is regressed, '
            'and uses BCE-to-dominance (union>=--occl_union_tau) when --edl_occl_potential is enabled. '
            '`mse` forces the continuous union regression regularizer, `bce_dom` forces dominance BCE, `off` disables it.'
        ),
    )
    parser.add_argument('--stage_occl_epoch', type=int, default=11, help='Epoch when occlusion losses/metrics turn on')
    parser.add_argument('--occl_union_tau', type=float, default=0.5, help='Dominant direction threshold on union for masking/metrics')
    parser.add_argument('--occl_potential_eps', type=float, default=0.05, help='Potential threshold for occluder retrieval metrics')
    parser.add_argument(
        '--occl_potential_dir_sampling',
        type=str,
        default="stratified",
        choices=["dominant_only", "stratified"],
        help=(
            'Which union directions to include for the occlusion potential BCE loss. '
            '`stratified` (default) uses all dominant dirs + a few mid dirs + a few low dirs (see --occl_potential_*). '
            '`dominant_only` reproduces legacy behavior (union>=--occl_union_tau only).'
        ),
    )
    parser.add_argument(
        '--occl_potential_union_tau_low',
        type=float,
        default=0.25,
        help='Low union threshold for stratified potential supervision (low<=tau_low, mid in (tau_low, --occl_union_tau)).',
    )
    parser.add_argument(
        '--occl_potential_n_mid_dirs',
        type=int,
        default=3,
        help='How many mid-occlusion directions to sample per fruit for potential BCE when --occl_potential_dir_sampling=stratified.',
    )
    parser.add_argument(
        '--occl_potential_n_low_dirs',
        type=int,
        default=1,
        help='How many low-occlusion directions to sample per fruit for potential BCE when --occl_potential_dir_sampling=stratified.',
    )
    parser.add_argument(
        '--occl_potential_eps_actionable',
        type=float,
        default=0.15,
        help='(Deprecated) Previously used for extra "actionable" leaf retrieval metrics; those metrics were removed.',
    )
    parser.add_argument('--occl_max_leaves_per_fruit', type=int, default=64, help='Cap candidate leaves per focus fruit for occlusion targets')
    parser.add_argument(
        '--occl_leafset_policy',
        type=str,
        default="geom_stratified",
        choices=["geom_stratified", "sidecar_priority"],
        help=(
            'How to pick the leafset per fruit for occlusion supervision/prediction (occl_pair_index). '
            '`geom_stratified` is deployable (geometry-only; no sidecar/GT-priority leaf picking). '
            '`sidecar_priority` reproduces legacy behavior (include occl-aug sidecar-selected pos/neg leaves first).'
        ),
    )
    parser.add_argument(
        '--occl_supervise_max_fruits_per_scene',
        type=int,
        default=0,
        help=(
            'If >0, supervise up to this many fruits per sampled subgraph for occlusion (including focus fruits). '
            'Extra fruits are selected only when their GT-dominant directions (union>=--occl_union_tau) remain dominant '
            'when considering only leaves present in the sampled subgraph (present-only explainability filter). '
            '0 keeps backwards-compatible behavior: supervise only the focus fruits.'
        ),
    )
    parser.add_argument(
        '--occl_supervise_remain_rate_min',
        type=float,
        default=0.9,
        help='Minimum fraction of GT-dominant directions that remain dominant under present-only union (for extra supervised fruits).',
    )
    parser.add_argument(
        '--occl_supervise_min_dom_dirs',
        type=int,
        default=1,
        help='Minimum number of GT-dominant directions required to consider a fruit for extra supervision.',
    )
    parser.add_argument('--occl_dir_embed_dim', type=int, default=16, help='Direction embedding dim for occlusion pair scoring')
    parser.add_argument('--occl_attn_dim', type=int, default=64, help='Attention hidden dim for occlusion pair pooling')
    parser.add_argument('--occl_attn_temp', type=float, default=1.0, help='Attention softmax temperature for union pooling (lower=sharper)')
    parser.add_argument(
        '--occl_node_feat_source',
        type=str,
        default="combined",
        choices=["combined", "gnn_only", "identity_only"],
        help=(
            'Which per-node embedding to feed into the occlusion module (res model only). '
            '`combined` (default) uses [h_gnn, identity]. '
            '`gnn_only` uses [h_gnn, 0]. '
            '`identity_only` uses [0, identity].'
        ),
    )
    parser.add_argument(
        '--occl_pair_geom_mode',
        type=str,
        default="full",
        choices=["full", "none"],
        help=(
            'Whether to feed fruit-local pair geometry features into the occlusion SA/scorer. '
            '`full` (default) uses all 11 dims. '
            '`none` replaces pair_geom with zeros (ablation; shape-preserving).'
        ),
    )
    parser.add_argument(
        '--occl_leaf_sa_layers',
        type=int,
        default=0,
        help='Number of leaf-set self-attention (Transformer encoder) layers per fruit. 0 disables.',
    )
    parser.add_argument(
        '--occl_leaf_sa_mode',
        type=str,
        default="per_fruit_dir",
        choices=["per_fruit", "per_fruit_dir"],
        help=(
            'Leaf-set self-attention grouping mode. '
            '`per_fruit` runs SA once per fruit and shares contextualized leaf features across all directions. '
            '`per_fruit_dir` runs SA separately per (fruit, direction) with additive direction conditioning (more powerful, '
            'but higher memory). Only used when --occl_leaf_sa_layers>0.'
        ),
    )
    parser.add_argument(
        '--occl_leaf_sa_dim',
        type=int,
        default=128,
        help='Leaf-set self-attention model dim (d_model). Only used when --occl_leaf_sa_layers>0.',
    )
    parser.add_argument(
        '--occl_leaf_sa_heads',
        type=int,
        default=4,
        help='Leaf-set self-attention number of heads. Only used when --occl_leaf_sa_layers>0.',
    )
    parser.add_argument(
        '--occl_leaf_sa_ff_mult',
        type=int,
        default=4,
        help='Leaf-set self-attention FFN width multiplier. Only used when --occl_leaf_sa_layers>0.',
    )
    parser.add_argument(
        '--occl_leaf_sa_dropout',
        type=float,
        default=0.1,
        help='Leaf-set self-attention dropout. Only used when --occl_leaf_sa_layers>0.',
    )
    parser.add_argument('--occl_rank_gamma', type=float, default=1.0, help='Exponent applied to k-buffer mass targets before listwise ranking')
    parser.add_argument('--reversed_curriculum', action='store_true', help='Reversed Curriculum: Local -> Reg -> GNN')
    parser.add_argument('--hard_pair_constraint', action='store_true', help='Enforce Fruit-Peduncle pairing in subgraphs', default=True)
    parser.add_argument('--no_hard_pair_constraint', action='store_false', dest='hard_pair_constraint')
    
    # Staged Curriculum (v12 overrides)
    parser.add_argument('--stage_reg_epoch', type=int, default=6)
    parser.add_argument('--stage_gnn_epoch', type=int, default=11)
    parser.add_argument('--log_interval', type=int, default=10, help='Batch interval for printing logs')
    parser.add_argument('--node_feat_mode', type=str, default=None, choices=['gt', 'local'], help='Inject semantic features into GNN')
    parser.add_argument('--use_4mm', action='store_true', default=True, help='Use 4mm voxelized files')
    parser.add_argument('--no_4mm', action='store_false', dest='use_4mm')
    parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'res'], 
                        help='Model type: baseline (GINE) or res (ResGINE + Identity Concatenation)')
    
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional reproducibility controls. Only active when --seed is provided.
    if args.seed is not None:
        try:
            _seed = int(args.seed)
            random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed)
        except Exception as e:
            print(f"[warn] failed to set RNG seeds (--seed={args.seed}): {e}", flush=True)

    def _metric_is_better(new_v: float, best_v: Optional[float], mode: str) -> bool:
        if new_v is None or not np.isfinite(new_v):
            return False
        if best_v is None or (isinstance(best_v, float) and not np.isfinite(best_v)):
            return True
        if mode == "min":
            return new_v < best_v
        return new_v > best_v

    def _save_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        best_monitor_value: Optional[float],
        monitor_name: str,
        monitor_mode: str,
        monitor_value: Optional[float] = None,
    ) -> None:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_monitor_value": float(best_monitor_value) if best_monitor_value is not None else None,
            "monitor_name": str(monitor_name),
            "monitor_mode": str(monitor_mode),
            "monitor_value": float(monitor_value) if monitor_value is not None else None,
            "args": vars(args),
            # Reproducibility helpers for crash-resume.
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate(),
        }
        if torch.cuda.is_available():
            ckpt["rng_state_cuda_all"] = torch.cuda.get_rng_state_all()
        torch.save(ckpt, str(path))

    def _load_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> dict:
        ckpt = torch.load(str(path), map_location=device or "cpu")
        if strict:
            model.load_state_dict(ckpt["model_state"])
        else:
            incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing or unexpected:
                def _short(xs):
                    xs = list(xs)
                    return xs[:8] + (["..."] if len(xs) > 8 else [])
                print(
                    f"[checkpoint] loaded with strict=False: missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
                    flush=True,
                )
                if missing:
                    print(f"[checkpoint] missing_keys (sample): {_short(missing)}", flush=True)
                if unexpected:
                    print(f"[checkpoint] unexpected_keys (sample): {_short(unexpected)}", flush=True)
        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        # Restore RNG states when present (best-effort).
        try:
            if "rng_state_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_state_torch"])
            if "rng_state_numpy" in ckpt:
                np.random.set_state(ckpt["rng_state_numpy"])
            if "rng_state_python" in ckpt:
                random.setstate(ckpt["rng_state_python"])
            if torch.cuda.is_available() and "rng_state_cuda_all" in ckpt:
                torch.cuda.set_rng_state_all(ckpt["rng_state_cuda_all"])
        except Exception:
            pass
        return ckpt

    # Common gotcha: occl-aug sampling changes the subgraph, but does not enable occlusion supervision.
    if args.occl_aug_enabled and (not args.occl_enabled):
        print(
            "[warn] --occl_aug_enabled was set but --occl_enabled is OFF. "
            "Occlusion losses/metrics will stay at 0 (placeholders) because no occlusion supervision is active.",
            flush=True,
        )

    # Important: geometric point augmentations can destroy the semantics of the "blind" geometry path.
    # In `--no_geom_injection` mode, the model derives node positions from point clusters (cluster_mean).
    # If we jitter/drop points, cluster_mean moves randomly, which can rewire candidate edges and add noise.
    if args.no_geom_injection:
        if float(args.xyz_jitter) > 0.0 or float(args.point_dropout) > 0.0:
            print(
                "[warn] `--no_geom_injection` with geometric point augs "
                f"(--xyz_jitter={args.xyz_jitter}, --point_dropout={args.point_dropout}) "
                "can hurt generalization because cluster means (and candidate edges) move randomly. "
                "If performance drops, try `--xyz_jitter 0 --point_dropout 0` (keep color/rotation if desired).",
                flush=True,
            )
    
    cfg = SimpleNamespace(
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
            class_dropout_rates=[args.pd_stem, args.pd_leaf, args.pd_fruit, args.pd_peduncle],
            occl_enabled=args.occl_enabled,
            occl_num_dirs=18,
            occl_dir_embed_dim=args.occl_dir_embed_dim,
            occl_attn_dim=args.occl_attn_dim,
            occl_attn_temp=args.occl_attn_temp,
            occl_node_feat_source=args.occl_node_feat_source,
            occl_pair_geom_mode=args.occl_pair_geom_mode,
            occl_leaf_sa_layers=args.occl_leaf_sa_layers,
            occl_leaf_sa_mode=args.occl_leaf_sa_mode,
            occl_leaf_sa_dim=args.occl_leaf_sa_dim,
            occl_leaf_sa_heads=args.occl_leaf_sa_heads,
            occl_leaf_sa_ff_mult=args.occl_leaf_sa_ff_mult,
            occl_leaf_sa_dropout=args.occl_leaf_sa_dropout,
        )
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.exp_name}"
    if args.zero_inputs: run_name += "_ZERO"
    if args.shuffle_labels: run_name += "_SHUFFLE"

    # Local run directory for configs/checkpoints.
    run_dir = Path(args.runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI + config for reproducibility.
    try:
        with open(run_dir / "cmd.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[warn] failed to write run metadata into {run_dir}: {e}", flush=True)

    # Lightweight progress tracking for dashboards (independent of W&B).
    progress_path = run_dir / "progress.json"

    def _write_progress(payload: Dict) -> None:
        try:
            tmp = progress_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            tmp.replace(progress_path)
        except Exception as e:
            print(f"[warn] failed to write progress file {progress_path}: {e}", flush=True)

    _write_progress(
        {
            "run_name": run_name,
            "exp_name": str(args.exp_name),
            "ablation_key": str(args.ablation_key) if args.ablation_key else None,
            "pid": int(os.getpid()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "seed": int(args.seed) if args.seed is not None else None,
            "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "epoch": 0,
            "phase": "init",
            "best_monitor": {
                "name": str(args.ckpt_monitor),
                "mode": str(args.ckpt_mode).lower(),
                "value": None,
                "epoch": None,
            },
        }
    )

    # Optional eval-only restriction: evaluate on a fixed set of scenes (explicit allowlist),
    # or on the first N scenes for which occlusion sidecars exist.
    # This is primarily used for camera-ray GT eval where sidecars may be produced progressively.
    eval_scene_allowlist: Optional[Set[int]] = None
    eval_scene_ids: Optional[List[int]] = None
    scene_ids_file = getattr(args, "occl_eval_scene_ids_file", None)
    if scene_ids_file:
        p = Path(str(scene_ids_file))
        if not p.exists():
            raise SystemExit(f"[abort] --occl_eval_scene_ids_file does not exist: {p}")
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    eval_scene_ids = [int(x) for x in obj]
                elif isinstance(obj, dict) and "scene_ids" in obj:
                    eval_scene_ids = [int(x) for x in obj["scene_ids"]]
                elif isinstance(obj, dict) and "ids" in obj:
                    eval_scene_ids = [int(x) for x in obj["ids"]]
                else:
                    raise ValueError("expected a JSON list of ints (or dict with scene_ids/ids)")
            else:
                ids: List[int] = []
                for ln in p.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    ids.append(int(ln))
                eval_scene_ids = ids
        except Exception as e:
            raise SystemExit(f"[abort] failed to parse --occl_eval_scene_ids_file {p}: {e}")
        eval_scene_ids = sorted({int(s) for s in eval_scene_ids})
        if not eval_scene_ids:
            raise SystemExit(f"[abort] --occl_eval_scene_ids_file was empty: {p}")
        eval_scene_allowlist = set(eval_scene_ids)
        try:
            out_path = run_dir / "occl_eval_scene_ids.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"source": str(p), "scene_ids": eval_scene_ids}, f, indent=2)
        except Exception as e:
            print(f"[warn] failed to write eval scene id allowlist: {e}", flush=True)
        print(
            f"[eval] restricting VAL to explicit scene allowlist ({len(eval_scene_ids)} ids) from {p} "
            f"(min={eval_scene_ids[0]}, max={eval_scene_ids[-1]})",
            flush=True,
        )
    else:
        try:
            n_first = int(getattr(args, "occl_eval_first_n_scenes", 0) or 0)
        except Exception:
            n_first = 0
        if n_first > 0:
            if not args.occl_sidecar_dir:
                raise SystemExit("[abort] --occl_eval_first_n_scenes requires --occl_sidecar_dir")
            sidecar_dir = Path(str(args.occl_sidecar_dir))
            if not sidecar_dir.exists():
                raise SystemExit(f"[abort] --occl_sidecar_dir does not exist: {sidecar_dir}")
            pat = "scene_*_occl_cam_*.npz" if str(getattr(args, "occl_gt_source", "zbuf")) in ("cam_ortho", "cam_persp") else "scene_*_occl_zbuf_*.npz"
            ids: Set[int] = set()
            for p in sorted(sidecar_dir.glob(pat)):
                parts = p.name.split("_")
                if len(parts) < 2 or parts[0] != "scene":
                    continue
                try:
                    ids.add(int(parts[1]))
                except Exception:
                    continue
            ids_sorted = sorted(ids)
            if not ids_sorted:
                raise SystemExit(f"[abort] no occlusion sidecars found in {sidecar_dir} (pattern: {pat})")
            eval_scene_ids = ids_sorted[: min(n_first, len(ids_sorted))]
            eval_scene_allowlist = set(eval_scene_ids)
            try:
                out_path = run_dir / f"occl_eval_scene_ids_first{n_first}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(eval_scene_ids, f, indent=2)
                out_txt = run_dir / f"occl_eval_scene_ids_first{n_first}.txt"
                with open(out_txt, "w", encoding="utf-8") as f:
                    for sid in eval_scene_ids:
                        f.write(f"{sid}\n")
            except Exception as e:
                print(f"[warn] failed to write eval scene id allowlist: {e}", flush=True)
            print(
                f"[eval] restricting VAL to first {len(eval_scene_ids)} scenes with sidecars "
                f"(pattern={pat}) in {sidecar_dir} (min={eval_scene_ids[0]}, max={eval_scene_ids[-1]})",
                flush=True,
            )

    # W&B Init (online by default)
    mode = str(args.wandb_mode).lower() if args.wandb_mode is not None else ("offline" if args.offline else "online")
    
    config = vars(args)
    config.update({
        'dataset.subgraph_enabled': args.subgraph_enabled,
        'dataset.subgraph_size': args.subgraph_size,
        'dataset.hard_pair_constraint': args.hard_pair_constraint
    })
    if args.ablation_key:
        config["ablation_key"] = str(args.ablation_key)
    config["env.cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if eval_scene_ids is not None:
        config["eval_scene_count"] = int(len(eval_scene_ids))
        config["eval_scene_min"] = int(eval_scene_ids[0]) if eval_scene_ids else None
        config["eval_scene_max"] = int(eval_scene_ids[-1]) if eval_scene_ids else None
    
    wandb_run_dir = run_dir / "wandb"
    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project=str(getattr(args, "wandb_project", "pepper_oracle")),
        name=run_name,
        config=config,
        mode=mode,
        dir=str(wandb_run_dir),
        group=getattr(args, "wandb_group", None),
        tags=getattr(args, "wandb_tag", None),
    )

    # Aug gating:
    # - Point-level augs run when any rate is >0.
    # - Rotation aug is controlled by --z_rot_aug_mode (set to 'none' to disable).
    point_aug_enabled = (
        float(args.color_dropout) > 0.0 or float(args.xyz_jitter) > 0.0 or float(args.point_dropout) > 0.0
    )
    rot_aug_enabled = str(getattr(args, "z_rot_aug_mode", "continuous")).lower() != "none"

    train_ds = OraclePepperDataset(
        data_root=args.data_root,
        split='train',
        npoint_per_instance=args.npoint,
        augment=point_aug_enabled,
        rotate_aug=rot_aug_enabled,
        rotate_mode=str(args.z_rot_aug_mode),
        drop_shoots=args.drop_shoots,
        k_candidates=args.k_candidates,
        extra_stems_per_node=args.extra_stems_per_node,
        dist_threshold=args.dist_threshold,
        use_rgb=args.use_rgb,
        color_dropout=args.color_dropout,
        xyz_jitter=args.xyz_jitter,
        point_dropout=args.point_dropout,
        # v12 Subgraph Args
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
        occl_aug_reserve_nodes=args.occl_aug_reserve_nodes,
        occl_targets_enabled=args.occl_enabled,
        occl_gt_source=args.occl_gt_source,
        occl_cam_jitter_level=args.occl_cam_jitter_level,
        occl_targets_union_tau=args.occl_union_tau,
        occl_targets_max_leaves_per_fruit=args.occl_max_leaves_per_fruit,
        occl_leafset_policy=args.occl_leafset_policy,
        occl_supervise_max_fruits_per_scene=args.occl_supervise_max_fruits_per_scene,
        occl_supervise_remain_rate_min=args.occl_supervise_remain_rate_min,
        occl_supervise_min_dom_dirs=args.occl_supervise_min_dom_dirs,
        use_4mm=args.use_4mm,
    )
    
    val_ds = OraclePepperDataset(
        data_root=args.data_root,
        split='val',
        augment=False,
        rotate_aug=False,
        rotate_mode=str(args.z_rot_aug_mode),
        drop_shoots=args.drop_shoots,
        k_candidates=args.k_candidates,
        extra_stems_per_node=args.extra_stems_per_node,
        dist_threshold=args.dist_threshold,
        use_rgb=args.use_rgb,
        color_dropout=0.0,
        xyz_jitter=0.0,
        point_dropout=0.0,
        # Validation Args
        subgraph_enabled=args.subgraph_enabled,
        subgraph_size=args.subgraph_size,
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
        occl_aug_reserve_nodes=args.occl_aug_reserve_nodes,
        occl_targets_enabled=args.occl_enabled,
        occl_gt_source=args.occl_gt_source,
        occl_cam_jitter_level=args.occl_cam_jitter_level,
        occl_targets_union_tau=args.occl_union_tau,
        occl_targets_max_leaves_per_fruit=args.occl_max_leaves_per_fruit,
        occl_leafset_policy=args.occl_leafset_policy,
        occl_supervise_max_fruits_per_scene=args.occl_supervise_max_fruits_per_scene,
        occl_supervise_remain_rate_min=args.occl_supervise_remain_rate_min,
        occl_supervise_min_dom_dirs=args.occl_supervise_min_dom_dirs,
        use_4mm=args.use_4mm,
        scene_id_allowlist=eval_scene_allowlist,
    )
    
    # Use custom collate if subgraph enabled (variable sizes), else standard is fine but safe to use custom always for consistency
    train_collate = oracle_collate_fn # Always use oracle_collate_fn for consistency with PyG Data objects

    dl_generator = None
    worker_init_fn = None
    if args.seed is not None:
        try:
            _seed = int(args.seed)
            dl_generator = torch.Generator()
            dl_generator.manual_seed(_seed)

            def _worker_init_fn(worker_id: int) -> None:
                s = _seed + int(worker_id)
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)

            worker_init_fn = _worker_init_fn
        except Exception as e:
            print(f"[warn] failed to configure DataLoader seeding: {e}", flush=True)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        collate_fn=train_collate,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )
    
    # Val loader can use default collate or custom (custom is safer for GNN batching)
    # Actually, standard PyTorch DataLoader with default_collate STACKS tensors.
    # PyG collation CONCATENATES attributes.
    # We MUST use oracle_collate_fn for ALL loaders to ensure consistent 'batch' attribute handling.
    # v12 Update: Always use oracle_collate_fn for safety.
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size if args.subgraph_enabled else args.val_batch_size, 
        shuffle=False, 
        num_workers=args.batch_size if args.subgraph_enabled else args.val_batch_size, # Use 1 worker per val batch for memory safety
        pin_memory=True,
        collate_fn=oracle_collate_fn,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )

    if args.model_type == 'res':
        print("Using OracleResBaseline (Elite v12 Architecture)", flush=True)
        model = OracleResBaseline(cfg, node_classes=4, edge_labels=3).to(DEVICE)
    else:
        print("Using OracleGTBaseline (Standard v12 Architecture)", flush=True)
        model = OracleGTBaseline(cfg, node_classes=4, edge_labels=3).to(DEVICE)
    
    # Compute node weights from training set (Phase 4 / v13)
    node_weights = compute_weights(train_ds, num_classes=4).to(DEVICE)
    
    if args.val_only:
        print("Starting Validation Only Run...", flush=True)
        # Optional: load a checkpoint for eval-only runs.
        if args.resume is not None:
            ckpt_path = Path(args.resume)
            print(f"[val_only] loading checkpoint: {ckpt_path}", flush=True)
            _load_checkpoint(ckpt_path, model=model, optimizer=None, scheduler=None, device=DEVICE, strict=False)
        # `validate()` gates some metrics/prints by curriculum epoch (stage_*). For eval-only runs we
        # want the full set of metrics, so we evaluate at a "late" epoch by default.
        val_epoch = int(getattr(args, "epochs", 0) or 0)
        val_epoch = max(
            val_epoch,
            int(getattr(args, "stage_reg_epoch", 0) or 0),
            int(getattr(args, "stage_gnn_epoch", 0) or 0),
            int(getattr(args, "stage_occl_epoch", 0) or 0),
        )
        metrics = validate(model, val_loader, DEVICE, args, epoch=val_epoch)
        # Persist a JSON copy of the validation metrics for later comparison.
        try:
            def _jsonify(v):
                if isinstance(v, (np.floating, np.integer)):
                    return v.item()
                if isinstance(v, np.ndarray):
                    return v.tolist()
                if isinstance(v, (float, int, str, bool)) or v is None:
                    return v
                try:
                    return float(v)
                except Exception:
                    return str(v)
            metrics_out = {str(k): _jsonify(v) for k, v in (metrics or {}).items()}
            # If the run directory was moved/renamed mid-run, ensure the original path exists.
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_out, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"[warn] failed to write val metrics: {e}", flush=True)
        print("Validation Only Run Complete.", flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Starting Training: {run_name}", flush=True)
    print(f"[run_dir] {run_dir}", flush=True)
    # validate(model, val_loader, DEVICE, args, 0) # Removed per user request

    # Checkpoint paths (oracle runs).
    ckpt_best_path = run_dir / "best_model.pt"
    ckpt_last_path = run_dir / "last_model.pt"
    best_monitor_value: Optional[float] = None
    best_monitor_epoch: Optional[int] = None
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        print(f"[resume] loading checkpoint: {resume_path}", flush=True)
        ckpt = _load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, device=DEVICE)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_monitor_value = ckpt.get("best_monitor_value", None)
        if best_monitor_value is not None:
            try:
                best_monitor_value = float(best_monitor_value)
            except Exception:
                best_monitor_value = None
        print(f"[resume] start_epoch={start_epoch} best_monitor_value={best_monitor_value}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        # Epoch-mean occlusion losses (helps diagnose "flat" per-batch curves).
        epoch_occl_union_losses: List[float] = []
        epoch_occl_potential_losses: List[float] = []
        epoch_occl_rank_losses: List[float] = []
        epoch_occl_consistency_losses: List[float] = []
        node_local_metrics_list = []
        node_metrics_list = []
        edge_label_metrics_list = []
        exist_stats = {"prec": [], "rec": [], "f1": [], "pos_rate": [], "recall_cand": []}
        occl_dir_stats = {
            "prec": [],
            "rec": [],
            "f1": [],
            "brier_dom": [],
            "brier_union": [],
            "ece_dom": [],
            "pos_rate": [],
            "pred_pos_rate": [],
        }
        occl_leaf_stats = {
            "recall_at_1": [],
            "recall_at_3": [],
            "recall_at_5": [],
        }
        occl_leaf_mass_stats = {
            "mass_at_1": [],
            "mass_at_3": [],
            "mass_at_5": [],
        }
        occl_leaf_ndcg_stats = {
            "ndcg_at_1": [],
            "ndcg_at_3": [],
            "ndcg_at_5": [],
        }
        centroid_maes = []
        extent_maes = []
        
        all_node_targets = []
        all_node_local_preds = []
        all_node_global_preds = []
        
        node_classes = ["Stem", "Leaf", "Ped", "Fruit"]
        edge_classes = ["Stem-Leaf", "Stem-Ped", "Ped-Fruit"]
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=os.environ.get("DISABLE_TQDM", "False") == "True")
        for i, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            
            if args.zero_inputs:
                batch.x_organ.zero_()
                batch.n_organ.zero_()
                batch.json_geom.zero_()
                
            if args.shuffle_labels:
                batch.y_node = batch.y_node[torch.randperm(batch.y_node.size(0))]
            
            optimizer.zero_grad()
            
            # [Zero-Cheat Stress Test]
            # If no_geom_injection is on, zero out the GT geom features before forward pass.
            # This proves the model HAS to use cluster_mean and local shape.
            gt_geom_clone = None
            if getattr(args, 'no_geom_injection', False):
                gt_geom_clone = batch.json_geom.clone()
                batch.json_geom = torch.zeros_like(batch.json_geom) # Absolute Zero-Cheat
            
            out = model(batch)
            
            # Restore for loss calculation
            if gt_geom_clone is not None:
                batch.json_geom = gt_geom_clone
                
            losses, targs = compute_oracle_loss(out, batch, args, epoch, batch_idx=i, node_weights=node_weights)
            
            losses["loss_total"].backward()
            optimizer.step()
            
            epoch_losses.append(losses["loss_total"].item())
            if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
                epoch_occl_union_losses.append(float(losses.get("occl_union", 0.0).item()))
                epoch_occl_potential_losses.append(float(losses.get("occl_potential", 0.0).item()))
                epoch_occl_rank_losses.append(float(losses.get("occl_rank", 0.0).item()))
                epoch_occl_consistency_losses.append(float(losses.get("occl_consistency", 0.0).item()))
            
            w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
            w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0

            # Get targets built during loss
            y_exist = targs["y_exist"]
            y_edge = targs["y_edge"]
            num_gt_found = targs["num_gt_in_cand"]

            # 1. Node Local (Always)
            n_mets_local = compute_metrics(out['local_node_logits'], batch.y_node, labels=list(range(len(node_classes))))
            if n_mets_local: node_local_metrics_list.append(n_mets_local)
            all_node_local_preds.append(out['local_node_logits'].argmax(dim=-1).cpu().numpy())
            all_node_targets.append(batch.y_node.cpu().numpy())

            # 2. GNN Metrics (Gated)
            if w_gnn > 0:
                n_mets = compute_metrics(out['node_logits'], batch.y_node, labels=list(range(len(node_classes))))
                if n_mets: node_metrics_list.append(n_mets)
                all_node_global_preds.append(out['node_logits'].argmax(dim=-1).cpu().numpy())

                if y_exist.numel() > 0:
                    edge_logits = out['edge_exist_logits'].squeeze(-1)
                    if bool(getattr(args, "edl_edge_exist", False)):
                        ev_fn = _get_edl_evidence_fn(args)
                        if ev_fn is None:
                            probs = torch.sigmoid(edge_logits)
                        else:
                            alpha = _edl_tied_beta_alpha(edge_logits, ev_fn)  # [E,2]
                            probs = alpha[..., 1] / (alpha.sum(dim=-1) + 1e-12)
                    else:
                        probs = torch.sigmoid(edge_logits)
                    preds_exist = (probs > 0.5).long().cpu().numpy()
                    y_exist_np = y_exist.cpu().numpy()
                    p, r, f, _ = precision_recall_fscore_support(y_exist_np, preds_exist, average='binary', zero_division=0)
                    exist_stats["prec"].append(p)
                    exist_stats["rec"].append(r)
                    exist_stats["f1"].append(f)
                    
                    total_gt = batch.num_gt_total.sum().item()
                    caught_gt = num_gt_found
                    num_cand = len(y_exist) / 2
                    if total_gt > 0: exist_stats["recall_cand"].append(caught_gt / total_gt)
                    if num_cand > 0: exist_stats["pos_rate"].append(caught_gt / num_cand)

                edge_mask = (y_edge != -1)
                if edge_mask.any():
                    e_mets = compute_metrics(out['edge_label_logits'], y_edge, mask=edge_mask, labels=list(range(len(edge_classes))))
                    if e_mets: edge_label_metrics_list.append(e_mets)

            # 3. Regression (Gated)
            if w_reg_curric > 0:
                centroid_maes.append(losses["mae_centroid"].item())
                extent_maes.append(losses["mae_extent"].item())

            # 4. Occlusion metrics (only after stage, if enabled).
            if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
                ds = compute_occl_dir_stats(
                    out.get("occl_fruit_union_logits", None),
                    getattr(batch, "occl_fruit_union", None),
                    union_tau=float(getattr(args, "occl_union_tau", 0.5)),
                )
                _append_stats(occl_dir_stats, ds)

                # Use evidential mean probability for leaf scoring when enabled; otherwise sigmoid(logit).
                pair_logits = out.get("occl_pair_potential_logits", None)
                pair_probs = None
                if bool(getattr(args, "edl_occl_potential", False)) and pair_logits is not None and pair_logits.numel() > 0:
                    ev_fn = _get_edl_evidence_fn(args)
                    if ev_fn is not None:
                        alpha_p = _edl_tied_beta_alpha(pair_logits, ev_fn)  # [M,D,2]
                        pair_probs = alpha_p[..., 1] / (alpha_p.sum(dim=-1) + 1e-12)  # [M,D]

                ls = compute_occl_leaf_retrieval_stats(
                    pair_logits,
                    getattr(batch, "occl_pair_potential", None),
                    pair_probs,
                    getattr(batch, "occl_pair_index", None),
                    getattr(batch, "occl_fruit_idx", None),
                    getattr(batch, "occl_fruit_union", None),
                    union_tau=None,
                    pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                    ks=(1, 3, 5),
                )
                _append_stats(occl_leaf_stats, ls)

                pk = getattr(batch, "occl_pair_k_rank", None)
                if pk is not None and pk.numel() > 0:
                    mass = pk.sum(dim=-1)  # [M,D]
                    ms = compute_occl_leaf_mass_at_k_stats(
                        pair_logits,
                        mass,
                        pair_probs,
                        getattr(batch, "occl_pair_index", None),
                        getattr(batch, "occl_fruit_idx", None),
                        getattr(batch, "occl_fruit_union", None),
                        union_tau=None,
                        pot=getattr(batch, "occl_pair_potential", None),
                        pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                        ks=(1, 3, 5),
                    )
                    _append_stats(occl_leaf_mass_stats, ms)

                    nd = compute_occl_leaf_ndcg_at_k_stats(
                        pair_logits,
                        mass,
                        pair_probs,
                        getattr(batch, "occl_pair_index", None),
                        getattr(batch, "occl_fruit_idx", None),
                        getattr(batch, "occl_fruit_union", None),
                        union_tau=None,
                        pot=getattr(batch, "occl_pair_potential", None),
                        pot_eps=float(getattr(args, "occl_potential_eps", 0.05)),
                        ks=(1, 3, 5),
                    )
                    _append_stats(occl_leaf_ndcg_stats, nd)

            # Per-batch Print Logging
            if (i + 1) % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses)
                msg = f"  Epoch {epoch} [{i+1}/{len(train_loader)}] loss: {avg_loss:.4f}"
                
                # Local Node
                curr_local = node_local_metrics_list[-1]
                msg += f" | NodeLoc(A:{curr_local['acc']:.2f} P:{curr_local['prec_macro']:.2f} R:{curr_local['rec_macro']:.2f})"
                
                if w_gnn > 0:
                    curr_glob = node_metrics_list[-1]
                    msg += f" | NodeGlob(A:{curr_glob['acc']:.2f} P:{curr_glob['prec_macro']:.2f} R:{curr_glob['rec_macro']:.2f})"
                    if exist_stats["f1"]:
                        msg += f" | Exist(F1:{exist_stats['f1'][-1]:.2f} PosR:{exist_stats['pos_rate'][-1]:.2f})"
                    if edge_label_metrics_list:
                        curr_edge = edge_label_metrics_list[-1]
                        msg += f" | EdgeLab(A:{curr_edge['acc']:.2f} P:{curr_edge['prec_macro']:.2f} R:{curr_edge['rec_macro']:.2f})"
                
                if w_reg_curric > 0:
                    msg += f" | MAE(C:{centroid_maes[-1]:.2f} E:{extent_maes[-1]:.2f})"

                if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
                    if occl_dir_stats["f1"] and occl_leaf_mass_stats["mass_at_3"]:
                        msg += (
                            f" | Occl(F1:{occl_dir_stats['f1'][-1]:.2f} "
                            f"Mass@3:{occl_leaf_mass_stats['mass_at_3'][-1]:.2f})"
                        )
                
                print(msg, flush=True)

            pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")
            
            if wandb.run:
                metrics = {f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
                metrics["train/node_local_acc"] = n_mets_local.get("acc", 0)
                metrics["train/node_local_f1"] = n_mets_local.get("f1_macro", 0)
                if w_gnn > 0:
                    metrics["train/node_acc"] = n_mets.get("acc", 0)
                    metrics["train/node_f1"] = n_mets.get("f1_macro", 0)
                wandb.log(metrics, step=epoch)

            # Optional cap for smoke tests / quick sanity runs.
            if int(getattr(args, "max_train_batches", 0)) > 0 and (i + 1) >= int(args.max_train_batches):
                print(f"[info] reached --max_train_batches={args.max_train_batches}; ending epoch early.", flush=True)
                break

        # Training metrics aggregation
        train_log_dict = {"epoch": epoch}
        if node_local_metrics_list:
            node_local_agg = aggregate_per_class_and_return(node_local_metrics_list, node_classes, "train/node_local")
            train_log_dict.update(node_local_agg)
        if node_metrics_list:
            node_global_agg = aggregate_per_class_and_return(node_metrics_list, node_classes, "train/node")
            train_log_dict.update(node_global_agg)
        if edge_label_metrics_list:
            edge_label_agg = aggregate_per_class_and_return(edge_label_metrics_list, edge_classes, "train/edge_label")
            train_log_dict.update(edge_label_agg)
        
        for k, v in exist_stats.items():
            if v: train_log_dict[f"train/edge_exist_{k}"] = np.mean(v)

        if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
            for k, v in occl_dir_stats.items():
                if v:
                    train_log_dict[f"train/occl_dir_{k}"] = float(np.mean(v))
            for k, v in occl_leaf_stats.items():
                if v:
                    train_log_dict[f"train/occl_leaf_{k}"] = float(np.mean(v))
            for k, v in occl_leaf_mass_stats.items():
                if v:
                    train_log_dict[f"train/occl_leaf_{k}"] = float(np.mean(v))
            for k, v in occl_leaf_ndcg_stats.items():
                if v:
                    train_log_dict[f"train/occl_leaf_{k}"] = float(np.mean(v))

            # Epoch-mean occlusion losses (per-batch logging can look flat because `wandb.log(step=epoch)` is repeated).
            if epoch_occl_union_losses:
                train_log_dict["train/occl_loss_union_epoch"] = float(np.mean(epoch_occl_union_losses))
                train_log_dict["train/occl_loss_potential_epoch"] = float(np.mean(epoch_occl_potential_losses))
                train_log_dict["train/occl_loss_rank_epoch"] = float(np.mean(epoch_occl_rank_losses))
                train_log_dict["train/occl_loss_consistency_epoch"] = float(np.mean(epoch_occl_consistency_losses))

        train_log_dict["train/centroid_mae"] = np.mean(centroid_maes) if centroid_maes else 0.0
        train_log_dict["train/extent_mae"] = np.mean(extent_maes) if extent_maes else 0.0

        if wandb.run:
            wandb.log(train_log_dict, step=epoch)

        # Update dashboard progress (train-side).
        try:
            _train_subset_keys = [
                "train/occl_leaf_ndcg_at_3",
                "train/occl_leaf_mass_at_3",
                "train/occl_dir_f1",
                "train/edge_exist_f1",
                "train/mae_centroid",
                "train/mae_extent",
            ]
            _train_subset = {k: float(train_log_dict[k]) for k in _train_subset_keys if k in train_log_dict}
            _write_progress(
                {
                    "run_name": run_name,
                    "exp_name": str(args.exp_name),
                    "ablation_key": str(args.ablation_key) if args.ablation_key else None,
                    "pid": int(os.getpid()),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "seed": int(args.seed) if args.seed is not None else None,
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "epoch": int(epoch),
                    "phase": "train",
                    "train": _train_subset,
                    "best_monitor": {
                        "name": str(args.ckpt_monitor),
                        "mode": str(args.ckpt_mode).lower(),
                        "value": float(best_monitor_value) if best_monitor_value is not None else None,
                        "epoch": int(best_monitor_epoch) if best_monitor_epoch is not None else None,
                    },
                }
            )
        except Exception:
            pass
            
        w_gnn = 1.0 if epoch >= args.stage_gnn_epoch else 0.0
        w_reg_curric = 1.0 if epoch >= args.stage_reg_epoch else 0.0

        print(f"\nTraining Epoch {epoch} Summary:", flush=True)
        # 1. Node Local (Always)
        print(f"  Node Local: Acc={train_log_dict.get('train/node_local_acc', 0):.4f}, "
              f"P(Macro)={train_log_dict.get('train/node_local_prec_macro', 0):.4f}, "
              f"R(Macro)={train_log_dict.get('train/node_local_rec_macro', 0):.4f}", flush=True)
        if node_local_metrics_list:
            for i, name in enumerate(node_classes):
                p = train_log_dict.get(f"train/node_local_{name}_precision", 0)
                r = train_log_dict.get(f"train/node_local_{name}_recall", 0)
                f = train_log_dict.get(f"train/node_local_{name}_f1", 0)
                print(f"        {name:7}: P={p:.3f}, R={r:.3f}, F1={f:.3f}", flush=True)
            
            # Confusion Matrix
            cm_local = confusion_matrix(np.concatenate(all_node_targets), np.concatenate(all_node_local_preds), labels=range(len(node_classes)))
            print_text_confusion_matrix(cm_local, node_classes, "Node Local Confusion Matrix (Train)")
            
        # 2. GNN (Gated)
        if w_gnn > 0:
            print(f"  Node Global: Acc={train_log_dict.get('train/node_acc', 0):.4f}, "
                  f"P(Macro)={train_log_dict.get('train/node_prec_macro', 0):.4f}, "
                  f"R(Macro)={train_log_dict.get('train/node_rec_macro', 0):.4f}", flush=True)
            if node_metrics_list:
                for i, name in enumerate(node_classes):
                    p = train_log_dict.get(f"train/node_{name}_precision", 0)
                    r = train_log_dict.get(f"train/node_{name}_recall", 0)
                    f = train_log_dict.get(f"train/node_{name}_f1", 0)
                    print(f"        {name:7}: P={p:.3f}, R={r:.3f}, F1={f:.3f}", flush=True)
                
                # Confusion Matrix
                cm_glob = confusion_matrix(np.concatenate(all_node_targets), np.concatenate(all_node_global_preds), labels=range(len(node_classes)))
                print_text_confusion_matrix(cm_glob, node_classes, "Node Global Confusion Matrix (Train)")
            
            print(f"  Edge Exist: F1={train_log_dict.get('train/edge_exist_f1', 0):.4f}, "
                  f"P={train_log_dict.get('train/edge_exist_prec', 0):.4f}, "
                  f"R={train_log_dict.get('train/edge_exist_rec', 0):.4f}", flush=True)
            print(f"              CandRecall={train_log_dict.get('train/edge_exist_recall_cand', 0):.4f}, "
                  f"PosRate={train_log_dict.get('train/edge_exist_pos_rate', 0):.4f}", flush=True)
            
            print(f"  Edge Label: Acc={train_log_dict.get('train/edge_label_acc', 0):.4f}, "
                  f"P(Macro)={train_log_dict.get('train/edge_label_prec_macro', 0):.4f}, "
                  f"R(Macro)={train_log_dict.get('train/edge_label_rec_macro', 0):.4f}", flush=True)
            if edge_label_metrics_list:
                for i, name in enumerate(edge_classes):
                    p = train_log_dict.get(f"train/edge_label_{name}_precision", 0)
                    r = train_log_dict.get(f"train/edge_label_{name}_recall", 0)
                    f = train_log_dict.get(f"train/edge_label_{name}_f1", 0)
                    print(f"        {name:9}: P={p:.3f}, R={r:.3f}, F1={f:.3f}", flush=True)
        
        if w_reg_curric > 0:
            print(
                f"\n  MAE: Centroid={train_log_dict.get('train/centroid_mae', 0):.4f}, "
                f"Extent={train_log_dict.get('train/extent_mae', 0):.4f}",
                flush=True,
            )

        if getattr(args, "occl_enabled", False) and epoch >= getattr(args, "stage_occl_epoch", 0):
            print(
                f"\n  Occl Dir: F1={train_log_dict.get('train/occl_dir_f1', 0):.4f}, "
                f"P={train_log_dict.get('train/occl_dir_prec', 0):.4f}, "
                f"R={train_log_dict.get('train/occl_dir_rec', 0):.4f}, "
                f"ECE={train_log_dict.get('train/occl_dir_ece_dom', 0):.4f}, "
                f"Brier(dom)={train_log_dict.get('train/occl_dir_brier_dom', 0):.4f}",
                flush=True,
            )
            print(
                f"  Occl Leaf: Recall@1/3/5={train_log_dict.get('train/occl_leaf_recall_at_1', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_recall_at_3', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_recall_at_5', 0):.3f}",
                flush=True,
            )
            print(
                f"            Mass@1/3/5={train_log_dict.get('train/occl_leaf_mass_at_1', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_mass_at_3', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_mass_at_5', 0):.3f}",
                flush=True,
            )
            print(
                f"            NDCG@1/3/5={train_log_dict.get('train/occl_leaf_ndcg_at_1', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_ndcg_at_3', 0):.3f}/"
                f"{train_log_dict.get('train/occl_leaf_ndcg_at_5', 0):.3f}",
                flush=True,
            )
            if "train/occl_loss_union_epoch" in train_log_dict:
                print(
                    f"  Occl Loss (epoch mean): union={train_log_dict.get('train/occl_loss_union_epoch', 0):.4f} "
                    f"pot={train_log_dict.get('train/occl_loss_potential_epoch', 0):.4f} "
                    f"rank={train_log_dict.get('train/occl_loss_rank_epoch', 0):.4f} "
                    f"cons={train_log_dict.get('train/occl_loss_consistency_epoch', 0):.4f}",
                    flush=True,
                )

        # Validation every val_interval epochs
        if epoch % args.val_interval == 0:
            val_log = validate(model, val_loader, DEVICE, args, epoch)
            monitor_name = str(args.ckpt_monitor)
            monitor_mode = str(args.ckpt_mode).lower()
            monitor_value = None
            if isinstance(val_log, dict):
                v = val_log.get(monitor_name, None)
                try:
                    monitor_value = float(v) if v is not None else None
                except Exception:
                    monitor_value = None
            if monitor_value is None or (isinstance(monitor_value, float) and not np.isfinite(monitor_value)):
                print(
                    f"[ckpt] monitor '{monitor_name}' not available/finite at epoch {epoch}. "
                    "Skipping best_model update this validation.",
                    flush=True,
                )
            else:
                if _metric_is_better(monitor_value, best_monitor_value, monitor_mode):
                    best_monitor_value = float(monitor_value)
                    best_monitor_epoch = int(epoch)
                    print(
                        f"[ckpt] new best at epoch {epoch}: {monitor_name}={best_monitor_value:.6f} "
                        f"(mode={monitor_mode}) -> saving {ckpt_best_path}",
                        flush=True,
                    )
                    _save_checkpoint(
                        ckpt_best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_monitor_value=best_monitor_value,
                        monitor_name=monitor_name,
                        monitor_mode=monitor_mode,
                        monitor_value=monitor_value,
                    )

            # Update dashboard progress (val-side).
            try:
                if isinstance(val_log, dict):
                    _val_subset_keys = [
                        "val/occl_leaf_ndcg_at_3",
                        "val/occl_leaf_mass_at_3",
                        "val/occl_dir_best_f1",
                        "val/edge_exist_best_f1",
                        "val/mae_centroid",
                        "val/mae_extent",
                    ]
                    _val_subset = {k: float(val_log[k]) for k in _val_subset_keys if k in val_log}
                    _write_progress(
                        {
                            "run_name": run_name,
                            "exp_name": str(args.exp_name),
                            "ablation_key": str(args.ablation_key) if args.ablation_key else None,
                            "pid": int(os.getpid()),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "seed": int(args.seed) if args.seed is not None else None,
                            "time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "epoch": int(epoch),
                            "phase": "val",
                            "val": _val_subset,
                            "monitor": {
                                "name": str(monitor_name),
                                "mode": str(monitor_mode),
                                "value": float(monitor_value) if monitor_value is not None else None,
                            },
                            "best_monitor": {
                                "name": str(args.ckpt_monitor),
                                "mode": str(args.ckpt_mode).lower(),
                                "value": float(best_monitor_value) if best_monitor_value is not None else None,
                                "epoch": int(best_monitor_epoch) if best_monitor_epoch is not None else None,
                            },
                        }
                    )
            except Exception:
                pass
        scheduler.step()

        # Always save "last" checkpoint for crash-resume.
        _save_checkpoint(
            ckpt_last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_monitor_value=best_monitor_value,
            monitor_name=str(args.ckpt_monitor),
            monitor_mode=str(args.ckpt_mode).lower(),
            monitor_value=None,
        )
        
    # Final progress marker (useful for dashboards when W&B is offline/disabled).
    try:
        _write_progress(
            {
                "run_name": run_name,
                "exp_name": str(args.exp_name),
                "ablation_key": str(args.ablation_key) if args.ablation_key else None,
                "pid": int(os.getpid()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "seed": int(args.seed) if args.seed is not None else None,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "epoch": int(args.epochs),
                "phase": "done",
                "best_monitor": {
                    "name": str(args.ckpt_monitor),
                    "mode": str(args.ckpt_mode).lower(),
                    "value": float(best_monitor_value) if best_monitor_value is not None else None,
                    "epoch": int(best_monitor_epoch) if best_monitor_epoch is not None else None,
                },
            }
        )
    except Exception:
        pass

    wandb.finish()

if __name__ == "__main__":
    main()
