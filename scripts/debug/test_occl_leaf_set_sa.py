#!/usr/bin/env python3
"""
Sanity checks for the occlusion leaf-set self-attention packing/unpacking logic.

This does NOT depend on the oracle dataset; it directly exercises the model helper
that packs variable-length leaf sets into a padded batch for a Transformer encoder.

Run (from repo root):
  python -u -m scripts.debug.test_occl_leaf_set_sa
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline


class _IdentitySA(nn.Module):
  def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    return x


def _make_cfg(*, sa_layers: int = 1, sa_dim: int = 128, sa_heads: int = 4) -> SimpleNamespace:
  return SimpleNamespace(
    model=SimpleNamespace(
      hidden_dim=128,
      num_layers=2,
      dropout=0.1,
      oracle_msg_npoint=16,
      oracle_msg_radius_list=[0.01, 0.02, 0.04],
      oracle_msg_nsample_list=[8, 8, 8],
      node_feat_mode=None,
      k_candidates=8,
      dist_threshold=0.15,
      extra_stems_per_node=2,
      use_normals=False,
      use_rgb=False,
      no_geom_injection=True,
      occl_enabled=True,
      occl_num_dirs=18,
      occl_dir_embed_dim=16,
      occl_attn_dim=64,
      occl_attn_temp=1.0,
      occl_leaf_sa_layers=sa_layers,
      occl_leaf_sa_dim=sa_dim,
      occl_leaf_sa_heads=sa_heads,
      occl_leaf_sa_ff_mult=4,
      occl_leaf_sa_dropout=0.0,
    )
  )


def main() -> None:
  torch.manual_seed(0)
  device = torch.device("cpu")

  cfg = _make_cfg(sa_layers=1, sa_dim=128, sa_heads=4)
  model = OracleGTBaseline(cfg, node_classes=4, edge_labels=3).to(device)

  # Case A: mapping sanity via identity "transformer"
  print("[case A] identity mapping check")
  assert model.occl_leaf_sa is not None
  model.occl_leaf_sa = _IdentitySA()

  F = 5
  # Create variable number of tokens per fruit, including an empty fruit row.
  counts = torch.tensor([3, 0, 7, 1, 5], dtype=torch.long)
  pair_row = torch.repeat_interleave(torch.arange(F), counts)
  M = int(pair_row.numel())
  leaf_tok = torch.randn((M, 128), device=device)

  out = model._occl_leaf_set_encode(leaf_tok=leaf_tok, pair_row=pair_row, F=F)
  max_err = (out - leaf_tok).abs().max().item() if out.numel() else 0.0
  print(f"  M={M} F={F} max_abs_err={max_err:.3e}")
  if max_err != 0.0:
    raise RuntimeError("Packing/unpacking mismatch under identity SA.")

  # Case B: actual Transformer forward should run and produce finite outputs.
  print("[case B] real Transformer forward (finite outputs check)")
  cfg2 = _make_cfg(sa_layers=1, sa_dim=128, sa_heads=4)
  model2 = OracleGTBaseline(cfg2, node_classes=4, edge_labels=3).to(device)

  out2 = model2._occl_leaf_set_encode(leaf_tok=leaf_tok, pair_row=pair_row, F=F)
  finite = torch.isfinite(out2).all().item() if out2.numel() else True
  print(f"  out_mean={out2.mean().item():.4f} out_std={out2.std().item():.4f} finite={finite}")
  if not finite:
    raise RuntimeError("Non-finite outputs from leaf-set Transformer.")

  # Case C: full occlusion scorer forward shapes (uses leaf-set SA internally).
  print("[case C] occlusion scorer forward shapes")
  N = 32
  node_dim = 128
  h_node = torch.randn((N, node_dim), device=device)
  fruit_nodes = torch.tensor([3, 7, 11], dtype=torch.long, device=device)
  # 6 leaf candidates per fruit
  pair_f = fruit_nodes.repeat_interleave(6)
  pair_l = torch.randint(low=0, high=N, size=(pair_f.numel(),), device=device)
  pair_index = torch.stack([pair_f, pair_l], dim=0)
  M2 = int(pair_index.size(1))
  pair_geom2 = torch.randn((M2, 11), device=device)
  dir_geom2 = torch.randn((M2, 18, 2), device=device)

  pair_pot_logits, pair_rank_logits, union_logits = model2._occl_score_pairs_and_union_attn(
    h_node=h_node,
    fruit_nodes=fruit_nodes,
    pair_index=pair_index,
    pair_geom=pair_geom2,
    dir_geom=dir_geom2,
    attn_temp=1.0,
  )
  print(f"  pair_pot_logits: {tuple(pair_pot_logits.shape)} pair_rank_logits: {tuple(pair_rank_logits.shape)} union_logits: {tuple(union_logits.shape)}")
  if tuple(pair_pot_logits.shape) != (M2, 18):
    raise RuntimeError("Unexpected pair_pot_logits shape.")
  if tuple(pair_rank_logits.shape) != (M2, 18):
    raise RuntimeError("Unexpected pair_rank_logits shape.")
  if tuple(union_logits.shape) != (fruit_nodes.numel(), 18):
    raise RuntimeError("Unexpected union_logits shape.")

  print("OK")


if __name__ == "__main__":
  main()
