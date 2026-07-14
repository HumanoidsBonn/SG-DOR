Changelog
=========

2.2.0 (2026-07-14)
------------------
- **Public Release Preparation**:
  - Added MIT License to the repository for its public release.
  - Cleaned up the root directory of temporary files (including an accidental `~` folder, local CSV files, and screen logs).
  - Moved the local experimental script `audit_sampling_agg.py` to `scripts/analysis/audit_sampling_agg.py` to prevent root pollution.
  - Fully rewrote and standardized `README.md` formatting to remove outdated placeholders and add installation and utility scripts documentation.

2.1.9 (2026-02-18)
------------------
- **Oracle occlusion: split pair logits for continuous potential vs evidential rank**:
  - The model now emits two separate per-(fruit,leaf,dir) score tensors:
    - `occl_pair_potential_logits`: supervised by potential BCE regression; used for union pooling and consistency.
    - `occl_pair_rank_logits`: supervised by the listwise rank loss (and `--edl_occl_rank`).
  - Fixes a critical incompatibility where enabling `--edl_occl_rank` could prevent continuous potential regression from learning
    when both losses shared the same logits.
  - Validation `val/edl_occl_rank_{S_mean,u_mean}` now uses `occl_pair_rank_logits` (falls back to potential logits for older checkpoints).

2.1.8 (2026-02-18)
------------------
- **Oracle evidence: per-head EDL calibration metrics (W&B)**:
  - Added evidence-specific validation metrics for enabled EDL heads:
    - Node: `val/edl_node_{ece,brier,S_mean,u_mean,S_correct,u_correct,S_wrong,u_wrong}`
    - Edge existence: `val/edl_edge_exist_{S_mean,u_mean,S_correct,u_correct,S_wrong,u_wrong}` (ECE/Brier remain `val/edge_exist_{ece,brier}`)
    - Edge label: `val/edl_edge_label_{ece,brier,S_mean,u_mean,S_correct,u_correct,S_wrong,u_wrong}`
    - Occlusion potential: `val/edl_occl_potential_{ece,brier,S_mean,u_mean,S_correct,u_correct,S_wrong,u_wrong}`
    - Occlusion rank: `val/edl_occl_rank_{S_mean,u_mean}`
  - Motivation: make it easy to check that EDL uncertainty behaves properly (wrong predictions should have higher uncertainty).
- **Oracle occlusion: consistency loss is now EDL-aware**:
  - Added `--occl_consistency_mode {auto,mse,bce_dom,off}` (default: `auto`).
  - When `--edl_occl_potential` is enabled, `auto` switches the consistency term to a dominance-consistency BCE:
    - aggregate predicted per-leaf `p_mean` into `u_pred = 1 - Π(1 - p_leaf)`
    - supervise against `y_dom = 1[union >= --occl_union_tau]` (using the same stratified direction mask as potential BCE)
    - weight positive directions by the union explainability ratio to avoid punishing missing-leaf cases.
  - When EDL is not enabled for occlusion potential, `auto` keeps the original MSE-to-continuous-union regularizer.
- **Oracle evidence: decouple edge existence tuning from other heads**:
  - Added `--pos_weight_edge_exist` to override `--pos_weight` *for edge existence only*.
  - Added `--edl_kl_coeff_edge_exist` to override `--edl_kl_coeff` *for edge existence only* (uses the same warmup schedule).
  - Motivation: edge existence can be significantly more sensitive to EDL regularization and positive weighting than occlusion.

2.1.7 (2026-02-17)
------------------
- **Oracle occlusion: deployable leafset selection (no sidecar-priority leaf picking)**:
  - `occl_pair_index` leaf candidates are now selected by **geometry-only** heuristics by default (`--occl_leafset_policy geom_stratified`).
  - Candidate pools are built from **point-derived centroids** (and a point-derived size proxy) within `--occl_aug_max_dist`, then stratified by a simple size-vs-distance score to keep a diverse set under `--occl_max_leaves_per_fruit`.
  - Legacy behavior (prioritizing occl-sidecar-selected pos/neg leaves) remains available via `--occl_leafset_policy sidecar_priority`.
- **Oracle occlusion: stratified direction sampling for potential BCE (default)**:
  - The potential BCE loss no longer trains only on dominant directions by default.
  - New default `--occl_potential_dir_sampling stratified`: supervise all dominant dirs (union `>= --occl_union_tau`) plus a small sample of mid/low dirs (controls: `--occl_potential_union_tau_low`, `--occl_potential_n_mid_dirs`, `--occl_potential_n_low_dirs`).
  - `--occl_potential_dir_sampling dominant_only` reproduces legacy behavior (dominant dirs only).

2.1.6 (2026-02-17)
------------------
- **Oracle augmentations: configurable Z-rotation mode + restore point aug defaults**:
  - Added `--z_rot_aug_mode {none,continuous,discrete90}` to control the Z-rotation augmentation mode.
  - Z-rotation is now enabled by default (default: `continuous`); disable via `--z_rot_aug_mode none`.
  - Rotation mode no longer depends on whether occlusion supervision is enabled; it is controlled only by `--z_rot_aug_mode`.
  - Removed `--z_rot_aug` and `--no_train_augment` (to disable point augs, set `--color_dropout 0 --xyz_jitter 0 --point_dropout 0`).
  - Restored point-level augmentation defaults to `--color_dropout 0.25 --xyz_jitter 0.001 --point_dropout 0.1`.

2.1.5 (2026-02-17)
------------------
- **Oracle Z-rotation augmentation: fix occlusion target permutation**:
  - Fixed a bug where we permuted direction-indexed occlusion targets under Z-rotation augmentation.
  - The z-buffer sidecars define directions in a **fruit-local frame** (+X toward stem in XY, +Z up), so global Z-rotation augmentation rotates fruit and stem together and **does not change the direction index semantics**.
  - After this change, Z-rotation augmentation rotates points/normals/centroids but leaves occlusion targets in the canonical direction order (no permutation).

2.1.4 (2026-02-16)
------------------
- **Oracle occlusion: toggle per-fruit vs direction-aware leaf-set SA**:
  - Added `--occl_leaf_sa_mode {per_fruit,per_fruit_dir}` to switch the leaf-set Transformer encoder between:
    - `per_fruit`: run SA once per fruit and share contextualized leaf features across all directions (cheaper).
    - `per_fruit_dir`: run SA per (fruit, direction) with additive direction conditioning (more powerful, higher memory).
  - Default is `per_fruit_dir` (keeps current behavior).
- **Oracle occlusion: supervise more than the 4 focus fruits**:
  - Added an optional “extra supervised fruits” mode that can supervise additional fruits present in the sampled subgraph (beyond the occl-aug focus fruits).
  - Extra fruits are only supervised if their GT-dominant directions (union `>= --occl_union_tau`) remain dominant when considering **present leaves only** (estimated by summing exclusive / rank-1 sidecar mass over present leaves).
  - Motivation: the previous setup supervised only ~4 fruits per sampled subgraph, which can slow occlusion convergence substantially.
  - New CLI knobs: `--occl_supervise_max_fruits_per_scene`, `--occl_supervise_remain_rate_min`, `--occl_supervise_min_dom_dirs`.
- **Oracle occl-aug sampling: reserve nodes so occl-aug has room**:
  - Added `--occl_aug_reserve_nodes` to sample a smaller stem-biased BFS core (`subgraph_size - reserve`) and let occl-aug fill back up to `subgraph_size`.
  - This reduces the “dominant direction” recall ceiling caused by missing occluder leaves in the sampled subgraph.
  - Coverage script updates: `scripts/debug/analyze_oracle_subgraph_occlusion_coverage.py` now reports “dominant remain rate” (and focus-only variants) to quantify this ceiling directly.

2.1.3 (2026-02-15)
------------------
- **Oracle occlusion metrics: NDCG@K**:
  - Added ordering-aware graded ranking metrics `occl_leaf_ndcg_at_{1,3,5}` (and actionable variants) computed on dominant directions using k-buffer mass as relevance.
  - Intended to complement Mass@K by rewarding correct ordering within the top-K.

2.1.2 (2026-02-15)
------------------
- **Oracle occlusion: direction-aware leaf-set self-attention**:
  - Upgraded the leaf-set Transformer encoder from **direction-independent** (one leaf-set pass per fruit) to **direction-aware** (one pass per fruit *and* per occlusion direction).
  - Additive direction conditioning for each leaf token:
    - `tok = proj_base([h_leaf, pair_geom]) + proj_dir([depth_norm, lateral_norm]) + proj_emb(dir_emb)`
  - Implemented vectorized packing over `(fruit, dir)` groups (`F*D`) using padding + `src_key_padding_mask`, avoiding per-fruit/per-direction Python loops.
  - No new CLI flags; existing `--occl_leaf_sa_*` knobs now control this direction-aware encoder (recommended: `--occl_leaf_sa_layers 2`).

2.1.1 (2026-02-15)
------------------
- **Oracle occlusion: leaf-set self-attention (Transformer encoder)**:
  - Added an optional direction-independent Transformer encoder over the per-fruit candidate leaf set to enable leaf-leaf interactions (competition/redundancy) before direction-conditioned scoring and ranking.
  - Implemented with padding + `src_key_padding_mask` so all focus fruits in a batch can be processed together efficiently (avoids per-fruit Python loops).
  - New CLI knobs (default disabled): `--occl_leaf_sa_layers`, `--occl_leaf_sa_dim`, `--occl_leaf_sa_heads`, `--occl_leaf_sa_ff_mult`, `--occl_leaf_sa_dropout`.
  - Added `scripts/debug/test_occl_leaf_set_sa.py` to sanity-check the packing/unpacking logic and ensure finite outputs.
- **Occlusion retrieval metrics: oracle upper bounds + normalization**:
  - Added `oracle_*` and `*_norm` variants for leaf Recall@K and Mass@K metrics to make performance interpretable relative to the best possible under the current candidate set (including actionable variants).

2.1.0 (2026-02-15)
------------------
- **Oracle occlusion: attention pooling over leaf neighborhoods**:
  - Replaced the previous independent MLP occlusion heads with a direction-conditioned attention-style scorer that explicitly pools over leaf candidates per (fruit, direction) to predict the union occlusion signal.
  - Wired both `OracleGTBaseline` and `OracleResBaseline` through the shared `_forward_occl()` path (fixes the Res baseline occlusion path after the refactor).
- **Union loss: explainability weighting to avoid "impossible" supervision**:
  - Weighted the fruit-union BCE loss by how much of the GT union is explainable by the **present** candidate leaves (estimated by summing exclusive/rank-1 sidecar mass over the leaves in `occl_pair_index`).
  - This prevents systematically punishing the model for occlusion caused by leaves that are missing from the sampled subgraph.
- **Occlusion ranking target: k-buffer mass + mass@k metrics**:
  - Switched the listwise leaf ranking loss target from raw potential occlusion to **k-buffer mass** (`k_sum = sum_k occl_pair_k_rank[...,k]`, k=3), aligning supervision with “backup occluders” that remain problematic after removing the frontmost leaf.
  - Added GT-mass-weighted retrieval metrics `*occl_leaf_mass_at_{1,3,5}` (and actionable variants) to better track “top occluder” discovery quality than Recall@K alone.
- **Occlusion CLI**:
  - Added attention/ranking knobs: `--occl_attn_dim`, `--occl_attn_temp`, `--occl_rank_gamma`.
  - Changed `--occl_aug_num_fruits` default from 3 to 4.

2.0.3 (2026-02-15)
------------------
- **Oracle occlusion: fruit-local physics features**:
  - Updated `pepper_scene_graphs/models/oracle_joint_model.py` leaf-fruit occlusion scorer to use a fruit-local coordinate frame aligned with the z-buffer targets (+X toward nearest stem in XY with z forced to 0, +Z up).
  - Added point-derived size cues (extent + RMS radius from point clusters) and normalized pair geometry by fruit radius.
  - Added per-direction geometric conditioning (`depth_norm`, `lateral_norm`) into the direction-conditioned occlusion scorer.
- **Oracle occlusion logging: actionable threshold + epoch-mean losses**:
  - Added `--occl_potential_eps_actionable` (default 0.15) to compute extra leaf retrieval metrics under `train/occl_leaf_actionable_*` and `val/occl_leaf_actionable_*` without changing training gating (still uses `--occl_potential_eps`).
  - Added epoch-mean occlusion loss scalars `train/occl_loss_*_epoch` to make "flat vs learning" easier to interpret.

2.0.2 (2026-02-15)
------------------
- **Oracle pointcloud augmentations default OFF**:
  - Changed `scripts/train_oracle.py` augmentation defaults to be disabled unless explicitly enabled (avoids accidental train/val distribution shifts in `--no_geom_injection` mode).
  - Added `--z_rot_aug` to explicitly enable Z-rotation augmentation. Point-level augmentations are now opt-in via nonzero `--color_dropout`, `--xyz_jitter`, `--point_dropout` (and can still be force-disabled via `--no_train_augment`).
- **Occlusion learning: direction-conditioned scoring + ranking + consistency**:
  - Updated `pepper_scene_graphs/models/oracle_joint_model.py` to score leaf-fruit occlusion **per direction** using a learned direction embedding (`--occl_dir_embed_dim`) instead of treating direction as an implicit output index.
  - Added a **listwise ranking loss** for leaf occluder retrieval: for each (fruit, direction) on dominant directions (`union>=tau`), form a soft target distribution over leaves proportional to potential occlusion (after `potential>=eps` gating) and optimize cross-entropy against `softmax` scores.
  - Added a **leaf->union consistency regularizer**: aggregate predicted leaf occlusion probabilities into a union estimate (noisy-OR) and regress it towards the GT union (MSE), evaluated on dominant directions that have at least one leaf with `potential>=eps`.
  - Applied the agreed “dust” filtering in leaf supervision by clamping `potential<eps` targets to 0 in the leaf BCE loss.
  - Added CLI weights `--w_occl_rank` and `--w_occl_consistency` to tune the new objectives.

2.0.1 (2026-02-14)
------------------
- **Oracle occlusion supervision (pointcloud z-buffer sidecars)**:
  - Added optional occlusion target loading to `OraclePepperDataset` from the z-buffer sidecars produced in 2.0.0.
  - Dataset now emits batched tensors for occlusion training, including focus-fruit union targets and leaf-fruit candidate pairs with per-direction potential/exclusive/k-rank values.
  - Added occlusion-augmented sampling hooks so the same focus-fruit selection used for occl-aug can also anchor occlusion supervision (prevents “supervising leaves the subgraph doesn’t contain”).
- **Oracle model occlusion heads**:
  - Added lightweight occlusion heads to `pepper_scene_graphs/models/oracle_joint_model.py`:
    - fruit-direction head to predict union occlusion per direction
    - leaf-fruit-direction head to predict potential occlusion per direction (uses node embeddings + relative centroid geometry).
  - Added `LayerNorm` in the occlusion heads to stabilize initial logit scales.
- **Oracle training integration (losses + metrics + CLI)**:
  - Extended `scripts/train_oracle.py` with occlusion CLI flags and staged activation (`--occl_enabled`, `--stage_occl_epoch`, weights, thresholds).
  - Added occlusion losses:
    - union: BCE-with-logits against soft union targets in `[0,1]`
    - potential: BCE-with-logits on dominant directions only (`union>=tau`) to avoid flooding with trivial negatives.
  - Added occlusion metrics:
    - dominant-direction precision/recall/F1 + ECE/Brier
    - leaf retrieval `Recall@{1,3,5}` on dominant directions (positives via `potential>=eps`).
- **Augmentation safety (oracle pointcloud)**:
  - Kept z-rotation augmentation when occlusion supervision is enabled by restricting rotations to `{0, 90, 180, 270}` degrees and permuting direction-indexed occlusion labels accordingly.
  - Added safe point-level augmentations (train split only): point dropout (keeps at least one point per instance), xyz jitter, and color dropout (all CLI-configurable via `--point_dropout`, `--xyz_jitter`, `--color_dropout`).
  - Added `scripts/debug/analyze_oracle_point_spacing.py` to estimate point spacing statistics and choose a jitter scale grounded in the dataset.

2.0.0 (2026-02-14)
------------------
- **Pointcloud occlusion parameters (z-buffer)**:
  - Added `scripts/debug/analyze_pointcloud_occlusion_raster_params.py` to empirically select raster `pixel_size` and depth tolerance `delta` for pointcloud z-buffer occlusion targets (grounded in 4mm voxelized point clouds).
- **Pointcloud occlusion targets as binary sidecars**:
  - Added `scripts/dataset_generation/postprocess_pointcloud_occlusion_zbuffer.py` to compute pointcloud-based occlusion targets from existing `*_pc_4mm.npz` + `*_graph.json` without regenerating scenes.
  - Writes per-scene compressed `.npz` sidecars (keeps JSON lean) including:
    - fruit-level union occlusion per direction
    - leaf-fruit pair metrics per direction: potential occlusion (non-stealing), exclusive/frontmost occlusion, and k-buffer depth ranks (k=3)

1.9.2 (2026-02-13)
------------------
- **Evidential edge-existence stability (binary EDL)**:
  - Switched evidential edge-existence head to a tied-Beta/Dirichlet parameterization: predict a single logit `z` and form `alpha=[e(-z)+1, e(z)+1]`. This avoids the degenerate case where both class logits go very negative and `alpha≈[1,1]`, which caused no-learning / pathological high-recall behavior.
- **Thresholding semantics for evidential edge existence**:
  - Changed binary decision rule and threshold sweep comparisons from `>=` to strict `>` so exact ties (`p==0.5`) are treated as negative (important because `p==0.5` often corresponds to low-evidence priors in EDL).
  - Aligned the `pred_thresh=0.5` TP/FP/FN summary counts with the strict `>` rule (`fn` uses `<=`).
- **Eval-only reproducibility and richer diagnostics**:
  - Added `--eval-config {default,run}` for `--eval-only`. `run` auto-loads the stored `config.json` from the checkpoint’s run folder (found by walking upward from `--checkpoint-path`), preventing silent config drift when defaults change.
  - Extended eval artifacts with:
    - strict per-type metrics at the global best threshold (`edge_exist_per_type_at_best_thresh`)
    - strict per-type metrics at each type’s own best threshold (`edge_exist_per_type_at_own_best_thresh`)
    - grouped (stem-segment equivalence) metrics at the global best threshold (`edge_exist_group_*_at_best_thresh`)
    - grouped per-type metrics at each stem type’s own best threshold (`edge_exist_group_per_type_at_own_best_thresh`)
- **Occlusion helper decoupling**:
  - Removed hard dependency on `pepper_scene_graphs/training/occlusion.py` (missing in this branch) by providing guarded placeholders; training/eval no longer fail on import when occlusion is disabled.

1.9.1 (2026-02-12)
------------------
- **Stem-segment equivalence for edge supervision**:
  - Added representative-segment supervision mode (`edge_exist.stem_segment_equiv=True`) so stem-attached relations are trained at stem-group level instead of penalizing all sibling stem segments.
  - For each non-stem <-> stem-group candidate set, one representative candidate is kept positive/valid while sibling segment candidates are ignored in edge-existence loss (`valid_mask=False`).
  - Added synthetic representative relation-label fill when needed so edge-label supervision stays aligned.
- **Dataset/cache support for stem-group logic**:
  - Added `node_stem_group_id` construction from segmented stem IDs (strip `_seg\\d+`) in dataset conversion.
  - Added cache compatibility check: when stem-segment equivalence is enabled and cached data lacks `node_stem_group_id`, auto-rebuild from JSON.
- **Logging/eval updates for segment-equivalence analysis**:
  - Kept strict representative-based per-type edge-existence metrics (leaf--stem, peduncle--stem, peduncle--fruit) in train/val logging.
  - Added eval-only detailed grouped stem metrics (any segment of correct stem-group counts) and wrote them to `per_type_eval.json`/`metrics_eval.json` as `edge_exist_group_per_type` and `edge_exist_group_overall`.
  - Extended eval-only artifact export path (`training/<run>/eval/`) to consistently include structured summary JSON outputs for downstream analysis.

1.9.0 (2026-02-12)
------------------
- **Oracle -> OBB Port (Phase 1 Candidate Parity)**:
  - Updated edge candidate construction to Oracle-style union graph: `radius_graph ∪ knn_graph ∪ extra stem links`.
  - Added radius candidate controls (`dist_threshold`, `radius_max_neighbors`) and kept bidirectional dedupe.
  - Added XY-based extra nearest stems per non-stem (`extra_stems_per_node=2`) to improve stem-related candidate recall.
- **Oracle -> OBB Port (Phase 2 Semantic + Residual)**:
  - Added **local node head** (pre-GNN) and semantic injection modes (`node_feat_mode`: `none|local|gt`).
  - Added `detach_local_semantics` option for controlled gradient flow through semantic injection.
  - Switched default backbone to **ResGINE** and enabled residual message passing by default.
  - Enabled node-head identity persistence via concatenation of raw node features with refined GNN embeddings (`node_identity_concat=True`).
  - Added explicit local-node loss integration (`loss_weights.node_local`) for Oracle-style two-stage node supervision.
- **Oracle -> OBB Port (Phase 3 Curriculum)**:
  - Added staged curriculum config block (`curriculum`) with per-task activation epochs:
    - `stage_gnn_epoch`
    - `stage_edge_label_epoch`
    - `stage_occlusion_epoch`
  - Integrated epoch-dependent curriculum multipliers into train/eval total loss composition.
  - Kept local-node loss active from epoch 1; delayed downstream heads by curriculum stage.
  - Removed temporary curriculum telemetry keys from final logging after validation of behavior.
- **Defaults aligned for structure-learning runs**:
  - `heads`: node/edge-exist/edge-label enabled, occlusion disabled by default.
  - `evidence`: disabled by default for this porting baseline.
  - `type_constraint.enabled=False` by default to reduce hand-crafted filtering and stay closer to Oracle candidate behavior.

1.8.0 (2026-02-11)
------------------
- **GNN Elite Architecture (Oracle v12)**:
  - **Identity Concatenation**: Implemented geometric feature persistence by concatenating original PointNet++ features with GNN-refined features.
  - **ResGINEBackbone**: Added residual skip-connections and layer-unique MLPs to prevent node identity over-smoothing.
- **Proposal v12 Restoration**:
  - **LR/Jitter Tuning**: Fixed 10x Learning Rate deficit (2e-3) and reduced spatial jitter (1mm) to stabilize botanical organ discovery.
- **Oracle V12 (GPU-Native)**: Migrated instance masking and cluster extraction from CPU to GPU using sort-based vectorized sampling.
- **Throughput Milestone**: Achieved ~4.5 scenes/sec training throughput (BS=32, n=256, s=64).
- **Edge Hashing**: Implemented vectorized GPU-side hashing for collision-free edge matching in $O(1)$ time.
- **Botanical Logic**: Integrated Shoot-to-Leaf merging directly into GNN targets for topological stability.


1.7.0 (2026-02-09)
------------------
- **Phase 13 (Decoupled Geometry)**: Refactored offset prediction into independent Direction (normalized unit vector) and Magnitude (dist) heads. Integrated Softplus activations and 5mm directional masking.
- **Merged Topology (V6/V4)**: Established consolidated class hierarchy, yielding a **+10.2% mIoU boost** in zero-color robustness.
- **Robust Transfer Learning**: Added support for selective weight loading across disparate class topologies (e.g. 5-class pre-trained to 4-class production).
- **Phase 12 (Relative Geometry)**: Implemented Log-Space Extent regression and Weighted Absolute Offset loss for scale-invariant error minimization.
- **Phase 2.1 (Full Scene Inference)**: Introduced `full_scene_inference.py` for block-wise processing of dense cloud scenes.
- **Automation**: Added `antigravity_watchdog.py` for training log monitoring.

1.6.0 (2026-02-09)
------------------
- **Phase 11: Multi-Head Instance Segmentation**:
  - Implemented `PointNet2MultiHead` architecture fork in the decoder for simultaneous Semantic, Offset, and Extent prediction.
  - Developed `train_instance_seg.py` trained with multi-task loss ($L_{sem}$ + 10$L_{off}$ + $L_{ext}$).
  - Refactored `PepperPointDataset` to provide per-point ground truth instance centroids and AABB dimensions.
  - Enhanced validation with confusion matrices and regression error metrics (L2, L1, Cosine).

1.5.0 (2026-02-08)
------------------
- **Phase 9: Robustness Watchdog**: Implemented automated monitoring and auto-tuning for geometric verification at Epoch 20.
- **Phase 10: Heavy-Jitter Experiments**: Added support for configurable color dropout and launched 60%/80% dropout trials.
- **Biological Color Integration**: Transitioned from grey-scale viewport signatures to realistic biological RGB ranges for all plant organs.
- **Dataset Surgery**: Implemented `update_dataset_colors.py` for efficient per-instance color remapping in existing NPZ datasets.
- **Weak-Signal Augmentation**: Added dynamic color augmentation to `PepperPointDataset` (Hue/Saturation jitter, contrast, and segment-level dropout) to force geometric feature learning.
- **Class Consolidation**: Integrated support for merging "Shoot" (ID 5) into "Leaf" (ID 2) to simplify segmentation tasks.
- **4-Class Baseline**: Completed 100-epoch training run for consolidated organ model, achieving 50.5% mIoU.

1.4.0 (2026-02-06)
------------------
- **Industrial Generation Pipeline**: Developed a high-volume, headless pipeline for generating 500+ scenes with parallel driver support.
- **Integrated extraction**: Implemented in-memory point cloud extraction (npz) directly in Blender driver, eliminating .blend disk overhead (~93% space saved).
- **ID Sync**: End-to-end integer ID synchronization (instance/semantic) across scene graphs, point clouds, and metadata.
- **Occlusion Learning**: Fully integrated "Union-of-Shadows" logic and Hybrid Node-Level storage.
- **Diagnostics**: Completed terminal piping diagnostic plan; optimized agent visibility by relocating memory files to root.
- **Rules**: Formalized agent memory persistence and stable terminal patterns (file-redirection protocol) in workspace rules.


1.3.0 (2026-02-05)
----------
- Added `generate_annotated_pointcloud.py` for dense (44M+ pts) annotated point cloud extraction from Blender scenes.
- Added `analyze_dimensions.py` for automated plant component metric (L/W/T) reporting from scene graphs.
- Implemented non-uniform stem scaling (`stem_width_scale`, `stem_height_scale`) in `scene_generation_obb.py` to achieve realistic 1-2m plants.
- Added proportional scaling for attachment points to maintain branch/fruit alignment during non-uniform stem deformation.
- Decoupled `fruit_scale_range` and `leaf_scale_range` for independent category-level size control.
- Added `visualize_cloud.py` for high-throughput point cloud inspection using Glasbey-style instance coloring and voxel downsampling.
- Integrated "shoots" into the point cloud extraction pipeline with Semantic ID 3.
- Optimized point cloud extraction with label-preserving per-instance voxelization (fixing collision bugs), oriented normal estimation, float32/uint dtypes, and `np.savez_compressed` for ~10x storage efficiency.
- Established a persistent documentation protocol with `DATASET_GENERATION.md` and automated summary logs.


1.2.5 (2026-02-03)
----------
- Added LightGBM dependency to the environment specification.
- Added LightGBM occlusion baseline (scene-split training, explicit per-direction 3-bin targets) with dedicated config and script.
- Implemented two-stage LightGBM training (any-vs-none + high-vs-low) with capped pos-weights and merged 3-class probabilities.
- Added feature importance exports for Model A and Model B.
- Added coarse→refine threshold tuning for two-stage (t_any/t_hi) with stored best thresholds and metrics.
- Added eval-curve logging (JSONL) and run-folder report PNGs (eval curve, confusion matrix, class metrics, thresholds, per-direction F1).
- Added overlap-inspired geometric features (inter_norm / overlap_flag / IoU) and overlap feature summary stats for GT none vs non-none.
- Added dead-zone filtering for Model A (drop/downweight) with logged removal fraction.
- Added per-stage min_data_in_leaf overrides so Model A/B can be tuned independently.
- Added projected OBB corner (u,v) features per direction (32 dims) with fruit-radius normalization and direction one-hot encoding.


1.2.4 (2026-02-03)
----------
- Removed backbone-embedding concat for occlusion head.
- Added focal loss option for occlusion classification (configurable alpha/gamma; ignored when evidential occlusion is enabled).


1.2.3 (2026-02-03)
----------
- Added per-direction `none_keep_prob` masking and masked occlusion loss support.
- Added masked occlusion metrics and post-mask bin distribution logging.
- Simplified occlusion metrics to overall precision/recall/F1 (removed any/per-direction breakdowns).
- Added optional backbone-embedding concat for occlusion head (configurable).
- Added debug scripts for occlusion value histograms, extent histograms, and tiny dataset slicing.

1.2.2 (2026-02-02)
----------
- Added linear KL warm-up schedule for evidential losses (epoch-based scaling via config).
- Added class-weighted evidential CE for occlusion bins (weights apply to data-fit term only).
- Fixed duplicate `dir` argument in `wandb.init` after run-folder changes.
- Switched occlusion to explicit per-direction 3-bin targets and added direction-aware geometric features for the occlusion head.

1.2.1 (2026-02-01)
----------
- Collapsed occlusion targets to 5-class bins per axis (none/+low/+high/−low/−high) to respect ± exclusivity.
- Added candidate-stage visualization script with per-fruit stats JSON for occlusion sampling (base/ring/far, thinning, none-keep).
- Fixed occlusion candidate extra-negative count computation for float ratios.
- Fixed metrics history JSONL export to handle tensor values safely.
- Logged occlusion precision/recall/F1 and per-axis metrics to W&B.
- Cached occlusion candidate pools per scene to avoid per-epoch distance recomputation.
- Vectorized occlusion candidate assembly using cached pools and aligned values for faster runtime.
- Added binary none-vs-any occlusion TP/FP/TN/FN metrics.
- Added occlusion cache sanity-check script to validate pool consistency and value alignment.

1.2.0 (2026-02-01)
----------

- Added occlusion prediction head for leaf-fruit pairs with per-direction 3-bin classification.
- Introduced occlusion candidate sampling (base radius + ring/far negatives + per-epoch thinning) and target binning config.
- Added occlusion loss with optional evidential mode and integrated occlusion loss weighting into training/eval loops.
- Added occlusion data loading (`occl_pairs`, `occl_raw`, `occl_weighted`) and dropout-safe remapping.
- Added occlusion geometry feature embedding (fruit-local frame, leaf axes/extents, distance cues) and occlusion head MLP.
- Added `pepper_scene_graphs/training/occlusion.py` to encapsulate occlusion candidates, binning, and metrics.
- Added occlusion distribution analysis script (raw/weighted × axes3/axes9) for threshold selection.
- Added occlusion candidate downsampling for all-none pairs (`none_keep_prob`) and improved weight handling in stats output.



1.1.2 (2026-02-01)
----------

- Switched occlusion overlap to silhouette (convex hull) polygons and aligned visualization with the metric.
- Added occlusion silhouette postprocess script for existing datasets.

1.1.1 (2026-02-01)
----------
- Added stem segmentation postprocess and visualization scripts.
- Updated stem segmentation postprocess CLI to derive segment length from cutoff height and segment count, and to preserve original stem axes/orientation while adapting extents.
- Apply segment length along the stem’s main extent axis instead of assuming index 0.
- Export per-epoch metrics history to JSONL and copy into the wandb run folder for offline analysis.

1.1.0 (2026-01-30)
------------------

- Implemented GINE (Graph Isomorphism Network with Edge features) backbone support.
- Added geometric edge feature calculation (relative position, distance) for backbone messaging.
- Enforced workspace rules: comprehensive docstrings, type hints, and inline comments for complex logic.
- Documented all configuration parameters in `config.py` for research transparency.
- Added `scripts/download_datasets.py` for automated dataset acquisition and recursive unzipping.
- Created `docs/installation.md` with comprehensive environment (Python 3.10) and data setup instructions.
- Added `environment.yml` for streamlined Conda environment setup.
- Implemented automatic `.pt` binary caching and `ujson` integration for dataset loading, achieving ~400x speedup on subsequent runs.
- Completed comprehensive documentation sprint: 100% compliance with docstring, type hint, and inline comment policies.
- Updated repo structure with `.agent/rules.md` and `docs/RECENT_COMMITS.md`.
- **Critical Fix**: Decoupled graph geometry (`pos`) from input features (`x`) in `transforms.py` to prevent structural collapse when auxiliary features are disabled.
- **Performance**: Vectorized `build_edge_targets` (removing Python loops) and registered `allowed_type_matrix` as a GPU buffer.
- Refactored `trainer.py` to centralize node loss logic and remove duplication.
- Enforced strict determinism (CuDNN) in `seed.py`.

1.0.0 (2026-01-30)
------------------

- Split the original `train_joint_sgg.py` monolith into a `pepper_scene_graphs` package with dedicated configs, data loaders, models, and training modules.
- Added CLI parameter overrides, structured WandB logging (project `pepper_scene_graphs`), per-run `config.json`, and suffix-based run naming.
- Introduced calibration-aware evaluation metrics (ECE/Brier), confusion matrices, prediction dumps, and k-NN hit rate reporting.
- Built analytics scripts (`analyze_noise_stats`, `visualize_graph`, `visualize_graph_3d`) and dataset helpers to reuse the new modular interfaces.
- Registered GraphSAGE/GINE backbones via an import-time registry to ensure they are discoverable when instantiating the model.
