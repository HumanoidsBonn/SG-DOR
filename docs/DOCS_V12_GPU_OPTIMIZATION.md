# Oracle V12: GPU-Native Optimization & Fast Mode
**Date**: 2026-02-11
**Objective**: Eliminate CPU bottlenecks, optimize training throughput, and stabilize the Oracle ceiling-check pipeline.

## 1. GPU-Native Pipeline Architecture
The core bottleneck was identified as **Instance Masking & Sampling** on the CPU. The pipeline was migrated to be "GPU-Native":
- **Dataset**: Instead of looping through instances on the CPU, the dataset now returns a raw subgraph point cloud and a mapping tensor (`node_id_per_point`).
- **Model**: Added `_extract_node_samples` in `OracleGTBaseline`. It uses vectorized CUDA operations (`torch.sort`, `torch.bincount`, modulo-indexing) to extract sampled organ clusters directly in VRAM.
- **Batching**: Implemented `OracleData(Data)` with custom `__inc__` to allow proper PyTorch Geometric batching of the cluster-to-point mappings.

## 2. Speed Optimizations ("Fast Mode")
Significant gains in training throughput were achieved by optimizing hyper-parameters and graph complexity:
- **Subgraph Size**: Reduced from 128 to **64 nodes** per scene (sufficient for botanical structure).
- **Point Resolution**: Lowered from 512 to **256 points per organ** (faster PointNet++ encoding).
- **Edge Pruning**: Reduced `dist_threshold` to **0.15m** + kNN=8 to keep the GNN adjacency matrix sparse.
- **Batching**: Increased total batch size to **32 scenes** on 48GB VRAM cards.
- **Result**: Throughput increased from ~1 scene/sec to **~4.5 scenes/sec** (~7s/it at BS=32).

## 3. Proposal Training Curriculum (v12)
Restarted the Proposal training run with a more stable curriculum for high-resolution refinement:
- **Extended training**: 150 epochs total.
- **Semantic-only phase**: Increased to 20 epochs to stabilize the backbone.
- **Stage 1 (No Extents)**: Extended to 50 epochs.
- **Hyperparams**: Doubled batch size (32) and learning rate (2e-4).

## 4. Mathematical Verifications
- **Hashing**: Implemented $u \cdot N + v$ hashing for $O(1)$ GPU-side edge matching.
- **Sampling**: Implemented "Wrap-around" vectorized sampling to handle small organs (e.g., tiny peduncles) without bias.
- **Consistency**: Verified Stem-Leaf-Shoot remapping across the entire GNN pipeline.
