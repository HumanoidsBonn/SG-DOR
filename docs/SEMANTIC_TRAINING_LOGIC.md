# Pepper Plant Semantic Segmentation: Training Logic & Design

This document outlines the architecture, data pipeline, and technical decisions behind the PointNet++ training for semantic segmentation of pepper plant point clouds.

## 1. Goal
Classify every point in a scene into one of 5 semantic categories:
- **ID 1 (0): Stem**
- **ID 2 (1): Leaf**
- **ID 3 (2): Peduncle**
- **ID 4 (3): Fruit**
- **ID 5 (4): Shoot**

Target input is a high-density scene (~400k-1M points) with XYZ and Normals (6D).

## 2. Model Architecture: PointNet++ (SSG)
We use the **Set Abstraction (SA)** and **Feature Propagation (FP)** framework from PointNet++.

### Key Design Choices
- **Decoupled Sampling (Speed vs. Quality)**:
    - Farthest Point Sampling (FPS) is $O(N \cdot S)$. For $N=600k$, this is prohibitively slow on CPU/Basic PyTorch.
    - **Logic**: We randomly subsample to $N=16,384$ points to pick centroids via FPS. However, once centroids are picked, we query neighbors (`BallQuery`) from the **entire original point cloud**.
    - **Result**: FPS remains fast, but local features are built using the full available density.
- **SSG (Single-Scale Grouping)**:
    - Chosen for simplicity and speed in the initial phase.
    - Radii $(0.02, 0.05, 0.1, 0.2)$ are tuned for the scale of pepper plants (where fruits are ~5-10cm).

## 3. Data Pipeline & Logic
The `PepperPointDataset` handles complex preprocessing to ensure stable training.

### Data Flow
1. **Memory Efficiency**: Uses `mmap_mode='r'` to read `.npz` files without full RAM spikes.
2. **Indexing Optimization**: Points are indexed *before* conversion to NumPy arrays/Tensors to minimize copying.
3. **Centering**: Points are centered at $(0, 0, 0)$ per scene. We **exclude** unit-sphere normalization to preserve metric scale (meters), which is important for fixed-radius BallQuery.
4. **Resampling**: Scenes are fixed to $N=8,192$ points for consistent batching.

### Label Mapping & Safety
- **Underflow Protection**: Raw labels (uint8) are cast to `int64` before subtraction (`semantics - 1`). This prevents `0` (background) from wrapping around to `255`.
- **Mapping**: By default, we map 1-5 to 0-4 to fit exactly into a 5-class model output, which is standard for PyTorch `CrossEntropyLoss`.

### Augmentation (Train Only)
Applied to simulate real-world sensor noise and variability:
- **Z-Rotation**: Full 360-degree rotation.
- **Jitter**: 2mm Gaussian noise (matching our high-density generator resolution).
- **Scaling**: 0.9x to 1.1x isotropic scaling.
- **Dropout**: Randomly removes up to 30% of points (simulating occlusion).
- **Normal Re-normalization**: Normals are re-normalized to unit length after rotation/scaling to maintain geometric validity.

## 4. Training Strategy
- **Weighted Loss**: We calculate inverse-frequency weights per class because 'Leaf' points usually dominate 'Fruit' or 'Shoot' points.
- **Metric**: **mIoU (mean Intersection over Union)** is the primary metric, as it accounts for class imbalance better than raw accuracy.
- **Scheduler**: `CosineAnnealingLR` for smooth convergence over 100 epochs.

## 5. Environment Requirements
- **Python 3.8+** (Type hints fixed for 3.8 compatibility).
- **PyTorch 2.0+** with CUDA 11.7+.
- **Data Location**: `/home/data/menon/data_pepper_scenes/output_scenes/...`
