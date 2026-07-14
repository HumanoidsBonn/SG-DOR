# Dataset Generation Guide

This guide describes the pipeline for generating procedural pepper plant scenes and extracting annotated 3D point clouds. It covers both single-scene generation (for debugging) and the high-efficiency parallel pipeline (for large-scale dataset creation).

## 🚀 Quick Start (Single Scene)

To generate a single scene and extract its point cloud for debugging:

### 1. Scene Generation
Generate a procedural Blender scene with randomized stems, leaves, and fruits:
```bash
blenderproc run scripts/dataset_generation/scene_generation_obb.py \
    /path/to/template.blend \
    --out output/scene.blend \
    --graphs_out output/scene_graph.json
```

### 2. Annotated Point Cloud Extraction
Extract world-space vertices with semantic and instance labels from the `.blend` file:
```bash
blenderproc run scripts/dataset_generation/generate_annotated_pointcloud.py \
    --scene output/scene.blend \
    --json output/scene_graph.json \
    --out output/cloud.npz \
    --voxel_size 0.005  # Highly recommended: 5mm voxelization
```

### 3. Visualization
Visualize the instance point cloud with Glasbey colors and downsampling:
```bash
python3 scripts/dataset_generation/visualize_cloud.py output/cloud.npz
```

---

## 🏭 Industrial Generation Pipeline (Large-Scale)

The industrial pipeline is optimized for high-volume dataset generation (e.g., 500+ scenes) on servers. It integrates point cloud extraction directly into the generation loop and eliminates intermediate `.blend` files to save disk space (~2GB per scene).

### 🚀 Running Parallel Generation
Use the `run_chunked_generation.py` driver to manage large runs:
```bash
python scripts/dataset_generation/run_chunked_generation.py \
    --scene path/to/template.blend \
    --total-scenes 500 \
    --chunk-size 25 \
    --workers 4 \
    --extract-pc \
    --no-blend \
    --run-name MyExperiment \
    --scenes-root /path/to/output/
```

### Key Arguments
| Flag | Description |
| :--- | :--- |
| `--total-scenes` | Total number of scenes to generate. |
| `--workers` | Number of parallel Blender instances to run. |
| `--extract-pc` | **CRITICAL**: Enables integrated Point Cloud extraction. |
| `--no-blend` | **CRITICAL**: Skips saving large `.blend` files (highly recommended for large runs). |
| `--chunk-size` | Number of scenes per Blender process (prevents memory leaks). |

---

## ⚙️ Configuration & Tuning

The generation process is highly tunable via the `CONFIG` dictionary in `scene_generation_obb.py`, or by passing a `--config_override` JSON file:
- **`stem_width_scale`**: `(0.15, 0.20)` -> Controls stem thickness (~1.5cm).
- **`stem_height_scale`**: `(0.30, 0.50)` -> Controls stem height (1-2m).
- **`fruit_scale_range`**: `(0.8, 1.3)` -> Independent scale range for pepper fruits.
- **`leaf_scale_range`**: `(0.5, 1.2)` -> Independent scale range for leaves.
- **Density**: `num_stems_range`, `placements_per_attachment`.
- **RNG & Logic**: `seed`, `collision_enabled`, `occlusion_enabled`.

---

## 🏷️ Semantic Mapping & IDs

| Class | ID | Description |
| :--- | :--- | :--- |
| **Background** | 0 | Unlabeled space. |
| **Stem** | 1 | Main vertical plant branch. |
| **Leaf** | 2 | Foliage / photosynthetic organs. |
| **Shoot** | 3 | Emerging leaves / stems. |
| **Peduncle** | 4 | Small stem connecting pepper to the main branch. |
| **Fruit** | 5 | Sweet pepper fruit. |

*Note: The datasets class mapping can be simplified in the `PepperPointDataset` loader (e.g., merging Shoots into Leaves for a 4-class configuration).*

---

## 📦 Data Formats & Output Structure

### Point Cloud Format (`.npz`)
The point cloud is saved as a compressed NumPy archive (`np.savez_compressed`) containing:
- **`points`**: `(N, 3) float32` -> World-space coordinates.
- **`semantics`**: `(N,) uint8` -> Class labels (see table above).
- **`instances`**: `(N,) uint16` -> Unique integer per object instance in the scene.
- **`normals`**: `(N, 3) float32` -> Estimated surface normals.

### Output Directory Structure
```text
run_name_TIMESTAMP/
├── scene_graphs.json          # Master merged graph
├── chunks/                    # Temporary chunk files (deleted after merge)
└── data/                      # Stable Dataset
    ├── scene_00000.json       # Graph + Metadata + Full Config
    ├── scene_00000.npz        # Point Cloud (points, normals, semantics, instance)
    ├── scene_00001.json
    └── scene_00001.npz
```

---

## 📊 Verification & Auditing

After generating a dataset, check its integrity using the programmatic audit script:
```bash
python scripts/dataset_generation/verify_industrial_output.py /path/to/output/MyExperiment_TIMESTAMP/
```

The script performs the following validation checks:
- **Consistency**: Asserts individual files have matching `.json` and `.npz` pairs.
- **Data Integrity**: Correlates point counts in `.npz` with metadata in the `.json` file.
- **Traceability**: Ensures the generation `CONFIG` dictionary is properly embedded in every scene file.
