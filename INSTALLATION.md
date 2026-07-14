# Installation Guide

Follow these steps to set up the environment and datasets for the `pepper_scene_graphs` project.

## 1. Environment Setup

We recommend using **Conda** to manage your Python environment. This ensures compatibility and isolates dependencies.

```bash
# Create a new conda environment from the provided YAML file
conda env create -f environment.yml
conda activate pepper_sg
```

## 2. Dependency Installation

Install the required packages. Note that you may need to install the correct version of PyTorch and PyTorch Geometric depending on your hardware (CPU vs CUDA).

```bash
# Install PyTorch (example for CUDA 11.x)
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric

# Install other requirements (if any)
pip install -e .
```

## 3. Dataset Setup

A script is provided to automate the download and extraction of the datasets.

```bash
# Run the setup script
python scripts/download_datasets.py
```

This script will:
1. Verify if the data already exists in the `datasets/` folder.
2. Download the dataset archive from the Bonn University cloud.
3. Recursively unzip all contents into the correct directory structure.

## 4. Verification

To verify the installation, you can run a dry-run training epoch:

```bash
python -m pepper_scene_graphs.main \
  --dataset-path datasets/processed/occlusionData_20260109_1946/scene_graphs.json \
  --runs-root training \
  --param epochs=1 \
  --param backbone=gine
```
