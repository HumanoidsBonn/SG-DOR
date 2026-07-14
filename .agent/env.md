# Environment Specification

## Python Environment

- **Conda Environment Name**: `bproc`
- **Python Version**: `3.10`
- **Package Manager**: `conda` (supplemented by `pip` for `blenderproc`)
- **CUDA Version**: `11.8`
- **PyTorch Version**: Supported via `conda-forge`/`pyg` channels in `environment.yml`

## PYTHONPATH

- Repository root: `/home/user/menon/workspace/software/pepper_scene_graphs`

## Environment Setup

- To create the environment:

  ```bash
  conda env create -f environment.yml
  ```

- To activate:

  ```bash
  conda activate bproc
  ```

## Run Commands

- Joint model training:

  ```bash
  python -m pepper_scene_graphs.main --dataset-path /path/to/scene_graphs.json --runs-root training --param lr=0.001 --param bs=64
  ```

- Semantic segmentation training:

  ```bash
  python pepper_scene_graphs/training/train_semseg.py
  ```

- Oracle training:

  ```bash
  python scripts/train_oracle.py --data_root /path/to/dataset --runs_root training/oracle --batch_size 32 --lr 0.001 --epochs 100
  ```

- Oracle ablation launch:

  ```bash
  python scripts/ablations/launch_oracle_ablation.py --key S2S0 --gpu 0
  ```
