import torch
import numpy as np
import os
import sys

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from pepper_scene_graphs.data.pepper_dataset import PepperPointDataset

def verify_sampling(shoot_mode='merge'):
    data_root = '/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors'
    print(f"\n--- Verifying Sampling (Mode: {shoot_mode}) ---")
    
    dataset = PepperPointDataset(
        data_root=data_root,
        split='train',
        npoints=32768,
        shoot_mode=shoot_mode,
        use_bg=False,
        use_4mm=True
    )
    
    # Check first item
    data = dataset[0]
    sem = data['y_sem'].numpy()
    
    unique, counts = np.unique(sem, return_counts=True)
    dist = dict(zip(unique, counts))
    
    # Manual Mapping Check
    # 1:Stem (0), 2:Leaf (1), 3:Peduncle (2), 4:Fruit (3), 5:Shoot (4 if keep)
    class_names = {0:'Stem', 1:'Leaf', 2:'Peduncle', 3:'Fruit', 4:'Shoot'}
    
    print("Class Counts in Sample (Total 32768):")
    for c_id, count in dist.items():
        name = class_names.get(c_id, f"Class {c_id}")
        print(f"  {name}: {count}")

if __name__ == "__main__":
    verify_sampling(shoot_mode='merge')
    verify_sampling(shoot_mode='drop')
    verify_sampling(shoot_mode='keep')
