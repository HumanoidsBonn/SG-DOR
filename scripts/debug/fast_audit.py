# scripts/debug/fast_audit.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Ensure repository root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn

def audit(split):
    data_root = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors"
    ds = OraclePepperDataset(data_root, split=split, augment=False)
    # Check only first 5 indices
    ds.indices = ds.indices[:5]
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=oracle_collate_fn, num_workers=0)
    
    node_counts = {}
    edge_counts = {}
    
    print(f"\nAuditing {split} split (5 scenes)...")
    for batch in tqdm(loader):
        for l in batch.y_node:
            val = l.item()
            node_counts[val] = node_counts.get(val, 0) + 1
        for l in batch.y_edge:
            val = l.item()
            edge_counts[val] = edge_counts.get(val, 0) + 1
            
    print(f"Results for {split}:")
    print(f"  Node Label Distribution: {node_counts}")
    print(f"  Edge Label Distribution: {edge_counts}")

if __name__ == "__main__":
    audit('train')
    audit('val')
