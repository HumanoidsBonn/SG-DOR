# scripts/debug/audit_oracle_dataset.py
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
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=oracle_collate_fn, num_workers=4)
    
    node_counts = {i: 0 for i in range(6)}
    edge_counts = {i: 0 for i in range(3)}
    total_nodes = 0
    total_edges = 0
    
    print(f"\nAuditing {split} split...")
    for i, batch in enumerate(tqdm(loader)):
        for l in batch.y_node:
            node_counts[l.item()] += 1
            total_nodes += 1
        for l in batch.y_edge:
            edge_counts[l.item()] += 1
            total_edges += 1
        if i > 20: break # Just check first 21 batches
            
    print(f"Results for {split}:")
    print(f"  Total Nodes checked: {total_nodes}")
    print(f"  Node Label Distribution: {node_counts}")
    print(f"  Total Edges checked: {total_edges}")
    print(f"  Edge Label Distribution: {edge_counts}")

if __name__ == "__main__":
    audit('train')
    audit('val')
