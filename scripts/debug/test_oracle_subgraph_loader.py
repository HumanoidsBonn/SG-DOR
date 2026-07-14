import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from termcolor import colored

# Add repo root to path
sys.path.append(os.getcwd())
from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors')
    parser.add_argument('--subgraph_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    print(colored("Initializing OraclePepperDataset with Subgraph Sampling...", "cyan"))
    
    ds = OraclePepperDataset(
        data_root=args.data_root,
        split='train',
        subgraph_enabled=True,
        subgraph_size=args.subgraph_size,
        npoint_per_instance=256, # Reduced for speed in test
        hard_pair_constraint=True
    )
    
    print(f"Dataset size: {len(ds)}")
    
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=oracle_collate_fn)
    
    print(colored("Iterating DataLoader...", "cyan"))
    
    for i, batch in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"  Batch Keys: {batch.keys}")
        
        # Check Node Counts
        # y_node is [N_nodes]
        y = batch.y_node
        num_nodes = y.shape[0]
        
        # Check Stem Count (Stem is 0 after remapping)
        num_stems = (y == 0).sum().item()
        
        # Check Instance sizes
        # x_organ is [N_nodes, P, 6]
        x = batch.x_organ
        
        avg_nodes_per_graph = num_nodes / args.batch_size
        avg_stems_per_graph = num_stems / args.batch_size
        
        print(f"  Total Nodes: {num_nodes} (Avg {avg_nodes_per_graph:.1f}/graph)")
        print(colored(f"  Total Stems: {num_stems} (Avg {avg_stems_per_graph:.1f}/graph)", "yellow"))
        print(f"  Feature Shape: {x.shape}")
        
        if i >= 2: break
        
    print(colored("\nSubgraph Integration Test PASSED", "green"))

if __name__ == "__main__":
    main()
