# scripts/debug/oracle_sanity_checks.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from types import SimpleNamespace

# Ensure repository root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn
from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline

def run_checks():
    data_root = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors"
    ds = OraclePepperDataset(data_root, split='val', augment=False)
    # Just take 5 scenes
    ds.indices = ds.indices[:5]
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=oracle_collate_fn)
    
    print("\n=== STARTING ORACLE SANITY CHECKS ===\n")
    
    all_node_labels = []
    all_instance_ids = []
    all_edge_labels = []
    
    for batch in loader:
        all_node_labels.append(batch.y_node.numpy())
        # We need to find the original instance IDs. 
        # The dataset doesn't store them in the batch currently.
        # Let's check the first scene manually
        break

    batch = next(iter(loader))
    print(f"Batch has {batch.num_nodes} nodes and {batch.edge_index.shape[1]} edges.")
    
    # Check B1: Candidate edges vs GT edges
    # In current implementation, edge_index IS sourced from GT.
    print(f"Check B1: Edge Existence Triviality")
    print(f"  edge_index count: {batch.edge_index.shape[1]}")
    # In a real scenario, we'd have candidates. Here every edge we see IS a GT edge.
    
    # Check B3: Edge label distribution
    print(f"Check B3: Edge Label Distribution")
    edge_counts = torch.bincount(batch.y_edge)
    for i, count in enumerate(edge_counts):
        print(f"  Label {i}: {count.item()}")

    # Check D: Indexing
    print(f"Check D: Indexing Alignment")
    print(f"  First 5 edges: \n{batch.edge_index[:, :5]}")
    print(f"  First 5 edge labels: {batch.y_edge[:5]}")
    
    # Check A2: Instance ID Correlation (Simulation)
    # Let's peek at the first item's raw data
    item = ds[0]
    print(f"Check A2: Instance ID Correlation for first scene")
    # item is now a dictionary
    labels = item['node_y'].numpy()
    print(f"  Node Labels (first 50): {labels[:50]}")
    # Check if they are sorted
    is_sorted = all(labels[i] <= labels[i+1] for i in range(len(labels)-1))
    print(f"  Labels are sorted (lower is better): {is_sorted}")
    
    # Check B1: Pos rate
    # If we had a kNN candidate generator, we'd compare. 
    # For now, let's just confirm that ALL edges in the batch are positive.
    print(f"Check B1: Pos Rate (should be 1.0 for oracle pure edges)")
    # Since we don't have negatives in this dataset yet:
    print(f"  Current edge_index only contains GT positives.")
    
    print("\n=== ZERO-INFORMATION TEST (Simulation) ===")
    # We will simulate this by zeroing out the batch in a mock forward.
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            oracle_msg_npoint=128,
            oracle_msg_radius_list=[0.01, 0.02, 0.04],
            oracle_msg_nsample_list=[16, 32, 64]
        )
    )
    model = OracleGTBaseline(cfg, node_classes=6, edge_labels=3)
    
    print("Testing if model can run on zeros...")
    batch.x_organ.zero_()
    batch.n_organ.zero_()
    batch.json_geom.zero_()
    
    try:
        out = model(batch)
        print("Model ran successfully on ZERO inputs.")
        print(f"Node logits shape: {out['node_logits'].shape}")
    except Exception as e:
        print(f"Error running model on zeros: {e}")

if __name__ == "__main__":
    run_checks()
