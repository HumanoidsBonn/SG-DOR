
import os
import sys
import torch
from types import SimpleNamespace

# Add workspace to path
sys.path.append("/home/user/menon/workspace/software/pepper_scene_graphs")

from pepper_scene_graphs.data.oracle_dataset import OraclePepperDataset, oracle_collate_fn
from pepper_scene_graphs.models.oracle_joint_model import OracleGTBaseline
from torch.utils.data import DataLoader

def test_pipeline():
    data_root = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors"
    
    print("Initializing Table/Dataset...")
    dataset = OraclePepperDataset(
        data_root=data_root, 
        split='val', 
        npoint_per_instance=512,
        augment=False,
        merge_shoots=True,
        drop_json=False,
        voxel_size=0.01, # 1cm voxels for speed
        eps=0.015 # Adjusted eps accordingly
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=oracle_collate_fn)
    
    # 1. Test Dataset Loading
    print("Testing DataLoader...")
    batch = next(iter(loader))
    print(f"Batch loaded. Total nodes: {batch.x_organ.shape[0]}, Sampled points/node: {batch.x_organ.shape[1]}")
    print(f"Edge index shape: {batch.edge_index.shape}")
    print(f"Node labels shape: {batch.y_node.shape}")
    print(f"Edge labels shape: {batch.y_edge.shape}")
    if batch.y_edge.numel() > 0:
        print(f"Unique Edge Labels: {torch.unique(batch.y_edge).tolist()}")
    print(f"JSON Geom shape: {batch.json_geom.shape}")
    
    # 2. Test Model Instantiation
    print("\nInitializing OracleGTBaseline...")
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            oracle_msg_npoint=256,
            oracle_msg_radius_list=[0.01, 0.02, 0.04],
            oracle_msg_nsample_list=[16, 32, 64]
        )
    )
    
    # node_classes=6 (0..5), edge_labels=3 (placeholder)
    model = OracleGTBaseline(cfg, node_classes=6, edge_labels=3)
    
    # 3. Test Forward Pass
    print("Running Forward Pass...")
    out = model(batch)
    
    print("\nOutput Shapes:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
        
    print("\n[SUCCESS] Pipeline test passed.")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
