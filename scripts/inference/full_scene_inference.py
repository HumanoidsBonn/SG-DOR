import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import sys
import time

# Ensure repository root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from pepper_scene_graphs.models.pointnet2_multihead import PointNet2MultiHead

def full_scene_inference(model, points, normals, colors=None, npoints=8192, overlap=0.2, device='cuda'):
    """
    Processes a high-density point cloud by tiling it into blocks.
    Args:
        points: (M, 3) 
        normals: (M, 3)
        colors: (M, 3) or None
        npoints: points per block
        overlap: meter overlap between blocks
    Returns:
        sem_probs: (M, num_classes)
        offsets: (M, 3)
        extents: (M, 3) - in meters (un-scaled from log)
    """
    model.eval()
    M = points.shape[0]
    num_classes = model.semantic_head.out_channels
    # in_ch is determined by the model's first abstraction layer
    
    # Global accumulators
    sem_acc = np.zeros((M, num_classes), dtype=np.float32)
    off_acc = np.zeros((M, 3), dtype=np.float32)
    ext_acc = np.zeros((M, 3), dtype=np.float32)
    feat_acc = np.zeros((M, 128), dtype=np.float32)
    seen_counts = np.zeros(M, dtype=np.float32)
    
    # 1. Block Partitioning (Simple Grid for Industrial Scenes)
    # ... (rest of the setup)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    
    # Block size 1.0m
    block_size = 1.0
    stride = block_size - overlap
    
    x_steps = np.arange(mins[0], maxs[0], stride)
    y_steps = np.arange(mins[1], maxs[1], stride)
    # Z usually isn't very deep for these scenes, but we include it
    z_steps = np.arange(mins[2], maxs[2], stride)
    
    blocks = []
    for x in x_steps:
        for y in y_steps:
            for z in z_steps:
                # Mask points in this block
                mask = (points[:, 0] >= x) & (points[:, 0] < x + block_size) & \
                       (points[:, 1] >= y) & (points[:, 1] < y + block_size) & \
                       (points[:, 2] >= z) & (points[:, 2] < z + block_size)
                
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    blocks.append(indices)
    
    print(f"Divided scene into {len(blocks)} blocks for full coverage.")

    with torch.no_grad():
        for block_indices in tqdm(blocks, desc="Inferring Blocks"):
            # Process in chunks of npoints
            np.random.shuffle(block_indices)
            
            num_passes = int(np.ceil(len(block_indices) / npoints))
            
            for p in range(num_passes):
                start = p * npoints
                end = min(start + npoints, len(block_indices))
                
                pass_indices = block_indices[start:end]
                if len(pass_indices) < npoints:
                    pad = np.random.choice(block_indices, npoints - len(pass_indices))
                    eval_indices = np.concatenate([pass_indices, pad])
                else:
                    eval_indices = pass_indices
                
                # Prepare data
                b_pts = points[eval_indices]
                b_norms = normals[eval_indices]
                
                b_centroid = np.mean(b_pts, axis=0)
                b_pts_centered = b_pts - b_centroid
                
                if colors is not None:
                    b_colors = colors[eval_indices]
                    x_in = np.concatenate([b_pts_centered, b_norms, b_colors], axis=1)
                else:
                    x_in = np.concatenate([b_pts_centered, b_norms], axis=1)
                
                x_tensor = torch.from_numpy(x_in).float().unsqueeze(0).to(device)
                
                # Model Forward (returns 5 outputs now)
                sem_logits, unit_dir, mag, ext_preds, features = model(x_tensor)
                
                # Softmax for semantics
                probs = F.softmax(sem_logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Offsets (Absolute meters = unit_dir * mag)
                off_p = (unit_dir * mag).squeeze(0).cpu().numpy()
                
                # Extents (Log -> Abs)
                ext_p = np.exp(ext_preds.squeeze(0).cpu().numpy()) - 1e-3
                
                # Features
                feat_p = features.squeeze(0).cpu().numpy()
                
                active_len = len(pass_indices)
                target_indices = eval_indices[:active_len]
                
                sem_acc[target_indices] += probs[:active_len]
                off_acc[target_indices] += off_p[:active_len]
                ext_acc[target_indices] += ext_p[:active_len]
                feat_acc[target_indices] += feat_p[:active_len]
                seen_counts[target_indices] += 1
                
    # Average
    seen_counts = np.maximum(seen_counts, 1e-8)[:, np.newaxis]
    sem_acc /= seen_counts
    off_acc /= seen_counts
    ext_acc /= seen_counts
    feat_acc /= seen_counts
    
    return sem_acc, off_acc, ext_acc, feat_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npz file")
    parser.add_argument("--output", type=str, default="full_scene_output.npz")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--use_color", action="store_true", default=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    in_ch = 9 if args.use_color else 6
    model = PointNet2MultiHead(num_classes=args.num_classes, in_ch=in_ch).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    # Handle if it was saved with 'model_state_dict' wrapper
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.ckpt}")
    
    # Load Scene
    data = np.load(args.input)
    points = data['points']
    normals = data['normals']
    colors = None
    if args.use_color and 'colors' in data:
        colors = data['colors'].astype(np.float32) / 255.0
    
    print(f"Processing Scene: {args.input} ({len(points)} points)")
    
    # Inference
    start_time = time.time()
    sem_probs, offsets, extents, features = full_scene_inference(
        model, points, normals, colors, device=device
    )
    duration = time.time() - start_time
    print(f"Full-Scene Inference completed in {duration:.2f} seconds.")
    
    # Save results
    output_dict = {
        'points': points,
        'pred_sem_probs': sem_probs,
        'pred_offsets': offsets,
        'pred_extents': extents,
        'pred_features': features
    }
    
    # Include GT if available for benchmarking
    if 'semantics' in data: output_dict['semantics_gt'] = data['semantics']
    if 'instances' in data: output_dict['instances_gt'] = data['instances']
    
    np.savez_compressed(args.output, **output_dict)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
