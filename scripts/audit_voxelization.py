import numpy as np
import glob
import os
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

def voxel_grid_sample(points, voxel_size=0.004):
    """
    Simple voxel grid downsampling.
    Returns indices of kept points.
    """
    quantized = np.floor(points / voxel_size).astype(np.int32)
    
    # Pack into unique keys
    # finding unique rows
    _, indices = np.unique(quantized, axis=0, return_index=True)
    return indices

def get_instance_bbox(points):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    center = (min_xyz + max_xyz) / 2.0
    extent = max_xyz - min_xyz
    return center, extent

def audit_scene(file_path, voxel_size=0.004):
    data = np.load(file_path)
    points = data['points']
    semantics = data['semantics']
    instances = data['instances']
    
    # --- Original Stats ---
    unique_insts = np.unique(instances)
    unique_insts = unique_insts[unique_insts >= 0] # Filter invalid
    
    stats = {
        'orig_num_inst': len(unique_insts),
        'orig_pts_per_class': defaultdict(int),
        'orig_inst_per_class': defaultdict(int),
        'vox_num_inst': 0,
        'vox_pts_per_class': defaultdict(int),
        'vox_inst_per_class': defaultdict(int),
        'missing_instances': [],
        'centroid_shifts': [],
        'extent_deltas': [],
        'class_counts_orig': defaultdict(list), # list of num_pts per instance
        'class_counts_vox': defaultdict(list)
    }
    
    # Pre-calculate original geometry
    orig_geom = {}
    for iid in unique_insts:
        mask = instances == iid
        inst_pts = points[mask]
        sem = semantics[mask][0] # Assume consistent
        
        center, extent = get_instance_bbox(inst_pts)
        orig_geom[iid] = {'center': center, 'extent': extent, 'sem': sem, 'count': len(inst_pts)}
        
        stats['orig_pts_per_class'][sem] += len(inst_pts)
        stats['orig_inst_per_class'][sem] += 1
        stats['class_counts_orig'][sem].append(len(inst_pts))

    # --- Voxelization (Per-Instance) ---
    vox_geom = {}
    
    for iid in unique_insts:
        mask = instances == iid
        inst_pts = points[mask]
        
        # Voxelize
        if len(inst_pts) > 0:
            idxs = voxel_grid_sample(inst_pts, voxel_size)
            vox_pts = inst_pts[idxs]
            
            if len(vox_pts) > 0:
                stats['vox_num_inst'] += 1
                sem = orig_geom[iid]['sem']
                
                center, extent = get_instance_bbox(vox_pts)
                vox_geom[iid] = {'center': center, 'extent': extent}
                
                stats['vox_pts_per_class'][sem] += len(vox_pts)
                stats['vox_inst_per_class'][sem] += 1
                stats['class_counts_vox'][sem].append(len(vox_pts))
                
                # Geometry Delta
                c_shift = np.linalg.norm(center - orig_geom[iid]['center'])
                e_delta = np.abs(extent - orig_geom[iid]['extent']) # Absolute diff per dim
                
                stats['centroid_shifts'].append(c_shift)
                stats['extent_deltas'].append(e_delta)
            else:
                stats['missing_instances'].append(iid)
        else:
             stats['missing_instances'].append(iid)

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors')
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    
    search_path = os.path.join(args.data_root, "*.npz")
    print(f"Searching for data in: {search_path}")
    files = glob.glob(search_path)
    
    if len(files) == 0:
        print("No files found! Checking recursive...")
        search_path = os.path.join(args.data_root, "**/*.npz")
        files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} files.")
    if len(files) == 0:
        return

    # Sample random 50
    np.random.shuffle(files)
    subset = files[:args.samples]
    
    agg_stats = {
        'total_orig_pts': 0,
        'total_vox_pts': 0,
        'orig_class_counts': defaultdict(int),
        'vox_class_counts': defaultdict(int),
        'missing_inst_count': 0,
        'centroid_shifts': [],
        'extent_deltas': [],
        'class_pts_orig': defaultdict(list),
        'class_pts_vox': defaultdict(list)
    }
    
    print(f"Auditing {len(subset)} scenes...")
    
    for f in tqdm(subset):
        s = audit_scene(f)
        
        agg_stats['missing_inst_count'] += len(s['missing_instances'])
        agg_stats['centroid_shifts'].extend(s['centroid_shifts'])
        agg_stats['extent_deltas'].extend(s['extent_deltas'])
        
        for k, v in s['orig_pts_per_class'].items():
            agg_stats['orig_class_counts'][k] += v
            agg_stats['total_orig_pts'] += v
            
        for k, v in s['vox_pts_per_class'].items():
            agg_stats['vox_class_counts'][k] += v
            agg_stats['total_vox_pts'] += v
            
        for k, v in s['class_counts_orig'].items():
            agg_stats['class_pts_orig'][k].extend(v)
            
        for k, v in s['class_counts_vox'].items():
            agg_stats['class_pts_vox'][k].extend(v)

    # --- Report ---
    print("\n" + "="*50)
    print("AUDIT RESULTS (N={})".format(len(subset)))
    print("="*50)
    
    print(f"\n1. Data Reduction (Original -> 4mm Voxel)")
    print(f"   Total Points: {agg_stats['total_orig_pts']} -> {agg_stats['total_vox_pts']}")
    print(f"   Compression Ratio: {agg_stats['total_vox_pts']/agg_stats['total_orig_pts']:.4f}")
    
    print(f"\n2. Instance Integrity")
    print(f"   Missing Instances: {agg_stats['missing_inst_count']}")
    if len(agg_stats['centroid_shifts']) > 0:
        avg_shift = np.mean(agg_stats['centroid_shifts'])
        max_shift = np.max(agg_stats['centroid_shifts'])
        print(f"   Avg Centroid Shift: {avg_shift*1000:.2f} mm")
        print(f"   Max Centroid Shift: {max_shift*1000:.2f} mm")
        
        ext_deltas = np.array(agg_stats['extent_deltas']) # N x 3
        avg_ext_d = np.mean(ext_deltas, axis=0)
        print(f"   Avg Extent Delta (L,W,H): [{avg_ext_d[0]*1000:.2f}, {avg_ext_d[1]*1000:.2f}, {avg_ext_d[2]*1000:.2f}] mm")

    print(f"\n3. Class Distribution (Points %)")
    print("   Class | Orig % | Vox % | Orig Avg Pts/Inst | Vox Avg Pts/Inst")
    print("   -------------------------------------------------------------")
    
    classes = sorted(list(set(agg_stats['orig_class_counts'].keys()) | set(agg_stats['vox_class_counts'].keys())))
    
    class_names = {0: 'Stem', 1: 'Leaf', 2: 'Peduncle', 3: 'Fruit'}
    
    for c in classes:
        name = class_names.get(c, str(c))
        orig_p = agg_stats['orig_class_counts'][c] / agg_stats['total_orig_pts'] * 100
        vox_p = agg_stats['vox_class_counts'][c] / agg_stats['total_vox_pts'] * 100
        
        orig_avg = np.mean(agg_stats['class_pts_orig'][c]) if len(agg_stats['class_pts_orig'][c]) > 0 else 0
        vox_avg = np.mean(agg_stats['class_pts_vox'][c]) if len(agg_stats['class_pts_vox'][c]) > 0 else 0
        
        print(f"   {name:<5} | {orig_p:6.2f} | {vox_p:5.2f} | {orig_avg:17.1f} | {vox_avg:16.1f}")

if __name__ == "__main__":
    main()
