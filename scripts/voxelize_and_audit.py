import numpy as np
import glob
import os
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp

def voxel_grid_sample(points, voxel_size=0.004):
    quantized = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    return indices

def get_instance_bbox(points):
    if len(points) == 0:
        return np.zeros(3), np.zeros(3)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    center = (min_xyz + max_xyz) / 2.0
    extent = max_xyz - min_xyz
    return center, extent

def process_file(file_info):
    file_path, voxel_size, output_suffix = file_info
    try:
        data = np.load(file_path)
        points = data['points']
        semantics = data['semantics']
        instances = data['instances']
        
        # Normals if present
        normals = data['normals'] if 'normals' in data else None
        # Color if present
        colors = data['colors'] if 'colors' in data else None
        
        unique_insts = np.unique(instances)
        unique_insts = unique_insts[unique_insts >= 0]
        
        vox_points = []
        vox_semantics = []
        vox_instances = []
        vox_normals = []
        vox_colors = []
        
        local_stats = {
            'orig_pts': len(points),
            'vox_pts': 0,
            'orig_class_counts': defaultdict(int),
            'vox_class_counts': defaultdict(int),
            'missing_inst': 0,
            'centroid_shifts': [],
            'extent_deltas': [],
            'class_inst_pts_orig': defaultdict(list),
            'class_inst_pts_vox': defaultdict(list)
        }
        
        for iid in unique_insts:
            mask = instances == iid
            inst_pts = points[mask]
            inst_sem = semantics[mask]
            inst_inst = instances[mask]
            inst_normals = normals[mask] if normals is not None else None
            inst_colors = colors[mask] if colors is not None else None
            
            sem_val = inst_sem[0]
            local_stats['orig_class_counts'][sem_val] += len(inst_pts)
            local_stats['class_inst_pts_orig'][sem_val].append(len(inst_pts))
            
            orig_center, orig_extent = get_instance_bbox(inst_pts)
            
            idxs = voxel_grid_sample(inst_pts, voxel_size)
            if len(idxs) > 0:
                v_pts = inst_pts[idxs]
                vox_points.append(v_pts)
                vox_semantics.append(inst_sem[idxs])
                vox_instances.append(inst_inst[idxs])
                if inst_normals is not None:
                    vox_normals.append(inst_normals[idxs])
                if inst_colors is not None:
                    vox_colors.append(inst_colors[idxs])
                
                local_stats['vox_pts'] += len(v_pts)
                local_stats['vox_class_counts'][sem_val] += len(v_pts)
                local_stats['class_inst_pts_vox'][sem_val].append(len(v_pts))
                
                vox_center, vox_extent = get_instance_bbox(v_pts)
                local_stats['centroid_shifts'].append(np.linalg.norm(vox_center - orig_center))
                local_stats['extent_deltas'].append(np.abs(vox_extent - orig_extent))
            else:
                local_stats['missing_inst'] += 1
                
        # Save voxelized file
        if len(vox_points) > 0:
            save_path = file_path.replace('.npz', f'{output_suffix}.npz')
            save_dict = {
                'points': np.concatenate(vox_points, axis=0),
                'semantics': np.concatenate(vox_semantics, axis=0),
                'instances': np.concatenate(vox_instances, axis=0)
            }
            if vox_normals:
                save_dict['normals'] = np.concatenate(vox_normals, axis=0)
            if vox_colors:
                save_dict['colors'] = np.concatenate(vox_colors, axis=0)
            
            np.savez_compressed(save_path, **save_dict)
            
        return local_stats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors')
    parser.add_argument('--voxel_size', type=float, default=0.004)
    parser.add_argument('--output_suffix', type=str, default='_4mm')
    parser.add_argument('--workers', type=int, default=mp.cpu_count())
    args = parser.parse_args()
    
    files = [f for f in glob.glob(os.path.join(args.data_root, "**/*.npz"), recursive=True) if not f.endswith(f"{args.output_suffix}.npz")]
    print(f"Found {len(files)} files to process.")
    
    file_info = [(f, args.voxel_size, args.output_suffix) for f in files]
    
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
    
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process_file, file_info), total=len(files)))
        
    for s in results:
        if s is None: continue
        agg_stats['total_orig_pts'] += s['orig_pts']
        agg_stats['total_vox_pts'] += s['vox_pts']
        agg_stats['missing_inst_count'] += s['missing_inst']
        agg_stats['centroid_shifts'].extend(s['centroid_shifts'])
        agg_stats['extent_deltas'].extend(s['extent_deltas'])
        
        for k, v in s['orig_class_counts'].items():
            agg_stats['orig_class_counts'][k] += v
        for k, v in s['vox_class_counts'].items():
            agg_stats['vox_class_counts'][k] += v
        for k, v in s['class_inst_pts_orig'].items():
            agg_stats['class_pts_orig'][k].extend(v)
        for k, v in s['class_inst_pts_vox'].items():
            agg_stats['class_pts_vox'][k].extend(v)

    # --- Final Report ---
    report_path = os.path.join(args.data_root, "audit_final_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"FINAL AUDIT RESULTS (N={len(files)})\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"1. Data Reduction (Original -> 4mm Voxel)\n")
        f.write(f"   Total Points: {agg_stats['total_orig_pts']} -> {agg_stats['total_vox_pts']}\n")
        f.write(f"   Compression Ratio: {agg_stats['total_vox_pts']/agg_stats['total_orig_pts']:.4f}\n\n")
        
        f.write(f"2. Instance Integrity\n")
        f.write(f"   Missing Instances: {agg_stats['missing_inst_count']}\n")
        if agg_stats['centroid_shifts']:
            f.write(f"   Avg Centroid Shift: {np.mean(agg_stats['centroid_shifts'])*1000:.2f} mm\n")
            f.write(f"   Avg Extent Delta: {np.mean(agg_stats['extent_deltas'], axis=0)*1000} mm\n\n")

        f.write(f"3. Class Distribution\n")
        f.write("   Class | Orig % | Vox % | Orig Avg Pts/Inst | Vox Avg Pts/Inst\n")
        f.write("   -------------------------------------------------------------\n")
        
        class_names = {0: 'Stem', 1: 'Leaf', 2: 'Peduncle', 3: 'Fruit', 4: 'Shoot'}
        for c in sorted(agg_stats['orig_class_counts'].keys()):
            name = class_names.get(c, str(c))
            orig_p = agg_stats['orig_class_counts'][c] / agg_stats['total_orig_pts'] * 100
            vox_p = agg_stats['vox_class_counts'][c] / agg_stats['total_vox_pts'] * 100
            orig_avg = np.mean(agg_stats['class_pts_orig'][c])
            vox_avg = np.mean(agg_stats['class_pts_vox'][c])
            f.write(f"   {name:<7} | {orig_p:6.2f} | {vox_p:5.2f} | {orig_avg:17.1f} | {vox_avg:16.1f}\n")

    print(f"\nAudit complete. Results saved to {report_path}")
    # Also print to stdout for screen capture
    with open(report_path, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
