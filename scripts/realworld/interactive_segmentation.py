import open3d as o3d
import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy.spatial import cKDTree
import sys

def main():
    if len(sys.argv) > 1:
        pcd_path = sys.argv[1]
    else:
        pcd_path = 'output_vertices_only.pcd'
        
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print("Voxelizing to 4mm for fast interaction & analysis...")
    pcd = pcd.voxel_down_sample(voxel_size=0.004)
    
    print("Filtering out pure black and white noise points...")
    tmp_colors = np.asarray(pcd.colors)
    tmp_hsv = rgb_to_hsv(tmp_colors)
    v_vals = tmp_hsv[:, 2]
    s_vals = tmp_hsv[:, 1]
    
    is_white = (v_vals > 0.9) & (s_vals < 0.1)
    is_black = (v_vals < 0.1)
    keep_mask = ~(is_white | is_black)
    
    pcd = pcd.select_by_index(np.where(keep_mask)[0])
    print(f"Dropped {len(tmp_colors) - len(pcd.points)} black/white noise points.")
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print("Calculating initial masks (Red vs Green)...")
    hsv_colors = rgb_to_hsv(colors)
    hues = hsv_colors[:, 0]
    sats = hsv_colors[:, 1]
    
    pepper_mask = ((hues >= 0.9) | (hues <= 0.05)) & (sats > 0.2)
    green_mask = (hues >= 0.10) & (hues <= 0.45) & (sats > 0.1) & (~pepper_mask)
    
    pepper_idx = np.where(pepper_mask)[0]
    green_idx = np.where(green_mask)[0]
    
    print("Computing LOCAL structural PCA to cleanly sever Leaves from Stems...")
    green_pts = points[green_idx]
    is_stem = np.zeros(len(green_pts), dtype=bool)
    
    if len(green_pts) > 0:
        tree = cKDTree(green_pts)
        distances, indices = tree.query(green_pts, k=40, distance_upper_bound=0.03)
        
        linearity = np.zeros(len(green_pts))
        planarity = np.zeros(len(green_pts))
        
        for i in range(len(green_pts)):
            valid_idx = indices[i][distances[i] < 0.03]
            if len(valid_idx) >= 10:
                local_neighborhood = green_pts[valid_idx]
                cov = np.cov(local_neighborhood.T)
                evals, _ = np.linalg.eigh(cov)
                e1, e2, e3 = np.sort(np.abs(evals))[::-1]
                sum_e = e1 + e2 + e3
                if sum_e > 1e-6:
                    linearity[i] = (e1 - e2) / sum_e
                    planarity[i] = (e2 - e3) / sum_e
                    
        is_stem = (linearity > planarity) & (linearity > 0.25)
        
    stem_global_idx = green_idx[is_stem]
    leaf_global_idx = green_idx[~is_stem]
    
    # Pre-compute independent DBSCANs so they NEVER cross boundaries
    print("Pre-computing boundary-isolated structural clusters...")
    db_labels = np.full(len(points), -1, dtype=int)
    cluster_offset = 0
    
    def cluster_subset(idx_list, eps, min_pts):
        nonlocal cluster_offset
        if len(idx_list) == 0: return
        sub_pcd = pcd.select_by_index(idx_list)
        sub_db = np.array(sub_pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
        max_id = sub_db.max()
        for i in range(max_id + 1):
            local_pts = idx_list[np.where(sub_db == i)[0]]
            db_labels[local_pts] = cluster_offset
            cluster_offset += 1

    cluster_subset(pepper_idx, eps=0.02, min_pts=10)
    cluster_subset(stem_global_idx, eps=0.03, min_pts=8)
    cluster_subset(leaf_global_idx, eps=0.008, min_pts=8)
    
    # Track final assignment
    semantics = np.full(len(points), -1, dtype=int)
    instances = np.full(len(points), -1, dtype=int)
    current_instance = 0
    
    while True:
        print("\n" + "="*60)
        print(" INTERACTIVE INSTANCE SEGMENTATION ")
        print("="*60)
        print("  s   = Stem\n  l   = Leaf\n  f   = Pepper\n  p = Peduncle")
        print("  q   = Quit & Save")
        print("="*60)
        
        choice = input("Enter class for the NEXT instance to pick: ").strip().lower()
        
        if choice == 'q':
            break
            
        class_id = -1
        if choice == 's': class_id = 0
        elif choice == 'l': class_id = 1
        elif choice == 'p': class_id = 2
        elif choice == 'f': class_id = 3
        
        if class_id == -1:
            print("Invalid choice. Please try again.")
            continue
            
        print("\n[INSTRUCTIONS]")
        print("1. An Open3D window will now open.")
        print("2. Orient the camera to see your target instance.")
        print("3. Hold [SHIFT] and [LEFT CLICK] on a few points scattered across your instance.")
        print("4. Close the window when you are done picking points for THIS instance.")
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"Pick points for Instance {current_instance} (Class: {choice.upper()})")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        
        picked = vis.get_picked_points()
        if len(picked) == 0:
            print("No points picked. Skipping this instance.")
            continue
            
        clusters_hit = set()
        for pt_idx in picked:
            c_id = db_labels[pt_idx]
            if c_id != -1:
                clusters_hit.add(c_id)
                
        assigned_pts = set(picked)
        for c_id in clusters_hit:
            c_pts = np.where(db_labels == c_id)[0]
            assigned_pts.update(c_pts)
            
        assigned_pts = list(assigned_pts)
        
        overwritten = 0
        for pt in assigned_pts:
            if semantics[pt] != -1:
                overwritten += 1
            semantics[pt] = class_id
            instances[pt] = current_instance
            
        if overwritten > 0:
            print(f"Warning: {overwritten} points were reassigned from a previously labeled instance.")
            
        print(f"-> Instance {current_instance} captured {len(assigned_pts)} points in total!")
        
        inst_color = np.random.uniform(0.1, 0.9, size=3)
        np.asarray(pcd.colors)[assigned_pts] = inst_color
        current_instance += 1
        
    print("\nSaving final interactive segmentation...")
    out_semantic = pcd_path.replace('.pcd', '_interactive_semantic.pcd')
    out_instance = pcd_path.replace('.pcd', '_interactive_instance.pcd')
    
    with open(out_semantic, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z class\nSIZE 4 4 4 4\nTYPE F F F I\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {semantics[i]}\n")
            
    o3d.io.write_point_cloud(out_instance, pcd, write_ascii=True)
    print(f"Saved {out_semantic} and {out_instance}.")
    print("All done!")

if __name__ == "__main__":
    main()
