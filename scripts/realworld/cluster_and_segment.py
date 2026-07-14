import open3d as o3d
import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy.spatial import cKDTree
import sys

def point_to_line_distance(points, line_pt, line_dir):
    diff = points - line_pt
    cross = np.cross(diff, line_dir)
    dist = np.linalg.norm(cross, axis=-1) / max(np.linalg.norm(line_dir), 1e-6)
    return dist
    
def get_pca(points):
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    e_val = eigenvalues[idx]
    e_vec = eigenvectors[:, idx]
    sum_e = np.sum(e_val)
    if sum_e == 0: sum_e = 1e-6
    l1, l2, l3 = e_val / sum_e
    return l1, l2, l3, e_vec

def main():
    if len(sys.argv) > 1:
        pcd_path = sys.argv[1]
    else:
        pcd_path = 'output_vertices_only.pcd'
        
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_colors(): return
        
    print("Voxel downsampling to 4mm for structural analysis...")
    pcd = pcd.voxel_down_sample(voxel_size=0.004)
        
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Init semantic array: -1 = unknown, 0 = Stem, 1 = Leaf, 2 = Peduncle, 3 = Pepper
    semantics = np.full(len(points), -1, dtype=int)
    instances = np.full(len(points), -1, dtype=int)
    
    print("Filtering visual seeds (Peppers & Greens)...")
    hsv_colors = rgb_to_hsv(colors)
    hues = hsv_colors[:, 0]
    sats = hsv_colors[:, 1]
    
    pepper_mask = ((hues >= 0.9) | (hues <= 0.05)) & (sats > 0.2)
    # Broad green mask for ALL plant matter
    green_mask = (hues >= 0.10) & (hues <= 0.45) & (sats > 0.1) & (~pepper_mask)
    
    # 1. PEPPERS
    print("Clustering Peppers...")
    pepper_idx = np.where(pepper_mask)[0]
    pepper_pcd = pcd.select_by_index(pepper_idx)
    pepper_dbscan = np.array(pepper_pcd.cluster_dbscan(eps=0.02, min_points=15, print_progress=False))
    
    current_instance_id = 0
    max_pepp = pepper_dbscan.max()
    for local_id in range(max_pepp + 1):
        idx_in_pcd = pepper_idx[np.where(pepper_dbscan == local_id)[0]]
        semantics[idx_in_pcd] = 3
        instances[idx_in_pcd] = current_instance_id
        current_instance_id += 1

    # 2. LOCAL PCA STRUCTURAL IDENTIFICATION (Stems vs Leaves)
    print("Computing LOCAL structural PCA for all Green points to isolate Cylinders (Stems) vs Planes (Leaves)...")
    green_idx = np.where(green_mask)[0]
    green_pts = points[green_idx]
    
    if len(green_pts) > 0:
        # We only query neighbors within the green plant structure (radius 3cm, max 40 pts)
        tree = cKDTree(green_pts)
        distances, indices = tree.query(green_pts, k=40, distance_upper_bound=0.03)
        
        linearity = np.zeros(len(green_pts))
        planarity = np.zeros(len(green_pts))
        
        for i in range(len(green_pts)):
            # get valid neighbors within 3cm
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
        
        # A true cylinder has high linearity, low planarity. A leaf has high planarity.
        # We classify every point locally:
        is_stem = (linearity > planarity) & (linearity > 0.25)
        is_leaf = ~is_stem
        
        stem_global_idx = green_idx[is_stem]
        leaf_global_idx = green_idx[is_leaf]
        
        # 3. CLUSTER STEMS
        print(f"Isolated {len(stem_global_idx)} structural Stem points. Clustering...")
        if len(stem_global_idx) > 0:
            stem_pcd = pcd.select_by_index(stem_global_idx)
            # Use smaller min_points so we don't drop thin stem sections
            stem_db = np.array(stem_pcd.cluster_dbscan(eps=0.03, min_points=8, print_progress=False))
            
            for local_id in range(stem_db.max() + 1):
                idx_in_pcd = stem_global_idx[np.where(stem_db == local_id)[0]]
                if len(idx_in_pcd) < 8: continue
                
                semantics[idx_in_pcd] = 0
                instances[idx_in_pcd] = current_instance_id
                current_instance_id += 1

        # 4. LEAVES
        print(f"Clustering {len(leaf_global_idx)} structural Leaf points...")
        # Add remaining unclassified points that might have been dropped by stems
        leftover_green = green_idx[semantics[green_idx] == -1]
        if len(leftover_green) > 0:
            leaf_pcd = pcd.select_by_index(leftover_green)
            leaf_db = np.array(leaf_pcd.cluster_dbscan(eps=0.008, min_points=8, print_progress=False))
            
            for l_id in range(leaf_db.max() + 1):
                l_loc = leftover_green[np.where(leaf_db == l_id)[0]]
                semantics[l_loc] = 1 # Leaf
                instances[l_loc] = current_instance_id
                current_instance_id += 1

    print(f"Discovered total {current_instance_id} instances.")

    out_semantic = pcd_path.replace('.pcd', '_semantic.pcd')
    print(f"Exporting semantic to {out_semantic}...")
    with open(out_semantic, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb class\nSIZE 4 4 4 4 4\nTYPE F F F F I\nCOUNT 1 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        
        r255 = (np.clip(colors[:, 0], 0, 1) * 255.0).astype(np.uint32)
        g255 = (np.clip(colors[:, 1], 0, 1) * 255.0).astype(np.uint32)
        b255 = (np.clip(colors[:, 2], 0, 1) * 255.0).astype(np.uint32)
        packed_rgb = (r255 << 16) | (g255 << 8) | b255
        packed_float = np.frombuffer(packed_rgb.astype(np.uint32).tobytes(), dtype=np.float32)

        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {packed_float[i]:.6e} {semantics[i]}\n")
            
    out_instance = pcd_path.replace('.pcd', '_instance.pcd')
    print(f"Exporting instances to {out_instance}...")
    np.random.seed(42)
    distinct_colors = np.random.uniform(0.1, 0.9, size=(current_instance_id, 3))
    inst_colors = np.zeros_like(colors)
    for i in range(len(points)):
        if instances[i] >= 0:
            inst_colors[i] = distinct_colors[instances[i]]
        else:
            inst_colors[i] = [0.1, 0.1, 0.1]
            
    instance_pcd = o3d.geometry.PointCloud()
    instance_pcd.points = pcd.points
    instance_pcd.colors = o3d.utility.Vector3dVector(inst_colors)
    
    o3d.io.write_point_cloud(out_instance, instance_pcd, write_ascii=True)
    o3d.visualization.draw_geometries([instance_pcd], window_name="Local Structure Segmentation")

if __name__ == "__main__":
    main()
