import open3d as o3d
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
import sys

def main():
    obj_path = '300_cleaned_rembg_base_1.55_512_resized.obj'
    tex_path = 'uv.png'
    
    print(f"Loading Mesh from {obj_path}...")
    mesh = o3d.io.read_triangle_mesh(obj_path, True)
    
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        print("Mesh lacks triangles or vertex colors. Ensure the OBJ has 'v x y z r g b' format.")
        return
        
    print("Using native vertex colors from OBJ...")
    v_colors = np.asarray(mesh.vertex_colors)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Convert vertex colors to HSV
    hsv_colors = rgb_to_hsv(v_colors)
    hues = hsv_colors[:, 0]
    sats = hsv_colors[:, 1]
    
    print("Masking vertices by material rules...")
    # 0=Stem, 1=Leaf, 3=Pepper, -1=Unassigned
    v_class = np.full(len(vertices), -1, dtype=int)
    
    pepper_mask = ((hues >= 0.9) | (hues <= 0.05)) & (sats > 0.2)
    leaf_mask = (hues >= 0.15) & (hues <= 0.30) & (sats > 0.1)
    stem_mask = (hues > 0.30) & (hues <= 0.45) & (sats > 0.1)
    
    v_class[stem_mask] = 0
    v_class[leaf_mask] = 1
    v_class[pepper_mask] = 3
    
    # To split them perfectly, we drop all faces that span across different materials
    # A face is valid for class X if at least 2 of its 3 vertices are class X
    def process_class(mesh, cls_id, cls_name):
        # Find which triangles belong to this class (majority vote on vertices)
        tri_v_classes = v_class[triangles] # Shape (N, 3)
        # Count how many vertices in each triangle belong to cls_id
        matches = np.sum(tri_v_classes == cls_id, axis=1)
        
        # Keep triangles where 2 or 3 vertices are this class
        triangles_to_keep = np.where(matches >= 2)[0]
        
        if len(triangles_to_keep) == 0:
            return []
            
        # Select submesh properly by dropping unwanted triangles
        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = mesh.vertices
        sub_mesh.vertex_colors = mesh.vertex_colors
        
        extracted_triangles = np.asarray(mesh.triangles)[triangles_to_keep]
        sub_mesh.triangles = o3d.utility.Vector3iVector(extracted_triangles)
        sub_mesh.remove_unreferenced_vertices()
        
        # Now run robust Connected Component split on this pure submesh
        print(f"[{cls_name}] Running Connected Components on {len(sub_mesh.triangles)} triangles...")
        triangle_clusters, cluster_n_triangles, cluster_area = sub_mesh.cluster_connected_triangles()
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        print(f"[{cls_name}] Topologically split into {len(cluster_n_triangles)} instances.")
        
        instances = []
        for c_id in range(len(cluster_n_triangles)):
            # Ignore tiny floating artifact meshes (e.g. < 20 triangles)
            if cluster_n_triangles[c_id] < 20: continue
                
            inst_tri_indices = np.where(triangle_clusters == c_id)[0]
            inst_mesh = sub_mesh.select_by_index(inst_tri_indices)
            instances.append(inst_mesh)
            
        return instances
        
    pepper_instances = process_class(mesh, 3, "Peppers")
    stem_instances = process_class(mesh, 0, "Stems")
    leaf_instances = process_class(mesh, 1, "Leaves")
    
    all_instances = pepper_instances + stem_instances + leaf_instances
    print(f"Total valid Instances across all classes: {len(all_instances)}")
    
    # Visualize by assigning strictly distinct vertex colors and merging them back into a single viewer
    print("Colorizing for visualization...")
    np.random.seed(42)
    distinct_colors = np.random.uniform(0.1, 0.9, size=(len(all_instances), 3))
    
    final_merged = o3d.geometry.TriangleMesh()
    
    for i, inst_mesh in enumerate(all_instances):
        inst_mesh.paint_uniform_color(distinct_colors[i])
        final_merged += inst_mesh
        
    print("Saving obj_instance_output.ply...")
    o3d.io.write_triangle_mesh('obj_instance_output.ply', final_merged)
    
    print("Opening Open3D Viewer...")
    o3d.visualization.draw_geometries([final_merged], window_name="Topology/Mesh Instances")

if __name__ == "__main__":
    main()
