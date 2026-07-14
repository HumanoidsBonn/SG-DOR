import open3d as o3d
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_vertices_to_pcd.py <input.obj> <output.pcd>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    print(f"Loading mesh from {input_path}...")
    # True argument instructs Open3D to print progress and attempt to load materials/colors
    mesh = o3d.io.read_triangle_mesh(input_path, True)
    
    if len(mesh.vertices) == 0:
        print("Error: The mesh has zero vertices.")
        sys.exit(1)
        
    print(f"Mesh loaded with {len(mesh.vertices)} vertices.")
    
    # Strictly extract vertices without surface sampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    
    # Retain native vertex colors if they exist
    if mesh.has_vertex_colors():
        print("Transferring native vertex colors to point cloud...")
        pcd.colors = mesh.vertex_colors
    else:
        print("Warning: Mesh does not have native vertex colors.")
        
    # Retain normals if they exist
    if mesh.has_vertex_normals():
        print("Transferring native vertex normals...")
        pcd.normals = mesh.vertex_normals
        
    print(f"Saving exact vertices to {output_path} as Point Cloud (ASCII format)...")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print("Saved successfully!")

if __name__ == "__main__":
    main()
