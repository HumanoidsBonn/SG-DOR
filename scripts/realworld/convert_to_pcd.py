import open3d as o3d
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert OBJ to PCD using Open3D")
    parser.add_argument("input", help="Path to input .obj file")
    parser.add_argument("output", help="Path to output .pcd file")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample from the mesh (default: 100000)")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Reading mesh from '{input_path}'...")
    try:
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(input_path, True)
    except Exception as e:
        print(f"Error reading mesh: {e}")
        sys.exit(1)
        
    print(len(mesh.vertices), "vertices read.")
    
    if len(mesh.vertices) == 0:
        print("Warning: The mesh has zero vertices. Maybe it specifies point clouds natively? Reading as point cloud directly.")
        try:
            pcd = o3d.io.read_point_cloud(input_path)
        except Exception as e:
            print(f"Failed to read as point cloud: {e}")
            sys.exit(1)
    else:
        # Convert to point cloud by sampling points uniformly from the mesh
        print(f"Sampling {args.num_points} points from the mesh...")
        pcd = mesh.sample_points_uniformly(number_of_points=args.num_points)
        print("Sampling complete.")
        
    print(f"Saving point cloud to '{output_path}'...")
    try:
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        print("Saved successfully!")
    except Exception as e:
        print(f"Error writing PCD: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
