import numpy as np
import open3d as o3d
import argparse
import os

def get_glasbey_palette():
    """Returns a list of 256 maximally distinct colors."""
    # This is a curated set of distinct colors (Glasbey-like)
    # Using a known set of high-contrast RGB values
    palette = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.75, 0.25, 0.0], [0.25, 0.75, 0.0], [0.0, 0.25, 0.75], [0.75, 0.0, 0.25],
        [0.25, 0.0, 0.75], [0.0, 0.75, 0.25], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0], [1.0, 0.0, 0.5], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]
    ]
    # Fill up to 256 with pseudo-random but deterministic colors if needed
    np.random.seed(42)
    for _ in range(256 - len(palette)):
        palette.append(np.random.rand(3).tolist())
    return np.array(palette)

def visualize_instances(npz_path, voxel_size=0.005):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    points = data['points']
    instances = data['instances']
    
    print(f"Original point count: {len(points)}")
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Apply Glasbey-like colors based on instance_id
    palette = get_glasbey_palette()
    unique_instances = np.unique(instances)
    print(f"Unique instances: {len(unique_instances)}")
    
    # Map each instance to a color
    # instance_id % size of palette
    colors = palette[instances % len(palette)]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Downsample for visualization performance
    if voxel_size > 0:
        print(f"Downsampling with voxel size: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"Downsampled point count: {len(pcd.points)}")
    
    # Visualize
    print("Opening Open3D visualization window...")
    o3d.visualization.draw_geometries([pcd], window_name="Instance Visualization (Glasbey Colors)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Path to .npz file")
    parser.add_argument('--voxel', type=float, default=0.005, help="Voxel size for downsampling (default: 0.005)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist.")
    else:
        visualize_instances(args.input, args.voxel)
