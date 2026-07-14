import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_color_histogram.py <path_to_pcd>")
        sys.exit(1)
        
    pcd_path = sys.argv[1]
    
    if not os.path.exists(pcd_path):
        print(f"Error: {pcd_path} not found.")
        sys.exit(1)
        
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if not pcd.has_colors():
        print(f"Error: The point cloud does not have color information.")
        sys.exit(1)
        
    colors = np.asarray(pcd.colors)
    print(f"Found color information... shape {colors.shape}")
    
    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.hist(r, bins=50, color='red', alpha=0.5, label='Red', range=(0, 1))
    plt.hist(g, bins=50, color='green', alpha=0.5, label='Green', range=(0, 1))
    plt.hist(b, bins=50, color='blue', alpha=0.5, label='Blue', range=(0, 1))
    
    plt.title('Color Histogram of Point Cloud')
    plt.xlabel('Color Value (0.0 - 1.0)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = pcd_path.replace('.pcd', '_color_histogram.png')
    plt.savefig(output_img)
    print(f"Histogram saved to {output_img}")

if __name__ == "__main__":
    main()
