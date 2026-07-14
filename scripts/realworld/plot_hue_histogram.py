import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_hue_histogram.py <path_to_pcd>")
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
    print(f"Found RGB color information... shape {colors.shape}")
    
    # Convert RGB to HSV
    # rgb_to_hsv expects RGB values in range 0-1 and outputs HSV in ranges: H (0-1), S (0-1), V (0-1).
    print("Converting RGB space to HSV space...")
    hsv_colors = rgb_to_hsv(colors)
    
    # Hue is the 0th index
    hues = hsv_colors[:, 0]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Optional: we can color the bins by their actual hue for a nicer visualization. 
    # To do this, we can use a bar chart instead of a standard histogram, or just do a standard one.
    counts, bins, patches = plt.hist(hues, bins=60, range=(0, 1), edgecolor='black', linewidth=0.5)
    
    # Colorize each bin corresponding to its central hue
    for i, patch in enumerate(patches):
        # We find the center hue of the bin
        center_hue = (bins[i] + bins[i+1]) / 2.0
        # Convert it back to RGB for matplotlib facecolor
        rgb_color = plt.matplotlib.colors.hsv_to_rgb([center_hue, 1.0, 1.0])
        patch.set_facecolor(rgb_color)
    
    plt.title('Distribution of Hues (HSV Space)')
    plt.xlabel('Hue Value (0 = Red, 0.33 = Green, 0.66 = Blue, 1 = Red)')
    plt.ylabel('Point Frequency')
    plt.grid(True, alpha=0.3)
    
    output_img = pcd_path.replace('.pcd', '_hue_histogram.png')
    plt.savefig(output_img)
    print(f"Hue histogram successfully saved to {output_img}")

if __name__ == "__main__":
    main()
