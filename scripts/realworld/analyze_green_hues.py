import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from scipy.signal import find_peaks

def main():
    pcd_path = 'output.pcd'
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    colors = np.asarray(pcd.colors)
    hsv_colors = rgb_to_hsv(colors)
    
    hues = hsv_colors[:, 0]
    saturations = hsv_colors[:, 1]
    values = hsv_colors[:, 2]
    
    # Filter for standard "green-ish" and "yellow-ish" range (0.10 to 0.45)
    # Also ignore extremely dark/grey points (e.g. shadows / background context)
    valid_mask = (hues >= 0.10) & (hues <= 0.45) & (saturations > 0.1) & (values > 0.1)
    green_hues = hues[valid_mask]
    
    if len(green_hues) == 0:
        print("No points found in the green/yellow hue range!")
        return

    # Create high-res histogram for just the greens
    counts, bins = np.histogram(green_hues, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Smooth a bit to find reliable peaks
    window = np.ones(3)/3
    smoothed_counts = np.convolve(counts, window, mode='same')
    
    peaks, properties = find_peaks(smoothed_counts, height=np.max(smoothed_counts)*0.1, distance=10)
    
    print("\n========= Green/Yellow Hue Analysis =========")
    print(f"Total points analyzed in range: {len(green_hues)}")
    print("Found potential distinct clusters at Hue values:")
    
    for peak in peaks:
        hue_val = bin_centers[peak]
        print(f" - Peak around Hue: {hue_val:.4f} (Count: {counts[peak]})")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    patches = plt.bar(bin_centers, counts, width=(bins[1]-bins[0]), edgecolor=None)
    
    for i, patch in enumerate(patches):
        center_hue = bin_centers[i]
        rgb_color = plt.matplotlib.colors.hsv_to_rgb([center_hue, 0.8, 0.8])
        patch.set_facecolor(rgb_color)
        
    plt.plot(bin_centers[peaks], counts[peaks], "x", color='red', markersize=10, label='Detected Peaks')
    plt.title('High-Res Hue Histogram for Stems & Leaves (0.1 to 0.45)')
    plt.xlabel('Hue Value')
    plt.ylabel('Point Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('green_hue_peaks.png')
    print("Saved high-res zoomed histogram to green_hue_peaks.png")

if __name__ == "__main__":
    main()
