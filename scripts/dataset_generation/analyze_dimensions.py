import json
import argparse
import numpy as np
from collections import defaultdict

def analyze_dimensions(file_path):
    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    stats = defaultdict(list)

    for frame in data:
        nodes = frame.get("nodes", [])
        for node in nodes:
            node_type = node.get("type")
            obb = node.get("obb")
            if obb and "extents" in obb:
                # Open3D's obb.extent is the full dimension, not half-extent
                dims = np.array(obb["extents"])
                # Sort dimensions to consistently handle length, width, thickness
                # (since 'height' vs 'width' is orientation dependent)
                sorted_dims = np.sort(dims)[::-1] # [Max, Mid, Min]
                stats[node_type].append(sorted_dims)

    print("\n" + "="*50)
    print(f"{'Type':12} | {'Count':6} | {'Dimension (Max)':15} | {'Dimension (Mid)':15} | {'Dimension (Min)':15}")
    print("-" * 50)

    for node_type, dim_list in stats.items():
        dim_array = np.array(dim_list)
        means = np.mean(dim_array, axis=0)
        stds = np.std(dim_array, axis=0)
        
        print(f"{node_type:12} | {len(dim_list):6} | "
              f"{means[0]:.4f} (±{stds[0]:.3f}) | "
              f"{means[1]:.4f} (±{stds[1]:.3f}) | "
              f"{means[2]:.4f} (±{stds[2]:.3f})")

    print("="*50)
    print("\nNote: Dimensions are in meters. [Max, Mid, Min] corresponds to [Length, Width, Thickness] for leaves/stems.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to the scene graph JSON file")
    args = parser.parse_args()
    analyze_dimensions(args.file)
