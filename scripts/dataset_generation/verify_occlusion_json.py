import json
import sys
from pathlib import Path

def verify_json(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both single scene and list of scenes
    scenes = data if isinstance(data, list) else [data]

    for i, scene in enumerate(scenes):
        print(f"Checking Scene {i}...")
        nodes = scene.get('nodes', [])
        fruits = [n for n in nodes if n.get('type') == 'fruit']
        
        if not fruits:
            print("  No fruit nodes found in this scene.")
            continue

        valid_count = 0
        for fruit in fruits:
            targets = fruit.get('occlusion_targets')
            if targets and 'union_raw' in targets:
                union_raw = targets['union_raw']
                if isinstance(union_raw, list) and len(union_raw) == 18:
                    valid_count += 1
                else:
                    print(f"  Warning: Fruit {fruit['id']} has invalid union_raw (len={len(union_raw) if isinstance(union_raw, list) else 'N/A'})")
            else:
                print(f"  Warning: Fruit {fruit['id']} missing occlusion_targets or union_raw")

        print(f"  Verified {valid_count}/{len(fruits)} fruit nodes have valid 18-dim union_raw vectors.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_occlusion_json.py <path_to_json>")
    else:
        verify_json(sys.argv[1])
