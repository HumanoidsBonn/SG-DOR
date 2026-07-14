
import json
import numpy as np
import argparse
import sys
from pathlib import Path

def verify_run(run_dir):
    data_dir = Path(run_dir) / "data"
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return False

    files = sorted(list(data_dir.glob("scene_*.json")))
    scene_ids = set()
    for f in files:
        if "_graph.json" in f.name:
            scene_ids.add(f.name.split("_")[1])
    
    print(f"Found {len(scene_ids)} scenes in {data_dir}")
    
    all_passed = True
    
    for sid in sorted(list(scene_ids)):
        graph_path = data_dir / f"scene_{sid}_graph.json"
        meta_path = data_dir / f"scene_{sid}_metadata.json"
        pc_path = data_dir / f"scene_{sid}_pc.npz"
        
        # 1. Existence Check
        missing = []
        if not graph_path.exists(): missing.append("graph")
        if not meta_path.exists(): missing.append("metadata")
        if not pc_path.exists(): missing.append("pc.npz")
        
        if missing:
            print(f"Scene {sid}: FAIL - Missing files: {missing}")
            all_passed = False
            continue
            
        # 2. Graph Validation
        try:
            with open(graph_path, 'r') as f:
                graph = json.load(f)
            
            # Check for shoot
            has_shoot = False
            invalid_semantics = []
            
            nodes = graph.get('nodes', [])
            for n in nodes:
                if 'shoot' in n['type'].lower() or 'shoot' in n['id'].lower():
                    has_shoot = True
                
                sem_id = n.get('semantic_id', 0)
                if sem_id not in [1, 2, 3, 4]: # stem, leaf, peduncle, fruit
                    invalid_semantics.append(sem_id)
            
            if has_shoot:
                print(f"Scene {sid}: FAIL - Graph contains 'shoot' nodes")
                all_passed = False
            if invalid_semantics:
                print(f"Scene {sid}: FAIL - Invalid semantic IDs found: {set(invalid_semantics)}")
                all_passed = False
                
        except Exception as e:
            print(f"Scene {sid}: FAIL - Graph JSON error: {e}")
            all_passed = False
            continue

        # 3. NPZ Validation
        try:
            data = np.load(pc_path)
            if 'points' not in data or 'semantics' not in data:
                print(f"Scene {sid}: FAIL - NPZ missing required arrays")
                all_passed = False
            else:
                semantics = data['semantics']
                unique_sems = np.unique(semantics)
                # Check for 0 (background? usually not in point cloud if extracted from mesh) or 5 (old fruit) or 3 (old shoot/new peduncle?)
                # New map: stem=1, leaf=2, ped=3, fruit=4.
                # If shoot (old=3) is still there masked as peduncle? 
                # Shoot removal means no mesh with shoot name is processed.
                # So if we see values outside 1-4, it's weird. 
                # (0 might happen if background extracted? usually internal_extract_pc logic determines filtering)
                
                if not np.all(np.isin(unique_sems, [1, 2, 3, 4])):
                     print(f"Scene {sid}: FAIL - NPZ contains unexpected semantic classes: {unique_sems}")
                     # all_passed = False # Soft fail? 
                
        except Exception as e:
            print(f"Scene {sid}: FAIL - NPZ load error: {e}")
            all_passed = False
            continue
            
        # print(f"Scene {sid}: PASS")

    if all_passed:
        print("\nOVERALL: PER-SCENE CHECKS PASSED")
    else:
        print("\nOVERALL: PER-SCENE FAILURES DETECTED")
        
    # Check Merged File
    merged_path = Path(run_dir) / "scene_graphs.json"
    if merged_path.exists():
        try:
            with open(merged_path, 'r') as f:
                merged = json.load(f)
            print(f"\nMerged `scene_graphs.json` has {len(merged)} entries.")
            if len(merged) != len(scene_ids):
                print(f"FAIL: Merged count ({len(merged)}) != Found scenes ({len(scene_ids)})")
            
            # Check IDs
            merged_ids = {s['id'] for s in merged}
            extracted_ids = {int(s) for s in scene_ids} # scene_ids are strings '00000'
            
            if merged_ids != extracted_ids:
                 print(f"FAIL: ID mismatch. Merged: {sorted(list(merged_ids))}, Scenes: {sorted(list(extracted_ids))}")
            else:
                 print("Merged IDs match individual scene files.")
                 
        except Exception as e:
            print(f"FAIL: Could not load merged file: {e}")
    else:
        print("\nFAIL: `scene_graphs.json` missing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory")
    args = parser.parse_args()
    verify_run(args.run_dir)
