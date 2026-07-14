import numpy as np
import json
import os
import random
from pathlib import Path

def audit_scene(data_dir, scene_id):
    graph_path = data_dir / f"scene_{scene_id:05d}_graph.json"
    pc_path = data_dir / f"scene_{scene_id:05d}_pc.npz"
    metadata_path = data_dir / f"scene_{scene_id:05d}_metadata.json"
    
    results = {"scene_id": scene_id}
    
    # 1. Audit Graph
    if graph_path.exists():
        with open(graph_path, 'r') as f:
            graph = json.load(f)
            results["node_count"] = len(graph.get("nodes", []))
            results["edge_count"] = len(graph.get("edges", []))
            
            # Count semantics
            semantics = {}
            for node in graph.get("nodes", []):
                t = node.get("type", "unknown")
                semantics[t] = semantics.get(t, 0) + 1
            results["semantics"] = semantics
            
            # Check for BBox/OBB
            has_obb = all("obb" in n for n in graph.get("nodes", []) if n.get("type") in ["pepper", "leaf", "stem"])
            results["has_obb"] = has_obb
    else:
        results["graph_error"] = "Missing"

    # 2. Audit Point Cloud
    if pc_path.exists():
        try:
            data = np.load(pc_path)
            # Logic: npz usually has 'points', 'colors', 'instances', 'semantics'
            # Note: The generation script uses internal_extract_pc which saves 'points', 'colors', 'instances', 'semantics'
            points = data['points']
            results["point_count"] = points.shape[0]
            results["pc_keys"] = list(data.keys())
            
            if 'instances' in data:
                unique_instances = np.unique(data['instances'])
                results["pc_instance_count"] = len(unique_instances)
            
            if 'semantics' in data:
                unique_semantics = np.unique(data['semantics'])
                results["pc_semantic_ids"] = unique_semantics.tolist()
                
            # Check for emptiness
            results["pc_is_valid"] = points.shape[0] > 0
        except Exception as e:
            results["pc_error"] = str(e)
    else:
        results["pc_error"] = "Missing"
        
    return results

def main():
    data_dir = Path("/home/data/menon/data_pepper_scenes/output_scenes/industrial_v1_2mm_20260207_1446/data")
    import glob
    all_graphs = glob.glob(str(data_dir / "*_graph.json"))
    scene_ids = [int(os.path.basename(f).split('_')[1]) for f in all_graphs]
    
    if not scene_ids:
        print("No scenes found.")
        return

    # Sample 5 random scenes
    samples = random.sample(scene_ids, min(5, len(scene_ids)))
    # Also include first and last
    samples.extend([min(scene_ids), max(scene_ids)])
    samples = sorted(list(set(samples)))
    
    print(f"Auditing {len(samples)} scenes: {samples}")
    print("-" * 40)
    
    for sid in samples:
        res = audit_scene(data_dir, sid)
        print(json.dumps(res, indent=2))
        print("-" * 20)

if __name__ == "__main__":
    main()
