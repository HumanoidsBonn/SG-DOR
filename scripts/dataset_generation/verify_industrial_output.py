#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path
import argparse

def verify_dataset(run_dir):
    run_path = Path(run_dir)
    data_dir = run_path / "data"
    master_json = run_path / "scene_graphs.json"

    print(f"--- Auditing Run: {run_path.name} ---")
    
    if not master_json.exists():
        print(f"CRITICAL: Master JSON not found at {master_json}")
        return

    with open(master_json, "r") as f:
        master_data = json.load(f)
    print(f"Master JSON: {len(master_data)} scenes found.")

    if not data_dir.exists():
        print(f"CRITICAL: Data directory not found at {data_dir}")
        return

    # Updated glob to match new naming convention: scene_XXXXX_graph.json
    graph_files = sorted(list(data_dir.glob("scene_*_graph.json")))
    npz_files = sorted(list(data_dir.glob("scene_*_pc.npz")))

    print(f"Individual Files: {len(graph_files)} Graph JSONs, {len(npz_files)} NPZs")

    if len(graph_files) != len(master_data):
        print(f"WARNING: Graph JSON count ({len(graph_files)}) mismatch with Master ({len(master_data)})")

    all_passed = True

    for g_path in graph_files:
        # Extract ID from scene_XXXXX_graph.json
        scene_id_str = g_path.name.split("_")[1]
        npz_path = data_dir / f"scene_{scene_id_str}_pc.npz"
        meta_path = data_dir / f"scene_{scene_id_str}_metadata.json"
        
        print(f"Checking Scene {scene_id_str}...", end="\r")
        
        # 1. Graph JSON Audit
        try:
            with open(g_path, "r") as f:
                scene = json.load(f)
            
            # Semantic/Type Check (Shoot Removal)
            nodes = scene.get("nodes", [])
            for n in nodes:
                if "shoot" in n.get("type", "").lower() or "shoot" in n.get("id", "").lower():
                    print(f"\nERROR: Scene {scene_id_str} contains 'shoot' node: {n.get('id')}")
                    all_passed = False
                
                sem_id = n.get("semantic_id")
                if sem_id not in [1, 2, 3, 4]:
                    print(f"\nERROR: Scene {scene_id_str} has invalid semantic_id {sem_id} for node {n.get('id')}")
                    all_passed = False

            # Occlusion Check
            if "occlusion" not in scene:
                print(f"\nERROR: Scene {scene_id_str} missing 'occlusion' data")
                all_passed = False
            elif not scene["occlusion"]:
                print(f"\nWARNING: Scene {scene_id_str} 'occlusion' list is empty")

        except Exception as e:
            print(f"\nERROR: Failed to load Graph JSON {g_path.name}: {e}")
            all_passed = False

        # 2. Metadata Check
        if not meta_path.exists():
            print(f"\nERROR: Metadata file missing for scene {scene_id_str}: {meta_path.name}")
            all_passed = False
        else:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if "metadata" not in meta or "config" not in meta["metadata"]:
                    print(f"\nERROR: Scene {scene_id_str} metadata is missing config details")
                    all_passed = False
            except Exception as e:
                print(f"\nERROR: Failed to load Metadata JSON {meta_path.name}: {e}")
                all_passed = False

        # 3. NPZ Audit
        if not npz_path.exists():
            print(f"\nERROR: NPZ missing for scene {scene_id_str}: {npz_path.name}")
            all_passed = False
            continue
            
        try:
            data = np.load(npz_path)
            # User requirement: points, semantic_id, instance_id, x, y, z, normals
            # In our implementation, npz keys are "points", "semantics", "instances", "normals"
            expected_arrays = ["points", "semantics", "instances", "normals"]
            for arr in expected_arrays:
                if arr not in data:
                    print(f"\nERROR: NPZ {npz_path.name} missing expected array '{arr}'")
                    all_passed = False
            
            if "points" in data:
                pts = data["points"]
                if pts.shape[1] != 3:
                    print(f"\nERROR: NPZ {npz_path.name} points shape {pts.shape} is invalid (expected NX3)")
                    all_passed = False
                    
            if "normals" in data:
                norms = data["normals"]
                if norms.shape != data["points"].shape:
                    print(f"\nERROR: NPZ {npz_path.name} normals shape {norms.shape} mismatch with points")
                    all_passed = False

            # Semantic values in NPZ should also be in [1, 2, 3, 4]
            if "semantics" in data:
                unique_sems = np.unique(data["semantics"])
                if not np.all(np.isin(unique_sems, [1, 2, 3, 4])):
                    print(f"\nERROR: NPZ {npz_path.name} contains invalid semantic classes: {unique_sems}")
                    all_passed = False

        except Exception as e:
            print(f"\nERROR: Failed to load/audit NPZ {npz_path.name}: {e}")
            all_passed = False

    print("\n--- Audit Complete ---")
    if all_passed:
        print("RESULT: SUCCESS (All integrity checks passed)")
    else:
        print("RESULT: FAILURE (Integrity issues detected)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the industrial run directory")
    args = parser.parse_args()
    verify_dataset(args.run_dir)
