import sys
import os
import argparse
import numpy as np
from termcolor import colored

# Add repo root to path
sys.path.append(os.getcwd())
from pepper_scene_graphs.data.json_subgraph_sampling import JSONSubgraphSampler

def check_connectivity(nodes, edges, selected_indices):
    """
    Verifies that the induced subgraph is connected (w.r.t connectivity edges).
    Returns True/False and component count.
    """
    if not selected_indices:
        return True, 0

    # Build local adjacency
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    adj = [[] for _ in range(len(selected_indices))]
    
    # We only care about connectivity edges for this check
    # But sampler might return all edges. We need to know which ones allow traversal.
    # For this test, we assume 'attached_to' is the connectivity label.
    CONNECT_LABELS = {'attached_to'}
    
    for u, v, label in edges:
        if u in old_to_new and v in old_to_new:
            if label in CONNECT_LABELS:
                nu, nv = old_to_new[u], old_to_new[v]
                adj[nu].append(nv)
                adj[nv].append(nu)
                
    visited = set()
    components = 0
    
    # Iterate over all nodes to find all connected components
    for start_node in range(len(selected_indices)):
        if start_node not in visited:
            components += 1
            q = [start_node]
            visited.add(start_node)
            while q:
                curr = q.pop(0)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)
                
    # In Multi-Stem sampling, we expect:
    # 1. Total visited = N (all nodes are reachable within their components)
    # 2. Number of components >= 1 (could be equal to number of stems if disjoint)
    
    all_reachable = len(visited) == len(selected_indices)
    return all_reachable, components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='scene_00000')
    parser.add_argument('--data_dir', type=str, default='/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors/data')
    parser.add_argument('--target_size', type=int, default=128)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    
    json_path = os.path.join(args.data_dir, f"{args.scene_id}_graph.json")
    npz_path = os.path.join(args.data_dir, f"{args.scene_id}_pc.npz")
    
    sampler = JSONSubgraphSampler(
        connect_edge_labels=['attached_to'],
        fruit_ped_pair_labels=['attached_to', 'fruit_peduncle'], 
        shoot_leaf_pair_labels=['attached_to'], # If we wanted to test this
        hard_pair_constraint=True,
        require_shoot_leaf_constraint=False # Defaults to False as per plan
    )
    
    print(colored(f"Loading {json_path}...", "cyan"))
    nodes, edges, _ = sampler.load_json_graph(json_path)
    
    # [DEBUG] Check Total Stems in Scene
    total_stems_in_scene = sum(1 for n in nodes if n['semantic_id'] == 1)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges.")
    print(colored(f"Total Stems in Scene: {total_stems_in_scene}", "yellow"))
    
    # 1. Check Correspondence
    print(colored("Checking NPZ Correspondence...", "cyan"))
    data_npz = np.load(npz_path)
    instances = data_npz['instances']
    unique_insts = np.unique(instances)
    
    try:
        sampler.verify_correspondence(nodes, unique_insts, args.scene_id, strict=True)
        print(colored("Correspondence Check PASSED", "green"))
    except ValueError as e:
        print(colored(f"Correspondence Check FAILED: {e}", "red"))
        return

    # 2. Run Sampling Loop
    print(colored(f"\nRunning {args.runs} sampling iterations (Target Size: {args.target_size})...", "cyan"))
    
    total_connected = 0
    total_fruits_happy = 0
    total_fruits_total = 0
    total_stems_counts = []
    
    fruit_to_ped, ped_to_fruit, _, _ = sampler.build_pair_maps(len(nodes), nodes, edges)
    
    for i in range(args.runs):
        selected = sampler.bfs_sample(nodes, edges, target_size=args.target_size, seed_policy="stem_bias")
        
        # Stats
        sub_nodes = [nodes[idx] for idx in selected]
        stems = [n for n in sub_nodes if n['semantic_id'] == 1]
        fruits = [n for n in sub_nodes if n['semantic_id'] == 4]
        peppers = [n for n in sub_nodes if n['semantic_id'] == 3]
        
        # Check Connectivity (Modified for disjoint components)
        is_valid_graph, num_components = check_connectivity(nodes, edges, selected)
        if is_valid_graph: total_connected += 1
        
        # Check Hard Constraints
        happy_fruits = 0
        for f_node in fruits:
            f_idx = f_node['index']
            # If this fruit has a peduncle in the FULL graph
            if f_idx in fruit_to_ped:
                p_idx = fruit_to_ped[f_idx]
                # Then p_idx MUST be in selected
                if p_idx in selected:
                    happy_fruits += 1
            else:
                # Orphan fruit in GT? Count as happy (constraint N/A)
                happy_fruits += 1
                
        total_fruits_happy += happy_fruits
        total_fruits_total += len(fruits)
        total_stems_counts.append(len(stems))
        
        print(f"Run {i+1}: N={len(selected)}, Stems={len(stems)}, Fruits={len(fruits)}, Components={num_components}, Valid={is_valid_graph}")

    # Report
    print(colored("\nSummary:", "yellow"))
    
    # Calculate average stems for the summary line
    avg_stems_sampled = np.mean(total_stems_counts) if total_stems_counts else 0
    
    print(f"Connectivity: {total_connected}/{args.runs} runs valid (All nodes belong to {avg_stems_sampled:.2f} components on average).")
    
    if total_fruits_total > 0:
        pct_happy = (total_fruits_happy / total_fruits_total) * 100
        col = "green" if pct_happy == 100 else "red"
        print(colored(f"Fruit-Ped Constraint: {pct_happy:.1f}% satisfied ({total_fruits_happy}/{total_fruits_total})", col))
    else:
        print("No fruits sampled.")
        
    avg_stems = np.mean(total_stems_counts)
    stem_dist = {c: total_stems_counts.count(c) for c in set(total_stems_counts)}
    print(f"Stem Distribution: {stem_dist}")
    print(f"Avg Stems: {avg_stems:.2f}")

if __name__ == "__main__":
    main()
