import json
import numpy as np
import random
from collections import deque

class JSONSubgraphSampler:
    def __init__(self, 
                 connect_edge_labels=None, 
                 fruit_ped_pair_labels=None, 
                 shoot_leaf_pair_labels=None,
                 hard_pair_constraint=True,
                 require_shoot_leaf_constraint=False):
        """
        Args:
            connect_edge_labels: list of edge labels used for BFS connectivity.
            fruit_ped_pair_labels: list of edge labels defining Fruit-Peduncle bonds.
            shoot_leaf_pair_labels: list of edge labels defining Shoot-Leaf bonds.
            hard_pair_constraint: If True, paired nodes are forced into the subgraph together.
            require_shoot_leaf_constraint: If True, apply hard constraint to Shoot-Leaf pairs too.
        """
        self.connect_edge_labels = set(connect_edge_labels) if connect_edge_labels else set()
        self.fruit_ped_pair_labels = set(fruit_ped_pair_labels) if fruit_ped_pair_labels else set()
        self.shoot_leaf_pair_labels = set(shoot_leaf_pair_labels) if shoot_leaf_pair_labels else set()
        self.hard_pair_constraint = hard_pair_constraint
        self.require_shoot_leaf_constraint = require_shoot_leaf_constraint
        
        # Prototype IDs (Fixed for this dataset version)
        self.STEM_SEM_ID = 1
        self.LEAF_SEM_ID = 2
        self.PED_SEM_ID = 3
        self.FRUIT_SEM_ID = 4
        self.SHOOT_SEM_ID = 5

    def load_json_graph(self, json_path):
        """
        Loads JSON, normalizes indexing to 0..N-1.
        Returns: 
            nodes: list of dicts (with 'index', 'instance_id', 'semantic_id', 'type')
            edges: list of (u, v, label) tuples
            str_id_to_idx: dict map
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        nodes = data.get('nodes', [])
        node_list = []
        str_id_to_idx = {}
        
        for i, n in enumerate(nodes):
            info = {
                'index': i,
                'id': n['id'],
                'instance_id': int(n['instance_id']),
                'semantic_id': int(n['semantic_id']),
                'type': n.get('type', 'unknown'),
                'centroid': np.array(n['centroid'], dtype=np.float32),
                'extent': np.array(n['obb']['extents'], dtype=np.float32) if 'obb' in n else np.zeros(3)
            }
            node_list.append(info)
            str_id_to_idx[n['id']] = i
            
        edge_list = []
        for e in data.get('edges', []):
            u_str = e['source'] if 'source' in e else e.get('src')
            v_str = e['target'] if 'target' in e else e.get('dst')
            label = e.get('label', 'unknown')
            
            if u_str in str_id_to_idx and v_str in str_id_to_idx:
                u = str_id_to_idx[u_str]
                v = str_id_to_idx[v_str]
                edge_list.append((u, v, label))
                
        return node_list, edge_list, str_id_to_idx

    def verify_correspondence(self, nodes, npz_instance_ids, scene_id, strict=True):
        """Checks if all JSON nodes exist in the NPZ instance list."""
        npz_set = set(npz_instance_ids)
        missing = []
        for n in nodes:
            if n['instance_id'] not in npz_set:
                missing.append(n['instance_id'])
        
        if missing and strict:
            raise ValueError(f"Scene {scene_id}: JSON nodes missing in NPZ! Instances: {missing}")
        return len(missing) == 0

    def build_adjacency(self, num_nodes, edges):
        """Builds adjacency list for BFS traversal."""
        adj = [[] for _ in range(num_nodes)]
        for u, v, label in edges:
            if label in self.connect_edge_labels:
                adj[u].append(v)
                adj[v].append(u)
        return adj

    def build_pair_maps(self, num_nodes, nodes, edges):
        """
        Builds maps for hard constraints:
        fruit_to_ped[u] = v
        ped_to_fruit[v] = u
        shoot_to_leaf[s] = [l1, l2, ...]  (One shoot can have multiple leaves)
        leaf_to_shoot[l] = s
        """
        fruit_to_ped = {}
        ped_to_fruit = {}
        shoot_to_leaves = {} # Shoot -> List[Leaf]
        leaf_to_shoot = {}   # Leaf -> Shoot

        for u, v, label in edges:
            # Check Fruit-Peduncle
            if label in self.fruit_ped_pair_labels:
                t_u = nodes[u]['semantic_id']
                t_v = nodes[v]['semantic_id']
                
                # Case 1: u=Ped, v=Fruit
                if t_u == self.PED_SEM_ID and t_v == self.FRUIT_SEM_ID:
                    ped_to_fruit[u] = v
                    fruit_to_ped[v] = u
                # Case 2: u=Fruit, v=Ped
                elif t_u == self.FRUIT_SEM_ID and t_v == self.PED_SEM_ID:
                    fruit_to_ped[u] = v
                    ped_to_fruit[v] = u

            # Check Shoot-Leaf (if required)
            if self.require_shoot_leaf_constraint and label in self.shoot_leaf_pair_labels:
                t_u = nodes[u]['semantic_id']
                t_v = nodes[v]['semantic_id']

                # Shoot (5) <-> Leaf (2)
                # Note: 'semantics' in JSON is usually 1-based.
                # Only map if nodes match valid semantic IDs.
                
                s_idx, l_idx = None, None
                if t_u == self.SHOOT_SEM_ID and t_v == self.LEAF_SEM_ID:
                    s_idx, l_idx = u, v
                elif t_u == self.LEAF_SEM_ID and t_v == self.SHOOT_SEM_ID:
                    s_idx, l_idx = v, u
                
                if s_idx is not None:
                    if s_idx not in shoot_to_leaves: shoot_to_leaves[s_idx] = []
                    shoot_to_leaves[s_idx].append(l_idx)
                    leaf_to_shoot[l_idx] = s_idx

        return fruit_to_ped, ped_to_fruit, shoot_to_leaves, leaf_to_shoot

    def bfs_sample(self, nodes, edges, target_size=128, seed_policy="stem_bias"):
        """
        Main sampling function.
        seed_policy: 'stem_bias' (prefer 2+ stems), 'uniform' (random).
        """
        num_nodes = len(nodes)
        adj = self.build_adjacency(num_nodes, edges)
        fruit_to_ped, ped_to_fruit, shoot_to_leaves, leaf_to_shoot = self.build_pair_maps(num_nodes, nodes, edges)
        
        # 1. Choose Seed
        stem_indices = [i for i, n in enumerate(nodes) if n['semantic_id'] == self.STEM_SEM_ID]
        
        # [PARALLEL BFS STRATEGY]
        # We initialize the queue with MULTIPLE stems (if available) to grow distinct 
        # disconnected components simultaneously (Round-Robin expansion).
        
        seed_indices = []
        if seed_policy == "stem_bias":
            # [Refinement] At least 2 stems with 80% prob, 1 stem with 20% prob
            if random.random() < 0.8 and len(stem_indices) >= 2:
                num_to_pick = random.randint(2, len(stem_indices))
                seed_indices = random.sample(stem_indices, num_to_pick)
            elif stem_indices:
                seed_indices = [random.choice(stem_indices)]
        
        # Fallback if no stems found or policy is uniform
        if not seed_indices:
            seed_indices = [random.randint(0, num_nodes - 1)]

        # BFS Init
        # Initializing queue with multiple seeds creates "Parallel BFS"
        queue = deque(seed_indices)
        selected_set = set(seed_indices)
        selected_list = list(seed_indices)
        
        # Helper to add node if space permits
        def try_add(idx):
            if idx not in selected_set and len(selected_list) < target_size:
                selected_set.add(idx)
                selected_list.append(idx)
                queue.append(idx)
                return True
            return False

        # Helper to handle hard constraints
        def check_constraints_and_add(curr):
            # 1. Fruit <-> Peduncle
            if self.hard_pair_constraint:
                if curr in fruit_to_ped:
                    friend = fruit_to_ped[curr]
                    if friend not in selected_set:
                        try_add(friend)
                if curr in ped_to_fruit:
                    friend = ped_to_fruit[curr]
                    if friend not in selected_set:
                        try_add(friend)

            # 2. Shoot <-> Leaf
            if self.hard_pair_constraint and self.require_shoot_leaf_constraint:
                if curr in shoot_to_leaves:
                    for leaf_idx in shoot_to_leaves[curr]:
                        if leaf_idx not in selected_set:
                            try_add(leaf_idx)
                if curr in leaf_to_shoot:
                    shoot_idx = leaf_to_shoot[curr]
                    if shoot_idx not in selected_set:
                        try_add(shoot_idx)

        # Initial constraint checks for all seeds
        for s in seed_indices:
            check_constraints_and_add(s)

        # 2. Main BFS Loop (Round-Robin Expansion)
        while queue and len(selected_list) < target_size:
            u = queue.popleft()
            nbrs = adj[u]
            random.shuffle(nbrs)
            
            # Prioritize stems? Not strictly necessary if we already seeded them.
            # But kept for consistency within a component.
            nbrs.sort(key=lambda x: 0 if nodes[x]['semantic_id'] == self.STEM_SEM_ID else 1)

            for v in nbrs:
                if v not in selected_set:
                    if try_add(v):
                        check_constraints_and_add(v)
                    if len(selected_list) >= target_size:
                        break
        
        return selected_list

    def induce_subgraph(self, original_nodes, original_edges, selected_indices):
        """
        Creates the subgraph data structures.
        Returns:
            sub_nodes: list of node dicts
            sub_edges: list of (new_u, new_v, label)
            old_to_new: dict map
        """
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        sub_nodes = [original_nodes[i] for i in selected_indices]
        sub_edges = []
        
        for u, v, label in original_edges:
            if u in old_to_new and v in old_to_new:
                sub_edges.append((old_to_new[u], old_to_new[v], label))
                
        return sub_nodes, sub_edges, old_to_new
