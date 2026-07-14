import numpy as np
import sys
import os

file_path = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors/data/scene_00397_pc.npz"
data = np.load(file_path)
pts = data['points']
sem = data['semantics']
inst = data['instances']

print(f"File: {file_path}")
print(f"Points: {len(pts)}")
unique_sem, counts_sem = np.unique(sem, return_counts=True)
print("Raw Semantic Counts (1=Stem, 2=Leaf, 3=Peduncle, 4=Fruit, 5=Shoot):")
for s, c in zip(unique_sem, counts_sem):
    print(f"  ID {s}: {c} points")
    
unique_inst = np.unique(inst)
print(f"Total Instances: {len(unique_inst)}")

# Check if any instances have semantic ID 2 (Leaf)
leaf_mask = (sem == 2)
leaf_insts = np.unique(inst[leaf_mask])
print(f"Leaf Instances (ID 2): {len(leaf_insts)}")
