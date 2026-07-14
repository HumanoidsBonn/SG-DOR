import os
import sys
sys.path.append("/home/user/menon/workspace/software/pepper_scene_graphs")
from pepper_scene_graphs.data.pepper_dataset import PepperPointDataset
import numpy as np

DATA_ROOT = "/home/data/menon/data_pepper_scenes/industrial_v2_800_20260207_2310/data"
dataset = PepperPointDataset(DATA_ROOT, split='train', npoints=1000, merge_shoots=True, use_color=False)

print(f"Dataset size: {len(dataset)}")
all_labels = set()
for i in range(min(50, len(dataset))):
    _, labels = dataset[i]
    all_labels.update(labels.numpy().tolist())

print(f"Unique labels in first 50 samples: {sorted(list(all_labels))}")
expected = [0, 1, 2, 3]
if sorted(list(all_labels)) == expected:
    print("SUCCESS: Mapping is correct (0, 1, 2, 3)")
else:
    print(f"ERROR: Mapping produced {sorted(list(all_labels))}, expected {expected}")
