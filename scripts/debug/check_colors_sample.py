import numpy as np
path = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors/data/scene_00001_pc.npz"
data = np.load(path)
semantics = data['semantics']
colors = data['colors'].astype(np.float32) / 255.0

print(f"File: {path}")
for cid in [1, 2, 3, 4, 5]:
    mask = (semantics == cid)
    if np.any(mask):
        mean_col = np.mean(colors[mask], axis=0)
        print(f"Class {cid} Mean Color: {mean_col}")
