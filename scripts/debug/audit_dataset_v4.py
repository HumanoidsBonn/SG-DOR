import torch
from pepper_scene_graphs.data.pepper_dataset import PepperPointDataset
import numpy as np

DATA_ROOT = "/home/data/menon/data_pepper_scenes/industrial_v2_800_bio_colors/data"

ds = PepperPointDataset(DATA_ROOT, split='val', npoints=8192, merge_shoots=True, return_proposals_gt=True)

from pepper_scene_graphs.training.gt_helpers import build_stem_segment_targets

for i in range(10):
    sample = ds[i]
    mask = sample['gt_prop_mask']
    classes = sample['gt_prop_class']
    valid_classes = classes[mask > 0.5]
    
    # Debug Stem Segmentation
    xyz = sample['x'][:, :3] # This is a Tensor
    sem = sample['y_sem']    # This is a Tensor
    inst = sample['instances'].long() if hasattr(sample, 'instances') else torch.zeros_like(sem)
    
    # Use existing build_stem_segment_targets directly with tensors
    stem_centers, _ = build_stem_segment_targets(xyz, sem, inst)
    
    print(f"Scene {i}: {len(valid_classes)} valid GT proposals")
    print(f"  Stem centers built: {len(stem_centers)}")
    u, counts = torch.unique(valid_classes, return_counts=True)
    for cls_id, count in zip(u, counts):
        print(f"  Class {cls_id.item()}: {count.item()}")

    # Check Stem (0) specifically from semantics
    stem_pts = (sem == 0).sum()
    print(f"  Stem Pts: {stem_pts.item()}")
