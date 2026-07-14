import torch
from pepper_scene_graphs.models.pointnet2_proposals import PointNet2Proposals

def test_msg_forward():
    print("Testing MSG Model Initialization...")
    model = PointNet2Proposals(num_classes=5, in_ch=9, use_msg=True).cuda()
    print("Model Initialized.")
    
    B, N = 2, 8192
    x = torch.randn(B, N, 9).cuda()
    
    print("Running Forward Pass...")
    out = model(x, use_gt_masks=False)
    
    print("Output Keys:", out.keys())
    print("Sem Logits:", out['sem_logits'].shape)
    print("Feat Out:", out['feat_out'].shape)
    
    # Check if proposals are generated (might be None if random logits don't predict classes)
    # But usually at init, logits are random so some points might be assigned classes 0-3
    for c, prop in out['proposals'].items():
        if prop is not None:
             print(f"Class {c} Proposals: {prop['center'].shape}")
        else:
             print(f"Class {c}: No proposals (expected with random init)")

    print("MSG Test Passed!")

if __name__ == "__main__":
    test_msg_forward()
