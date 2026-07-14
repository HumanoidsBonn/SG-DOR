
import torch
import sys
import os

# Add workspace to path
sys.path.append("/home/user/menon/workspace/software/pepper_scene_graphs")

print("Testing imports...")
try:
    from pepper_scene_graphs.models.pointnet2_utils import SetAbstractionMSG, SetAbstractionSSG
    print("[PASS] pointnet2_utils import")
except ImportError as e:
    print(f"[FAIL] pointnet2_utils: {e}")

try:
    from pepper_scene_graphs.models.pointnet2_semantic_seg import PointNet2SemanticSeg
    model = PointNet2SemanticSeg(num_classes=6)
    print("[PASS] PointNet2SemanticSeg instantiation")
except Exception as e:
    print(f"[FAIL] PointNet2SemanticSeg: {e}")

try:
    from pepper_scene_graphs.models.pointnet2_multihead import PointNet2MultiHead
    model = PointNet2MultiHead(num_classes=6)
    print("[PASS] PointNet2MultiHead instantiation")
except Exception as e:
    print(f"[FAIL] PointNet2MultiHead: {e}")

try:
    from pepper_scene_graphs.models.oracle_encoder import SharedMSGEncoder
    model = SharedMSGEncoder()
    print("[PASS] SharedMSGEncoder instantiation")
except Exception as e:
    print(f"[FAIL] SharedMSGEncoder: {e}")
