import blenderproc as bproc
import argparse
import numpy as np
import random

# -------------------- Argument parser --------------------
parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="camera_files/camera", help="Path to the camera pose file")
parser.add_argument('scene', nargs='?', default="blend/blenderproc_pepper_rgy.blend", help="Path to the scene .blend file")
parser.add_argument('output_dir', nargs='?', default="output/segment/", help="Path to the output directory")
args = parser.parse_args()

# -------------------- Initialize BlenderProc --------------------
bproc.init()

# -------------------- Load scene with collections and objects --------------------
objs = bproc.loader.load_blend(args.scene, data_blocks=["collections", "objects"])
print(f"\nTotal loaded objects: {len(objs)}")

# -------------------- Group by name patterns (optional, for category_id logic) --------------------
stem_objs = []
leaf_objs = []
shoot_objs = []
pepper_green_objs = []
pepper_yellow_objs = []
pepper_red_objs = []
pedduncle_objs = []
cuts_objs = []

for obj in objs:
    name = obj.get_name()
    if "stem_" in name or "stems_" in name:
        stem_objs.append(obj)
    elif "leafs_" in name:
        leaf_objs.append(obj)
    elif "shoots_" in name:
        shoot_objs.append(obj)
    elif "peduncles_" in name:
        pedduncle_objs.append(obj)
    elif "pepper_green" in name:
        pepper_green_objs.append(obj)
    elif "pepper_yellow" in name:
        pepper_yellow_objs.append(obj)
    elif "peppers_7" in name:
        pepper_red_objs.append(obj)
    elif "cuts_" in name:   
        cuts_objs.append(obj)

# -------------------- Assign unique color per object (for instance coloring) --------------------
# def make_json_safe_color(color):
#     return [float(x) for x in list(color)]

# for i, obj in enumerate(objs):
#     random.seed(i)
#     color = make_json_safe_color([random.random(), random.random(), random.random()])
#     try:
#         obj.set_cp("instance_color", color)
#     except Exception as e:
#         print(f"⚠️ Skipping color assignment for {obj.get_name()}: {e}")
# -------------------- Set camera intrinsics and poses --------------------
bproc.camera.set_resolution(360, 512)

with open(args.camera, "r") as f:
    for line in f:
        values = [float(x) for x in line.strip().split()]
        position = values[:3]
        rotation = values[3:]
        cam_pose = bproc.math.build_transformation_mat(position, rotation)
        bproc.camera.add_camera_pose(cam_pose)

bproc.camera.set_intrinsics_from_blender_params(
    lens_unit="MILLIMETERS",
    lens=50,
    clip_start=0.01,
    clip_end=1000,
)

# -------------------- Renderer setup --------------------
bproc.renderer.set_world_background([0.5, 0.5, 0.5])
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
# bproc.renderer.enable_segmentation_output(
#     map_by=["category_id", "instance", "name"],
#     default_values={"instance_color": [0.0, 0.0, 0.0]}  # fallback for objects like "World"
# )

# -------------------- Render the scene --------------------
data = bproc.renderer.render()

# -------------------- Print available outputs --------------------
print("🧾 Render keys:", data.keys())

# -------------------- Print Visible Instances --------------------
instance_maps = data["instance_segmaps"]
attribute_maps = data["instance_attribute_maps"]
color_maps = data["colors"]
print (f"\n📸 Rendered {len(color_maps)} frames with instance segmentation.")

for frame_idx, attr_map in enumerate(attribute_maps):
    print(f"\n🖼️ Frame {frame_idx}:")
    visible_ids = np.unique(instance_maps[frame_idx])

    for entry in attr_map:
        inst_id = entry["idx"]
        if inst_id in visible_ids:
            name = entry.get("name", "N/A")
            cat_id = entry.get("category_id", "N/A")
            print(f"  Instance ID {inst_id}: Name = {name}, Category ID = {cat_id}")

# -------------------- Save to HDF5 --------------------
bproc.writer.write_hdf5(args.output_dir, data)
