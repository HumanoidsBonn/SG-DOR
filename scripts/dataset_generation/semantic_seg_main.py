import blenderproc as bproc
import argparse
import numpy as np

# Argument parser for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="camera_files/camera", help="Path to the camera pose file (each line: x y z rot_x rot_y rot_z in degrees)")
parser.add_argument('scene', nargs='?', default="blend/building_blocks/stem.blend", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="output/segment/", help="Path to where the final files, will be saved")
args = parser.parse_args()

# Initialize BlenderProc
bproc.init()

objs = bproc.loader.load_blend(args.scene,data_blocks=["collections", "objects"])

print(f"\nTotal loaded objects: {len(objs)}")

# -------------------- Group by name keywords --------------------
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
        # obj.set_cp("category_id", 1)
    elif "leafs_" in name:
        leaf_objs.append(obj)
        # obj.set_cp("category_id", 2)
    elif "shoots_" in name:
        shoot_objs.append(obj)
        # obj.set_cp("category_id", 3)
    elif "peduncles_" in name:
        pedduncle_objs.append(obj)
        # obj.set_cp("category_id", 4)
    elif "pepper_green" in name:
        pepper_green_objs.append(obj)
        # obj.set_cp("category_id", 5)
    elif "pepper_yellow" in name:
        pepper_yellow_objs.append(obj)
        # obj.set_cp("category_id", 6)
    elif "peppers_7" in name:
        pepper_red_objs.append(obj)
        # obj.set_cp("category_id", 7)
    elif "cuts_" in name:   
        cuts_objs.append(obj)
        # obj.set_cp("category_id", 8)


# -------------------- Print results --------------------
def print_group(group_name, group):
    print(f"\nGroup: {group_name} ({len(group)} objects)")
    for obj in group:
        col_names = [col.name for col in obj.blender_obj.users_collection]
        print(f"  - {obj.get_name()} (Collections: {col_names})")
        print(f"    - Category ID: {obj.get_cp('category_id') if obj.has_cp('category_id') else 'N/A'}")
    

# print_group("Stem", stem_objs)
# print_group("Leaf", leaf_objs)
# print_group("Shoot", shoot_objs)
# print_group("Pepper", pepper_green_objs)
# print_group("Pepper Yellow", pepper_yellow_objs)
# print_group("Pepper Red", pepper_red_objs)
# print_group("Peduncles", pedduncle_objs)
# print_group("Cuts", cuts_objs)


# Optional: set resolution
res_x, res_y = 360, 512
bproc.camera.set_resolution(res_x, res_y)

# Read camera positions from the file and add them
with open(args.camera, "r") as f:
    for line in f:
        values = [float(x) for x in line.strip().split()]
        position = values[:3]
        euler_rotation = values[3:]
        cam_pose = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(cam_pose)

# Set intrinsics
bproc.camera.set_intrinsics_from_blender_params(
    lens_unit="MILLIMETERS",
    lens=50,
    clip_start=0.01,
    clip_end=1000,
)

# Add optional depth of field
bproc.camera.add_depth_of_field(
    focal_distance=2.062,
    fstop_value=105.0,
    focal_point_obj=None
)

# Set background color
bproc.renderer.set_world_background([0.5, 0.5, 0.5])

# Enable normals and depth output and segmentation masks
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# render the whole pipeline
data = bproc.renderer.render()

print("Render keys:", data.keys())

# Get segmentation maps
instance_maps = data["instance_segmaps"]
attribute_maps = data["instance_attribute_maps"]

# For each frame, print visible instance info
for frame_idx, attr_map in enumerate(attribute_maps):
    print(f"\n🖼️ Frame {frame_idx}:")
    visible_ids = np.unique(instance_maps[frame_idx])
    
    for entry in attr_map:
        inst_id = entry["idx"]
        if inst_id in visible_ids:
            name = entry.get("name", "N/A")
            cat_id = entry.get("category_id", "N/A")
            print(f"  Instance ID {inst_id}: Name = {name}, Category ID = {cat_id}")

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
