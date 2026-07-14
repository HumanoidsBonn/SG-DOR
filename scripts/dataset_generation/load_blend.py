import blenderproc as bproc
import argparse
import numpy as np

# Argument parser for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('camera', help="Path to the camera pose file (each line: x y z rot_x rot_y rot_z in degrees)")
parser.add_argument('scene', help="Path to the .blend file")
parser.add_argument('output_dir', help="Directory where output will be saved")
args = parser.parse_args()

# Initialize BlenderProc
bproc.init()

# Load the .blend file
bproc.loader.load_blend(args.scene)

# objs = bproc.loader.load_blend(args.scene)
# print(f"Loaded {len(objs)} objects from the scene.")
# print("Objects in the scene:")
# for obj in objs:
#     print(f" - {obj.get_name()})")  

# Optional: set resolution
res_x, res_y = 360, 512
bproc.camera.set_resolution(res_x, res_y)

# (Optional) Add ambient background if scene is too dark
# bproc.world.set_world_background_hdr_img([1, 1, 1])

# Read camera positions from the file and add them
with open(args.camera, "r") as f:
    for line in f:
        values = [float(x) for x in line.strip().split()]
        position = values[:3]
        euler_rotation = values[3:]
        cam_pose = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(cam_pose)

bproc.camera.set_intrinsics_from_blender_params(
    lens_unit="MILLIMETERS",  # Lens unit
    lens=50,  # Focal length in mm
    clip_start=0.01,  # Near clipping plane
    clip_end=1000,  # Far clipping plane
)

bproc.camera.add_depth_of_field(
    focal_distance=2.062,  # Distance to the focus point
    fstop_value=105.0,  # Aperture value
    focal_point_obj=None
)

bproc.renderer.set_world_background([0.5, 0.5, 0.5])  # Set the world background color

# Enable output of normals and depth
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# read objects in the scene



# Render images from all camera poses
data = bproc.renderer.render()



# Write all outputs to output_dir
bproc.writer.write_hdf5(args.output_dir, data)
