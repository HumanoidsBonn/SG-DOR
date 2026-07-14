import bpy
import os

template_path = "/home/data/menon/data_pepper_scenes/template/building_block_models.blend"

# Load the blend file (using bpy.open_main or library load)
# Since we are in the blender env, we can just open it or load nodes.
# Let's try to load it into a fresh scene.

def inspect_blend():
    with bpy.data.libraries.load(template_path) as (data_from, data_to):
        data_to.objects = data_from.objects

    print("\n--- Template Color Audit ---")
    for obj in data_to.objects:
        if obj and obj.type == 'MESH':
            # Skip objects not in the primary collections if possible, 
            # but let's just list all meshes.
            mats = obj.data.materials
            color_str = "No Material"
            if mats and mats[0]:
                c = mats[0].diffuse_color
                color_str = f"R:{c[0]:.3f}, G:{c[1]:.3f}, B:{c[2]:.3f}"
            
            print(f"Object: {obj.name:<25} | Viewport Color: {color_str}")

if __name__ == "__main__":
    inspect_blend()
