import blenderproc as bproc
import numpy as np
import argparse

def inspect_blend_units(scene_path):
    bproc.init()
    objs = bproc.loader.load_blend(scene_path)
    
    print(f"\n{'Name':20} | {'Type':10} | {'Dimensions (m)':30}")
    print("-" * 65)
    
    for obj in objs:
        b_obj = obj.blender_obj
        if b_obj.type == 'MESH':
            # Local dimensions in Blender
            dims = b_obj.dimensions
            print(f"{b_obj.name:20} | {b_obj.type:10} | {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f}")
            
    bproc.clean_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to the .blend file")
    args = parser.parse_args()
    inspect_blend_units(args.file)
