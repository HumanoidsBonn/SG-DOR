import blenderproc as bproc
import bpy
import argparse

def check_parenting(scene_path):
    bproc.init()
    # Load with bpy directly to check parenting more easily
    bpy.ops.wm.open_mainfile(filepath=scene_path)
    
    print(f"\n{'Object Name':25} | {'Parent':25}")
    print("-" * 55)
    
    for obj in bpy.data.objects:
        if obj.parent:
            print(f"{obj.name:25} | {obj.parent.name:25}")
        elif obj.name == 'stemM':
            # Check children of stemM
            children = [c.name for c in obj.children]
            print(f"{obj.name:25} | Children: {children[:10]}...")

    bproc.clean_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to the .blend file")
    args = parser.parse_args()
    check_parenting(args.file)
