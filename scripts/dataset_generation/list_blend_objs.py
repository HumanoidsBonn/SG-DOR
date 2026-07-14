import blenderproc as bproc
import bpy
import argparse

def list_all_objects(scene_path):
    bpy.ops.wm.open_mainfile(filepath=scene_path)
    print(f"\nTotal objects: {len(bpy.data.objects)}")
    for obj in bpy.data.objects:
        if 'shoot' in obj.name.lower():
            print(f"FOUND SHOOT: {obj.name} (Type: {obj.type})")
        elif 'pepper' in obj.name.lower() or 'leaf' in obj.name.lower() or 'stem' in obj.name.lower() or 'peduncle' in obj.name.lower():
            # Just to see some others
            pass
        else:
            # print(f"Other: {obj.name}")
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    list_all_objects(args.file)
