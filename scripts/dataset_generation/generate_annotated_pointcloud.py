import blenderproc as bproc
import numpy as np
import open3d as o3d
import os
import argparse
import json

def get_semantic_id(obj_name, obj_type):
    """Maps object name/type to semantic ID."""
    if obj_type == "stem":
        return 1
    elif obj_type == "leaf":
        return 2
    elif obj_type == "peduncle":
        return 4
    elif obj_type == "shoot":
        return 3
    elif obj_type == "fruit":
        return 5
    return 0  # Unknown

def generate_pointcloud(scene_path, json_path, output_path, voxel_size=0.0, compute_normals=True):
    bproc.init()
    
    # 1. Load scene
    print(f"Loading scene: {scene_path}")
    objs = bproc.loader.load_blend(scene_path)
    
    # 2. Load JSON and build ID Registry
    print(f"Loading JSON and building ID Registry: {json_path}")
    with open(json_path, 'r') as f:
        graph_data = json.load(f)
    
    # Map node_id (string) -> (instance_id, semantic_id)
    id_registry = {}
    instance_to_node = {} # For metadata
    
    # Handle list of scenes (chunked generation) or single scene dict
    scene_list = graph_data if isinstance(graph_data, list) else [graph_data]
    
    for scene_entry in scene_list:
        for node in scene_entry.get('nodes', []):
            nid = node['id']
            inst_id = node['instance_id']
            sem_id = node['semantic_id']
            id_registry[nid] = (inst_id, sem_id)
            instance_to_node[int(inst_id)] = nid
    
    all_points = []
    all_semantics = []
    all_instances = []

    # 3. Process objects using ID Registry
    unmapped_objects = []
    for obj in objs:
        if obj.blender_obj.type != 'MESH':
            continue
            
        node_id = obj.get_name()
        
        if node_id in id_registry:
            instance_id, semantic_id = id_registry[node_id]
        else:
            # Fallback or background
            unmapped_objects.append(node_id)
            instance_id = 0
            semantic_id = 0

        # Get vertices in world space
        matrix = np.asarray(obj.get_local2world_mat(), dtype=np.float32)
        mesh_data = obj.blender_obj.data
        
        for vert in mesh_data.vertices:
            local_pos = np.array([vert.co.x, vert.co.y, vert.co.z, 1.0], dtype=np.float32)
            world_pos = (matrix @ local_pos)[:3]
            
            all_points.append(world_pos)
            all_semantics.append(semantic_id)
            all_instances.append(instance_id)

    if unmapped_objects:
        print(f"Warning: {len(unmapped_objects)} objects were not found in the scene graph and labeled as background.")

    if not all_points:
        print("No points found!")
        bproc.clean_up()
        return

    # Force compact dtypes
    final_points = np.asarray(all_points, dtype=np.float32)
    final_semantics = np.asarray(all_semantics, dtype=np.uint8)
    final_instances = np.asarray(all_instances, dtype=np.uint16)

    # 4. Correct Voxelization (per instance)
    if voxel_size > 0:
        print(f"Applying voxel downsampling (size={voxel_size}m)...")
        voxel_size = float(voxel_size)
        
        voxel_indices = np.floor(final_points / voxel_size).astype(np.int32)
        
        # Include instance_id in key to prevent cross-instance collisions
        unique_voxels = {}
        for i, (vidx, inst) in enumerate(zip(voxel_indices, final_instances)):
            key = (int(inst), int(vidx[0]), int(vidx[1]), int(vidx[2]))
            if key not in unique_voxels:
                unique_voxels[key] = i
        
        keep_indices = np.fromiter(unique_voxels.values(), dtype=np.int64)
        final_points = final_points[keep_indices]
        final_semantics = final_semantics[keep_indices]
        final_instances = final_instances[keep_indices]
        print(f"Downsampled to {len(final_points)} points.")

    # 5. Save dict
    save_dict = {
        "points": final_points,
        "semantics": final_semantics,
        "instances": final_instances
    }

    # 6. Optional Normal Estimation
    if compute_normals:
        print("Estimating normals...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        radius = voxel_size * 2 if voxel_size > 0 else 0.01
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
        save_dict["normals"] = np.asarray(pcd.normals, dtype=np.float32)

    # 7. Use compressed .npz
    np.savez_compressed(output_path, **save_dict)
    
    # 8. Export Metadata
    meta_path = output_path.replace(".npz", ".meta.json")
    metadata = {
        "scene_source": os.path.abspath(scene_path),
        "graph_source": os.path.abspath(json_path),
        "voxel_size": voxel_size,
        "compute_normals": compute_normals,
        "point_count": len(final_points),
        "instance_id_to_node_id": instance_to_node,
        "dtypes": {
            "points": str(final_points.dtype),
            "semantics": str(final_semantics.dtype),
            "instances": str(final_instances.dtype)
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved to {output_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Final points: {len(final_points)}")

    bproc.clean_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help="Path to .blend file")
    parser.add_argument('--json', required=True, help="Path to scene graph JSON")
    parser.add_argument('--out', default="annotated_cloud.npz", help="Output .npz path")
    parser.add_argument('--voxel_size', type=float, default=0.0, help="Voxel size for downsampling (e.g. 0.005 for 5mm). 0 to disable.")
    parser.add_argument('--no_normals', action='store_true', help="Disable normal estimation")
    args = parser.parse_args()
    
    generate_pointcloud(args.scene, args.json, args.out, args.voxel_size, compute_normals=not args.no_normals)
