#!/usr/bin/env python3

import blenderproc as bproc
import argparse
import os
import random
import json
import bpy

import numpy as np
import copy
import time
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import open3d as o3d
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from shapely.geometry import Polygon
from shapely.ops import unary_union
import logging
import sys
from pathlib import Path

# Configure Logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Initializing industrial generation script...")


def resolve_dir(path):
    d = os.path.abspath(path)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Path to input .blend file containing the collections (stemC, attachC, leafC, pepperC)')
    parser.add_argument('--out', default='blend/debug/placed.blend', help='Output .blend path (default: ./blend/debug/placed.blend)')
    parser.add_argument('--graphs_out', default='graphs/scene_graphs.json', help='Output path for generated scene graphs JSON')
    parser.add_argument('--scene_start', type=int, default=0, help='Starting scene index (for chunked generation)')
    parser.add_argument('--scene_count', type=int, default=None, help='Number of scenes to generate in this run (overrides num_datapoints)')
    parser.add_argument('--config_override', default=None, help='Path to JSON file to override CONFIG values')
    parser.add_argument('--no-blend', action='store_true', help='Skip saving final .blend file to save disk space')
    parser.add_argument('--extract-pc', action='store_true', help='Extract point cloud directly after generation')
    parser.add_argument('--voxel-size', type=float, default=0.002, help='Voxel size for PC downsampling (default: 0.002m)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level (default: INFO)')
    parser.add_argument('--data_dir', default=None, help='Directory to save individual scene graphs, metadata, and point clouds (e.g. data/)')
    # Optional camera-based ray occlusion sidecars (off by default)
    parser.add_argument('--occl-cam', action='store_true', help='Enable camera-based ray occlusion sidecars')
    parser.add_argument('--occl-cam-out-dir', default=None, help='Output directory for ray occlusion sidecars')
    parser.add_argument('--occl-cam-px', type=float, default=None, help='Pixel size (meters) for ray grid (default: 0.004)')
    parser.add_argument('--occl-cam-delta', type=float, default=None, help='Depth tolerance for in-front test (default: 0.004)')
    parser.add_argument('--occl-cam-max-dist', type=float, default=None, help='Candidate leaf radius (default: occlusion_max_dist)')
    parser.add_argument('--occl-cam-min-pixels', type=int, default=None, help='Skip fruit if silhouette pixels < this')
    parser.add_argument('--occl-cam-max-pixels', type=int, default=None, help='Optional cap on silhouette pixels (downsample)')
    parser.add_argument('--occl-cam-debug', action='store_true', help='Write ray occlusion debug JSON summaries')
    parser.add_argument('--occl-cam-debug-dir', default=None, help='Output directory for ray occlusion debug files')
    parser.add_argument('--occl-cam-compare-pc-dir', default=None, help='Optional pointcloud zbuf sidecar dir for ray-vs-pc sanity checks')
    parser.add_argument('--occl-cam-progress-every', type=int, default=10, help='Log ray-occl fruit progress every N fruits')
    parser.add_argument('--occl-cam-mode', default=None, choices=['ortho', 'persp', 'both'],
                        help='Ray occlusion camera mode: ortho, persp, or both (default: ortho)')
    parser.add_argument('--occl-cam-persp-dist', type=float, default=None,
                        help='Perspective camera distance along -d (default: occl_cam_max_dist)')
    parser.add_argument('--occl-cam-jitter-small-deg', type=float, default=None,
                        help='Small jitter angle in degrees for per-direction variants (default: 5.0)')
    parser.add_argument('--occl-cam-jitter-large-deg', type=float, default=None,
                        help='Larger jitter angle in degrees for per-direction variants (default: 15.0)')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    # -----------------------
    # CONFIGURATION
    # -----------------------

    CONFIG = {
        # number of dataset entries (scene graphs) to generate
        'num_datapoints': 10,

        # rng seed
        'seed': 0,
        # linked duplicates share mesh data, False to make full copies
        'linked_duplicates': True,

        # number of stems to create
        'num_stems_range': (4, 6),
        # per-stem X step range (accumulated along a line)
        'stem_offset_x_range': (0.12, 0.25),
        # per-stem Y jitter range (relative to the original stem Y, no accumulation)
        'stem_offset_y_range': (-0.05, 0.10),

        # P(fruit) per attachment point
        'fruit_prob': 0.45,
        # per-placement uniform scale jitter
        'fruit_scale_range': (1.2, 1.8), # Aiming for 7-12cm (Base ~5.4cm)
        'leaf_scale_range': (0.7, 1.4),  # Aiming for 10-20cm for shoots/leaves
        # per-stem non-uniform scale jitter
        'stem_width_scale': (0.15, 0.20),
        'stem_height_scale': (0.30, 0.60), # Aiming for 84-168cm (Base ~280cm)
        # ranges for rotation after building block placement (degrees)
        'rot_x_deg': (-8.0, 8.0),
        'rot_y_deg': (-8.0, 8.0),
        'rot_z_deg': (0.0, 360.0),
        # range for Z offsets of attachment points
        'attach_z_offset': (-0.0035, 0.0035),
        # number of placement variants per attachment
        'placements_per_attachment': 6,
        # jitter range (degrees) applied around each evenly spaced rotation offset
        'placement_rot_jitter_deg': 30.0,
        # probability to force variants beyond the first to be leaves (overrides fruit_prob for those variants)
        'extra_variant_leaf_prob': 0.6,
        # bending of meshes
        'bend_enabled': False,
        'bend_axes': ('X', 'Y'),
        'bend_leaf_angle_deg': (-8.0, 8.0),
        'bend_peduncle_angle_deg': (-5.0, 5.0),
        
        # collision checks
        'collision_enabled': True,
        # maximum attempts to resolve collisions for a single attachment
        'collision_max_attempts': 3,
        # scales to try (uniform) when resolving collisions
        'collision_scale_candidates': [1.0, 0.9, 0.8],
        # accept small overlaps if the intersection volume is below this fraction of the new block's OBB volume
        'collision_minor_overlap_ratio': 0.12,
        # absolute volume threshold to guard against extremely small OBBs
        'collision_minor_overlap_abs': 2e-7,
        # spatial hash grid cell size for collision candidate pruning
        'collision_grid_cell_size': 0.05,

        # leaf -> fruit occlusion estimation
        'occlusion_enabled': True,
        'occlusion_max_dist': 0.20,
        # weight for inverse-square falloff using distance normalized by fruit size
        'occlusion_weight': 0.2,
        'occlusion_grid_cell_size': 0.05,
        'occlusion_print_stats': True,
        'occlusion_print_every': 1,
        # camera-based ray occlusion sidecars (off by default)
        'occl_cam_enabled': False,
        'occl_cam_px': 0.004,
        'occl_cam_delta': 0.004,
        'occl_cam_max_dist': 0.20,
        'occl_cam_k': 3,
        'occl_cam_min_pixels': 16,
        'occl_cam_max_pixels': None,
        'occl_cam_debug': False,
        'occl_cam_progress_every': 10,
        'occl_cam_leaves_only': True,
        'occl_cam_mode': 'ortho',
        'occl_cam_persp_dist': None,
        'occl_cam_jitter_small_deg': 5.0,
        'occl_cam_jitter_large_deg': 15.0,
    }

    # optional config override via JSON
    if args.config_override and os.path.exists(args.config_override):
        try:
            with open(args.config_override, 'r', encoding='utf-8') as f:
                override_cfg = json.load(f)
            if isinstance(override_cfg, dict):
                CONFIG.update(override_cfg)
                print(f"Loaded CONFIG override from {args.config_override}")
        except Exception as e:
            print(f"Warning: failed to load config override {args.config_override}: {e}")

    # CLI overrides for ray occlusion sidecars
    if args.occl_cam:
        CONFIG['occl_cam_enabled'] = True
    if args.occl_cam_px is not None:
        CONFIG['occl_cam_px'] = float(args.occl_cam_px)
    if args.occl_cam_delta is not None:
        CONFIG['occl_cam_delta'] = float(args.occl_cam_delta)
    if args.occl_cam_max_dist is not None:
        CONFIG['occl_cam_max_dist'] = float(args.occl_cam_max_dist)
    if args.occl_cam_min_pixels is not None:
        CONFIG['occl_cam_min_pixels'] = int(args.occl_cam_min_pixels)
    if args.occl_cam_max_pixels is not None:
        CONFIG['occl_cam_max_pixels'] = int(args.occl_cam_max_pixels)
    if args.occl_cam_debug:
        CONFIG['occl_cam_debug'] = True
    if args.occl_cam_progress_every is not None:
        CONFIG['occl_cam_progress_every'] = int(args.occl_cam_progress_every)
    if args.occl_cam_mode is not None:
        CONFIG['occl_cam_mode'] = args.occl_cam_mode
    if args.occl_cam_persp_dist is not None:
        CONFIG['occl_cam_persp_dist'] = float(args.occl_cam_persp_dist)
    if args.occl_cam_jitter_small_deg is not None:
        CONFIG['occl_cam_jitter_small_deg'] = float(args.occl_cam_jitter_small_deg)
    if args.occl_cam_jitter_large_deg is not None:
        CONFIG['occl_cam_jitter_large_deg'] = float(args.occl_cam_jitter_large_deg)

    # -----------------------
    #  HELPERS
    # -----------------------

    # Return XY translation components from a 4x4 transform matrix.
    def mat_pos(mat):
        return float(mat[0, 3]), float(mat[1, 3])

    # Ensure a collection with the given name exists and return it.
    def make_collection(name):
        if name in bpy.data.collections:
            return bpy.data.collections[name]
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
        return coll

    # Move an entity into a collection, unlinking it from previous ones.
    def move_entity_to_collection(entity, coll):
        for c in list(entity.blender_obj.users_collection):
            c.objects.unlink(entity.blender_obj)
        coll.objects.link(entity.blender_obj)

    # Collect all mesh objects belonging to an entity and its descendants.
    def mesh_objects_of(entity):
        meshes = []
        def add_mesh(ent):
            if ent.blender_obj.type == 'MESH':
                meshes.append(ent if isinstance(ent, MeshObject) else MeshObject(ent.blender_obj))
        add_mesh(entity)
        for child in entity.get_children(return_all_offspring=True):
            add_mesh(child)
        return meshes

    def internal_extract_pc(graph_nodes, output_path, voxel_size, compute_normals=True):
        """Integrated Point Cloud extraction from in-memory Blender objects."""
        # Map node_id (string) -> (instance_id, semantic_id, color)
        id_registry = {}
        # Palettes for biological colors (Mean RGB [0, 1])
        PALETTE_BIO = {
            1: [0.14, 0.45, 0.20], # Stem: Dark Green
            3: [0.12, 0.40, 0.18], # Peduncle: Dark Green
            2: [0.11, 0.50, 0.22], # Leaf: Deep Green
            5: [0.35, 0.65, 0.35], # Shoot: Vibrant Green
        }

        for n in graph_nodes:
            sid = n['semantic_id']
            iid = n['instance_id']
            
            # Deterministic seed for per-instance base variation
            inst_rng = random.Random(iid % 10000)
            
            if sid == 4: # Fruit
                # Varieties: 70% Red, 10% Yellow, 10% Orange, 10% Green
                r = inst_rng.random()
                if r < 0.70:
                    base = [0.85, 0.15, 0.15] # Red
                elif r < 0.80:
                    base = [0.95, 0.85, 0.25] # Yellow
                elif r < 0.90:
                    base = [1.0, 0.5, 0.0]    # Orange
                else:
                    base = [0.2, 0.6, 0.2]    # Green Fruit
            else:
                base = PALETTE_BIO.get(sid, [0.5, 0.5, 0.5])
            
            # Apply small subtle variation (±0.02)
            jitter = [inst_rng.uniform(-0.02, 0.02) for _ in range(3)]
            final_col = [np.clip(base[i] + jitter[i], 0.0, 1.0) for i in range(3)]
            col_uint8 = [int(c * 255) for c in final_col]
            
            id_registry[n['id']] = (iid, sid, col_uint8)
        
        all_points = []
        all_semantics = []
        all_instances = []
        all_colors = []

        # We iterate over all blender objects that are mesh types
        for obj_bl in bpy.data.objects:
            if obj_bl.type != 'MESH':
                continue
            
            node_id = obj_bl.name
            if node_id not in id_registry:
                continue
                
            instance_id, semantic_id, col = id_registry[node_id]
            matrix = np.asarray(obj_bl.matrix_world, dtype=np.float32)
            mesh_data = obj_bl.data
            
            # Vertices only extraction
            for vert in mesh_data.vertices:
                world_pos = (matrix @ np.array([vert.co.x, vert.co.y, vert.co.z, 1.0], dtype=np.float32))[:3]
                all_points.append(world_pos)
                all_semantics.append(semantic_id)
                all_instances.append(instance_id)
                all_colors.append(col)

        if not all_points:
            return None

        final_points = np.asarray(all_points, dtype=np.float32)
        final_semantics = np.asarray(all_semantics, dtype=np.uint8)
        final_instances = np.asarray(all_instances, dtype=np.uint16)
        final_colors = np.asarray(all_colors, dtype=np.uint8)

        if voxel_size > 0:
            voxel_indices = np.floor(final_points / voxel_size).astype(np.int32)
            unique_voxels = {}
            for i, (vidx, inst) in enumerate(zip(voxel_indices, final_instances)):
                key = (int(inst), int(vidx[0]), int(vidx[1]), int(vidx[2]))
                if key not in unique_voxels:
                    unique_voxels[key] = i
            keep_indices = np.fromiter(unique_voxels.values(), dtype=np.int64)
            final_points = final_points[keep_indices]
            final_semantics = final_semantics[keep_indices]
            final_instances = final_instances[keep_indices]
            final_colors = final_colors[keep_indices]

        save_dict = {
            "points": final_points,
            "semantics": final_semantics,
            "instances": final_instances,
            "colors": final_colors
        }

        if compute_normals:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_points)
            radius = voxel_size * 2 if voxel_size > 0 else 0.01
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
            save_dict["normals"] = np.asarray(pcd.normals, dtype=np.float32)

        np.savez_compressed(output_path, **save_dict)
        return len(final_points)

    def find_child_mesh(entity, name_substr):
        """Return first mesh child whose name contains the given substring."""
        for child in entity.get_children(return_all_offspring=True):
            if child.blender_obj.type == 'MESH' and name_substr in child.get_name():
                return child if isinstance(child, MeshObject) else MeshObject(child.blender_obj)
        return None

    def is_leaf_mesh(mesh_obj):
        name = mesh_obj.get_name().lower()
        return 'leaf' in name and 'shoot' not in name

    def is_shoot_mesh(mesh_obj):
        name = mesh_obj.get_name().lower()
        return 'shoot' in name

    def ensure_single_user(mesh_obj):
        """Ensure the mesh datablock is single-user so modifiers can be applied on linked duplicates."""
        if mesh_obj is None:
            return None
        obj = mesh_obj.blender_obj
        if obj.data and obj.data.users > 1:
            obj.data = obj.data.copy()
        return mesh_obj

    def apply_bend_modifier(mesh_obj, angle_range_deg, axes=('X', 'Y')):
        """Apply a SimpleDeform BEND modifier with a small random angle to the given mesh."""
        if mesh_obj is None:
            return
        try:
            mesh_obj = ensure_single_user(mesh_obj)
            obj = mesh_obj.blender_obj
            angle = np.deg2rad(random.uniform(angle_range_deg[0], angle_range_deg[1]))
            axis = random.choice(axes)
            prev_active = bpy.context.view_layer.objects.active
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            mod = obj.modifiers.new(name="AutoBend", type='SIMPLE_DEFORM')
            mod.deform_method = 'BEND'
            mod.deform_axis = axis
            mod.angle = angle
            # move modifier to the top to avoid Blender warning about apply order
            try:
                obj.modifiers.move(len(obj.modifiers) - 1, 0)
            except Exception:
                pass
            bpy.ops.object.modifier_apply(modifier=mod.name)
            obj.select_set(False)
            bpy.context.view_layer.objects.active = prev_active
        except Exception:
            # if anything goes wrong, skip bending silently
            pass

    # Gather all world-space vertices for a mesh object.
    def world_vertices_for_mesh(mesh_obj: MeshObject):
        mat = Matrix(mesh_obj.get_local2world_mat())
        verts = []
        for v in mesh_obj.get_mesh().vertices:
            w = mat @ v.co
            verts.append([w.x, w.y, w.z])
        if not verts:
            return np.zeros((0, 3))
        return np.array(verts, dtype=float)

    # Gather all world-space vertices for an entity.
    def gather_world_vertices(entity):
        verts = []
        for mesh in mesh_objects_of(entity):
            verts.append(world_vertices_for_mesh(mesh))
        if not verts:
            return np.zeros((0, 3))
        return np.vstack(verts)

    # Compute an oriented bounding box from point samples plus AABB for coarse pruning.
    def compute_obb_from_points(verts: np.ndarray):
        if verts.shape[0] < 3:
            center = verts.mean(axis=0) if verts.shape[0] > 0 else np.zeros(3)
            axes = np.eye(3)
            if verts.shape[0] > 0:
                mins = verts.min(axis=0)
                maxs = verts.max(axis=0)
                extents = (maxs - mins)
            else:
                extents = np.zeros(3)
            obb = None
            aabb_min = mins if verts.shape[0] > 0 else np.zeros(3)
            aabb_max = maxs if verts.shape[0] > 0 else np.zeros(3)
        else:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(verts)
            obb = pc.get_oriented_bounding_box()
            center = np.array(obb.center)
            axes = np.array(obb.R)
            extents = np.array(obb.extent)
            aabb_min = verts.min(axis=0)
            aabb_max = verts.max(axis=0)
        half_extents = extents * 0.5
        volume = float(np.prod(extents)) if extents.size else 0.0
        return {
            'center': center,
            'axes': axes,
            'extents': extents,
            'half_extents': half_extents,
            'volume': volume,
            'obb': obb,
            'aabb_min': aabb_min,
            'aabb_max': aabb_max
        }

    # Generate cached oriented bounding box entries for all meshes of an entity.
    def obb_entries_for_entity(entity):
        entries = []
        for mesh in mesh_objects_of(entity):
            verts = world_vertices_for_mesh(mesh)
            obb = compute_obb_from_points(verts)
            obb['mesh'] = mesh
            obb['meshes'] = [mesh]
            obb['key'] = mesh.get_name()
            entries.append(obb)
        return entries

    def group_obb_entry_for_entity(entity):
        meshes = mesh_objects_of(entity)
        verts = []
        for mesh in meshes:
            verts.append(world_vertices_for_mesh(mesh))
        if not verts:
            all_verts = np.zeros((0, 3))
        else:
            all_verts = np.vstack(verts)
        obb = compute_obb_from_points(all_verts)
        obb['mesh'] = meshes[0] if meshes else None
        obb['meshes'] = meshes
        obb['key'] = f"{entity.get_name()}_group"
        return obb

    # Average diffuse colors across all materials of an entity.
    def average_material_color(entity):
        colors = []
        for mesh in mesh_objects_of(entity):
            mats = mesh.blender_obj.data.materials
            for mat in mats:
                if mat and hasattr(mat, 'diffuse_color'):
                    colors.append(mat.diffuse_color[:3])
        if not colors:
            return None
        arr = np.array(colors, dtype=float)
        return arr.mean(axis=0).tolist()

    # Build a node dictionary with geometric attributes for graph export.
    def build_node_entry(entity, node_type, node_id, prototype_name, instance_id):
        verts = gather_world_vertices(entity)
        obb = compute_obb_from_points(verts)
        axes = obb['axes']
        extents = obb['extents']
        if extents.size and axes.size:
            idx = int(np.argmax(extents)) if extents.ndim == 1 else 0
            orientation_vec = axes[:, idx].tolist()
        else:
            orientation_vec = [0.0, 0.0, 1.0]
        color = average_material_color(entity)
        
        # Semantic ID mapping (Continuous indices)
        semantic_map = {"stem": 1, "leaf": 2, "peduncle": 3, "fruit": 4, "shoot": 5}
        semantic_id = semantic_map.get(node_type, 0)
        
        return {
            'id': node_id,
            'instance_id': instance_id,
            'semantic_id': semantic_id,
            'type': node_type,
            'prototype': prototype_name,
            'obb': {
                'center': obb['center'].tolist(),
                'axes': obb['axes'].tolist(),
                'extents': obb['extents'].tolist()
            },
            'centroid': obb['center'].tolist(),
            'orientation': orientation_vec,
            'color': color,
        }

    # Compute SAT overlap between two oriented bounding boxes; return (collides, overlap_volume_approx).
    def obb_overlap(entry_a, entry_b):
        center_a = entry_a['center']
        center_b = entry_b['center']
        axes_a = entry_a['axes']
        axes_b = entry_b['axes']
        half_a = entry_a['half_extents']
        half_b = entry_b['half_extents']
        if axes_a.size == 0 or axes_b.size == 0:
            return False, 0.0
        ra = axes_a
        rb = axes_b
        R = ra.T @ rb
        absR = np.abs(R) + 1e-8
        t = center_b - center_a
        t_a = ra.T @ t
        overlap_a = np.zeros(3)
        for i in range(3):
            radius_a = half_a[i]
            radius_b = half_b[0] * absR[i, 0] + half_b[1] * absR[i, 1] + half_b[2] * absR[i, 2]
            if abs(t_a[i]) > radius_a + radius_b:
                return False, 0.0
            overlap_a[i] = max(0.0, radius_a + radius_b - abs(t_a[i]))
        t_b = rb.T @ t
        for j in range(3):
            radius_a = half_a[0] * absR[0, j] + half_a[1] * absR[1, j] + half_a[2] * absR[2, j]
            radius_b = half_b[j]
            if abs(t_b[j]) > radius_a + radius_b:
                return False, 0.0
        for i in range(3):
            for j in range(3):
                ra_term = half_a[(i + 1) % 3] * absR[(i + 2) % 3, j] + half_a[(i + 2) % 3] * absR[(i + 1) % 3, j]
                rb_term = half_b[(j + 1) % 3] * absR[i, (j + 2) % 3] + half_b[(j + 2) % 3] * absR[i, (j + 1) % 3]
                t_term = abs(t_a[(i + 2) % 3] * R[(i + 1) % 3, j] - t_a[(i + 1) % 3] * R[(i + 2) % 3, j])
                if t_term > ra_term + rb_term:
                    return False, 0.0
        overlap_volume = overlap_a[0] * overlap_a[1] * overlap_a[2]
        return True, overlap_volume

    # Spatial hash helpers for pruning collision candidates.
    def cells_for_aabb(aabb_min, aabb_max, cell_size):
        mins = np.floor(np.array(aabb_min) / cell_size).astype(int)
        maxs = np.floor(np.array(aabb_max) / cell_size).astype(int)
        cells = []
        for ix in range(mins[0], maxs[0] + 1):
            for iy in range(mins[1], maxs[1] + 1):
                for iz in range(mins[2], maxs[2] + 1):
                    cells.append((ix, iy, iz))
        return cells

    def add_entry_to_grid(entry, grid, cell_size):
        for cell in cells_for_aabb(entry['aabb_min'], entry['aabb_max'], cell_size):
            grid.setdefault(cell, []).append(entry)

    def candidates_from_grid(entry, grid, cell_size):
        seen = set()
        candidates = []
        for cell in cells_for_aabb(entry['aabb_min'], entry['aabb_max'], cell_size):
            for other in grid.get(cell, []):
                key = other.get('key')
                if key is None:
                    mesh = other.get('mesh')
                    key = mesh.get_name() if mesh is not None else str(id(other))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(other)
        return candidates

    # Return 8 OBB corners from center/axes/extents.
    def obb_corners(center, axes, extents):
        half = extents * 0.5
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = (
                        axes[:, 0] * (sx * half[0])
                        + axes[:, 1] * (sy * half[1])
                        + axes[:, 2] * (sz * half[2])
                    )
                    corners.append(center + offset)
        return np.array(corners, dtype=float)

    # Build 18 canonical directions (axes + diagonals).
    def build_occlusion_dirs():
        base = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]
        dirs = []
        for v in base:
            dirs.append(v)
            dirs.append(-v)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = base[i] * s1 + base[j] * s2
                    v = v / np.linalg.norm(v)
                    dirs.append(v)
        return dirs

    # Build per-direction orthonormal bases for projections.
    def build_dir_bases(dirs):
        bases = []
        z = np.array([0.0, 0.0, 1.0], dtype=float)
        x = np.array([1.0, 0.0, 0.0], dtype=float)
        for d in dirs:
            d = d / np.linalg.norm(d)
            tmp = z if abs(np.dot(d, z)) < 0.9 else x
            u = np.cross(tmp, d)
            if np.linalg.norm(u) < 1e-6:
                tmp = x
                u = np.cross(tmp, d)
            u = u / np.linalg.norm(u)
            v = np.cross(d, u)
            v = v / np.linalg.norm(v)
            bases.append((d, u, v))
        return bases

    def _normalize_vec(v):
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v
        return v / n

    def _random_perp_axis(rng, d):
        # Sample a random axis perpendicular to d.
        v = rng.normal(size=3).astype(float)
        v = v - d * float(np.dot(d, v))
        if np.linalg.norm(v) < 1e-6:
            # Fallback to a fixed axis.
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(tmp, d))) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            v = np.cross(d, tmp)
        return _normalize_vec(v)

    def _rotate_vec(v, axis, angle_rad):
        # Rodrigues' rotation formula.
        axis = _normalize_vec(axis)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1.0 - c)

    def build_camera_dirs_with_jitter(base_dirs, jitter_small_deg, jitter_large_deg, seed):
        rng = np.random.default_rng(seed)
        dirs_all = []
        base_idx = []
        jitter_level = []
        jitter_deg = []
        jitter_axis = []
        for i, d in enumerate(base_dirs):
            d = _normalize_vec(np.array(d, dtype=float))
            # Base direction.
            dirs_all.append(d)
            base_idx.append(i)
            jitter_level.append(0)
            jitter_deg.append(0.0)
            jitter_axis.append([0.0, 0.0, 0.0])
            # Jittered variants.
            for lvl, deg in ((1, jitter_small_deg), (2, jitter_large_deg)):
                if deg is None or deg <= 0.0:
                    continue
                axis = _random_perp_axis(rng, d)
                d_j = _rotate_vec(d, axis, np.deg2rad(deg))
                dirs_all.append(_normalize_vec(d_j))
                base_idx.append(i)
                jitter_level.append(lvl)
                jitter_deg.append(float(deg))
                jitter_axis.append(axis.tolist())
        dirs_all = np.asarray(dirs_all, dtype=np.float32)
        meta = {
            "base_index": np.asarray(base_idx, dtype=np.int32),
            "jitter_level": np.asarray(jitter_level, dtype=np.int8),
            "jitter_deg": np.asarray(jitter_deg, dtype=np.float32),
            "jitter_axis": np.asarray(jitter_axis, dtype=np.float32),
        }
        return dirs_all, meta

    # Build a local frame for a fruit, with +X pointing toward its stem (in XY plane).
    def fruit_local_frame(fruit_center, stem_center):
        x_vec = np.array(
            [stem_center[0] - fruit_center[0], stem_center[1] - fruit_center[1], 0.0],
            dtype=float,
        )
        if np.linalg.norm(x_vec) < 1e-6:
            x_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        x_unit = x_vec / np.linalg.norm(x_vec)
        z_unit = np.array([0.0, 0.0, 1.0], dtype=float)
        y_vec = np.cross(z_unit, x_unit)
        if np.linalg.norm(y_vec) < 1e-6:
            y_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        y_unit = y_vec / np.linalg.norm(y_vec)
        x_unit = np.cross(y_unit, z_unit)
        x_unit = x_unit / (np.linalg.norm(x_unit) + 1e-8)
        return np.vstack([x_unit, y_unit, z_unit])

    # Build a simple spatial grid for leaf candidates.
    def build_leaf_grid(leaf_entries, cell_size):
        grid = {}
        for entry in leaf_entries:
            cell = tuple(np.floor(entry['center'] / cell_size).astype(int).tolist())
            grid.setdefault(cell, []).append(entry)
        return grid

    def leaf_candidates_from_grid(fruit_center, grid, cell_size, max_dist):
        if not grid:
            return []
        base = np.floor(fruit_center / cell_size).astype(int)
        rad = int(np.ceil(max_dist / cell_size))
        seen = set()
        candidates = []
        for ix in range(base[0] - rad, base[0] + rad + 1):
            for iy in range(base[1] - rad, base[1] + rad + 1):
                for iz in range(base[2] - rad, base[2] + rad + 1):
                    for entry in grid.get((ix, iy, iz), []):
                        if entry['id'] in seen:
                            continue
                        seen.add(entry['id'])
                        dist = np.linalg.norm(entry['center'] - fruit_center)
                        if dist <= max_dist:
                            candidates.append(entry)
        return candidates

    # Compute occlusion overlaps for leaf-fruit pairs.
    def _cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _convex_hull(points_2d):
        # Compute the 2D convex hull using the monotonic chain algorithm
        # (a.k.a. Andrew's algorithm). This yields the silhouette polygon
        # of the projected OBB corners in the u/v plane.
        if len(points_2d) <= 1:
            return points_2d
        pts = sorted(set((float(p[0]), float(p[1])) for p in points_2d))
        if len(pts) <= 1:
            return pts

        def build_half(seq):
            # Build one half of the hull (lower or upper) using a stack.
            half = []
            for p in seq:
                while len(half) >= 2 and _cross_2d(
                    (half[-1][0] - half[-2][0], half[-1][1] - half[-2][1]),
                    (p[0] - half[-1][0], p[1] - half[-1][1]),
                ) <= 0:
                    half.pop()
                half.append(p)
            return half

        lower = build_half(pts)
        upper = build_half(reversed(pts))
        hull = lower[:-1] + upper[:-1]
        return hull

    def _polygon_area(poly):
        # Shoelace formula for polygon area (convex or general).
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _clip_polygon(subject, clipper):
        # Sutherland–Hodgman polygon clipping for convex polygons.
        # This computes the intersection polygon between two silhouettes.
        if not subject or not clipper:
            return []

        def inside(p, a, b):
            return _cross_2d((b[0] - a[0], b[1] - a[1]), (p[0] - a[0], p[1] - a[1])) >= 0

        def intersection(p1, p2, a, b):
            s = np.array(p1, dtype=float)
            e = np.array(p2, dtype=float)
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            d1 = e - s
            d2 = b - a
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 1e-8:
                return e.tolist()
            t = ((a[0] - s[0]) * d2[1] - (a[1] - s[1]) * d2[0]) / denom
            return (s + t * d1).tolist()

        output = subject
        for i in range(len(clipper)):
            # Clip the subject polygon by each edge of the clipper.
            input_list = output
            if not input_list:
                break
            output = []
            a = clipper[i]
            b = clipper[(i + 1) % len(clipper)]
            prev = input_list[-1]
            for curr in input_list:
                if inside(curr, a, b):
                    if not inside(prev, a, b):
                        output.append(intersection(prev, curr, a, b))
                    output.append(curr)
                elif inside(prev, a, b):
                    output.append(intersection(prev, curr, a, b))
                prev = curr
        return output

    def compute_leaf_fruit_occlusion(
        graph_nodes,
        graph_edges,
        occlusion_dirs,
        occlusion_dir_bases,
        max_dist,
        weight,
        grid_cell_size,
    ):
        node_by_id = {n['id']: n for n in graph_nodes}
        attached_to = {
            e['source']: e['target']
            for e in graph_edges
            if e.get('label') == 'attached_to'
        }
        leaf_entries = []
        for node in graph_nodes:
            if node['type'] != 'leaf':
                continue
            center = np.array(node['centroid'], dtype=float)
            axes = np.array(node['obb']['axes'], dtype=float)
            extents = np.array(node['obb']['extents'], dtype=float)
            corners = obb_corners(center, axes, extents)
            leaf_entries.append({'id': node['id'], 'center': center, 'corners': corners})
        leaf_grid = build_leaf_grid(leaf_entries, grid_cell_size)

        occlusion_edges = []
        fruit_count = 0
        fruit_with_any = 0
        total_candidates = 0
        num_dirs = len(occlusion_dirs)
        raw_count = 0
        raw_sum = 0.0
        raw_min = None
        raw_max = 0.0
        w_count = 0
        w_sum = 0.0
        w_min = None
        w_max = 0.0

        for node in graph_nodes:
            if node['type'] != 'fruit':
                continue
            fruit_id = node['id']
            ped_id = attached_to.get(fruit_id)
            stem_id = attached_to.get(ped_id) if ped_id else None
            stem_node = node_by_id.get(stem_id) if stem_id else None
            if stem_node is None:
                continue
            fruit_count += 1
            fruit_center = np.array(node['centroid'], dtype=float)
            stem_center = np.array(stem_node['centroid'], dtype=float)
            R = fruit_local_frame(fruit_center, stem_center)
            fruit_axes = np.array(node['obb']['axes'], dtype=float)
            fruit_extents = np.array(node['obb']['extents'], dtype=float)
            fruit_radius = 0.5 * float(np.linalg.norm(fruit_extents))
            if fruit_radius <= 1e-6:
                fruit_radius = 1e-6
            fruit_corners = obb_corners(fruit_center, fruit_axes, fruit_extents)
            fruit_local = (R @ (fruit_corners - fruit_center).T).T
            fruit_proj = []
            for (_d, u, v) in occlusion_dir_bases:
                u_vals = fruit_local @ u
                v_vals = fruit_local @ v
                fruit_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
                fruit_hull = _convex_hull(fruit_pts)
                fruit_area = _polygon_area(fruit_hull)
                fruit_proj.append((fruit_hull, fruit_area))

            union_parts = [[] for _ in range(num_dirs)]

            candidates = leaf_candidates_from_grid(fruit_center, leaf_grid, grid_cell_size, max_dist)
            total_candidates += len(candidates)
            fruit_has_any = False

            for leaf in candidates:
                dist = float(np.linalg.norm(leaf['center'] - fruit_center))
                dist_ratio = dist / fruit_radius
                weight_fac = 1.0 / (1.0 + weight * dist_ratio * dist_ratio)
                leaf_center_local = R @ (leaf['center'] - fruit_center)
                leaf_local = (R @ (leaf['corners'] - fruit_center).T).T
                raw_vals = []
                weighted_vals = []
                has_overlap = False

                for dir_id, ((d, u, v), fp) in enumerate(zip(occlusion_dir_bases, fruit_proj)):
                    fruit_hull, fruit_area = fp
                    raw = 0.0
                    if float(np.dot(leaf_center_local, d)) > 0.0 and fruit_area > 0.0:
                        u_vals = leaf_local @ u
                        v_vals = leaf_local @ v
                        leaf_pts = list(zip(u_vals.tolist(), v_vals.tolist()))
                        leaf_hull = _convex_hull(leaf_pts)
                        if leaf_hull:
                            inter = _clip_polygon(leaf_hull, fruit_hull)
                            inter_area = _polygon_area(inter)
                            raw = inter_area / fruit_area if fruit_area > 0.0 else 0.0

                            if inter_area > 0.0 and inter and len(inter) >= 3:
                                try:
                                    p = Polygon(inter).buffer(0)
                                    if p.is_valid and not p.is_empty and p.area > 1e-12:
                                        union_parts[dir_id].append(p)
                                except Exception:
                                    pass
                    raw = float(raw)
                    weighted = raw * weight_fac
                    raw_vals.append(raw)
                    weighted_vals.append(float(weighted))
                    if raw > 0.0:
                        has_overlap = True
                        raw_count += 1
                        raw_sum += raw
                        raw_min = raw if raw_min is None or raw < raw_min else raw_min
                        raw_max = raw if raw > raw_max else raw_max
                        w_count += 1
                        w_sum += weighted
                        w_min = weighted if w_min is None or weighted < w_min else w_min
                        w_max = weighted if weighted > w_max else w_max

                if has_overlap:
                    occlusion_edges.append(
                        {
                            'leaf_id': leaf['id'],
                            'fruit_id': fruit_id,
                            'dist': dist,
                            'raw_overlap': raw_vals,
                            'weighted_overlap': weighted_vals,
                        }
                    )
                    fruit_has_any = True

            if fruit_has_any:
                fruit_with_any += 1

            fruit_union_raw = []
            for dir_id, (_fh, fruit_area) in enumerate(fruit_proj):
                if fruit_area <= 0.0:
                    fruit_union_raw.append(0.0)
                    continue
                parts = union_parts[dir_id]
                if not parts:
                    fruit_union_raw.append(0.0)
                    continue
                try:
                    U = unary_union(parts)
                    union_area = float(U.area) if U is not None else 0.0
                    fruit_union_raw.append(union_area / float(fruit_area))
                except Exception:
                    fruit_union_raw.append(0.0)

            # Hybrid Injection: Store directly in node entry
            node['occlusion_targets'] = {
                'union_raw': fruit_union_raw,
                'weight_mode': 'inv_sq_fruit_radius'
            }

        stats = {
            'fruit_count': fruit_count,
            'fruit_with_any': fruit_with_any,
            'avg_candidates': (total_candidates / fruit_count) if fruit_count > 0 else 0.0,
            'raw_min': raw_min if raw_min is not None else 0.0,
            'raw_max': raw_max if raw_count > 0 else 0.0,
            'raw_avg': (raw_sum / raw_count) if raw_count > 0 else 0.0,
            'weighted_min': w_min if w_min is not None else 0.0,
            'weighted_max': w_max if w_count > 0 else 0.0,
            'weighted_avg': (w_sum / w_count) if w_count > 0 else 0.0,
            'edge_count': len(occlusion_edges),
            'avg_leaves_per_fruit': (len(occlusion_edges) / fruit_count) if fruit_count > 0 else 0.0,
        }
        occlusion = {
            'dirs': [d.tolist() for d in occlusion_dirs],
            'edges': occlusion_edges,
            'max_dist': max_dist,
            'weight': weight,
            'weight_mode': 'inv_sq_fruit_radius',
        }
        return occlusion, stats

    # -----------------------
    # Ray-based camera occlusion (sidecar generation)
    # -----------------------
    def meshes_for_node_id(node_id):
        obj = bpy.data.objects.get(node_id)
        if obj is None:
            return []
        ent = MeshObject(obj)
        return mesh_objects_of(ent)

    def bvh_for_mesh(mesh_obj, bvh_cache, depsgraph):
        key = mesh_obj.blender_obj.name_full
        if key in bvh_cache:
            return bvh_cache[key]
        bvh = None
        inv = None
        inv3 = None
        mat = None
        try:
            obj_eval = mesh_obj.blender_obj.evaluated_get(depsgraph)
            mat = obj_eval.matrix_world.copy()
            inv = mat.inverted()
            inv3 = inv.to_3x3()
            mesh_eval = obj_eval.to_mesh()
            if mesh_eval and mesh_eval.polygons:
                verts = [v.co.copy() for v in mesh_eval.vertices]
                polys = [p.vertices[:] for p in mesh_eval.polygons]
                bvh = BVHTree.FromPolygons(verts, polys)
            obj_eval.to_mesh_clear()
        except Exception:
            bvh = None
        bvh_cache[key] = (bvh, inv, inv3, mat)
        return bvh_cache[key]

    def ray_cast_min_dist(meshes, origin, direction, bvh_cache, depsgraph, max_dist=100.0):
        best = None
        for mesh in meshes:
            bvh, inv, inv3, mat = bvh_for_mesh(mesh, bvh_cache, depsgraph)
            if bvh is None or inv is None or inv3 is None or mat is None:
                continue
            # Cast in local space, then convert hit distance back to world.
            origin_local = inv @ origin
            dir_local = (inv3 @ direction).normalized()
            hit = bvh.ray_cast(origin_local, dir_local, max_dist)
            if hit is None or hit[0] is None or hit[3] is None:
                continue
            hit_world = mat @ (origin_local + dir_local * hit[3])
            dist = (hit_world - origin).length
            if best is None or dist < best:
                best = dist
        return best

    def compute_ray_occlusion_sidecar(
        scene_id,
        graph_nodes,
        graph_edges,
        occlusion_dirs,
        occlusion_dir_bases,
        out_dir,
        max_dist,
        pix,
        delta,
        k=3,
        min_pixels=16,
        max_pixels=None,
        debug=False,
        debug_dir=None,
        compare_pc_dir=None,
        progress_every=10,
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir_path = None
        if debug:
            debug_dir_path = Path(debug_dir) if debug_dir else (out_dir / "debug")
            debug_dir_path.mkdir(parents=True, exist_ok=True)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        node_by_id = {n['id']: n for n in graph_nodes}
        attached_to = {
            e['source']: e['target']
            for e in graph_edges
            if e.get('label') == 'attached_to'
        }

        # Build leaf entries + grid for candidates.
        leaf_entries = []
        for node in graph_nodes:
            if node['type'] != 'leaf':
                continue
            center = np.array(node['centroid'], dtype=float)
            leaf_entries.append({'id': node['id'], 'center': center})
        leaf_grid = build_leaf_grid(leaf_entries, CONFIG['occlusion_grid_cell_size'])

        # Camera directions (base + jittered variants).
        cam_mode = CONFIG.get('occl_cam_mode', 'ortho')
        if cam_mode not in ('ortho', 'persp', 'both'):
            cam_mode = 'ortho'
        jitter_small = float(CONFIG.get('occl_cam_jitter_small_deg', 0.0) or 0.0)
        jitter_large = float(CONFIG.get('occl_cam_jitter_large_deg', 0.0) or 0.0)
        jitter_seed = int(CONFIG.get('seed', 0)) * 1000 + int(scene_id)
        dirs_base = np.asarray(occlusion_dirs, dtype=np.float32)
        dirs_all, dir_meta = build_camera_dirs_with_jitter(
            dirs_base, jitter_small, jitter_large, jitter_seed
        )
        occlusion_dir_bases_all = build_dir_bases(dirs_all) if dirs_all.size else []
        num_dirs = len(dirs_all)

        # Mode setup.
        modes_to_run = []
        if cam_mode in ('ortho', 'both'):
            modes_to_run.append(('ortho', False))
        if cam_mode in ('persp', 'both'):
            modes_to_run.append(('persp', True))

        def _init_mode_data():
            return {
                "fruit_iids": [],
                "fruit_union": [],
                "fruit_pix": [],
                "pair_fruit_iid": [],
                "pair_leaf_iid": [],
                "pair_dist": [],
                "pair_potential": [],
                "pair_exclusive": [],
                "pair_k_rank": [],
                "debug_summary": {
                    "scene_id": scene_id,
                    "fruit_total": 0,
                    "fruit_processed": 0,
                    "fruit_skipped_min_pixels": 0,
                    "fruit_skipped_no_mesh": 0,
                    "fruit_skipped_no_candidates": 0,
                    "pix_min": None,
                    "pix_max": None,
                    "u_cam_mean": None,
                    "u_cam_max": None,
                },
                "debug_pixels": [] if debug_dir_path is not None else None,
                "debug_dir_stats": None,
                "debug_center_hits": None,
            }

        mode_data = {}
        for mode_name, _use_persp in modes_to_run:
            mode_data[mode_name] = _init_mode_data()

        bvh_cache = {}
        fruit_nodes = [n for n in graph_nodes if n['type'] == 'fruit']
        total_fruits = len(fruit_nodes)
        for m in mode_data.values():
            m["debug_summary"]["fruit_total"] = total_fruits

        debug_fruit_local_bvh = None
        debug_fruit_local_inv = None
        debug_fruit_local_inv3 = None

        persp_cam_dist = CONFIG.get('occl_cam_persp_dist', None)
        if persp_cam_dist is None:
            persp_cam_dist = max_dist
        persp_cam_dist = float(persp_cam_dist)

        for f_idx, node in enumerate(fruit_nodes):
            fruit_id = node['id']
            fruit_iid = int(node['instance_id'])
            ped_id = attached_to.get(fruit_id)
            stem_id = attached_to.get(ped_id) if ped_id else None
            stem_node = node_by_id.get(stem_id) if stem_id else None
            if stem_node is None:
                continue
            fruit_center = np.array(node['centroid'], dtype=float)
            stem_center = np.array(stem_node['centroid'], dtype=float)
            R = fruit_local_frame(fruit_center, stem_center)

            fruit_meshes = meshes_for_node_id(fruit_id)
            if not fruit_meshes:
                for m in mode_data.values():
                    m["debug_summary"]["fruit_skipped_no_mesh"] += 1
                    if m["debug_pixels"] is not None:
                        m["debug_pixels"].append({
                            "fruit_id": fruit_id,
                            "fruit_iid": fruit_iid,
                            "candidate_leaves": 0,
                            "pix_counts": [0] * num_dirs,
                            "skipped_reason": "no_mesh",
                        })
                continue

            # Candidate leaves by centroid distance.
            candidates = leaf_candidates_from_grid(
                fruit_center, leaf_grid, CONFIG['occlusion_grid_cell_size'], max_dist
            )
            if not candidates:
                # Still record union=0 for this fruit (no candidates).
                for m in mode_data.values():
                    m["fruit_iids"].append(fruit_iid)
                    m["fruit_union"].append(np.zeros((num_dirs,), dtype=np.float32))
                    m["fruit_pix"].append(np.zeros((num_dirs,), dtype=np.int32))
                    m["debug_summary"]["fruit_processed"] += 1
                    m["debug_summary"]["fruit_skipped_no_candidates"] += 1
                    if m["debug_pixels"] is not None:
                        m["debug_pixels"].append({
                            "fruit_id": fruit_id,
                            "fruit_iid": fruit_iid,
                            "candidate_leaves": 0,
                            "pix_counts": [0] * num_dirs,
                            "skipped_reason": "no_candidates",
                        })
                continue

            cand_leaf_ids = [c['id'] for c in candidates]
            cand_leaf_centroids = np.stack([c['center'] for c in candidates], axis=0)
            cand_leaf_dists = np.linalg.norm(cand_leaf_centroids - fruit_center[None, :], axis=1)
            cand_leaf_meshes = [meshes_for_node_id(cid) for cid in cand_leaf_ids]

            # Precompute fruit vertices in local frame for u/v bounds.
            fruit_verts_list = []
            for fm in fruit_meshes:
                fruit_verts_list.append(world_vertices_for_mesh(fm))
            if fruit_verts_list:
                fruit_verts = np.vstack(fruit_verts_list)
            else:
                fruit_verts = np.zeros((0, 3))
            fruit_local = (R @ (fruit_verts - fruit_center).T).T if fruit_verts.size else np.zeros((0, 3))
            if fruit_local.size == 0:
                for m in mode_data.values():
                    m["debug_summary"]["fruit_skipped_no_mesh"] += 1
                    if m["debug_pixels"] is not None:
                        m["debug_pixels"].append({
                            "fruit_id": fruit_id,
                            "fruit_iid": fruit_iid,
                            "candidate_leaves": len(cand_leaf_ids),
                            "pix_counts": [0] * num_dirs,
                            "skipped_reason": "no_mesh_verts",
                        })
                continue

            # Debug BVH for first fruit only.
            if debug_dir_path is not None and f_idx == 0 and debug_fruit_local_bvh is None:
                if fruit_meshes:
                    try:
                        dbg_obj = fruit_meshes[0].blender_obj
                        dbg_mesh = dbg_obj.data
                        if dbg_mesh and dbg_mesh.polygons:
                            dbg_verts = [v.co.copy() for v in dbg_mesh.vertices]
                            dbg_polys = [p.vertices[:] for p in dbg_mesh.polygons]
                            debug_fruit_local_bvh = BVHTree.FromPolygons(dbg_verts, dbg_polys)
                            debug_fruit_local_inv = dbg_obj.matrix_world.inverted()
                            debug_fruit_local_inv3 = debug_fruit_local_inv.to_3x3()
                    except Exception:
                        debug_fruit_local_bvh = None

            M = len(cand_leaf_ids)

            for mode_name, use_persp in modes_to_run:
                m = mode_data[mode_name]
                union = np.zeros((num_dirs,), dtype=np.float32)
                pix_counts = np.zeros((num_dirs,), dtype=np.int32)
                pot = np.zeros((M, num_dirs), dtype=np.float32)
                excl = np.zeros((M, num_dirs), dtype=np.float32)
                krank = np.zeros((M, num_dirs, k), dtype=np.float32)

                if debug_dir_path is not None and f_idx == 0:
                    if m["debug_dir_stats"] is None:
                        m["debug_dir_stats"] = []
                        m["debug_center_hits"] = []

                small_dir_count = 0
                for di, (d_local, u_local, v_local) in enumerate(occlusion_dir_bases_all):
                    # Local u/v bounds from fruit mesh.
                    u_vals = fruit_local @ u_local
                    v_vals = fruit_local @ v_local
                    min_u = float(u_vals.min(initial=0.0))
                    max_u = float(u_vals.max(initial=0.0))
                    min_v = float(v_vals.min(initial=0.0))
                    max_v = float(v_vals.max(initial=0.0))
                    margin = 1e-6
                    u0 = min_u - margin
                    v0 = min_v - margin
                    W = int(np.floor((max_u - u0) / pix)) + 1
                    H = int(np.floor((max_v - v0) / pix)) + 1
                    if W <= 0 or H <= 0:
                        if m["debug_dir_stats"] is not None:
                            m["debug_dir_stats"].append({
                                "dir_index": di,
                                "base_index": int(dir_meta["base_index"][di]),
                                "jitter_level": int(dir_meta["jitter_level"][di]),
                                "jitter_deg": float(dir_meta["jitter_deg"][di]),
                                "d_local": d_local.tolist(),
                                "W": int(W),
                                "H": int(H),
                                "P": 0,
                                "min_u": float(min_u),
                                "max_u": float(max_u),
                                "min_v": float(min_v),
                                "max_v": float(max_v),
                            })
                        continue

                    # Map local basis to world.
                    d_w = (R.T @ d_local).astype(np.float64)
                    d_w = d_w / (np.linalg.norm(d_w) + 1e-8)
                    u_w = (R.T @ u_local).astype(np.float64)
                    v_w = (R.T @ v_local).astype(np.float64)
                    u_vec = Vector(u_w.tolist())
                    v_vec = Vector(v_w.tolist())

                    if use_persp:
                        cam_origin = Vector((fruit_center - d_w * persp_cam_dist).tolist())
                        cam_plane_origin = Vector(fruit_center.tolist())
                        ray_max = persp_cam_dist + max_dist
                    else:
                        cam_origin = Vector((fruit_center - d_w * max_dist).tolist())
                        # Use a conservative ray limit so ortho rays don't miss due to local-space scaling.
                        ray_max = max_dist + 1.0
                        d_vec_const = Vector(d_w.tolist())

                    # Pixel loop.
                    P = 0
                    pot_counts = np.zeros((M,), dtype=np.int32)
                    k_counts = np.zeros((M, k), dtype=np.int32)

                    if m["debug_dir_stats"] is not None and m["debug_center_hits"] is not None:
                        # Center ray sanity check (no u/v offset).
                        if use_persp:
                            center_origin = cam_origin.copy()
                            center_dir = (cam_plane_origin - center_origin).normalized()
                        else:
                            center_origin = cam_origin.copy()
                            center_dir = d_vec_const.copy()
                        fruit_dist_world = ray_cast_min_dist(
                            fruit_meshes, center_origin, center_dir, bvh_cache, depsgraph, max_dist=ray_max
                        )
                        fruit_dist_local = None
                        fruit_dist_local_world = None
                        if debug_fruit_local_bvh is not None and debug_fruit_local_inv is not None:
                            try:
                                origin_local = debug_fruit_local_inv @ center_origin
                                dir_local = (debug_fruit_local_inv3 @ center_dir).normalized()
                                hit = debug_fruit_local_bvh.ray_cast(origin_local, dir_local, ray_max)
                                if hit and hit[0] is not None and hit[3] is not None:
                                    fruit_dist_local = float(hit[3])
                                    hit_world = debug_fruit_local_inv.inverted() @ (origin_local + dir_local * fruit_dist_local)
                                    fruit_dist_local_world = float((hit_world - center_origin).length)
                            except Exception:
                                fruit_dist_local = None
                                fruit_dist_local_world = None
                        m["debug_center_hits"].append({
                            "dir_index": di,
                            "base_index": int(dir_meta["base_index"][di]),
                            "jitter_level": int(dir_meta["jitter_level"][di]),
                            "jitter_deg": float(dir_meta["jitter_deg"][di]),
                            "d_local": d_local.tolist(),
                            "fruit_dist_world": None if fruit_dist_world is None else float(fruit_dist_world),
                            "fruit_dist_local": fruit_dist_local,
                            "fruit_dist_local_world": fruit_dist_local_world,
                        })

                    for iy in range(H):
                        v = v0 + (iy + 0.5) * pix
                        for ix in range(W):
                            u = u0 + (ix + 0.5) * pix
                            if use_persp:
                                plane_point = cam_plane_origin + u_vec * u + v_vec * v
                                origin = cam_origin
                                d_vec = (plane_point - cam_origin).normalized()
                            else:
                                origin = cam_origin + u_vec * u + v_vec * v
                                d_vec = d_vec_const
                            fruit_dist = ray_cast_min_dist(
                                fruit_meshes, origin, d_vec, bvh_cache, depsgraph, max_dist=ray_max
                            )
                            if fruit_dist is None:
                                continue
                            P += 1
                            # occluding leaf candidates
                            occluders = []
                            thresh = fruit_dist - delta
                            for j, meshes in enumerate(cand_leaf_meshes):
                                if not meshes:
                                    continue
                                leaf_dist = ray_cast_min_dist(
                                    meshes, origin, d_vec, bvh_cache, depsgraph, max_dist=ray_max
                                )
                                if leaf_dist is None:
                                    continue
                                if leaf_dist < thresh:
                                    pot_counts[j] += 1
                                    occluders.append((leaf_dist, j))

                            if not occluders:
                                continue
                            occluders.sort(key=lambda x: x[0])
                            topk = occluders[:k]
                            # rank-1
                            k_counts[topk[0][1], 0] += 1
                            # rank-2/3
                            for rk, (_d, jidx) in enumerate(topk[1:], start=1):
                                k_counts[jidx, rk] += 1

                    if P < min_pixels:
                        pix_counts[di] = int(P)
                        small_dir_count += 1
                        continue
                    if max_pixels and P > max_pixels:
                        # downsample effect: scale counts to capped pixel count
                        scale = float(max_pixels) / float(P)
                        P = int(max_pixels)
                        pot_counts = np.round(pot_counts * scale).astype(np.int32)
                        k_counts = np.round(k_counts * scale).astype(np.int32)

                    pix_counts[di] = int(P)
                    if m["debug_dir_stats"] is not None:
                        m["debug_dir_stats"].append({
                            "dir_index": di,
                            "base_index": int(dir_meta["base_index"][di]),
                            "jitter_level": int(dir_meta["jitter_level"][di]),
                            "jitter_deg": float(dir_meta["jitter_deg"][di]),
                            "d_local": d_local.tolist(),
                            "W": int(W),
                            "H": int(H),
                            "P": int(P),
                            "min_u": float(min_u),
                            "max_u": float(max_u),
                            "min_v": float(min_v),
                            "max_v": float(max_v),
                        })
                    if P > 0:
                        pot[:, di] = pot_counts.astype(np.float32) / float(P)
                        excl[:, di] = k_counts[:, 0].astype(np.float32) / float(P)
                        for rk in range(k):
                            krank[:, di, rk] = k_counts[:, rk].astype(np.float32) / float(P)
                        union[di] = float(k_counts[:, 0].sum()) / float(P)

                if small_dir_count == num_dirs:
                    m["debug_summary"]["fruit_skipped_min_pixels"] += 1
                    if m["debug_pixels"] is not None:
                        m["debug_pixels"].append({
                            "fruit_id": fruit_id,
                            "fruit_iid": fruit_iid,
                            "candidate_leaves": len(cand_leaf_ids),
                            "pix_counts": pix_counts.tolist(),
                            "skipped_reason": "min_pixels_all_dirs",
                        })
                    continue

                m["fruit_iids"].append(fruit_iid)
                m["fruit_union"].append(union.astype(np.float16))
                m["fruit_pix"].append(pix_counts.astype(np.int32))

                for j, liid in enumerate(cand_leaf_ids):
                    row_pot = pot[j]
                    row_excl = excl[j]
                    row_k = krank[j]
                    if np.all(row_pot == 0) and np.all(row_k == 0) and np.all(row_excl == 0):
                        continue
                    m["pair_fruit_iid"].append(fruit_iid)
                    m["pair_leaf_iid"].append(int(node_by_id[liid]['instance_id']))
                    m["pair_dist"].append(float(cand_leaf_dists[j]))
                    m["pair_potential"].append(row_pot.astype(np.float16, copy=False))
                    m["pair_exclusive"].append(row_excl.astype(np.float16, copy=False))
                    m["pair_k_rank"].append(row_k.astype(np.float16, copy=False))

                m["debug_summary"]["fruit_processed"] += 1
                if m["debug_pixels"] is not None:
                    m["debug_pixels"].append({
                        "fruit_id": fruit_id,
                        "fruit_iid": fruit_iid,
                        "candidate_leaves": len(cand_leaf_ids),
                        "pix_counts": pix_counts.tolist(),
                        "skipped_reason": None,
                    })

            if progress_every and ((f_idx + 1) % progress_every == 0 or (f_idx + 1) == total_fruits):
                logger.info(
                    f"[Scene {scene_id}] Ray occl progress {f_idx+1}/{total_fruits} fruits"
                )

        def _assemble_mode_arrays(m):
            fruit_iid_arr = np.asarray(m["fruit_iids"], dtype=np.int32)
            if m["fruit_union"]:
                fruit_union_arr = np.stack(m["fruit_union"], axis=0).astype(np.float16, copy=False)
                fruit_pix_arr = np.stack(m["fruit_pix"], axis=0).astype(np.int32, copy=False)
            else:
                fruit_union_arr = np.zeros((0, num_dirs), dtype=np.float16)
                fruit_pix_arr = np.zeros((0, num_dirs), dtype=np.int32)
            pair_fruit_iid_arr = np.asarray(m["pair_fruit_iid"], dtype=np.int32)
            pair_leaf_iid_arr = np.asarray(m["pair_leaf_iid"], dtype=np.int32)
            pair_dist_arr = np.asarray(m["pair_dist"], dtype=np.float32)
            if m["pair_potential"]:
                pair_potential_arr = np.stack(m["pair_potential"], axis=0).astype(np.float16, copy=False)
                pair_exclusive_arr = np.stack(m["pair_exclusive"], axis=0).astype(np.float16, copy=False)
                pair_k_rank_arr = np.stack(m["pair_k_rank"], axis=0).astype(np.float16, copy=False)
            else:
                pair_potential_arr = np.zeros((0, num_dirs), dtype=np.float16)
                pair_exclusive_arr = np.zeros((0, num_dirs), dtype=np.float16)
                pair_k_rank_arr = np.zeros((0, num_dirs, k), dtype=np.float16)
            return {
                "fruit_iid": fruit_iid_arr,
                "fruit_union": fruit_union_arr,
                "fruit_pix": fruit_pix_arr,
                "pair_fruit_iid": pair_fruit_iid_arr,
                "pair_leaf_iid": pair_leaf_iid_arr,
                "pair_dist": pair_dist_arr,
                "pair_potential": pair_potential_arr,
                "pair_exclusive": pair_exclusive_arr,
                "pair_k_rank": pair_k_rank_arr,
            }

        mode_arrays = {}
        for mode_name, _use_persp in modes_to_run:
            mode_arrays[mode_name] = _assemble_mode_arrays(mode_data[mode_name])

        ps_mm = int(round(pix * 1000.0))
        d_mm = int(round(delta * 1000.0))
        base_name = f"scene_{scene_id:05d}_occl_cam_k{k}_ps{ps_mm}mm_d{d_mm}mm"
        if cam_mode == 'ortho':
            out_path = out_dir / f"{base_name}.npz"
        else:
            out_path = out_dir / f"{base_name}_{cam_mode}.npz"

        primary_mode = "ortho" if cam_mode == "both" else cam_mode
        payload = {
            "schema_version": np.asarray(2, dtype=np.int32),
            "scene_id": np.asarray(scene_id),
            "pixel_size_m": np.asarray(pix, dtype=np.float32),
            "delta_m": np.asarray(delta, dtype=np.float32),
            "max_dist_m": np.asarray(max_dist, dtype=np.float32),
            "dirs": np.asarray(dirs_all, dtype=np.float32),
            "dirs_base": np.asarray(dirs_base, dtype=np.float32),
            "dir_base_index": dir_meta["base_index"],
            "dir_jitter_level": dir_meta["jitter_level"],
            "dir_jitter_deg": dir_meta["jitter_deg"],
            "dir_jitter_axis": dir_meta["jitter_axis"],
            "cam_mode": np.asarray(cam_mode),
            "primary_mode": np.asarray(primary_mode),
            "persp_cam_dist_m": np.asarray(persp_cam_dist, dtype=np.float32),
            "jitter_small_deg": np.asarray(jitter_small, dtype=np.float32),
            "jitter_large_deg": np.asarray(jitter_large, dtype=np.float32),
        }

        if cam_mode == 'ortho':
            ortho = mode_arrays.get("ortho")
            payload.update({
                "fruit_iid": ortho["fruit_iid"],
                "fruit_union": ortho["fruit_union"],
                "fruit_pixel_count": ortho["fruit_pix"],
                "pair_fruit_iid": ortho["pair_fruit_iid"],
                "pair_leaf_iid": ortho["pair_leaf_iid"],
                "pair_dist": ortho["pair_dist"],
                "pair_potential": ortho["pair_potential"],
                "pair_exclusive": ortho["pair_exclusive"],
                "pair_k_rank": ortho["pair_k_rank"],
            })
        elif cam_mode == 'persp':
            persp = mode_arrays.get("persp")
            payload.update({
                "persp_fruit_iid": persp["fruit_iid"],
                "persp_fruit_union": persp["fruit_union"],
                "persp_fruit_pixel_count": persp["fruit_pix"],
                "persp_pair_fruit_iid": persp["pair_fruit_iid"],
                "persp_pair_leaf_iid": persp["pair_leaf_iid"],
                "persp_pair_dist": persp["pair_dist"],
                "persp_pair_potential": persp["pair_potential"],
                "persp_pair_exclusive": persp["pair_exclusive"],
                "persp_pair_k_rank": persp["pair_k_rank"],
            })
        else:
            ortho = mode_arrays.get("ortho")
            persp = mode_arrays.get("persp")
            payload.update({
                # Unprefixed keys map to ortho for backward compatibility.
                "fruit_iid": ortho["fruit_iid"],
                "fruit_union": ortho["fruit_union"],
                "fruit_pixel_count": ortho["fruit_pix"],
                "pair_fruit_iid": ortho["pair_fruit_iid"],
                "pair_leaf_iid": ortho["pair_leaf_iid"],
                "pair_dist": ortho["pair_dist"],
                "pair_potential": ortho["pair_potential"],
                "pair_exclusive": ortho["pair_exclusive"],
                "pair_k_rank": ortho["pair_k_rank"],
                # Explicit ortho / persp blocks.
                "ortho_fruit_iid": ortho["fruit_iid"],
                "ortho_fruit_union": ortho["fruit_union"],
                "ortho_fruit_pixel_count": ortho["fruit_pix"],
                "ortho_pair_fruit_iid": ortho["pair_fruit_iid"],
                "ortho_pair_leaf_iid": ortho["pair_leaf_iid"],
                "ortho_pair_dist": ortho["pair_dist"],
                "ortho_pair_potential": ortho["pair_potential"],
                "ortho_pair_exclusive": ortho["pair_exclusive"],
                "ortho_pair_k_rank": ortho["pair_k_rank"],
                "persp_fruit_iid": persp["fruit_iid"],
                "persp_fruit_union": persp["fruit_union"],
                "persp_fruit_pixel_count": persp["fruit_pix"],
                "persp_pair_fruit_iid": persp["pair_fruit_iid"],
                "persp_pair_leaf_iid": persp["pair_leaf_iid"],
                "persp_pair_dist": persp["pair_dist"],
                "persp_pair_potential": persp["pair_potential"],
                "persp_pair_exclusive": persp["pair_exclusive"],
                "persp_pair_k_rank": persp["pair_k_rank"],
            })

        np.savez_compressed(out_path, **payload)

        # Debug summary (file-based)
        if debug_dir_path is not None:
            multi_mode = len(modes_to_run) > 1
            for mode_name, _use_persp in modes_to_run:
                m = mode_data[mode_name]
                arrays = mode_arrays.get(mode_name, {})
                if arrays.get("fruit_union", np.array([])).size:
                    m["debug_summary"]["u_cam_mean"] = float(arrays["fruit_union"].mean())
                    m["debug_summary"]["u_cam_max"] = float(arrays["fruit_union"].max())
                if arrays.get("fruit_pix", np.array([])).size:
                    m["debug_summary"]["pix_min"] = int(arrays["fruit_pix"].min())
                    m["debug_summary"]["pix_max"] = int(arrays["fruit_pix"].max())
                m["debug_summary"]["mode"] = mode_name
                suffix = f"_{mode_name}" if (multi_mode or mode_name != "ortho") else ""
                summary_path = debug_dir_path / f"scene_{scene_id:05d}_ray_summary{suffix}.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(m["debug_summary"], f, indent=2)
                if m["debug_pixels"] is not None:
                    pixels_path = debug_dir_path / f"scene_{scene_id:05d}_ray_pixels{suffix}.json"
                    with open(pixels_path, "w", encoding="utf-8") as f:
                        json.dump(m["debug_pixels"], f, indent=2)
                if m["debug_dir_stats"] is not None:
                    dir_path = debug_dir_path / f"scene_{scene_id:05d}_ray_dir_debug{suffix}.json"
                    with open(dir_path, "w", encoding="utf-8") as f:
                        json.dump(m["debug_dir_stats"], f, indent=2)
                if m["debug_center_hits"] is not None:
                    center_path = debug_dir_path / f"scene_{scene_id:05d}_ray_center_debug{suffix}.json"
                    with open(center_path, "w", encoding="utf-8") as f:
                        json.dump(m["debug_center_hits"], f, indent=2)

        # Optional ray-vs-pc sanity checks
        if debug_dir_path is not None and compare_pc_dir and "ortho" in mode_arrays:
            try:
                pc_dir = Path(compare_pc_dir)
                pc_matches = list(pc_dir.glob(f"scene_{scene_id:05d}_occl_zbuf_*.npz"))
                if pc_matches:
                    pc = np.load(pc_matches[0])
                    ortho = mode_arrays["ortho"]
                    ray = {
                        "fruit_iid": ortho["fruit_iid"],
                        "fruit_union": ortho["fruit_union"],
                        "pair_fruit_iid": ortho["pair_fruit_iid"],
                        "pair_leaf_iid": ortho["pair_leaf_iid"],
                        "pair_k_rank": ortho["pair_k_rank"],
                    }
                    pc_iid = pc["fruit_iid"]
                    pc_union = pc["fruit_union"]
                    # union MAE / correlation
                    common = np.intersect1d(ray["fruit_iid"], pc_iid)
                    if common.size > 0:
                        ray_idx = np.searchsorted(ray["fruit_iid"], common)
                        pc_idx = np.searchsorted(pc_iid, common)
                        r = ray["fruit_union"][ray_idx].astype(np.float32)
                        p = pc_union[pc_idx].astype(np.float32)
                        diff = np.abs(r - p)
                        mae = float(diff.mean())
                        corr = float(np.corrcoef(r.flatten(), p.flatten())[0, 1]) if r.size > 1 else 0.0
                    else:
                        mae, corr = None, None
                    sanity = {
                        "scene_id": scene_id,
                        "common_fruits": int(common.size),
                        "union_mae": mae,
                        "union_corr": corr,
                    }
                    sanity_path = debug_dir_path / f"scene_{scene_id:05d}_ray_vs_pc.json"
                    with open(sanity_path, "w", encoding="utf-8") as f:
                        json.dump(sanity, f, indent=2)
            except Exception as e:
                logger.warning(f"[Scene {scene_id}] Ray-vs-PC sanity check failed: {e}")

        return out_path

    def is_peduncle(mesh):
        return 'peduncle' in mesh.get_name().lower()

    def entry_is_peduncle(entry):
        meshes = entry.get('meshes')
        if not meshes:
            mesh = entry.get('mesh')
            meshes = [mesh] if mesh is not None else []
        return bool(meshes) and all(is_peduncle(m) for m in meshes)

    # Check collisions with previously placed geometry using oriented bounding boxes and BVH trees.
    def collides_with_existing(entity, existing_entries, grid, bvh_cache, minor_ratio, minor_abs, cell_size, group_leaf=False):
        new_entries = [group_obb_entry_for_entity(entity)] if group_leaf else obb_entries_for_entity(entity)
        last_info = None
        overlap_ratio = 0.0
        for entry in new_entries:
            if entry_is_peduncle(entry):
                continue
            allowed = max(minor_abs, minor_ratio * entry['volume'])
            for mesh in entry.get('meshes', []):
                bvh_cache.pop(mesh.get_name(), None)
            candidates = candidates_from_grid(entry, grid, cell_size) if grid else existing_entries
            for existing in candidates:
                if entry_is_peduncle(existing):
                    continue
                collides_obb, overlap = obb_overlap(entry, existing)
                if not collides_obb:
                    continue
                ratio = overlap / entry['volume'] if entry['volume'] > 0 else (1.0 if overlap > 0 else 0.0)
                overlap_ratio = max(overlap_ratio, ratio)
                if overlap <= allowed:
                    continue
                intersects = False
                for mesh in entry.get('meshes', []):
                    for other_mesh in existing.get('meshes', []):
                        intersects, bvh_cache = CollisionUtility.check_mesh_intersection(
                            mesh,
                            other_mesh,
                            skip_inside_check=True,
                            bvh_cache=bvh_cache
                        )
                        if intersects:
                            break
                    if intersects:
                        break
                if intersects:
                    other_name = existing.get('key')
                    if other_name is None:
                        other_mesh = existing.get('mesh')
                        other_name = other_mesh.get_name() if other_mesh is not None else 'unknown'
                    last_info = (overlap, allowed, other_name)
                    return True, new_entries, last_info, overlap_ratio, bvh_cache
        return False, new_entries, last_info, overlap_ratio, bvh_cache

    # Duplicate stems and attachments for a scene instance along a mostly linear layout.
    def instantiate_stem_instances(stem_template, attach_templates, num_stems, config):
        instances = []
        orig_pos = mat_pos(stem_template.get_local2world_mat())
        x_prev = orig_pos[0]
        y_base = orig_pos[1]
        for si in range(num_stems):
            stem_dup = stem_template.duplicate(duplicate_children=True, linked=config['linked_duplicates'])
            stem_dup.blender_obj.hide_viewport = False
            stem_dup.blender_obj.hide_render = False
            attach_dups = [
                attach.duplicate(duplicate_children=True, linked=config['linked_duplicates'])
                for attach in attach_templates
            ]
            coll = make_collection(f'stem_inst_{si}')
            move_entity_to_collection(stem_dup, coll)
            for a_dup in attach_dups:
                move_entity_to_collection(a_dup, coll)

            if si == 0:
                new_pos = orig_pos
            else:
                x_offset = random.uniform(config['stem_offset_x_range'][0], config['stem_offset_x_range'][1])
                y_offset = random.uniform(config['stem_offset_y_range'][0], config['stem_offset_y_range'][1])
                new_pos = (x_prev + x_offset, y_base + y_offset)
                x_prev = new_pos[0]

            cur_mat = stem_dup.get_local2world_mat()
            cur_x, cur_y = mat_pos(cur_mat)
            dx = new_pos[0] - cur_x
            dy = new_pos[1] - cur_y

            new_stem_mat = cur_mat.copy()
            new_stem_mat[0, 3] = cur_mat[0, 3] + dx
            new_stem_mat[1, 3] = cur_mat[1, 3] + dy
            stem_dup.set_local2world_mat(new_stem_mat)
            
            # Apply non-uniform stem scaling
            # Local Y is height in the template stemM
            sw = random.uniform(config['stem_width_scale'][0], config['stem_width_scale'][1])
            sh = random.uniform(config['stem_height_scale'][0], config['stem_height_scale'][1])
            stem_dup.set_scale([sw, sh, sw])

            for a_dup in attach_dups:
                a_mat = a_dup.get_local2world_mat()
                # 1. Translate with stem
                a_new = a_mat.copy()
                a_new[0, 3] = a_mat[0, 3] + dx
                a_new[1, 3] = a_mat[1, 3] + dy
                
                # 2. Scale position relative to stem base center (new_pos)
                # Since stemM local Y is world Z (upright), we scale:
                # (X, Y) relative to center by sw
                # Z relative to base by sh
                rel_x = a_new[0, 3] - new_pos[0]
                rel_y = a_new[1, 3] - new_pos[1]
                rel_z = a_new[2, 3]
                
                a_new[0, 3] = new_pos[0] + rel_x * sw
                a_new[1, 3] = new_pos[1] + rel_y * sw
                a_new[2, 3] = rel_z * sh
                
                a_dup.set_local2world_mat(a_new)

            instances.append((stem_dup, attach_dups, coll))
        return instances

    # Delete a list of BlenderProc entities from the scene.
    def delete_entities(entities):
        if not entities:
            return
        bproc.object.delete_multiple(entities, remove_all_offspring=True)

    # Delete stem instances along with their attachment empties.
    def delete_stem_instances(instances):
        to_delete = []
        for stem_ent, attach_list, _ in instances:
            to_delete.append(stem_ent)
            to_delete.extend(attach_list)
        delete_entities(to_delete)

    # Delete only attachment empties for cleanup.
    def delete_attachment_empties(instances):
        to_delete = []
        for _, attach_list, _ in instances:
            to_delete.extend(attach_list)
        delete_entities(to_delete)

    # Render a simple text progress bar for dataset creation.
    def print_progress(current, total):
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\rScenes: [{bar}] {current}/{total}", end='', flush=True)
        if current == total:
            print()

    # find entities by collection name
    def in_collection(ent, cname):
        return any(c.name == cname for c in ent.blender_obj.users_collection)

    # -----------------------
    # MAIN LOGIC
    # -----------------------    
    # High-level overview:
    # 1) Initialize BlenderProc and load the template .blend file.
    # 2) Collect prototype meshes/empties and prepare helper metadata.
    # 3) For each requested datapoint, duplicate stems, attach blocks with collision-aware placement,
    #    and log the resulting scene graph plus placement metadata.
    # 4) After all datapoints are built, clean up helper empties and persist the final .blend + graphs.

    # spin up BlenderProc runtime and bring the .blend into memory
    print("TRACE: bproc.init()...", flush=True)
    bproc.init()
    print("TRACE: bproc.loader.load_blend...", flush=True)
    objs = bproc.loader.load_blend(args.scene, data_blocks=["collections", "objects"])
    print("TRACE: load_blend done.", flush=True)

    # collect prototype empties for leaves/fruits plus all attachment empties
    leaf_protos = [o for o in objs if in_collection(o, 'leafC') and o.is_empty()]
    pepper_protos = [o for o in objs if in_collection(o, 'pepperC') and o.is_empty()]
    attach_points = [o for o in objs if in_collection(o, 'attachC') and o.is_empty()]

    # find stem - try stemM or any mesh in stemC
    stem_objs = [o for o in objs if in_collection(o, 'stemC')]
    stem = None
    for o in stem_objs:
        if o.get_name() == 'stemM':
            stem = o
            break
    if stem is None and stem_objs:
        # fallback: first mesh in stemC
        for o in stem_objs:
            if o.blender_obj.type == 'MESH':
                stem = o
                break
    if stem is None:
        raise RuntimeError('Could not find stem object in collection `stemC`.')

    print(f"Found {len(attach_points)} attach points, {len(leaf_protos)} leaf prototypes, {len(pepper_protos)} pepper prototypes")

    # make sure scene contains only stems + attachment helpers while prototypes stay hidden
    for coll in bpy.data.collections:
        if coll.name in ['stemC', 'attachC']:
            coll.hide_viewport = False
            coll.hide_render = False
        else:
            coll.hide_viewport = True
            coll.hide_render = True

    stem_template = stem
    attach_templates = attach_points

    stem_template.blender_obj.hide_viewport = True
    stem_template.blender_obj.hide_render = True
    for a in attach_templates:
        a.blender_obj.hide_viewport = True
        a.blender_obj.hide_render = True

    graphs = []
    chunk_accum = []
    chunk_start_idx = args.scene_start
    placements_per_attach = CONFIG['placements_per_attachment']
    final_stem_instances = []
    occlusion_dirs = build_occlusion_dirs() if (CONFIG['occlusion_enabled'] or CONFIG.get('occl_cam_enabled', False)) else []
    occlusion_dir_bases = build_dir_bases(occlusion_dirs) if occlusion_dirs else []

    total_scenes = args.scene_count if args.scene_count is not None else CONFIG['num_datapoints']
    graphs_out_path = resolve_dir(args.graphs_out)

    for dp_local in range(total_scenes):
        dp_idx = args.scene_start + dp_local
        t_scene_start = time.time()
        t_bend = 0.0
        t_collision = 0.0
        t_place = 0.0
        t_occlusion = 0.0
        # reseed per datapoint so graph generation stays reproducible but distinct
        scene_seed = CONFIG['seed'] + dp_idx
        random.seed(scene_seed)
        np.random.seed(scene_seed)

        # duplicate stems/attachments for this datapoint and prepare bookkeeping containers
        logger.info(f"--- Generating Scene {dp_idx} ---")
        logger.debug(f"Scene seed: {scene_seed}")
        num_stems = random.randint(CONFIG['num_stems_range'][0], CONFIG['num_stems_range'][1])
        stem_instances = instantiate_stem_instances(stem_template, attach_templates, num_stems, CONFIG)
        placed_entities = []
        graph_nodes = []
        graph_edges = []
        placements_log = []

        collision_entries = []
        collision_grid = {}
        collision_bvh_cache = {}
        
        instance_counter = 1 # Start from 1 (0 is background)
        
        # collision stats
        stats = {
            'attempts': 0,
            'success': 0,
            'fail': 0,
            'overlap_min': None,
            'overlap_max': 0.0,
            'overlap_avg': 0.0,
            'overlap_count': 0,
        }

        stem_node_ids = {}
        for si, (stem_ent, _attach_list, _coll) in enumerate(stem_instances):
            # add each duplicate stem as a node in the scene graph
            node_id = f"stem_d{dp_idx}_s{si}"
            stem_ent.set_name(node_id)
            graph_nodes.append(build_node_entry(stem_ent, 'stem', node_id, stem_template.get_name(), instance_counter))
            stem_node_ids[si] = node_id
            instance_counter += 1

        def place_variant(si, attach, coll, attach_idx, variant_idx, base_rz_offset_deg):
            nonlocal collision_entries, collision_bvh_cache, t_bend, t_collision, t_place, instance_counter
            t0_place = time.time()
            # sample either fruit or leaf depending on configured probability
            r = random.random()
            if variant_idx > 0 and random.random() < CONFIG['extra_variant_leaf_prob']:
                is_fruit = False
            else:
                is_fruit = r < CONFIG['fruit_prob']
            pool = pepper_protos if is_fruit else leaf_protos
            if len(pool) == 0:
                logger.warning(f"No prototypes found for {'fruit' if is_fruit else 'leaf'} -> skipping attach {attach.get_name()} (variant {variant_idx})")
                return False

            logger.debug(f"    - Attach {attach.get_name()} (v{variant_idx}): Trying {'fruit' if is_fruit else 'leaf'}")
            stats['attempts'] += 1
            proto = random.choice(pool)
            node_type = 'fruit' if is_fruit else 'leaf'

            dup = proto.duplicate(duplicate_children=True, linked=CONFIG['linked_duplicates'])
            children = dup.get_children(return_all_offspring=True)

            attach_mat = attach.get_local2world_mat().copy()
            dz = random.uniform(CONFIG['attach_z_offset'][0], CONFIG['attach_z_offset'][1])
            attach_mat[2, 3] = attach_mat[2, 3] + dz
            dup.set_local2world_mat(attach_mat)

            dup.blender_obj.hide_viewport = False
            dup.blender_obj.hide_render = False
            # ensure the duplicate and all its children live inside the stem instance collection
            for c in list(dup.blender_obj.users_collection):
                c.objects.unlink(dup.blender_obj)
            coll.objects.link(dup.blender_obj)
            for child in children:
                for c in list(child.blender_obj.users_collection):
                    c.objects.unlink(child.blender_obj)
                coll.objects.link(child.blender_obj)

            rx = random.uniform(CONFIG['rot_x_deg'][0], CONFIG['rot_x_deg'][1])
            ry = random.uniform(CONFIG['rot_y_deg'][0], CONFIG['rot_y_deg'][1])

            if CONFIG['bend_enabled']:
                t0 = time.time()
                if node_type == 'leaf':
                    for m in mesh_objects_of(dup):
                        apply_bend_modifier(m, CONFIG['bend_leaf_angle_deg'], CONFIG['bend_axes'])
                else:
                    ped_mesh = find_child_mesh(dup, 'peduncle')
                    if ped_mesh:
                        apply_bend_modifier(ped_mesh, CONFIG['bend_peduncle_angle_deg'], CONFIG['bend_axes'])
                t_bend += time.time() - t0

            placed_ok = False
            last_failure = None
            scale_candidates = CONFIG['collision_scale_candidates']
            overlap_ratio_percent = 0.0
            
            s_range = CONFIG['fruit_scale_range'] if is_fruit else CONFIG['leaf_scale_range']
            base_scale = random.uniform(s_range[0], s_range[1])
            
            for attempt in range(CONFIG['collision_max_attempts']):
                base_rz = random.uniform(CONFIG['rot_z_deg'][0], CONFIG['rot_z_deg'][1])
                rz = (base_rz + base_rz_offset_deg) % 360.0
                scale = base_scale * scale_candidates[min(attempt, len(scale_candidates) - 1)]
                dup.set_scale([scale, scale, scale])
                dup.set_rotation_euler(np.deg2rad([rx, ry, rz]).tolist())

                if CONFIG['collision_enabled']:
                    tc0 = time.time()
                    collides, new_entries, overlap_info, max_ratio, collision_bvh_cache = collides_with_existing(
                        dup,
                        collision_entries,
                        collision_grid,
                        collision_bvh_cache,
                        CONFIG['collision_minor_overlap_ratio'],
                        CONFIG['collision_minor_overlap_abs'],
                        CONFIG['collision_grid_cell_size'],
                        group_leaf=(node_type == 'leaf')
                    )
                    t_collision += time.time() - tc0
                    if not collides:
                        logger.debug(f"      - Placement OK (Attempt {attempt}, Scale {scale:.2f})")
                        placed_ok = True
                        overlap_ratio_percent = max_ratio * 100.0
                        for e in new_entries:
                            if entry_is_peduncle(e):
                                continue
                            collision_entries.append(e)
                            add_entry_to_grid(e, collision_grid, CONFIG['collision_grid_cell_size'])
                        break
                    if overlap_info:
                        last_failure = (*overlap_info, scale, rz)
                else:
                    placed_ok = True
                    overlap_ratio_percent = 0.0
                    break

            if not placed_ok:
                logger.debug(f"    - Attach {attach.get_name()} (v{variant_idx}): FAILED after {CONFIG['collision_max_attempts']} attempts")
                stats['fail'] += 1
                if last_failure:
                    overlap, allowed, other_name, fail_scale, fail_rz = last_failure
                to_remove = [dup] + children
                delete_entities(to_remove)
                t_place += time.time() - t0_place
                return False

            stats['success'] += 1
            stats['overlap_count'] += 1
            count = stats['overlap_count']
            if stats['overlap_min'] is None or overlap_ratio_percent < stats['overlap_min']:
                stats['overlap_min'] = overlap_ratio_percent
            if overlap_ratio_percent > stats['overlap_max']:
                stats['overlap_max'] = overlap_ratio_percent
            if count == 1:
                stats['overlap_avg'] = overlap_ratio_percent
            else:
                stats['overlap_avg'] = (stats['overlap_avg'] * (count - 1) + overlap_ratio_percent) / count

            # register nodes/edges; fruit placements also produce a peduncle node attached to stem
            success_idx = stats['success']
            if node_type == 'fruit':
                ped_mesh = find_child_mesh(dup, 'peduncle')
                fruit_mesh = find_child_mesh(dup, 'pepper') or dup
                ped_id = f"peduncle_d{dp_idx}_s{si}_a{attach_idx}_v{variant_idx}_{success_idx}"
                fruit_id = f"fruit_d{dp_idx}_s{si}_a{attach_idx}_v{variant_idx}_{success_idx}"
                
                # Sync Blender names
                if ped_mesh: ped_mesh.set_name(ped_id)
                if fruit_mesh: fruit_mesh.set_name(fruit_id)

                ped_node = build_node_entry(ped_mesh or dup, 'peduncle', ped_id, proto.get_name(), instance_counter)
                instance_counter += 1
                fruit_node = build_node_entry(fruit_mesh, 'fruit', fruit_id, proto.get_name(), instance_counter)
                instance_counter += 1

                graph_nodes.append(ped_node)
                graph_nodes.append(fruit_node)
                # peduncle to stem, fruit to peduncle
                graph_edges.append({'source': ped_id, 'target': stem_node_ids[si], 'label': 'attached_to'})
                graph_edges.append({'source': fruit_id, 'target': ped_id, 'label': 'attached_to'})
                placements_log.append({
                    'node_id': ped_id,
                    'instance_id': ped_node['instance_id'],
                    'stem_node_id': stem_node_ids[si],
                    'attach_point': attach.get_name(),
                    'variant': variant_idx,
                    'prototype': proto.get_name(),
                    'matrix_world': (ped_mesh or dup).get_local2world_mat().tolist(),
                    'rotation_euler': (ped_mesh or dup).get_rotation_euler().tolist(),
                    'scale': (ped_mesh or dup).get_scale().tolist()
                })
                placements_log.append({
                    'node_id': fruit_id,
                    'instance_id': fruit_node['instance_id'],
                    'stem_node_id': ped_id,
                    'attach_point': attach.get_name(),
                    'variant': variant_idx,
                    'prototype': proto.get_name(),
                    'matrix_world': fruit_mesh.get_local2world_mat().tolist(),
                    'rotation_euler': fruit_mesh.get_rotation_euler().tolist(),
                    'scale': fruit_mesh.get_scale().tolist()
                })
            else:
                leaf_meshes = [m for m in mesh_objects_of(dup) if is_leaf_mesh(m)]
                shoot_meshes = [m for m in mesh_objects_of(dup) if is_shoot_mesh(m)]
                leaf_meshes.sort(key=lambda m: m.get_name())
                shoot_meshes.sort(key=lambda m: m.get_name())

                if leaf_meshes or shoot_meshes:
                    for li, leaf_mesh in enumerate(leaf_meshes):
                        node_id = f"leaf_d{dp_idx}_s{si}_a{attach_idx}_v{variant_idx}_{success_idx}_l{li}"
                        leaf_mesh.set_name(node_id)
                        node_entry = build_node_entry(leaf_mesh, "leaf", node_id, proto.get_name(), instance_counter)
                        graph_nodes.append(node_entry)
                        graph_edges.append({'source': node_id, 'target': stem_node_ids[si], 'label': 'attached_to'})
                        placements_log.append({
                            'node_id': node_id,
                            'instance_id': instance_counter,
                            'stem_node_id': stem_node_ids[si],
                            'attach_point': attach.get_name(),
                            'variant': variant_idx,
                            'prototype': proto.get_name(),
                            'matrix_world': leaf_mesh.get_local2world_mat().tolist(),
                            'rotation_euler': leaf_mesh.get_rotation_euler().tolist(),
                            'scale': leaf_mesh.get_scale().tolist()
                        })
                        instance_counter += 1
                    
                    for shi, shoot_mesh in enumerate(shoot_meshes):
                        node_id = f"shoot_d{dp_idx}_s{si}_a{attach_idx}_v{variant_idx}_{success_idx}_sh{shi}"
                        shoot_mesh.set_name(node_id)
                        node_entry = build_node_entry(shoot_mesh, "shoot", node_id, proto.get_name(), instance_counter)
                        graph_nodes.append(node_entry)
                        graph_edges.append({'source': node_id, 'target': stem_node_ids[si], 'label': 'attached_to'})
                        placements_log.append({
                            'node_id': node_id,
                            'instance_id': instance_counter,
                            'stem_node_id': stem_node_ids[si],
                            'attach_point': attach.get_name(),
                            'variant': variant_idx,
                            'prototype': proto.get_name(),
                            'matrix_world': shoot_mesh.get_local2world_mat().tolist(),
                            'rotation_euler': shoot_mesh.get_rotation_euler().tolist(),
                            'scale': shoot_mesh.get_scale().tolist()
                        })
                        instance_counter += 1
                else:
                    node_id = f"{node_type}_d{dp_idx}_s{si}_a{attach_idx}_v{variant_idx}_{success_idx}"
                    dup.set_name(node_id)
                    node_entry = build_node_entry(dup, node_type, node_id, proto.get_name(), instance_counter)
                    graph_nodes.append(node_entry)
                    graph_edges.append({'source': node_id, 'target': stem_node_ids[si], 'label': 'attached_to'})
                    placements_log.append({
                        'node_id': node_id,
                        'instance_id': instance_counter,
                        'stem_node_id': stem_node_ids[si],
                        'attach_point': attach.get_name(),
                        'variant': variant_idx,
                        'prototype': proto.get_name(),
                        'matrix_world': dup.get_local2world_mat().tolist(),
                        'rotation_euler': dup.get_rotation_euler().tolist(),
                        'scale': dup.get_scale().tolist()
                    })
                    instance_counter += 1
            placed_entities.append(dup)
            t_place += time.time() - t0_place
            return True

        for si, (stem_ent, attach_list, coll) in enumerate(stem_instances):
            logger.info(f"  Instantiating variants for stem {si}...")
            for attach_idx, attach in enumerate(attach_list):
                # evenly space variants around 360 degrees
                for variant_idx in range(placements_per_attach):
                    base_offset = (360.0 / placements_per_attach) * variant_idx
                    jitter = random.uniform(-CONFIG['placement_rot_jitter_deg'], CONFIG['placement_rot_jitter_deg'])
                    place_variant(si, attach, coll, attach_idx, variant_idx, base_offset + jitter)
        
        logger.info(f"  Finished placement loop for scene {dp_idx}")

        logger.info(f"[Scene {dp_idx}] Placed {len(placed_entities)} instances.")
        total_attempts = stats['attempts']
        successes = stats['success']
        failures = stats['fail']
        success_pct = (successes / total_attempts * 100.0) if total_attempts else 0.0
        failure_pct = (failures / total_attempts * 100.0) if total_attempts else 0.0
        overlap_min = stats['overlap_min'] if stats['overlap_min'] is not None else 0.0
        overlap_max = stats['overlap_max'] if stats['overlap_count'] else 0.0
        overlap_avg = stats['overlap_avg'] if stats['overlap_count'] else 0.0
        logger.info(f"[Scene {dp_idx}] Collision stats: {successes}/{total_attempts} succeeded ({success_pct:.1f}%). Overlap avg: {overlap_avg:.2f}%.")

        occlusion_block = None
        occlusion_stats = None
        if CONFIG['occlusion_enabled']:
            t_occ0 = time.time()
            occlusion_block, occlusion_stats = compute_leaf_fruit_occlusion(
                graph_nodes,
                graph_edges,
                occlusion_dirs,
                occlusion_dir_bases,
                CONFIG['occlusion_max_dist'],
                CONFIG['occlusion_weight'],
                CONFIG['occlusion_grid_cell_size'],
            )
            t_occlusion = time.time() - t_occ0
            if CONFIG['occlusion_print_stats'] and (
                (dp_local + 1) % CONFIG['occlusion_print_every'] == 0
                or (dp_local + 1) == total_scenes
            ):
                fruit_count = occlusion_stats['fruit_count']
                fruit_with_any = occlusion_stats['fruit_with_any']
                fruit_pct = (100.0 * fruit_with_any / fruit_count) if fruit_count else 0.0
                logger.info(
                    f"[Scene {dp_idx}] OccStats "
                    f"cand_avg={occlusion_stats['avg_candidates']:.2f} "
                    f"edges={occlusion_stats['edge_count']} "
                    f"fruits_with_occ={fruit_with_any}/{fruit_count} ({fruit_pct:.1f}%) "
                    f"leaves_per_fruit={occlusion_stats['avg_leaves_per_fruit']:.2f} "
                    f"raw(avg/min/max)={occlusion_stats['raw_avg']:.3f}/{occlusion_stats['raw_min']:.3f}/{occlusion_stats['raw_max']:.3f} "
                    f"weighted(avg/min/max)={occlusion_stats['weighted_avg']:.3f}/{occlusion_stats['weighted_min']:.3f}/{occlusion_stats['weighted_max']:.3f}"
                )

        # capture a full graph snapshot plus metadata about stems and placement RNG so scenes can be reproduced
        graph = {
            'id': dp_idx,
            'nodes': graph_nodes,
            'edges': graph_edges,
        }
        if occlusion_block is not None:
            graph['occlusion'] = occlusion_block
        graph['metadata'] = {
            'scene_index': dp_idx,
            'scene_seed': scene_seed,
            'num_stems': num_stems,
            'config': copy.deepcopy(CONFIG),
            'collision_stats': {
                'attempts': total_attempts,
                'successes': successes,
                'failures': failures,
                'success_pct': success_pct,
                'failure_pct': failure_pct,
                'overlap_min_pct': overlap_min,
                'overlap_max_pct': overlap_max,
                'overlap_avg_pct': overlap_avg,
            },
            'stem_transforms': [
                {
                    'stem_id': stem_node_ids[si],
                    'matrix_world': stem_ent.get_local2world_mat().tolist()
                }
                for si, (stem_ent, _attach_list, _coll) in enumerate(stem_instances)
            ],
            'placements': placements_log
        }

        # Optional: camera-based ray occlusion sidecars
        if CONFIG.get('occl_cam_enabled', False):
            if args.occl_cam_out_dir:
                occl_cam_out_dir = Path(args.occl_cam_out_dir)
            elif args.data_dir:
                occl_cam_out_dir = Path(args.data_dir).parent / "occl_cam_sidecars"
            else:
                occl_cam_out_dir = Path(graphs_out_path).parent / "occl_cam_sidecars"
            occl_cam_out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Scene {dp_idx}] Ray occlusion sidecar -> {occl_cam_out_dir}")
            ray_path = compute_ray_occlusion_sidecar(
                scene_id=dp_idx,
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                occlusion_dirs=occlusion_dirs,
                occlusion_dir_bases=occlusion_dir_bases,
                out_dir=occl_cam_out_dir,
                max_dist=CONFIG['occl_cam_max_dist'],
                pix=CONFIG['occl_cam_px'],
                delta=CONFIG['occl_cam_delta'],
                k=CONFIG['occl_cam_k'],
                min_pixels=CONFIG['occl_cam_min_pixels'],
                max_pixels=CONFIG['occl_cam_max_pixels'],
                debug=CONFIG['occl_cam_debug'],
                debug_dir=args.occl_cam_debug_dir,
                compare_pc_dir=args.occl_cam_compare_pc_dir,
                progress_every=CONFIG['occl_cam_progress_every'],
            )
            graph['metadata']['occl_cam_sidecar'] = str(ray_path)

        # Individual Scene Saving
        if args.data_dir:
            logger.info(f"  Saving individual data for scene {dp_idx} to {args.data_dir}...")
            data_dir_path = Path(args.data_dir)
            data_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save graph JSON
            graph_json_name = f"scene_{dp_idx:05d}_graph.json"
            with open(data_dir_path / graph_json_name, 'w', encoding='utf-8') as f_graph:
                json.dump(graph, f_graph, indent=2)
            
            # Save metadata JSON
            if 'metadata' in graph:
                # We extract metadata to a separate file to keep the graph clean
                meta_to_save = {
                    'id': dp_idx,
                    'metadata': graph['metadata']
                }
                meta_json_name = f"scene_{dp_idx:05d}_metadata.json"
                with open(data_dir_path / meta_json_name, 'w', encoding='utf-8') as f_meta:
                    json.dump(meta_to_save, f_meta, indent=2)

        # Integrated Extraction
        if args.extract_pc:
            # Determine PC output path
            if args.data_dir:
                pc_out_path = str(Path(args.data_dir) / f"scene_{dp_idx:05d}_pc.npz")
            else:
                pc_out_path = graphs_out_path.replace('.json', f'_s{dp_idx}.npz')
                
            logger.info(f"[Scene {dp_idx}] Extracting point cloud to {pc_out_path}...")
            pc_timer = time.time()
            pt_count = internal_extract_pc(graph_nodes, pc_out_path, args.voxel_size)
            if pt_count:
                logger.info(f"[Scene {dp_idx}] Extracted {pt_count} points in {time.time() - pc_timer:.2f}s")
                # update the graph object with PC metadata (even if saved individually)
                graph['metadata']['pc_extraction'] = {
                    'voxel_size': args.voxel_size,
                    'point_count': pt_count
                }

        graphs.append(graph)
        chunk_accum.append(graph)
        print_progress(dp_local + 1, total_scenes)

        if dp_local < total_scenes - 1:
            # wipe placed geometry when generating intermediate datapoints to keep memory usage stable
            delete_entities(placed_entities)
            delete_stem_instances(stem_instances)
        else:
            final_stem_instances = stem_instances

        if final_stem_instances:
            # remove helper empties but keep visible geometry for the very last scene (helps manual inspection)
            delete_attachment_empties(final_stem_instances)
            for (_stem, _attach_list, coll) in final_stem_instances:
                for obj in coll.objects:
                    if obj.type == 'EMPTY':
                        obj.hide_viewport = True
                        obj.hide_render = True

        t_scene = time.time() - t_scene_start
        logger.info(
            f"[Scene {dp_idx}] Time total {t_scene:.2f}s (bend {t_bend:.2f}s, collision {t_collision:.2f}s, "
            f"occlusion {t_occlusion:.2f}s, place bookkeeping {t_place:.2f}s)"
        )

    # serialize generated graphs to disk for downstream dataset consumers
    with open(graphs_out_path, 'w', encoding='utf-8') as f:
        json.dump(graphs, f, indent=2)
    logger.info(f"Wrote {len(graphs)} scene graphs to {graphs_out_path}")

    # finally save the Blender scene corresponding to the last datapoint for visual debugging
    if not args.no_blend:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        logger.info(f"Saving final .blend to {out_path}")
        bpy.ops.wm.save_mainfile(filepath=out_path)
    else:
        logger.debug("Skipping .blend save as requested.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL ERROR during execution: {e}", exc_info=True)
        sys.exit(1)
