#!/usr/bin/env python3
# IMPORTANT: BlenderProc requires this to be the first import in scripts launched via `blenderproc run`.
import blenderproc as bproc  # type: ignore

# Export a manually-instance-segmented Blender point cloud scene into our synthetic-compatible format.
#
# Input expectation (user-authored .blend):
#   - Collections named like: plant1, plant2, plant3, ...
#   - Each plant collection contains:
#       - exactly one stem object (name starts with "stem", e.g. stem1)
#       - any number of leaves (name starts with "leaf", e.g. leaf7)
#       - optional peduncles (name starts with "peduncle" or "ped", e.g. peduncle4)
#       - optional fruits (name starts with "fruit", e.g. fruit4)
#     Fruit i is attached to peduncle i (same numeric suffix). Peduncle is attached to the stem.
#     Leaves are attached directly to the stem (within the same plant collection).
#
# Outputs (to <out_root>/data/ by default):
#   - scene_00000_pc_4mm.npz:
#       points   float32 [N,3] (world coords)
#       normals  float32 [N,3] (estimated via Open3D by default)
#       colors   uint8   [N,3] (from Blender vertex color attributes when available; else white)
#       semantics uint8  [N]   (1..5: stem/leaf/peduncle/fruit/shoot)
#       instances uint16 [N]   (1..M, unique per exported object)
#   - scene_00000_graph.json:
#       nodes[] compatible with our synthetic scene graphs (centroid + OBB + orientation + color)
#       edges[] with label "attached_to" from your plant/peduncle/fruit conventions
#   - scene_00000_instance_map.json:
#       debugging map from instance_id -> (plant_collection, blender_object_name, type, semantic_id)
#
# Run:
#   blenderproc run scripts/utils/blend_pointcloud_instances_to_dataset.py -- \
#     --blend /abs/path/to/your_scene.blend \
#     --out_root datasets/real_data/real300_export \
#     --scene_id 0

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[abort] numpy is required. Run via BlenderProc (conda env 'bproc') so numpy is available.\n"
            f"Import error: {e}"
        )
    return np


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        o3d = None
    return o3d


def _try_import_bpy():
    try:
        import bpy  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[abort] bpy is required. This script must run inside Blender (e.g. via `blenderproc run`).\n"
            f"Import error: {e}"
        )
    return bpy


SEMANTIC_ID = {
    "stem": 1,
    "leaf": 2,
    "peduncle": 3,
    "fruit": 4,
    "shoot": 5,
}

TYPE_ORDER = {"stem": 0, "peduncle": 1, "fruit": 2, "leaf": 3, "shoot": 4}


def _strip_blender_numeric_suffix(name: str) -> str:
    # "leaf7.001" -> "leaf7"
    return name.split(".", 1)[0]


def _sanitize_id(s: str) -> str:
    # Keep it readable and JSON/filename safe.
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = s.replace(":", "_")
    s = s.replace(".", "_")
    return s


def _parse_type_and_index(obj_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Return (type, idx) from an object name.
    Type in {stem,leaf,peduncle,fruit,shoot} and idx is the trailing integer if present.
    """
    base = _strip_blender_numeric_suffix(obj_name).lower()
    t: Optional[str] = None
    if base.startswith("stem"):
        t = "stem"
    elif base.startswith("leaf"):
        t = "leaf"
    elif base.startswith("fruit"):
        t = "fruit"
    elif base.startswith("peduncle") or base.startswith("ped"):
        t = "peduncle"
    elif base.startswith("shoot"):
        t = "shoot"
    else:
        return None, None
    m = re.search(r"(\d+)$", base)
    idx = int(m.group(1)) if m else None
    return t, idx


def _get_vertex_colors_uint8(np, obj) -> Optional["np.ndarray"]:
    """
    Try to read per-vertex colors from Blender mesh color attributes.

    Returns uint8 [V,3] or None if no colors found.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    color_attrs = getattr(mesh, "color_attributes", None)
    if not color_attrs or len(color_attrs) == 0:
        return None

    # Prefer active attribute, else common names, else first.
    attr = None
    try:
        attr = color_attrs.active
    except Exception:
        attr = None
    if attr is None:
        for nm in ("Col", "color", "Color", "RGB", "rgba"):
            if nm in color_attrs:
                attr = color_attrs[nm]
                break
    if attr is None:
        attr = color_attrs[0]

    # Blender stores colors as RGBA floats in [0,1] in most cases.
    domain = getattr(attr, "domain", None)
    if domain == "POINT":
        if len(attr.data) != len(mesh.vertices):
            return None
        c = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        for i in range(len(mesh.vertices)):
            rgba = attr.data[i].color
            r = float(rgba[0]) if len(rgba) > 0 else 1.0
            g = float(rgba[1]) if len(rgba) > 1 else 1.0
            b = float(rgba[2]) if len(rgba) > 2 else 1.0
            c[i, 0] = int(max(0.0, min(1.0, r)) * 255.0 + 0.5)
            c[i, 1] = int(max(0.0, min(1.0, g)) * 255.0 + 0.5)
            c[i, 2] = int(max(0.0, min(1.0, b)) * 255.0 + 0.5)
        return c

    # CORNER domain needs faces/loops; point clouds typically won't have them.
    return None


def _compute_obb_from_points(np, o3d, pts: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Return (center[3], axes[3,3], extents[3]).
    Uses Open3D if available; otherwise fall back to AABB with identity axes.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((3,), dtype=np.float32), np.eye(3, dtype=np.float32), np.zeros((3,), dtype=np.float32)
    if pts.shape[0] < 3 or o3d is None:
        c = pts.mean(axis=0)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        e = mx - mn
        return c.astype(np.float32), np.eye(3, dtype=np.float32), e.astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center, dtype=np.float32)
    axes = np.asarray(obb.R, dtype=np.float32)
    extents = np.asarray(obb.extent, dtype=np.float32)
    return center, axes, extents


def _estimate_normals(np, o3d, points: "np.ndarray", voxel: float, max_nn: int) -> "np.ndarray":
    if o3d is None:
        # Fallback: up normals.
        n = np.zeros_like(points, dtype=np.float32)
        if n.size:
            n[:, 2] = 1.0
        return n
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    radius = float(max(voxel * 2.0, 1e-6))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(max_nn)))
    # Keep it consistent with synthetic: prefer +Z orientation.
    try:
        pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    except Exception:
        pass
    return np.asarray(pcd.normals, dtype=np.float32)


@dataclass(frozen=True)
class ObjRec:
    plant: str
    obj_name: str
    obj: object
    typ: str
    idx: Optional[int]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export segmented Blender pointcloud .blend to synthetic-compatible NPZ+JSON.")
    ap.add_argument("--blend", type=str, required=True, help="Input .blend file (with plant* collections).")
    ap.add_argument("--out_root", type=str, required=True, help="Output root directory (writes into <out_root>/data).")
    ap.add_argument("--scene_id", type=int, default=0, help="Scene index for output filenames (default: 0).")
    ap.add_argument("--plant_prefix", type=str, default="plant", help="Collection name prefix (default: plant).")
    ap.add_argument("--voxel", type=float, default=0.004, help="Voxel size in meters (for normal estimation radius; default: 0.004).")
    ap.add_argument("--max_nn", type=int, default=30, help="Max neighbors for normal estimation (default: 30).")
    ap.add_argument("--no_estimate_normals", action="store_true", help="Do not estimate normals (fallback normals will be +Z).")
    ap.add_argument("--include_shoots", action="store_true", help="Export shoot* objects too (default: off).")
    args = ap.parse_args()

    np = _try_import_numpy()
    o3d = _try_import_open3d()
    bpy = _try_import_bpy()

    # Ensure BlenderProc environment is initialized (device selection, etc).
    # We open the user .blend afterwards.
    try:
        bproc.init()
    except Exception:
        pass

    blend_path = Path(args.blend)
    if not blend_path.exists():
        raise SystemExit(f"[abort] blend does not exist: {blend_path}")

    out_root = Path(args.out_root)
    out_data = out_root / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    scene_id = int(args.scene_id)
    pc_path = out_data / f"scene_{scene_id:05d}_pc_4mm.npz"
    graph_path = out_data / f"scene_{scene_id:05d}_graph.json"
    map_path = out_data / f"scene_{scene_id:05d}_instance_map.json"

    print(f"[info] opening blend: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    plant_prefix = str(args.plant_prefix).strip().lower()

    # Identify candidate plant collections.
    plant_colls = []
    for coll in bpy.data.collections:
        if not coll.name.lower().startswith(plant_prefix):
            continue
        # Require it contains at least one stem mesh object.
        stems = []
        for obj in coll.all_objects:
            if obj.type != "MESH":
                continue
            typ, _ = _parse_type_and_index(obj.name)
            if typ == "stem":
                stems.append(obj)
        if stems:
            plant_colls.append(coll)

    if not plant_colls:
        raise SystemExit(
            f"[abort] no plant collections found with prefix '{plant_prefix}' containing a stem mesh.\n"
            "Expected collections like plant1/plant2/... with objects named stem*, leaf*, fruit*, peduncle*."
        )

    plant_colls.sort(key=lambda c: c.name)
    print(f"[info] found {len(plant_colls)} plant collections: {[c.name for c in plant_colls]}")

    # Gather all exportable objects with deterministic ordering.
    recs: List[ObjRec] = []
    seen_obj_names: set = set()
    for coll in plant_colls:
        plant = coll.name
        for obj in coll.all_objects:
            if obj.type != "MESH":
                continue
            if obj.name in seen_obj_names:
                # Object linked into multiple plant collections; skip duplicates deterministically.
                continue
            typ, idx = _parse_type_and_index(obj.name)
            if typ is None:
                continue
            if typ == "shoot" and not args.include_shoots:
                continue
            seen_obj_names.add(obj.name)
            recs.append(ObjRec(plant=plant, obj_name=obj.name, obj=obj, typ=typ, idx=idx))

    if not recs:
        raise SystemExit("[abort] no labeled mesh objects found (stem/leaf/peduncle/fruit[/shoot]).")

    def _rec_sort_key(r: ObjRec):
        return (r.plant, TYPE_ORDER.get(r.typ, 99), _strip_blender_numeric_suffix(r.obj_name), r.obj_name)

    recs.sort(key=_rec_sort_key)
    print(f"[info] exporting {len(recs)} objects as instances")

    # Export point cloud arrays.
    points_list: List[List[float]] = []
    colors_list: List[List[int]] = []
    semantics_list: List[int] = []
    instances_list: List[int] = []

    instance_map: Dict[int, Dict[str, object]] = {}
    inst_ranges: Dict[int, Tuple[int, int]] = {}
    node_id_of_inst: Dict[int, str] = {}
    inst_of_objname: Dict[str, int] = {}

    inst_id = 1
    for r in recs:
        sem = SEMANTIC_ID.get(r.typ, 0)
        if sem <= 0:
            continue
        obj = r.obj
        mesh = obj.data
        vcount = len(mesh.vertices)
        if vcount == 0:
            print(f"[warn] skipping empty mesh: {r.obj_name}")
            continue

        # World-space points.
        mw = obj.matrix_world
        start = len(points_list)
        for v in mesh.vertices:
            co = mw @ v.co
            points_list.append([float(co.x), float(co.y), float(co.z)])

        # Vertex colors (if present), else white.
        cols = _get_vertex_colors_uint8(np, obj)
        if cols is None or cols.shape[0] != vcount:
            colors_list.extend([[255, 255, 255] for _ in range(vcount)])
        else:
            colors_list.extend(cols.astype(np.uint8, copy=False).tolist())

        semantics_list.extend([int(sem)] * vcount)
        instances_list.extend([int(inst_id)] * vcount)
        end = len(points_list)

        inst_ranges[inst_id] = (start, end)
        inst_of_objname[r.obj_name] = inst_id

        node_id = _sanitize_id(f"{r.plant}__{r.obj_name}")
        node_id_of_inst[inst_id] = node_id

        instance_map[inst_id] = {
            "plant_collection": r.plant,
            "blender_object_name": r.obj_name,
            "type": r.typ,
            "semantic_id": int(sem),
            "name_index": None if r.idx is None else int(r.idx),
        }
        inst_id += 1

    if not points_list:
        raise SystemExit("[abort] no points exported (all meshes empty?)")

    points = np.asarray(points_list, dtype=np.float32)
    colors = np.asarray(colors_list, dtype=np.uint8)
    semantics = np.asarray(semantics_list, dtype=np.uint8)
    instances = np.asarray(instances_list, dtype=np.uint16)

    if colors.shape[0] != points.shape[0]:
        raise SystemExit(f"[abort] internal error: colors {colors.shape} != points {points.shape}")

    if args.no_estimate_normals:
        normals = np.zeros_like(points, dtype=np.float32)
        normals[:, 2] = 1.0
    else:
        print(f"[info] estimating normals (N={points.shape[0]}, voxel={float(args.voxel)}m, max_nn={int(args.max_nn)})")
        normals = _estimate_normals(np, o3d, points, float(args.voxel), int(args.max_nn))

    # Save NPZ.
    np.savez_compressed(
        pc_path,
        points=points,
        normals=normals.astype(np.float32, copy=False),
        colors=colors,
        semantics=semantics,
        instances=instances,
    )
    print(f"[ok] wrote point cloud -> {pc_path} (N={points.shape[0]}, instances={int(instances.max())})")

    # Build graph nodes with OBBs.
    nodes: List[Dict[str, object]] = []
    for iid, meta in instance_map.items():
        start, end = inst_ranges[iid]
        pts_i = points[start:end]
        cols_i = colors[start:end]
        centroid = pts_i.mean(axis=0)
        center, axes, extents = _compute_obb_from_points(np, o3d, pts_i)
        # Orientation: pick the main axis by extent.
        if extents.size:
            main_idx = int(np.argmax(extents))
            orient = axes[:, main_idx].astype(np.float32)
            on = float(np.linalg.norm(orient))
            if on > 1e-8:
                orient = orient / on
        else:
            orient = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Average RGB in [0,1]
        if cols_i.size:
            col_mean = (cols_i.astype(np.float32).mean(axis=0) / 255.0).tolist()
        else:
            col_mean = [0.8, 0.8, 0.8]

        node_id = node_id_of_inst[iid]
        typ = str(meta["type"])
        sem_id = int(meta["semantic_id"])
        nodes.append(
            {
                "id": node_id,
                "instance_id": int(iid),
                "semantic_id": int(sem_id),
                "type": typ,
                "prototype": str(meta["blender_object_name"]),
                "obb": {
                    "center": center.tolist(),
                    "axes": axes.tolist(),
                    "extents": extents.tolist(),
                },
                "centroid": centroid.astype(np.float32).tolist(),
                "orientation": orient.tolist(),
                "color": col_mean,
            }
        )

    # Build edges from plant structure:
    # - leaf -> stem (same plant)
    # - peduncle -> stem (same plant)
    # - fruit -> peduncle (same plant, same numeric suffix) else fruit -> stem
    edges: List[Dict[str, str]] = []

    # Build per-plant lookup tables.
    by_plant: Dict[str, Dict[str, List[ObjRec]]] = {}
    for r in recs:
        if r.obj_name not in inst_of_objname:
            continue
        by_plant.setdefault(r.plant, {}).setdefault(r.typ, []).append(r)

    for plant, d in by_plant.items():
        stems = d.get("stem", [])
        if not stems:
            print(f"[warn] plant {plant}: no stem found; skipping edges")
            continue
        if len(stems) > 1:
            print(f"[warn] plant {plant}: multiple stems found; using first by name")
            stems = sorted(stems, key=lambda r: r.obj_name)
        stem_rec = stems[0]
        stem_iid = inst_of_objname.get(stem_rec.obj_name)
        if stem_iid is None:
            continue
        stem_node = node_id_of_inst[stem_iid]

        # Leaves.
        for leaf_rec in d.get("leaf", []):
            liid = inst_of_objname.get(leaf_rec.obj_name)
            if liid is None:
                continue
            edges.append({"source": node_id_of_inst[liid], "target": stem_node, "label": "attached_to"})

        # Peduncles.
        ped_by_idx: Dict[int, ObjRec] = {}
        for ped_rec in d.get("peduncle", []):
            piid = inst_of_objname.get(ped_rec.obj_name)
            if piid is None:
                continue
            edges.append({"source": node_id_of_inst[piid], "target": stem_node, "label": "attached_to"})
            if ped_rec.idx is not None:
                ped_by_idx[int(ped_rec.idx)] = ped_rec

        # Fruits.
        for fruit_rec in d.get("fruit", []):
            fiid = inst_of_objname.get(fruit_rec.obj_name)
            if fiid is None:
                continue
            target_node = stem_node
            if fruit_rec.idx is not None:
                ped_match = ped_by_idx.get(int(fruit_rec.idx))
                if ped_match is not None:
                    piid = inst_of_objname.get(ped_match.obj_name)
                    if piid is not None:
                        target_node = node_id_of_inst[piid]
                else:
                    print(
                        f"[warn] plant {plant}: fruit {fruit_rec.obj_name} has idx={fruit_rec.idx} "
                        "but no matching peduncle; attaching fruit to stem"
                    )
            edges.append({"source": node_id_of_inst[fiid], "target": target_node, "label": "attached_to"})

    graph = {
        "id": int(scene_id),
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "source": "real_blend_pointcloud",
            "blend_path": str(blend_path),
            "plant_prefix": str(args.plant_prefix),
            "voxel_m": float(args.voxel),
            "normals_estimated": (not bool(args.no_estimate_normals)),
        },
    }

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(instance_map, f, indent=2)

    print(f"[ok] wrote graph -> {graph_path} (nodes={len(nodes)}, edges={len(edges)})")
    print(f"[ok] wrote instance map -> {map_path}")


if __name__ == "__main__":
    main()
