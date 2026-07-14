#!/usr/bin/env python3
"""
Blender helper: apply an occlusion inference "case" JSON to a .blend for easy visualization.

This is intentionally a plain Blender (`bpy`) script (not BlenderProc).

Usage (background):
  blender --background --python scripts/utils/blender_apply_occl_case.py -- \
    --blend datasets/real_data/300_cleaned_rembg_base_1.55_512_resized.blend \
    --case_json datasets/real_data/inference/<run>/cases/good/case_000_fruitXX_dirYY.json \
    --out_blend datasets/real_data/inference/<run>/viz_case_000.blend

What it does:
  - Opens the input .blend.
  - Colors leaf objects listed in the case by predicted rank (top-K saturated, rest muted).
  - Adds a direction arrow (Empty) pointing along d_world, positioned so the arrow tip stops shortly
    before the fruit centroid (to keep the fruit unobstructed in the viz).
  - Saves a new .blend for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser(description="Apply an occlusion case JSON to a blend for visualization.")
    ap.add_argument("--blend", required=True, help="Input .blend file")
    ap.add_argument("--case_json", required=True, help="Case JSON produced by infer_oracle_real_scene.py")
    ap.add_argument("--out_blend", default=None, help="Output .blend (default: <blend>_viz_case.blend)")
    ap.add_argument("--top_k", type=int, default=5, help="How many top-ranked leaves to highlight strongly")
    ap.add_argument(
        "--rank_source",
        type=str,
        default="rank_logit",
        choices=["rank_logit", "pred_rank", "pot_rank", "potential"],
        help=(
            "Which score to use for ranking leaves. "
            "rank_logit uses pred_rank_logit when present; "
            "pred_rank uses pred_rank from the case file; "
            "pot_rank uses pred_pot_rank (or pred_rank fallback); "
            "potential ranks by pred_potential."
        ),
    )
    ap.add_argument("--mute_others", action="store_true", help="Mute non-topK leaves (set to gray)")
    ap.add_argument("--no_boxes", action="store_true", help="Do not draw wire boxes around top-K leaves (default: draw).")
    ap.add_argument("--box_margin", type=float, default=0.002, help="Extra margin (meters) added to each leaf wire box.")
    ap.add_argument("--arrow_len", type=float, default=0.15, help="Arrow length in meters")
    ap.add_argument(
        "--arrow_tip_gap",
        type=float,
        default=0.03,
        help="Distance (meters) to stop the arrow tip short of the fruit centroid (default: 0.03m).",
    )
    ap.add_argument("--arrow_name", type=str, default="occl_case_dir", help="Name for the arrow object")
    ap.add_argument("--collection", type=str, default="occl_case_viz", help="Collection for viz objects")
    return ap.parse_args(argv)


def _find_obj(name: str) -> Optional["bpy.types.Object"]:
    if not name:
        return None
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    # Fallback: try stripped numeric suffix match (leaf7.001 vs leaf7).
    base = name.split(".", 1)[0]
    if base and base in bpy.data.objects:
        return bpy.data.objects.get(base)
    # Last resort: linear scan for base-name equality.
    for o in bpy.data.objects:
        if o.name.split(".", 1)[0] == base:
            return o
    return None


def _srgb(c: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def _rank_color(rank: int, top_k: int) -> Tuple[float, float, float, float]:
    # Hand-tuned palette for visibility in Blender.
    if rank <= 0:
        return _srgb((0.6, 0.6, 0.6))
    if rank == 1:
        return _srgb((0.90, 0.10, 0.10))  # red
    if rank == 2:
        return _srgb((0.95, 0.55, 0.10))  # orange
    if rank == 3:
        return _srgb((0.95, 0.90, 0.20))  # yellow
    if rank <= int(top_k):
        return _srgb((0.20, 0.80, 0.20))  # green
    return _srgb((0.45, 0.65, 0.45))  # muted green


def _ensure_mat(name: str, rgba: Tuple[float, float, float, float]) -> "bpy.types.Material":
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return mat
    # Ensure Principled exists.
    bsdf = None
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        out = None
        for n in nt.nodes:
            if n.type == "OUTPUT_MATERIAL":
                out = n
                break
        if out is None:
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            out.location = (250, 0)
        nt.links.new(bsdf.outputs.get("BSDF"), out.inputs.get("Surface"))
    # Set base color + some emission for point visibility.
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = rgba
    if "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = rgba
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.5
    return mat


def _set_obj_color(obj: "bpy.types.Object", rgba: Tuple[float, float, float, float], mat_name: str) -> None:
    try:
        obj.color = rgba
    except Exception:
        pass
    # Assign material for Material Preview / Rendered view.
    if getattr(obj, "data", None) is not None and hasattr(obj.data, "materials"):
        mat = _ensure_mat(mat_name, rgba)
        mats = obj.data.materials
        if len(mats) == 0:
            mats.append(mat)
        else:
            mats[0] = mat


def _make_collection(name: str) -> "bpy.types.Collection":
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name=name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _add_wire_box(
    *,
    obj: "bpy.types.Object",
    rgba: Tuple[float, float, float, float],
    name: str,
    collection: "bpy.types.Collection",
    margin: float,
    mat_name: str,
) -> Optional["bpy.types.Object"]:
    if obj is None:
        return None
    if getattr(obj, "bound_box", None) is None:
        return None
    try:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        mn = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
        mx = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
        center = (mn + mx) * 0.5
        half = (mx - mn) * 0.5
        m = float(max(margin, 0.0))
        half = Vector((half.x + m, half.y + m, half.z + m))
        if half.length < 1e-9:
            return None

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
        box = bpy.context.active_object
        box.name = name
        box.scale = half
        try:
            box.display_type = "WIRE"
        except Exception:
            pass
        try:
            box.show_in_front = True
        except Exception:
            pass
        _set_obj_color(box, rgba, mat_name)

        # Move into viz collection.
        for c in list(box.users_collection):
            c.objects.unlink(box)
        collection.objects.link(box)
        return box
    except Exception:
        return None


def _add_direction_arrow(
    *,
    name: str,
    collection: "bpy.types.Collection",
    origin: Vector,
    direction: Vector,
    length: float,
) -> "bpy.types.Object":
    # Use an Empty as a cheap arrow. In Blender 4.x, 'SINGLE_ARROW' exists.
    bpy.ops.object.empty_add(type="SINGLE_ARROW", location=origin)
    arrow = bpy.context.active_object
    arrow.name = name
    arrow.empty_display_size = float(max(length, 1e-3))

    d = direction.copy()
    if d.length < 1e-9:
        d = Vector((0.0, 0.0, 1.0))
    d.normalize()

    # Align empty's local +Z with d (track quaternion).
    try:
        arrow.rotation_mode = "QUATERNION"
        arrow.rotation_quaternion = d.to_track_quat("Z", "Y")
    except Exception:
        # Fallback: do nothing.
        pass

    # Move into the viz collection.
    for c in list(arrow.users_collection):
        c.objects.unlink(arrow)
    collection.objects.link(arrow)
    return arrow


def main() -> None:
    args = _parse_args()
    blend_path = Path(args.blend).expanduser().resolve()
    case_path = Path(args.case_json).expanduser().resolve()
    if not blend_path.exists():
        raise SystemExit(f"[abort] blend does not exist: {blend_path}")
    if not case_path.exists():
        raise SystemExit(f"[abort] case_json does not exist: {case_path}")

    out_blend = Path(args.out_blend).expanduser().resolve() if args.out_blend else None
    if out_blend is None:
        out_blend = blend_path.with_name(blend_path.stem + "_viz_case.blend")

    case = _load_json(case_path)
    leaves = case.get("leaves", []) or []
    fruit = case.get("fruit", {}) or {}
    direction = case.get("direction", {}) or {}

    fruit_name = str(fruit.get("blender_object_name", ""))
    fruit_center = fruit.get("centroid_world", None)
    d_world = direction.get("d_world", None)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    viz_coll = _make_collection(str(args.collection))

    # Build rank lookup.
    rank_by_leaf_name: Dict[str, int] = {}
    rank_source = str(args.rank_source).strip().lower()
    if rank_source in ("pred_rank", "pot_rank"):
        key = "pred_rank" if rank_source == "pred_rank" else "pred_pot_rank"
        for entry in leaves:
            nm = str(entry.get("leaf_name", ""))
            if not nm:
                continue
            v = entry.get(key, None)
            if v is None:
                v = entry.get("pred_rank", 0)
            try:
                rank_by_leaf_name[nm] = int(v)
            except Exception:
                rank_by_leaf_name[nm] = 0
    elif rank_source == "potential":
        scored = []
        for entry in leaves:
            nm = str(entry.get("leaf_name", ""))
            if not nm:
                continue
            try:
                s = float(entry.get("pred_potential", float("-inf")))
            except Exception:
                s = float("-inf")
            scored.append((nm, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        for r, (nm, _s) in enumerate(scored, start=1):
            rank_by_leaf_name[nm] = int(r)
    else:
        # rank_logit: fall back to pred_rank if logits are missing.
        scored = []
        n_finite = 0
        for entry in leaves:
            nm = str(entry.get("leaf_name", ""))
            if not nm:
                continue
            try:
                s = float(entry.get("pred_rank_logit", float("nan")))
            except Exception:
                s = float("nan")
            if math.isfinite(s):
                n_finite += 1
            scored.append((nm, s, entry))
        if n_finite == 0:
            for entry in leaves:
                nm = str(entry.get("leaf_name", ""))
                if not nm:
                    continue
                try:
                    rank_by_leaf_name[nm] = int(entry.get("pred_rank", 0))
                except Exception:
                    rank_by_leaf_name[nm] = 0
        else:
            scored.sort(key=lambda t: (t[1] if math.isfinite(t[1]) else float("-inf")), reverse=True)
            for r, (nm, _s, _e) in enumerate(scored, start=1):
                rank_by_leaf_name[nm] = int(r)

    # Color leaves.
    for entry in leaves:
        leaf_name = str(entry.get("leaf_name", ""))
        rank = int(rank_by_leaf_name.get(leaf_name, 0))
        obj = _find_obj(leaf_name)
        if obj is None:
            continue
        rgba = _rank_color(rank, int(args.top_k))
        mat_name = f"viz_rank_{rank:02d}"
        _set_obj_color(obj, rgba, mat_name)
        try:
            obj.show_in_front = True
        except Exception:
            pass
        if (not bool(args.no_boxes)) and (0 < int(rank) <= int(args.top_k)):
            _add_wire_box(
                obj=obj,
                rgba=rgba,
                name=f"viz_box_{obj.name}_r{rank:02d}",
                collection=viz_coll,
                margin=float(args.box_margin),
                mat_name=f"viz_box_rank_{rank:02d}",
            )

    if bool(args.mute_others):
        # Mute leaves that are not in the top-K by setting them to gray if they are leaf-like objects.
        topk_names = {nm for nm, r in rank_by_leaf_name.items() if int(r) <= int(args.top_k)}
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            nm = obj.name.lower()
            if not nm.startswith("leaf"):
                continue
            if obj.name in rank_by_leaf_name and obj.name in topk_names:
                continue
            rgba = _srgb((0.25, 0.25, 0.25))
            _set_obj_color(obj, rgba, "viz_muted")

    # Highlight fruit object (optional).
    fruit_obj = _find_obj(fruit_name)
    if fruit_obj is not None:
        _set_obj_color(fruit_obj, _srgb((0.95, 0.20, 0.20)), "viz_fruit")
        try:
            fruit_obj.show_in_front = True
        except Exception:
            pass

    # Add direction arrow.
    if fruit_center is not None and d_world is not None:
        try:
            fruit_c = Vector((float(fruit_center[0]), float(fruit_center[1]), float(fruit_center[2])))
            dv = Vector((float(d_world[0]), float(d_world[1]), float(d_world[2])))
            if dv.length < 1e-9:
                dv = Vector((0.0, 0.0, 1.0))
            dv = dv.normalized()

            # Place the arrow so it points along +d_world, but doesn't start at the fruit centroid.
            # Tail is "earlier" along -d_world; tip stops short of the fruit by `arrow_tip_gap`.
            tip_gap = float(max(args.arrow_tip_gap, 0.0))
            length = float(max(args.arrow_len, 1e-3))
            origin = fruit_c - dv * (tip_gap + length)
            _add_direction_arrow(
                name=str(args.arrow_name),
                collection=viz_coll,
                origin=origin,
                direction=dv,
                length=length,
            )
        except Exception as e:
            print(f"[warn] failed to add arrow: {e}", file=sys.stderr)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[ok] wrote {out_blend}")


if __name__ == "__main__":
    main()
