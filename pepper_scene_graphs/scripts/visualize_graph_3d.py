"""Interactive 3D visualization of GT vs predicted scene graphs using Open3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from pepper_scene_graphs.data.dataset import load_raw_graphs


NODE_COLOR_PALETTE = {
    "stem": (0.2, 0.6, 0.2),
    "leaf": (0.2, 0.4, 0.9),
    "fruit": (0.9, 0.5, 0.1),
    "peduncle": (0.8, 0.1, 0.5),
}
DEFAULT_NODE_COLOR = (0.6, 0.6, 0.6)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="3D Open3D visualization of pepper scene graph")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--pred-dump", required=True)
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--node-scale", type=float, default=0.02, help="Approximate sphere radius for nodes")
    return parser.parse_args()


def build_node_info(pred_dump: dict) -> tuple[dict, dict]:
    """Extract node positions and colors from prediction dump.
    
    Args:
        pred_dump: Dictionary containing 'nodes' list.
        
    Returns:
        Tuple of (positions_dict, colors_dict).
    """
    nodes = pred_dump.get("nodes", [])
    positions = {node["idx"]: np.array(node["pos"], dtype=np.float32) for node in nodes}
    colors = {
        node["idx"]: NODE_COLOR_PALETTE.get(node.get("pred_label") or node.get("gt_label"), DEFAULT_NODE_COLOR)
        for node in nodes
    }
    return positions, colors


def build_line_set(pred_edges: list, positions: dict, threshold: float) -> o3d.geometry.LineSet:
    """Build Open3D LineSet for edges.
    
    Args:
        pred_edges: List of edge dictionaries.
        positions: Dictionary mapping node idx to position.
        threshold: Probability threshold for displaying edges.
        
    Returns:
        Open3D LineSet geometry.
    """
    points = []
    lines = []
    colors = []
    idx_map = {}
    for edge in pred_edges:
        u = edge["u"]
        v = edge["v"]
        if u not in positions or v not in positions:
            continue
        for idx in (u, v):
            if idx not in idx_map:
                idx_map[idx] = len(points)
                points.append(positions[idx])
        lines.append([idx_map[u], idx_map[v]])
        prob = float(edge.get("pred_exist_prob", 0.0))
        gt = bool(edge.get("gt_exists"))
        if gt:
            colors.append((0.0, 1.0, 0.0))
        elif prob >= threshold:
            colors.append((1.0, 0.2, 0.2))
        else:
            colors.append((0.2, 0.5, 1.0))
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if colors:
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def make_node_meshes(positions: dict, colors: dict, radius: float) -> list[o3d.geometry.TriangleMesh]:
    """Create spherical meshes for nodes.
    
    Args:
        positions: Dictionary of node positions.
        colors: Dictionary of node colors.
        radius: Sphere radius.
        
    Returns:
        List of Open3D TriangleMesh objects.
    """
    meshes = []
    for idx, pos in positions.items():
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(pos, relative=False)
        mesh.paint_uniform_color(colors.get(idx, DEFAULT_NODE_COLOR))
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes


def main():
    args = parse_args()
    graphs = load_raw_graphs(args.dataset_path)
    if args.graph_index >= len(graphs):
        raise IndexError("Graph index out of range")
    pred_dump = json.load(open(args.pred_dump, "r", encoding="utf-8"))
    positions, node_colors = build_node_info(pred_dump)
    if not positions:
        raise RuntimeError("No nodes found in pred_dump")

    line_set = build_line_set(pred_dump.get("edges", []), positions, args.edge_threshold)
    node_meshes = make_node_meshes(positions, node_colors, args.node_scale)

    o3d.visualization.draw_geometries([*node_meshes, line_set], window_name=f"Graph #{args.graph_index}")


if __name__ == "__main__":
    main()
