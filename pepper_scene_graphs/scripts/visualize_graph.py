"""Visualize ground truth vs predicted scene graph edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pepper_scene_graphs.data.dataset import load_raw_graphs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot GT vs pred scene graph for a single graph")
    parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
    parser.add_argument("--pred-dump", required=True, help="JSON produced by `pred_dump`")
    parser.add_argument("--graph-index", type=int, default=0, help="Index of scene graph to visualize")
    parser.add_argument("--edge-threshold", type=float, default=0.5, help="Probability threshold for predicted edges")
    parser.add_argument("--save", default=None, help="Path to save figure (PNG); default = show on screen")
    return parser.parse_args()


def build_node_map(nodes: list) -> dict:
    """Map node index -> 3D centroid.
    
    Args:
        nodes: List of node dictionaries.
        
    Returns:
        Dictionary mapping node index to centroid list.
    """
    node_map = {}
    for idx, node in enumerate(nodes):
        centroid = node.get("centroid", [0.0, 0.0, 0.0])
        node_map[idx] = centroid
    return node_map


def collect_gt_edges(graph: dict) -> list[tuple[int, int, str]]:
    """Return GT edges as (u, v, label).
    
    Args:
        graph: Raw graph dictionary.
        
    Returns:
        List of (source, target, label) tuples.
    """
    nodes = graph.get("nodes", [])
    id_to_idx = {node["id"]: idx for idx, node in enumerate(nodes)}
    edges = []
    for edge in graph.get("edges", []):
        u = id_to_idx.get(edge["source"])
        v = id_to_idx.get(edge["target"])
        if u is None or v is None or u == v:
            continue
        label = edge.get("label")
        edges.append((u, v, label))
    return edges


def segmentize(pairs: Iterable[tuple[int, int]], positions: dict[int, Sequence[float]]) -> list[list[tuple[float, float]]]:
    """Convert edge pairs into line segments for plotting.
    
    Args:
        pairs: Iterable of (u, v) indices.
        positions: Dictionary mapping index to position.
        
    Returns:
        List of [(x1, y1), (x2, y2)] segments.
    """
    segments = []
    for u, v in pairs:
        if u not in positions or v not in positions:
            continue
        ux, uy = positions[u][:2]
        vx, vy = positions[v][:2]
        segments.append([(ux, uy), (vx, vy)])
    return segments


def build_pred_lines(edges, positions, threshold):
    lines = []
    colors = []
    linestyles = []
    for edge in edges:
        u = edge["u"]
        v = edge["v"]
        if u not in positions or v not in positions:
            continue
        prob = edge.get("pred_exist_prob", 0.0) or 0.0
        gt = bool(edge.get("gt_exists"))
        segment = segmentize([(u, v)], positions)[0]
        lines.append(segment)
        colors.append("tab:red" if gt else "tab:blue")
        linestyles.append("-" if prob >= threshold else "--")
    return lines, colors, linestyles


def draw_nodes(ax, positions, labels=None):
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    ax.scatter(xs, ys, s=80, c="k", zorder=3)
    if labels:
        for idx, label in labels.items():
            if idx not in positions:
                continue
            ax.text(positions[idx][0], positions[idx][1], label, fontsize=8, ha="right", va="bottom")


def plot_scene(ax, positions, gt_edges, pred_edges, threshold, labels=None):
    draw_nodes(ax, positions)
    gt_segments = segmentize([(u, v) for u, v, _ in gt_edges], positions)
    if gt_segments:
        ax.add_collection(LineCollection(gt_segments, colors="tab:green", linewidths=2, linestyles=":", label="GT"))
    pred_lines, colors, linestyles = build_pred_lines(pred_edges, positions, threshold)
    for segment, color, ls in zip(pred_lines, colors, linestyles):
        ax.add_collection(LineCollection([segment], colors=color, linewidths=1.5, linestyles=ls, alpha=0.7))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    args = parse_args()
    graphs = load_raw_graphs(args.dataset_path)
    if args.graph_index >= len(graphs):
        raise IndexError("Graph index out of range")
    graph = graphs[args.graph_index]
    with open(args.pred_dump, "r", encoding="utf-8") as f:
        pred = json.load(f)

    node_positions = {node["idx"]: node["pos"] for node in pred.get("nodes", [])}
    node_labels = {node["idx"]: node.get("pred_label") for node in pred.get("nodes", [])}
    gt_edges = collect_gt_edges(graph)
    pred_edges = pred.get("edges", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Ground-truth graph")
    axes[1].set_title("Predicted edges")
    plot_scene(axes[0], node_positions, gt_edges, [], args.edge_threshold, labels=node_labels)
    plot_scene(axes[1], node_positions, gt_edges, pred_edges, args.edge_threshold, labels=node_labels)

    fig.suptitle(f"Graph #{args.graph_index}")
    handles = [
        plt.Line2D([0], [0], color="tab:green", ls=":", lw=2, label="GT edge"),
        plt.Line2D([0], [0], color="tab:red", ls="-", lw=2, label="Pred edge (prob>=thr)"),
        plt.Line2D([0], [0], color="tab:blue", ls="--", lw=2, label="Pred edge (prob<thr)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
