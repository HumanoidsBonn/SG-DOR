#!/usr/bin/env python3

import argparse
import colorsys
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pepper_scene_graphs.configs.config import Config
from pepper_scene_graphs.data.dataset import load_raw_graphs


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dataset stats vs noise config")
    parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs.json")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    return parser.parse_args()


def try_load_noise_cfg() -> dict:
    """Attempt to load default noise configuration."""
    cfg = Config()
    return asdict(cfg.noise)


def summarize_array(arr: Sequence[Any]) -> dict | None:
    """Compute statistics for an array of vectors.
    
    Args:
        arr: Input array-like.
        
    Returns:
        Dictionary with count, mean, std, min, max, and percentiles.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[:, None]
    p05, p50, p95 = np.percentile(arr, [5, 50, 95], axis=0)
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "p05": p05.tolist(),
        "p50": p50.tolist(),
        "p95": p95.tolist(),
    }


def summarize_scalar(arr: Sequence[float]) -> dict | None:
    """Compute statistics for a scalar array.
    
    Args:
        arr: Input array-like.
        
    Returns:
        Dictionary with count, mean, std, min, max, and percentiles.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    p05, p50, p95 = np.percentile(arr, [5, 50, 95])
    return {
        "count": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def collect_stats(graphs):
    all_centroids = []
    all_extents = []
    all_orientations = []
    all_axes_cols = []
    all_axes_dot = []
    all_axes_det = []
    all_colors = []
    all_hsv = []
    graph_bbox_diag = []
    by_type = {}

    for graph in graphs:
        nodes = graph.get("nodes", [])
        if not nodes:
            continue
        centroids = []
        for node in nodes:
            ntype = node.get("type", "unknown")
            by_type.setdefault(ntype, {"centroid": [], "extents": [], "orientation": [], "color": []})

            centroid = np.asarray(node.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
            centroids.append(centroid)
            all_centroids.append(centroid)
            by_type[ntype]["centroid"].append(centroid)

            obb = node.get("obb", {})
            extents = np.asarray(obb.get("extents", [0.0, 0.0, 0.0]), dtype=np.float32)
            all_extents.append(extents)
            by_type[ntype]["extents"].append(extents)

            orientation = np.asarray(node.get("orientation", [0.0, 0.0, 1.0]), dtype=np.float32)
            all_orientations.append(orientation)
            by_type[ntype]["orientation"].append(orientation)

            axes = np.asarray(obb.get("axes", np.eye(3)), dtype=np.float32)
            if axes.shape == (3, 3):
                all_axes_cols.append(np.linalg.norm(axes, axis=0))
                all_axes_dot.append(
                    [
                        float(np.dot(axes[:, 0], axes[:, 1])),
                        float(np.dot(axes[:, 0], axes[:, 2])),
                        float(np.dot(axes[:, 1], axes[:, 2])),
                    ]
                )
                all_axes_det.append(float(np.linalg.det(axes)))

            color = node.get("color")
            if color is not None:
                rgb = np.asarray(color, dtype=np.float32)
                if rgb.shape == (3,):
                    rgb = np.clip(rgb, 0.0, 1.0)
                    all_colors.append(rgb)
                    by_type[ntype]["color"].append(rgb)
                    h, s, v = colorsys.rgb_to_hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
                    all_hsv.append([h, s, v])

        if centroids:
            centroids = np.stack(centroids)
            mins = centroids.min(axis=0)
            maxs = centroids.max(axis=0)
            diag = float(np.linalg.norm(maxs - mins))
            graph_bbox_diag.append(diag)

    report = {
        "global": {
            "centroid": summarize_array(all_centroids),
            "extents": summarize_array(all_extents),
            "orientation": summarize_array(all_orientations),
            "orientation_norm": summarize_scalar(np.linalg.norm(all_orientations, axis=1) if all_orientations else []),
            "axes_col_norm": summarize_array(all_axes_cols),
            "axes_dot": summarize_array(all_axes_dot),
            "axes_det": summarize_scalar(all_axes_det),
            "color_rgb": summarize_array(all_colors),
            "color_hsv": summarize_array(all_hsv),
            "graph_bbox_diag": summarize_scalar(graph_bbox_diag),
        },
        "by_type": {},
    }

    for ntype, vals in by_type.items():
        report["by_type"][ntype] = {
            "centroid": summarize_array(vals["centroid"]),
            "extents": summarize_array(vals["extents"]),
            "orientation": summarize_array(vals["orientation"]),
            "color_rgb": summarize_array(vals["color"]),
        }

    return report


def add_noise_ratio(report, noise_cfg):
    if not noise_cfg:
        return
    global_stats = report.get("global", {})
    ratios = {}
    for key, noise_key in [
        ("centroid", "centroid_std"),
        ("orientation", "orientation_std"),
        ("extents", "extent_std"),
    ]:
        stats = global_stats.get(key)
        if stats and "std" in stats:
            std = np.asarray(stats["std"], dtype=np.float32)
            denom = np.maximum(std, 1e-8)
            ratios[key] = (float(noise_cfg.get(noise_key, 0.0)) / denom).tolist()
    if global_stats.get("axes_col_norm") and "std" in global_stats["axes_col_norm"]:
        std = np.asarray(global_stats["axes_col_norm"]["std"], dtype=np.float32)
        denom = np.maximum(std, 1e-8)
        ratios["axes"] = (float(noise_cfg.get("axis_std", 0.0)) / denom).tolist()
    hsv_stats = global_stats.get("color_hsv")
    if hsv_stats and "std" in hsv_stats:
        std = np.asarray(hsv_stats["std"], dtype=np.float32)
        denom = np.maximum(std, 1e-8)
        ratios["color_hsv"] = [
            float(noise_cfg.get("color_hue_std", 0.0)) / float(denom[0]),
            float(noise_cfg.get("color_sat_std", 0.0)) / float(denom[1]),
            float(noise_cfg.get("color_val_std", 0.0)) / float(denom[2]),
        ]
    report["noise_vs_data_std"] = ratios


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    graphs = load_raw_graphs(str(dataset_path))

    report = collect_stats(graphs)
    add_noise_ratio(report, try_load_noise_cfg())

    payload = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
