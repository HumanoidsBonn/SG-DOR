#!/usr/bin/env python3
"""Audit stem-segment identity structure in scene-graph JSON files.

This helps validate whether we can safely treat "any segment of the same stem"
as equivalent for edge-existence supervision/metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total > 0 else 0.0


def summarize_counts(values: list[int]) -> str:
    if not values:
        return "n=0"
    vals = sorted(values)
    n = len(vals)
    p90 = vals[min(n - 1, int(0.9 * (n - 1)))]
    p99 = vals[min(n - 1, int(0.99 * (n - 1)))]
    return (
        f"n={n} mean={mean(vals):.2f} median={median(vals):.2f} "
        f"p90={p90} p99={p99} max={vals[-1]}"
    )


def node_type(node: dict[str, Any]) -> str:
    return str(node.get("type", "<missing>"))


def get_instance_id(node: dict[str, Any]) -> Any:
    # Keep raw type so we can detect weird variants quickly.
    return node.get("instance_id", None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit stem-segment equivalence assumptions.")
    ap.add_argument("--dataset", required=True, help="Path to scene_graphs*.json")
    ap.add_argument("--max-scenes", type=int, default=None, help="Optional scene cap for quick audits")
    ap.add_argument("--show-problems", type=int, default=20, help="How many problematic samples to print")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        graphs = json.load(f)

    if not isinstance(graphs, list):
        raise ValueError("Expected top-level list of scene graphs")

    if args.max_scenes is not None:
        graphs = graphs[: args.max_scenes]

    print(f"Dataset: {path}")
    print(f"Scenes: {len(graphs)}")

    total_nodes = 0
    total_edges = 0
    missing_instance_by_type: Counter[str] = Counter()
    node_count_by_type: Counter[str] = Counter()

    # Stem structure stats
    stem_segments_per_instance_all: list[int] = []
    stem_instances_per_scene: list[int] = []

    # Attachment structure stats
    # For each non-stem node, how many direct attached_to stem-segment neighbors?
    nonstem_to_stem_segment_deg: list[int] = []
    # For each non-stem node, how many unique stem instance_ids among those neighbors?
    nonstem_to_stem_instance_deg: list[int] = []

    # Diagnostics
    multi_stem_instance_nodes: list[tuple[int, str, str, list[Any]]] = []
    missing_instance_edges: list[tuple[int, str, str, str, str]] = []

    for si, g in enumerate(graphs):
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])
        total_nodes += len(nodes)
        total_edges += len(edges)

        idx_by_id: dict[str, dict[str, Any]] = {}
        stem_nodes_by_instance: dict[Any, list[str]] = defaultdict(list)

        for n in nodes:
            nid = str(n.get("id", ""))
            ntype = node_type(n)
            iid = get_instance_id(n)
            idx_by_id[nid] = n
            node_count_by_type[ntype] += 1
            if iid is None:
                missing_instance_by_type[ntype] += 1
            if ntype == "stem":
                stem_nodes_by_instance[iid].append(nid)

        # stem segment stats in this scene
        valid_stem_instances = [k for k in stem_nodes_by_instance.keys() if k is not None]
        stem_instances_per_scene.append(len(valid_stem_instances))
        for iid in valid_stem_instances:
            stem_segments_per_instance_all.append(len(stem_nodes_by_instance[iid]))

        # Build undirected attached_to neighborhood lookup for convenience.
        attached = defaultdict(set)
        for e in edges:
            if e.get("label") != "attached_to":
                continue
            u = str(e.get("source", ""))
            v = str(e.get("target", ""))
            if u not in idx_by_id or v not in idx_by_id:
                continue
            attached[u].add(v)
            attached[v].add(u)

        for nid, n in idx_by_id.items():
            ntype = node_type(n)
            if ntype == "stem":
                continue
            # direct attached_to neighbors that are stem segments
            stem_neigh = [m for m in attached.get(nid, []) if node_type(idx_by_id[m]) == "stem"]
            nonstem_to_stem_segment_deg.append(len(stem_neigh))

            stem_iids = []
            for sid in stem_neigh:
                stem_iid = get_instance_id(idx_by_id[sid])
                if stem_iid is None:
                    missing_instance_edges.append((si, nid, ntype, sid, "stem_missing_instance_id"))
                stem_iids.append(stem_iid)
            unique_iids = sorted(set(stem_iids), key=lambda x: str(x))
            nonstem_to_stem_instance_deg.append(len(unique_iids))

            if len([x for x in unique_iids if x is not None]) > 1:
                multi_stem_instance_nodes.append((si, nid, ntype, unique_iids))

    print("\nNode counts by type:")
    for k, v in sorted(node_count_by_type.items()):
        miss = missing_instance_by_type[k]
        print(f"  {k:10s} total={v:<8d} missing_instance_id={miss} ({pct(miss, v):.2f}%)")

    print("\nStem segmentation structure:")
    print(f"  Stem instances per scene: {summarize_counts(stem_instances_per_scene)}")
    print(f"  Stem segments per stem-instance: {summarize_counts(stem_segments_per_instance_all)}")

    print("\nAttachment structure (direct attached_to):")
    print(f"  Non-stem -> #stem-segment neighbors: {summarize_counts(nonstem_to_stem_segment_deg)}")
    print(f"  Non-stem -> #unique stem-instance neighbors: {summarize_counts(nonstem_to_stem_instance_deg)}")

    zero_seg = sum(1 for x in nonstem_to_stem_segment_deg if x == 0)
    one_seg = sum(1 for x in nonstem_to_stem_segment_deg if x == 1)
    many_seg = sum(1 for x in nonstem_to_stem_segment_deg if x > 1)
    print(
        f"  Segment-neighbor bins: 0={zero_seg} ({pct(zero_seg, len(nonstem_to_stem_segment_deg)):.2f}%), "
        f"1={one_seg} ({pct(one_seg, len(nonstem_to_stem_segment_deg)):.2f}%), "
        f">1={many_seg} ({pct(many_seg, len(nonstem_to_stem_segment_deg)):.2f}%)"
    )

    zero_inst = sum(1 for x in nonstem_to_stem_instance_deg if x == 0)
    one_inst = sum(1 for x in nonstem_to_stem_instance_deg if x == 1)
    many_inst = sum(1 for x in nonstem_to_stem_instance_deg if x > 1)
    print(
        f"  Instance-neighbor bins: 0={zero_inst} ({pct(zero_inst, len(nonstem_to_stem_instance_deg)):.2f}%), "
        f"1={one_inst} ({pct(one_inst, len(nonstem_to_stem_instance_deg)):.2f}%), "
        f">1={many_inst} ({pct(many_inst, len(nonstem_to_stem_instance_deg)):.2f}%)"
    )

    print("\nPotential blockers for stem-segment equivalence:")
    print(f"  Non-stem nodes attached to >1 stem-instance: {len(multi_stem_instance_nodes)}")
    print(f"  attached_to edges involving stem with missing instance_id: {len(missing_instance_edges)}")

    if args.show_problems > 0:
        if multi_stem_instance_nodes:
            print("\nExamples: non-stem nodes attached to multiple stem instances")
            for row in multi_stem_instance_nodes[: args.show_problems]:
                si, nid, ntype, iids = row
                print(f"  scene={si} node={nid} type={ntype} stem_instance_ids={iids}")
        if missing_instance_edges:
            print("\nExamples: edges with missing stem instance_id")
            for row in missing_instance_edges[: args.show_problems]:
                si, nid, ntype, sid, reason = row
                print(f"  scene={si} node={nid} type={ntype} stem_node={sid} reason={reason}")

    print("\nTotals:")
    print(f"  Nodes={total_nodes}  Edges={total_edges}")


if __name__ == "__main__":
    main()
