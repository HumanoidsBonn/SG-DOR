#!/usr/bin/env python3
"""
Analyze how well the occlusion-augmented BFS subgraph covers the *full-scene* occlusion
candidate leaves stored in the z-buffer sidecars.

This answers: for a given sampling configuration (subgraph_size + occl-aug budgets),
how many (fruit, leaf) occluders that exist in the sidecar are actually present in the
sampled subgraph (and thus get node embeddings from the GNN).

Notes:
- Any node included in the sampled subgraph is seen by the GNN. The real "missing" issue
  is leaves within max_dist (0.2m) that exist in the full scene but are *not* present in
  the sampled subgraph. This script measures that directly using the sidecar lists.
- Sidecars are searched in:
  1) <data_root>/data (or <data_root> if no data/ dir)
  2) analysis/occl_zbuf_sidecars_<dataset_name> (workspace-local fallback)
  and can be overridden via --occl_sidecar_dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise SystemExit(
            "numpy is required for this script (needed for .npz sidecars). "
            "Install it in your env (e.g. `pip install numpy`).\n"
            f"Import error: {e}"
        )
    return np


np = _import_numpy()


from pepper_scene_graphs.data.json_subgraph_sampling import JSONSubgraphSampler


@dataclass
class OcclAugCfg:
    enabled: bool
    num_fruits: int
    union_tau: float
    resample_tries: int
    max_dist: float
    max_pos_leaves: int
    max_neg_leaves: int
    max_total_leaves: int


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_candidate_adjacency(
    *,
    pos: np.ndarray,  # [N,3]
    semantic_id: np.ndarray,  # [N] JSON semantic_id (1-based): stem=1 leaf=2 ped=3 fruit=4 ...
    k_candidates: int,
    dist_threshold: float,
    extra_stems_per_node: int,
    radius_max_neighbors: int = 64,
) -> List[List[int]]:
    """
    Approximate the Oracle candidate graph (radius_graph + knn_graph + extra stems).

    This uses JSON centroids as node positions and JSON semantic_id for identifying stems.
    It is not a byte-for-byte reproduction of the GPU path (which uses predicted labels
    and cluster_mean in no-geom-injection mode), but is good enough to estimate whether
    fruit nodes can receive messages from occluder leaves within <=L hops.
    """
    N = int(pos.shape[0])
    if N == 0:
        return []

    # Pairwise distances (N is <=256 typically; O(N^2) is fine).
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1).astype(np.float32, copy=False)  # [N,N]
    np.fill_diagonal(d, np.inf)

    undirected: Dict[Tuple[int, int], None] = {}

    # 1) Radius edges (directed in torch_geometric; we'll treat undirected for message passing).
    r = float(dist_threshold)
    if r > 0:
        for i in range(N):
            js = np.nonzero(d[i] <= r)[0]
            if js.size == 0:
                continue
            # Cap neighbors by distance (torch_geometric uses max_num_neighbors).
            if int(js.size) > int(radius_max_neighbors):
                order = np.argsort(d[i, js])[: int(radius_max_neighbors)]
                js = js[order]
            for j in js.tolist():
                undirected[_pair_key(int(i), int(j))] = None

    # 2) kNN edges (directed; again treat undirected).
    k = int(max(k_candidates, 0))
    if k > 0 and N > 1:
        kk = min(k, N - 1)
        for i in range(N):
            js = np.argsort(d[i])[:kk]
            for j in js.tolist():
                undirected[_pair_key(int(i), int(j))] = None

    # 3) Extra stems (XY-only) from every non-stem to nearest stems.
    extra = int(max(extra_stems_per_node, 0))
    if extra > 0 and N > 1:
        stem_mask = (semantic_id.astype(np.int32, copy=False) == 1)
        if bool(stem_mask.any()) and bool((~stem_mask).any()):
            stem_idx = np.nonzero(stem_mask)[0]
            non_idx = np.nonzero(~stem_mask)[0]
            pos_xy = pos[:, :2].astype(np.float32, copy=False)
            # Distances in XY only.
            dxy = np.linalg.norm(pos_xy[non_idx][:, None, :] - pos_xy[stem_idx][None, :, :], axis=-1)
            kk = min(extra, int(stem_idx.size))
            order = np.argsort(dxy, axis=1)[:, :kk]  # [Nnon, kk]
            for r0, i in enumerate(non_idx.tolist()):
                for jj in order[r0].tolist():
                    j = int(stem_idx[int(jj)])
                    undirected[_pair_key(int(i), int(j))] = None

    # Build adjacency list.
    adj: List[List[int]] = [[] for _ in range(N)]
    for (u, v) in undirected.keys():
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_dist(adj: List[List[int]], src: int, max_hops: int) -> np.ndarray:
    N = len(adj)
    dist = np.full((N,), fill_value=10 ** 9, dtype=np.int32)
    dist[src] = 0
    q = [src]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = int(dist[u])
        if du >= int(max_hops):
            continue
        for v in adj[u]:
            if dist[v] > du + 1:
                dist[v] = du + 1
                q.append(v)
    return dist


def _default_data_dir(data_root: str) -> str:
    d = os.path.join(data_root, "data")
    return d if os.path.exists(d) else data_root


def _split_indices(num_scenes: int, split: str) -> List[int]:
    # Matches OraclePepperDataset: fixed seed + 80/20 split.
    idxs = np.arange(num_scenes, dtype=np.int64)
    rng = np.random.RandomState(42)
    rng.shuffle(idxs)
    split_idx = int(0.8 * num_scenes)
    if split == "train":
        return idxs[:split_idx].tolist()
    if split == "val":
        return idxs[split_idx:].tolist()
    return idxs.tolist()


def _list_scene_ids(data_dir: str, use_4mm: bool) -> List[str]:
    suffix = "_pc_4mm.npz" if use_4mm else "_pc.npz"
    files = sorted(glob.glob(os.path.join(data_dir, f"*{suffix}")))
    out: List[str] = []
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        if len(parts) < 2:
            continue
        scene_id = parts[0] + "_" + parts[1]
        out.append(scene_id)
    return out


def _default_occl_sidecar_dirs(data_root: str, data_dir: str, occl_sidecar_dir: Optional[str]) -> List[Path]:
    if occl_sidecar_dir:
        return [Path(occl_sidecar_dir)]
    dirs: List[Path] = []
    dirs.append(Path(data_dir))
    dirs.append(Path("analysis") / f"occl_zbuf_sidecars_{Path(data_root).name}")
    return dirs


def _find_occl_sidecar_path(scene_id: str, data_root: str, data_dir: str, occl_sidecar_dir: Optional[str]) -> Optional[Path]:
    filename_prefer = f"{scene_id}_occl_zbuf_k3_ps4mm_d4mm.npz"
    glob_fallback = f"{scene_id}_occl_zbuf_k3_*.npz"
    for d in _default_occl_sidecar_dirs(data_root, data_dir, occl_sidecar_dir):
        p = d / filename_prefer
        if p.exists():
            return p
        matches = sorted(d.glob(glob_fallback))
        if matches:
            return matches[0]
    return None


def _load_occl_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with np.load(path) as z:
            return {
                "fruit_iid": z["fruit_iid"].astype(np.int32, copy=False),
                "fruit_union": z["fruit_union"].astype(np.float32, copy=False),
                "pair_fruit_iid": z["pair_fruit_iid"].astype(np.int32, copy=False),
                "pair_leaf_iid": z["pair_leaf_iid"].astype(np.int32, copy=False),
                "pair_potential": z["pair_potential"].astype(np.float32, copy=False),
            }
    except Exception:
        return None


def _augment_subgraph_with_occlusion(
    *,
    sampler: JSONSubgraphSampler,
    scene_id: str,
    json_nodes_list: List[Dict[str, Any]],
    json_edges_raw: List[Tuple[int, int, str]],
    selected_indices_core: List[int],
    target_size: int,
    cfg: OcclAugCfg,
    sidecar: Optional[Dict[str, Any]],
    debug: Optional[Dict[str, Any]] = None,
) -> List[int]:
    # JSON semantic IDs (1-based).
    STEM, LEAF, PED, FRUIT = 1, 2, 3, 4

    fruits_in_core = [i for i in selected_indices_core if int(json_nodes_list[i].get("semantic_id", -1)) == FRUIT]
    if not fruits_in_core:
        return selected_indices_core

    union_max_by_iid: Dict[int, float] = {}
    union_vec_by_iid: Dict[int, np.ndarray] = {}
    if sidecar is not None and int(sidecar["fruit_iid"].size) > 0:
        fruit_iid_arr = sidecar["fruit_iid"]
        union_arr = sidecar["fruit_union"]  # [F,D]
        max_union = union_arr.max(axis=1)
        for iid, mu, uv in zip(fruit_iid_arr.tolist(), max_union.tolist(), union_arr):
            union_max_by_iid[int(iid)] = float(mu)
            union_vec_by_iid[int(iid)] = uv

    num_focus = min(max(cfg.num_fruits, 1), len(fruits_in_core))
    high_union_fruits: List[int] = []
    if union_max_by_iid:
        for idx in fruits_in_core:
            iid = int(json_nodes_list[idx].get("instance_id", -1))
            if union_max_by_iid.get(iid, 0.0) >= cfg.union_tau:
                high_union_fruits.append(idx)

    focus_fruits: List[int] = []
    tries = max(cfg.resample_tries, 1)
    for _ in range(tries):
        focus_fruits = random.sample(fruits_in_core, k=num_focus)
        if not high_union_fruits:
            break
        if any(i in high_union_fruits for i in focus_fruits):
            break
    if high_union_fruits and not any(i in high_union_fruits for i in focus_fruits):
        forced = random.choice(high_union_fruits)
        if forced not in focus_fruits:
            focus_fruits[random.randrange(len(focus_fruits))] = forced

    if debug is not None:
        debug["focus_fruits"] = list(focus_fruits)

    num_nodes = len(json_nodes_list)
    iid_to_json_idx: Dict[int, int] = {}
    for i, n in enumerate(json_nodes_list):
        iid = n.get("instance_id")
        if iid is None:
            continue
        iid_to_json_idx[int(iid)] = i

    # Precompute leaf pool with centroids.
    leaf_indices_all: List[int] = []
    leaf_centroids_all: List[np.ndarray] = []
    for i, n in enumerate(json_nodes_list):
        if int(n.get("semantic_id", -1)) != LEAF:
            continue
        c = n.get("centroid")
        if c is None:
            continue
        leaf_indices_all.append(i)
        leaf_centroids_all.append(np.asarray(c, dtype=np.float32))
    if not leaf_indices_all:
        return selected_indices_core
    leaf_centroids_all_arr = np.stack(leaf_centroids_all, axis=0)  # [L,3]

    adj = sampler.build_adjacency(num_nodes, json_edges_raw)
    fruit_to_ped, _, _, _ = sampler.build_pair_maps(num_nodes, json_nodes_list, json_edges_raw)

    core_required: List[int] = []
    pos_leaves_ordered: List[int] = []
    neg_leaves_ordered: List[int] = []

    for fidx in focus_fruits:
        if fidx not in core_required:
            core_required.append(fidx)

    for fidx in focus_fruits:
        fruit_node = json_nodes_list[fidx]
        fruit_iid = int(fruit_node.get("instance_id", -1))
        fruit_c = np.asarray(fruit_node.get("centroid"), dtype=np.float32)

        ped_idx = fruit_to_ped.get(fidx)
        if ped_idx is not None:
            if int(ped_idx) not in core_required:
                core_required.append(int(ped_idx))
            for nbr in adj[int(ped_idx)]:
                if int(json_nodes_list[nbr].get("semantic_id", -1)) == STEM:
                    if int(nbr) not in core_required:
                        core_required.append(int(nbr))
                    break

        dists = np.linalg.norm(leaf_centroids_all_arr - fruit_c[None, :], axis=1)
        cand_mask = dists <= float(cfg.max_dist)
        cand_pos = np.nonzero(cand_mask)[0]
        if cand_pos.size == 0:
            continue
        cand_leaf_json_idx = [leaf_indices_all[j] for j in cand_pos.tolist()]
        cand_leaf_json_set = set(cand_leaf_json_idx)

        picked_pos: List[int] = []
        if sidecar is not None and fruit_iid >= 0:
            union_vec = union_vec_by_iid.get(fruit_iid, None)
            dom_mask = None
            if union_vec is not None:
                dom_mask = union_vec >= float(cfg.union_tau)

            pair_mask = (sidecar["pair_fruit_iid"] == fruit_iid)
            if bool(pair_mask.any()):
                leaf_iids = sidecar["pair_leaf_iid"][pair_mask].astype(np.int32, copy=False)
                pot = sidecar["pair_potential"][pair_mask].astype(np.float32, copy=False)  # [R,D]
                if dom_mask is not None and bool(dom_mask.any()):
                    score = pot[:, dom_mask].max(axis=1)
                else:
                    score = pot.max(axis=1)
                order = np.argsort(-score)
                for j in order.tolist():
                    if len(picked_pos) >= int(cfg.max_pos_leaves):
                        break
                    liid = int(leaf_iids[j])
                    lj = iid_to_json_idx.get(liid)
                    if lj is None:
                        continue
                    if lj not in cand_leaf_json_set:
                        continue
                    if lj not in picked_pos:
                        picked_pos.append(lj)

        if not picked_pos:
            order = np.argsort(dists[cand_pos])
            for j in order.tolist():
                if len(picked_pos) >= int(cfg.max_pos_leaves):
                    break
                picked_pos.append(cand_leaf_json_idx[j])

        for lj in picked_pos:
            if lj not in pos_leaves_ordered:
                pos_leaves_ordered.append(lj)

        if int(cfg.max_neg_leaves) > 0:
            neg_pool = [lj for lj in cand_leaf_json_idx if lj not in set(picked_pos)]
            if neg_pool:
                k = min(int(cfg.max_neg_leaves), len(neg_pool))
                picked_neg = random.sample(neg_pool, k=k)
                for lj in picked_neg:
                    if lj not in neg_leaves_ordered:
                        neg_leaves_ordered.append(lj)

    if int(cfg.max_total_leaves) > 0:
        if len(pos_leaves_ordered) > int(cfg.max_total_leaves):
            pos_leaves_ordered = pos_leaves_ordered[: int(cfg.max_total_leaves)]
            neg_leaves_ordered = []
        else:
            rem = int(cfg.max_total_leaves) - len(pos_leaves_ordered)
            if rem < len(neg_leaves_ordered):
                neg_leaves_ordered = neg_leaves_ordered[:rem]

    if debug is not None:
        debug["aug_pos_leaf_json_idx"] = list(pos_leaves_ordered)
        debug["aug_neg_leaf_json_idx"] = list(neg_leaves_ordered)

    selected = list(selected_indices_core)
    for idx in core_required + pos_leaves_ordered + neg_leaves_ordered:
        if idx not in selected:
            selected.append(int(idx))

    # Keep deterministic cap (prefer core nodes first).
    if len(selected) > int(target_size):
        # Preserve the original BFS ordering, plus forced chain nodes, then occl leaves.
        # If we still overflow, truncate the tail (occl leaves).
        # (This mirrors the "keep BFS core distribution" intent.)
        selected = selected[: int(target_size)]

    return selected


def _agg_add(d: Dict[str, float], k: str, v: float) -> None:
    d[k] = float(d.get(k, 0.0) + float(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--use_4mm", action="store_true", help="Use the *_pc_4mm.npz list for scene ids (matches training)")
    ap.add_argument("--n_scenes", type=int, default=200, help="Number of scenes to sample for analysis")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--subgraph_size", type=int, default=256)
    ap.add_argument("--subgraph_size_min", type=int, default=256)

    ap.add_argument("--occl_sidecar_dir", type=str, default=None)

    ap.add_argument("--no_occl_aug", action="store_true", help="Disable occl-aug (baseline BFS core only)")
    ap.add_argument("--compare", action="store_true", help="Run both baseline BFS and occl-aug and print both")

    ap.add_argument("--occl_aug_num_fruits", type=int, default=4)
    ap.add_argument("--occl_aug_union_tau", type=float, default=0.5)
    ap.add_argument("--occl_aug_resample_tries", type=int, default=5)
    ap.add_argument("--occl_aug_max_dist", type=float, default=0.2)
    ap.add_argument("--occl_aug_max_pos_leaves", type=int, default=16)
    ap.add_argument("--occl_aug_max_neg_leaves", type=int, default=8)
    ap.add_argument("--occl_aug_max_total_leaves", type=int, default=96)
    ap.add_argument("--occl_aug_reserve_nodes", type=int, default=0, help="Reserve nodes for occl-aug by sampling a smaller BFS core (subgraph_size - reserve)")

    ap.add_argument("--pot_eps", type=float, default=0.05, help="Positive leaf threshold (potential>=eps) for coverage stats")
    ap.add_argument("--actionable_eps", type=float, default=0.15, help="Actionable leaf threshold for coverage stats")

    # Candidate-graph connectivity approximation (to estimate message passing reachability).
    ap.add_argument("--cand_k", type=int, default=8, help="Candidate kNN size (matches model k_candidates)")
    ap.add_argument("--cand_r", type=float, default=0.15, help="Candidate radius threshold (matches model dist_threshold)")
    ap.add_argument("--cand_extra_stems", type=int, default=2, help="Extra stems per node (matches extra_stems_per_node)")
    ap.add_argument("--gnn_hops", type=int, default=3, help="How many message-passing hops to treat as 'seen by fruit embedding'")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_dir = _default_data_dir(args.data_root)
    scene_ids = _list_scene_ids(data_dir, use_4mm=bool(args.use_4mm))
    if not scene_ids:
        raise SystemExit(f"No scenes found in {data_dir}")

    split_idxs = _split_indices(len(scene_ids), args.split)
    if not split_idxs:
        raise SystemExit(f"No indices for split={args.split}")
    # Pick a subset for speed.
    n = min(int(args.n_scenes), len(split_idxs))
    picked = split_idxs[:]
    random.shuffle(picked)
    picked = picked[:n]

    sampler = JSONSubgraphSampler(
        connect_edge_labels=["attached_to"],
        fruit_ped_pair_labels=["attached_to", "fruit_peduncle"],
        shoot_leaf_pair_labels=["attached_to"],
        hard_pair_constraint=True,
        require_shoot_leaf_constraint=False,
    )

    cfg = OcclAugCfg(
        enabled=not bool(args.no_occl_aug),
        num_fruits=int(args.occl_aug_num_fruits),
        union_tau=float(args.occl_aug_union_tau),
        resample_tries=int(args.occl_aug_resample_tries),
        max_dist=float(args.occl_aug_max_dist),
        max_pos_leaves=int(args.occl_aug_max_pos_leaves),
        max_neg_leaves=int(args.occl_aug_max_neg_leaves),
        max_total_leaves=int(args.occl_aug_max_total_leaves),
    )

    def run_once(enable_aug: bool) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        n_used = 0
        n_missing_sidecar = 0

        for scene_idx in picked:
            scene_id = scene_ids[int(scene_idx)]
            json_path = os.path.join(data_dir, f"{scene_id}_graph.json")
            if not os.path.exists(json_path):
                continue

            nodes, edges, _ = sampler.load_json_graph(json_path)

            # Match dataset: variable target size in [min, max] if subgraph_size_min < subgraph_size.
            if int(args.subgraph_size_min) is not None and int(args.subgraph_size_min) < int(args.subgraph_size):
                target_size = int(np.random.randint(int(args.subgraph_size_min), int(args.subgraph_size) + 1))
            else:
                target_size = int(args.subgraph_size)

            core_target = int(target_size)
            if enable_aug and int(args.occl_aug_reserve_nodes) > 0:
                core_target = max(1, int(target_size) - int(args.occl_aug_reserve_nodes))
            core = sampler.bfs_sample(nodes, edges, target_size=core_target, seed_policy="stem_bias")

            sidecar_path = _find_occl_sidecar_path(scene_id, args.data_root, data_dir, args.occl_sidecar_dir)
            sidecar = _load_occl_sidecar(sidecar_path) if sidecar_path is not None else None
            if sidecar is None:
                n_missing_sidecar += 1

            debug: Dict[str, Any] = {}
            if enable_aug and sidecar is not None:
                selected = _augment_subgraph_with_occlusion(
                    sampler=sampler,
                    scene_id=scene_id,
                    json_nodes_list=nodes,
                    json_edges_raw=edges,
                    selected_indices_core=core,
                    target_size=target_size,
                    cfg=cfg,
                    sidecar=sidecar,
                    debug=debug,
                )
            else:
                selected = list(core)

            selected_iids = {int(nodes[i].get("instance_id", -1)) for i in selected}

            # Build candidate graph adjacency on the sampled subgraph nodes.
            # Node order: stable order of `selected` indices.
            sel_nodes = [nodes[int(i)] for i in selected]
            sel_iids = np.asarray([int(n.get("instance_id", -1)) for n in sel_nodes], dtype=np.int64)
            sel_sem = np.asarray([int(n.get("semantic_id", -1)) for n in sel_nodes], dtype=np.int32)
            sel_pos = np.asarray([n.get("centroid", [0.0, 0.0, 0.0]) for n in sel_nodes], dtype=np.float32)
            # Some nodes may not have centroids; guard by leaving them at 0 (rare).
            adj = _build_candidate_adjacency(
                pos=sel_pos,
                semantic_id=sel_sem,
                k_candidates=int(args.cand_k),
                dist_threshold=float(args.cand_r),
                extra_stems_per_node=int(args.cand_extra_stems),
            )
            iid_to_sel = {int(iid): int(j) for j, iid in enumerate(sel_iids.tolist()) if int(iid) > 0}

            # Choose focus fruits for coverage evaluation.
            focus_fruits = debug.get("focus_fruits", [])
            if not focus_fruits:
                focus_fruits = [i for i in selected if int(nodes[i].get("semantic_id", -1)) == 4]
                if focus_fruits:
                    k = min(int(cfg.num_fruits), len(focus_fruits))
                    focus_fruits = random.sample(focus_fruits, k=k)

            if sidecar is None:
                continue

            fruit_iid_arr = sidecar["fruit_iid"]
            union_arr = sidecar["fruit_union"]
            fruit_iid_to_row = {int(iid): int(r) for r, iid in enumerate(fruit_iid_arr.tolist())}

            pair_f_iid = sidecar["pair_fruit_iid"]
            pair_l_iid = sidecar["pair_leaf_iid"]
            pair_pot = sidecar["pair_potential"]

            for fidx in focus_fruits:
                fiid = int(nodes[int(fidx)].get("instance_id", -1))
                row = fruit_iid_to_row.get(fiid)
                if row is None:
                    continue
                union_vec = union_arr[row]  # [D]
                dom_mask = union_vec >= float(cfg.union_tau)

                m = (pair_f_iid == fiid)
                if not bool(m.any()):
                    continue

                leaf_iids = pair_l_iid[m].astype(np.int64, copy=False)
                pot = pair_pot[m].astype(np.float32, copy=False)  # [R,D]
                if bool(dom_mask.any()):
                    score_rows = pot[:, dom_mask].max(axis=1)
                else:
                    score_rows = pot.max(axis=1)

                # Aggregate by leaf iid (in case of duplicate rows).
                score_by_leaf: Dict[int, float] = {}
                for liid, sc in zip(leaf_iids.tolist(), score_rows.tolist()):
                    liid = int(liid)
                    sc = float(sc)
                    prev = score_by_leaf.get(liid)
                    if prev is None or sc > prev:
                        score_by_leaf[liid] = sc

                if not score_by_leaf:
                    continue

                all_leaf = list(score_by_leaf.keys())
                n_total = len(all_leaf)
                n_present = sum(1 for liid in all_leaf if int(liid) in selected_iids)

                pos_leaf = [liid for liid, sc in score_by_leaf.items() if float(sc) >= float(args.pot_eps)]
                n_pos = len(pos_leaf)
                n_pos_present = sum(1 for liid in pos_leaf if int(liid) in selected_iids)

                act_leaf = [liid for liid, sc in score_by_leaf.items() if float(sc) >= float(args.actionable_eps)]
                n_act = len(act_leaf)
                n_act_present = sum(1 for liid in act_leaf if int(liid) in selected_iids)

                # Top-k by score coverage.
                order = sorted(score_by_leaf.items(), key=lambda kv: -float(kv[1]))
                top1 = [order[0][0]] if order else []
                top3 = [liid for liid, _ in order[: min(3, len(order))]]
                top5 = [liid for liid, _ in order[: min(5, len(order))]]
                top1_cov = 1.0 if (top1 and int(top1[0]) in selected_iids) else 0.0
                top3_cov = float(sum(1 for liid in top3 if int(liid) in selected_iids)) / float(len(top3) or 1)
                top5_cov = float(sum(1 for liid in top5 if int(liid) in selected_iids)) / float(len(top5) or 1)

                # Message-passing reachability within <=gnn_hops in the candidate graph (for present leaves only).
                fruit_sel = iid_to_sel.get(int(fiid))
                if fruit_sel is not None:
                    dist = _bfs_dist(adj, int(fruit_sel), max_hops=int(args.gnn_hops))

                    def _reach_stats(liids: List[int]) -> Tuple[float, float]:
                        pres = [liid for liid in liids if int(liid) in selected_iids]
                        if not pres:
                            return 0.0, 0.0
                        sel_idx = [iid_to_sel.get(int(liid), None) for liid in pres]
                        sel_idx = [i for i in sel_idx if i is not None]
                        if not sel_idx:
                            return 0.0, 0.0
                        # Direct neighbor: dist==1
                        direct = sum(1 for i in sel_idx if int(dist[int(i)]) == 1)
                        within = sum(1 for i in sel_idx if int(dist[int(i)]) <= int(args.gnn_hops))
                        return float(direct) / float(len(sel_idx)), float(within) / float(len(sel_idx))

                    # Fractions among PRESENT leaves.
                    d_all, w_all = _reach_stats(all_leaf)
                    d_pos, w_pos = _reach_stats(pos_leaf)
                    d_act, w_act = _reach_stats(act_leaf)
                    d_t3, w_t3 = _reach_stats(top3)
                    d_t5, w_t5 = _reach_stats(top5)

                    _agg_add(agg, "reach_direct_all", d_all)
                    _agg_add(agg, "reach_within_all", w_all)
                    _agg_add(agg, "reach_direct_pos", d_pos)
                    _agg_add(agg, "reach_within_pos", w_pos)
                    _agg_add(agg, "reach_direct_act", d_act)
                    _agg_add(agg, "reach_within_act", w_act)
                    _agg_add(agg, "reach_direct_top3", d_t3)
                    _agg_add(agg, "reach_within_top3", w_t3)
                    _agg_add(agg, "reach_direct_top5", d_t5)
                    _agg_add(agg, "reach_within_top5", w_t5)

                _agg_add(agg, "n_focus_fruits", 1.0)
                _agg_add(agg, "leaves_total", float(n_total))
                _agg_add(agg, "leaves_present", float(n_present))
                _agg_add(agg, "pos_total", float(n_pos))
                _agg_add(agg, "pos_present", float(n_pos_present))
                _agg_add(agg, "act_total", float(n_act))
                _agg_add(agg, "act_present", float(n_act_present))
                _agg_add(agg, "top1_cov", float(top1_cov))
                _agg_add(agg, "top3_cov", float(top3_cov))
                _agg_add(agg, "top5_cov", float(top5_cov))

            # Subgraph change stats.
            _agg_add(agg, "subgraph_core_n", float(len(core)))
            _agg_add(agg, "subgraph_final_n", float(len(selected)))
            n_used += 1

        agg["n_scenes_used"] = float(n_used)
        agg["n_scenes_missing_sidecar"] = float(n_missing_sidecar)
        return agg

    def summarize(name: str, agg: Dict[str, float]) -> None:
        n_focus = max(1.0, agg.get("n_focus_fruits", 0.0))
        leaves_total = agg.get("leaves_total", 0.0)
        leaves_present = agg.get("leaves_present", 0.0)
        pos_total = agg.get("pos_total", 0.0)
        pos_present = agg.get("pos_present", 0.0)
        act_total = agg.get("act_total", 0.0)
        act_present = agg.get("act_present", 0.0)

        print(f"\n[{name}] scenes_used={int(agg.get('n_scenes_used', 0.0))} missing_sidecar={int(agg.get('n_scenes_missing_sidecar', 0.0))}")
        print(f"  focus_fruits={int(agg.get('n_focus_fruits', 0.0))} (avg per scene: {agg.get('n_focus_fruits', 0.0)/max(1.0, agg.get('n_scenes_used', 1.0)):.2f})")
        print(f"  leaf candidates per focus fruit (sidecar): mean={leaves_total/n_focus:.2f}")
        print(f"  leaf coverage in subgraph: {leaves_present/max(1.0, leaves_total):.3f} (missing {1.0 - leaves_present/max(1.0, leaves_total):.3f})")
        print(f"  pos coverage (pot>={args.pot_eps:g}): {pos_present/max(1.0, pos_total):.3f} (pos mean per fruit {pos_total/n_focus:.2f})")
        print(f"  actionable coverage (pot>={args.actionable_eps:g}): {act_present/max(1.0, act_total):.3f} (act mean per fruit {act_total/n_focus:.2f})")
        print(f"  top-k coverage (by score): top1={agg.get('top1_cov', 0.0)/n_focus:.3f} top3={agg.get('top3_cov', 0.0)/n_focus:.3f} top5={agg.get('top5_cov', 0.0)/n_focus:.3f}")
        # Message passing reachability estimates.
        print(f"  candidate-graph reachability among PRESENT leaves (<= {int(args.gnn_hops)} hops):")
        print(f"    all: direct={agg.get('reach_direct_all', 0.0)/n_focus:.3f} within={agg.get('reach_within_all', 0.0)/n_focus:.3f}")
        print(f"    pos(pot>={args.pot_eps:g}): direct={agg.get('reach_direct_pos', 0.0)/n_focus:.3f} within={agg.get('reach_within_pos', 0.0)/n_focus:.3f}")
        print(f"    actionable(pot>={args.actionable_eps:g}): direct={agg.get('reach_direct_act', 0.0)/n_focus:.3f} within={agg.get('reach_within_act', 0.0)/n_focus:.3f}")
        print(f"    top3: direct={agg.get('reach_direct_top3', 0.0)/n_focus:.3f} within={agg.get('reach_within_top3', 0.0)/n_focus:.3f}")
        print(f"    top5: direct={agg.get('reach_direct_top5', 0.0)/n_focus:.3f} within={agg.get('reach_within_top5', 0.0)/n_focus:.3f}")
        core_n = agg.get("subgraph_core_n", 0.0) / max(1.0, agg.get("n_scenes_used", 1.0))
        final_n = agg.get("subgraph_final_n", 0.0) / max(1.0, agg.get("n_scenes_used", 1.0))
        print(f"  subgraph size: core_mean={core_n:.1f} final_mean={final_n:.1f} (+{final_n-core_n:.1f})")

    if bool(args.compare):
        a0 = run_once(enable_aug=False)
        a1 = run_once(enable_aug=True)
        summarize("baseline_bfs", a0)
        summarize("occl_aug", a1)
    else:
        a = run_once(enable_aug=cfg.enabled)
        summarize("occl_aug" if cfg.enabled else "baseline_bfs", a)


if __name__ == "__main__":
    main()
