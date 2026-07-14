import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from pepper_scene_graphs.data.json_subgraph_sampling import JSONSubgraphSampler

from torch_geometric.data import Data

class OracleData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # Any [2, E] index tensor should concatenate along the edge dimension.
        if key in ("gt_edge_index", "occl_pair_index"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'node_id_per_point':
            return self.y_node.size(0)
        if key == 'gt_edge_index':
            return self.y_node.size(0)
        if key == "occl_fruit_idx":
            return self.y_node.size(0)
        if key == "occl_pair_index":
            return self.y_node.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class OraclePepperDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        npoint_per_instance: int = 512,
        # Train-time augmentations:
        # - Point-level augs are enabled implicitly by non-zero rates (only applied for split=='train').
        # - Rotation aug is controlled separately via `rotate_aug` (only applied for split=='train').
        augment: bool = False,
        rotate_aug: bool = False,
        rotate_mode: str = "continuous",
        # Point-level augmentations (only applied for split=='train').
        # Applied only when `augment=True`; defaults match the training CLI defaults.
        color_dropout: float = 0.25,
        xyz_jitter: float = 0.001,
        point_dropout: float = 0.1,
        merge_shoots: bool = True,
        drop_shoots: bool = False,
        drop_json: bool = False,
        subgraph_enabled: bool = False,
        subgraph_size: int = 128,
        subgraph_size_min: Optional[int] = None,
        use_4mm: bool = True,
        hard_pair_constraint: bool = True,
        # Occlusion-augmented subgraph sampling (keeps stem-biased BFS as the core).
        occl_aug_enabled: bool = False,
        occl_sidecar_dir: Optional[str] = None,
        occl_aug_num_fruits: int = 4,
        occl_aug_union_tau: float = 0.5,
        occl_aug_resample_tries: int = 5,
        occl_aug_max_dist: float = 0.2,
        occl_aug_max_pos_leaves: int = 16,
        occl_aug_max_neg_leaves: int = 8,
        occl_aug_max_total_leaves: int = 96,
        # Reserve capacity in the subgraph for occlusion augmentation. When >0, we sample a smaller BFS core
        # (subgraph_size - reserve) and then let occl-aug fill back up to subgraph_size. This avoids "no room"
        # and reduces disruption vs evicting arbitrary BFS nodes.
        occl_aug_reserve_nodes: int = 0,
        # Occlusion targets (pointcloud z-buffer sidecars).
        occl_targets_enabled: bool = False,
        # Occlusion GT source for targets/aug (default: pointcloud z-buffer sidecars).
        # - zbuf: scene_XXXXX_occl_zbuf_*.npz (18 dirs)
        # - cam_ortho/cam_persp: scene_XXXXX_occl_cam_*.npz (54 dirs incl jitter; we map back to 18 base dirs)
        occl_gt_source: str = "zbuf",
        # Camera GT only: which direction set to use from cam sidecars.
        # 0 = base dirs (canonical), 1 = small jitter, 2 = large jitter.
        # We always map to the 18 canonical indices via dir_base_index.
        occl_cam_jitter_level: int = 0,
        # Threshold used to define "dominant directions" on fruit union occlusion (must match training `--occl_union_tau`).
        occl_targets_union_tau: float = 0.5,
        occl_targets_max_leaves_per_fruit: int = 64,
        # How we pick the *leafset* per fruit for occlusion supervision/prediction (occl_pair_index).
        # - geom_stratified: geometry-only stratified sampling (deployable; avoids sidecar/GT-driven priority leaf picking)
        # - sidecar_priority: legacy behavior: include occl-aug selected pos/neg leaves first (uses sidecar-driven leaf picks)
        occl_leafset_policy: str = "geom_stratified",
        # Extra occlusion supervision: optionally supervise additional fruits in the sampled subgraph whose dominant
        # directions remain explainable under the *present* leaves (avoids supervising "impossible" targets).
        # If <=0, we supervise only the focus fruits (backwards compatible behavior).
        occl_supervise_max_fruits_per_scene: int = 0,
        occl_supervise_remain_rate_min: float = 0.9,
        occl_supervise_min_dom_dirs: int = 1,
        # Optional: restrict dataset to a fixed set of scene numbers (ints, e.g. {0,1,2}).
        # Useful for eval-only runs (e.g. first N scenes with camera sidecars).
        scene_id_allowlist: Optional[Set[int]] = None,
        **kwargs
    ):
        self.data_root = data_root
        self.split = split
        self.npoint_per_instance = npoint_per_instance
        self.augment = augment
        self.rotate_aug = bool(rotate_aug)
        self.rotate_mode = str(rotate_mode)
        if self.rotate_mode not in ("none", "continuous", "discrete90"):
            raise ValueError(f"rotate_mode must be 'none', 'continuous' or 'discrete90', got {self.rotate_mode!r}")
        if self.rotate_mode == "none":
            # Ensure the mode alone can disable rotation cleanly.
            self.rotate_aug = False
        self.color_dropout = float(color_dropout)
        self.xyz_jitter = float(xyz_jitter)
        self.point_dropout = float(point_dropout)
        self.merge_shoots = merge_shoots
        self.drop_shoots = drop_shoots
        self.drop_json = drop_json
        self.use_4mm = use_4mm
        self.hard_pair_constraint = hard_pair_constraint

        # Occlusion-augmented sampling settings.
        self.occl_aug_enabled = occl_aug_enabled
        self.occl_sidecar_dir = occl_sidecar_dir
        self.occl_aug_num_fruits = int(occl_aug_num_fruits)
        self.occl_aug_union_tau = float(occl_aug_union_tau)
        self.occl_aug_resample_tries = int(occl_aug_resample_tries)
        self.occl_aug_max_dist = float(occl_aug_max_dist)
        self.occl_aug_max_pos_leaves = int(occl_aug_max_pos_leaves)
        self.occl_aug_max_neg_leaves = int(occl_aug_max_neg_leaves)
        self.occl_aug_max_total_leaves = int(occl_aug_max_total_leaves)
        self.occl_aug_reserve_nodes = int(occl_aug_reserve_nodes)

        # Occlusion targets.
        self.occl_targets_enabled = bool(occl_targets_enabled)
        self.occl_gt_source = str(occl_gt_source).strip().lower()
        if self.occl_gt_source not in ("zbuf", "cam_ortho", "cam_persp"):
            raise ValueError(
                "occl_gt_source must be one of {'zbuf','cam_ortho','cam_persp'}, "
                f"got {self.occl_gt_source!r}"
            )
        try:
            self.occl_cam_jitter_level = int(occl_cam_jitter_level)
        except Exception:
            self.occl_cam_jitter_level = 0
        if self.occl_cam_jitter_level not in (0, 1, 2):
            raise ValueError(
                "occl_cam_jitter_level must be one of {0,1,2} "
                f"(got {occl_cam_jitter_level!r})"
            )
        self.occl_targets_union_tau = float(occl_targets_union_tau)
        self.occl_targets_max_leaves_per_fruit = int(occl_targets_max_leaves_per_fruit)
        self.occl_leafset_policy = str(occl_leafset_policy).strip().lower()
        if self.occl_leafset_policy not in ("geom_stratified", "sidecar_priority"):
            raise ValueError(
                "occl_leafset_policy must be one of {'geom_stratified','sidecar_priority'}, "
                f"got {self.occl_leafset_policy!r}"
            )
        self.occl_supervise_max_fruits_per_scene = int(occl_supervise_max_fruits_per_scene)
        self.occl_supervise_remain_rate_min = float(occl_supervise_remain_rate_min)
        self.occl_supervise_min_dom_dirs = int(occl_supervise_min_dom_dirs)
        
        # Subgraph Init
        self.subgraph_enabled = subgraph_enabled
        self.subgraph_size = subgraph_size
        self.subgraph_size_min = subgraph_size_min
        self.sampler = JSONSubgraphSampler(
            connect_edge_labels=['attached_to'],
            fruit_ped_pair_labels=['attached_to', 'fruit_peduncle'],
            shoot_leaf_pair_labels=['attached_to'],
            hard_pair_constraint=self.hard_pair_constraint,
            require_shoot_leaf_constraint=not self.merge_shoots
        )

        self.data_dir = os.path.join(data_root, 'data')
        if not os.path.exists(self.data_dir):
            self.data_dir = data_root
            
        import glob
        suffix = "_4mm.npz" if use_4mm else "_pc.npz"
        all_pc_files = sorted(glob.glob(os.path.join(self.data_dir, f"*{suffix}")))
        self.pc_files = [os.path.basename(f) for f in all_pc_files]
        
        num_scenes = len(self.pc_files)
        indices = np.arange(num_scenes)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(0.8 * num_scenes)
        if split == 'train':
            self.indices = indices[:split_idx]
        elif split == 'val':
            self.indices = indices[split_idx:]
        else:
            self.indices = indices

        # Optional override: use an explicit allowlist of scene numbers.
        if scene_id_allowlist is not None:
            allow = sorted({int(s) for s in scene_id_allowlist})
            scene_num_to_idx: Dict[int, int] = {}
            for i, fn in enumerate(self.pc_files):
                # filenames look like: scene_00000_pc_4mm.npz
                parts = fn.split("_")
                if len(parts) < 2 or parts[0] != "scene":
                    continue
                try:
                    scene_num = int(parts[1])
                except Exception:
                    continue
                scene_num_to_idx[scene_num] = int(i)
            selected = [scene_num_to_idx[s] for s in allow if s in scene_num_to_idx]
            self.indices = np.asarray(selected, dtype=np.int64)

        print(f"OracleDataset: {len(self.indices)} scenes for {split}")

    def __len__(self):
        return len(self.indices)

    def _default_occl_sidecar_dirs(self) -> List[Path]:
        if self.occl_sidecar_dir:
            return [Path(self.occl_sidecar_dir)]
        dirs: List[Path] = []
        dirs.append(Path(self.data_dir))
        # Default postprocess fallback output (workspace-local).
        dirs.append(Path("analysis") / f"occl_zbuf_sidecars_{Path(self.data_root).name}")
        return dirs

    def _find_occl_sidecar_path(self, scene_id: str) -> Optional[Path]:
        """
        Locate a z-buffer occlusion sidecar for this scene.

        We prefer the grounded defaults (ps4mm/d4mm), but fall back to any matching file.
        """
        if self.occl_gt_source in ("cam_ortho", "cam_persp"):
            # Prefer the common "both" sidecar (contains ortho + persp; ortho keys are unprefixed).
            prefer = [
                f"{scene_id}_occl_cam_k3_ps4mm_d4mm_both.npz",
                f"{scene_id}_occl_cam_k3_ps4mm_d4mm_persp.npz",
                f"{scene_id}_occl_cam_k3_ps4mm_d4mm.npz",
            ]
            glob_fallbacks = [
                f"{scene_id}_occl_cam_k3_*.npz",
                f"{scene_id}_occl_cam_*.npz",
            ]
        else:
            prefer = [f"{scene_id}_occl_zbuf_k3_ps4mm_d4mm.npz"]
            glob_fallbacks = [f"{scene_id}_occl_zbuf_k3_*.npz"]
        for d in self._default_occl_sidecar_dirs():
            for fn in prefer:
                p = (d / fn)
                if p.exists():
                    return p
            for pat in glob_fallbacks:
                matches = sorted(d.glob(pat))
                if matches:
                    return matches[0]
        return None

    def _load_occl_sidecar(
        self,
        scene_id: str,
        *,
        cam_jitter_level_override: Optional[int] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        path = self._find_occl_sidecar_path(scene_id)
        if path is None:
            return None
        try:
            with np.load(path) as z:
                if self.occl_gt_source in ("cam_ortho", "cam_persp"):
                    # Camera ray sidecars store 54 directions (18 base + jitter variants).
                    # For oracle training/eval we map back to the 18 base dirs (jitter_level==0),
                    # ordered by base index.
                    prefix = "" if self.occl_gt_source == "cam_ortho" else "persp_"
                    fruit_iid_key = f"{prefix}fruit_iid"
                    fruit_union_key = f"{prefix}fruit_union"
                    pair_fruit_key = f"{prefix}pair_fruit_iid"
                    pair_leaf_key = f"{prefix}pair_leaf_iid"
                    pair_pot_key = f"{prefix}pair_potential"
                    pair_exc_key = f"{prefix}pair_exclusive"
                    pair_k_key = f"{prefix}pair_k_rank"

                    fruit_iid = z[fruit_iid_key].astype(np.int32, copy=False)
                    fruit_union = z[fruit_union_key].astype(np.float32, copy=False)
                    pair_fruit_iid = z[pair_fruit_key].astype(np.int32, copy=False)
                    pair_leaf_iid = z[pair_leaf_key].astype(np.int32, copy=False)
                    pair_potential = z[pair_pot_key].astype(np.float32, copy=False)
                    pair_exclusive = z[pair_exc_key].astype(np.float32, copy=False) if pair_exc_key in z else None
                    pair_k_rank = z[pair_k_key].astype(np.float32, copy=False)

                    base_cols = None
                    if ("dir_jitter_level" in z) and ("dir_base_index" in z):
                        lvl = z["dir_jitter_level"].astype(np.int8, copy=False).reshape(-1)
                        base = z["dir_base_index"].astype(np.int32, copy=False).reshape(-1)
                        want_lvl = int(self.occl_cam_jitter_level) if cam_jitter_level_override is None else int(cam_jitter_level_override)
                        cols = np.nonzero(lvl == want_lvl)[0]
                        if cols.size != 18 and want_lvl != 0:
                            # Fallback to base dirs when jitter level isn't present.
                            cols = np.nonzero(lvl == 0)[0]
                        if cols.size == 18:
                            order = np.argsort(base[cols], kind="stable")
                            base_cols = cols[order]
                    if base_cols is None:
                        # Fallback: assume the first 18 columns correspond to the canonical base dirs.
                        base_cols = np.arange(min(18, int(fruit_union.shape[1])), dtype=np.int64)

                    fruit_union = fruit_union[:, base_cols]
                    pair_potential = pair_potential[:, base_cols]
                    if pair_exclusive is not None:
                        pair_exclusive = pair_exclusive[:, base_cols]
                    pair_k_rank = pair_k_rank[:, base_cols, :]

                    dirs = None
                    if "dirs_base" in z:
                        dirs = z["dirs_base"].astype(np.float32, copy=False)
                    elif "dirs" in z and z["dirs"].ndim == 2 and int(z["dirs"].shape[0]) >= 18:
                        # Best-effort fallback.
                        dirs = z["dirs"][:18].astype(np.float32, copy=False)

                    return {
                        "fruit_iid": fruit_iid,
                        "fruit_union": fruit_union,
                        "pair_fruit_iid": pair_fruit_iid,
                        "pair_leaf_iid": pair_leaf_iid,
                        "pair_potential": pair_potential,
                        "pair_exclusive": pair_exclusive,
                        "pair_k_rank": pair_k_rank,
                        "dirs": dirs,
                    }

                # Default: pointcloud z-buffer sidecars.
                return {
                    "fruit_iid": z["fruit_iid"].astype(np.int32, copy=False),
                    "fruit_union": z["fruit_union"].astype(np.float32, copy=False),
                    "pair_fruit_iid": z["pair_fruit_iid"].astype(np.int32, copy=False),
                    "pair_leaf_iid": z["pair_leaf_iid"].astype(np.int32, copy=False),
                    "pair_potential": z["pair_potential"].astype(np.float32, copy=False),
                    "pair_exclusive": z["pair_exclusive"].astype(np.float32, copy=False) if "pair_exclusive" in z else None,
                    "pair_k_rank": z["pair_k_rank"].astype(np.float32, copy=False),
                    "dirs": z["dirs"].astype(np.float32, copy=False) if "dirs" in z else None,
                }
        except Exception:
            return None

    def _augment_subgraph_with_occlusion(
        self,
        scene_id: str,
        json_nodes_list: List[Dict[str, Any]],
        json_edges_raw: List[Tuple[int, int, str]],
        selected_indices: List[int],
        target_size: int,
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Occlusion-augmented subgraph sampling:
          - Keep the original stem-biased BFS subgraph as the core.
          - Choose a few focus fruits *randomly* from the core.
          - Add a budgeted set of nearby leaves around those fruits (top occluders when sidecars exist).
          - Guardrail: try a few resamples to include at least one "high-union" fruit (union>=tau) if available.
        """
        # JSON semantic IDs (1-based).
        STEM, LEAF, PED, FRUIT = 1, 2, 3, 4

        # Focus fruits must already be in the BFS core (preserves existing structural distribution).
        fruits_in_core = [i for i in selected_indices if int(json_nodes_list[i].get("semantic_id", -1)) == FRUIT]
        if not fruits_in_core:
            return selected_indices

        # For stable sampling, always use the canonical base directions from cam sidecars
        # (jittered directions are an eval robustness target, not a sampling control signal).
        sidecar = self._load_occl_sidecar(scene_id, cam_jitter_level_override=0)
        union_max_by_iid: Dict[int, float] = {}
        union_vec_by_iid: Dict[int, np.ndarray] = {}
        if sidecar is not None and sidecar["fruit_iid"].size > 0:
            fruit_iid_arr = sidecar["fruit_iid"]
            union_arr = sidecar["fruit_union"]  # [F, D]
            max_union = union_arr.max(axis=1)
            for iid, mu, uv in zip(fruit_iid_arr.tolist(), max_union.tolist(), union_arr):
                union_max_by_iid[int(iid)] = float(mu)
                union_vec_by_iid[int(iid)] = uv

        # Pick focus fruits (random) with the "at least one high-union" guardrail.
        num_focus = min(max(self.occl_aug_num_fruits, 1), len(fruits_in_core))
        high_union_fruits = []
        if union_max_by_iid:
            for idx in fruits_in_core:
                iid = int(json_nodes_list[idx].get("instance_id", -1))
                if union_max_by_iid.get(iid, 0.0) >= self.occl_aug_union_tau:
                    high_union_fruits.append(idx)

        focus_fruits: List[int] = []
        tries = max(self.occl_aug_resample_tries, 1)
        for _ in range(tries):
            focus_fruits = random.sample(fruits_in_core, k=num_focus)
            if not high_union_fruits:
                break
            if any(i in high_union_fruits for i in focus_fruits):
                break
        # If we repeatedly failed but high-union fruits exist, force-include one.
        if high_union_fruits and not any(i in high_union_fruits for i in focus_fruits):
            forced = random.choice(high_union_fruits)
            if forced not in focus_fruits:
                # Replace a random slot.
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
            return selected_indices
        leaf_centroids_all_arr = np.stack(leaf_centroids_all, axis=0)  # [L,3]

        # Build adjacency for finding stem neighbors of peduncles.
        adj = self.sampler.build_adjacency(num_nodes, json_edges_raw)
        fruit_to_ped, _, _, _ = self.sampler.build_pair_maps(num_nodes, json_nodes_list, json_edges_raw)

        core_required: List[int] = []
        pos_leaves_ordered: List[int] = []
        neg_leaves_ordered: List[int] = []

        # Always keep the focus fruits themselves.
        for fidx in focus_fruits:
            if fidx not in core_required:
                core_required.append(fidx)

        # Add chain nodes and leaf candidates for each focus fruit.
        for fidx in focus_fruits:
            fruit_node = json_nodes_list[fidx]
            fruit_iid = int(fruit_node.get("instance_id", -1))
            fruit_c = np.asarray(fruit_node.get("centroid"), dtype=np.float32)

            # Fruit <-> Peduncle (hard constraint).
            ped_idx = fruit_to_ped.get(fidx)
            if ped_idx is not None:
                if ped_idx not in core_required:
                    core_required.append(int(ped_idx))
                # Peduncle -> Stem neighbor (via attached_to adjacency).
                for nbr in adj[int(ped_idx)]:
                    if int(json_nodes_list[nbr].get("semantic_id", -1)) == STEM:
                        if nbr not in core_required:
                            core_required.append(int(nbr))
                        break

            # Candidate leaves by centroid radius (world).
            dists = np.linalg.norm(leaf_centroids_all_arr - fruit_c[None, :], axis=1)
            cand_mask = dists <= self.occl_aug_max_dist
            cand_pos = np.nonzero(cand_mask)[0]
            if cand_pos.size == 0:
                continue
            cand_leaf_json_idx = [leaf_indices_all[j] for j in cand_pos.tolist()]
            cand_leaf_json_set = set(cand_leaf_json_idx)

            # Pick positive occluder leaves (prefer sidecar-based scoring).
            picked_pos: List[int] = []
            if sidecar is not None and fruit_iid >= 0:
                union_vec = union_vec_by_iid.get(fruit_iid, None)
                dom_mask = None
                if union_vec is not None:
                    dom_mask = union_vec >= self.occl_aug_union_tau

                pair_mask = (sidecar["pair_fruit_iid"] == fruit_iid)
                if pair_mask.any():
                    leaf_iids = sidecar["pair_leaf_iid"][pair_mask].astype(np.int32, copy=False)
                    pot = sidecar["pair_potential"][pair_mask].astype(np.float32, copy=False)  # [R,D]
                    if dom_mask is not None and bool(dom_mask.any()):
                        score = pot[:, dom_mask].max(axis=1)
                    else:
                        score = pot.max(axis=1)
                    order = np.argsort(-score)
                    for j in order.tolist():
                        if len(picked_pos) >= self.occl_aug_max_pos_leaves:
                            break
                        liid = int(leaf_iids[j])
                        lj = iid_to_json_idx.get(liid)
                        if lj is None:
                            continue
                        if lj not in cand_leaf_json_set:
                            continue
                        if lj not in picked_pos:
                            picked_pos.append(lj)

            # Fallback: nearest leaves.
            if not picked_pos:
                # nearest by distance to fruit
                order = np.argsort(dists[cand_pos])
                for j in order.tolist():
                    if len(picked_pos) >= self.occl_aug_max_pos_leaves:
                        break
                    picked_pos.append(cand_leaf_json_idx[j])

            for lj in picked_pos:
                if lj not in pos_leaves_ordered:
                    pos_leaves_ordered.append(lj)

            # Random negatives from the candidate leaf pool (excludes picked positives).
            if self.occl_aug_max_neg_leaves > 0:
                neg_pool = [lj for lj in cand_leaf_json_idx if lj not in set(picked_pos)]
                if neg_pool:
                    k = min(self.occl_aug_max_neg_leaves, len(neg_pool))
                    picked_neg = random.sample(neg_pool, k=k)
                    for lj in picked_neg:
                        if lj not in neg_leaves_ordered:
                            neg_leaves_ordered.append(lj)

        # Global cap on leaf nodes to avoid blowing up the subgraph.
        if self.occl_aug_max_total_leaves > 0:
            if len(pos_leaves_ordered) > self.occl_aug_max_total_leaves:
                pos_leaves_ordered = pos_leaves_ordered[: self.occl_aug_max_total_leaves]
                neg_leaves_ordered = []
            else:
                rem = self.occl_aug_max_total_leaves - len(pos_leaves_ordered)
                if len(neg_leaves_ordered) > rem:
                    neg_leaves_ordered = neg_leaves_ordered[:rem]

        if debug is not None:
            debug["core_required"] = list(core_required)
            debug["pos_leaves"] = list(pos_leaves_ordered)
            debug["neg_leaves"] = list(neg_leaves_ordered)

        # Assemble final selection.
        #
        # Two modes:
        # 1) If BFS core is smaller than the target size, we *preserve* the BFS core and only append
        #    occl-relevant nodes into the remaining capacity. This is the preferred "reserve capacity"
        #    behavior because it minimizes disruption to existing tasks.
        # 2) If BFS already filled the full budget (len(core)==target_size), we fall back to the older
        #    behavior that prioritizes occlusion signal first and may evict some BFS nodes (otherwise
        #    occl-aug would add nothing).
        preserve_core = (len(selected_indices) < int(target_size))

        final: List[int] = []
        final_set: Set[int] = set()

        def _try_add(i: int) -> None:
            if i in final_set:
                return
            if len(final) >= int(target_size):
                return
            final_set.add(i)
            final.append(i)

        if preserve_core:
            # Start from the BFS core and append augmentation into free slots.
            for i in selected_indices:
                _try_add(int(i))
            for i in core_required:
                _try_add(int(i))
            for i in pos_leaves_ordered:
                _try_add(int(i))
            for i in neg_leaves_ordered:
                _try_add(int(i))
            return final

        # Fallback: prioritize occlusion signal, then add as much BFS structure as possible.
        for i in core_required:
            _try_add(int(i))
        for i in pos_leaves_ordered:
            _try_add(int(i))
        for i in neg_leaves_ordered:
            _try_add(int(i))
        for i in selected_indices:
            _try_add(int(i))

        return final

    def _augment(self, points, normals, centroids, mode: str = "continuous"):
        """
        Z-rotation augmentation.

        Mode:
          - continuous: random theta ~ U(0, 2pi)
          - discrete90: theta in {0, 90, 180, 270} deg (avoid tiny floating-point remnants)

        NOTE: Our z-buffer occlusion targets are defined in a fruit-local frame (+X toward stem in XY, +Z up),
        so global Z-rotation does not require permuting direction-indexed occlusion labels.
        """
        if mode == "discrete90":
            # {0, 90, 180, 270} deg (closed under the 18-dir basis).
            k = int(np.random.randint(0, 4))
            theta = float(k) * (0.5 * np.pi)
        else:
            theta = float(np.random.uniform(0, 2 * np.pi))

        c, s = float(np.cos(theta)), float(np.sin(theta))
        if mode == "discrete90":
            # Avoid tiny floating point remnants (e.g. 6e-17).
            c = float(np.round(c))
            s = float(np.round(s))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return points @ R.T, normals @ R.T, centroids @ R.T, R

    @staticmethod
    def _compute_dir_inv_perm(dirs: np.ndarray, R: np.ndarray, tol: float = 1e-4) -> Optional[np.ndarray]:
        """
        Compute an inverse permutation for direction-indexed labels under a scene rotation.

        dirs: [D,3] original direction vectors (world)
        R: [3,3] rotation applied to scene points (row-vector convention: p' = p @ R.T)

        Returns inv_perm [D] such that new_labels[..., i] = old_labels[..., inv_perm[i]].
        """
        if dirs is None:
            return None
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            return None
        D = int(dirs.shape[0])
        if D == 0:
            return None

        # Normalize to unit vectors for robust matching.
        dn = dirs.astype(np.float32, copy=False)
        dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-12)

        rot = dn @ R.T
        rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12)

        # rot[j] should match some dn[i].
        dot = rot @ dn.T  # [D,D]
        perm = np.argmax(dot, axis=1).astype(np.int64)  # old j -> new i
        best = dot[np.arange(D), perm]

        # Verify near-exact match and bijection.
        if float(best.min()) < (1.0 - float(tol)):
            return None
        if np.unique(perm).size != D:
            return None

        inv_perm = np.empty((D,), dtype=np.int64)
        inv_perm[perm] = np.arange(D, dtype=np.int64)
        return inv_perm

    def __getitem__(self, idx: int):
        scene_idx = self.indices[idx]
        pc_filename = self.pc_files[scene_idx]
        scene_id = pc_filename.split('_')[0] + '_' + pc_filename.split('_')[1]
        pc_path = os.path.join(self.data_dir, pc_filename)
        json_path = os.path.join(self.data_dir, f"{scene_id}_graph.json")
        
        data_npz = np.load(pc_path)
        points, normals, colors = data_npz['points'], data_npz['normals'], data_npz['colors']
        semantics, instances = data_npz['semantics'], data_npz['instances']
        
        mask = instances > 0
        if self.drop_shoots: mask = mask & (semantics != 5)
        points, normals, colors, semantics, instances = points[mask], normals[mask], colors[mask], semantics[mask], instances[mask]
        
        if self.merge_shoots and not self.drop_shoots: semantics[semantics == 5] = 2
        semantics = semantics - 1 # 0-indexed
            
        unique_insts = np.unique(instances)
        json_nodes_list, json_edges_raw = [], []
        load_json = (not self.drop_json) and os.path.exists(json_path)
        if load_json:
            json_nodes_list, json_edges_raw, _ = self.sampler.load_json_graph(json_path)
            
        if self.subgraph_enabled and load_json:
            target_size = self.subgraph_size
            if self.subgraph_size_min is not None and self.subgraph_size_min < self.subgraph_size:
                target_size = np.random.randint(self.subgraph_size_min, self.subgraph_size + 1)

            # If occl-aug is enabled, optionally reserve capacity so augmentation can inject leaves
            # without evicting arbitrary BFS nodes.
            core_target = int(target_size)
            if self.occl_aug_enabled and int(self.occl_aug_reserve_nodes) > 0:
                core_target = max(1, int(target_size) - int(self.occl_aug_reserve_nodes))

            selected_indices = self.sampler.bfs_sample(json_nodes_list, json_edges_raw, target_size=core_target, seed_policy="stem_bias")
            if self.occl_aug_enabled:
                occl_debug = {} if self.occl_targets_enabled else None
                selected_indices = self._augment_subgraph_with_occlusion(
                    scene_id,
                    json_nodes_list,
                    json_edges_raw,
                    selected_indices,
                    target_size=target_size,
                    debug=occl_debug,
                )
            else:
                occl_debug = None
            selected_inst_ids = {json_nodes_list[idx]['instance_id'] for idx in selected_indices}
            subgraph_mask = np.isin(instances, list(selected_inst_ids))
            points, normals, colors, semantics, instances = points[subgraph_mask], normals[subgraph_mask], colors[subgraph_mask], semantics[subgraph_mask], instances[subgraph_mask]
            unique_insts = np.unique(instances)
        else:
            occl_debug = None

        # Point-level augmentations (safe for direction-indexed occlusion targets).
        # Enabled implicitly by non-zero rates; `self.augment` acts as an explicit master switch.
        if self.split == "train" and self.augment:
            # 1) Point dropout: randomly drop points but keep at least one per instance.
            p_drop = float(self.point_dropout)
            if p_drop > 0.0:
                keep = (np.random.rand(points.shape[0]) >= p_drop)
                # Ensure every instance keeps at least one point.
                _, first_idx = np.unique(instances, return_index=True)
                keep[first_idx] = True
                points, normals, colors, semantics, instances = (
                    points[keep],
                    normals[keep],
                    colors[keep],
                    semantics[keep],
                    instances[keep],
                )

            # 2) XYZ jitter (Gaussian).
            j = float(self.xyz_jitter)
            if j > 0.0:
                points = points + np.random.normal(0.0, j, size=points.shape).astype(points.dtype, copy=False)

            # 3) Color dropout: zero RGB for a random subset of points.
            c_drop = float(self.color_dropout)
            if c_drop > 0.0:
                m = (np.random.rand(colors.shape[0]) < c_drop)
                if m.any():
                    colors = np.asarray(colors).copy()
                    colors[m] = 0

        unique_insts = np.unique(instances)
        np.random.shuffle(unique_insts)
        inst_map = {val: i for i, val in enumerate(unique_insts)}
        
        # Build GT edges
        json_nodes_dict = {n['instance_id']: n for n in json_nodes_list} if load_json else {}
        json_edges = []
        if load_json:
            for u_idx, v_idx, _ in json_edges_raw:
                u_iid, v_iid = json_nodes_list[u_idx]['instance_id'], json_nodes_list[v_idx]['instance_id']
                if u_iid in inst_map and v_iid in inst_map:
                    json_edges.append((u_iid, v_iid))

        # Optimized Node Attribute Gathering
        centroids, extents, node_labels = [], [], []
        # Point-derived geometry (deployable): cluster means and a simple size proxy for leafset selection.
        # These do NOT use JSON centroids/extents and are stable under Z-rotation (size proxy is rotation-invariant).
        centroids_pc, radii_pc = [], []
        # Pre-group indices for speed
        inst_to_idx = {iid: np.where(instances == iid)[0] for iid in unique_insts}
        
        for i, iid in enumerate(unique_insts):
            idx_list = inst_to_idx[iid]
            node_labels.append(semantics[idx_list[0]])
            inst_pts = points[idx_list]
            c_pc = inst_pts.mean(axis=0)
            centroids_pc.append(c_pc)
            # Size proxy: max radius from centroid (rotation-invariant, robust enough for stratified sampling).
            if inst_pts.shape[0] > 0:
                dif = inst_pts - c_pc[None, :]
                r_pc = float(np.sqrt((dif * dif).sum(axis=1)).max())
            else:
                r_pc = 0.0
            radii_pc.append(r_pc)
            if iid in json_nodes_dict:
                node = json_nodes_dict[iid]
                centroids.append(node['centroid'])
                extents.append(node['extent'])
            else:
                e = inst_pts.max(axis=0) - inst_pts.min(axis=0)
                centroids.append(c_pc); extents.append(e)

        centroids = np.array(centroids)
        extents = np.array(extents)
        node_labels = np.array(node_labels)
        centroids_pc = np.asarray(centroids_pc, dtype=np.float32)
        radii_pc = np.asarray(radii_pc, dtype=np.float32)

        # Z-rotation augmentation:
        # - Controlled by `rotate_aug` and `rotate_mode` (train split only).
        # - IMPORTANT: our z-buffer occlusion targets are defined in a fruit-local frame, so global Z-rotation
        #   does NOT permute direction indices. (Earlier versions attempted to permute and hurt learning.)
        R_aug = None
        if self.split == "train" and self.rotate_aug:
            points, normals, centroids, R_aug = self._augment(points, normals, centroids, mode=self.rotate_mode)
            # Keep point-derived centroids consistent with rotated points for candidate radius selection.
            centroids_pc = centroids_pc @ R_aug.T

        # Build GT Edge Tensors
        gt_rel_map = {}
        for s_iid, d_iid in json_edges:
            s_idx, d_idx = inst_map[s_iid], inst_map[d_iid]
            pair = tuple(sorted((s_idx, d_idx)))
            s_sem, d_sem = node_labels[s_idx], node_labels[d_idx]
            pair_sems = {s_sem, d_sem}
            label = -1
            if pair_sems == {0, 1}: label = 0 # Stem-Leaf
            elif pair_sems == {0, 2}: label = 1 # Stem-Ped
            elif pair_sems == {2, 3}: label = 2 # Ped-Fruit
            if label != -1: gt_rel_map[pair] = label

        gt_u, gt_v, gt_lab = [], [], []
        for (u, v), l in gt_rel_map.items():
            gt_u.append(u); gt_v.append(v); gt_lab.append(l)

        # [v12 GPU Acceleration] Mapping points to node IDs
        # Instead of looping M times, we do one scan
        node_id_per_point = np.full(points.shape[0], -1, dtype=np.int64)
        for i, iid in enumerate(unique_insts):
            node_id_per_point[inst_to_idx[iid]] = i

        # Optional: occlusion supervision targets from sidecars.
        # We always materialize tensors (possibly empty) to keep PyG batching stable.
        D = 18
        K = 3
        occl_fruit_idx = torch.empty((0,), dtype=torch.long)
        occl_fruit_union = torch.empty((0, D), dtype=torch.float32)
        occl_pair_index = torch.empty((2, 0), dtype=torch.long)
        occl_pair_potential = torch.empty((0, D), dtype=torch.float32)
        occl_pair_exclusive = torch.empty((0, D), dtype=torch.float32)
        occl_pair_k_rank = torch.empty((0, D, K), dtype=torch.float32)

        if self.occl_targets_enabled and load_json:
            sidecar = self._load_occl_sidecar(scene_id)
            if sidecar is not None:
                # NOTE: The z-buffer sidecars store direction-indexed targets in a *fruit-local* coordinate
                # frame (+X toward stem in XY, +Z up). Global Z-rotation augmentation rotates fruit and stem
                # together, so the canonical direction indices remain invariant. Therefore we intentionally
                # do NOT permute direction-indexed occlusion labels under Z-rotation augmentation (`--z_rot_aug_mode`).
                inv_perm = None

                # Determine focus fruits: prefer occl-aug focus list when available.
                focus_fruit_iids: List[int] = []
                if isinstance(occl_debug, dict) and occl_debug.get("focus_fruits"):
                    for jidx in occl_debug["focus_fruits"]:
                        try:
                            focus_fruit_iids.append(int(json_nodes_list[int(jidx)]["instance_id"]))
                        except Exception:
                            continue
                else:
                    # Fallback: random fruits present in this sample.
                    fruit_iids_present = [int(iid) for iid in unique_insts.tolist() if int(iid) in json_nodes_dict and json_nodes_dict[int(iid)].get("type") == "fruit"]
                    if fruit_iids_present:
                        k = min(max(self.occl_aug_num_fruits, 1), len(fruit_iids_present))
                        focus_fruit_iids = random.sample(fruit_iids_present, k=k)

                # Map focus fruit iids -> node indices and fetch union vectors.
                fruit_iid_arr = sidecar["fruit_iid"]
                fruit_union_arr = sidecar["fruit_union"]  # [F, D]
                fruit_iid_to_row = {int(iid): int(r) for r, iid in enumerate(fruit_iid_arr.tolist())}

                # Sidecar pair tensors used for both supervision materialization and (optionally) for selecting
                # additional supervised fruits that are explainable under the present leaves.
                pair_fruit_iid_all = sidecar["pair_fruit_iid"]
                pair_leaf_iid_all = sidecar["pair_leaf_iid"]
                pot_all = sidecar["pair_potential"]  # [R, D]
                exc_all = sidecar.get("pair_exclusive", None)  # [R, D] or None
                k_rank_all = sidecar["pair_k_rank"]  # [R, D, 3]

                # Leaf iid set present in the sampled subgraph (used for "present-only" explainability checks).
                leaf_node_idx_all = np.nonzero(node_labels == 1)[0]
                leaf_iids_present: Set[int] = set(int(unique_insts[int(i)]) for i in leaf_node_idx_all.tolist())

                def _present_union_for_fruit_iid(fruit_iid: int) -> np.ndarray:
                    """Approximate union occlusion using only leaves present in this sampled subgraph.

                    We sum exclusive (rank-1) occlusion mass over present leaves; this partitions union mass over leaves.
                    Returns [D] float32 in the canonical fruit-local direction indexing.
                    """
                    fmask = (pair_fruit_iid_all == int(fruit_iid))
                    if not bool(fmask.any()):
                        out = np.zeros((fruit_union_arr.shape[1],), dtype=np.float32)
                        return out
                    rows = np.nonzero(fmask)[0]
                    if rows.size == 0:
                        out = np.zeros((fruit_union_arr.shape[1],), dtype=np.float32)
                        return out
                    # Filter to leaves that exist in this sampled subgraph.
                    keep = [int(r) for r in rows.tolist() if int(pair_leaf_iid_all[int(r)]) in leaf_iids_present]
                    if not keep:
                        out = np.zeros((fruit_union_arr.shape[1],), dtype=np.float32)
                        return out
                    if exc_all is not None:
                        u = exc_all[np.asarray(keep, dtype=np.int64)].sum(axis=0)
                    else:
                        # Fallback: rank-1 (frontmost) mass from k-buffer.
                        u = k_rank_all[np.asarray(keep, dtype=np.int64), :, 0].sum(axis=0)
                    u = np.clip(u, 0.0, 1.0).astype(np.float32, copy=False)
                    return u

                fruit_node_indices: List[int] = []
                fruit_iids_used: List[int] = []
                fruit_union_rows: List[np.ndarray] = []
                for iid in focus_fruit_iids:
                    if iid not in inst_map:
                        continue
                    nidx = int(inst_map[iid])
                    # Ensure semantics agrees (fruit=3 in 0-indexed Oracle scheme).
                    if int(node_labels[nidx]) != 3:
                        continue
                    row = fruit_iid_to_row.get(int(iid))
                    if row is None:
                        continue
                    fruit_iids_used.append(int(iid))
                    fruit_node_indices.append(nidx)
                    uv = fruit_union_arr[row]
                    fruit_union_rows.append(uv)

                # Optionally supervise additional fruits in the sampled subgraph that have "explainable" dominant
                # directions under the present leaves. This increases occlusion supervision density without asking the
                # model to predict dominant directions caused by missing leaves.
                extra_fruit_iids_added: List[int] = []
                if int(self.occl_supervise_max_fruits_per_scene) > 0 and len(fruit_iids_used) > 0:
                    max_total = int(self.occl_supervise_max_fruits_per_scene)
                    tau = float(self.occl_targets_union_tau)
                    min_dom_dirs = max(int(self.occl_supervise_min_dom_dirs), 1)
                    remain_min = float(self.occl_supervise_remain_rate_min)

                    # Candidate fruits present in the sampled subgraph.
                    fruit_node_idx_all = np.nonzero(node_labels == 3)[0]
                    cand_fruit_iids = [int(unique_insts[int(i)]) for i in fruit_node_idx_all.tolist()]

                    # Score candidates by union strength (descending).
                    cand_scored: List[Tuple[float, int, int, np.ndarray]] = []
                    used_set = set(int(i) for i in fruit_iids_used)
                    for fiid in cand_fruit_iids:
                        if fiid in used_set:
                            continue
                        row = fruit_iid_to_row.get(int(fiid))
                        if row is None:
                            continue
                        nidx = int(inst_map.get(int(fiid), -1))
                        if nidx < 0 or int(node_labels[nidx]) != 3:
                            continue
                        uv = fruit_union_arr[row]
                        uv = uv.astype(np.float32, copy=False)
                        if float(uv.max()) < tau:
                            continue
                        dom_mask = (uv >= tau)
                        if int(dom_mask.sum()) < int(min_dom_dirs):
                            continue
                        u_present = _present_union_for_fruit_iid(int(fiid))
                        # Fraction of GT-dominant directions that remain dominant under present leaves.
                        remain_rate = float((u_present[dom_mask] >= tau).mean()) if bool(dom_mask.any()) else 0.0
                        if remain_rate < remain_min:
                            continue
                        cand_scored.append((float(uv.max()), int(fiid), int(nidx), uv))

                    cand_scored.sort(key=lambda t: t[0], reverse=True)
                    for _, fiid, nidx, uv in cand_scored:
                        if len(fruit_iids_used) >= max_total:
                            break
                        fruit_iids_used.append(int(fiid))
                        fruit_node_indices.append(int(nidx))
                        fruit_union_rows.append(uv)
                        extra_fruit_iids_added.append(int(fiid))

                if isinstance(occl_debug, dict) and extra_fruit_iids_added:
                    occl_debug["occl_supervised_extra_fruits_iid"] = list(extra_fruit_iids_added)
                    occl_debug["occl_supervised_total_fruits"] = int(len(fruit_iids_used))

                if fruit_node_indices:
                    occl_fruit_idx = torch.tensor(fruit_node_indices, dtype=torch.long)
                    occl_fruit_union = torch.from_numpy(np.stack(fruit_union_rows, axis=0)).float()  # [Ff, D]

                    # Leaf candidate pool (within max_dist of each focus fruit).
                    # (Reuses leaf_node_idx_all computed above.)

                    pairs_f: List[List[int]] = []
                    pairs_pot: List[np.ndarray] = []
                    pairs_exc: List[np.ndarray] = []
                    pairs_k: List[np.ndarray] = []

                    max_leaves = max(int(self.occl_targets_max_leaves_per_fruit), 1)

                    # Optional priority leaves from occl-aug (if enabled and debug was collected).
                    aug_pos_iids: Set[int] = set()
                    aug_neg_iids: Set[int] = set()
                    if isinstance(occl_debug, dict):
                        for jidx in occl_debug.get("pos_leaves", []) or []:
                            try:
                                aug_pos_iids.add(int(json_nodes_list[int(jidx)]["instance_id"]))
                            except Exception:
                                pass
                        for jidx in occl_debug.get("neg_leaves", []) or []:
                            try:
                                aug_neg_iids.add(int(json_nodes_list[int(jidx)]["instance_id"]))
                            except Exception:
                                pass

                    for fruit_iid, fruit_node_idx in zip(fruit_iids_used, fruit_node_indices):
                        # Candidate leaves by centroid radius.
                        # Use point-derived centroids (cluster means) rather than JSON centroids so this leafset
                        # selection remains deployable and does not depend on GT geometry injection.
                        fruit_c = centroids_pc[fruit_node_idx]
                        leaf_pool = leaf_node_idx_all
                        if leaf_pool.size == 0:
                            continue
                        dists = np.linalg.norm(centroids_pc[leaf_pool] - fruit_c[None, :], axis=1)
                        cand_mask = (dists <= self.occl_aug_max_dist)
                        cand = leaf_pool[cand_mask]
                        if cand.size == 0:
                            continue

                        cand_list = cand.tolist()
                        picked: List[int] = []
                        if self.occl_leafset_policy == "sidecar_priority":
                            # Legacy: include occl-aug selected leaves (pos/neg), then fill with random nearby leaves.
                            def _try_add_leaf_iid(iid: int) -> None:
                                if iid not in inst_map:
                                    return
                                li = int(inst_map[iid])
                                if li in picked:
                                    return
                                if li not in cand_list:
                                    return
                                picked.append(li)

                            for iid in list(aug_pos_iids):
                                _try_add_leaf_iid(iid)
                            for iid in list(aug_neg_iids):
                                _try_add_leaf_iid(iid)

                            # Fill remaining budget randomly from candidate pool.
                            if len(picked) < max_leaves:
                                rem = [li for li in cand_list if li not in picked]
                                if rem:
                                    k = min(max_leaves - len(picked), len(rem))
                                    picked.extend(random.sample(rem, k=k))
                        else:
                            # New default: geometry-only stratified selection (no sidecar/GT-driven priority leaf picks).
                            if len(cand_list) <= max_leaves:
                                picked = cand_list
                            else:
                                # Stable per (scene, fruit) RNG: keeps selection consistent across workers.
                                try:
                                    scene_num = int(str(scene_id).split("_")[1])
                                except Exception:
                                    scene_num = 0
                                seed = (int(scene_num) * 1000003 + int(fruit_iid)) & 0xFFFFFFFF
                                rng = np.random.default_rng(seed)

                                # Score: bigger + closer leaves are more likely to occlude.
                                d_cand = dists[cand_mask].astype(np.float32, copy=False)
                                sizes = radii_pc[cand].astype(np.float32, copy=False)
                                score = sizes / (d_cand + 1e-3)
                                order = np.argsort(-score)  # indices into `cand`

                                n = int(cand.shape[0])
                                k_total = min(int(max_leaves), n)
                                k_low = max(1, int(round(0.125 * float(k_total))))
                                k_mid = k_low
                                k_top = k_total - k_mid - k_low
                                if k_top < 1:
                                    k_top = max(1, k_total - 2)
                                    k_mid = max(1, (k_total - k_top) // 2)
                                    k_low = k_total - k_top - k_mid

                                picked_set: Set[int] = set()

                                # 1) Top: deterministic highest-score leaves.
                                for oi in order[:k_top].tolist():
                                    li = int(cand[int(oi)])
                                    picked.append(li)
                                    picked_set.add(li)

                                # 2) Mid: sample from middle quantiles of score ranks.
                                mid_lo = int(0.25 * float(n))
                                mid_hi = int(0.75 * float(n))
                                mid_pool = [int(cand[int(oi)]) for oi in order[mid_lo:mid_hi].tolist() if int(cand[int(oi)]) not in picked_set]
                                if mid_pool and k_mid > 0:
                                    take = min(k_mid, len(mid_pool))
                                    mid_sel = rng.choice(np.asarray(mid_pool, dtype=np.int64), size=take, replace=False)
                                    for li in mid_sel.tolist():
                                        picked.append(int(li))
                                        picked_set.add(int(li))

                                # 3) Low: sample from the lowest-score quartile (small/far/unlikely occluders).
                                low_pool = [int(cand[int(oi)]) for oi in order[mid_hi:].tolist() if int(cand[int(oi)]) not in picked_set]
                                if low_pool and k_low > 0:
                                    take = min(k_low, len(low_pool))
                                    low_sel = rng.choice(np.asarray(low_pool, dtype=np.int64), size=take, replace=False)
                                    for li in low_sel.tolist():
                                        picked.append(int(li))
                                        picked_set.add(int(li))

                                # 4) Fill any remaining budget uniformly from what's left.
                                if len(picked) < k_total:
                                    rem = [li for li in cand_list if li not in picked_set]
                                    if rem:
                                        take = min(k_total - len(picked), len(rem))
                                        fill_sel = rng.choice(np.asarray(rem, dtype=np.int64), size=take, replace=False)
                                        for li in fill_sel.tolist():
                                            picked.append(int(li))
                                            picked_set.add(int(li))

                        # Build lookup for sidecar rows for this fruit.
                        fmask = (pair_fruit_iid_all == int(fruit_iid))
                        leaf_iid_to_row: Dict[int, int] = {}
                        if fmask.any():
                            rows = np.nonzero(fmask)[0]
                            for r in rows.tolist():
                                leaf_iid_to_row[int(pair_leaf_iid_all[r])] = int(r)

                        # Emit candidate pairs and their labels (zeros if absent).
                        for leaf_node_idx in picked:
                            leaf_iid = int(unique_insts[leaf_node_idx])
                            r = leaf_iid_to_row.get(int(leaf_iid))
                            if r is None:
                                pot = np.zeros((pot_all.shape[1],), dtype=np.float32)
                                exc = np.zeros((pot_all.shape[1],), dtype=np.float32)
                                kk = np.zeros((pot_all.shape[1], k_rank_all.shape[2]), dtype=np.float32)
                            else:
                                pot = pot_all[r].astype(np.float32, copy=False)
                                if exc_all is not None:
                                    exc = exc_all[r].astype(np.float32, copy=False)
                                else:
                                    exc = np.zeros((pot_all.shape[1],), dtype=np.float32)
                                kk = k_rank_all[r].astype(np.float32, copy=False)

                            pairs_f.append([int(fruit_node_idx), int(leaf_node_idx)])
                            pairs_pot.append(pot)
                            pairs_exc.append(exc)
                            pairs_k.append(kk)

                    if pairs_f:
                        pair_arr = np.asarray(pairs_f, dtype=np.int64).T  # [2, M]
                        occl_pair_index = torch.from_numpy(pair_arr).long()
                        occl_pair_potential = torch.from_numpy(np.stack(pairs_pot, axis=0)).float()
                        occl_pair_exclusive = torch.from_numpy(np.stack(pairs_exc, axis=0)).float()
                        occl_pair_k_rank = torch.from_numpy(np.stack(pairs_k, axis=0)).float()

        return OracleData(
            pos_subgraph=torch.from_numpy(points).float(),
            norm_subgraph=torch.from_numpy(normals).float(),
            color_subgraph=torch.from_numpy(colors).float(),
            node_id_per_point=torch.from_numpy(node_id_per_point).long(),
            json_geom=torch.from_numpy(np.concatenate([centroids, extents], axis=-1)).float(),
            y_node=torch.from_numpy(node_labels).long(),
            node_iid=torch.from_numpy(unique_insts.astype(np.int64, copy=False)).long(),
            scene_id=scene_id,
            num_nodes=len(unique_insts),
            gt_edge_index=torch.tensor([gt_u, gt_v], dtype=torch.long),
            gt_edge_label=torch.tensor(gt_lab, dtype=torch.long),
            num_gt_total=torch.tensor([len(gt_rel_map)]),
            occl_fruit_idx=occl_fruit_idx,
            occl_fruit_union=occl_fruit_union,
            occl_pair_index=occl_pair_index,
            occl_pair_potential=occl_pair_potential,
            occl_pair_exclusive=occl_pair_exclusive,
            occl_pair_k_rank=occl_pair_k_rank,
        )

def oracle_collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
