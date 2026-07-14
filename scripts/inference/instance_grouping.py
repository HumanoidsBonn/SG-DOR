import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import mode
import os

# Classes: 0: Stem, 1: LeafMerged, 2: Peduncle, 3: Fruit
# Global ids: 0: Stem, 1: Leaf, 2: Shoot, 3: Peduncle, 4: Fruit
MERGED_TO_GLOBAL = {
    0: 0,  # Stem -> Stem
    1: 1,  # LeafMerged -> Leaf (Shoot is logically part of this)
    2: 2,  # Peduncle -> Peduncle (Mapped to 2)
    3: 3,  # Fruit -> Fruit (Mapped to 3)
}

class InstanceGrouper:
    def __init__(self, tau_sem=None, cluster_params=None, tau_purity=0.7, tau_quality=0.01):
        """
        Args:
            tau_sem (float or dict): Hard Semantic Gate. Points with max_prob < tau_sem are discarded.
                                     Can be a dict {class_id: threshold}.
                                     Default: {0:0.7, 1:0.7, 2:0.3, 3:0.6} (Peduncle permissive)
            tau_purity: Unused (Redundant with hard gating).
            tau_quality: Final quality threshold.
        """
        self.tau_sem = tau_sem or {
            0: 0.75, 
            1: 0.70, 
            2: 0.30, 
            3: 0.60 
        }
        self.tau_purity = tau_purity
        self.tau_quality = tau_quality
        
        # Default clustering parameters per class (Robust V4)
        self.params = cluster_params or {
            0: {'eps': 0.10,  'min_samples': 20, 'm_max': 0.00, 'd_max': 1.00, 'ignore_tightness': True}, # Stem: Keep loose
            1: {'eps': 0.018, 'min_samples': 15,  'm_max': 0.00, 'd_max': 0.08}, # Leaf: Revert to V4 (Golden: 525/529)
            2: {'eps': 0.030, 'min_samples': 4,   'm_max': 0.12, 'd_max': 0.10, 'ignore_tightness': True}, # Peduncle: Revert to V4 (Golden: 55/59)
            3: {'eps': 0.040, 'min_samples': 20,  'm_max': 0.07, 'd_max': 0.15}, # Fruit: Middle ground (0.033 over-seg, 0.050 lost)
        }

    def group(self, points, sem_probs, offsets, extents, features, exclude_stems=True):
        """
        Groups points into instances and synthesizes node features.
        """
        N = len(points)
        pred_labels = np.argmax(sem_probs, axis=1)
        max_probs = np.max(sem_probs, axis=1)

        # 1. Gating & Voted Shifting
        voted_points = np.copy(points)
        valid_mask = np.zeros(N, dtype=bool)

        for c, p in self.params.items():
            if isinstance(self.tau_sem, dict):
                t_s = self.tau_sem.get(c, 0.5)
            else:
                t_s = self.tau_sem
            
            class_mask = (pred_labels == c) & (max_probs > t_s)
            if not np.any(class_mask): continue
            
            # Magnitude Clamp & Voting
            off_c = offsets[class_mask]
            m_c = np.linalg.norm(off_c, axis=1, keepdims=True)
            scale = np.minimum(m_c, p['m_max']) / (m_c + 1e-8)
            voted_points[class_mask] = points[class_mask] + off_c * scale
            
            valid_mask[class_mask] = True

        all_nodes = []

        # 2. Class-wise Clustering
        for c, p in self.params.items():
            if exclude_stems and c == 0: continue
            
            c_indices = np.where((pred_labels == c) & valid_mask)[0]
            if len(c_indices) < p['min_samples']: continue

            v_c = voted_points[c_indices]
            db = DBSCAN(eps=p['eps'], min_samples=p['min_samples']).fit(v_c)
            labels = db.labels_

            # DEBUG: Cluster Counts
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sizes = [np.sum(labels==k) for k in set(labels) if k!=-1]
            print(f"DEBUG: Msg Class {c} -> Clusters: {n_clusters}, Median Size: {np.median(sizes) if sizes else 0:.1f}")

            unique_labels = set(labels)
            if -1 in unique_labels: unique_labels.remove(-1)

            for label in unique_labels:
                cluster_indices = c_indices[labels == label]
                
                # 3. Anti-Overmerge Split (Radius/Dispersion based)
                if c == 3:
                    sub_clusters =  [np.arange(len(cluster_indices))]
                else:
                    sub_clusters = self._recursive_split(voted_points[cluster_indices], p['d_max'])
                
                for sub_indices_relative in sub_clusters:
                    actual_indices = cluster_indices[sub_indices_relative]
                    if len(actual_indices) < p['min_samples']: continue
                    
                    # 4. Node Synthesis
                    node = self._synthesize_node(
                        actual_indices, points, sem_probs, offsets, extents, features, voted_points, c
                    )
                    if node is not None:
                        all_nodes.append(node)

        return all_nodes

    def _recursive_split(self, v_points, d_max):
        """Recursively splits clusters using KMeans if cluster radius exceeds d_max/2."""
        if len(v_points) < 2:
            return [np.arange(len(v_points))]

        center = np.mean(v_points, axis=0)
        r = np.max(np.linalg.norm(v_points - center, axis=1))  # max radius
        if 2.0 * r <= d_max:
            return [np.arange(len(v_points))]

        # Split into 2
        try:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(v_points)
            labels = km.labels_
        except:
             return [np.arange(len(v_points))] # Fallback

        res = []
        for i in [0, 1]:
            mask = (labels == i)
            if np.sum(mask) < 2:
                continue
            sub_res = self._recursive_split(v_points[mask], d_max)
            for sr in sub_res:
                res.append(np.where(mask)[0][sr])
        return res

    def _synthesize_node(self, idx, points, sem_probs, offsets, extents, features, voted_points, class_id):
        p_c = points[idx]
        f_c = features[idx]
        s_c = sem_probs[idx]
        v_c = voted_points[idx]
        off_c = offsets[idx]
        # ext_c = extents[idx] # Unused for now, AABB is primary

        # Median Centroid
        centroid = np.median(p_c, axis=0)

        # AABB Extents
        aabb = np.max(p_c, axis=0) - np.min(p_c, axis=0)
        
        # Orientation (Weighted Mean Direction)
        m_c = np.linalg.norm(off_c, axis=1, keepdims=True)
        u_c = off_c / (m_c + 1e-8)
        weights = np.max(s_c, axis=1, keepdims=True)
        orientation = np.sum(weights * u_c, axis=0)
        
        norm_o = np.linalg.norm(orientation)
        if norm_o < 1e-6:
            orientation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            orientation /= norm_o

        # Embedding (Mean + Max)
        z_node = np.concatenate([np.mean(f_c, axis=0), np.max(f_c, axis=0)]) # 256-D

        # Quality Score
        # 1. Tightness
        v_mean = np.mean(v_c, axis=0)
        var = np.mean(np.sum((v_c - v_mean)**2, axis=1))
        # Use class-specific tightness scale s_c
        if self.params[class_id].get('ignore_tightness', False):
             q_t = 1.0
        else:
             s_c_val = self.params[class_id]['eps'] * 0.5 
             q_t = np.exp(-var / (s_c_val**2 + 1e-8))

        # 2. Direction Consensus
        q_c = np.mean((1 + np.sum(u_c * orientation, axis=1)) / 2.0)

        # 3. Semantic Purity (Mode Agreement)
        # Note: Since we hard-gate by class before clustering, the cluster is by definition 100% pure 
        # for the predicted label. 
        # q_s = 1.0 (Redundant calculation removed)

        quality = q_t * q_c # * q_s (Implicitly 1.0)

        # Robust Gating
        # if q_s < self.tau_purity: return None (Removed as q_s is implicitly 1.0)
        if quality < self.tau_quality: return None

        # Assemble Feature Vector (271-D) using Global Schema
        sem_one_hot = np.zeros(5, dtype=np.float32)
        sem_one_hot[MERGED_TO_GLOBAL[class_id]] = 1.0
        
        feature_vec = np.concatenate([
            z_node, aabb, orientation, centroid, [quality], sem_one_hot
        ])

        return {
            'indices': idx,
            'class_id': class_id,
            'class_id_merged': class_id,
            'global_class_id': MERGED_TO_GLOBAL.get(class_id, class_id),
            'centroid': centroid,
            'extents': aabb,
            'orientation': orientation,
            'quality': quality,
            'feature_vec': feature_vec
        }

def build_pyg_data(nodes):
    """Converts node list to PyG Data object."""
    import torch
    from torch_geometric.data import Data
    
    if not nodes:
        return None
        
    x = torch.from_numpy(np.stack([n['feature_vec'] for n in nodes])).float()
    pos = torch.from_numpy(np.stack([n['centroid'] for n in nodes])).float()
    
    data = Data(x=x, pos=pos)
    data.node_quality = torch.from_numpy(np.array([n['quality'] for n in nodes])).float()
    data.node_class = torch.from_numpy(np.array([n['global_class_id'] for n in nodes])).long()
    data.node_size = torch.tensor([len(n['indices']) for n in nodes], dtype=torch.long)
    
    return data
