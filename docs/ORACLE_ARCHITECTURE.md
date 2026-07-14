# Oracle Joint Model Architecture: V12 Elite (OracleResBaseline)

This document details the architectural specifications of the **Oracle Joint Model (V12 Elite)**, implemented as `OracleResBaseline`. This model is designed for end-to-end scene graph generation and geometric reasoning directly from segmented 3D point clouds of complex botanical structures. The architecture synergistically combines local geometric feature extraction with global topological message passing, supplemented by specialized prediction heads for structure estimation and occlusion reasoning.

## 1. Local Node Feature Extraction (PointNet++ MSG)

At the foundation of the architecture is the `SharedMSGEncoder`, a PointNet++ Multi-Scale Grouping (MSG) encoder. Unlike standard semantic segmentation networks that classify individual points, this encoder operates on pre-segmented *instances* (organs).

- **Input Representation**: Each node $i$ in the initial graph represents a segmented organ, provided as an unordered set of points $P_i \in \mathbb{R}^{N \times 3}$, alongside corresponding surface normals $N_i \in \mathbb{R}^{N \times 3}$ and optional RGB colors $C_i \in \mathbb{R}^{N \times 3}$.
- **Multi-Scale Grouping**: The encoder processes points at multiple spatial resolutions (e.g., radii of 0.01m, 0.02m, and 0.04m) to capture both fine-grained surface textures and coarser topological shapes.
- **Aggregation**: Extracted multi-scale features are concatenated and max-pooled across the sample dimension, yielding a single, permutation-invariant identity embedding $x_{local}^{(i)} \in \mathbb{R}^{D_{feat}}$ for each node.
- **Pre-GNN Supervision**: To ensure that the local encoder learns robust biological representations independently of the graph structure, a *Local Node Head* applies auxiliary cross-entropy supervision directly to $x_{local}^{(i)}$.

## 2. Graph Construction & Geometric Edge Features

Before global routing, a candidate graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is constructed.
- **Candidate Edges**: Edges are proposed dynamically based on spatial proximity utilizing a combined Radius Graph and k-Nearest Neighbors (k-NN) strategy, supplemented by deterministic heuristic links (e.g., connecting nodes to the nearest main stems).
- **Geometric Edge Attributes**: For every directed edge $e_{u,v}$, a geometric attribute vector $w_{u,v} \in \mathbb{R}^4$ is calculated involving the relative 3D displacement vector between centroids and the absolute Euclidean distance.

## 3. Structural Reasoning: ResGINE Backbone

To model the complex relational topology of plant organs (e.g., leaves attached to stems, fruits borne on peduncles), the initial node embeddings are refined via message passing.

- **Architecture**: The backbone employs a **Residual Graph Isomorphism Network with Edge Features (ResGINE)**.
- **Message Formulation**: The standard GINE operator is used to integrate edge attributes into node messages.
- **Anti-Oversmoothing (Residual Connections)**: In highly connected and dense botanical subgraphs, standard GNNs suffer from oversmoothing, where node representations become indistinguishable. The `ResGINEBackbone` combats this by utilizing skip connections (residual additions) between layers and applying layer-unique Multi-Layer Perceptrons (MLPs) to preserve distinct, high-frequency signals at each depth.

Let the output of the GNN for node $i$ be $x_{gnn}^{(i)} \in \mathbb{R}^{D_{gnn}}$.

## 4. Feature Persistence: Identity Concatenation

A hallmark of the V12 Elite architecture is **Identity Concatenation** (the "Safety Valve" mechanism). Rather than discarding the local geometric descriptor after message passing, the model concatenates the GNN-refined features with the original PointNet++ representation:

$$ x_{final}^{(i)} = x_{gnn}^{(i)} \parallel x_{local}^{(i)} $$

This ensures that critical low-level geometric signatures (e.g., the specific curvature of a fruit) are protected against topological blurring and are directly available to the terminal prediction heads.

## 5. Terminal Prediction Heads

The architecture utilizes the high-dimensional concatenated embeddings $x_{final}$ to perform multiple simultaneous downstream tasks.

### 5.1. Classification Heads
- **Global Node Head**: Predicts final organ semantics (Stem, Leaf, Peduncle, Fruit) from $x_{final}^{(i)}$.
- **Edge Existence & Relation Heads**: For every candidate edge $e_{u,v}$, the representations of the source node, target node, and the edge attributes are concatenated: $[x_{final}^{(u)} \parallel x_{final}^{(v)} \parallel w_{u,v}]$. This joint embedding is passed through parallel MLPs to predict the binary existence of the edge and classify its biological relation.
- **Evidential Deep Learning (EDL)**: The classification heads (Node, Edge Existence, Edge Label) employ an EDL formulation using a tied-Beta/Dirichlet parameterization. This replaces standard Softmax outputs with evidence parameters $(\alpha)$, enabling the model to express epistemic uncertainty (e.g., distinguishing between confident background vs. unknown structures).

### 5.2. Geometric Regression
- **Centroid Offset & Extent Deltas**: Using $x_{final}^{(i)}$, the model regresses corrections to the initial geometric assumptions of the bounding boxes, outputting a 3D relative centroid shift and an absolute extent (dimensions) prediction.

### 5.3. Direction-Aware Occlusion Reasoning
The architecture features a highly specialized subsystem for explicit, direction-aware occlusion prediction, substituting independent point-wise evaluations with listwise set operations.

1. **Leaf-Set Self-Attention**: Candidate occluding leaves surrounding a target fruit are processed via a Transformer encoder with padding masks. The leaf embeddings interact via self-attention, allowing the model to detect redundant occluders and spatial competition.
2. **Direction Conditioning**: Occlusion is not absolute; it is dependent on the viewing angle. The model injects additive, direction-specific geometric conditions via a learned local coordinate frame (Fruit-local +X toward stem).
3. **Cross-Attention Pooling**: An attention-style scorer pools the contextualized leaf tokens against the direction-embedded fruit token.
4. **Multi-Target Output**:
   - **Union Score**: Predicts the total cross-sectional area occluded by *all* leaves from a specific direction.
   - **Potential & Rank Scores**: For each leaf-fruit pair along a specific direction, the model outputs a *potential score* (continuous regression) and a *listwise rank score*. The rank score is supervised using $k$-buffer mass distribution to teach the model to identify the dominant ("front-most") occluders.

## Conclusion
The `OracleResBaseline` achieves biologically accurate scene graph generation by balancing high-fidelity point cloud processing (Shared MSG) with robust graph reasoning (ResGINE). The introduction of Identity Concatenation and Evidential Deep Learning ensures that geometric nuances are preserved and uncertainties are rigorously quantified, while the direction-aware attention pooling facilitates advanced physics-based occlusion estimation.

## 6. Training Objectives and Loss Formulations

The Oracle model is trained end-to-end using a heavily structured, multi-task curriculum. The total loss $\mathcal{L}_{total}$ is a dynamic combination of the following objectives, weighted by global hyperparameters and curriculum schedule gating:

### 6.1. Node Classification Losses

Because the model predicts semantics both before and after the GNN, the node loss is bipartite:
- **Local Pre-GNN Node Loss ($\mathcal{L}_{node\_local}$)**: Standard Cross-Entropy applied directly to the initial MSG representations ($x_{local}$). This primes the underlying PointNet++ to capture organic shapes independent of the graph environment.
- **Global Post-GNN Node Loss ($\mathcal{L}_{node}$)**: Applied to $x_{final}$. If Evidential Deep Learning (EDL) is enabled, this is cast as the expected Dirichlet Cross-Entropy augmented with a KL-divergence regularizer to penalize flat (high-uncertainty) evidence distributions. Otherwise, standard Cross-Entropy is used.

### 6.2. Structural Edge Losses

- **Edge Existence ($\mathcal{L}_{edge\_exist}$)**: Formulated as a Binary Cross Entropy (BCE) task. Because plant scene graphs are exceedingly sparse (most nodes are not connected), a strong `pos_weight` is necessary. The codebase supports both statically defined weights and *Dynamic Edge Weighting*, which adjusts the gradient scalar at every batch based on the precise ratio of negative to positive candidate edges.
- **Edge Labeling ($\mathcal{L}_{edge\_label}$)**: A standard multi-class objective (Cross-Entropy or EDL Dirichlet CE) trained *only* on the subset of edges where the ground-truth relationship exists ($y_{edge} \neq -1$).

### 6.3. Geometric Regression Losses

Both size and positional corrections are optimized using **Smooth L1 (Huber) Loss** to provide stable gradients against outliers (common in clustered point clouds):
- **Centroid Offset ($\mathcal{L}_{centroid}$)**: $\text{SmoothL1}(\Delta\hat{c}, \Delta c)$
- **Extent Delta ($\mathcal{L}_{extent}$)**: $\text{SmoothL1}(\Delta\hat{s}, \Delta s)$

### 6.4. Occlusion Objectives

The direction-aware occlusion system requires four interacting loss terms to supervise the attention pooling mechanism:
1. **Union Loss ($\mathcal{L}_{occl\_union}$)**: A BCE loss against the pointcloud z-buffer target. Crucially, it incorporates an *Explainability Weight* ($W_{explain}$): the loss is downscaled if the candidate sub-graph is missing leaves that contribute to the ground-truth occlusion, preventing the model from being unfairly penalized for missing data.
2. **Potential Loss ($\mathcal{L}_{occl\_potential}$)**: BCE applied per-(leaf, direction) pair. To prevent trivial optimization, training employs *Stratified Sampling*, actively selecting a balanced mix of dominant (highly occluded), mid, and low occlusion viewing angles rather than just filtering by a hard threshold.
3. **Listwise Rank Loss ($\mathcal{L}_{occl\_rank}$)**: To force the model to correctly identify front-most occluders, this term treats the leaf potential scores as a distribution and matches it to a continuous target distribution constructed from the geometric "K-buffer mass" (how often that leaf is in the top-K depth layers). It optimizes the negative expected log-likelihood.
4. **Consistency Regularizer ($\mathcal{L}_{occl\_consistency}$)**: A self-supervised constraint forcing the aggregate sum of the individual leaf occlusion potentials to match the predicted total fruit union. Depending on the EDL configuration, this acts either as an MSE regression constraint or a BCE dominance term.

### 6.5. Curriculum Learning

Training complex GNNs from scratch often leads to representational collapse if the structural updates outpace the local geometric learning. To prevent this, the loss terms are gated temporally:
1. **Phase 1 (Epochs 0 to $E_{reg}$)**: *Local Initialization*. Only $\mathcal{L}_{node\_local}$ is active. The PointNet++ MSG learns solid foundational features.
2. **Phase 2 (Epochs $E_{reg}$ to $E_{gnn}$)**: *Geometric Bootstrapping*. Regression losses ($\mathcal{L}_{centroid}, \mathcal{L}_{extent}$) turn on.
3. **Phase 3 (Epochs $E_{gnn}$ onward)**: *Global Reasoning*. $\mathcal{L}_{node}, \mathcal{L}_{edge\_exist}, \mathcal{L}_{edge\_label}$ switch on simultaneously, routing properly grounded local features through the ResGINE layers. Occlusion ($\mathcal{L}_{occl\_*}$), if enabled, joins shortly after graph convergence is achieved.
