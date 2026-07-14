# Occlusion learning: experiment log + findings (supervisor report)

## 0) Goal & setup
**Goal:** Predict a *viewpoint-conditioned occlusion metric* between **fruit–leaf** pairs in 3D pepper-plant scenes.

**Ground truth (important):** occlusion is defined as **projected OBB silhouette overlap**, i.e. overlap of the **projected 8-corner OBB polygons** in the 2D plane orthogonal to a direction `d`. Concretely, for each direction `d` we build an orthonormal basis `(u,v)` spanning the plane perpendicular to `d`, project both OBBs onto that plane, compute their convex hull polygons, and define:

- `overlap_raw(d) = Area( Hull_fruit(d) ∩ Hull_leaf(d) ) / Area( Hull_fruit(d) )`

Optionally a distance weight is applied in data generation:
- `overlap_weighted(d) = overlap_raw(d) * w(dist)`
- `w(dist) = 1 / (1 + α * (dist / fruit_radius)^2)`
- `fruit_radius = 0.5 * ||fruit_extents||`

**Directions:** primarily **dirs6** = ±X, ±Y, ±Z in a fruit-local frame:
- `+X`: from fruit centroid toward stem centroid projected into the XY plane
- `+Z`: up axis (stem/world up)
- `+Y`: completes right-handed system

**Targets:** per direction `d` binned into **3 classes**:
- none / low / high  
(using thresholds like `none_thresh` and `high_thresh`, e.g. 0.015 and 0.15 in most runs)

**Scale & distribution:**
- ~1000 scenes, ~1.525M fruit–leaf pairs within `max_dist`, times directions.
- Extremely zero-heavy distribution (90%+ exactly 0).
- Main challenge: reliable “any vs none” detection (gate) under extreme class imbalance.

---

## 1) Initial neural approach (GNN backbone + occlusion MLP)
- Graph backbone: GraphSAGE / GINE on a kNN graph for node type + edge existence; occlusion head is an MLP using fruit-local relative geometry features.
- **Observed:** node type works well; edge existence meh; occlusion head often collapses to “none” or plateaus early.
- Multiple imbalance strategies tested (filtering none, class weights, focal loss, importance weighting); results remained unusable.
- **Sanity check:** tiny overfit test on 2 scenes with only occlusion head achieved ~85% F1 → pipeline not fundamentally broken; issue is generalization / distribution / feature-task mismatch.

**Conclusion:** plain MLP head is insufficient / unstable under the extreme class structure; need either (i) a more suitable model class for tabular geometry, or (ii) much richer geometric features.

---

## 2) Switch to Gradient-Boosted Decision Trees (LightGBM)
Motivation: strong tabular baseline, good with mixed features, faster iteration, easier diagnostics (feature importance), and stable optimization.

### 2.1 Single-stage 3-class LightGBM
- Trained multi-class LGBM on per-(fruit,leaf,dir) features.
- Inverse-frequency class weighting: minority recall improves, precision collapses (overpredicting non-none).
- Fixed weights (e.g. 1/5/10): accuracy and macro-F1 improved, but low/high recall dropped (too conservative).
- Threshold sweeps (post-processing) gave modest gains (PR tradeoff changes, no large breakthrough).

**Conclusion:** single-stage is limited by the “any vs none” boundary.

### 2.2 Two-stage LightGBM (recommended architecture)
Split into:
- **Model A:** binary `any vs none` (gate)
- **Model B:** `high vs low`, trained only on GT positives, optionally with `p_any` as extra feature

Inference combines:
- `P_none = 1 - p_any`
- `P_high = p_any * p_hi`
- `P_low  = p_any * (1 - p_hi)`

**Key observation:**
- Model **B** tends to be decent once given true positives (often F1 ~0.6+ on GT positives).
- Overall bottleneck is almost always **Model A** (false positives / missed positives).

---

## 3) Feature engineering progression (what we added over time)

Below, features are per **(fruit, leaf, direction)** row. We denote:
- fruit centroid `c_f ∈ R^3`, leaf centroid `c_l ∈ R^3`
- fruit axes matrix `A_f ∈ R^{3×3}` (columns are unit axes), extents `e_f = (e_x,e_y,e_z)` (full lengths)
- leaf axes `A_l`, extents `e_l`
- half-extents `h_f = 0.5 * e_f`, `h_l = 0.5 * e_l`
- direction-specific orthonormal basis `(d,u,v)` where `u ⟂ d`, `v ⟂ d`, `u ⟂ v`

### 3.1 “Near-sufficient scalar projection features” (early LGBM)
These encode the *relative configuration in the projection plane* using a few scalars.

**(a) Center offsets in the projection plane**
Project the leaf center into fruit coordinates (or subtract fruit center):
- `Δ = c_l - c_f`
- `o_u = Δ · u`
- `o_v = Δ · v`
Often we normalized by fruit radius:
- `fruit_radius = 0.5 * ||e_f||`
- `o_u_norm = o_u / (fruit_radius + eps)`
- `o_v_norm = o_v / (fruit_radius + eps)`

**(b) Depth / separation along direction (front-back cue)**
- `gap = Δ · d` (or a refined version using OBB intervals; see below)
- `gap_norm = gap / (fruit_radius + eps)`

Interpretation: positive gap means leaf lies “in front” of the fruit along `d`. (This mirrors the dataset’s front-check logic.)

**(c) Projected “radii” along u and v (size in the projection plane)**
For an OBB, the maximum extent along a unit axis `q` is:
- `r(q) = |q · A[:,0]| h0 + |q · A[:,1]| h1 + |q · A[:,2]| h2`

So we compute:
- fruit: `r_f_u = r_f(u)`, `r_f_v = r_f(v)`
- leaf:  `r_l_u = r_l(u)`, `r_l_v = r_l(v)`
Sometimes also the extent along `d`:
- `r_f_d = r_f(d)`, `r_l_d = r_l(d)`

We also included normalized leaf size terms:
- `leaf_u_norm = r_l_u / (fruit_radius + eps)`
- `leaf_v_norm = r_l_v / (fruit_radius + eps)`

**(d) Distance cue for weighted overlaps**
Because GT may be distance-weighted:
- `dist = ||c_l - c_f||`
- `dist_ratio = dist / (fruit_radius + eps)`

**(e) Direction indicator**
Include `dir_id` (categorical) or one-hot encoding so the model can learn direction-specific biases.

**Outcome:** Performance plateaued around macro-F1 ~0.47 and binary non-none F1 ~0.31 (after tuning). Many runs changed logloss but not task metrics → indicates feature ceiling.

---

### 3.2 Added fast rectangle-overlap proxy features (attempt to help Model A)
Goal: approximate overlap without full polygon intersection by treating projected shapes as axis-aligned rectangles in (u,v).

Given offsets and radii:
- overlap length along u:
  - `inter_u = max(0, (r_f_u + r_l_u) - |o_u| )`
- overlap length along v:
  - `inter_v = max(0, (r_f_v + r_l_v) - |o_v| )`

Then:
- `inter_area = inter_u * inter_v`
- `rect_iou = inter_area / (A_f + A_l - inter_area + eps)` where `A_f = 4 r_f_u r_f_v`, `A_l = 4 r_l_u r_l_v`
- `overlap_flag = 1[ inter_u > 0 and inter_v > 0 ]`
- `inter_norm = inter_area / (A_f + eps)` (a “coverage-like” proxy)

**Outcome:** only weakly separated none vs non-none (many GT positives had near-zero proxy overlap; many GT none had small positive proxy overlap). Metrics barely moved.

---

### 3.3 Breakthrough: projected OBB corner coordinates (u,v) features (32 dims)
Motivation: the previous scalars do not fully describe the 2D polygons. Two different OBB orientations can share the same center offsets and radii but yield different convex-hull overlap.

**Compute the 8 OBB corners in local coordinates**
For each sign triple `s ∈ {−1,+1}^3`:
- `corner = c + s0 h0 A[:,0] + s1 h1 A[:,1] + s2 h2 A[:,2]`

We do this for fruit and leaf (8 corners each).

**Project each corner into the (u,v) plane**
For each corner `p`, using relative coords to fruit center:
- `p_rel = p - c_f`
- `p_u = p_rel · u`
- `p_v = p_rel · v`

**Normalize**
- `p_u_norm = p_u / (fruit_radius + eps)`
- `p_v_norm = p_v / (fruit_radius + eps)`

Flatten in deterministic order:
- fruit: `[u0,v0, u1,v1, ..., u7,v7]`
- leaf:  `[u0,v0, ..., u7,v7]`
Total: **32 features** per direction.

**Outcome:** This gave Model A access to the effective 2D polygon geometry. Metrics improved substantially:
- macro-F1 increased to ~0.492–0.493 (best so far)
- binary non-none F1 increased to ~0.337 baseline, ~0.351 after threshold tuning
- confusion matrix showed meaningful gains in minority precision/recall
- threshold tuning began to have noticeable impact (improved non-none F1)

**Interpretation:** This is strong evidence the prior plateau was a feature ceiling: the model needed more information to approximate polygon overlap.

---

## 4) Why small metric gains (e.g. +0.02 macro-F1) can be *very* meaningful here
In this task, **macro-F1 is dominated by minority-class performance**, and small absolute increases can correspond to large changes in concrete errors.

Example (illustrative): If low/high precision is around 0.14 and improves to 0.24, that may look like “only +0.10”, but it can mean **tens of thousands fewer false positives** at validation scale. Because you evaluate millions of rows, even a 1–2% absolute shift in the “none → non-none” error rate corresponds to **many** corrected predictions.

Two practical reasons this makes gains look “numerically small”:
1) **Class 0 is already strong** (none precision/recall near ~0.95+), so macro-F1 is bottlenecked by low/high.
2) **The decision boundary is extremely narrow near 0**, so improvements come from shaving off a subset of hard errors (often a small fraction of the dataset) that still matter a lot operationally.

Thus, moving macro-F1 from ~0.47 to ~0.49 is not “noise”: it reflects a real and measurable reduction in the error modes that dominate minority performance.

---

## 5) “Realistic” significant-overlap thresholds experiment (10–30% / >30%)
Experiment: redefine labels:
- none if overlap < 0.1
- low if 0.1–0.3
- high if >0.3

Per-fruit prevalence stats indicated that most fruits have *some* high overlap somewhere among candidates, but per-row class counts remained extremely imbalanced, e.g.:
- none ~8.3M, low ~122k, high ~61k (~2% positives)

Empirically, even with large `any_pos_weight` caps, the gate still struggled to achieve non-none F1 comparable to the best small-threshold regime.

**Interpretation:** the per-row learning problem is still dominated by many irrelevant leaf candidates. A high per-fruit max-overlap does not imply that most (fruit,leaf,dir) rows are positive. Even in dense scenes, only a small subset of leaves per fruit-direction produce substantial overlap.

---

## 6) Current best and what it implies
**Best-performing learned setup so far:**
- Two-stage LGBM + scalar projection features + rectangle proxies + **corner-UV features** + threshold tuning
- macro-F1 ~0.493; tuned non-none F1 ~0.351
- precision≥0.5 remains achievable only at very low recall (~0.10), indicating persistent ambiguity / overlap near zero.

**Key takeaway:**
- For predicting OBB-projected overlap, providing the model the projected corner geometry brings it close to learning a proxy for the deterministic overlap computation.
- Remaining limitations likely reflect (i) intrinsic ambiguity near zero, and/or (ii) the per-row formulation with many negatives per fruit-direction.


