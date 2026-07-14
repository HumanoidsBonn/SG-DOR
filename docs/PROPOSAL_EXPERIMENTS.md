# Object Proposal Discovery: Experimental Iterations (v10 - v11.2)

This document tracks the iterative debugging and optimization of the Evidential Object Proposal framework.

## The Objective
To solve the "Zero Recall" problem in sparse 3D point clouds by substituting standard Binary Cross-Entropy with a **Beta-Bernoulli Evidential Head** that explicitly models Vacuous Uncertainty.

---

## Experiment Log

| Version | Strategy | Result | Key Insights |
| :--- | :--- | :--- | :--- |
| **v10.0** | Initial Evidential Head | **FAILED** | "Infinite Beta Collapse": The model optimized for maximum vacuity, crushing Alpha/Beta to 1.0. |
| **v10.1** | Neutral Prior + max=10 Clamp | **STABLE** | Recall > 0 achieved, but "Bayesian Rigidity" (background dominated the gradient). |
| **v10.3** | Pro-Object Bias (Alpha > Beta) | **IMPROVED** | High initial recall, but background eventually overwhelmed the model. |
| **v10.4** | Hard Negative Mining (EHNM) | **NOISE** | Aggressive negative mining created "Gradient Whiplash" on a single shared head. |
| **v10.6** | **Decoupled Heads** | **BREAKTHROUGH** | Splitting Discovery (Alpha/Beta) from Localization (Center/Extent) stabilized the gradients. |
| **v11.0** | **Certainty Ceiling (max=100)** | **STABLE** | Replacing Softmax with Softplus + 100.0 clamp allowed for "Hyper-Certainty" in valid centers. |
| **v11.1** | **Contrastive Disbelief (10x)** | **OPTIMAL** | Forced differentiation between object centers and surface points using a massive negative weight. |
| **v11.2** | **Precision Audit** | **VERIFIED** | Tightened thresholds ($P>0.6, U<0.3$) to guarantee high-quality proposal generation. |
| **v11.3** | **Sparse Refinement** | **DEPLOYED** | v2: Tightened to $U<0.2$, fixed validation bugs, and added "AvgRef" shift metrics. |

---

## Core Infrastructure Changes

### 1. The Decoupled Head (`proposal_module.py`)
Previously shared layers caused localization errors to corrupt objectness predictions. Now, separate MLP branches handle Discovery and Geometry independently.

### 2. Certainty-Weighted Ranking (`evidential_nms.py`)
Introduced a new ranking metric for NMS:
$$\text{Ranking Score} = P \times (1 - U)$$
This prioritizes points that are both highly likely to be an object **and** have low vacuity (uncertainty).

### 3. Contrastive Disbelief Push (`proposal_loss.py`)
To prevent the model from getting stuck in a "safe" uncertain state, we implemented a 25:10 (Positive to Negative) weighting scheme. This forces the model to actively "disbelieve" surface noise rather than just ignoring it.

---

## Running v11.3 Sparse Zoom (Refinement)
```bash
python pepper_scene_graphs/training/train_proposals.py \
  --batch_size 64 \
  --epochs 100 \
  --exp_name votenet_multiclass_v11_3_zoom
```

🚣🚀⚖️📊⚖️📊⚖️📊⚖️📊⚖️📊⚖️📊
