**Stem rule massively improves candidate coverage.**  
base_hit_rate ≈ 0.067 → stem_hit_rate ≈ 0.94.  
That’s exactly what we wanted: almost all GT edges are in the candidate set after stem augmentation.

**Leaf–stem and peduncle–stem coverage is essentially fixed.**  
Leaf–stem hit rate: 3.6% → 94.2%  
Peduncle–stem hit rate: 7.3% → 99.6%  
Fruit–peduncle is unchanged (as expected).

**Dedup looks consistent.**  
Added directed edges ≈ 2,886 → added undirected ≈ 1,435 (roughly half). That’s sane.

So the stem rule itself is not broken; it’s doing what it’s meant to do.

### Why training could still be bad (and what these results hint at)

**Candidate set explodes after stem rule.**  
Type‑filtered candidates go from ~79 to ~1,456 per scene.  
This means the edge classifier suddenly sees huge numbers of candidates.

**Stems become huge hubs.**  
Mean stem degree ≈ 357, vs non‑stem ≈ 2.  
That’s a very “star‑graph” structure; message passing might be dominated by stems, and edge existence becomes harder to learn cleanly.

**Type filter drops ~72% of candidate edges, but still leaves a large pool.**  
This may still be too many negatives if the classifier can’t separate them using current features.

### Post‑type‑filter balance is already high

After type‑filtering, your candidate set is already ~48% positives.  
stem_pos_rate_type ≈ 0.476 (median ~0.479).  
That is extremely high and explains why the usual imbalance tricks (pos_weight, neg_sampling) keep behaving weirdly.

**Implication:**  
The edge existence task, after type filter, is almost a balanced binary classification problem.  
Using large pos_weight or aggressive focal loss will distort training (and can tank recall or precision).

### Extra stem edges are not garbage

Stem rule “extra edges” are not garbage — they’re ~47% correct  
stem_extra_precision ≈ 0.467  
So the stem rule is not flooding the model with pure noise; almost half of those extra edges are true positives. That’s pretty good. It also means:

- The model is not overwhelmed by low‑quality edges.  
- If training is still bad, it’s not because the stem rule is “wrong.”

### Distance is not discriminative

stem_extra_dist_true and stem_extra_dist_false are almost identical (means/medians match).  
That means distance alone cannot separate true vs false stem‑rule edges — which is why a naive distance cutoff won’t fix precision/recall.

### Candidate explosion + hub dominance

Stem degree mean ~357 vs non‑stem ~2  
→ graph is extremely hub‑dominated.  
This can cause message passing to blur or overwhelm features, making edge existence harder than it looks from candidate stats alone.

## 7) What this suggests for next steps

Turn off pos_weight (and keep it off) when type‑filtering is on.  
You’re already near balanced. Weighted loss is likely hurting.

For focal, use alpha > 0.5 only if positives are rare; they are not.  
Focal is probably unnecessary here. If you try it, use alpha≈0.5 and small gamma (0–1).

Edge classifier may be under‑powered / poorly conditioned.  
Candidates are ~50/50, but the model still struggles. That points to representation quality rather than data imbalance.

Consider making edge head see more input:  
e.g., concatenate raw relative geometry (delta xyz, distance, maybe color diff) to h_src/h_dst before the edge head.

Consider limiting stem degree  
Cap extra stem edges per stem or per node (or per stem per scene). The hub‑heavy structure might be hurting the GNN backbone.

## 8) TL;DR

The diagnostics show:

✅ Stem rule works and massively boosts GT coverage  
✅ Extra edges are ~47% true — not noisy  
❌ Distance doesn’t separate true vs false  
❌ Post‑type‑filter pos rate is ~0.48 → imbalance tricks are unnecessary or harmful  
❌ Stem hubs are extremely high‑degree → likely hurting message passing

This makes the training issues less about candidate correctness and more about representation and graph structure.

If you want, a follow‑up diagnostic could:
- compute edge‑feature separability (e.g., simple logistic regression on raw geometry)
- test a “non‑GNN” baseline edge classifier to see if the features alone can solve it
