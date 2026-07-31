# Evaluation (`eval.py`) — genre-free

`python -m anther_ml.eval --index models/index_phase{N}` prints a scorecard:
known-related-pairs self-retrieval (recall@1/5/10, MRR), and — with
`--labels models/labels_phase{N}.npy` — cluster size distribution + silhouette.
`--audit-out <path>` dumps a neighbor CSV/HTML for spot-listening.

This is the **arbiter** for every "compare/evaluate" design choice (standardize
on/off, resolution, all-layer vs last-layer, loudness on/off): run it before and
after each change.

## Current numbers

8 auto-detected related groups in the 107-track personal corpus:

- **Phase 2, rebuilt** (all-layer / multi-window / loudness): recall@1 = 0.625, recall@10 = 0.875, MRR = 0.666; 5 clusters, 0 noise, silhouette 0.202.
- **Phase 2, earlier** (single-layer, single-window): recall@1 = 0.875.

⚠️ The richer embedding **lowered** top-1 self-retrieval on this small corpus
(0.875 → 0.625), though recall@10 held. Per Workstream E ("keep the change only
if the eval improves"), this is an open **keep-or-revert decision** that needs
more data — embed fma_small and add a hand-labeled `related_pairs.json` — before
committing. Toggle `LAYER_AGG='last'` / `NORMALIZE=False` in notebook 04 to A/B it.

## Metrics (all label-free)

1. **Self-retrieval** (primary) — a track's alternate mixes/versions/stems should rank near the top of its neighbors.
2. **Cluster stability** — ARI between bootstrap-resampled refits.
3. **Internal quality** — silhouette on the clustering-space embedding + size distribution.
4. **Neighbor audit** — top-k neighbors per seed exported for human spot-listening.

Genre is **never** a metric — see [invariants.md](invariants.md).
