# Clustering (`cluster.py`)

## Primary path — Leiden community detection

`fit_clusters_leiden` builds a k-NN graph directly in the PCA/embedding space and
partitions it with Leiden. Every node gets a community, so there is structurally
**no giant catch-all cluster and no noise bucket**. UMAP is kept strictly for 2D
visualization (`embedding_2d`), never for clustering. This is the single-cell
RNA-seq playbook (Traag 2019 Leiden; Wolf 2018 SCANPY).

`fit_clusters_leiden` returns a **dict**, not a tuple:

```python
out = fit_clusters_leiden(X, resolution=1.0, n_pca_components=100, standardize=True)
# keys: labels, embedding_2d, scaler, pca, reducer_2d, clustering_space,
#       n_neighbors, resolution, metric, diagnostics
save_leiden('models/pipeline_phase1.pkl', out)
out = load_leiden('models/pipeline_phase1.pkl')

# Assign a new song by k-NN vote against the labeled corpus (same scale/PCA):
cluster_id, confidence = assign_cluster_knn(
    vec, out['clustering_space'], out['labels'], scaler=out['scaler'], pca=out['pca'])
```

- `resolution` is the one interpretable knob (low → few broad clusters, high → many fine). Sweep it with `resolution_sweep` and pick via the eval harness ([evaluation.md](evaluation.md)).
- `n_neighbors` auto-scales with corpus size (`n_neighbors_for_corpus`): ~10 for tiny corpora (Phase-2 n≈107), up to 30 for large Phase-1.
- `cluster_diagnostics` prints n_clusters / noise% / largest-share on every fit and warns if any cluster exceeds 60% of tracks (degenerate).
- The graph is built with one cosine metric, eliminating the old cosine-UMAP → euclidean-HDBSCAN metric mismatch.

## Legacy path — HDBSCAN-on-UMAP (`fit_clusters`)

Retained only for comparison. Returns 7 values
(`labels, embedding_2d, scaler, pca, reducer, clusterer, reducer_2d`);
`load_pipeline` returns 5. It clusters the UMAP projection, which is the wrong
space (see [invariants.md](invariants.md)) — **do not use it for new work.**
