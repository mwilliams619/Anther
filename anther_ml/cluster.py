"""
Clustering pipeline for both Phase 1 (FMA pre-computed features)
and Phase 2 (MERT embeddings).

The same API works for both — the only difference is the input array shape
(518-dim for Phase 1, 1024-dim for Phase 2).

**Primary path: Leiden community detection on a k-NN graph** (Workstream D).
Borrowed from the single-cell RNA-seq playbook (Traag 2019; Wolf 2018), which
solves the identical problem — label-free grouping of noisy, high-dim
embeddings. The graph is built directly in the embedding/PCA space with a
single cosine metric, and Leiden partitions it; every node gets a community, so
there is structurally no giant catch-all cluster and no noise bucket. UMAP is
kept strictly for 2D visualization (``embedding_2d``), never for clustering —
2D projections distort density and inter-cluster distance.

``fit_clusters`` (HDBSCAN-on-UMAP) is retained only as an optional comparison
method; prefer ``fit_clusters_leiden``.
"""

import pickle
from pathlib import Path

import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def fit_clusters(
    features: np.ndarray,
    n_umap_components: int = 32,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    min_cluster_size: int = 100,
    cluster_selection_method: str = "eom",
    n_pca_components: int | None = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA | None, umap.UMAP, hdbscan.HDBSCAN]:
    """
    Full clustering pipeline:
      features (N, D) → scale → [PCA] → UMAP(32d) → HDBSCAN → labels

    The 2D visualization embedding is derived from the 32D clustering embedding
    so cluster boundaries align with the plot.

    Returns:
      labels        — cluster assignment per track (-1 = noise)
      embedding_2d  — 2D UMAP coordinates for visualization
      scaler        — fit StandardScaler
      pca           — fit PCA (or None if n_pca_components is None)
      reducer       — fit UMAP reducer (32D)
      clusterer     — fit HDBSCAN model
    """
    if len(features) < 10:
        raise ValueError(
            f"fit_clusters needs at least 10 tracks, got {len(features)}. "
            "Add more songs to data/audio/personal/ before clustering."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Optional PCA pre-step: removes noise dimensions and speeds up UMAP
    max_pca = min(X_scaled.shape[0], X_scaled.shape[1]) - 1
    effective_pca = min(n_pca_components, max_pca) if n_pca_components is not None else None
    if effective_pca is not None and effective_pca < X_scaled.shape[1] and effective_pca > 0:
        pca = PCA(n_components=effective_pca, random_state=random_state)
        X_pre = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_.sum()
        print(f'PCA: {X_scaled.shape[1]}D → {effective_pca}D  ({explained:.1%} variance retained)')
    else:
        pca = None
        X_pre = X_scaled

    # High-dim UMAP for clustering
    reducer = umap.UMAP(
        n_components=n_umap_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    X_reduced = reducer.fit_transform(X_pre)

    # 2D UMAP derived from the clustering embedding so the viz aligns with clusters
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.05,
        metric="euclidean",
        random_state=random_state,
    )
    embedding_2d = reducer_2d.fit_transform(X_reduced)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_reduced)

    return labels, embedding_2d, scaler, pca, reducer, clusterer, reducer_2d


def assign_cluster(
    feature_vec: np.ndarray,
    scaler: StandardScaler,
    reducer: umap.UMAP,
    clusterer: hdbscan.HDBSCAN,
    pca: PCA | None = None,
) -> tuple[int, float]:
    """
    Assign a new song (feature vector) to an existing cluster.
    Returns (cluster_id, membership_strength). cluster_id=-1 means noise.
    """
    X = scaler.transform(feature_vec.reshape(1, -1))
    if pca is not None:
        X = pca.transform(X)
    X_reduced = reducer.transform(X)
    labels, strengths = hdbscan.approximate_predict(clusterer, X_reduced)
    return int(labels[0]), float(strengths[0])


def save_pipeline(path: str | Path, scaler, reducer, clusterer, pca=None, reducer_2d=None):
    """Persist the fit pipeline to disk."""
    with open(path, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "reducer": reducer,
                     "clusterer": clusterer, "reducer_2d": reducer_2d}, f)


def load_pipeline(path: str | Path) -> tuple:
    """Load a previously saved pipeline. Returns (scaler, pca, reducer, clusterer, reducer_2d)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["scaler"], obj.get("pca"), obj["reducer"], obj["clusterer"], obj.get("reducer_2d")


# --------------------------------------------------------------------------- #
# Workstream D — Leiden community detection on a k-NN graph (primary path)
# --------------------------------------------------------------------------- #

def n_neighbors_for_corpus(n: int) -> int:
    """
    Scale ``n_neighbors`` with corpus size instead of hardcoding it (D3).
    Single-cell defaults are 10–30: smaller for tiny corpora (Phase-2, n≈107),
    larger for big ones (Phase-1). Always < n.
    """
    if n < 200:
        k = 10
    elif n < 2000:
        k = 15
    elif n < 20000:
        k = 20
    else:
        k = 30
    return max(2, min(k, n - 1))


def build_knn_graph(
    X: np.ndarray, n_neighbors: int, metric: str = "cosine"
):
    """
    Build a weighted, undirected k-nearest-neighbor graph in the embedding
    space. Edge weights are similarities (1 - distance), so stronger neighbors
    pull harder in Leiden. Returns an ``igraph.Graph``.
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    edges = {}
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):  # skip self
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a == b:
                continue
            # Cosine distance → similarity, clamped at 0: anti-correlated
            # neighbors (dist > 1) would give negative weights, which Leiden
            # rejects outright.
            w = max(0.0, 1.0 - float(dist))
            # Keep the strongest weight for a mutual/duplicate edge.
            if (a, b) not in edges or w > edges[(a, b)]:
                edges[(a, b)] = w

    g = ig.Graph(n=n)
    if edges:
        g.add_edges(list(edges.keys()))
        g.es["weight"] = list(edges.values())
    return g


def leiden_labels(
    X: np.ndarray,
    resolution: float = 1.0,
    n_neighbors: int | None = None,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Partition the k-NN graph of ``X`` with the Leiden algorithm and return a
    label per row. Leiden (not Louvain) guarantees well-connected communities.
    """
    import leidenalg as la

    n = X.shape[0]
    if n_neighbors is None:
        n_neighbors = n_neighbors_for_corpus(n)
    g = build_knn_graph(X, n_neighbors, metric=metric)

    weights = g.es["weight"] if g.ecount() else None
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=random_state,
    )
    return np.asarray(partition.membership, dtype=int)


def cluster_diagnostics(labels: np.ndarray, warn_threshold: float = 0.60) -> dict:
    """
    Always-printed diagnostics after a fit (D4): n_clusters, largest-cluster
    share, noise %, and a size histogram. Logs a WARNING when a single cluster
    holds more than ``warn_threshold`` of the tracks (degenerate run).
    """
    labels = np.asarray(labels)
    n = len(labels)
    noise = int(np.sum(labels == -1))
    clustered = labels[labels != -1]
    from collections import Counter
    sizes = Counter(clustered.tolist())
    largest = max(sizes.values()) if sizes else 0
    largest_frac = largest / n if n else 0.0

    diag = {
        "n_clusters": len(sizes),
        "n_tracks": n,
        "noise_frac": noise / n if n else 0.0,
        "largest_cluster_frac": largest_frac,
        "sizes": dict(sorted(sizes.items())),
    }
    print(f"clusters={diag['n_clusters']}  "
          f"noise={diag['noise_frac']:.1%}  "
          f"largest={largest_frac:.1%}  "
          f"sizes={sorted(sizes.values(), reverse=True)[:10]}")
    if largest_frac > warn_threshold:
        print(f"  WARNING: largest cluster is {largest_frac:.1%} of tracks "
              f"(> {warn_threshold:.0%}) — likely degenerate")
    return diag


def fit_clusters_leiden(
    features: np.ndarray,
    resolution: float = 1.0,
    n_neighbors: int | None = None,
    n_pca_components: int | None = 100,
    standardize: bool = True,
    pca_whiten: bool = False,
    metric: str = "cosine",
    random_state: int = 42,
    compute_2d: bool = True,
) -> dict:
    """
    Primary clustering pipeline (Workstream D):
      features → [scale] → [PCA] → k-NN graph → Leiden → labels
      (+ 2D UMAP of the SAME embedding, for visualization only)

    Returns a dict with: labels, embedding_2d (or None), scaler, pca,
    reducer_2d, clustering_space (the array Leiden ran on — reused for
    assign_cluster_knn and eval), plus the fit params and diagnostics.

    The k-NN graph is built on the clustering-space embedding, NOT on any UMAP
    projection (D2), eliminating the old cosine-UMAP → euclidean-HDBSCAN metric
    mismatch entirely.
    """
    if len(features) < 10:
        raise ValueError(
            f"fit_clusters_leiden needs at least 10 tracks, got {len(features)}."
        )

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
    else:
        scaler = None
        X = np.asarray(features, dtype=np.float32)

    max_pca = min(X.shape[0], X.shape[1]) - 1
    eff_pca = min(n_pca_components, max_pca) if n_pca_components else None
    if eff_pca and 0 < eff_pca < X.shape[1]:
        pca = PCA(n_components=eff_pca, whiten=pca_whiten, random_state=random_state)
        X = pca.fit_transform(X)
        print(f"PCA → {eff_pca}D ({pca.explained_variance_ratio_.sum():.1%} var)")
    else:
        pca = None

    if n_neighbors is None:
        n_neighbors = n_neighbors_for_corpus(X.shape[0])
    print(f"Leiden: n_neighbors={n_neighbors}, resolution={resolution}, "
          f"metric={metric}")

    labels = leiden_labels(
        X, resolution=resolution, n_neighbors=n_neighbors,
        metric=metric, random_state=random_state,
    )
    diagnostics = cluster_diagnostics(labels)

    embedding_2d = None
    reducer_2d = None
    if compute_2d:
        reducer_2d = umap.UMAP(
            n_components=2, n_neighbors=min(50, X.shape[0] - 1),
            min_dist=0.05, metric="euclidean", random_state=random_state,
        )
        embedding_2d = reducer_2d.fit_transform(X)

    return {
        "labels": labels,
        "embedding_2d": embedding_2d,
        "scaler": scaler,
        "pca": pca,
        "reducer_2d": reducer_2d,
        "clustering_space": X,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "metric": metric,
        "diagnostics": diagnostics,
    }


def resolution_sweep(
    features: np.ndarray,
    resolutions=(0.2, 0.5, 1.0, 1.5, 2.0),
    **fit_kwargs,
) -> list[dict]:
    """
    Fit Leiden at each resolution and report the size distribution, so the knob
    can be chosen with the Workstream F metrics rather than by guessing. Low
    resolution → few broad clusters, high → many fine ones. Returns one summary
    dict per resolution (skips the 2D UMAP for speed).
    """
    fit_kwargs.setdefault("compute_2d", False)
    results = []
    for res in resolutions:
        out = fit_clusters_leiden(features, resolution=res, **fit_kwargs)
        results.append({"resolution": res, **out["diagnostics"]})
    return results


def assign_cluster_knn(
    vec: np.ndarray,
    clustering_space: np.ndarray,
    labels: np.ndarray,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
    k: int = 15,
    metric: str = "cosine",
) -> tuple[int, float]:
    """
    Assign a new song to a Leiden cluster by majority vote of its k nearest
    labeled neighbors (graph partitioning has no native ``transform``). The
    query is put through the SAME scale/PCA transform as the corpus.

    Returns (cluster_id, confidence = winning_votes / k).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    assert X.shape[1] == clustering_space.shape[1], (
        f"query dim {X.shape[1]} != corpus dim {clustering_space.shape[1]}"
    )

    k = min(k, clustering_space.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(clustering_space)
    _, idx = nn.kneighbors(X)
    neighbor_labels = np.asarray(labels)[idx[0]]
    from collections import Counter
    winner, votes = Counter(neighbor_labels.tolist()).most_common(1)[0]
    return int(winner), votes / k


def save_leiden(path: str | Path, result: dict) -> None:
    """
    Persist a ``fit_clusters_leiden`` result. Stores everything needed to assign
    a new song later (labels, scaler, pca, clustering_space, reducer_2d, params)
    but drops the fitted UMAP-2D transform's ability to project new points — 2D
    is viz-only, so ``reducer_2d`` is kept solely to re-plot the same picture.
    """
    keep = {
        "labels": result["labels"],
        "scaler": result["scaler"],
        "pca": result["pca"],
        "reducer_2d": result.get("reducer_2d"),
        "clustering_space": result["clustering_space"],
        "n_neighbors": result["n_neighbors"],
        "resolution": result["resolution"],
        "metric": result["metric"],
        "diagnostics": result.get("diagnostics"),
    }
    with open(path, "wb") as f:
        pickle.dump(keep, f)


def load_leiden(path: str | Path) -> dict:
    """Load a saved Leiden pipeline dict (see ``save_leiden``)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def cluster_summary(labels: np.ndarray, genre_labels: list[str] | None = None) -> dict:
    """
    Print a summary table: cluster_id → size, dominant genre (if provided), noise %.
    Returns a dict of {cluster_id: {'size': int, 'top_genre': str}}.
    """
    from collections import Counter
    unique = sorted(set(labels))
    summary = {}
    for cid in unique:
        mask = labels == cid
        size = mask.sum()
        top_genre = None
        if genre_labels is not None:
            genres = [genre_labels[i] for i, m in enumerate(mask) if m]
            top_genre = Counter(genres).most_common(1)[0][0]
        summary[cid] = {"size": int(size), "top_genre": top_genre}
    return summary
