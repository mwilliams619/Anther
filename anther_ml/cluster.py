"""
Clustering pipeline for both Phase 1 (FMA pre-computed features)
and Phase 2 (MERT embeddings).

The same API works for both — the only difference is the input array shape
(568-dim for Phase 1, 1024-dim for Phase 2).
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

    return labels, embedding_2d, scaler, pca, reducer, clusterer


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


def save_pipeline(path: str | Path, scaler, reducer, clusterer, pca=None):
    """Persist the fit pipeline to disk."""
    with open(path, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "reducer": reducer, "clusterer": clusterer}, f)


def load_pipeline(path: str | Path) -> tuple:
    """Load a previously saved pipeline. Returns (scaler, pca, reducer, clusterer)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["scaler"], obj.get("pca"), obj["reducer"], obj["clusterer"]


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
