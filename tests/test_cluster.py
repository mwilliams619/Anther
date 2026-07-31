"""Workstream D — Leiden graph community detection + diagnostics."""
import numpy as np
import pytest

from anther_ml.cluster import (
    n_neighbors_for_corpus,
    build_knn_graph,
    leiden_labels,
    cluster_diagnostics,
    fit_clusters_leiden,
    resolution_sweep,
    assign_cluster_knn,
    save_leiden,
    load_leiden,
)


def _three_blobs(n_per=40, d=16, sep=8.0, seed=0):
    """Three well-separated Gaussian blobs — a corpus that *should* yield 3
    non-degenerate clusters."""
    rng = np.random.default_rng(seed)
    centers = np.array([[sep, 0], [-sep, 0], [0, sep]], dtype=np.float32)
    X, y = [], []
    for c, center in enumerate(centers):
        pts = rng.standard_normal((n_per, d)).astype(np.float32)
        pts[:, :2] += center
        X.append(pts)
        y += [c] * n_per
    return np.vstack(X), np.array(y)


def test_n_neighbors_scales_with_corpus():
    assert n_neighbors_for_corpus(100) == 10
    assert n_neighbors_for_corpus(1000) == 15
    assert n_neighbors_for_corpus(10000) == 20
    assert n_neighbors_for_corpus(50000) == 30
    assert n_neighbors_for_corpus(5) == 4  # capped below n


def test_build_knn_graph_shape():
    X, _ = _three_blobs()
    g = build_knn_graph(X, n_neighbors=10)
    assert g.vcount() == X.shape[0]
    assert g.ecount() > 0
    assert "weight" in g.es.attributes()


def test_leiden_recovers_separated_blobs():
    X, y = _three_blobs()
    labels = leiden_labels(X, resolution=1.0, n_neighbors=10)
    # Every node gets a community — no noise bucket (structural guarantee).
    assert (labels == -1).sum() == 0
    # Well-separated blobs → at least 3 clusters recovered.
    assert len(set(labels.tolist())) >= 3
    # Agreement with ground truth is high.
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(y, labels) > 0.9


def test_no_giant_catchall_cluster():
    """The specific failure mode D fixes: one mega-cluster holding >60%."""
    X, _ = _three_blobs()
    labels = leiden_labels(X, resolution=1.0, n_neighbors=10)
    diag = cluster_diagnostics(labels)
    assert diag["largest_cluster_frac"] < 0.60
    assert diag["noise_frac"] == 0.0


def test_resolution_knob_monotonic_ish():
    """Higher resolution → at least as many clusters (the one interpretable
    knob). We assert the extremes, not every step."""
    X, _ = _three_blobs()
    low = leiden_labels(X, resolution=0.1, n_neighbors=10)
    high = leiden_labels(X, resolution=3.0, n_neighbors=10)
    assert len(set(high.tolist())) >= len(set(low.tolist()))


def test_fit_clusters_leiden_end_to_end():
    X, y = _three_blobs()
    out = fit_clusters_leiden(X, resolution=1.0, n_pca_components=8)
    assert out["labels"].shape[0] == X.shape[0]
    assert out["clustering_space"].shape[0] == X.shape[0]
    assert out["embedding_2d"].shape == (X.shape[0], 2)
    assert out["scaler"] is not None
    assert out["diagnostics"]["n_clusters"] >= 3


def test_resolution_sweep():
    X, _ = _three_blobs()
    sweep = resolution_sweep(X, resolutions=(0.2, 1.0, 2.0), n_pca_components=8)
    assert len(sweep) == 3
    assert all("n_clusters" in s and "resolution" in s for s in sweep)


def test_assign_cluster_knn_matches_training():
    X, y = _three_blobs()
    out = fit_clusters_leiden(X, resolution=1.0, n_pca_components=8, compute_2d=False)
    labels = out["labels"]
    # A held-out point near blob 0's center should get blob 0's Leiden label.
    rng = np.random.default_rng(99)
    near0 = rng.standard_normal((1, X.shape[1])).astype(np.float32) * 0.1
    near0[0, :2] += [8.0, 0.0]
    cid, conf = assign_cluster_knn(
        near0[0], out["clustering_space"], labels,
        scaler=out["scaler"], pca=out["pca"], k=10,
    )
    # Whatever label blob-0 training points got, the new point should match it.
    blob0_label = np.bincount(labels[y == 0]).argmax()
    assert cid == blob0_label
    assert 0.0 < conf <= 1.0


def test_save_load_leiden_roundtrip(tmp_path):
    X, _ = _three_blobs()
    out = fit_clusters_leiden(X, resolution=1.0, n_pca_components=8, compute_2d=False)
    p = tmp_path / "leiden.pkl"
    save_leiden(p, out)
    loaded = load_leiden(p)
    np.testing.assert_array_equal(loaded["labels"], out["labels"])
    np.testing.assert_allclose(loaded["clustering_space"], out["clustering_space"])
    assert loaded["resolution"] == 1.0
