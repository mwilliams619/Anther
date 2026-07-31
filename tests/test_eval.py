"""Workstream F — genre-free evaluation harness."""
import json

import numpy as np
import pytest

from anther_ml import eval as ev
from anther_ml.similarity import SongIndex


def test_title_stem_collapses_alternates():
    assert ev.title_stem("Mic Check") == ev.title_stem("Mic Check vocals")
    assert ev.title_stem("Mic Check vocals_1") == ev.title_stem("Mic Check")
    assert ev.title_stem("hang v1") == ev.title_stem("hang v1-1")
    assert ev.title_stem("colture rough mix 1") == ev.title_stem("colture rough mix 2")
    # Genuinely different songs must not collapse.
    assert ev.title_stem("Blue Ocean") != ev.title_stem("Red Desert")


def test_auto_related_groups():
    names = ["Mic Check", "Mic Check vocals", "Solo Track", "hang v1", "hang v1-1"]
    groups = ev.auto_related_groups(names)
    # Two groups: the Mic Check pair and the hang pair; Solo Track is alone.
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [2, 2]
    assert all(len(v) > 1 for v in groups.values())


def test_load_related_groups_from_json(tmp_path):
    names = ["Mic Check", "Mic Check vocals", "Solo"]
    p = tmp_path / "related.json"
    p.write_text(json.dumps({"mic": ["Mic Check", "Mic Check vocals"]}))
    groups = ev.load_related_groups(p, names)
    assert list(groups) == ["mic"]
    assert set(groups["mic"]) == {0, 1}


def _grouped_embeddings():
    """
    6 tracks in 3 related pairs. Each pair sits close together (small offset),
    pairs are far apart — so a correct retriever ranks a track's partner #1.
    """
    rng = np.random.default_rng(0)
    centers = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    emb = np.repeat(centers, 2, axis=0) + 0.01 * rng.standard_normal((6, 3)).astype(np.float32)
    names = ["A v1", "A v2", "B v1", "B v2", "C v1", "C v2"]
    return emb, names


def test_self_retrieval_perfect_when_partners_are_nearest():
    emb, names = _grouped_embeddings()
    idx = SongIndex(emb, [{"name": n} for n in names], standardize=False)
    groups = ev.auto_related_groups(names)
    assert len(groups) == 3
    res = ev.self_retrieval(idx.embeddings, groups, ks=(1, 5))
    assert res["n_queries"] == 6
    assert res["recall@1"] == 1.0
    assert res["mrr"] == 1.0


def test_self_retrieval_recall_below_one_when_scrambled():
    """If embeddings carry no group structure, recall@1 should be poor."""
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((6, 8)).astype(np.float32)
    names = ["A v1", "A v2", "B v1", "B v2", "C v1", "C v2"]
    idx = SongIndex(emb, [{"name": n} for n in names], standardize=False)
    groups = ev.auto_related_groups(names)
    res = ev.self_retrieval(idx.embeddings, groups, ks=(1,))
    assert res["recall@1"] < 1.0


def test_size_distribution():
    labels = np.array([0, 0, 0, 1, 1, -1])
    sd = ev.size_distribution(labels)
    assert sd["n_clusters"] == 2
    assert sd["noise_frac"] == pytest.approx(1 / 6)
    assert sd["largest_cluster_frac"] == pytest.approx(3 / 6)
    assert sd["sizes"] == {0: 3, 1: 2}


def test_silhouette_and_stability_with_kmeans():
    from sklearn.cluster import KMeans

    emb, _ = _grouped_embeddings()
    idx = SongIndex(emb, [{"name": str(i)} for i in range(6)], standardize=False)

    def cluster_fn(X):
        return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)

    labels = cluster_fn(idx.embeddings)
    sil = ev.silhouette(idx.embeddings, labels)
    assert -1.0 <= sil <= 1.0
    stab = ev.cluster_stability(idx.embeddings, cluster_fn, frac=0.8, n_seeds=3)
    assert 0.0 <= stab["mean_ari"] <= 1.0
    assert stab["n_seeds"] == 3


def test_neighbor_audit_export(tmp_path):
    emb, names = _grouped_embeddings()
    idx = SongIndex(emb, [{"name": n} for n in names], standardize=False)
    rows = ev.neighbor_audit(idx.embeddings, names, seeds=[0], k=3)
    assert len(rows) == 3
    assert rows[0]["seed"] == "A v1"
    assert rows[0]["neighbor"] == "A v2"  # partner is nearest
    out = tmp_path / "audit"
    ev.write_audit(rows, out)
    assert (out.with_suffix(".csv")).exists()
    assert (out.with_suffix(".html")).exists()


def test_build_scorecard_end_to_end():
    emb, names = _grouped_embeddings()
    idx = SongIndex(emb, [{"name": n} for n in names], standardize=False,
                    config={"phase": 2})
    card = ev.build_scorecard(idx)
    assert card["n_tracks"] == 6
    assert card["n_related_groups"] == 3
    assert card["self_retrieval"]["recall@1"] == 1.0


def test_cli_main(tmp_path, capsys):
    emb, names = _grouped_embeddings()
    idx = SongIndex(emb, [{"name": n} for n in names], standardize=False)
    path = tmp_path / "idx"
    idx.save(path)
    card = ev.main(["--index", str(path), "--json"])
    assert card["self_retrieval"]["recall@1"] == 1.0
    out = capsys.readouterr().out
    assert "self_retrieval" in out
