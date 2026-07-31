"""Placement regime: neighbor/cluster/2D placement and calibrated playlist fit."""
import numpy as np
import pytest

from anther_ml.corpus import ReferenceCorpus, calibrate_fit, place, playlist_fit, rank_playlists
from anther_ml.similarity import SongIndex
from corpus_fixtures import blob_vector, build_test_corpus


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("place")
    return build_test_corpus(tmp)


def test_place_corpus_row_retrieves_itself(built):
    corpus, _ = built
    vec = corpus.embeddings[0]  # raw pre-transform vector, as a query would be
    result = place(corpus, vec, top_k=3, knn_k=5)

    assert result["neighbors"][0]["name"] == corpus.metadata[0]["name"]
    assert result["neighbors"][0]["score"] > 0.999
    assert result["cluster"]["id"] == int(corpus.labels[0])
    assert result["cluster"]["confidence"] >= 0.6
    assert result["cluster"]["profile"]["cluster_id"] == int(corpus.labels[0])
    assert len(result["coords_2d"]) == 2
    assert np.isfinite(result["coords_2d"]).all()


def test_place_rejects_wrong_dimension(built):
    corpus, _ = built
    with pytest.raises(ValueError, match="dim"):
        place(corpus, np.ones(7, dtype=np.float32))


# --------------------------------------------------------------------------- #
# playlist fit
# --------------------------------------------------------------------------- #

def _micro_corpus():
    """4 orthogonal tracks; tracks 0 and 1 are on playlist 'pl'."""
    emb = np.eye(4, dtype=np.float32)
    meta = [
        {"id": f"t{i}", "name": f"t{i}", "playlists": [{"pid": 1, "name": "pl"}] if i < 2 else []}
        for i in range(4)
    ]
    index = SongIndex(emb, meta, standardize=False)
    leiden = {
        "labels": np.zeros(4, dtype=int),
        "scaler": None,
        "pca": None,
        "reducer_2d": None,
        "clustering_space": emb,
    }
    manifest = {"corpus_format_version": 1, "embedding_config": {}}
    return ReferenceCorpus(
        emb, index, leiden, np.zeros((4, 2)), emb[:1], [], manifest
    )


def test_playlist_fit_matches_hand_computation():
    corpus = _micro_corpus()
    # Query [1,1,0,0]/√2: cosine 1/√2 to each member → top-2 mean = 1/√2.
    vec = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert playlist_fit(corpus, vec, "pl", k=2) == pytest.approx(1 / np.sqrt(2))
    # k=1 takes only the best member.
    vec2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert playlist_fit(corpus, vec2, "pl", k=1) == pytest.approx(1.0)


def test_playlist_fit_unknown_playlist_raises():
    corpus = _micro_corpus()
    with pytest.raises(ValueError, match="no tracks"):
        playlist_fit(corpus, np.ones(4, dtype=np.float32), "nope")


def test_calibrated_fit_separates_blobs(built):
    corpus, _ = built
    query_blob0 = blob_vector(90)  # blob 0 (index 90 unseen: corpus used 0-29)
    query_blob1 = blob_vector(91)  # blob 1

    fit_match = calibrate_fit(corpus, query_blob0, "pl_a")
    fit_off = calibrate_fit(corpus, query_blob1, "pl_a")

    # A blob-0 song fits the blob-0 playlist better than nearly all null tracks.
    assert fit_match["percentile"] > 90
    assert fit_match["raw_fit"] > fit_off["raw_fit"]
    assert fit_match["percentile"] > fit_off["percentile"]

    # Null distribution excludes the playlist's own members.
    assert fit_match["n_members"] == 10
    assert fit_match["n_null"] == corpus.n_tracks - 10


def test_rank_playlists_puts_matching_playlist_first(built):
    corpus, _ = built
    ranked_0 = rank_playlists(corpus, blob_vector(90))
    ranked_1 = rank_playlists(corpus, blob_vector(91))
    assert ranked_0[0]["name"] == "pl_a"
    assert ranked_1[0]["name"] == "pl_b"
    # Every corpus playlist scored.
    assert {r["name"] for r in ranked_0} == {"pl_a", "pl_b"}
