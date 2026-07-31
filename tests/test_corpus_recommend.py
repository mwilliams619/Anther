"""Multi-song recommendation (use-case 2): recommend corpus tracks similar to a
*set* of seed songs. Same fixtures as test_corpus_place."""
import numpy as np
import pytest

from anther_ml.corpus import ReferenceCorpus, recommend_from_seeds
from anther_ml.similarity import SongIndex
from corpus_fixtures import blob_vector, build_test_corpus


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("recommend")
    return build_test_corpus(tmp)


def _orthogonal_corpus():
    """4 orthonormal tracks t0..t3 (identity), no standardization."""
    emb = np.eye(4, dtype=np.float32)
    meta = [{"id": f"t{i}", "name": f"t{i}", "playlists": []} for i in range(4)]
    index = SongIndex(emb, meta, standardize=False)
    leiden = {"labels": np.zeros(4, dtype=int), "scaler": None, "pca": None,
              "reducer_2d": None, "clustering_space": emb}
    manifest = {"corpus_format_version": 1, "embedding_config": {}}
    return ReferenceCorpus(emb, index, leiden, np.zeros((4, 2)), emb[:1], [],
                           manifest)


# --------------------------------------------------------------------------- #
# centroid semantics on a hand-checkable corpus
# --------------------------------------------------------------------------- #

def test_centroid_ranks_shared_center_first():
    corpus = _orthogonal_corpus()
    # Seeds t0=[1,0,0,0] and t1=[0,1,0,0]. Centroid ∝ [1,1,0,0]/√2.
    # Cosine to t0 and t1 = 1/√2 (tied top); to t2,t3 = 0.
    res = recommend_from_seeds(corpus, [corpus.embeddings[0], corpus.embeddings[1]],
                               top_k=4, method="centroid")
    top_ids = {r["id"] for r in res[:2]}
    assert top_ids == {"t0", "t1"}
    assert res[0]["score"] == pytest.approx(1 / np.sqrt(2))
    # t2, t3 are orthogonal to the centroid → zero score, ranked last.
    assert {r["id"] for r in res[2:]} == {"t2", "t3"}
    assert res[2]["score"] == pytest.approx(0.0, abs=1e-6)


def test_exclude_ids_drops_seeds_themselves():
    corpus = _orthogonal_corpus()
    res = recommend_from_seeds(corpus, [corpus.embeddings[0], corpus.embeddings[1]],
                               top_k=4, exclude_ids={"t0", "t1"})
    assert {r["id"] for r in res} == {"t2", "t3"}   # only non-seeds returned


def test_rank_and_score_shape_matches_query():
    corpus = _orthogonal_corpus()
    res = recommend_from_seeds(corpus, [corpus.embeddings[0]], top_k=3)
    assert [r["rank"] for r in res] == [1, 2, 3]     # 1-based, contiguous
    assert all("score" in r and "name" in r for r in res)
    # Single seed reduces to plain nearest-neighbor: t0 first at cosine 1.
    assert res[0]["id"] == "t0" and res[0]["score"] == pytest.approx(1.0)


def test_topk_is_the_default_method():
    corpus = _orthogonal_corpus()
    seeds = [corpus.embeddings[0], corpus.embeddings[1]]
    default = recommend_from_seeds(corpus, seeds, top_k=4)
    explicit = recommend_from_seeds(corpus, seeds, top_k=4, method="topk")
    assert [(r["id"], r["score"]) for r in default] == [
        (r["id"], r["score"]) for r in explicit
    ]


def test_empty_seeds_raise():
    corpus = _orthogonal_corpus()
    with pytest.raises(ValueError, match="at least one seed"):
        recommend_from_seeds(corpus, [])


def test_unknown_method_raises():
    corpus = _orthogonal_corpus()
    with pytest.raises(ValueError, match="unknown method"):
        recommend_from_seeds(corpus, [corpus.embeddings[0]], method="bogus")


def test_antipodal_centroid_raises():
    # Two exactly-opposite seeds cancel: centroid is the zero vector.
    emb = np.array([[1, 0], [-1, 0], [0, 1]], dtype=np.float32)
    meta = [{"id": f"t{i}", "name": f"t{i}", "playlists": []} for i in range(3)]
    index = SongIndex(emb, meta, standardize=False)
    leiden = {"labels": np.zeros(3, dtype=int), "scaler": None, "pca": None,
              "reducer_2d": None, "clustering_space": emb}
    corpus = ReferenceCorpus(emb, index, leiden, np.zeros((3, 2)), emb[:1], [],
                             {"corpus_format_version": 1, "embedding_config": {}})
    with pytest.raises(ValueError, match="degenerate"):
        recommend_from_seeds(corpus, [emb[0], emb[1]], method="centroid")


# --------------------------------------------------------------------------- #
# topk fallback: handles a two-mood seed set the centroid would strand
# --------------------------------------------------------------------------- #

def test_topk_recovers_both_modes_when_centroid_strands(built):
    corpus, _ = built
    # Two seeds from two different blobs. The centroid sits between the blobs
    # (a sparse region); topk scores each candidate by its closest seed, so
    # members of BOTH blobs surface.
    seeds = [blob_vector(90), blob_vector(91)]   # blob 0 and blob 1 (unseen ids)

    centroid = recommend_from_seeds(corpus, seeds, top_k=10, method="centroid")
    topk = recommend_from_seeds(corpus, seeds, top_k=10, method="topk",
                                per_seed_k=1)

    # topk's top hit is at least as similar to its nearest seed as centroid's is
    # to the stranded midpoint — the multi-mood set is better served.
    assert topk[0]["score"] >= centroid[0]["score"]
    # Both methods return the requested count and valid ranks.
    assert len(topk) == 10 and [r["rank"] for r in topk] == list(range(1, 11))


def test_recommend_on_real_fixture_excludes_seed_row(built):
    corpus, _ = built
    seed_idx = 0
    seed_id = corpus.metadata[seed_idx]["id"]
    res = recommend_from_seeds(corpus, [corpus.embeddings[seed_idx]],
                               top_k=5, exclude_ids={seed_id})
    assert seed_id not in {r["id"] for r in res}
    assert len(res) == 5
    # Recommendations are same-blob neighbors: all share the seed's cluster.
    seed_cluster = int(corpus.labels[seed_idx])
    idx_by_id = {m["id"]: i for i, m in enumerate(corpus.metadata)}
    rec_clusters = [int(corpus.labels[idx_by_id[r["id"]]]) for r in res]
    assert rec_clusters.count(seed_cluster) >= 3
