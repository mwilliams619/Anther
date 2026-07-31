"""
Per-track micro-genre tagging (anther_ml/corpus/tagging/).

Covers vocab normalization + longest-match-wins, Stage-A seed labels
(multi-hot, support, confidence monotonicity, catalog-name discount),
TagProbe fit/predict/save/load with the learnable floor and leak-free
artist split, the FMA crosswalk (unmapped dropped, never guessed), the
evaluate wall-off from anther_ml.eval, and tags in place() with and
without tag artifacts present.
"""

import inspect
import json
import re

import numpy as np
import pytest
from scipy import sparse

from anther_ml.corpus.tagging.crosswalk import (
    build_crosswalk,
    crosswalk_labels,
)
from anther_ml.corpus.tagging.probe import (
    TagProbe,
    group_split,
    knn_smooth,
    predict_tags,
)
from anther_ml.corpus.tagging.vocab import build_vocab_map, norm
from anther_ml.corpus.tagging.weak_labels import (
    build_seed_labels,
    genre_support,
    match_name,
)

from .corpus_fixtures import build_test_corpus, blob_vector

VOCAB = ["pop", "k-pop", "death metal", "hip hop", "r&b", "drum and bass",
         "soul", "funk", "jazz"]
VMAP = build_vocab_map(VOCAB)


# -- vocab / normalization ----------------------------------------------------


def test_norm_folds_separators_case_emoji():
    assert norm("Hip-Hop") == "hip hop"
    assert norm("  DEATH/metal ") == "death metal"
    assert norm("🔥 R&B 🔥") == "r&b"


def test_alias_folding():
    assert match_name("dnb", VMAP) == {"drum and bass": "exact"}
    assert match_name("HipHop", VMAP) == {"hip hop": "exact"}
    assert match_name("R n B", VMAP) == {"r&b": "exact"}


def test_longest_vocab_match_wins():
    # "k-pop hits" must seed k-pop, and the freed "pop" token must NOT match.
    assert match_name("K-Pop Hits", VMAP) == {"k-pop": "substring"}
    assert match_name("90s death metal", VMAP) == {"death metal": "substring"}


def test_no_match_returns_empty():
    assert match_name("roadtrip 2019", VMAP) == {}
    assert match_name("", VMAP) == {}


def test_exact_only_genres_never_match_as_substring():
    vmap = build_vocab_map(VOCAB + ["sound", "sleep"])
    assert match_name("Sound of Summer", vmap) == {}
    assert match_name("songs for deep sleep vibes", vmap) == {}
    # …but a playlist literally named that IS the genre
    assert match_name("sleep", vmap) == {"sleep": "exact"}


# -- Stage A seed labels --------------------------------------------------------


def _track(names):
    return {"playlists": [{"pid": None, "name": n} for n in names]}


def test_build_seed_labels_multihot_and_support():
    metadata = [
        _track(["death metal"]),                  # exact
        _track(["90s death metal", "k-pop hits"]),  # two substrings
        _track(["roadtrip"]),                     # nothing
        _track([]),
    ]
    W, genres, conf, tiers = build_seed_labels(metadata, VOCAB)
    Y = (W > 0).toarray()
    gi = {g: j for j, g in enumerate(genres)}
    assert Y[0, gi["death metal"]] and not Y[0, gi["pop"]]
    assert Y[1, gi["death metal"]] and Y[1, gi["k-pop"]]
    assert not Y[2].any() and not Y[3].any()
    support = genre_support(W, genres)
    assert support["death metal"] == 2 and support["k-pop"] == 1
    assert tiers["exact"] == 1 and tiers["substring"] == 2


def test_confidence_monotone_in_matched_playlists():
    one = [_track(["death metal"])]
    two = [_track(["death metal", "metalcore & death metal"])]
    _, _, c1, _ = build_seed_labels(one, VOCAB)
    _, _, c2, _ = build_seed_labels(two, VOCAB)
    assert c2[0] > c1[0]


def test_catalog_name_discount():
    # A single name matching >=3 genres is a catalog playlist: each genre
    # gets tier_weight/n, so it alone can't clear the confident threshold.
    W, genres, conf, _ = build_seed_labels(
        [_track(["Soul, Funk, Jazz and more"])], VOCAB
    )
    gi = {g: j for j, g in enumerate(genres)}
    w_soul = W[0, gi["soul"]]
    assert w_soul == pytest.approx(0.6 / 3)
    assert conf[0] == pytest.approx(3 * 0.6 / 3)


# -- TagProbe --------------------------------------------------------------------


def _blob_training_set(n=240, dim=16, seed=7):
    """Two separable blobs + one starved genre; artists split across blobs."""
    rng = np.random.default_rng(seed)
    centers = np.stack([np.ones(dim) * 4, -np.ones(dim) * 4])
    blob = np.arange(n) % 2
    X = centers[blob] + rng.normal(size=(n, dim))
    rows, cols = [], []
    for i in range(n):
        rows.append(i)
        cols.append(blob[i])  # genre 0 for blob 0, genre 1 for blob 1
    for i in range(3):  # starved genre 2: 3 seeds only
        rows.append(i)
        cols.append(2)
    W = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, 3)
    )
    genres = ["blob zero core", "blob one wave", "rare genre"]
    artists = [f"artist {i % 24}" for i in range(n)]
    return X.astype(np.float32), W, genres, artists


def test_probe_fit_predict_roundtrip(tmp_path):
    X, W, genres, artists = _blob_training_set()
    probe = TagProbe.fit(X, W, genres, artists, min_support=10)
    # starved genre never gets a head
    assert set(probe.genre_names) == {"blob zero core", "blob one wave"}
    assert probe.fit_report["val_macro_f1"] > 0.9  # blobs are separable

    out = predict_tags(probe, X[:2])
    assert out[0]["tags"][0]["genre"] == "blob zero core"
    assert out[0]["tags"][0]["primary"] is True
    assert out[1]["tags"][0]["genre"] == "blob one wave"

    probe.save(tmp_path / "probe.pkl")
    again = TagProbe.load(tmp_path / "probe.pkl")
    np.testing.assert_allclose(
        probe.predict_proba(X[:5]), again.predict_proba(X[:5])
    )
    assert again.genre_names == probe.genre_names


def test_seed_fallback_for_unlearnable_genres():
    X, W, genres, artists = _blob_training_set()
    probe = TagProbe.fit(X, W, genres, artists, min_support=10)
    out = predict_tags(
        probe, X[:1], seed_weights=W[:1], full_genre_names=genres
    )
    sources = {t["genre"]: t["source"] for t in out[0]["tags"]}
    assert sources.get("rare genre") == "seed"  # row 0 carries the rare seed


def test_group_split_no_artist_leak():
    groups = [f"artist {i % 10}" for i in range(100)]
    train, val = group_split(groups, val_frac=0.3, seed=1)
    assert not (train & val).any() and (train | val).all()
    leaked = {g for g, t in zip(groups, train) if t} & {
        g for g, v in zip(groups, val) if v
    }
    assert leaked == set()


def test_knn_smooth_pulls_stray_toward_neighbors():
    probs = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    X = np.ones((4, 8), dtype=np.float32)  # everyone is everyone's neighbor
    sm = knn_smooth(probs, X, k=3, alpha=0.5)
    assert sm[0, 0] < 1.0 and sm[1, 0] > 0.0


# -- crosswalk -------------------------------------------------------------------


def test_crosswalk_exact_and_curated_and_dropped():
    mapping = build_crosswalk(
        ["Hip-Hop", "Death Metal", "Experimental", "Zorbian Throat Bass"],
        VOCAB,
    )
    assert mapping["Hip-Hop"] == ["hip hop"]      # curated
    assert mapping["Death Metal"] == ["death metal"]  # normalized exact
    assert mapping["Experimental"] == []          # curated drop
    assert mapping["Zorbian Throat Bass"] == []   # unmapped → dropped, not guessed


def test_crosswalk_labels_multihot():
    mapping = {"Hip-Hop": ["hip hop"], "Weird": []}
    Y = crosswalk_labels(
        [["Hip-Hop", "Weird"], ["Weird"], []], mapping, VOCAB
    )
    gi = VOCAB.index("hip hop")
    assert Y[0, gi] == 1 and Y[1].nnz == 0 and Y[2].nnz == 0


# -- evaluate: metrics + the wall ------------------------------------------------


def test_evaluate_reports_per_genre_f1_and_support():
    from anther_ml.corpus.tagging.evaluate import evaluate_probe

    X, W, genres, artists = _blob_training_set()
    probe = TagProbe.fit(X, W, genres, artists, min_support=10)
    report = evaluate_probe(probe, X, (W > 0).tocsr(), genres)
    assert report["n_eval_tracks"] == len(X)
    for g in probe.genre_names:
        assert {"support", "precision", "recall", "f1"} <= set(
            report["per_genre"][g]
        )
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["per_genre"]["blob zero core"]["f1"] > 0.9


def test_evaluate_is_walled_off_from_anther_ml_eval():
    """Import-guard (plan §5c): the tag scoreboard never touches the map's
    scoreboard — no import may resolve to anther_ml.eval."""
    import ast

    import anther_ml.corpus.tagging.evaluate as tag_eval

    tree = ast.parse(inspect.getsource(tag_eval))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("anther_ml.eval")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # absolute anther_ml.eval, or relative "...eval" (level 3 from
            # corpus/tagging/), or "from ... import eval"
            assert not mod.startswith("anther_ml.eval")
            if node.level == 3:
                assert mod.split(".")[0] != "eval"
                assert all(a.name != "eval" for a in node.names)
    assert "build_scorecard" not in {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    }


# -- placement wiring -------------------------------------------------------------


@pytest.fixture(scope="module")
def tagged_bundle(tmp_path_factory):
    """Fixture bundle + a probe fitted from synthetic seeds via the real
    orchestrator path (seed → fit → predict on the bundle dir)."""
    from anther_ml.corpus.tagging.build_tags import (
        run_fit,
        run_predict,
        run_seed,
    )

    tmp = tmp_path_factory.mktemp("tagged")
    corpus, bundle_dir = build_test_corpus(tmp, n=60, name="tagged")
    # vocab whose entries are exactly the fixture playlist names
    vocab_path = tmp / "vocab.txt"
    vocab_path.write_text("pl a\npl b\n")
    run_seed(bundle_dir, vocab_path=vocab_path)
    run_fit(bundle_dir, min_support=3, verbose=False)
    run_predict(bundle_dir, knn_smooth_k=5)
    return bundle_dir


def test_place_returns_tags_with_probe(tagged_bundle):
    from anther_ml.corpus import ReferenceCorpus
    from anther_ml.corpus.place import place

    corpus = ReferenceCorpus.load(tagged_bundle)
    assert corpus.tag_probe is not None
    result = place(corpus, blob_vector(0))  # blob 0 ↔ playlist "pl_a"
    assert result["tags"], "probe present → tags expected"
    tag = result["tags"][0]
    assert {"genre", "score", "primary", "source"} <= set(tag)
    assert tag["primary"] is True and tag["genre"] == "pl a"


def test_place_neighbor_inherited_tags_without_probe(tagged_bundle):
    from anther_ml.corpus import ReferenceCorpus
    from anther_ml.corpus.place import place

    (tagged_bundle / "tag_probe.pkl").rename(tagged_bundle / "tag_probe.bak")
    try:
        corpus = ReferenceCorpus.load(tagged_bundle)
        assert corpus.tag_probe is None and corpus.track_tags is not None
        result = place(corpus, blob_vector(0))
        assert result["tags"] and result["tags"][0]["source"] == "neighbors"
    finally:
        (tagged_bundle / "tag_probe.bak").rename(tagged_bundle / "tag_probe.pkl")


def test_place_empty_tags_without_any_artifacts(tmp_path):
    from anther_ml.corpus import ReferenceCorpus
    from anther_ml.corpus.place import place

    _, bundle_dir = build_test_corpus(tmp_path, n=30, name="bare")
    corpus = ReferenceCorpus.load(bundle_dir)
    result = place(corpus, blob_vector(0))
    assert result["tags"] == []


def test_recipe_mismatch_still_refused(tagged_bundle):
    from anther_ml.corpus import ReferenceCorpus

    corpus = ReferenceCorpus.load(tagged_bundle)
    bad = dict(corpus.embedding_config)
    bad["sr"] = 16000
    with pytest.raises(ValueError):
        corpus.assert_compatible(bad)
