"""
Playlist-name cluster labels (anther_ml/corpus/labels.py).

Covers name normalization, TF-IDF distinctiveness, entropy coherence, label
assembly on synthetic metadata, override persistence across re-labeling, and
an on-disk round-trip through a real fixture bundle.
"""

import json
from collections import Counter

import numpy as np
import pytest

from anther_ml.corpus.labels import (
    apply_labels_to_profiles,
    generate_cluster_labels,
    label_bundle,
    normalize_playlist_name,
    set_cluster_override,
    tag_entropy,
)

from .corpus_fixtures import build_test_corpus


# -- normalization -----------------------------------------------------------


def test_normalize_folds_case_and_whitespace_variants():
    variants = ["Texas Country", "texas country", "Texas  country ", "TEXAS COUNTRY"]
    keys = {normalize_playlist_name(v) for v in variants}
    assert keys == {"texas country"}


def test_normalize_strips_decoration():
    assert normalize_playlist_name("🔥 workout 🔥") == "workout"
    assert normalize_playlist_name("...jazz!!!") == "jazz"


@pytest.mark.parametrize("bad", ["", "🔥", "x", "   ", "!!!", None])
def test_normalize_rejects_unusable_names(bad):
    assert normalize_playlist_name(bad) is None


# -- entropy ------------------------------------------------------------------


def test_entropy_lower_for_coherent_cluster():
    single = Counter({"metal": 50})
    mixed = Counter({"metal": 10, "jazz": 10, "pop": 10, "edm": 10, "folk": 10})
    h_single, hn_single = tag_entropy(single)
    h_mixed, hn_mixed = tag_entropy(mixed)
    assert h_single == 0.0 and hn_single == 0.0
    assert h_mixed > h_single
    assert hn_mixed == pytest.approx(1.0)  # uniform → max normalized entropy


def test_entropy_empty_counter():
    assert tag_entropy(Counter()) == (0.0, 0.0)


# -- synthetic two-cluster metadata -------------------------------------------


def _two_cluster_fixture():
    """Cluster 0: metal names (+ ubiquitous 'favorites'); cluster 1: country
    names (+ 'favorites'); cluster 2: no playlists at all."""

    def track(playlist_names):
        return {"playlists": [{"pid": None, "name": n} for n in playlist_names]}

    metadata, labels = [], []
    for _ in range(6):
        metadata.append(track(["Death Metal", "favorites"]))
        labels.append(0)
    for _ in range(4):
        metadata.append(track(["death metal", "Power Metal"]))
        labels.append(0)
    for _ in range(6):
        metadata.append(track(["Texas Country", "favorites"]))
        labels.append(1)
    for _ in range(4):
        metadata.append(track(["texas country", "Red Dirt"]))
        labels.append(1)
    for _ in range(5):
        metadata.append(track([]))
        labels.append(2)
    return metadata, np.asarray(labels)


def test_generate_labels_distinctive_per_cluster():
    metadata, labels = _two_cluster_fixture()
    generated = generate_cluster_labels(metadata, labels)
    by_cid = {g["cluster_id"]: g for g in generated}
    assert set(by_cid) == {0, 1, 2}

    assert "Death Metal" in by_cid[0]["label"]
    assert "Power Metal" in by_cid[0]["label"]
    assert by_cid[0]["label_source"] == "playlist_tfidf"

    assert "Texas Country" in by_cid[1]["label"]
    assert "Red Dirt" in by_cid[1]["label"]

    # 'favorites' is a stopword — must not appear in any label.
    for g in generated:
        assert "favorites" not in g["label"].lower()

    # top_tags still report the raw normalized tally (stopwords included there).
    tag_names = [t[0].lower() for t in by_cid[0]["top_tags"]]
    assert "death metal" in tag_names

    # Case variants folded: death metal counted 10x, not split 6/4.
    counts = dict((t[0].lower(), t[1]) for t in by_cid[0]["top_tags"])
    assert counts["death metal"] == 10


def test_generate_labels_empty_playlist_cluster_no_fabrication():
    metadata, labels = _two_cluster_fixture()
    by_cid = {g["cluster_id"]: g for g in generate_cluster_labels(metadata, labels)}
    assert by_cid[2]["label"] == ""
    assert by_cid[2]["label_source"] == "none"
    assert by_cid[2]["top_tags"] == []


def test_tfidf_prefers_concentrated_over_ubiquitous():
    """A non-stopword name present in every cluster loses to a concentrated
    one even when the ubiquitous name has more raw occurrences."""

    def track(names):
        return {"playlists": [{"pid": None, "name": n} for n in names]}

    metadata, labels = [], []
    for _ in range(10):
        metadata.append(track(["road trip", "death metal"]))
        labels.append(0)
    for _ in range(10):
        metadata.append(track(["road trip"]))
        labels.append(1)
    generated = generate_cluster_labels(metadata, np.asarray(labels), label_n=1)
    by_cid = {g["cluster_id"]: g for g in generated}
    assert by_cid[0]["label"] == "Death Metal"


def test_min_support_floor():
    def track(names):
        return {"playlists": [{"pid": None, "name": n} for n in names]}

    metadata = [track(["rare gem"])] + [track([])] * 4
    labels = np.zeros(5, dtype=int)
    (g,) = generate_cluster_labels(metadata, labels, min_support=3)
    assert g["label"] == ""
    assert g["label_source"] == "none"


# -- merge + override persistence ---------------------------------------------


def test_apply_labels_preserves_profile_fields_and_computes_final():
    metadata, labels = _two_cluster_fixture()
    profiles = [
        {"cluster_id": 0, "size": 10, "exemplars": [], "top_playlists": []},
        {"cluster_id": 1, "size": 10, "exemplars": [], "top_playlists": []},
        {"cluster_id": 2, "size": 5, "exemplars": [], "top_playlists": []},
    ]
    generated = generate_cluster_labels(metadata, labels)
    merged = apply_labels_to_profiles(profiles, generated)
    for p in merged:
        assert p["size"] in (10, 5)  # untouched
        assert p["label_final"] == p["label"]
        assert "tag_entropy" in p and "tag_entropy_norm" in p


def test_override_survives_relabel():
    metadata, labels = _two_cluster_fixture()
    profiles = [
        {"cluster_id": cid, "size": 1, "exemplars": [], "top_playlists": []}
        for cid in (0, 1, 2)
    ]
    apply_labels_to_profiles(profiles, generate_cluster_labels(metadata, labels))
    profiles[0]["label_override"] = "Dreamy lo-fi (nostalgic)"
    profiles[0]["label_override_at"] = "2026-01-01T00:00:00+00:00"

    # Re-run labeling — the draft updates, the override is authoritative.
    apply_labels_to_profiles(profiles, generate_cluster_labels(metadata, labels))
    p0 = profiles[0]
    assert p0["label_override"] == "Dreamy lo-fi (nostalgic)"
    assert p0["label_final"] == "Dreamy lo-fi (nostalgic)"
    assert "Death Metal" in p0["label"]  # draft still recomputed
    assert profiles[1]["label_final"] == profiles[1]["label"]


# -- on-disk round-trip ---------------------------------------------------------


def test_label_bundle_round_trip(tmp_path):
    from anther_ml.corpus import ReferenceCorpus

    corpus, bundle_dir = build_test_corpus(tmp_path)
    before = {
        f.name: f.stat().st_mtime_ns
        for f in bundle_dir.iterdir()
        if f.name != "cluster_profiles.json"
    }

    profiles = label_bundle(bundle_dir)

    # Only cluster_profiles.json rewritten.
    after = {
        f.name: f.stat().st_mtime_ns
        for f in bundle_dir.iterdir()
        if f.name != "cluster_profiles.json"
    }
    assert after == before

    # Bundle still loads and verifies (format version unchanged).
    reloaded = ReferenceCorpus.load(bundle_dir)
    assert reloaded.manifest["corpus_format_version"] == 1

    # Every cluster with playlist data carries a label_final.
    for p in reloaded.profiles:
        assert "label_final" in p
        if p.get("top_tags"):
            assert p["label_final"]
    assert profiles == reloaded.profiles


def test_set_cluster_override_persists_on_disk(tmp_path):
    from anther_ml.corpus import ReferenceCorpus

    corpus, bundle_dir = build_test_corpus(tmp_path, name="ovr")
    cid = corpus.profiles[0]["cluster_id"]
    set_cluster_override(bundle_dir, cid, "Hand-named cluster")

    p = ReferenceCorpus.load(bundle_dir).cluster_profile(cid)
    assert p["label_override"] == "Hand-named cluster"
    assert p["label_final"] == "Hand-named cluster"
    assert p["label_override_at"]

    # A plain re-label keeps the override, only refreshes the draft/tallies.
    label_bundle(bundle_dir)
    p = ReferenceCorpus.load(bundle_dir).cluster_profile(cid)
    assert p["label_final"] == "Hand-named cluster"

    # Clearing the override falls back to the auto draft.
    set_cluster_override(bundle_dir, cid, "")
    p = ReferenceCorpus.load(bundle_dir).cluster_profile(cid)
    assert "label_override" not in p
    assert p["label_final"] == p["label"]


def test_build_path_ships_labels(tmp_path):
    """build_corpus itself now attaches label fields to every profile."""
    corpus, _ = build_test_corpus(tmp_path, name="shipped")
    for p in corpus.profiles:
        assert "label_final" in p and "label_source" in p
