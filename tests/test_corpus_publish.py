"""
Publishing a bundle: the portable Leiden format and `corpus publish`.

The load-bearing property is that a *published* bundle still places songs
identically to the bundle it came from — stripping metadata and de-pickling the
Leiden fit must not move any query. Everything here builds on the synthetic
blob corpus, so no MERT, no network, no FMA.
"""

import json
import pickle

import numpy as np
import pytest

from anther_ml.cluster import (
    FrozenPCA,
    FrozenStandardScaler,
    load_leiden,
    load_leiden_portable,
    save_leiden_portable,
)
from anther_ml.corpus.bundle import ReferenceCorpus, leiden_path
from anther_ml.corpus.publish import (
    STRIPPED_METADATA_KEYS,
    publish_bundle,
    strip_metadata_rows,
    verify_published,
)
from tests.corpus_fixtures import blob_vector, build_test_corpus


# --------------------------------------------------------------------------
# Portable Leiden format
# --------------------------------------------------------------------------


def test_frozen_scaler_matches_sklearn():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    sk = StandardScaler().fit(X)
    frozen = FrozenStandardScaler(sk.mean_, sk.scale_)
    np.testing.assert_allclose(frozen.transform(X), sk.transform(X), atol=1e-5)


@pytest.mark.parametrize("whiten", [False, True])
def test_frozen_pca_matches_sklearn(whiten):
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    sk = PCA(n_components=4, whiten=whiten, random_state=0).fit(X)
    frozen = FrozenPCA(sk.components_, sk.mean_, sk.explained_variance_, whiten=whiten)
    np.testing.assert_allclose(frozen.transform(X), sk.transform(X), atol=1e-5)


def test_portable_roundtrip_reproduces_transforms(tmp_path):
    """The whole point: npz transforms == pickled transforms, on new vectors."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    pickled = load_leiden(bundle_dir / "leiden.pkl")

    save_leiden_portable(tmp_path / "leiden.npz", pickled)
    portable = load_leiden_portable(tmp_path / "leiden.npz")

    query = np.stack([blob_vector(500 + i) for i in range(5)])
    a, b = query, query
    for key in ("scaler", "pca"):
        if pickled[key] is not None:
            a = pickled[key].transform(a)
            b = portable[key].transform(b)
    np.testing.assert_allclose(a, b, atol=1e-4)

    np.testing.assert_array_equal(portable["labels"], pickled["labels"])
    np.testing.assert_allclose(
        portable["clustering_space"], pickled["clustering_space"], atol=1e-5
    )
    assert portable["resolution"] == pickled["resolution"]
    assert portable["metric"] == pickled["metric"]
    # Not portable, and callers must see it as absent rather than stale.
    assert portable["reducer_2d"] is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frozen_shims_preserve_sklearn_dtype(dtype):
    """sklearn keeps float32 float32 (in-place scaler ops) and promotes the
    rest. The shims must follow, or a published bundle transforms queries at a
    different precision than the bundle it came from."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 16)).astype(dtype)
    sk_s = StandardScaler().fit(X)
    sk_p = PCA(n_components=4, random_state=0).fit(sk_s.transform(X))

    fz_s = FrozenStandardScaler(sk_s.mean_, sk_s.scale_)
    fz_p = FrozenPCA(sk_p.components_, sk_p.mean_, sk_p.explained_variance_)

    assert fz_s.transform(X).dtype == sk_s.transform(X).dtype
    assert (
        fz_p.transform(fz_s.transform(X)).dtype
        == sk_p.transform(sk_s.transform(X)).dtype
    )


def test_portable_npz_contains_no_pickle(tmp_path):
    """allow_pickle=False must suffice — that is the security property."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    save_leiden_portable(tmp_path / "leiden.npz", load_leiden(bundle_dir / "leiden.pkl"))
    with np.load(tmp_path / "leiden.npz", allow_pickle=False) as z:
        assert "labels" in z and "clustering_space" in z


def test_load_leiden_dispatches_on_suffix(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    save_leiden_portable(tmp_path / "leiden.npz", load_leiden(bundle_dir / "leiden.pkl"))
    assert load_leiden(tmp_path / "leiden.npz")["reducer_2d"] is None
    assert load_leiden(bundle_dir / "leiden.pkl")["labels"] is not None


def test_portable_rejects_unflattenable_transform(tmp_path):
    """A scaler this can't flatten must fail loudly, not publish silently."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    leiden = load_leiden(bundle_dir / "leiden.pkl")
    leiden["scaler"] = object()
    with pytest.raises(TypeError, match="cannot flatten scaler"):
        save_leiden_portable(tmp_path / "bad.npz", leiden)


def test_portable_version_mismatch_raises(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    save_leiden_portable(tmp_path / "leiden.npz", load_leiden(bundle_dir / "leiden.pkl"))
    with np.load(tmp_path / "leiden.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta")))
    meta["leiden_portable_format_version"] = 99
    np.savez_compressed(tmp_path / "bad.npz", meta=np.array(json.dumps(meta)), **arrays)
    with pytest.raises(ValueError, match="portable format version"):
        load_leiden_portable(tmp_path / "bad.npz")


def test_bundle_prefers_portable_leiden(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    assert leiden_path(bundle_dir).name == "leiden.pkl"
    save_leiden_portable(bundle_dir / "leiden.npz", load_leiden(bundle_dir / "leiden.pkl"))
    assert leiden_path(bundle_dir).name == "leiden.npz"


# --------------------------------------------------------------------------
# publish_bundle
# --------------------------------------------------------------------------


def test_strip_metadata_rows_counts_only_changed():
    rows = [{"id": "a", "playlists": [1]}, {"id": "b"}]
    out, touched = strip_metadata_rows(rows, ("playlists",))
    assert touched == 1
    assert out == [{"id": "a"}, {"id": "b"}]


def test_publish_strips_playlists_and_depickles(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    # The fixture puts blobs 0 and 1 on playlists, so there is something to strip.
    assert any(m.get("playlists") for m in corpus.metadata)

    out = tmp_path / "published"
    report = publish_bundle(bundle_dir, out)

    assert not (out / "leiden.pkl").exists()
    assert (out / "leiden.npz").exists()
    assert report["bytes_after"] < report["bytes_before"]
    assert report["stripped_metadata_keys"] == list(STRIPPED_METADATA_KEYS)

    with open(out / "index.json") as f:
        rows = json.load(f)["metadata"]
    assert rows and all("playlists" not in r for r in rows)
    # Track facts survive — this is what keeps playback and search working.
    assert all(r["id"] and r["name"] and r["artist"] for r in rows)


def test_published_bundle_places_identically(tmp_path):
    """The property that matters: publishing moves no query."""
    from anther_ml.corpus.place import place

    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)
    published = ReferenceCorpus.load(out)

    for i in range(6):
        vec = blob_vector(900 + i)
        a = place(corpus, vec, top_k=5)
        b = place(published, vec, top_k=5)
        assert a["cluster"]["id"] == b["cluster"]["id"]
        assert a["cluster"]["confidence"] == b["cluster"]["confidence"]
        assert [n["id"] for n in a["neighbors"]] == [n["id"] for n in b["neighbors"]]


def test_published_place_returns_no_2d_coords(tmp_path):
    """reducer_2d is not portable; the documented degradation must be graceful."""
    from anther_ml.corpus.place import place

    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)
    result = place(ReferenceCorpus.load(out), blob_vector(901), top_k=3)
    assert result["coords_2d"] is None


def test_publish_drops_backbone_and_scratch(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    (bundle_dir / "merit_backbone.npy").write_bytes(b"x" * 1024)
    (bundle_dir / "build.log").write_text("noise")

    out = tmp_path / "published"
    report = publish_bundle(bundle_dir, out)
    assert not (out / "merit_backbone.npy").exists()
    assert not (out / "build.log").exists()
    assert "merit_backbone.npy" in report["excluded_files"]


def test_publish_copies_unknown_sidecars(tmp_path):
    """Copy-unless-excluded: a future sidecar must travel without a code change."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    (bundle_dir / "some_future_sidecar.json").write_text('{"ok": true}')
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)
    assert (out / "some_future_sidecar.json").exists()


def test_publish_records_provenance_in_manifest(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    with open(out / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["publish"]["stripped_metadata_keys"] == ["playlists"]
    assert manifest["publish"]["portable_leiden"] is True
    # Source provenance is documented, not laundered.
    assert manifest["name"] == "test"
    assert "embedding_config" in manifest


def test_publish_keep_flags(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    (bundle_dir / "merit_backbone.npy").write_bytes(b"x" * 1024)

    out = tmp_path / "kept"
    publish_bundle(
        bundle_dir, out,
        strip_playlists=False, drop_backbone=False, portable_leiden=False,
    )
    assert (out / "merit_backbone.npy").exists()
    assert (out / "leiden.pkl").exists()
    with open(out / "index.json") as f:
        assert any("playlists" in r for r in json.load(f)["metadata"])


# --------------------------------------------------------------------------
# Lazy embeddings
# --------------------------------------------------------------------------


def test_load_does_not_read_embeddings_eagerly(tmp_path, monkeypatch):
    """406 MB on the 100k bundle; nothing on the placement path needs it."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)

    real_load = np.load
    loaded = []

    def spy(path, *a, **k):
        loaded.append(str(path))
        return real_load(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    reloaded = ReferenceCorpus.load(bundle_dir)
    assert not any("embeddings.npy" in p for p in loaded)

    # ...and it is memory-mapped, not read whole, on first touch.
    assert isinstance(reloaded.embeddings, np.memmap)
    assert any("embeddings.npy" in p for p in loaded)


def test_lazy_embeddings_match_and_stay_settable(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    reloaded = ReferenceCorpus.load(bundle_dir)
    np.testing.assert_allclose(
        np.asarray(reloaded.embeddings), np.asarray(corpus.embeddings), atol=0
    )

    # extend_corpus rebinds this; the setter must accept a plain array.
    grown = np.concatenate([np.asarray(reloaded.embeddings)] * 2, axis=0)
    reloaded.embeddings = grown
    assert reloaded.embeddings.shape[0] == 2 * corpus.n_tracks
    assert not isinstance(reloaded.embeddings, np.memmap)


def test_embeddings_without_dir_raises():
    with pytest.raises(ValueError, match="no bundle dir"):
        _ = ReferenceCorpus(
            embeddings=None, index=None, leiden={}, embedding_2d=np.zeros((1, 2)),
            centroids=np.zeros((1, 2)), profiles=[], manifest={},
        ).embeddings


# --------------------------------------------------------------------------
# Deduplicated metadata
# --------------------------------------------------------------------------


def test_publish_dedupes_merit_metadata(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    _fake_merit_sidecar(bundle_dir, corpus)

    out = tmp_path / "published"
    report = publish_bundle(bundle_dir, out)

    with open(out / "index_merit_agg.json") as f:
        payload = json.load(f)
    assert "metadata" not in payload
    assert payload["metadata_ref"] == "index"
    assert report["rewritten_indices"]["index_merit_agg.json"]["deduped_to"] == "index"


def test_deduped_merit_index_resolves_metadata(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    _fake_merit_sidecar(bundle_dir, corpus)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    published = ReferenceCorpus.load(out)
    merit = published.merit_index
    assert merit is not None
    assert len(merit.metadata) == corpus.n_tracks
    assert [m["id"] for m in merit.metadata] == [m["id"] for m in published.metadata]
    # Shared, not copied — that is the memory win.
    assert merit.metadata is published.index.metadata


def test_deduped_merit_index_resolves_standalone(tmp_path):
    """A caller without the main index in hand follows the pointer itself."""
    from anther_ml.corpus.merit_index import load_merit_aggregate_index

    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    _fake_merit_sidecar(bundle_dir, corpus)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    merit = load_merit_aggregate_index(out)
    assert merit is not None and len(merit.metadata) == corpus.n_tracks


def test_songindex_load_without_metadata_raises(tmp_path):
    from anther_ml.similarity import SongIndex

    np.save(tmp_path / "idx.npy", np.zeros((3, 4), dtype=np.float32))
    with open(tmp_path / "idx.json", "w") as f:
        json.dump({"format_version": 2, "metadata_ref": "index"}, f)
    with pytest.raises(ValueError, match="metadata_ref='index'"):
        SongIndex.load(tmp_path / "idx")


def test_publish_keep_duplicate_metadata(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    _fake_merit_sidecar(bundle_dir, corpus)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out, dedupe_metadata=False)

    with open(out / "index_merit_agg.json") as f:
        payload = json.load(f)
    assert len(payload["metadata"]) == corpus.n_tracks
    assert "metadata_ref" not in payload


def _fake_merit_sidecar(bundle_dir, corpus):
    """A minimal index_merit_agg pair — enough to exercise the dedupe path
    without MERIT heads."""
    from anther_ml.similarity import SongIndex

    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(corpus.n_tracks, 12)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    SongIndex(vecs, corpus.metadata, standardize=False, config={}).save(
        bundle_dir / "index_merit_agg"
    )


def test_publish_dry_run_writes_nothing(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    report = publish_bundle(bundle_dir, out, dry_run=True)
    assert not out.exists()
    assert report["dry_run"] is True
    assert report["bytes_after"] < report["bytes_before"]


def test_publish_refuses_to_overwrite_source(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    with pytest.raises(ValueError, match="overwrite the source"):
        publish_bundle(bundle_dir, bundle_dir)


def test_publish_rejects_nonempty_destination(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    out.mkdir()
    (out / "stale-artifact").write_text("old")

    with pytest.raises(FileExistsError, match="must not exist or must be empty"):
        publish_bundle(bundle_dir, out)


def test_publish_rejects_non_bundle(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no corpus bundle"):
        publish_bundle(tmp_path / "empty", tmp_path / "out")


def test_verify_published_standalone(tmp_path):
    """Without the source bundle: cosine agreement with clustering_space."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    checks = verify_published(out)
    assert checks["pickle_free"]
    assert checks["backbone_dropped"]
    assert checks["playlists_stripped"]
    assert checks["transform_ok"]
    assert checks["transform_checked_against"] == "stored clustering_space"
    assert checks["transform_min_cosine"] > 0.999
    assert checks["n_tracks"] == corpus.n_tracks


def test_verify_published_against_source(tmp_path):
    """With the source bundle: direct agreement with the pickled estimators."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    checks = verify_published(out, src_dir=bundle_dir)
    assert checks["transform_checked_against"] == "source bundle"
    assert checks["transform_max_abs_err"] < 1e-4
    assert checks["transform_ok"]


def test_verify_detects_a_corrupted_transform(tmp_path):
    """The check must actually fail when the published transform drifts."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    with np.load(out / "leiden.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = arrays.pop("meta")
    arrays["scaler_mean"] = arrays["scaler_mean"] + 5.0
    np.savez_compressed(out / "leiden.npz", meta=meta, **arrays)

    assert not verify_published(out, src_dir=bundle_dir)["transform_ok"]


def test_published_bundle_loads_without_pickle_module(tmp_path, monkeypatch):
    """A published bundle must not reach for pickle at load time."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    def _boom(*a, **k):
        raise AssertionError("published bundle unpickled something")

    monkeypatch.setattr(pickle, "load", _boom)
    loaded = ReferenceCorpus.load(out)
    assert loaded.n_tracks == corpus.n_tracks
