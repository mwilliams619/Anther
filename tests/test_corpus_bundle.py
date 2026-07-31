"""ReferenceCorpus bundle: save/load roundtrip, verification, lookups."""
import json
import shutil

import numpy as np
import pytest

from anther_ml.corpus import ReferenceCorpus
from corpus_fixtures import build_test_corpus


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    return build_test_corpus(tmp)


def _mutable_copy(bundle_dir, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(bundle_dir, dst)
    return dst


def test_save_load_roundtrip(built):
    corpus, bundle_dir = built
    loaded = ReferenceCorpus.load(bundle_dir)

    np.testing.assert_array_equal(loaded.labels, corpus.labels)
    np.testing.assert_allclose(loaded.embeddings, corpus.embeddings)
    np.testing.assert_allclose(loaded.embedding_2d, corpus.embedding_2d, rtol=1e-5)
    np.testing.assert_allclose(loaded.centroids, corpus.centroids, rtol=1e-5)
    assert loaded.profiles == json.loads(json.dumps(corpus.profiles))
    assert loaded.manifest == json.loads(json.dumps(corpus.manifest))

    # Queries must rank identically before and after the roundtrip.
    q = corpus.embeddings[0]
    before = [r["name"] for r in corpus.index.query(q, top_k=5)]
    after = [r["name"] for r in loaded.index.query(q, top_k=5)]
    assert before == after


def test_load_rejects_embedding_config_mismatch(built, tmp_path):
    _, bundle_dir = built
    dst = _mutable_copy(bundle_dir, tmp_path)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["embedding_config"]["window_seconds"] = 99.0
    (dst / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="window_seconds"):
        ReferenceCorpus.load(dst)


def test_load_rejects_unknown_format_version(built, tmp_path):
    _, bundle_dir = built
    dst = _mutable_copy(bundle_dir, tmp_path)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["corpus_format_version"] = 999
    (dst / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corpus_format_version"):
        ReferenceCorpus.load(dst)


def test_load_rejects_inconsistent_track_counts(built, tmp_path):
    # labels.npy is a derived copy for the eval CLI; load() takes labels from
    # leiden.pkl, so the consistency check watches the arrays load consumes.
    _, bundle_dir = built
    dst = _mutable_copy(bundle_dir, tmp_path)
    emb = np.load(dst / "embeddings.npy")
    np.save(dst / "embeddings.npy", emb[:-3])
    with pytest.raises(ValueError, match="counts disagree"):
        ReferenceCorpus.load(dst)


def test_load_without_verify_skips_checks(built, tmp_path):
    _, bundle_dir = built
    dst = _mutable_copy(bundle_dir, tmp_path)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["corpus_format_version"] = 999
    (dst / "manifest.json").write_text(json.dumps(manifest))
    ReferenceCorpus.load(dst, verify=False)  # must not raise


def test_playlist_lookups(built):
    corpus, _ = built
    playlists = {p["name"]: p for p in corpus.playlists()}
    assert set(playlists) == {"pl_a", "pl_b"}
    assert playlists["pl_a"]["n_tracks"] == 10  # blob 0 = every 3rd of 30

    by_name = corpus.playlist_member_indices("pl_a")
    by_pid = corpus.playlist_member_indices(1)
    np.testing.assert_array_equal(by_name, by_pid)
    assert len(by_name) == 10


def test_cluster_profile_lookup(built):
    corpus, _ = built
    cid = int(corpus.labels[0])
    profile = corpus.cluster_profile(cid)
    assert profile["cluster_id"] == cid
    with pytest.raises(KeyError):
        corpus.cluster_profile(9999)
