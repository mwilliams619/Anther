"""
Hugging Face transport for published bundles.

No network: `huggingface_hub` is stubbed. What is worth testing here is the
policy around the transport — that an unpublished bundle can't be pushed by
accident, that a fetch is never implicit, and that an existing bundle is never
silently replaced by a download.
"""

import json

import numpy as np
import pytest

from anther_ml.corpus import hub
from anther_ml.corpus.bundle import ReferenceCorpus
from anther_ml.corpus.publish import publish_bundle
from tests.corpus_fixtures import build_test_corpus


def test_publish_state_flags_unpublished_bundle(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    (bundle_dir / "merit_backbone.npy").write_bytes(b"x" * 16)

    state = hub.bundle_publish_state(bundle_dir)
    assert not state["published"]
    joined = " ".join(state["reasons"])
    assert "leiden.pkl" in joined
    assert "merit_backbone.npy" in joined
    assert "playlist membership" in joined
    assert "never went through" in joined


def test_publish_state_accepts_published_bundle(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    state = hub.bundle_publish_state(out)
    assert state["published"], state["reasons"]


def test_publish_state_rejects_non_bundle(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no corpus bundle"):
        hub.bundle_publish_state(tmp_path / "empty")


def test_publish_state_scans_playlist_membership_after_first_thousand_rows(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"publish": {}}))
    rows = [{} for _ in range(1000)] + [{"playlists": [{"pid": "secret"}]}]
    (bundle / "index.json").write_text(json.dumps({"metadata": rows}))

    state = hub.bundle_publish_state(bundle)
    assert not state["published"]
    assert any("playlist membership" in r for r in state["reasons"])


class _FakeApi:
    """Records calls instead of talking to the Hub."""

    def __init__(self):
        self.created = None
        self.branch = None
        self.uploaded = None

    def create_repo(self, repo_id, **kw):
        self.created = (repo_id, kw)

    def create_branch(self, repo_id, **kw):
        self.branch = (repo_id, kw)

    def upload_folder(self, **kw):
        self.uploaded = kw


def test_push_refuses_unpublished_bundle(tmp_path, monkeypatch):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    api = _FakeApi()
    monkeypatch.setattr(hub, "_api", lambda: api)

    with pytest.raises(ValueError, match="refusing to push an unpublished bundle"):
        hub.push_bundle(bundle_dir, "me/corpus")
    assert api.created is None, "must not create a repo before validating"


def test_push_uploads_published_bundle_privately(tmp_path, monkeypatch):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    out = tmp_path / "published"
    publish_bundle(bundle_dir, out)

    api = _FakeApi()
    monkeypatch.setattr(hub, "_api", lambda: api)
    url = hub.push_bundle(out, "me/corpus", revision="v1")

    assert url == "https://huggingface.co/datasets/me/corpus"
    assert api.created[0] == "me/corpus"
    assert api.created[1]["private"] is True  # public is a deliberate act
    assert api.created[1]["repo_type"] == "dataset"
    assert api.branch[1]["branch"] == "v1"
    assert api.uploaded["revision"] == "v1"
    assert "*.log" in api.uploaded["ignore_patterns"]


def test_push_rejects_unpublished_even_when_public(tmp_path, monkeypatch):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    api = _FakeApi()
    monkeypatch.setattr(hub, "_api", lambda: api)
    with pytest.raises(ValueError, match="refusing to push an unpublished bundle"):
        hub.push_bundle(bundle_dir, "me/public", private=False)
    assert api.created is None


def test_fetch_skips_backbone_by_default(tmp_path, monkeypatch):
    calls = {}

    def fake_snapshot(**kw):
        calls.update(kw)
        return str(tmp_path / "dl")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    hub.fetch_bundle("me/corpus", tmp_path / "dl", revision="v1")
    assert calls["ignore_patterns"] == ["merit_backbone.npy"]
    assert calls["repo_type"] == "dataset"
    assert calls["revision"] == "v1"

    hub.fetch_bundle("me/corpus", tmp_path / "dl", include_backbone=True)
    assert calls["ignore_patterns"] is None


def test_resolve_repo_id_prefers_argument(monkeypatch):
    monkeypatch.setenv(hub.CORPUS_REPO_ENV, "env/repo")
    assert hub.resolve_repo_id("arg/repo") == "arg/repo"
    assert hub.resolve_repo_id(None) == "env/repo"
    monkeypatch.delenv(hub.CORPUS_REPO_ENV)
    assert hub.resolve_repo_id(None) is None


def test_ensure_bundle_never_refetches_existing(tmp_path, monkeypatch):
    """A frozen map must not move under a running session."""
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)

    def _boom(*a, **k):
        raise AssertionError("re-fetched a bundle that was already on disk")

    monkeypatch.setattr(hub, "fetch_bundle", _boom)
    monkeypatch.setenv(hub.CORPUS_REPO_ENV, "me/corpus")
    assert hub.ensure_bundle(bundle_dir) == bundle_dir


def test_ensure_bundle_without_repo_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv(hub.CORPUS_REPO_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="corpus fetch --repo-id"):
        hub.ensure_bundle(tmp_path / "missing")


def test_load_does_not_download_implicitly(tmp_path, monkeypatch):
    """No repo configured → the old FileNotFoundError, no network attempt."""
    monkeypatch.delenv(hub.CORPUS_REPO_ENV, raising=False)

    def _boom(*a, **k):
        raise AssertionError("ReferenceCorpus.load attempted a download")

    monkeypatch.setattr(hub, "ensure_bundle", _boom)
    with pytest.raises(FileNotFoundError, match="no corpus bundle"):
        ReferenceCorpus.load(tmp_path / "missing")


def test_load_fetches_when_repo_given(tmp_path, monkeypatch):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=40)
    fetched = {}

    def fake_fetch(repo_id, local_dir, *, revision=None, include_backbone=False):
        fetched["repo_id"] = repo_id
        fetched["revision"] = revision
        return bundle_dir  # stand in for the downloaded copy

    monkeypatch.setattr(hub, "fetch_bundle", fake_fetch)
    loaded = ReferenceCorpus.load(
        tmp_path / "missing", repo_id="me/corpus", revision="v1"
    )
    assert fetched == {"repo_id": "me/corpus", "revision": "v1"}
    assert loaded.n_tracks == corpus.n_tracks
