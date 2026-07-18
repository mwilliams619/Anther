"""Corpus builder: dedupe, artist cap, end-to-end build, checkpoint resume."""
import numpy as np
import pytest

from anther_ml.corpus import build_corpus, dedupe_near_identical
from anther_ml.corpus.sources import _cap_by_artist
from corpus_fixtures import blob_vector, build_test_corpus, fake_embed_fn, make_items

EXPECTED_ARTIFACTS = (
    "manifest.json",
    "embeddings.npy",
    "index.npy",
    "index.json",
    "leiden.pkl",
    "labels.npy",
    "embedding_2d.npy",
    "centroids.npy",
    "cluster_profiles.json",
)


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #

def test_dedupe_drops_near_duplicate_keeps_first():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 8)).astype(np.float32)
    X[5] = X[2] + rng.normal(scale=1e-4, size=8)  # near-exact duplicate of row 2
    keep = dedupe_near_identical(X, threshold=0.98)
    assert keep[2] and not keep[5]  # first occurrence wins
    assert keep.sum() == 11


def test_dedupe_does_not_drop_distinct_same_blob_tracks():
    # Ten distinct tracks from the same blob: similar, but not duplicates.
    X = np.vstack([blob_vector(i) for i in range(0, 30, 3)])
    keep = dedupe_near_identical(X, threshold=0.98)
    assert keep.all()


def test_dedupe_ann_matches_exact():
    # Tier 2A: the ANN path must reproduce the exact O(N²) keep-mask.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 32)).astype(np.float32)
    for src, dst in [(10, 50), (10, 51), (200, 480), (5, 6), (300, 301)]:
        X[dst] = X[src] + rng.normal(scale=1e-4, size=32)  # planted near-dupes

    exact = dedupe_near_identical(X, threshold=0.98, method="exact")
    ann = dedupe_near_identical(X, threshold=0.98, method="ann", n_neighbors=20)

    np.testing.assert_array_equal(exact, ann)
    # sanity: first occurrence kept, later duplicates dropped
    for src, dst in [(10, 50), (10, 51), (200, 480), (5, 6), (300, 301)]:
        assert exact[src] and not exact[dst]


# --------------------------------------------------------------------------- #
# artist cap
# --------------------------------------------------------------------------- #

def test_artist_cap_keeps_first_n_per_artist():
    items = [{"artist": "Prolific", "id": i} for i in range(6)] + [
        {"artist": f"solo {i}", "id": 10 + i} for i in range(4)
    ]
    capped = _cap_by_artist(items, cap=2)
    assert len(capped) == 6  # 2 of Prolific + 4 solos
    assert [i["id"] for i in capped if i["artist"] == "Prolific"] == [0, 1]

    # The cap matches artists case-insensitively.
    shouty = [dict(i, artist=i["artist"].upper()) for i in items]
    assert [i["id"] for i in _cap_by_artist(shouty, cap=2)] == [
        i["id"] for i in capped
    ]


# --------------------------------------------------------------------------- #
# end-to-end build
# --------------------------------------------------------------------------- #

def test_build_end_to_end(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=30)

    for artifact in EXPECTED_ARTIFACTS:
        assert (bundle_dir / artifact).exists(), artifact
    assert not (bundle_dir / "checkpoint").exists()  # cleared on freeze

    assert corpus.n_tracks == 30
    assert corpus.manifest["n_tracks"] == 30
    assert corpus.manifest["diagnostics"]["n_clusters"] >= 2
    assert sum(p["size"] for p in corpus.profiles) == 30

    # Genre was never provided (make_items omits the key) — the build must not
    # need it: genre is display-only, never an input.
    assert all("genre" not in row for row in corpus.metadata)

    # Exemplars must carry their own cluster's label.
    for profile in corpus.profiles:
        for ex in profile["exemplars"]:
            assert corpus.labels[ex["idx"]] == profile["cluster_id"]


class _FakeMeritProcessor:
    def __call__(self, windows, sampling_rate, return_tensors, padding):
        import torch
        maxlen = max(len(w) for w in windows)
        arr = np.zeros((len(windows), maxlen), dtype=np.float32)
        for i, w in enumerate(windows):
            arr[i, : len(w)] = w
        return {"input_values": torch.tensor(arr)}


class _FakeMeritModel:
    """25 hidden states (real-MERT shape) so merit_concat's layer 23
    requirement is satisfiable; deterministic per-window/per-layer values."""
    H = 4
    N_LAYERS = 25

    def __call__(self, input_values, output_hidden_states):
        import torch
        b, t = input_values.shape
        wmean = input_values.mean(dim=1, keepdim=True)
        hidden = []
        for layer in range(self.N_LAYERS):
            base = (wmean + layer).unsqueeze(1)
            hidden.append(base.expand(b, 3, self.H).clone())

        class _Out:
            hidden_states = tuple(hidden)
        return _Out()


def _make_audio_items(n: int, sr: int) -> list[dict]:
    rng = np.random.default_rng(7)
    return [
        {
            "id": f"aud:{i:03d}",
            "name": f"audsong {i:03d}",
            "artist": f"artist {i % 5}",
            "source": "fake",
            "playlists": [],
            "audio": rng.standard_normal(int(sr * 5)).astype(np.float32),
            "sr": sr,
        }
        for i in range(n)
    ]


def test_build_captures_merit_backbone_alongside_mert(tmp_path, monkeypatch):
    """capture_merit_backbone=True must save a row-aligned merit_backbone.npy
    without changing the standard 1024-d (well, here 4-d fake-H) MERT path."""
    from anther_ml.embedding import SR

    items = _make_audio_items(12, SR)
    corpus = build_corpus(
        items,
        name="merit_capture",
        out_dir=tmp_path,
        model=_FakeMeritModel(),
        processor=_FakeMeritProcessor(),
        device="cpu",
        seed=42,
        capture_merit_backbone=True,
        batch_windows=4,
        use_fp16=False,
        dedupe_threshold=None,
    )
    bundle_dir = tmp_path / "corpus_merit_capture"
    backbone_path = bundle_dir / "merit_backbone.npy"
    assert backbone_path.exists()
    backbone = np.load(backbone_path)
    assert backbone.shape == (corpus.n_tracks, _FakeMeritModel.H * 5)  # H * len(MERIT_LAYERS)
    assert not (bundle_dir / "checkpoint_merit").exists()  # cleared on freeze

    # Row alignment: metadata order matches embeddings.npy order matches backbone order.
    assert len(corpus.metadata) == backbone.shape[0]


def test_build_requires_min_tracks(tmp_path):
    with pytest.raises(ValueError, match="at least 10"):
        build_corpus(
            make_items(5), name="tiny", out_dir=tmp_path, embed_fn=fake_embed_fn
        )


def test_build_limit_counts_source_items(tmp_path):
    corpus = build_corpus(
        make_items(30), name="lim", out_dir=tmp_path, limit=12,
        embed_fn=fake_embed_fn,
    )
    assert corpus.n_tracks == 12


# --------------------------------------------------------------------------- #
# checkpoint resume
# --------------------------------------------------------------------------- #

def _counting_embed_fn(crash_after: int | None = None):
    calls = []

    def embed(item):
        if crash_after is not None and len(calls) >= crash_after:
            raise KeyboardInterrupt  # simulate Ctrl-C mid-job
        calls.append(item["id"])
        return fake_embed_fn(item)

    return embed, calls


def test_checkpoint_resume_does_not_re_embed(tmp_path):
    items = make_items(30)

    crashing, first_calls = _counting_embed_fn(crash_after=15)
    with pytest.raises(KeyboardInterrupt):
        build_corpus(
            items, name="res", out_dir=tmp_path,
            embed_fn=crashing, checkpoint_flush_every=5,
        )
    assert len(first_calls) == 15

    resuming, second_calls = _counting_embed_fn()
    corpus = build_corpus(
        items, name="res", out_dir=tmp_path,
        embed_fn=resuming, checkpoint_flush_every=5, resume=True,
    )
    # Every track embedded exactly once across both runs.
    assert len(second_calls) == 15
    assert set(first_calls) | set(second_calls) == {i["id"] for i in items}
    assert corpus.n_tracks == 30

    # The resumed corpus is identical to an uninterrupted one.
    clean = build_corpus(
        items, name="clean", out_dir=tmp_path, embed_fn=fake_embed_fn
    )
    np.testing.assert_allclose(corpus.embeddings, clean.embeddings)
    assert [r["id"] for r in corpus.metadata] == [r["id"] for r in clean.metadata]


def test_fresh_discards_checkpoint(tmp_path):
    items = make_items(30)
    crashing, _ = _counting_embed_fn(crash_after=15)
    with pytest.raises(KeyboardInterrupt):
        build_corpus(
            items, name="fresh", out_dir=tmp_path,
            embed_fn=crashing, checkpoint_flush_every=5,
        )

    counting, calls = _counting_embed_fn()
    build_corpus(
        items, name="fresh", out_dir=tmp_path,
        embed_fn=counting, checkpoint_flush_every=5, resume=False,
    )
    assert len(calls) == 30  # everything re-embedded


def test_failed_tracks_are_skipped_not_fatal(tmp_path):
    items = make_items(15)

    def flaky(item):
        if item["id"].endswith("003"):
            raise RuntimeError("corrupt file")
        return fake_embed_fn(item)

    corpus = build_corpus(items, name="flaky", out_dir=tmp_path, embed_fn=flaky)
    assert corpus.n_tracks == 14
    assert all(r["id"] != "fake:003" for r in corpus.metadata)
