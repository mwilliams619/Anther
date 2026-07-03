"""Workstream A — similarity math (column standardization before cosine)."""
import json

import numpy as np
import pytest

from anther_ml.similarity import SongIndex


def _corpus():
    """
    3 tracks, columns = [big_spectral_hz, chroma_a, chroma_b].
      A: loud, pitch-profile "a"
      B: loud, pitch-profile "b"  (same loudness as A, different music)
      C: quiet, pitch-profile "a" (different loudness, same music as A)
    The big column spans ~10³ while chroma spans ~1 — the exact scale gap that
    lets one Hz-scale dimension dominate raw cosine.
    """
    emb = np.array(
        [
            [5000.0, 1.0, 0.0],  # A
            [5000.0, 0.0, 1.0],  # B
            [500.0, 1.0, 0.0],   # C
        ],
        dtype=np.float32,
    )
    meta = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    return emb, meta


def test_raw_cosine_is_dominated_by_scale():
    """Without standardization, B (same loudness) outranks C (same music)."""
    emb, meta = _corpus()
    idx = SongIndex(emb, meta, standardize=False)
    res = idx.query(emb[0], top_k=3)
    ranked = [r["name"] for r in res]
    # A is itself; the bug: B ranks above C purely because of the loud column.
    assert ranked.index("B") < ranked.index("C")


def test_standardization_lets_music_dominate():
    """With standardization, C (same pitch profile) outranks B."""
    emb, meta = _corpus()
    idx = SongIndex(emb, meta, standardize=True)
    res = idx.query(emb[0], top_k=3)
    ranked = [r["name"] for r in res]
    assert ranked.index("C") < ranked.index("B")


def test_shared_loud_band_not_a_near_duplicate_after_standardization():
    """
    Acceptance (A): two musically-different tracks that merely share a loud
    spectral band were ranked as near-duplicates before (raw cosine ≈ 1,
    because normalization collapses onto the dominant band) but must NOT be
    after standardization (their differing chroma now counts).
    """
    emb, meta = _corpus()  # A and B share the loud band (5000) but differ in chroma

    raw = SongIndex(emb, meta, standardize=False)
    raw_score = next(r["score"] for r in raw.query(emb[0], top_k=3) if r["name"] == "B")
    assert raw_score > 0.99  # the bug: looks like a near-duplicate

    std = SongIndex(emb, meta, standardize=True)
    std_score = next(r["score"] for r in std.query(emb[0], top_k=3) if r["name"] == "B")
    assert std_score < 0.9   # fixed: no longer a near-duplicate
    assert std_score < raw_score


def test_query_uses_corpus_stats_not_its_own():
    emb, meta = _corpus()
    idx = SongIndex(emb, meta, standardize=True)
    # A single query vector has no variance of its own; standardization must
    # still work by using the persisted corpus mean/scale.
    res = idx.query(emb[1], top_k=1)
    assert res[0]["name"] == "B"  # exact corpus row retrieves itself


def test_save_load_roundtrip_preserves_standardization(tmp_path):
    emb, meta = _corpus()
    cfg = {"phase": 1, "clip_seconds": 30}
    idx = SongIndex(emb, meta, standardize=True, config=cfg)
    path = tmp_path / "idx"
    idx.save(path)

    loaded = SongIndex.load(path)
    assert loaded.standardize is True
    assert loaded.format_version == 2
    assert loaded.config == cfg
    np.testing.assert_allclose(loaded.mean_, idx.mean_)
    np.testing.assert_allclose(loaded.scale_, idx.scale_)
    # Queries must return identical rankings after reload.
    a = [r["name"] for r in idx.query(emb[0], top_k=3)]
    b = [r["name"] for r in loaded.query(emb[0], top_k=3)]
    assert a == b


def test_legacy_v1_index_still_loads(tmp_path):
    """A pre-existing v1 index (bare metadata list, row-normalized) must load."""
    emb, meta = _corpus()
    path = tmp_path / "legacy"
    np.save(path.with_suffix(".npy"), emb)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)
    loaded = SongIndex.load(path)
    assert loaded.format_version == 1
    assert loaded.standardize is False
    assert loaded.metadata == meta


def test_assert_compatible_flags_config_mismatch():
    emb, meta = _corpus()
    idx = SongIndex(emb, meta, config={"clip_seconds": 30})
    idx.assert_compatible({"clip_seconds": 30})  # ok
    with pytest.raises(ValueError):
        idx.assert_compatible({"clip_seconds": 10})
