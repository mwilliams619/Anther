"""
Tests for anther_ml.corpus.merit_index: building the MERIT-aggregate SongIndex
+ per-factor sidecars from a bundle's merit_backbone.npy, without touching the
existing MERT-based index.npy/embeddings.npy/leiden.pkl.
"""

import json

import numpy as np
import torch

from anther_ml.corpus.merit_index import (
    build_merit_aggregate_index,
    load_factor_vectors,
    load_merit_aggregate_index,
)
from anther_ml.merit import FACTORS, ProjectionHead

from corpus_fixtures import build_test_corpus


def _write_fake_heads(heads_dir, in_dim=40, hidden_dim=16, out_dim=128, seed=0):
    """Tiny random-but-deterministic ProjectionHead checkpoints, matching the
    on-disk format load_head/load_heads expects (in_dim/hidden_dim/out_dim +
    state_dict) — no real MERIT download needed for these tests."""
    heads_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(FACTORS):
        torch.manual_seed(seed + i)
        head = ProjectionHead(in_dim, hidden_dim, out_dim)
        d = heads_dir / f"head_{f}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "out_dim": out_dim,
                "state_dict": head.state_dict(),
            },
            d / "best_head.pt",
        )


def _build_bundle_with_backbone(tmp_path, n=40, backbone_dim=40):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=n, name="merit_agg")
    rng = np.random.default_rng(7)
    backbone = rng.normal(size=(n, backbone_dim)).astype(np.float32)
    np.save(bundle_dir / "merit_backbone.npy", backbone)
    return corpus, bundle_dir


def test_build_merit_aggregate_index_writes_all_sidecars(tmp_path):
    corpus, bundle_dir = _build_bundle_with_backbone(tmp_path)
    heads_dir = tmp_path / "heads"
    _write_fake_heads(heads_dir)

    result = build_merit_aggregate_index(bundle_dir, heads_dir=heads_dir, device="cpu")

    assert result["n_tracks"] == 40
    assert result["agg_shape"] == (40, 384)
    for f in FACTORS:
        assert (bundle_dir / f"factor_{f}.npy").exists()
    assert (bundle_dir / "index_merit_agg.npy").exists()
    assert (bundle_dir / "index_merit_agg.json").exists()
    assert (bundle_dir / "link_calibration_merit.json").exists()

    # existing MERT artifacts untouched
    assert (bundle_dir / "index.npy").exists()
    assert (bundle_dir / "embeddings.npy").exists()
    assert (bundle_dir / "leiden.pkl").exists()


def test_aggregate_is_unit_norm_and_equals_mean_of_factor_cosines(tmp_path):
    corpus, bundle_dir = _build_bundle_with_backbone(tmp_path)
    heads_dir = tmp_path / "heads"
    _write_fake_heads(heads_dir)
    build_merit_aggregate_index(bundle_dir, heads_dir=heads_dir, device="cpu")

    agg_index = load_merit_aggregate_index(bundle_dir)
    factors = load_factor_vectors(bundle_dir)

    # each 384-d row is 3 concatenated unit vectors -> whole row is NOT unit
    # norm in general, but SongIndex(standardize=False) L2-normalizes at
    # query/index time internally via _l2_normalize on save/query — verify
    # cosine(concat_a, concat_b) == mean of the 3 per-factor cosines exactly,
    # since each sub-vector already has unit norm before concatenation.
    a, b = 0, 1
    concat_cos = float(np.dot(agg_index.embeddings[a], agg_index.embeddings[b]))
    factor_cos = np.mean(
        [float(np.dot(factors[f][a], factors[f][b])) for f in FACTORS]
    )
    assert abs(concat_cos - factor_cos) < 1e-4


def test_load_merit_aggregate_index_none_when_not_built(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=10, name="no_merit")
    assert load_merit_aggregate_index(bundle_dir) is None
    assert load_factor_vectors(bundle_dir) is None


def test_build_raises_without_merit_backbone(tmp_path):
    corpus, bundle_dir = build_test_corpus(tmp_path, n=10, name="missing_backbone")
    heads_dir = tmp_path / "heads"
    _write_fake_heads(heads_dir)
    import pytest

    with pytest.raises(FileNotFoundError):
        build_merit_aggregate_index(bundle_dir, heads_dir=heads_dir, device="cpu")


def test_build_raises_on_row_count_mismatch(tmp_path):
    corpus, bundle_dir = _build_bundle_with_backbone(tmp_path, n=40)
    # corrupt the backbone to a different row count than metadata
    np.save(bundle_dir / "merit_backbone.npy", np.zeros((5, 40), dtype=np.float32))
    heads_dir = tmp_path / "heads"
    _write_fake_heads(heads_dir)
    import pytest

    with pytest.raises(ValueError):
        build_merit_aggregate_index(bundle_dir, heads_dir=heads_dir, device="cpu")


def test_corpus_merit_index_lazy_properties_roundtrip(tmp_path):
    from anther_ml.corpus.bundle import ReferenceCorpus

    corpus, bundle_dir = _build_bundle_with_backbone(tmp_path)
    heads_dir = tmp_path / "heads"
    _write_fake_heads(heads_dir)
    build_merit_aggregate_index(bundle_dir, heads_dir=heads_dir, device="cpu")

    reloaded = ReferenceCorpus.load(bundle_dir)
    assert reloaded.merit_index is not None
    assert reloaded.merit_index.embeddings.shape == (40, 384)
    assert reloaded.merit_factors is not None
    assert set(reloaded.merit_factors) == set(FACTORS)
    assert reloaded.merit_calibration is not None
    assert reloaded.merit_calibration.qq_threshold > 0
