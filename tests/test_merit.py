"""Tests for the MERIT factor-head additions (embedding.merit_concat + merit.py).

The heavy path (MERT + real head weights) is exercised in the encode step; here
we verify the pure logic — aggregation dims, additivity of the 1024-d path, head
projection shape/normalization, and aggregate synthesis — without the 1.3 GB model.
"""

import numpy as np
import pytest
import torch

from anther_ml import embedding as E
from anther_ml import merit as M


def _fake_hidden_states(b=2, t=40, h=1024, n=25):
    return tuple(torch.randn(b, t, h) for _ in range(n))


def test_merit_concat_dims_and_layers():
    hs = _fake_hidden_states()
    out = E.aggregate_layers(hs, "merit_concat")
    assert out.shape == (2, 1024 * len(E.MERIT_LAYERS))  # 5120
    expect = torch.cat([hs[l].mean(dim=1) for l in E.MERIT_LAYERS], dim=-1)
    assert torch.allclose(out, expect)


def test_1024_path_unchanged_by_merit_addition():
    hs = _fake_hidden_states()
    per_layer = torch.stack([x.mean(dim=1) for x in hs], dim=0)
    assert torch.allclose(E.aggregate_layers(hs, "mean"), per_layer.mean(dim=0))
    assert torch.allclose(E.aggregate_layers(hs, "last"), hs[-1].mean(dim=1))


def test_merit_concat_needs_enough_layers():
    with pytest.raises(ValueError):
        E.aggregate_layers(_fake_hidden_states(n=10), "merit_concat")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        E.aggregate_layers(_fake_hidden_states(), "bogus")


def test_projection_head_shape_and_norm():
    head = M.ProjectionHead(5120, 512, 128)
    out = head(torch.randn(4, 5120))
    assert out.shape == (4, 128)
    norms = out.norm(dim=-1).detach().numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_project_batch_and_single():
    heads = {f: M.ProjectionHead(5120, 512, 128) for f in M.FACTORS}
    single = M.project(np.random.randn(5120).astype(np.float32), heads)
    batch = M.project(np.random.randn(3, 5120).astype(np.float32), heads)
    for f in M.FACTORS:
        assert single[f].shape == (128,)
        assert batch[f].shape == (3, 128)
        assert abs(np.linalg.norm(single[f]) - 1.0) < 1e-4


def test_aggregate_similarity_weighting():
    sims = {
        "mel": np.array([1.0, 0.0]),
        "rhy": np.array([0.0, 1.0]),
        "tim": np.array([0.5, 0.5]),
    }
    # equal weights → simple mean
    agg = M.aggregate_similarity(sims)
    assert np.allclose(agg, [0.5, 0.5])
    # melody-heavy preset
    agg_mel = M.aggregate_similarity(sims, {"mel": 8.0, "rhy": 1.0, "tim": 1.0})
    assert agg_mel[0] > agg_mel[1]  # first item is melody-similar


def test_merit_config_shape():
    cfg = M.merit_config()
    assert cfg["backbone_dim"] == 5120
    assert cfg["factor_dim"] == 128
    assert cfg["merit_layers"] == list(E.MERIT_LAYERS)
