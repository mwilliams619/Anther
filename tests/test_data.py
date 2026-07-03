"""Workstream C — corpus sizing (ordered-categorical subset filter)."""
import pandas as pd
import pytest

from anther_ml.data import load_fma_features, SUBSET_ORDER


def _make_tracks(subsets):
    """Build a minimal tracks.csv-shaped frame with a ('set','subset') col."""
    cols = pd.MultiIndex.from_tuples([("set", "subset")])
    return pd.DataFrame(
        {("set", "subset"): subsets},
        index=range(len(subsets)),
        columns=cols,
    )


@pytest.fixture
def fake_fma(tmp_path, monkeypatch):
    """
    A tiny in-memory FMA: 3 'small', 2 'medium', 4 'large' tracks, each with
    a distinct feature row. Patches the CSV readers so no real FMA is needed.
    """
    subsets = (["small"] * 3) + (["medium"] * 2) + (["large"] * 4)
    tracks = _make_tracks(subsets)

    feat_cols = pd.MultiIndex.from_tuples(
        [("chroma_cens", "mean", "01"), ("mfcc", "mean", "01")]
    )
    features = pd.DataFrame(
        [[float(i), float(i) * 2] for i in range(len(subsets))],
        index=range(len(subsets)),
        columns=feat_cols,
    )

    import anther_ml.data as data_mod

    def fake_read_csv(path, *args, **kwargs):
        return features if "features" in str(path) else tracks

    monkeypatch.setattr(data_mod.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(data_mod, "load_fma_tracks", lambda d: tracks)
    return tmp_path


def test_small_subset_is_not_everything(fake_fma):
    """The core bug: string '<=' matched all tracks; categorical must not."""
    small = load_fma_features(fake_fma, subset="small")
    assert len(small) == 3  # only the 'small'-labeled rows


def test_subsets_are_nested(fake_fma):
    """small ⊂ medium ⊂ large; each level is cumulative."""
    small = load_fma_features(fake_fma, subset="small")
    medium = load_fma_features(fake_fma, subset="medium")
    large = load_fma_features(fake_fma, subset="large")
    assert len(small) == 3
    assert len(medium) == 5  # small + medium
    assert len(large) == 9   # everything
    assert set(small.index).issubset(set(medium.index))
    assert set(medium.index).issubset(set(large.index))


def test_no_subset_returns_all(fake_fma):
    allrows = load_fma_features(fake_fma, subset=None)
    assert len(allrows) == 9


def test_bad_subset_raises(fake_fma):
    with pytest.raises(ValueError):
        load_fma_features(fake_fma, subset="tiny")


def test_subset_order_constant():
    assert SUBSET_ORDER == ["small", "medium", "large"]
