"""Tests for anther_ml.corpus.extend helpers."""
import numpy as np
import pytest

from anther_ml.corpus.extend import _append_npy_streaming


# (n_old, n_new): the first case is the real 100k+billboard bundle shape that
# regressed -- 180644 % 8192 == 420, so the last chunk of `old` is short.
@pytest.mark.parametrize(
    "n_old,n_new",
    [(180644, 20), (420, 20), (8192, 5), (16384, 8192), (1, 1), (9000, 3)],
)
def test_append_npy_streaming_preserves_rows(tmp_path, n_old, n_new):
    old = np.arange(n_old * 4, dtype=np.float32).reshape(n_old, 4)
    new = np.full((n_new, 4), -1.0, dtype=np.float32)
    path = tmp_path / "factor_mel.npy"
    np.save(path, old)

    _append_npy_streaming(path, path, new)

    got = np.load(path)
    assert got.shape == (n_old + n_new, 4)
    assert np.array_equal(got[:n_old], old)
    assert np.array_equal(got[n_old:], new)
    assert not list(tmp_path.glob("*.tmp.npy"))


def test_append_npy_streaming_writes_to_new_path(tmp_path):
    old = np.ones((100, 4), dtype=np.float32)
    np.save(tmp_path / "in.npy", old)

    _append_npy_streaming(tmp_path / "in.npy", tmp_path / "out.npy",
                          np.zeros((7, 4), dtype=np.float32))

    assert np.load(tmp_path / "in.npy").shape == (100, 4)
    assert np.load(tmp_path / "out.npy").shape == (107, 4)


def test_append_npy_streaming_rejects_shape_mismatch(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot append"):
        _append_npy_streaming(tmp_path / "a.npy", tmp_path / "a.npy",
                              np.ones((3, 5), dtype=np.float32))
