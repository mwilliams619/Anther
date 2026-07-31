"""Workstream I#1 — loudness normalization."""
import numpy as np
import pytest

from anther_ml.audio import loudness_normalize, loudness_config, DEFAULT_TARGET_LUFS


def _tone(sr, seconds, amp):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _measure(y, sr):
    import pyloudnorm as pyln
    return pyln.Meter(sr).integrated_loudness(y)


def test_normalizes_to_target_loudness():
    sr = 24000
    loud = _tone(sr, 3.0, 0.9)
    quiet = _tone(sr, 3.0, 0.05)
    ln = loudness_normalize(loud, sr)
    qn = loudness_normalize(quiet, sr)
    # Both clips end up near the same integrated loudness regardless of input gain.
    assert _measure(ln, sr) == pytest.approx(DEFAULT_TARGET_LUFS, abs=1.0)
    assert _measure(qn, sr) == pytest.approx(DEFAULT_TARGET_LUFS, abs=1.0)


def test_output_is_finite_and_bounded():
    sr = 24000
    y = _tone(sr, 3.0, 0.9)
    out = loudness_normalize(y, sr)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_short_signal_is_noop():
    sr = 24000
    y = _tone(sr, 0.1, 0.5)  # below the 400ms measurement window
    out = loudness_normalize(y, sr)
    np.testing.assert_array_equal(out, y)


def test_silence_is_noop():
    sr = 24000
    y = np.zeros(sr, dtype=np.float32)
    out = loudness_normalize(y, sr)
    np.testing.assert_array_equal(out, y)


def test_loudness_config_metadata():
    on = loudness_config(True)
    assert on["loudness_normalize"] is True
    assert on["target_lufs"] == DEFAULT_TARGET_LUFS
    assert on["loudness_standard"] == "EBU R128"
    off = loudness_config(False)
    assert off["loudness_normalize"] is False
    assert off["target_lufs"] is None
