"""
Audio loading with optional loudness normalization (Workstream I#1).

Two songs can land near each other in embedding space because they share
*production* characteristics — chiefly mastering loudness — rather than because
they are musically similar. This is the music-similarity analogue of a
single-cell "batch effect". The cheapest, highest-value correction is to
loudness-normalize every file (corpus and query alike) to a common target
before feature extraction or MERT inference.

We use EBU R128 integrated loudness (via ``pyloudnorm``), the same standard
streaming platforms normalize to. The normalization method and target are
recorded in the index metadata (see ``SongIndex(config=...)``) so a corpus and
a query can never be silently compared under different normalization.

Note on Phase 1: the FMA librosa corpus was pre-computed from un-normalized
audio and cannot be re-normalized, so Phase-1 queries stay un-normalized to
match it (Workstream I#2 down-weights loudness features instead). Loudness
normalization is applied on the Phase-2 (MERT) path, where we control both the
corpus and the query.
"""

from pathlib import Path

import librosa
import numpy as np

DEFAULT_TARGET_LUFS = -14.0  # streaming-platform reference (Spotify/YouTube)


def loudness_normalize(
    y: np.ndarray, sr: int, target_lufs: float = DEFAULT_TARGET_LUFS
) -> np.ndarray:
    """
    Normalize a mono waveform to ``target_lufs`` integrated loudness (EBU R128).

    Falls back to a no-op for signals that are silent or too short for a valid
    loudness measurement (pyloudnorm needs ≥ ~0.4s), so it never returns NaN/inf.
    Peaks are clipped to [-1, 1] after gain to avoid hard distortion.
    """
    import pyloudnorm as pyln

    y = np.asarray(y, dtype=np.float32)
    # pyloudnorm's block size is 400ms; shorter/near-silent signals can't be
    # measured reliably — leave them untouched.
    if y.size < int(0.4 * sr) or not np.any(np.abs(y) > 1e-6):
        return y

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    if not np.isfinite(loudness):
        return y

    normalized = pyln.normalize.loudness(y, loudness, target_lufs)
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)


def load_audio(
    path: str | Path,
    sr: int,
    mono: bool = True,
    duration: float | None = None,
    offset: float = 0.0,
    normalize: bool = False,
    target_lufs: float = DEFAULT_TARGET_LUFS,
) -> np.ndarray:
    """
    Load audio at ``sr`` and (optionally) loudness-normalize it. Returns the
    waveform only (sample rate is fixed by ``sr``).
    """
    y, _ = librosa.load(
        str(path), sr=sr, mono=mono, duration=duration, offset=offset
    )
    if normalize:
        y = loudness_normalize(y, sr, target_lufs=target_lufs)
    return y


def loudness_config(normalize: bool, target_lufs: float = DEFAULT_TARGET_LUFS) -> dict:
    """Metadata block describing the loudness normalization, for index config."""
    return {
        "loudness_normalize": bool(normalize),
        "target_lufs": target_lufs if normalize else None,
        "loudness_standard": "EBU R128" if normalize else None,
    }
