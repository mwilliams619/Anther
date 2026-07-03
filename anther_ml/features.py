"""
Audio feature extraction for Phase 1 (librosa hand-crafted features)
and Phase 2 preparation (mel spectrograms).

Phase 1: extract the same 518-feature vector FMA pre-computed, so personal
MP3s can be projected into the FMA feature space.

**Feature-ordering contract (Workstream B).** FMA's ``features.csv`` blocks
features in *alphabetical* order (chroma_cens, chroma_cqt, chroma_stft, mfcc,
rmse, spectral_bandwidth, spectral_centroid, spectral_contrast,
spectral_rolloff, tonnetz, zcr) with stats also alphabetical (kurtosis, max,
mean, median, min, skew, std). The previous implementation emitted a *different*
order but the same 518 length — so nothing errored while every dimension was
misaligned (chroma compared against MFCC, kurtosis against mean). To make
ordering explicit and self-correcting, ``extract_librosa_features`` now returns
a **labeled ``pandas.Series``** indexed by the exact FMA
``(feature, statistic, number)`` MultiIndex, reindexed to the canonical order.

This mirrors FMA's official ``features.py`` recipe (Defferrard et al., 2017):
full-track native-sample-rate load, CQT-derived chroma, tonnetz from
chroma_cens, STFT-derived spectral features, and MFCC from a log-mel
spectrogram. Aligning the recipe is what makes the round-trip test
(``tests/test_features.py``) reproduce a track's ``features.csv`` row.

Phase 2: mel spectrogram extraction for MERT input (handled in embedding.py).
"""

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# FMA processes the whole track at its native sample rate (librosa.load sr=None).
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 20

# Alphabetical, exactly as FMA emits them. Sizes are the number of sub-bands.
_FEATURE_SIZES = {
    "chroma_stft": 12,
    "chroma_cqt": 12,
    "chroma_cens": 12,
    "tonnetz": 6,
    "mfcc": 20,
    "rmse": 1,
    "zcr": 1,
    "spectral_centroid": 1,
    "spectral_bandwidth": 1,
    "spectral_contrast": 7,
    "spectral_rolloff": 1,
}
_MOMENTS = ("mean", "std", "skew", "kurtosis", "median", "min", "max")


@lru_cache(maxsize=1)
def fma_feature_columns() -> pd.MultiIndex:
    """
    The canonical FMA ``features.csv`` column MultiIndex, built programmatically
    and sorted the same way FMA sorts it. Verified to equal the real CSV's 518
    columns exactly. Cached — this is the single source of truth for ordering.
    """
    cols = []
    for name, size in _FEATURE_SIZES.items():
        for moment in _MOMENTS:
            cols.extend(
                (name, moment, f"{i + 1:02d}") for i in range(size)
            )
    idx = pd.MultiIndex.from_tuples(
        cols, names=("feature", "statistics", "number")
    )
    return idx.sort_values()


def _feature_stats(name: str, values: np.ndarray, out: dict) -> None:
    """
    Populate ``out`` with the 7 summary stats of a (n_bands, n_frames) matrix,
    keyed by (name, statistic, '01'..). Matches FMA's ``feature_stats``.
    """
    if values.ndim == 1:
        values = values[np.newaxis, :]
    stat_vectors = {
        "mean": np.mean(values, axis=1),
        "std": np.std(values, axis=1),
        "skew": skew(values, axis=1),
        "kurtosis": kurtosis(values, axis=1),
        "median": np.median(values, axis=1),
        "min": np.min(values, axis=1),
        "max": np.max(values, axis=1),
    }
    for stat, vec in stat_vectors.items():
        for i, val in enumerate(np.atleast_1d(vec)):
            out[(name, stat, f"{i + 1:02d}")] = float(val)


def extract_librosa_features(
    path: str | Path, sr: int | None = None
) -> pd.Series:
    """
    Extract the 518-dim FMA-schema feature vector for one audio file.

    Returns a ``pandas.Series`` indexed by the FMA ``(feature, statistic,
    number)`` MultiIndex and reindexed to the canonical column order — so it can
    be projected into the FMA feature space with no positional ambiguity. Use
    ``.values`` (or ``.to_numpy()``) to get the ordered array.

    ``sr=None`` (default) preserves FMA's native-sample-rate, full-track
    processing. Pass an int only if you deliberately want to resample.
    """
    x, sr = librosa.load(str(path), sr=sr, mono=True)

    out: dict = {}

    # Zero-crossing rate
    _feature_stats(
        "zcr",
        librosa.feature.zero_crossing_rate(
            x, frame_length=N_FFT, hop_length=HOP_LENGTH
        ),
        out,
    )

    # Constant-Q transform → CQT-based chroma + tonnetz (from chroma_cens)
    cqt = np.abs(
        librosa.cqt(
            y=x, sr=sr, hop_length=HOP_LENGTH,
            bins_per_octave=12, n_bins=7 * 12, tuning=None,
        )
    )
    _feature_stats(
        "chroma_cqt",
        librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7),
        out,
    )
    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    _feature_stats("chroma_cens", chroma_cens, out)
    _feature_stats(
        "tonnetz", librosa.feature.tonnetz(chroma=chroma_cens), out
    )
    del cqt

    # STFT-derived spectral features
    stft = np.abs(librosa.stft(y=x, n_fft=N_FFT, hop_length=HOP_LENGTH))
    _feature_stats(
        "chroma_stft",
        librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12),
        out,
    )
    _feature_stats("rmse", librosa.feature.rms(S=stft), out)
    _feature_stats(
        "spectral_centroid", librosa.feature.spectral_centroid(S=stft), out
    )
    _feature_stats(
        "spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft), out
    )
    _feature_stats(
        "spectral_contrast",
        librosa.feature.spectral_contrast(S=stft, n_bands=6),
        out,
    )
    _feature_stats(
        "spectral_rolloff", librosa.feature.spectral_rolloff(S=stft), out
    )

    # MFCC from a log-power mel spectrogram
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    del stft
    _feature_stats(
        "mfcc",
        librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC),
        out,
    )

    series = pd.Series(out)
    series.index = pd.MultiIndex.from_tuples(
        series.index, names=("feature", "statistics", "number")
    )
    # Reindex to canonical order — self-correcting regardless of insertion order.
    return series.reindex(fma_feature_columns())


def align_to_corpus(
    query: pd.Series, corpus_columns: pd.Index | pd.MultiIndex
) -> np.ndarray:
    """
    Reindex a query feature Series onto the corpus column order and return the
    ordered numpy vector, asserting the axes align exactly first (Workstream B).
    A silent misalignment here is the class of bug that made Phase-1 uploads
    meaningless, so fail loudly instead.
    """
    aligned = query.reindex(corpus_columns)
    assert list(aligned.index) == list(corpus_columns), (
        "query features do not align with corpus columns — "
        "feature-ordering contract violated"
    )
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise ValueError(
            f"query is missing {len(missing)} feature(s) present in the "
            f"corpus, e.g. {missing[:3]}"
        )
    return aligned.to_numpy(dtype=np.float32)


# Pure-production feature blocks (Workstream I#2). RMS energy is loudness by
# definition; absolute spectral-magnitude descriptors (centroid/bandwidth/
# rolloff in Hz) encode mix brightness/mastering more than musical content.
# Scale-invariant descriptors (chroma, MFCC shape, tonnetz) are preferred.
LOUDNESS_FEATURE_BLOCKS = (
    "rmse",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
)


def feature_weight_vector(
    columns: pd.MultiIndex,
    downweight: tuple[str, ...] = LOUDNESS_FEATURE_BLOCKS,
    factor: float = 0.0,
) -> np.ndarray:
    """
    Weight vector aligned to ``columns`` for down-weighting (or dropping, with
    ``factor=0``) production-driven feature blocks before similarity/clustering
    (Workstream I#2). Multiply standardized features by this to attenuate the
    loudness/brightness confound while keeping scale-invariant descriptors.
    """
    feats = columns.get_level_values("feature")
    return np.where(feats.isin(downweight), factor, 1.0).astype(np.float32)


def drop_loudness_features(
    df: pd.DataFrame, blocks: tuple[str, ...] = LOUDNESS_FEATURE_BLOCKS
) -> pd.DataFrame:
    """
    Return ``df`` (FMA-schema columns) with the given production feature blocks
    removed entirely — the hard version of ``feature_weight_vector``. Use the
    Workstream F neighbor audit to decide whether dropping vs. down-weighting
    gives more musically-coherent neighbors.
    """
    feats = df.columns.get_level_values("feature")
    return df.loc[:, ~feats.isin(blocks)]


def extract_mel_spectrogram(
    path: str | Path, sr: int = 22050, duration: float | None = 30.0
) -> np.ndarray:
    """
    Extract log-mel spectrogram for visualization or manual CNN use.
    Shape: (128, time_frames) — approximately (128, 1292) for 30s clips.

    Note: MERT embedding uses its own internal preprocessing via
    Wav2Vec2FeatureExtractor — use embedding.py for that path instead.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(S, ref=np.max)
