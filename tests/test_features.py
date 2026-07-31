"""Workstream B — feature-ordering contract + FMA round-trip fidelity."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from anther_ml.features import (
    extract_librosa_features,
    fma_feature_columns,
    align_to_corpus,
    feature_weight_vector,
    drop_loudness_features,
    LOUDNESS_FEATURE_BLOCKS,
)


_PERSONAL = Path("data/audio/personal")


@pytest.fixture(scope="module")
def synth_wav(tmp_path_factory):
    """
    A short real-music clip if any personal MP3 is on disk (truncated to 6s so
    the full-track CQT stays fast), else a synthetic fallback.

    Real audio is preferred because clean synthetic tones can produce
    exactly-constant chroma rows (undefined skew/kurtosis) that real music never
    has — an artifact of the tone, not of the extractor.
    """
    import librosa

    out = tmp_path_factory.mktemp("audio") / "clip.wav"
    real = sorted(
        p for p in _PERSONAL.glob("*.mp3") if not p.name.startswith("._")
    ) if _PERSONAL.exists() else []
    if real:
        y, sr = librosa.load(str(real[0]), sr=None, mono=True, duration=6.0)
        sf.write(out, y, sr)
        return out

    sr = 22050
    rng = np.random.default_rng(0)
    notes = [220.0 * 2 ** (k / 12) for k in range(12)]
    y = np.zeros(int(sr * 6.0), dtype=np.float32)
    seg = len(y) // len(notes)
    for i, f0 in enumerate(notes):
        s = i * seg
        tt = np.arange(seg) / sr
        y[s:s + seg] = 0.5 * np.sin(2 * np.pi * f0 * tt) + 0.25 * np.sin(
            2 * np.pi * 2 * f0 * tt
        )
    y += 0.02 * rng.standard_normal(len(y)).astype(np.float32)
    sf.write(out, y, sr)
    return out


def test_canonical_columns_match_real_csv_if_present():
    """fma_feature_columns() is the single source of truth; if the real FMA
    CSV is available, they must be identical."""
    cols = fma_feature_columns()
    assert len(cols) == 518
    assert cols.names == ["feature", "statistics", "number"]
    csv = Path("data/fma_metadata/features.csv")
    if csv.exists():
        real = pd.read_csv(csv, index_col=0, header=[0, 1, 2], nrows=1).columns
        assert list(cols) == list(real)


def test_extract_block_sizes(synth_wav):
    """Each feature block has the right number of sub-bands × 7 stats."""
    feats = extract_librosa_features(synth_wav)
    counts = feats.index.get_level_values("feature").value_counts()
    assert counts["mfcc"] == 20 * 7
    assert counts["chroma_cens"] == 12 * 7
    assert counts["tonnetz"] == 6 * 7
    assert counts["spectral_contrast"] == 7 * 7
    assert counts["zcr"] == 1 * 7


def test_align_to_corpus_missing_feature_raises(synth_wav):
    feats = extract_librosa_features(synth_wav)
    corpus_cols = feats.index
    incomplete = feats.iloc[:-1]  # drop one column
    with pytest.raises(ValueError):
        align_to_corpus(incomplete, corpus_cols)


def test_feature_weight_vector_zeros_loudness_blocks():
    cols = fma_feature_columns()
    w = feature_weight_vector(cols, factor=0.0)
    assert w.shape[0] == len(cols)
    feats = cols.get_level_values("feature")
    # Loudness blocks weighted 0, everything else 1.
    assert (w[feats.isin(LOUDNESS_FEATURE_BLOCKS)] == 0.0).all()
    assert (w[~feats.isin(LOUDNESS_FEATURE_BLOCKS)] == 1.0).all()
    # Musical descriptors survive.
    assert (w[feats == "chroma_cens"] == 1.0).all()
    assert (w[feats == "mfcc"] == 1.0).all()


def test_feature_weight_vector_downweight_factor():
    cols = fma_feature_columns()
    w = feature_weight_vector(cols, factor=0.25)
    feats = cols.get_level_values("feature")
    assert (w[feats == "rmse"] == 0.25).all()


def test_drop_loudness_features_removes_blocks():
    cols = fma_feature_columns()
    df = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
    dropped = drop_loudness_features(df)
    remaining = set(dropped.columns.get_level_values("feature"))
    assert remaining.isdisjoint(set(LOUDNESS_FEATURE_BLOCKS))
    assert "chroma_cens" in remaining
    assert "mfcc" in remaining
    assert dropped.shape[1] < df.shape[1]


@pytest.mark.skipif(
    not os.environ.get("FMA_ROUNDTRIP_AUDIO"),
    reason="set FMA_ROUNDTRIP_AUDIO=<path to an FMA mp3> and FMA_ROUNDTRIP_TID "
    "to run the param-fidelity round-trip",
)
def test_roundtrip_matches_fma_features_csv():
    """
    Extract features for one FMA track whose audio we have, reindex, and
    confirm the vector matches that track's features.csv row within tolerance.
    Validates ordering AND that librosa params match FMA's (Workstream B note).
    """
    audio = os.environ["FMA_ROUNDTRIP_AUDIO"]
    tid = int(os.environ["FMA_ROUNDTRIP_TID"])
    csv = pd.read_csv(
        "data/fma_metadata/features.csv", index_col=0, header=[0, 1, 2]
    )
    reference = csv.loc[tid]
    got = extract_librosa_features(audio).reindex(reference.index)
    # Correlation is a robust ordering/fidelity check; absolute tolerance is
    # loose because minor librosa-version drift shifts magnitudes slightly.
    corr = np.corrcoef(got.to_numpy(), reference.to_numpy())[0, 1]
    assert corr > 0.95, f"round-trip correlation only {corr:.3f}"
