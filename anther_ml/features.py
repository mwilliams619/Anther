"""
Audio feature extraction for Phase 1 (librosa hand-crafted features)
and Phase 2 preparation (mel spectrograms).

Phase 1: extract the same 568-feature vector FMA pre-computed,
so personal MP3s can be projected into the FMA feature space.

Phase 2: mel spectrogram extraction for MERT input (handled in embedding.py).
"""

from pathlib import Path

import librosa
import numpy as np

# Match FMA's feature extraction parameters
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20


def _stats(feature: np.ndarray) -> list[float]:
    """Compute 7 summary statistics for a feature matrix (mean, std, skew, kurtosis, median, min, max)."""
    from scipy.stats import kurtosis, skew
    if feature.ndim == 1:
        feature = feature[np.newaxis, :]
    result = []
    for row in feature:
        result.extend([
            np.mean(row), np.std(row), skew(row), kurtosis(row),
            np.median(row), np.min(row), np.max(row)
        ])
    return result


def extract_librosa_features(path: str | Path, sr: int = SR) -> np.ndarray:
    """
    Extract ~560 hand-crafted audio features matching FMA's features.csv schema.
    Use this to project a personal MP3 into the Phase 1 FMA feature space.

    Returns 1D numpy array.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=30.0)

    features = []

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features.extend(_stats(zcr))

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(_stats(chroma_stft))

    # Chroma CQT
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(_stats(chroma_cqt))

    # Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(_stats(chroma_cens))

    # Tonnetz
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    features.extend(_stats(tonnetz))

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    features.extend(_stats(mfcc))

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features.extend(_stats(rms))

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(_stats(spec_centroid))

    # Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(_stats(spec_bandwidth))

    # Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(_stats(spec_contrast))

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(_stats(spec_rolloff))

    return np.array(features, dtype=np.float32)


def extract_mel_spectrogram(path: str | Path, sr: int = SR) -> np.ndarray:
    """
    Extract log-mel spectrogram for visualization or manual CNN use.
    Shape: (N_MELS, time_frames) — approximately (128, 1292) for 30s clips.

    Note: MERT embedding uses its own internal preprocessing via
    Wav2Vec2FeatureExtractor — use embedding.py for that path instead.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=30.0)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(S, ref=np.max)
