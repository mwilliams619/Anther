"""
MERT-v1-330M embedding extraction.
Model: m-a-p/MERT-v1-330M (CC-BY-NC)
Trained on 160k hours of music with masked audio modeling.
No text alignment — embeddings reflect perceptual acoustic content.

Requires 24kHz audio. librosa.load(sr=24000) resamples any MP3/WAV automatically.

Embedding-quality design (Workstream E):
  * All-layer aggregation. MERT is a 24-layer transformer (25 hidden states);
    different layers encode different musical facets (lower ≈ pitch/timbre,
    higher ≈ structure). MERT's own guidance is to aggregate across all layers,
    not use only the last. We mean-pool each layer over time, then mean across
    layers. The layer count is read at runtime so a model swap can't break it.
  * Multi-window. We embed several evenly-spaced fixed-length windows and
    mean-pool them, so the vector represents the whole track — not just its
    (often unrepresentative) opening.
  * One consistent clip length. WINDOW_SECONDS/N_WINDOWS are applied identically
    to corpus and query and recorded in the index config, so indices built with
    different settings can't be silently mixed.
  * Loudness normalization (Workstream I#1) is applied before inference when
    requested, removing the mastering-loudness production confound.

The pure logic (window planning, layer aggregation) is factored out so it is
unit-testable without downloading the 1.3 GB model.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

from .audio import loudness_normalize, loudness_config

SR = 24000  # MERT's required sample rate
WINDOW_SECONDS = 10.0  # length of each analysis window
N_WINDOWS = 3          # evenly-spaced windows mean-pooled per track


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embedding_config(
    layer_aggregation: str = "mean",
    normalize: bool = True,
    window_seconds: float = WINDOW_SECONDS,
    n_windows: int = N_WINDOWS,
) -> dict:
    """Self-describing metadata for the Phase-2 index (Workstream E/G)."""
    return {
        "phase": 2,
        "model": "m-a-p/MERT-v1-330M",
        "sr": SR,
        "layer_aggregation": layer_aggregation,
        "window_seconds": window_seconds,
        "n_windows": n_windows,
        **loudness_config(normalize),
    }


def plan_windows(
    n_samples: int, sr: int = SR,
    window_seconds: float = WINDOW_SECONDS, n_windows: int = N_WINDOWS,
) -> list[tuple[int, int]]:
    """
    Evenly-spaced (start, end) sample indices for ``n_windows`` windows of
    ``window_seconds`` across a clip of ``n_samples``. Windows never run past
    the end. If the clip is shorter than one window, returns a single window
    covering the whole clip. Deterministic — same plan for corpus and query.
    """
    win = int(round(window_seconds * sr))
    if n_samples <= win or n_windows <= 1:
        return [(0, n_samples)]
    last_start = n_samples - win
    starts = np.linspace(0, last_start, n_windows).round().astype(int)
    # De-duplicate in case rounding collapses windows on short clips.
    seen, windows = set(), []
    for s in starts:
        s = int(s)
        if s not in seen:
            seen.add(s)
            windows.append((s, s + win))
    return windows


def aggregate_layers(hidden_states, mode: str = "mean") -> torch.Tensor:
    """
    Aggregate a tuple of per-layer hidden states into one vector per item.

    hidden_states: tuple/list of length (n_layers + 1), each (B, T, H).
    Returns (B, H). ``mode="mean"`` averages the time-mean of every layer;
    ``mode="last"`` uses only the final layer (the old behavior, for comparison).
    """
    n_layers = len(hidden_states)
    if n_layers == 0:
        raise ValueError("no hidden states to aggregate")
    if mode == "last":
        return hidden_states[-1].mean(dim=1)
    if mode == "mean":
        # (n_layers, B, H) → mean over layers
        per_layer = torch.stack([h.mean(dim=1) for h in hidden_states], dim=0)
        return per_layer.mean(dim=0)
    raise ValueError(f"unknown layer_aggregation mode: {mode!r}")


def _embed_windows(
    model, processor, windows: list[np.ndarray], device: str,
    layer_aggregation: str,
) -> np.ndarray:
    """Embed a list of waveform windows and mean-pool them → (H,) vector."""
    inputs = processor(windows, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # (n_windows, H) → mean over windows
    per_window = aggregate_layers(outputs.hidden_states, layer_aggregation)
    return per_window.mean(dim=0).cpu().numpy().astype(np.float32)


def get_embedding(
    model, processor, path: str | Path, device: str,
    layer_aggregation: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Load any MP3/WAV, resample to 24kHz, optionally loudness-normalize, then
    return a 1024-dim MERT embedding aggregated across all layers and several
    windows (see module docstring).
    """
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if normalize:
        y = loudness_normalize(y, SR)
    windows = [y[s:e] for s, e in plan_windows(len(y))]
    return _embed_windows(model, processor, windows, device, layer_aggregation)


def load_mert(device: str | None = None):
    """
    Download (first time) and load MERT-v1-330M from HuggingFace.
    Returns (model, processor, device).
    """
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    if device is None:
        device = _get_device()

    print(f"Loading MERT-v1-330M on {device}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("MERT loaded.")
    return model, processor, device


def embed_batch(
    model, processor, paths: list[str | Path], device: str,
    layer_aggregation: str = "mean",
    normalize: bool = True,
) -> tuple[np.ndarray, list[Path]]:
    """
    Extract embeddings for a list of audio files, one track at a time (each
    track is itself a mini-batch of windows). Skips macOS resource-fork files
    (._*) and any file that fails to load.

    Returns:
      embeddings — (N, 1024) float32 array
      good_paths — paths that were successfully embedded (same order)
    """
    from tqdm import tqdm

    paths = [Path(p) for p in paths if not Path(p).name.startswith("._")]

    embeddings, good_paths = [], []
    for p in tqdm(paths, desc="Embedding"):
        try:
            y, _ = librosa.load(str(p), sr=SR, mono=True)
            if normalize:
                y = loudness_normalize(y, SR)
            windows = [y[s:e] for s, e in plan_windows(len(y))]
            vec = _embed_windows(model, processor, windows, device, layer_aggregation)
        except Exception as e:  # noqa: BLE001 — keep going past bad files
            print(f"\nSkipping {p.name}: {e}")
            continue
        embeddings.append(vec)
        good_paths.append(p)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(embeddings).astype(np.float32), good_paths
