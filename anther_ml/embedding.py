"""
MERT-v1-330M embedding extraction.
Model: m-a-p/MERT-v1-330M (CC-BY-NC)
Trained on 160k hours of music with masked audio modeling.
No text alignment — embeddings reflect perceptual acoustic content.

Requires 24kHz audio. librosa.load(sr=24000) resamples any MP3 or WAV automatically.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

SR = 24000  # MERT's required sample rate
MAX_DURATION = 30.0  # seconds; longer files are truncated


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    print("MERT loaded.")
    return model, processor, device


def get_embedding(
    model, processor, path: str | Path, device: str
) -> np.ndarray:
    """
    Load any MP3 or WAV, resample to 24kHz, and return a 1024-dim
    MERT embedding (mean-pooled last hidden state).
    """
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=MAX_DURATION)

    inputs = processor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Mean-pool across time dimension → (1024,)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def embed_batch(
    model, processor, paths: list[str | Path], device: str,
    batch_size: int = 8
) -> np.ndarray:
    """
    Extract embeddings for a list of audio files.
    Returns array of shape (N, 1024).
    Processes in batches to avoid OOM on large libraries.
    """
    from tqdm import tqdm

    embeddings = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
        batch_paths = paths[i : i + batch_size]
        batch_audio = []
        for p in batch_paths:
            y, _ = librosa.load(str(p), sr=SR, mono=True, duration=MAX_DURATION)
            batch_audio.append(y)

        inputs = processor(
            batch_audio, sampling_rate=SR, return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_emb)

    return np.vstack(embeddings).astype(np.float32)
