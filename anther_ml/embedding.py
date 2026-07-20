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

# MERIT factor-head backbone (additive; the 1024-d MERT path is untouched).
# MERIT (Herremans et al., arXiv:2605.27346) freezes this same MERT-v1-330M and
# reads a fixed subset of hidden states — the time-mean of layers {3,4,5,6,23}
# concatenated into a 5120-d vector — which its three projection heads
# (melody / timbre / rhythm) map to 128-d unit vectors. hidden_states[0] is the
# embedding output and [1..24] the 24 transformer layers, so these indices are
# the transformer-layer numbers used in the paper. Selected via
# ``layer_aggregation="merit_concat"`` — see anther_ml/merit.py.
MERIT_LAYERS = (3, 4, 5, 6, 23)


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
    ``mode="mean"`` averages the time-mean of every layer → (B, H); ``mode="last"``
    uses only the final layer (the old behavior, for comparison) → (B, H).
    ``mode="merit_concat"`` time-means the MERIT backbone layers ``MERIT_LAYERS``
    and concatenates them → (B, H*len(MERIT_LAYERS)) — the 5120-d input the MERIT
    factor heads expect. This path is additive: the 1024-d ``mean``/``last``
    outputs are unchanged.
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
    if mode == "merit_concat":
        need = max(MERIT_LAYERS)
        if n_layers <= need:
            raise ValueError(
                f"merit_concat needs hidden states through layer {need}, "
                f"got only {n_layers}"
            )
        # time-mean each MERIT layer → concat along feature dim → (B, H*k)
        return torch.cat(
            [hidden_states[l].mean(dim=1) for l in MERIT_LAYERS], dim=-1
        )
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


def _embed_windows_dual(
    model, processor, windows: list[np.ndarray], device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-track counterpart to ``embed_tracks_batched_dual``: one forward
    pass over ``windows`` yields both the 1024-d MERT mean-pooled vector and
    the 5120-d MERIT backbone (``merit_concat``), mean-pooled over windows.
    Returns ``(mert_1024, merit_backbone_5120)``, each ``(H,)`` float32.
    """
    inputs = processor(windows, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    mert = aggregate_layers(outputs.hidden_states, "mean").mean(dim=0).cpu().numpy()
    merit = aggregate_layers(outputs.hidden_states, "merit_concat").mean(dim=0).cpu().numpy()
    return mert.astype(np.float32), merit.astype(np.float32)


def prepare_waveform(item: dict, normalize: bool = True) -> np.ndarray:
    """
    Turn a source item into a mono float32 waveform at ``SR``, loudness-normalized
    when requested — the CPU/IO half of embedding, split out so it can run in the
    prefetch pool (Tier 1B) while the GPU stays busy.

    Accepts an in-memory ``item['audio']`` (resampled if ``item['sr']`` differs)
    or an on-disk ``item['path']``.
    """
    audio = item.get("audio")
    if audio is not None:
        y = np.asarray(audio, dtype=np.float32)
        sr = int(item.get("sr", SR))
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    elif item.get("path"):
        y, _ = librosa.load(str(item["path"]), sr=SR, mono=True)
    else:
        raise ValueError(f"item {item.get('id')!r} has neither path nor audio")
    if normalize:
        y = loudness_normalize(y, SR)
    return y.astype(np.float32)


def embed_tracks_batched(
    model, processor, waveforms: list[np.ndarray], device: str,
    layer_aggregation: str = "mean",
    batch_windows: int = 32,
    use_fp16: bool | None = None,
) -> np.ndarray:
    """
    Embed several prepared waveforms in shared GPU forward passes → ``(N, H)``.

    Windows from all tracks are pooled into batches of up to ``batch_windows``,
    but **only equal-length windows share a forward pass**, so no padding is ever
    introduced — each track's pooled vector is identical to the one-track-at-a-time
    path (``_embed_windows``) up to fp16 rounding. This is the invariant the
    corpus↔query recipe depends on (``docs/invariants.md``): batching must not
    change the per-track result.

    On CUDA the forward runs under ``inference_mode`` + fp16 ``autocast`` (MERT is
    a frozen feature extractor, so fp16 perturbs vectors ~1e-2 cosine while roughly
    halving VRAM). Set ``use_fp16=False`` to force fp32.
    """
    from collections import defaultdict

    if use_fp16 is None:
        use_fp16 = device == "cuda"
    if not waveforms:
        return np.empty((0, 0), dtype=np.float32)

    # Flat list of (track_index, window) across all tracks, plus per-track slots.
    flat: list[tuple[int, np.ndarray]] = []
    for ti, y in enumerate(waveforms):
        for s, e in plan_windows(len(y)):
            flat.append((ti, y[s:e]))
    per_track_windows: list[list[np.ndarray]] = [[] for _ in waveforms]

    # Group window positions by exact length so a batch never needs padding.
    by_length: dict[int, list[int]] = defaultdict(list)
    for fi, (_, win) in enumerate(flat):
        by_length[len(win)].append(fi)

    for _, positions in by_length.items():
        for start in range(0, len(positions), batch_windows):
            chunk = positions[start : start + batch_windows]
            wins = [flat[fi][1] for fi in chunk]
            inputs = processor(
                wins, sampling_rate=SR, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs, output_hidden_states=True)
                else:
                    outputs = model(**inputs, output_hidden_states=True)
            pooled = aggregate_layers(outputs.hidden_states, layer_aggregation)
            pooled = pooled.float().cpu().numpy().astype(np.float32)  # (B, H)
            for j, fi in enumerate(chunk):
                per_track_windows[flat[fi][0]].append(pooled[j])

    return np.vstack(
        [np.mean(ws, axis=0) for ws in per_track_windows]
    ).astype(np.float32)


def embed_tracks_batched_dual(
    model, processor, waveforms: list[np.ndarray], device: str,
    batch_windows: int = 32,
    use_fp16: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like :func:`embed_tracks_batched`, but pulls **both** the standard 1024-d
    mean-pooled embedding and the 5120-d MERIT backbone out of the *same*
    forward pass — one GPU pass per waveform batch instead of two, since both
    are just different reductions over the same ``hidden_states``.

    Returns ``(mert_mean, merit_backbone)``, each ``(N, ·)`` in input order.
    The 1024-d output is byte-identical to ``embed_tracks_batched(...,
    layer_aggregation="mean")`` — same windows, same batching, same fp16
    behavior — so it can freely replace it as the corpus's raw MERT-1024
    space.
    """
    from collections import defaultdict

    if use_fp16 is None:
        use_fp16 = device == "cuda"
    if not waveforms:
        return (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32))

    flat: list[tuple[int, np.ndarray]] = []
    for ti, y in enumerate(waveforms):
        for s, e in plan_windows(len(y)):
            flat.append((ti, y[s:e]))
    per_track_mean: list[list[np.ndarray]] = [[] for _ in waveforms]
    per_track_backbone: list[list[np.ndarray]] = [[] for _ in waveforms]

    by_length: dict[int, list[int]] = defaultdict(list)
    for fi, (_, win) in enumerate(flat):
        by_length[len(win)].append(fi)

    for _, positions in by_length.items():
        for start in range(0, len(positions), batch_windows):
            chunk = positions[start : start + batch_windows]
            wins = [flat[fi][1] for fi in chunk]
            inputs = processor(
                wins, sampling_rate=SR, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs, output_hidden_states=True)
                else:
                    outputs = model(**inputs, output_hidden_states=True)
            pooled_mean = aggregate_layers(outputs.hidden_states, "mean")
            pooled_mean = pooled_mean.float().cpu().numpy().astype(np.float32)
            pooled_backbone = aggregate_layers(outputs.hidden_states, "merit_concat")
            pooled_backbone = pooled_backbone.float().cpu().numpy().astype(np.float32)
            for j, fi in enumerate(chunk):
                ti = flat[fi][0]
                per_track_mean[ti].append(pooled_mean[j])
                per_track_backbone[ti].append(pooled_backbone[j])

    mert_mean = np.vstack(
        [np.mean(ws, axis=0) for ws in per_track_mean]
    ).astype(np.float32)
    merit_backbone = np.vstack(
        [np.mean(ws, axis=0) for ws in per_track_backbone]
    ).astype(np.float32)
    return mert_mean, merit_backbone


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


def get_embedding_dual(
    model, processor, path: str | Path, device: str,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query-time counterpart to ``embed_tracks_batched_dual``: load one audio
    file and return both the 1024-d MERT vector (``layer_aggregation="mean"``)
    and the 5120-d MERIT backbone (``merit_concat``) from a single forward
    pass, so out-of-corpus queries (uploads, playlist imports) can be scored
    against the MERIT-aggregate index alongside the existing MERT index.
    """
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if normalize:
        y = loudness_normalize(y, SR)
    windows = [y[s:e] for s, e in plan_windows(len(y))]
    return _embed_windows_dual(model, processor, windows, device)


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
