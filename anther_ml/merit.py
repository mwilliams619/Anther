"""
MERIT disentangled factor heads (additive layer over frozen MERT).

MERIT (Herremans et al., *MERIT: Multi-view Embeddings for Retrieval of
audIo Tracks*, arXiv:2605.27346; heads: HF ``amaai-lab/merit``) keeps the same
frozen ``m-a-p/MERT-v1-330M`` backbone Anther already uses and adds three small
projection heads — **melody**, **rhythm**, **timbre** — each mapping a 5120-d
backbone vector (the time-mean of MERT layers {3,4,5,6,23} concatenated; see
``embedding.MERIT_LAYERS`` / ``aggregate_layers(mode="merit_concat")``) to a
128-d L2-normalized vector. Similarity within a factor is plain cosine.

There is **no native aggregate** — MERIT's thesis is that a single scalar mixes
perceptual axes, so it deliberately exposes three. An aggregate score, when
wanted, is synthesized downstream as a weighted combination of the three factor
cosines (see ``aggregate_similarity``).

This module is purely additive: it never touches the 1024-d MERT path. It loads
the pretrained heads (``models/merit_heads/head_{mel,rhy,tim}/best_head.pt``,
~11 MB each — no training required) and projects backbone vectors to factor
vectors. The head architecture is read from each checkpoint's own metadata
(``in_dim``/``hidden_dim``/``out_dim``), so a head reshape can't silently break.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FACTORS = ("mel", "rhy", "tim")
FACTOR_NAMES = {"mel": "melody", "rhy": "rhythm", "tim": "timbre"}
DEFAULT_HEADS_DIR = "models/merit_heads"


class ProjectionHead(nn.Module):
    """MERIT projection head: Linear→ReLU→Linear(no bias)→L2-normalize.

    Matches the reference implementation exactly (the second Linear has no bias
    and the output is unit-normalized, so factor similarity is cosine).
    """

    def __init__(self, in_dim: int = 5120, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def load_head(path: str | Path, device: str = "cpu") -> ProjectionHead:
    """Load one head checkpoint. Dims come from the checkpoint's own metadata."""
    ck = torch.load(str(path), map_location=device, weights_only=True)
    head = ProjectionHead(ck["in_dim"], ck["hidden_dim"], ck["out_dim"])
    head.load_state_dict(ck["state_dict"])
    return head.to(device).eval()


def load_heads(
    heads_dir: str | Path = DEFAULT_HEADS_DIR, device: str = "cpu"
) -> dict[str, ProjectionHead]:
    """Load all three factor heads → ``{"mel":..., "rhy":..., "tim":...}``."""
    heads_dir = Path(heads_dir)
    heads = {}
    for f in FACTORS:
        p = heads_dir / f"head_{f}" / "best_head.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"MERIT head not found: {p}. Download with "
                f"`hf download amaai-lab/merit head_{f}/best_head.pt "
                f"--local-dir {heads_dir}`."
            )
        heads[f] = load_head(p, device)
    return heads


def project(
    backbone: np.ndarray, heads: dict[str, ProjectionHead], device: str = "cpu"
) -> dict[str, np.ndarray]:
    """
    Project 5120-d MERIT backbone vector(s) to the three factor spaces.

    backbone: (5120,) or (N, 5120) float array (the ``merit_concat`` output).
    Returns ``{"mel": (…,128), "rhy": (…,128), "tim": (…,128)}`` unit vectors,
    matching the input's batch shape.
    """
    x = np.asarray(backbone, dtype=np.float32)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    t = torch.from_numpy(x).to(device)
    out = {}
    with torch.no_grad():
        for f, head in heads.items():
            v = head(t).cpu().numpy().astype(np.float32)
            out[f] = v[0] if single else v
    return out


def merit_query_vector(backbone: np.ndarray, heads: dict[str, ProjectionHead]) -> np.ndarray:
    """
    Project one raw 5120-d MERIT backbone vector to the 384-d query vector
    matching ``merit_index.py``'s aggregate index layout — a concat of the
    three unit factor vectors, in ``FACTORS`` order (mel, rhy, tim). Pass the
    result through the aggregate ``SongIndex.transform_query``/``.query`` for
    a cosine identical to a corpus row's own aggregate vector.
    """
    factors = project(backbone, heads)
    return np.concatenate([factors[f] for f in FACTORS]).astype(np.float32)


def merit_config(heads_dir: str | Path = DEFAULT_HEADS_DIR) -> dict:
    """Self-describing metadata for a MERIT factor index (stored in SongIndex config)."""
    from .embedding import MERIT_LAYERS, SR, WINDOW_SECONDS, N_WINDOWS

    return {
        "phase": 2,
        "space": "merit",
        "model": "m-a-p/MERT-v1-330M",
        "heads": "amaai-lab/merit",
        "merit_layers": list(MERIT_LAYERS),
        "backbone_dim": 1024 * len(MERIT_LAYERS),
        "factor_dim": 128,
        "sr": SR,
        "window_seconds": WINDOW_SECONDS,
        "n_windows": N_WINDOWS,
    }


def aggregate_similarity(
    sims: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Synthesize an aggregate similarity from the three factor cosine similarities.

    MERIT has no native aggregate; this is Anther's weighted combination
    ``S = α·S_mel + β·S_rhy + γ·S_tim`` (default equal weights). ``sims`` maps
    each factor to an array of cosine similarities (same shape across factors);
    returns the weighted sum. Weights are normalized to sum to 1 so the aggregate
    stays in cosine's [-1, 1] range.
    """
    if weights is None:
        weights = {f: 1.0 for f in FACTORS}
    total = sum(weights.get(f, 0.0) for f in sims)
    if total == 0:
        raise ValueError("aggregate weights sum to zero")
    return sum((weights.get(f, 0.0) / total) * sims[f] for f in sims)
