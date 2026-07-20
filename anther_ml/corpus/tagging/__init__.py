"""
Per-track micro-genre tagging over a frozen corpus bundle
(MICROGENRE_TAGGING_BUILD_PLAN.md).

Stage A (weak_labels): playlist names → noisy in-vocab seed tags.
Stage B (probe): linear probe on frozen MERT vectors + kNN propagation —
what makes tags track-level instead of artist-level.

Everything here is a display-only artifact of the frozen map. Genre never
feeds the embedding, SongIndex, Leiden, or anther_ml.eval, and the tag
metrics in tagging/evaluate.py never tune them (docs/invariants.md).
"""

from .build_tags import (
    PROBE_FILE,
    TRACK_TAGS_FILE,
    load_seeds,
    run_all,
    run_embed_eval,
    run_evaluate,
    run_fit,
    run_predict,
    run_seed,
)
from .probe import TagProbe, knn_smooth, predict_tags
from .vocab import load_vocab, norm
from .weak_labels import build_seed_labels, match_name

__all__ = [
    "PROBE_FILE",
    "TRACK_TAGS_FILE",
    "TagProbe",
    "build_seed_labels",
    "knn_smooth",
    "load_seeds",
    "load_vocab",
    "match_name",
    "norm",
    "predict_tags",
    "run_all",
    "run_embed_eval",
    "run_evaluate",
    "run_fit",
    "run_predict",
    "run_seed",
]
