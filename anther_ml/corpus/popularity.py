"""
Popularity sidecar + reranking blend (billboard_expansion project).

The sidecar (``popularity.json`` / ``popularity_by_track_id.json``, built by
``scripts/billboard_expansion/02_match_against_corpus.py``) is metadata-only —
it never touches embeddings, the SongIndex, or the Leiden fit. It is loaded
lazily and blended into a ranked result list *after* cosine scoring, exactly
like ``tags`` in ``place.py``: a read-out, not a clustering input (see
docs/invariants.md's genre carve-out — same principle applies here).

Score blend: ``final = alpha * cosine + beta * popularity_percentile``, where
popularity_percentile is the track's popularity_score rank-normalized to
[0, 1] over the whole sidecar (so it's stable regardless of score-scale
changes upstream). Both alpha/beta default to a mild blend (0.85 / 0.15) so
popularity nudges rank order without overriding acoustic similarity.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_ALPHA = 0.85  # cosine similarity weight
DEFAULT_BETA = 0.15   # popularity weight


def load_popularity_by_track_id(path: str | Path) -> dict[str, float]:
    """``{track_id: popularity_score}`` — raw scores, NOT yet percentiled."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def popularity_percentiles(track_id_scores: dict[str, float]) -> dict[str, float]:
    """Rank-normalize raw popularity_score values to [0, 1] percentiles."""
    if not track_id_scores:
        return {}
    ids = list(track_id_scores.keys())
    scores = np.array([track_id_scores[i] for i in ids], dtype=np.float64)
    order = np.argsort(scores)
    pct = np.empty_like(order, dtype=np.float64)
    pct[order] = np.linspace(0.0, 1.0, len(scores))
    return {i: float(p) for i, p in zip(ids, pct)}


def rerank_with_popularity(
    results: list[dict],
    popularity_pct: dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> list[dict]:
    """
    Blend a ranked result list (as returned by ``SongIndex.query``,
    ``recommend_from_seeds``, or ``place()``'s ``neighbors``) with the
    popularity percentile map. Tracks absent from the sidecar get
    popularity_percentile = 0 (no boost, no penalty vs. an unranked song).

    Returns a NEW list, re-sorted by blended score, with
    ``popularity_percentile`` and ``blended_score`` added to each row.
    Re-numbers ``rank`` 1..N on the output order. Does not mutate ``results``.
    """
    out = []
    for row in results:
        row = dict(row)
        tid = row.get("id")
        pct = popularity_pct.get(tid, 0.0)
        row["popularity_percentile"] = pct
        row["blended_score"] = alpha * row.get("score", 0.0) + beta * pct
        out.append(row)
    out.sort(key=lambda r: r["blended_score"], reverse=True)
    for rank, row in enumerate(out, 1):
        row["rank"] = rank
    return out
