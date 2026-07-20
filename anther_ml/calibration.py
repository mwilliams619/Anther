"""Shared link-calibration: the mapping from raw MERT cosine similarity to a
human-readable 0-100 score and to a 4-tier similarity band.

Two independent processes need to agree on this mapping:

  * ``ui/atlas.py`` calibrates once at corpus load and stamps every persisted
    query<->query map edge with a ``score`` field.
  * ``mentor/anther_service.py`` (a separate process — see
    docs/mentor-graph-aware.md) computes fresh cosines for ``neighbors``,
    ``compare``, and ``bridge`` and needs to band them the same way, or a
    song's on-map edge score and the mentor's spoken "close"/"related" would
    disagree with each other for no reason a user could see.

Both anchor to the SAME one-time 200k-random-pair cosine draw over the
corpus's fitted index (deterministic, seed=0), so either process reproduces
bit-identical thresholds on its own if the sidecar below is unavailable.
``save_calibration``/``load_calibration`` persist that draw's result as a
small JSON file next to the frozen corpus bundle (NOT part of
``ReferenceCorpus``'s own save/load — this is an optional, regenerable
sidecar) so it only has to be computed once per corpus build.
"""
from __future__ import annotations

import json
import os

import numpy as np

# The percentile of random corpus-pair cosines above which a pair is "similar
# enough to draw a map edge" (ui/atlas.py's original QUERY_LINK_PCTL).
QUERY_LINK_PCTL = float(os.environ.get("ANTHER_QQ_PCTL", "95"))

# Two additional percentiles of the SAME draw, for the "close" and
# "near-identical" band boundaries (see band_for_cosine).
CLOSE_PCTL = 99.0
NEAR_IDENTICAL_PCTL = 99.9

# mentor-graph-aware.md measured on-screen within-artist coherence at cosine
# mean 0.985-0.992, std 0.003-0.009. near-identical never fires below this,
# even on a corpus whose 99.9th percentile happens to sit lower.
NEAR_IDENTICAL_FLOOR = 0.985

SCORE_FLOOR_DISPLAY = 55.0  # a drawn edge's raw cosine (>= qq_threshold) never
                            # displays below this on the 0-100 scale.

# Bootstrap defaults — the values ui/atlas.py's globals held before the first
# corpus load calibrated them for real. Also the safety net for a corpus too
# small (<2 tracks) to draw random pairs from.
DEFAULT_QQ_THRESHOLD = 0.981
DEFAULT_SCORE_CEILING_RAW = 0.994

CALIBRATION_FILENAME = "link_calibration.json"
# Separate sidecar for the MERIT-aggregate index (anther_ml/corpus/merit_index.py)
# — never overwrites the MERT-based sidecar above, since the two indices live
# side by side in the same bundle dir and calibrate independently.
CALIBRATION_FILENAME_MERIT = "link_calibration_merit.json"
# Per-factor (melody/rhythm/timbre) sidecar. Each factor's own raw-cosine
# distribution can run much hotter or colder than the aggregate's — e.g. on
# the 100k+billboard corpus, timbre's 95th-percentile random-pair cosine is
# ~0.95 while the aggregate's is ~0.64. Reusing the aggregate-calibrated
# scale for a factor's display score means that factor either clips to 100
# constantly (if it runs hotter than the aggregate) or almost never reaches
# the top of the 0-100 range (if colder). Keyed by FACTOR_NAMES value
# ("melody"/"rhythm"/"timbre"), not the internal "mel"/"rhy"/"tim" code.
CALIBRATION_FILENAME_MERIT_FACTORS = "link_calibration_merit_factors.json"
CALIBRATION_FORMAT_VERSION = 1

BAND_NEAR_IDENTICAL = "near-identical"
BAND_CLOSE = "close"
BAND_RELATED = "related"
BAND_DISTANT = "distant"


class LinkThresholds:
    """The calibrated constants needed to score and band a raw cosine.

    Boundaries are monotonic by construction: ``near_identical_cutoff >=
    close_cutoff >= qq_threshold`` always holds, because each is a higher
    percentile of the same non-decreasing draw (the near-identical floor can
    only push its cutoff up, never below close_cutoff).
    """

    __slots__ = ("qq_threshold", "score_ceiling_raw", "close_cutoff",
                 "near_identical_cutoff", "pctl", "n_pairs", "seed")

    def __init__(self, qq_threshold, score_ceiling_raw, close_cutoff,
                 near_identical_cutoff, pctl=QUERY_LINK_PCTL, n_pairs=200_000,
                 seed=0):
        self.qq_threshold = float(qq_threshold)
        self.score_ceiling_raw = float(score_ceiling_raw)
        self.close_cutoff = float(close_cutoff)
        self.near_identical_cutoff = float(near_identical_cutoff)
        self.pctl = float(pctl)
        self.n_pairs = int(n_pairs)
        self.seed = int(seed)

    def to_dict(self) -> dict:
        return {
            "format_version": CALIBRATION_FORMAT_VERSION,
            "qq_threshold": self.qq_threshold,
            "score_ceiling_raw": self.score_ceiling_raw,
            "close_cutoff": self.close_cutoff,
            "near_identical_cutoff": self.near_identical_cutoff,
            "pctl": self.pctl,
            "n_pairs": self.n_pairs,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinkThresholds":
        qq = float(d["qq_threshold"])
        ceiling = float(d["score_ceiling_raw"])
        return cls(
            qq_threshold=qq,
            score_ceiling_raw=ceiling,
            close_cutoff=float(d.get("close_cutoff", qq)),
            near_identical_cutoff=float(
                d.get("near_identical_cutoff", max(ceiling, NEAR_IDENTICAL_FLOOR))),
            pctl=d.get("pctl", QUERY_LINK_PCTL),
            n_pairs=d.get("n_pairs", 200_000),
            seed=d.get("seed", 0),
        )

    @classmethod
    def defaults(cls) -> "LinkThresholds":
        return cls(DEFAULT_QQ_THRESHOLD, DEFAULT_SCORE_CEILING_RAW,
                    DEFAULT_QQ_THRESHOLD,
                    max(DEFAULT_SCORE_CEILING_RAW, NEAR_IDENTICAL_FLOOR))


def calibrate_link_thresholds(index, n_pairs: int = 200_000,
                               pctl: float = QUERY_LINK_PCTL,
                               seed: int = 0) -> LinkThresholds:
    """Draw ``n_pairs`` random (unordered, distinct) pairs from
    ``index.embeddings`` (already standardized + L2-normalized) and compute
    every threshold off that ONE draw, so the qq/close/near-identical bands
    are internally consistent. Deterministic (default seed=0): either process
    reproduces the exact same thresholds independently, so a missing sidecar
    never causes disagreement, only redundant computation.
    """
    E = index.embeddings
    n = E.shape[0]
    if n < 2:
        return LinkThresholds.defaults()
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    mask = a != b
    cos = np.einsum("ij,ij->i", E[a[mask]], E[b[mask]])
    qq_threshold = float(np.percentile(cos, pctl))
    score_ceiling_raw = float(cos.max())
    close_cutoff = float(np.percentile(cos, CLOSE_PCTL))
    near_identical_cutoff = max(
        float(np.percentile(cos, NEAR_IDENTICAL_PCTL)), NEAR_IDENTICAL_FLOOR)
    return LinkThresholds(qq_threshold, score_ceiling_raw, close_cutoff,
                           near_identical_cutoff, pctl=pctl, n_pairs=n_pairs,
                           seed=seed)


def display_score(raw_cos: float, thresholds: LinkThresholds,
                   clip_low: bool = True) -> float:
    """Map a raw cosine to the 0-100 human-readable similarity score.

    ``thresholds.qq_threshold`` -> SCORE_FLOOR_DISPLAY, ``score_ceiling_raw``
    -> 100, linear between. ``clip_low=True`` (map edges, which structurally
    can't fall below qq_threshold) floors the result at SCORE_FLOOR_DISPLAY;
    ``clip_low=False`` (a corpus-wide search that can surface pairs below the
    link threshold) lets the score read honestly below the floor, clipping
    only at 0. Both cases clip at 100 on top. Identical math to the
    pre-extraction ``ui/atlas.py._display_score``.
    """
    span = thresholds.score_ceiling_raw - thresholds.qq_threshold
    frac = (raw_cos - thresholds.qq_threshold) / span if span > 0 else 1.0
    score = SCORE_FLOOR_DISPLAY + (100.0 - SCORE_FLOOR_DISPLAY) * frac
    lo = SCORE_FLOOR_DISPLAY if clip_low else 0.0
    return float(np.clip(score, lo, 100.0))


def band_for_cosine(raw_cos: float, thresholds: LinkThresholds) -> str:
    """One of the 4 similarity-band names for a raw cosine, calibrated off
    the same random-pair draw as ``display_score``:

        near-identical  >= near_identical_cutoff  (matches measured within-
                                                     artist coherence, ~0.985+)
        close           >= close_cutoff            (~99th pctl of random pairs)
        related         >= qq_threshold             (~95th pctl -- "an edge
                                                       would be drawn on the map")
        distant         below qq_threshold
    """
    if raw_cos >= thresholds.near_identical_cutoff:
        return BAND_NEAR_IDENTICAL
    if raw_cos >= thresholds.close_cutoff:
        return BAND_CLOSE
    if raw_cos >= thresholds.qq_threshold:
        return BAND_RELATED
    return BAND_DISTANT


def _sidecar_path(corpus_dir, filename: str = CALIBRATION_FILENAME) -> str:
    return os.path.join(str(corpus_dir), filename)


def save_calibration(corpus_dir, thresholds: LinkThresholds,
                      filename: str = CALIBRATION_FILENAME) -> str:
    """Write the sidecar JSON next to the corpus bundle. Atomic (write-then-
    rename) so a concurrent reader never sees a half-written file. Pass
    ``filename=CALIBRATION_FILENAME_MERIT`` to write the MERIT-aggregate
    index's independent sidecar instead of the MERT one."""
    path = _sidecar_path(corpus_dir, filename)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(thresholds.to_dict(), f)
    os.replace(tmp, path)
    return path


def load_calibration(corpus_dir, filename: str = CALIBRATION_FILENAME):
    """The sidecar's ``LinkThresholds``, or ``None`` if it doesn't exist yet
    or fails to parse (caller should fall back to
    ``calibrate_link_thresholds`` in-process)."""
    path = _sidecar_path(corpus_dir, filename)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return LinkThresholds.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def calibrate_factor_link_thresholds(
    factor_vectors: dict, n_pairs: int = 200_000, pctl: float = QUERY_LINK_PCTL,
    seed: int = 0,
) -> dict:
    """Independent ``LinkThresholds`` per factor, one per key of
    ``factor_vectors`` (e.g. ``{"mel": (N,128), "rhy": (N,128), "tim":
    (N,128)}`` from ``merit_index.load_factor_vectors`` — unit-norm rows).
    Draws the SAME ``n_pairs`` random index pairs once (seed=0) and reuses
    them across all factors, so a pair that's "close" on melody and "close"
    on timbre are compared on the same underlying sample — only the
    per-factor cosine and its own distribution differ.

    Calibrating each factor off its own draw (rather than reusing the
    aggregate's ``calibrate_link_thresholds`` scale for all three) matters
    because factors can have very different raw-cosine spreads — e.g. timbre
    frequently runs close to 1.0 between unrelated tracks while rhythm
    spreads much wider, so a shared scale clips one factor to 100 constantly
    while the other rarely reaches the top.
    """
    any_vec = next(iter(factor_vectors.values()))
    n = any_vec.shape[0]
    if n < 2:
        return {f: LinkThresholds.defaults() for f in factor_vectors}
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    out = {}
    for f, E in factor_vectors.items():
        cos = np.einsum("ij,ij->i", E[a], E[b])
        qq_threshold = float(np.percentile(cos, pctl))
        score_ceiling_raw = float(cos.max())
        close_cutoff = float(np.percentile(cos, CLOSE_PCTL))
        near_identical_cutoff = max(
            float(np.percentile(cos, NEAR_IDENTICAL_PCTL)), NEAR_IDENTICAL_FLOOR)
        out[f] = LinkThresholds(qq_threshold, score_ceiling_raw, close_cutoff,
                                 near_identical_cutoff, pctl=pctl,
                                 n_pairs=len(a), seed=seed)
    return out


def save_factor_calibration(
    corpus_dir, thresholds_by_factor: dict,
    filename: str = CALIBRATION_FILENAME_MERIT_FACTORS,
) -> str:
    """Write the per-factor sidecar (``{factor_name: LinkThresholds.to_dict()}``).
    Atomic (write-then-rename), same convention as ``save_calibration``."""
    path = _sidecar_path(corpus_dir, filename)
    tmp = path + ".tmp"
    payload = {f: t.to_dict() for f, t in thresholds_by_factor.items()}
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return path


def load_factor_calibration(
    corpus_dir, filename: str = CALIBRATION_FILENAME_MERIT_FACTORS,
):
    """``{factor_name: LinkThresholds}`` from the sidecar, or ``None`` if it
    doesn't exist yet or fails to parse (caller falls back to a shared/
    aggregate scale)."""
    path = _sidecar_path(corpus_dir, filename)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return {f: LinkThresholds.from_dict(d) for f, d in data.items()}
    except (KeyError, TypeError, ValueError):
        return None
