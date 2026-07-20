"""
Stage A — playlist-name weak labels (MICROGENRE_TAGGING_BUILD_PLAN.md §3).

Each track's own playlist names are matched against the everynoise vocabulary
in three tiers (exact > token-substring > optional fuzzy) to produce a noisy
but in-vocab, in-distribution seed label matrix. Longest vocab match wins
inside a name ("k-pop hits" seeds *k-pop*, never *pop*).

Catalog-name discount: a single playlist name that matches many genres at once
("Soul, Funk, Jazz, Latin, Disco, Afrobeat...") assigns every one of them to
every member track — mostly wrongly. Names matching >= MULTI_GENRE_MIN genres
contribute each at weight/n_matched.

Seeds are training data for the display-only tag probe (probe.py). They never
touch the embedding, SongIndex, Leiden, or anther_ml.eval (docs/invariants.md).
"""

from collections import Counter

import numpy as np
from scipy import sparse

from .vocab import build_vocab_map, max_token_len, norm

TIER_WEIGHTS = {"exact": 1.0, "substring": 0.6, "fuzzy": 0.4}
MULTI_GENRE_MIN = 3  # >= this many genres from one name → catalog-name discount
FUZZY_THRESHOLD = 92  # rapidfuzz token_set_ratio floor (gated, off by default)

# Vocab entries that are also generic English playlist words. As substrings
# they tag thousands of tracks wrongly ("Sound of Summer" is not the genre
# *sound*), so they may only match a playlist named exactly that. Observed on
# the 100k MPD seed run: substring "sound" alone hit 4.4k tracks.
EXACT_ONLY = frozenset(
    {"sound", "focus", "sleep", "rain", "environmental", "background music"}
)


def match_name(
    name: str,
    vocab_map: dict[str, str],
    *,
    max_len: int | None = None,
    fuzzy: bool = False,
) -> dict[str, str]:
    """Match one playlist name → {canonical_genre: tier}.

    Exact wins outright. Otherwise whole-word n-grams are scanned longest
    first, and tokens covered by a longer match are dead to shorter ones.
    The fuzzy tier fires only when the first two tiers found nothing and the
    match is unambiguous (single vocab entry at/above threshold).
    """
    key = norm(name)
    if not key:
        return {}
    if key in vocab_map:
        return {vocab_map[key]: "exact"}

    if max_len is None:
        max_len = max_token_len(vocab_map)
    tokens = key.split()
    covered = [False] * len(tokens)
    out: dict[str, str] = {}
    for n in range(min(max_len, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(covered[i : i + n]):
                continue
            canon = vocab_map.get(" ".join(tokens[i : i + n]))
            if canon is not None and canon not in EXACT_ONLY:
                out.setdefault(canon, "substring")
                for j in range(i, i + n):
                    covered[j] = True
    if out or not fuzzy:
        return out

    try:
        from rapidfuzz import fuzz, process
    except ImportError as e:
        raise ImportError(
            "the fuzzy tier needs rapidfuzz: pip install rapidfuzz"
        ) from e
    hits = process.extract(
        key, list(vocab_map), scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD, limit=2,
    )
    if len(hits) == 1:  # unambiguous only
        return {vocab_map[hits[0][0]]: "fuzzy"}
    return {}


def build_seed_labels(
    metadata: list[dict],
    vocab: list[str],
    *,
    fuzzy: bool = False,
    aliases: dict[str, str] | None = None,
) -> tuple[sparse.csr_matrix, list[str], np.ndarray, Counter]:
    """Per-track seed tag weights from playlist names.

    Returns ``(W, genre_names, confidence, tier_counts)`` where ``W`` is an
    ``(N, G)`` CSR matrix of accumulated match weights (0 = no seed; binarize
    with ``W > 0`` for multi-hot labels), ``genre_names`` keeps the vocab's
    popularity order, ``confidence[i]`` is the total match weight of track i
    (monotone in how many of its playlists matched), and ``tier_counts``
    tallies track-genre assignments per tier.
    """
    vocab_map = build_vocab_map(vocab, aliases)
    max_len = max_token_len(vocab_map)
    col = {g: j for j, g in enumerate(vocab)}

    name_cache: dict[str, dict[str, str]] = {}
    rows, cols, vals = [], [], []
    tier_counts: Counter = Counter()

    for i, row in enumerate(metadata):
        acc: dict[int, float] = {}
        for pl in row.get("playlists") or []:
            name = pl.get("name") or ""
            matches = name_cache.get(name)
            if matches is None:
                matches = match_name(
                    name, vocab_map, max_len=max_len, fuzzy=fuzzy
                )
                name_cache[name] = matches
            if not matches:
                continue
            discount = len(matches) if len(matches) >= MULTI_GENRE_MIN else 1
            for genre, tier in matches.items():
                acc[col[genre]] = acc.get(col[genre], 0.0) + (
                    TIER_WEIGHTS[tier] / discount
                )
                tier_counts[tier] += 1
        for j, w in acc.items():
            rows.append(i)
            cols.append(j)
            vals.append(w)

    W = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(len(metadata), len(vocab)),
    )
    confidence = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    return W, list(vocab), confidence, tier_counts


def genre_support(W: sparse.csr_matrix, genre_names: list[str]) -> dict[str, int]:
    """Tracks carrying each genre seed (any weight), vocab order, zeros kept."""
    counts = np.asarray((W > 0).sum(axis=0)).ravel()
    return {g: int(c) for g, c in zip(genre_names, counts)}
