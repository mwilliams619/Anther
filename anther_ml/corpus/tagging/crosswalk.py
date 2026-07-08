"""
FMA taxonomy → everynoise vocabulary crosswalk
(MICROGENRE_TAGGING_BUILD_PLAN.md §5a).

FMA's 161 genres are a different, much coarser taxonomy. The mapping is
exact/normalized name match first, then a small hand-curated table; FMA
genres that map to nothing are **dropped from eval, never guessed**. The
same pattern is reusable for Jamendo's 87 tags if pretraining is added.

Caveat carried from the plan §10: after the crosswalk drops broad parents
("Experimental", "Sound Effects"…) the held-out eval covers the coarse end
of the vocabulary only — it is a floor/proxy, not a microgenre readout.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .vocab import norm

DEFAULT_CROSSWALK_PATH = "data/fma_to_everynoise.json"

# Hand-curated FMA → everynoise entries where names don't fold to each other.
# Targets are checked against the vocab at build time; missing ones dropped
# with a warning rather than guessed. Conservative on purpose: broad umbrella
# genres with no honest everynoise counterpart map to [] (excluded from eval).
CURATED: dict[str, list[str]] = {
    "Hip-Hop": ["hip hop"],
    "Rock": ["rock"],
    "Pop": ["pop"],
    "Folk": ["folk"],
    "Jazz": ["jazz"],
    "Country": ["country"],
    "Blues": ["blues"],
    "Classical": ["classical"],
    "Electronic": ["electronica"],
    "Soul-RnB": ["soul", "r&b"],
    "Punk": ["punk"],
    "Metal": ["metal"],
    "Reggae - Dub": ["reggae", "dub"],
    "Rap": ["rap"],
    "Hip-Hop Beats": ["instrumental hip hop"],
    "Breakbeat": ["breakbeat"],
    "Chip Music": ["chiptune"],
    "Trip-Hop": ["trip hop"],
    "Drum & Bass": ["drum and bass"],
    "Psych-Folk": ["psychedelic folk"],
    "Psych-Rock": ["psychedelic rock"],
    "Indie-Rock": ["indie rock"],
    "Singer-Songwriter": ["singer-songwriter"],
    "Old-Time / Historic": [],
    "Experimental": [],
    "Sound Effects": [],
    "Sound Art": [],
    "Sound Collage": [],
    "Sound Poetry": [],
    "Field Recordings": [],
    "Radio Art": [],
    "Audio Collage": [],
    "Spoken": [],
    "Spoken Weird": [],
    "Spoken Word": [],
    "Novelty": [],
    "Unclassifiable": [],
}


def build_crosswalk(
    fma_genre_names: list[str],
    vocab: list[str],
    curated: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """{fma_genre: [everynoise_genre, ...]} — [] means dropped from eval."""
    if curated is None:
        curated = CURATED
    vmap = {norm(g): g for g in reversed(vocab)}  # popular (early) wins ties
    canon = set(vocab)
    out: dict[str, list[str]] = {}
    for fg in fma_genre_names:
        if fg in curated:
            targets = [t for t in curated[fg] if t in canon]
            missing = [t for t in curated[fg] if t not in canon]
            if missing:
                print(f"crosswalk: curated targets not in vocab, dropped: {missing}")
            out[fg] = targets
        else:
            hit = vmap.get(norm(fg))
            out[fg] = [hit] if hit else []
    return out


def save_crosswalk(mapping: dict, path: str | Path = DEFAULT_CROSSWALK_PATH) -> None:
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)


def load_crosswalk(path: str | Path = DEFAULT_CROSSWALK_PATH) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def crosswalk_labels(
    per_track_fma_genres: list[list[str]],
    mapping: dict[str, list[str]],
    genre_names: list[str],
) -> sparse.csr_matrix:
    """Multi-hot (M, G) everynoise labels from per-track FMA genre-name lists.
    Unmapped FMA genres contribute nothing (dropped, not guessed)."""
    col = {g: j for j, g in enumerate(genre_names)}
    rows, cols = [], []
    for i, fgs in enumerate(per_track_fma_genres):
        hit = set()
        for fg in fgs:
            for target in mapping.get(fg, []):
                if target in col:
                    hit.add(col[target])
        rows.extend([i] * len(hit))
        cols.extend(sorted(hit))
    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(per_track_fma_genres), len(genre_names))
    )
