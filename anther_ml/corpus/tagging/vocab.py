"""
Everynoise micro-genre vocabulary: load, normalize, alias-fold.

The top-2000 everynoise genres (popularity-ordered, one per line) are the
label space for per-track micro-genre tagging
(MICROGENRE_TAGGING_BUILD_PLAN.md §3a). Everything built from them is a
display-only artifact of the frozen map — see docs/invariants.md.

``norm()`` here is the *matching* normalizer for genre strings and playlist
names. It is deliberately separate from ``labels.normalize_playlist_name``
(which folds names for display tallies and rejects unusable ones): this one
must also fold hyphen/slash variants ("hip-hop" == "hip hop") so vocabulary
lookups hit.
"""

import re
import unicodedata
from pathlib import Path

DEFAULT_VOCAB_PATH = "data/everynoise_genres_top2000.txt"

# Curated near-miss aliases observed in MPD playlist names. Keys are
# norm()-forms, values canonical vocab entries. Kept small on purpose;
# build_vocab_map drops any alias whose target is missing from the vocab.
ALIASES = {
    "hiphop": "hip hop",
    "rnb": "r&b",
    "r n b": "r&b",
    "r and b": "r&b",
    "dnb": "drum and bass",
    "d&b": "drum and bass",
    "drum n bass": "drum and bass",
    "drum & bass": "drum and bass",
    "lofi": "lo-fi",
    "lo fi": "lo-fi",
    "neosoul": "neo soul",
    "kpop": "k-pop",
    "jpop": "j-pop",
    "reggeaton": "reggaeton",
    "regueton": "reggaeton",
}

# Keep word chars and '&' ("r&b"); everything else (emoji, quotes, dots)
# is stripped after separators are folded to spaces.
_SEP_RE = re.compile(r"[-_/,+.]")
_PUNCT_RE = re.compile(r"[^\w&\s]")
_WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    """Matching form of a genre or playlist name: casefold, fold separators
    to spaces, strip remaining punctuation/emoji, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _SEP_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    return _WS_RE.sub(" ", s).strip()


def load_vocab(path: str | Path = DEFAULT_VOCAB_PATH) -> list[str]:
    """Canonical genre list, file (popularity) order preserved."""
    with open(path) as f:
        seen, out = set(), []
        for line in f:
            g = line.strip()
            if g and g not in seen:
                seen.add(g)
                out.append(g)
    return out


def build_vocab_map(
    vocab: list[str], aliases: dict[str, str] | None = None
) -> dict[str, str]:
    """{norm(name): canonical_genre}. On norm collisions the more popular
    (earlier) genre wins. Aliases pointing outside the vocab are dropped."""
    if aliases is None:
        aliases = ALIASES
    vmap: dict[str, str] = {}
    canon = set(vocab)
    for g in vocab:
        vmap.setdefault(norm(g), g)
    for key, target in aliases.items():
        if target in canon:
            vmap.setdefault(norm(key), target)
    return vmap


def max_token_len(vocab_map: dict[str, str]) -> int:
    """Longest vocab key in tokens — bounds the n-gram scan in weak_labels."""
    return max((len(k.split()) for k in vocab_map), default=1)
