"""
Field normalization for artist profiles (ARTIST_ENRICHMENT_PLAN.md §4).

Pure, network-free transforms that turn raw source data into the canonical
profile field shapes. Every function preserves source + (where available)
confidence, and deliberately does *not* invent data for absent inputs.
"""

from __future__ import annotations

from typing import Any

# Small alias map — collapse the most common spelling variants to one display
# form. Intentionally conservative; genre taxonomies are display-only here and
# must never feed clustering (docs/invariants.md).
_GENRE_ALIASES = {
    "hip hop": "hip-hop",
    "hiphop": "hip-hop",
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "drum and bass": "drum & bass",
    "dnb": "drum & bass",
    "electronica": "electronic",
    "alt rock": "alternative rock",
    "indie": "indie rock",
    "synth pop": "synth-pop",
    "synthpop": "synth-pop",
    "lo fi": "lo-fi",
    "lofi": "lo-fi",
}

MAX_GENRES = 5


def normalize_genre_name(name: str) -> str:
    n = " ".join((name or "").lower().split())
    return _GENRE_ALIASES.get(n, n)


def normalize_genres(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe + alias + cap a list of ``{name, source, weight?}`` genres.

    Highest-weight wins on collision; ties keep first-seen order. Caps to
    ``MAX_GENRES`` displayed entries (plan §4)."""
    merged: dict[str, dict[str, Any]] = {}
    for order, g in enumerate(raw or []):
        name = normalize_genre_name(g.get("name", ""))
        if not name:
            continue
        weight = g.get("weight")
        entry = merged.get(name)
        if entry is None:
            merged[name] = {"name": name, "source": g.get("source", ""),
                            "weight": weight, "_order": order}
        elif weight is not None and (entry["weight"] is None or weight > entry["weight"]):
            entry["weight"] = weight
            entry["source"] = g.get("source", entry["source"])
    ordered = sorted(merged.values(),
                     key=lambda e: (-(e["weight"] or 0), e["_order"]))
    out = []
    for e in ordered[:MAX_GENRES]:
        item = {"name": e["name"], "source": e["source"]}
        if e["weight"] is not None:
            item["weight"] = e["weight"]
        out.append(item)
    return out


def genres_from_musicbrainz(artist_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull genres (preferred) then tags from a MusicBrainz artist lookup doc."""
    raw = []
    for g in artist_doc.get("genres", []) or []:
        raw.append({"name": g.get("name", ""), "source": "musicbrainz",
                    "weight": g.get("count")})
    if not raw:  # fall back to folksonomy tags when curated genres are absent
        for t in artist_doc.get("tags", []) or []:
            raw.append({"name": t.get("name", ""), "source": "musicbrainz",
                        "weight": t.get("count")})
    return normalize_genres(raw)


def structure_hometown(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Build the origin object from a resolved MB search candidate.

    Prefers ``begin_area`` (birthplace / formation locale), then ``area``, then
    ``country``. Absent everything → ``None``. Records the MB relationship type
    so the UI can label uncertain cases 'Origin' rather than 'Hometown'
    (plan §2/§4)."""
    if candidate.get("begin_area"):
        return {"name": candidate["begin_area"], "type": "begin_area",
                "source": "musicbrainz"}
    if candidate.get("area"):
        return {"name": candidate["area"], "type": "area",
                "source": "musicbrainz"}
    if candidate.get("country"):
        return {"name": candidate["country"], "type": "country",
                "source": "musicbrainz"}
    return None


def normalize_labels(raw: list[dict[str, Any]], top_n: int = 4) -> list[dict[str, Any]]:
    """Cap + tag release-derived labels. Scope is fixed to ``release-derived`` so
    the UI never presents this as a current exclusive affiliation (plan §4)."""
    out = []
    for e in (raw or [])[:top_n]:
        if not e.get("name"):
            continue
        out.append({"name": e["name"], "mbid": e.get("mbid"),
                    "release_count": e.get("release_count"),
                    "source": "musicbrainz", "scope": "release-derived"})
    return out


def clean_image_url(url: str | None) -> str | None:
    """Accept only plausible http(s) image URLs; drop everything else."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    return url
