"""Resolve an artist's discography from the iTunes lookup API.

iTunes is used (rather than Deezer) because its song objects carry
``releaseDate`` and ``collectionName``, which the era-grouping needs. This
module only does light HTTP + dedupe; it does not touch the GPU model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from anther_ml import itunes

ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
OTHER_ERA = "Singles / Soundtrack / Other"


@dataclass
class EraRule:
    """One substring->era mapping; first matching rule (in order) wins."""
    substring: str
    era: str


def _base_title(title: str) -> str:
    """Normalized key for de-duplicating the same song across releases."""
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", (title or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def era_mapper(rules: list[EraRule], default: str = OTHER_ERA) -> Callable[[str], str]:
    """Build an era-of-collection function from ordered substring rules."""
    lowered = [(r.substring.lower(), r.era) for r in rules]

    def era_of(collection: str) -> str:
        c = (collection or "").lower()
        for sub, era in lowered:
            if sub in c:
                return era
        return default

    return era_of


def resolve_discography(
    artist_name: str,
    era_rules: list[EraRule],
    *,
    artist_id: int | None = None,
    limit: int = 200,
    timeout: int = 30,
) -> "list[dict[str, Any]]":
    """Return de-duplicated discography rows with era labels.

    Each row: track_id, itunes_id, title, album, release_date, year,
    duration_ms, genre_itunes, era, preview_url. Deduped by base-title,
    keeping the earliest release. Raises ``LookupError`` if the artist can't
    be resolved.
    """
    if artist_id is None:
        info = itunes.resolve_artist(artist_name)
        if not info:
            raise LookupError(f"artist not found on iTunes: {artist_name!r}")
        artist_id = info["artist_id"]

    resp = requests.get(
        ITUNES_LOOKUP,
        params={"id": artist_id, "entity": "song", "limit": limit},
        timeout=timeout,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    songs = [
        x for x in results
        if x.get("wrapperType") == "track" and x.get("kind") == "song"
        and x.get("previewUrl") and x.get("trackId")
        and str(x.get("artistId")) == str(artist_id)
    ]

    earliest: dict[str, dict] = {}
    for x in songs:
        key = _base_title(x.get("trackName", ""))
        rd = x.get("releaseDate", "9999")
        if key not in earliest or rd < earliest[key].get("releaseDate", "9999"):
            earliest[key] = x

    era_of = era_mapper(era_rules)
    rows: list[dict[str, Any]] = []
    for x in earliest.values():
        rd = x.get("releaseDate", "")
        year = int(rd[:4]) if rd[:4].isdigit() else 0
        rows.append({
            "track_id": f"itunes:{x['trackId']}",
            "itunes_id": x["trackId"],
            "title": x.get("trackName", ""),
            "album": x.get("collectionName", ""),
            "release_date": rd,
            "year": year,
            "duration_ms": x.get("trackTimeMillis"),
            "genre_itunes": x.get("primaryGenreName"),
            "era": era_of(x.get("collectionName", "")),
            "preview_url": x["previewUrl"],
        })
    rows.sort(key=lambda r: (r["year"], r["album"], r["title"]))
    return rows
