"""
Deezer artist adapter — profile image + fan count (ARTIST_ENRICHMENT_PLAN.md §2).

Deezer's simple API needs no key. We reuse the throttled/retrying ``_deezer_get``
already used for track matching (``anther_ml.spotify_deezer``) so all Deezer
traffic shares one backoff policy. ``nb_fan`` is the follower proxy; it changes
over time, so callers stamp it with an observation date (plan §4).
"""

from __future__ import annotations

from typing import Any

from anther_ml.spotify_deezer import _deezer_get, _artist_ratio


class DeezerArtistClient:
    """Search + fetch Deezer artists. Stateless apart from a throttle knob."""

    source = "deezer"

    def __init__(self, throttle: float = 0.06):
        self.throttle = throttle

    def search_artist(self, name: str, limit: int = 8) -> list[dict[str, Any]]:
        """Return scored Deezer artist candidates for ``name``.

        Each candidate: ``{deezer_id, name, nb_fan, image_url, link, name_ratio}``.
        ``name_ratio`` is a 0–1 fuzzy match against the query so the resolver can
        cross-check Deezer's pick against MusicBrainz rather than trusting
        Deezer's own ranking blindly (plan §3 — no name-only enrichment).
        """
        if not name or not name.strip():
            return []
        data = _deezer_get("search/artist", {"q": name, "limit": limit},
                           throttle=self.throttle)
        rows = (data or {}).get("data") or []
        out = []
        for row in rows:
            cand_name = row.get("name", "")
            out.append({
                "deezer_id": row.get("id"),
                "name": cand_name,
                "nb_fan": row.get("nb_fan"),
                "image_url": _best_image(row),
                "link": row.get("link"),
                "name_ratio": _artist_ratio(name, cand_name),
            })
        out.sort(key=lambda c: -c["name_ratio"])
        return out

    def get_artist(self, deezer_id: int | str) -> dict[str, Any] | None:
        """Full artist record by id, for a refresh of just image/fan count."""
        data = _deezer_get(f"artist/{deezer_id}", throttle=self.throttle)
        if not data or data.get("error"):
            return None
        return {
            "deezer_id": data.get("id"),
            "name": data.get("name", ""),
            "nb_fan": data.get("nb_fan"),
            "image_url": _best_image(data),
            "link": data.get("link"),
            "raw": data,
        }


def _best_image(row: dict[str, Any]) -> str | None:
    """Prefer the largest non-placeholder picture Deezer offers."""
    for key in ("picture_xl", "picture_big", "picture_medium", "picture"):
        url = row.get(key)
        # Deezer serves a generic placeholder for artists with no photo; its URL
        # contains no artist hash, so treat an empty/blank value as absent.
        if url:
            return url
    return None
