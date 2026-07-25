"""
MusicBrainz adapter — origin, genres, and release-derived labels
(ARTIST_ENRICHMENT_PLAN.md §2/§3).

MusicBrainz requires:
- a descriptive User-Agent with contact info, and
- an average of no more than one request per second.

This client enforces both: a process-wide minimum interval between calls and a
mandatory UA. It returns *candidates* and raw sub-documents; deciding which
candidate is the artist is the resolver's job.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from anther_ml.spotify_deezer import _artist_ratio

MB_BASE = "https://musicbrainz.org/ws/2"
DEFAULT_USER_AGENT = (
    "Anther-ArtistEnrichment/0.1 ( https://github.com/mwilliams619/Anther )"
)


class MusicBrainzError(RuntimeError):
    pass


class MusicBrainzClient:
    """Rate-limited MusicBrainz web-service client.

    ``min_interval`` is the floor between requests (seconds). MusicBrainz allows
    ~1 req/s for anonymous clients, so the default is conservative.
    """

    source = "musicbrainz"

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT,
                 min_interval: float = 1.05, timeout: float = 15.0,
                 retries: int = 3):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    # -- transport -------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "fmt": "json"}
        last_err = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                r = self._session.get(f"{MB_BASE}/{path}", params=params,
                                      timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 503:  # rate-limited — back off harder
                    last_err = "HTTP 503 (rate limited)"
                    time.sleep(self.min_interval * (attempt + 2))
                    continue
                last_err = f"HTTP {r.status_code}"
            except requests.RequestException as e:  # pragma: no cover - network
                last_err = str(e)
            time.sleep(self.min_interval * (attempt + 1))
        raise MusicBrainzError(last_err or "unknown MusicBrainz error")

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    # -- queries ---------------------------------------------------------
    def search_artist(self, name: str, limit: int = 8) -> list[dict[str, Any]]:
        """Candidate artists for ``name``, each with MB's relevance ``score``
        (0–100), a fuzzy ``name_ratio`` we compute, disambiguation, country and
        begin-area. No release/genre data yet — that needs a lookup."""
        if not name or not name.strip():
            return []
        data = self._get("artist", {"query": name, "limit": limit})
        out = []
        for a in data.get("artists", []):
            cand_name = a.get("name", "")
            out.append({
                "mbid": a.get("id"),
                "name": cand_name,
                "mb_score": float(a.get("score", 0)),
                "name_ratio": _artist_ratio(name, cand_name),
                "disambiguation": a.get("disambiguation", ""),
                "country": a.get("country"),
                "type": a.get("type"),
                "begin_area": (a.get("begin-area") or {}).get("name"),
                "area": (a.get("area") or {}).get("name"),
                "raw": a,
            })
        return out

    def lookup_artist(self, mbid: str) -> dict[str, Any]:
        """Full artist doc with tags, genres and area relationships — the source
        of truth for origin + genres."""
        return self._get(f"artist/{mbid}",
                          {"inc": "tags+genres+aliases+area-rels"})

    def artist_release_labels(self, mbid: str, limit: int = 100) -> list[dict[str, Any]]:
        """Distinct labels derived from this artist's official releases.

        Returned as ``[{name, mbid, release_count}]``, most releases first. This
        is *release-derived* provenance, never a claim of current affiliation
        (plan §4)."""
        data = self._get("release",
                          {"artist": mbid, "inc": "labels",
                           "status": "official", "limit": limit})
        counts: dict[str, dict[str, Any]] = {}
        for rel in data.get("releases", []):
            for li in rel.get("label-info", []):
                label = li.get("label") or {}
                lname = label.get("name")
                if not lname:
                    continue
                key = label.get("id") or lname
                entry = counts.setdefault(
                    key, {"name": lname, "mbid": label.get("id"), "release_count": 0})
                entry["release_count"] += 1
        return sorted(counts.values(), key=lambda e: -e["release_count"])
