"""
Canonical artist-profile schema for the enrichment layer.

A *profile* is a durable, provenance-carrying record about a corpus artist,
kept entirely separate from the immutable clustering artifacts in
``artist_meta.json`` (see docs/invariants.md — those artifacts are row-count
locked and must never be mutated). Profiles are keyed by the *normalized artist
name* (``_norm(...)`` from ``anther_ml.spotify_deezer``) so they survive a
corpus rebuild that would shift the positional ``corpus:{index}`` ids.

Every enriched field records where it came from and when, so data can be
refreshed field-by-field and audited later. See ARTIST_ENRICHMENT_PLAN.md §1/§4.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Fields that carry external, refreshable data (used by `--missing <field>`).
ENRICHABLE_FIELDS = ("profile_image", "following", "genres", "hometown", "labels")

# Enrichment lifecycle states.
STATUS_COMPLETE = "complete"    # matched + at least one enriched field present
STATUS_PARTIAL = "partial"      # matched but some core fields missing
STATUS_UNMATCHED = "unmatched"  # no confident identity match found
STATUS_REVIEW = "review"        # ambiguous — parked in the review queue


@dataclass
class Provenance:
    """Source + timestamp stamp attached to every enriched value."""
    source: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "updated_at": self.updated_at}


@dataclass
class ArtistProfile:
    """The canonical profile document (ARTIST_ENRICHMENT_PLAN.md §1).

    ``artist_key`` is the normalized-name join key. ``identities`` holds
    source-specific ids so a refresh can re-fetch without re-resolving.
    Absent data is represented by ``None`` / empty list — never a fabricated
    default (plan §6).
    """
    artist_key: str
    name: str
    identities: dict[str, Any] = field(default_factory=dict)
    profile_image: dict[str, Any] | None = None   # {url, source, updated_at}
    following: dict[str, Any] | None = None        # {count, source, updated_at}
    genres: list[dict[str, Any]] = field(default_factory=list)  # [{name, source, weight?}]
    hometown: dict[str, Any] | None = None         # {name, city?, region?, country?, type, source}
    labels: list[dict[str, Any]] = field(default_factory=list)  # [{name, mbid?, release_count?, source, scope}]
    match_confidence: float | None = None
    status: str = STATUS_UNMATCHED
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtistProfile":
        known = {f: data.get(f) for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Preserve dataclass defaults for keys the stored blob omits.
        clean = {k: v for k, v in known.items() if v is not None or k in ("artist_key", "name")}
        return cls(**{**_defaults(), **clean})

    def missing_fields(self) -> list[str]:
        """Enrichable fields that are still empty — drives `--missing`."""
        out = []
        for name in ENRICHABLE_FIELDS:
            val = getattr(self, name)
            if val in (None, [], {}, ""):
                out.append(name)
        return out

    def to_api_profile(self) -> dict[str, Any]:
        """Compact object surfaced under ``/api/artist/<id>``'s ``profile`` key
        (plan §6). Flattens provenance to the couple of fields the UI shows and
        never fabricates absent values."""
        return {
            "image_url": (self.profile_image or {}).get("url"),
            "image_source": (self.profile_image or {}).get("source"),
            "following": (self.following or {}).get("count"),
            "following_source": (self.following or {}).get("source"),
            "following_as_of": (self.following or {}).get("updated_at"),
            "genres": [g["name"] for g in self.genres][:5],
            "origin": (self.hometown or {}).get("name"),
            "origin_source": (self.hometown or {}).get("source"),
            "labels": [l["name"] for l in self.labels],
            "match_confidence": self.match_confidence,
            "enrichment_status": self.status,
            "updated_at": self.updated_at,
        }


def _defaults() -> dict[str, Any]:
    return {
        "identities": {},
        "profile_image": None,
        "following": None,
        "genres": [],
        "hometown": None,
        "labels": [],
        "match_confidence": None,
        "status": STATUS_UNMATCHED,
        "updated_at": None,
    }
