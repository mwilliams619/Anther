"""
SQLite persistence for artist profiles, raw source payloads, and the
ambiguous-match review queue (ARTIST_ENRICHMENT_PLAN.md §5).

Three tables:

- ``profiles``       — one canonical :class:`ArtistProfile` per normalized name,
                       stored as a JSON blob plus a few indexed columns.
- ``raw_cache``      — verbatim source payloads keyed by (source, source_id,
                       kind) so a refresh/refit never needs a re-crawl and every
                       field stays auditable.
- ``review_queue``   — artists whose identity was too ambiguous to accept
                       automatically, with the candidate list that caused it.

Writes are idempotent upserts, so the DB doubles as the crawl checkpoint:
resuming just skips keys already marked ``complete``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator

from .schema import ArtistProfile, STATUS_COMPLETE

DEFAULT_DB_PATH = "data/artist_profiles.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    artist_key       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    profile          TEXT NOT NULL,          -- full ArtistProfile JSON
    match_confidence REAL,
    status           TEXT NOT NULL,
    updated_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(status);

CREATE TABLE IF NOT EXISTS raw_cache (
    source     TEXT NOT NULL,
    source_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,                -- e.g. 'artist', 'releases'
    payload    TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (source, source_id, kind)
);

CREATE TABLE IF NOT EXISTS review_queue (
    artist_key TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    candidates TEXT NOT NULL,                -- JSON list of scored candidates
    created_at TEXT NOT NULL
);
"""


class ArtistProfileStore:
    """Thin, dependency-free wrapper around the profiles SQLite DB."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ArtistProfileStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- profiles --------------------------------------------------------
    def upsert(self, profile: ArtistProfile) -> None:
        self._conn.execute(
            "INSERT INTO profiles (artist_key, name, profile, match_confidence, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(artist_key) DO UPDATE SET "
            "name=excluded.name, profile=excluded.profile, "
            "match_confidence=excluded.match_confidence, status=excluded.status, "
            "updated_at=excluded.updated_at",
            (profile.artist_key, profile.name, json.dumps(profile.to_dict()),
             profile.match_confidence, profile.status, profile.updated_at),
        )
        self._conn.commit()

    def get(self, artist_key: str) -> ArtistProfile | None:
        row = self._conn.execute(
            "SELECT profile FROM profiles WHERE artist_key = ?", (artist_key,)
        ).fetchone()
        if row is None:
            return None
        return ArtistProfile.from_dict(json.loads(row["profile"]))

    def has_complete(self, artist_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM profiles WHERE artist_key = ? AND status = ?",
            (artist_key, STATUS_COMPLETE),
        ).fetchone()
        return row is not None

    def iter_profiles(self) -> Iterator[ArtistProfile]:
        cur = self._conn.execute("SELECT profile FROM profiles ORDER BY artist_key")
        for row in cur:
            yield ArtistProfile.from_dict(json.loads(row["profile"]))

    def load_all(self) -> dict[str, dict[str, Any]]:
        """Return {artist_key: api_profile} for a fast read-time join.

        Used by the UI/atlas warm-up: one query, compact objects, ready to
        splice into ``/api/artist/<id>``.
        """
        out: dict[str, dict[str, Any]] = {}
        for prof in self.iter_profiles():
            out[prof.artist_key] = prof.to_api_profile()
        return out

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0])

    def status_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM profiles GROUP BY status"
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    # -- raw payload cache ----------------------------------------------
    def cache_raw(self, source: str, source_id: str, kind: str,
                  payload: Any, fetched_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO raw_cache (source, source_id, kind, payload, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source, str(source_id), kind, json.dumps(payload), fetched_at),
        )
        self._conn.commit()

    def get_raw(self, source: str, source_id: str, kind: str) -> Any | None:
        row = self._conn.execute(
            "SELECT payload FROM raw_cache WHERE source = ? AND source_id = ? AND kind = ?",
            (source, str(source_id), kind),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    # -- review queue ----------------------------------------------------
    def queue_review(self, artist_key: str, name: str, reason: str,
                     candidates: list[dict[str, Any]], created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO review_queue (artist_key, name, reason, candidates, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (artist_key, name, reason, json.dumps(candidates), created_at),
        )
        self._conn.commit()

    def dequeue_review(self, artist_key: str) -> None:
        """Drop an artist from the review queue — called when it later resolves
        to a non-ambiguous status, so a rescued match doesn't linger as a stale
        review row."""
        self._conn.execute("DELETE FROM review_queue WHERE artist_key = ?", (artist_key,))
        self._conn.commit()

    def review_rows(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT artist_key, name, reason, candidates, created_at FROM review_queue "
            "ORDER BY created_at"
        )
        out = []
        for r in cur:
            out.append({"artist_key": r["artist_key"], "name": r["name"],
                        "reason": r["reason"], "candidates": json.loads(r["candidates"]),
                        "created_at": r["created_at"]})
        return out

    def review_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0])


def open_readonly(path: str | Path = DEFAULT_DB_PATH) -> "ArtistProfileStore | None":
    """Open an existing store for the read path, or ``None`` if it isn't there.

    The atlas warm-up calls this: a missing profiles DB is not an error, it just
    means no profile fields are surfaced yet.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return ArtistProfileStore(p)
    except sqlite3.Error:
        return None
