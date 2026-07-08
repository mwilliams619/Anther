"""
anther_ml.mpd_sql — MySQL ``mysqldump`` of the MPD/Spotify DB → queryable SQLite
→ a sampled, artist-capped corpus query set.

Why this exists
---------------
The corpus builder historically read the RecSys ``mpd.slice.*.json`` files (see
``mpd_ingest.py``). The dataset we actually have is a single ~10 GB **MySQL dump**
(``spotifydbdumpshare.sql``) with a normalized schema::

    album(id, name, uri)
    artist(id, name, uri)
    playlist(id, name, followers, uri, total_tracks)
    track(id, name, duration, popularity, explicit, preview_url, uri, album_id)
    track_artist1(track_id, artist_id)          -- 17.5M rows
    track_playlist1(track_id, playlist_id)       -- 125M  rows

Crucially, the ``track`` table carries a **``preview_url``** (a Spotify
``p.scdn.co`` 30 s preview) directly — so, unlike the JSON path, no Deezer/ISRC
matching is required to get audio; the URL is fetched directly (with an optional
Deezer fallback in ``sources.sql_source``).

Two stages, both here
---------------------
1. :func:`load_dump_to_sqlite` — a streaming loader. ``mysqldump`` writes one
   ``INSERT INTO `<table>` VALUES (...),(...),...;`` **per line**, so we iterate
   lines, parse the extended-INSERT tuples for the tables we need, and bulk-insert
   into a fresh SQLite file. It is:
     * **resumable** — the last committed byte offset is recorded; a re-run seeks
       past it, and every insert is ``INSERT OR IGNORE`` (primary/unique keys make
       re-processing idempotent), so a crash never corrupts or duplicates.
     * **logged** — progress per N rows via the stdlib ``logging`` module.
     * **robust** — a malformed line is logged and skipped, not fatal.
2. :func:`sample_tracks` — pull a deterministic, artist-capped random sample of
   tracks that have a preview, plus each sampled track's playlist membership.
   The heavy lifting (filter + per-artist cap + sample) runs inside SQLite, so
   memory stays flat regardless of the 13M-row table.

Because 13.3M tracks is far past the corpus design's 10k–100k sweet spot (exact
O(N²) dedupe and exact-cosine ``SongIndex` do not scale past ~100k), the intended
use is always to *sample down* — never to embed the whole dump.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger("anther_ml.mpd_sql")

# Tables we load and the source-column indices we keep. mysqldump emits every
# column positionally in the CREATE-TABLE order (see module docstring); we pluck
# only what the corpus needs and give each a trimmed SQLite schema.
_INSERT_PREFIX = "INSERT INTO `"

# name -> (create_sql, [(src_col_index, dest_col_name), ...])
TABLE_SPEC: dict[str, tuple[str, list[tuple[int, str]]]] = {
    "artist": (
        "CREATE TABLE IF NOT EXISTS artist (id TEXT PRIMARY KEY, name TEXT)",
        [(0, "id"), (1, "name")],
    ),
    "playlist": (
        "CREATE TABLE IF NOT EXISTS playlist (id TEXT PRIMARY KEY, name TEXT)",
        [(0, "id"), (1, "name")],
    ),
    "track": (
        "CREATE TABLE IF NOT EXISTS track ("
        "id TEXT PRIMARY KEY, name TEXT, popularity REAL, duration INTEGER, "
        "preview_url TEXT, album_id TEXT)",
        [(0, "id"), (1, "name"), (3, "popularity"), (2, "duration"),
         (5, "preview_url"), (7, "album_id")],
    ),
    "track_artist1": (
        "CREATE TABLE IF NOT EXISTS track_artist1 ("
        "track_id TEXT, artist_id TEXT, UNIQUE(track_id, artist_id))",
        [(0, "track_id"), (1, "artist_id")],
    ),
    "track_playlist1": (
        "CREATE TABLE IF NOT EXISTS track_playlist1 ("
        "track_id TEXT, playlist_id TEXT, UNIQUE(track_id, playlist_id))",
        [(0, "track_id"), (1, "playlist_id")],
    ),
}

# Indices built once the load completes — the joins sample_tracks() needs.
_POST_LOAD_INDICES = [
    "CREATE INDEX IF NOT EXISTS ix_ta_track ON track_artist1(track_id)",
    "CREATE INDEX IF NOT EXISTS ix_tp_track ON track_playlist1(track_id)",
]

# MySQL string escapes → their literal characters.
_ESCAPES = {
    "0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t",
    "Z": "\x1a", "\\": "\\", "'": "'", '"': '"',
}


# ──────────────────────────────────────────────────────────────────────────
# Extended-INSERT parsing
# ──────────────────────────────────────────────────────────────────────────
def _unescape(s: str) -> str:
    """Resolve MySQL backslash escapes inside an already-unquoted string."""
    if "\\" not in s:
        return s
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _iter_tuples(values: str) -> Iterator[str]:
    """
    Yield each ``(...)`` tuple body from the VALUES portion of an INSERT, quote-
    and escape-aware so commas/parens inside a string literal don't split it.
    """
    i, n = 0, len(values)
    while i < n:
        while i < n and values[i] != "(":
            i += 1
        if i >= n:
            return
        i += 1  # past '('
        buf: list[str] = []
        in_str = False
        while i < n:
            c = values[i]
            if in_str:
                if c == "\\" and i + 1 < n:
                    buf.append(c)
                    buf.append(values[i + 1])
                    i += 2
                    continue
                buf.append(c)
                i += 1
                if c == "'":
                    in_str = False
                continue
            if c == "'":
                in_str = True
                buf.append(c)
                i += 1
                continue
            if c == ")":
                i += 1
                yield "".join(buf)
                break
            buf.append(c)
            i += 1


def _split_fields(body: str) -> list[object]:
    """Split one tuple body into coerced Python values (str / int / float / None)."""
    fields: list[str] = []
    buf: list[str] = []
    in_str = False
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                buf.append(c)
                buf.append(body[i + 1])
                i += 2
                continue
            buf.append(c)
            i += 1
            if c == "'":
                in_str = False
            continue
        if c == "'":
            in_str = True
            buf.append(c)
            i += 1
            continue
        if c == ",":
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    fields.append("".join(buf))
    return [_coerce(f) for f in fields]


def _coerce(raw: str) -> object:
    raw = raw.strip()
    if raw == "NULL" or raw == "":
        return None if raw == "NULL" else ""
    if raw[0] == "'" and raw[-1] == "'":
        return _unescape(raw[1:-1])
    # Numeric literal — leave as string; SQLite column affinity coerces on insert.
    return raw


def parse_insert_line(line: str) -> tuple[str, list[list[object]]] | None:
    """
    Parse one ``INSERT INTO `tbl` VALUES ...`` line into ``(table, [rows])`` for a
    table in :data:`TABLE_SPEC`, projecting to that table's kept columns. Returns
    ``None`` for any other line (comments, DDL, unrelated tables).
    """
    if not line.startswith(_INSERT_PREFIX):
        return None
    table, sep, values = line[len(_INSERT_PREFIX):].partition("` VALUES ")
    if not sep or table not in TABLE_SPEC:
        return None
    _, cols = TABLE_SPEC[table]
    max_idx = cols[-1][0] if table != "track" else 7  # track keeps col 7
    rows: list[list[object]] = []
    for body in _iter_tuples(values):
        fields = _split_fields(body)
        if len(fields) <= max_idx:
            log.warning("skipping malformed %s tuple (%d fields)", table, len(fields))
            continue
        rows.append([fields[src] for src, _ in cols])
    return table, rows


# ──────────────────────────────────────────────────────────────────────────
# Streaming load: mysqldump → SQLite
# ──────────────────────────────────────────────────────────────────────────
def _connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF")      # restartable, so durability is moot
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=-262144")    # ~256 MB page cache
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def _init_schema(con: sqlite3.Connection, with_membership: bool) -> None:
    for name, (create_sql, _) in TABLE_SPEC.items():
        if name == "track_playlist1" and not with_membership:
            continue
        con.execute(create_sql)
    con.execute(
        "CREATE TABLE IF NOT EXISTS _load_progress ("
        "id INTEGER PRIMARY KEY CHECK (id = 0), byte_offset INTEGER, done INTEGER)"
    )
    con.execute(
        "INSERT OR IGNORE INTO _load_progress (id, byte_offset, done) VALUES (0, 0, 0)"
    )
    con.commit()


def _progress(con: sqlite3.Connection) -> tuple[int, bool]:
    row = con.execute(
        "SELECT byte_offset, done FROM _load_progress WHERE id = 0"
    ).fetchone()
    return (0, False) if row is None else (int(row[0]), bool(row[1]))


def is_loaded(db_path: str | Path) -> bool:
    """True iff ``db_path`` exists and holds a completed load."""
    p = Path(db_path)
    if not p.exists():
        return False
    try:
        con = sqlite3.connect(str(p))
        try:
            _, done = _progress(con)
            return done
        finally:
            con.close()
    except sqlite3.Error:
        return False


def load_dump_to_sqlite(
    sql_path: str | Path,
    db_path: str | Path,
    *,
    with_membership: bool = True,
    batch_size: int = 20000,
    commit_every: int = 200000,
    resume: bool = True,
    force: bool = False,
    log_every: int = 1000000,
) -> Path:
    """
    Stream ``sql_path`` (a mysqldump) into the SQLite file ``db_path``.

    Idempotent and resumable: the last committed byte offset is persisted, so a
    re-run seeks past already-loaded content, and all inserts are
    ``INSERT OR IGNORE``. Pass ``force=True`` to rebuild from scratch, or
    ``with_membership=False`` to skip the 125M-row ``track_playlist1`` table
    (much faster; playlist-fit workflow then unavailable).

    Returns ``db_path`` on success. Raises ``FileNotFoundError`` if the dump is
    missing.
    """
    sql_path = Path(sql_path)
    db_path = Path(db_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL dump not found: {sql_path}")

    if force and db_path.exists():
        log.info("force: removing existing DB %s", db_path)
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(db_path)
    try:
        _init_schema(con, with_membership)
        start_offset, already_done = _progress(con)
        if already_done and not force:
            log.info("DB %s already fully loaded — nothing to do", db_path)
            return db_path
        if not resume:
            start_offset = 0
            con.execute("UPDATE _load_progress SET byte_offset = 0, done = 0 WHERE id = 0")
            con.commit()

        wanted = set(TABLE_SPEC)
        if not with_membership:
            wanted.discard("track_playlist1")
        insert_sql = {
            name: "INSERT OR IGNORE INTO {t} VALUES ({q})".format(
                t=name, q=",".join("?" * len(cols))
            )
            for name, (_, cols) in TABLE_SPEC.items()
        }

        batches: dict[str, list[list[object]]] = {name: [] for name in wanted}
        counts: dict[str, int] = {name: 0 for name in wanted}
        total_since_commit = 0
        t0 = time.time()

        def flush(name: str) -> None:
            if batches[name]:
                con.executemany(insert_sql[name], batches[name])
                batches[name].clear()

        with open(sql_path, "rb") as f:
            if start_offset:
                f.seek(start_offset)
                log.info("resuming load at byte offset %d", start_offset)
            offset = start_offset
            for raw in f:
                offset += len(raw)
                try:
                    line = raw.decode("utf-8", "replace")
                except Exception:  # noqa: BLE001 — never let one line kill the load
                    continue
                if not line.startswith(_INSERT_PREFIX):
                    continue
                # Cheap table-name check *before* parsing tuples, so an unwanted
                # table (e.g. the 125M-row track_playlist1 under --no-membership)
                # is skipped without paying to parse rows we'd only discard.
                table = line[len(_INSERT_PREFIX):].partition("` VALUES ")[0]
                if table not in wanted:
                    continue
                try:
                    parsed = parse_insert_line(line)
                except Exception as e:  # noqa: BLE001
                    log.warning("parse error (skipped): %s", e)
                    continue
                if parsed is None:
                    continue
                table, rows = parsed
                if not rows:
                    continue
                batches[table].extend(rows)
                counts[table] += len(rows)
                total_since_commit += len(rows)
                if len(batches[table]) >= batch_size:
                    flush(table)
                if total_since_commit >= commit_every:
                    for name in wanted:
                        flush(name)
                    con.execute(
                        "UPDATE _load_progress SET byte_offset = ? WHERE id = 0",
                        (offset,),
                    )
                    con.commit()
                    total_since_commit = 0
                    done_total = sum(counts.values())
                    if done_total % max(log_every, commit_every) < commit_every:
                        rate = done_total / max(time.time() - t0, 1e-6)
                        log.info(
                            "loaded %s rows (%.0f/s) — %s",
                            f"{done_total:,}", rate,
                            ", ".join(f"{k}={v:,}" for k, v in counts.items()),
                        )

            for name in wanted:
                flush(name)
            con.execute(
                "UPDATE _load_progress SET byte_offset = ? WHERE id = 0", (offset,)
            )
            con.commit()

        log.info("row counts: %s", ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        log.info("building indices…")
        for stmt in _POST_LOAD_INDICES:
            if "track_playlist1" in stmt and not with_membership:
                continue
            con.execute(stmt)
        # Mark done only after indices exist, so is_loaded() never advertises a
        # DB that would query without them.
        con.execute("UPDATE _load_progress SET done = 1 WHERE id = 0")
        con.commit()
        log.info("load complete → %s (%.1f min)", db_path, (time.time() - t0) / 60)
        return db_path
    finally:
        con.close()


def ensure_db(
    *,
    db_path: str | Path | None = None,
    sql_dump: str | Path | None = None,
    with_membership: bool = True,
    force: bool = False,
) -> Path:
    """
    Return a ready-to-query SQLite path, building it from ``sql_dump`` if needed.

    If ``db_path`` is omitted it defaults to ``sql_dump`` with a ``.sqlite``
    suffix. If the DB is already fully loaded (and not ``force``), it is reused.
    """
    if db_path is None:
        if sql_dump is None:
            raise ValueError("ensure_db needs db_path or sql_dump")
        db_path = Path(sql_dump).with_suffix(".sqlite")
    db_path = Path(db_path)
    if is_loaded(db_path) and not force:
        log.info("using existing SQLite DB %s", db_path)
        return db_path
    if sql_dump is None:
        raise FileNotFoundError(
            f"{db_path} is not a completed load and no sql_dump was given to build it"
        )
    return load_dump_to_sqlite(
        sql_dump, db_path, with_membership=with_membership, force=force
    )


# ──────────────────────────────────────────────────────────────────────────
# Sampling: DB → artist-capped, deterministic track sample + membership
# ──────────────────────────────────────────────────────────────────────────
def _seeded_rand_fn(seed: int):
    """A deterministic [0,1) hash of a track id, salted by ``seed``."""
    import hashlib

    salt = f"{seed}:".encode()

    def rnd(track_id: object) -> float:
        h = hashlib.md5(salt + str(track_id).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2.0**64

    return rnd


def _chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def sample_tracks(
    db_path: str | Path,
    *,
    sample_n: int | None = 8000,
    artist_cap: int | None = 5,
    seed: int = 42,
    min_popularity: float | None = None,
    require_preview: bool = True,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Deterministic, artist-capped random sample of tracks, plus playlist membership.

    Returns ``(rows, membership)`` where each row is
    ``{"track_id", "name", "preview_url", "artist_name"}`` and ``membership`` maps
    ``track_id -> [{"pid", "name"}, ...]``. The filter (preview present, optional
    popularity floor), the per-artist cap, and the sample all run inside SQLite;
    only the ~``sample_n`` chosen rows are materialized in Python.

    The cap mirrors the corpus design's hygiene rule (no single artist forms a
    fake dense cluster) and is applied *before* embedding, so capped tracks cost
    zero GPU time.
    """
    con = sqlite3.connect(str(db_path))
    try:
        con.create_function("seeded_rand", 1, _seeded_rand_fn(seed), deterministic=True)

        where = ["1=1"]
        params: dict[str, object] = {}
        if require_preview:
            where.append("t.preview_url IS NOT NULL AND t.preview_url <> ''")
        if min_popularity is not None:
            where.append("t.popularity >= :minpop")
            params["minpop"] = min_popularity
        where_sql = " AND ".join(where)

        # One representative artist per track (MIN is arbitrary but deterministic).
        candidate = f"""
            SELECT t.id AS track_id, t.name AS name, t.preview_url AS preview_url,
                   MIN(ta.artist_id) AS artist_id
            FROM track t
            JOIN track_artist1 ta ON ta.track_id = t.id
            WHERE {where_sql}
            GROUP BY t.id
        """
        cap = artist_cap if artist_cap is not None else 1_000_000_000
        params["cap"] = cap
        limit_sql = ""
        if sample_n is not None:
            params["n"] = sample_n
            limit_sql = "LIMIT :n"

        query = f"""
            WITH cand AS ({candidate}),
                 ranked AS (
                     SELECT cand.*,
                            ROW_NUMBER() OVER (
                                PARTITION BY artist_id ORDER BY seeded_rand(track_id)
                            ) AS rn
                     FROM cand
                 )
            SELECT r.track_id, r.name, r.preview_url, a.name AS artist_name
            FROM ranked r
            LEFT JOIN artist a ON a.id = r.artist_id
            WHERE r.rn <= :cap
            ORDER BY seeded_rand(r.track_id)
            {limit_sql}
        """
        try:
            cur = con.execute(query, params)
        except sqlite3.OperationalError as e:
            # Old SQLite without window functions → uncapped fallback.
            log.warning("windowed sample failed (%s); falling back to uncapped sample", e)
            fallback = f"""
                SELECT t.id, t.name, t.preview_url,
                       (SELECT a.name FROM track_artist1 ta JOIN artist a
                        ON a.id = ta.artist_id WHERE ta.track_id = t.id LIMIT 1)
                FROM track t
                WHERE {where_sql}
                ORDER BY seeded_rand(t.id)
                {limit_sql}
            """
            cur = con.execute(fallback, params)

        rows = [
            {"track_id": tid, "name": name, "preview_url": purl, "artist_name": artist}
            for tid, name, purl, artist in cur.fetchall()
        ]
        log.info("sampled %d tracks (cap=%s, seed=%d)", len(rows), artist_cap, seed)

        membership = _fetch_membership(con, [r["track_id"] for r in rows])
        return rows, membership
    finally:
        con.close()


def _fetch_membership(
    con: sqlite3.Connection, track_ids: list[str]
) -> dict[str, list[dict]]:
    """Playlist membership for the given tracks, or ``{}`` if the DB has no
    ``track_playlist1`` table (built with ``--no-membership``)."""
    from collections import defaultdict

    has_table = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='track_playlist1'"
    ).fetchone()
    if not has_table or not track_ids:
        return {}
    membership: dict[str, list[dict]] = defaultdict(list)
    for chunk in _chunks(track_ids, 900):
        placeholders = ",".join("?" * len(chunk))
        cur = con.execute(
            f"SELECT tp.track_id, p.id, p.name FROM track_playlist1 tp "
            f"JOIN playlist p ON p.id = tp.playlist_id "
            f"WHERE tp.track_id IN ({placeholders})",
            chunk,
        )
        for tid, pid, pname in cur:
            membership[tid].append({"pid": pid, "name": pname})
    return dict(membership)


# ──────────────────────────────────────────────────────────────────────────
# CLI: build the DB standalone (also runnable via the corpus builder)
# ──────────────────────────────────────────────────────────────────────────
def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m anther_ml.mpd_sql",
        description="Load a MySQL MPD dump into a queryable SQLite DB.",
    )
    p.add_argument("--dump", required=True, help="path to spotifydbdumpshare.sql")
    p.add_argument("--db", default=None, help="output .sqlite (default: dump + .sqlite)")
    p.add_argument("--no-membership", action="store_true",
                   help="skip the 125M-row track_playlist1 table (much faster)")
    p.add_argument("--force", action="store_true", help="rebuild from scratch")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db = args.db or str(Path(args.dump).with_suffix(".sqlite"))
    load_dump_to_sqlite(
        args.dump, db, with_membership=not args.no_membership, force=args.force
    )


if __name__ == "__main__":
    main()
