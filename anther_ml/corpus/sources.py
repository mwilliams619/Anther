"""
Track sources for the corpus builder.

Every source yields plain dicts under one contract:

    {
        "id":        str   — stable, unique (checkpoint/resume key)
        "name":      str
        "artist":    str | None
        "source":    str   — "fma" | "local" | "spotify_deezer" | ...
        "genre":     str | None   — DISPLAY ONLY, never a clustering input
        "playlists": list[{"pid", "name"}]  — design §5 membership tags
        and exactly one of:
        "path":      str   — audio file on disk
        "audio":     np.ndarray — mono waveform (+ "sr": int, 24 kHz)
    }

Composition is the corpus's most important design axis (design §3A): sources
apply the per-artist cap *before* embedding so capped tracks cost zero GPU
time. Near-duplicate removal needs vectors, so it lives post-embedding in
build.py.
"""

import logging
import random
from pathlib import Path

log = logging.getLogger("anther_ml.corpus.sources")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff"}

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"


def _cap_by_artist(items: list[dict], cap: int) -> list[dict]:
    """Keep at most ``cap`` items per (lowercased) artist, preserving order."""
    counts: dict = {}
    kept = []
    for item in items:
        artist = (item.get("artist") or "").strip().lower()
        counts[artist] = counts.get(artist, 0) + 1
        if counts[artist] <= cap:
            kept.append(item)
    return kept


def fma_source(
    audio_dir: str | Path = "data/audio/fma_small",
    metadata_dir: str | Path = "data/fma_metadata",
    subset: str = "small",
    artist_cap: int | None = None,
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    FMA tracks with title/artist/genre metadata. Seeded shuffle so a
    ``limit``-ed smoke build samples across the collection (FMA is ordered by
    track id, which correlates with upload era) yet stays deterministic per
    seed — a resumed build sees the same tracks.
    """
    from ..data import SUBSET_ORDER, get_audio_path, load_fma_tracks

    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir() or not any(audio_dir.iterdir()):
        raise FileNotFoundError(
            f"FMA audio not found at {audio_dir}. Download and unzip "
            f"{FMA_SMALL_URL} (7.2 GB) there, or use --source local."
        )

    import pandas as pd

    tracks = load_fma_tracks(metadata_dir)
    subset_col = pd.Categorical(
        tracks[("set", "subset")], categories=SUBSET_ORDER, ordered=True
    )
    track_ids = list(tracks.index[subset_col <= subset])
    if shuffle:
        random.Random(seed).shuffle(track_ids)

    items = []
    for tid in track_ids:
        path = get_audio_path(audio_dir, tid)
        if not path.exists():
            continue
        row = tracks.loc[tid]
        genre = row.get(("track", "genre_top"))
        items.append(
            {
                "id": f"fma:{tid}",
                "name": str(row.get(("track", "title")) or f"track {tid}"),
                "artist": str(row.get(("artist", "name")) or "") or None,
                "source": "fma",
                "genre": str(genre) if isinstance(genre, str) else None,
                "playlists": [],
                "path": str(path),
            }
        )
    if artist_cap is not None:
        items = _cap_by_artist(items, artist_cap)
    yield from items[:limit]


def local_source(
    audio_dir: str | Path,
    source_name: str = "local",
    limit: int | None = None,
):
    """
    Walk a folder of audio files (recursive, sorted, macOS ``._*`` junk
    skipped). Filename stem becomes the track name; no artist/genre metadata.
    This is what makes a smoke build runnable without any dataset download.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"no such audio directory: {audio_dir}")
    paths = sorted(
        p
        for p in audio_dir.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("._")
    )
    n = 0
    for p in paths:
        if limit is not None and n >= limit:
            return
        n += 1
        yield {
            "id": f"{source_name}:{p.relative_to(audio_dir).as_posix()}",
            "name": p.stem,
            "artist": None,
            "source": source_name,
            "genre": None,
            "playlists": [],
            "path": str(p),
        }


def mpd_source(
    mpd_dir: str | Path,
    sample_n: int | None = 8000,
    tracks_per_artist_cap: int | None = 5,
    max_slices: int | None = None,
    enrich: bool = True,
    clip_seconds: float | None = 30.0,
    seed: int = 42,
    min_ratio: float = 0.82,
):
    """
    Million Playlist Dataset tracks resolved to Deezer 30 s previews, streamed
    one waveform at a time (materializing 8k previews up front would hold
    ~23 GB in RAM). Playlist membership rides along on each item — the
    metadata the playlist-fit workflow (design §5) requires. Unmatched tracks
    are skipped with a count printed at the end.
    """
    from ..embedding import SR
    from ..mpd_ingest import build_mpd_queries, enrich_isrc
    from ..spotify_deezer import (
        fetch_preview_waveform,
        match_deezer_track,
        spotify_track_to_query,
    )

    objs, membership = build_mpd_queries(
        mpd_dir,
        max_slices=max_slices,
        sample_n=sample_n,
        tracks_per_artist_cap=tracks_per_artist_cap,
        seed=seed,
    )
    if enrich:
        enrich_isrc(objs)

    n_missed = 0
    for obj in objs:
        query = spotify_track_to_query(obj)
        if not query or not query.get("sp_id"):
            n_missed += 1
            continue
        match = match_deezer_track(query, min_ratio=min_ratio)
        if "error" in match:
            n_missed += 1
            continue
        try:
            wav, _ = fetch_preview_waveform(
                match["preview"], target_sr=SR, clip_seconds=clip_seconds
            )
        except Exception:  # noqa: BLE001 — network flake on one preview
            n_missed += 1
            continue
        yield {
            "id": f"spotify:{query['sp_id']}",
            "name": query["title"],
            "artist": query["artist"] or None,
            "source": "spotify_deezer",
            "genre": None,
            "playlists": membership.get(query["sp_id"], []),
            "audio": wav,
            "sr": SR,
        }
    if n_missed:
        print(f"mpd_source: {n_missed} tracks had no usable preview (skipped)")


def sql_source(
    db_path: str | Path | None = None,
    sql_dump: str | Path | None = None,
    sample_n: int | None = 8000,
    tracks_per_artist_cap: int | None = 5,
    seed: int = 42,
    min_popularity: float | None = None,
    with_membership: bool = True,
    prefer_spotify_preview: bool = True,
    deezer_fallback: bool = True,
    clip_seconds: float | None = 30.0,
    min_ratio: float = 0.82,
    n_workers: int = 12,
):
    """
    MPD tracks from the **MySQL dump** (``spotifydbdumpshare.sql``), streamed one
    waveform at a time. This is the SQL analogue of :func:`mpd_source`; it reads
    the normalized ``track``/``artist``/``playlist`` tables via
    :mod:`anther_ml.mpd_sql` instead of ``mpd.slice.*.json`` files.

    Audio comes from the Spotify ``preview_url`` stored on each ``track`` row —
    fetched directly, so no Deezer/ISRC matching is needed (``deezer_fallback``
    covers dead/expired preview URLs). ``db_path`` is built from ``sql_dump`` on
    first use if it doesn't exist yet (see :func:`mpd_sql.ensure_db`).

    Sampling is deterministic per ``seed``, artist-capped before embedding, and
    carries playlist membership — the same contract the JSON path provides.

    Previews are fetched with a bounded ``n_workers``-thread pool (Tier 1B): the
    fetch/decode is I/O-bound, so overlapping it keeps the downstream GPU fed.
    Results are yielded in sample order (deterministic), so a resumed build and
    the dedupe "first occurrence wins" rule stay stable.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..embedding import SR
    from ..mpd_sql import ensure_db, sample_tracks
    from ..spotify_deezer import fetch_preview_waveform, match_deezer_track

    db_path = ensure_db(
        db_path=db_path, sql_dump=sql_dump, with_membership=with_membership
    )
    rows, membership = sample_tracks(
        db_path,
        sample_n=sample_n,
        artist_cap=tracks_per_artist_cap,
        seed=seed,
        min_popularity=min_popularity,
    )
    log.info(
        "sql_source: fetching audio for %d sampled tracks (%d workers)…",
        len(rows), n_workers,
    )

    def fetch_one(row: dict) -> dict | None:
        track_id, name = row["track_id"], row["name"] or ""
        artist_name, preview_url = row["artist_name"], row["preview_url"]

        wav = None
        if prefer_spotify_preview and preview_url:
            try:
                wav, _ = fetch_preview_waveform(
                    preview_url, target_sr=SR, clip_seconds=clip_seconds
                )
            except Exception as e:  # noqa: BLE001 — dead/expired URL → try fallback
                log.debug("spotify preview failed for %s: %s", track_id, e)
                wav = None

        if wav is None and deezer_fallback:
            match = match_deezer_track(
                {"sp_id": track_id, "isrc": None, "title": name,
                 "artist": artist_name or "", "duration_ms": None},
                min_ratio=min_ratio,
            )
            if "error" not in match:
                try:
                    wav, _ = fetch_preview_waveform(
                        match["preview"], target_sr=SR, clip_seconds=clip_seconds
                    )
                except Exception:  # noqa: BLE001 — network flake on one preview
                    wav = None

        if wav is None:
            return None
        return {
            "id": f"spotify:{track_id}",
            "name": name,
            "artist": artist_name or None,
            "source": "mpd_sql",
            "genre": None,
            "playlists": membership.get(track_id, []),
            "audio": wav,
            "sr": SR,
        }

    # Ordered, bounded parallelism: process in windows so at most ~one window of
    # fetched waveforms is buffered at a time (memory), while up to n_workers
    # fetches run concurrently.
    n_missed = 0
    window = max(n_workers * 4, 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for start in range(0, len(rows), window):
            for item in pool.map(fetch_one, rows[start : start + window]):
                if item is None:
                    n_missed += 1
                    continue
                yield item

    if n_missed:
        log.warning(
            "sql_source: %d/%d tracks had no usable preview (skipped)",
            n_missed, len(rows),
        )


def sql_source_oversampled(
    db_path: str | Path | None = None,
    sql_dump: str | Path | None = None,
    target_n: int = 8000,
    oversample_ratio: float = 1.15,
    tracks_per_artist_cap: int | None = 5,
    seed: int = 42,
    min_popularity: float | None = None,
    with_membership: bool = True,
    prefer_spotify_preview: bool = True,
    deezer_fallback: bool = True,
    clip_seconds: float | None = 30.0,
    min_ratio: float = 0.82,
    n_workers: int = 12,
):
    """
    Like :func:`sql_source`, but guarantees ``target_n`` *successfully fetched*
    tracks instead of yielding whatever a fixed sample happens to resolve.

    Dead/expired preview URLs (and Deezer-fallback misses) are the one source
    of yield loss in the SQL path. Rather than accept a shrunken corpus or add
    gap-handling downstream, this draws a larger deterministic candidate pool
    up front (``target_n * oversample_ratio``, same seed → same prefix as a
    plain ``sql_source`` call) and substitutes the next unused candidate,
    in sampled order, whenever a fetch fails — so the *set* of tracks shifts
    slightly at the margin but the *count* is exact and no per-track fallback
    logic is needed by callers. Raises if the oversampled pool itself runs out
    before reaching ``target_n`` (i.e. the dead-URL rate exceeds the buffer).

    Fetches run with the same bounded ``n_workers``-thread pool as
    ``sql_source``, processed in deterministic sample order so a resumed
    build stays stable (the checkpoint's ``done_ids()`` just skips whatever
    already landed).
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..embedding import SR
    from ..mpd_sql import ensure_db, sample_tracks
    from ..spotify_deezer import fetch_preview_waveform, match_deezer_track

    db_path = ensure_db(
        db_path=db_path, sql_dump=sql_dump, with_membership=with_membership
    )
    pool_n = int(target_n * oversample_ratio)
    rows, membership = sample_tracks(
        db_path,
        sample_n=pool_n,
        artist_cap=tracks_per_artist_cap,
        seed=seed,
        min_popularity=min_popularity,
    )
    log.info(
        "sql_source_oversampled: target=%d, pool=%d candidates (%d workers)…",
        target_n, len(rows), n_workers,
    )

    def fetch_one(row: dict) -> dict | None:
        track_id, name = row["track_id"], row["name"] or ""
        artist_name, preview_url = row["artist_name"], row["preview_url"]

        wav = None
        if prefer_spotify_preview and preview_url:
            try:
                wav, _ = fetch_preview_waveform(
                    preview_url, target_sr=SR, clip_seconds=clip_seconds
                )
            except Exception as e:  # noqa: BLE001 — dead/expired URL → try fallback
                log.debug("spotify preview failed for %s: %s", track_id, e)
                wav = None

        if wav is None and deezer_fallback:
            match = match_deezer_track(
                {"sp_id": track_id, "isrc": None, "title": name,
                 "artist": artist_name or "", "duration_ms": None},
                min_ratio=min_ratio,
            )
            if "error" not in match:
                try:
                    wav, _ = fetch_preview_waveform(
                        match["preview"], target_sr=SR, clip_seconds=clip_seconds
                    )
                except Exception:  # noqa: BLE001 — network flake on one preview
                    wav = None

        if wav is None:
            return None
        return {
            "id": f"spotify:{track_id}",
            "name": name,
            "artist": artist_name or None,
            "source": "mpd_sql",
            "genre": None,
            "playlists": membership.get(track_id, []),
            "audio": wav,
            "sr": SR,
        }

    # Ordered, bounded parallelism over the *whole* oversampled pool, but we
    # stop pulling from the pool as soon as target_n successes have yielded —
    # later candidates in a partially-consumed window are simply never
    # submitted, so no wasted fetches beyond the window in flight.
    n_yielded = 0
    n_missed = 0
    window = max(n_workers * 4, 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for start in range(0, len(rows), window):
            if n_yielded >= target_n:
                break
            chunk = rows[start : start + window]
            for item in pool.map(fetch_one, chunk):
                if n_yielded >= target_n:
                    break
                if item is None:
                    n_missed += 1
                    continue
                n_yielded += 1
                yield item

    if n_yielded < target_n:
        raise RuntimeError(
            f"sql_source_oversampled: only {n_yielded}/{target_n} tracks "
            f"fetched from a pool of {len(rows)} candidates ({n_missed} "
            f"failed). Raise oversample_ratio (currently {oversample_ratio}) "
            f"and retry."
        )
    log.info(
        "sql_source_oversampled: reached target_n=%d (%d candidates skipped/failed, "
        "%d unused pool remainder)",
        target_n, n_missed, len(rows) - n_yielded - n_missed,
    )
