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

import random
from pathlib import Path

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
