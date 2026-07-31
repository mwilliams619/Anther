"""
anther_ml.mpd_ingest — Spotify Million Playlist Dataset (MPD) → corpus queries.

Adapts the RecSys-2018 Spotify **Million Playlist Dataset** into the shape the
existing ``spotify_deezer`` pipeline already consumes, so MPD tracks flow
straight into ``load_spotify_via_deezer`` (Deezer preview → 24 kHz waveform)
with no changes to that module.

Why an adapter is needed
------------------------
``spotify_deezer.spotify_track_to_query`` expects a Spotify **track object**::

    {"id", "name", "artists": [{"name": ...}], "duration_ms",
     "external_ids": {"isrc": ...}}

MPD track objects are flatter and, crucially, carry **no ISRC**::

    {"track_name", "artist_name", "track_uri", "album_name",
     "artist_uri", "album_uri", "duration_ms", "pos"}

This module (a) reshapes MPD tracks into Spotify-track objects so the existing
orchestrator handles them unchanged, (b) preserves **playlist membership**
(the one metadata field the EP-to-playlist workflow needs — see
REFERENCE_CORPUS_DESIGN.md §5), and (c) optionally **enriches ISRC** via the
Spotify Web API so matching can use the high-precision ISRC path instead of
falling back to fuzzy title/artist for every track.

MPD on-disk layout
------------------
The download is a ``data/`` folder of JSON slice files::

    mpd.slice.0-999.json, mpd.slice.1000-1999.json, ...

each shaped ``{"info": {...}, "playlists": [ {playlist}, ... ]}`` where a
playlist is ``{"pid", "name", "num_tracks", "tracks": [ {track}, ... ], ...}``.

Typical usage
-------------
    from anther_ml.mpd_ingest import build_mpd_queries, enrich_isrc, attach_playlist_membership
    from anther_ml.spotify_deezer import load_spotify_via_deezer

    objs, membership = build_mpd_queries(
        "data/mpd/data", sample_n=8000, tracks_per_artist_cap=5, seed=42,
    )
    # optional but recommended — turns fuzzy matches into ISRC matches:
    enrich_isrc(objs, client_id=..., client_secret=...)

    corpus, misses = load_spotify_via_deezer(objs, clip_seconds=30.0)
    attach_playlist_membership(corpus, membership)   # adds 'playlists' to each row

    # corpus rows are now embed-ready (item['audio']) and carry playlist tags.

Nothing here downloads audio or hits Deezer — that stays in ``spotify_deezer``.
Spotify enrichment is the only network call in this module and is optional.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import requests

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"


# ──────────────────────────────────────────────────────────────────────────
# MPD parsing
# ──────────────────────────────────────────────────────────────────────────
def uri_to_id(uri: str | None) -> str | None:
    """'spotify:track:6b2o...' → '6b2o...'.  None-safe."""
    if not uri:
        return None
    return uri.rsplit(":", 1)[-1]


def iter_mpd_playlists(
    mpd_dir: str | Path, max_slices: int | None = None
) -> Iterator[dict]:
    """
    Yield playlist dicts from every ``mpd.slice.*.json`` file in ``mpd_dir``.

    Slices are read in numeric order (by their start index), not lexical, so
    ``max_slices`` gives a deterministic prefix of the dataset.
    """
    files = glob.glob(str(Path(mpd_dir) / "mpd.slice.*.json"))
    if not files:
        raise FileNotFoundError(
            f"No 'mpd.slice.*.json' files under {mpd_dir!r}. Point mpd_dir at the "
            "folder that contains the slice files (usually the dataset's 'data/' dir)."
        )

    def _start_index(path: str) -> int:
        # 'mpd.slice.1000-1999.json' → 1000
        stem = Path(path).name
        try:
            return int(stem.split(".")[2].split("-")[0])
        except (IndexError, ValueError):
            return 0

    files.sort(key=_start_index)
    if max_slices is not None:
        files = files[:max_slices]

    for path in files:
        with open(path) as f:
            slice_obj = json.load(f)
        for pl in slice_obj.get("playlists", []):
            yield pl


def mpd_track_to_spotify_obj(track: dict, isrc: str | None = None) -> dict:
    """
    Reshape one MPD track into a Spotify-track object that
    ``spotify_deezer.spotify_track_to_query`` understands.

    ISRC is left empty unless supplied (MPD has none) — attach it later via
    ``enrich_isrc`` to unlock the ISRC-first Deezer match path.
    """
    return {
        "id": uri_to_id(track.get("track_uri")),
        "name": track.get("track_name") or "",
        "artists": [{"name": track.get("artist_name") or ""}],
        "duration_ms": track.get("duration_ms"),
        "external_ids": ({"isrc": isrc} if isrc else {}),
    }


# ──────────────────────────────────────────────────────────────────────────
# Build a deduped, sampled query set + playlist-membership map
# ──────────────────────────────────────────────────────────────────────────
def build_mpd_queries(
    mpd_dir: str | Path,
    max_slices: int | None = None,
    sample_n: int | None = None,
    tracks_per_artist_cap: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Read MPD slices → unique Spotify-track objects + a playlist-membership map.

    Dedup is by Spotify track id (``track_uri``). Membership is recorded across
    **all** occurrences before any sampling, so a sampled track still knows
    every playlist it belonged to.

    Parameters
    ----------
    max_slices            : cap how many slice files to read (1 slice = 1000
                            playlists). None = all.
    sample_n              : randomly keep this many unique tracks. None = keep all.
    tracks_per_artist_cap : keep at most this many tracks per artist_name — the
                            "cap tracks per artist" hygiene rule from the design
                            doc (prevents one prolific artist forming a fake
                            dense cluster). Applied before sample_n.
    seed                  : RNG seed for reproducible sampling / capping.

    Returns
    -------
    objs        : list of Spotify-track objects (feed to load_spotify_via_deezer)
    membership  : {spotify_track_id: [ {"pid": int, "name": str}, ... ]}
                  — pass to attach_playlist_membership() after the corpus is built.
    """
    rng = random.Random(seed)

    first_seen: dict[str, dict] = {}          # track_id -> mpd track dict
    artist_of: dict[str, str] = {}            # track_id -> artist_name (lower)
    membership: dict[str, list[dict]] = defaultdict(list)
    seen_pl_for_track: set[tuple[str, int]] = set()

    n_playlists = 0
    for pl in iter_mpd_playlists(mpd_dir, max_slices=max_slices):
        n_playlists += 1
        pid = pl.get("pid")
        pname = pl.get("name")
        for tr in pl.get("tracks", []):
            tid = uri_to_id(tr.get("track_uri"))
            if not tid:
                continue
            if tid not in first_seen:
                first_seen[tid] = tr
                artist_of[tid] = (tr.get("artist_name") or "").strip().lower()
            # membership: one entry per (track, playlist)
            key = (tid, pid)
            if key not in seen_pl_for_track:
                seen_pl_for_track.add(key)
                membership[tid].append({"pid": pid, "name": pname})

    if verbose:
        print(f"Read {n_playlists} playlists → {len(first_seen)} unique tracks")

    track_ids = list(first_seen.keys())

    # Cap tracks per artist (hygiene: no single artist dominates the corpus)
    if tracks_per_artist_cap is not None:
        rng.shuffle(track_ids)
        per_artist: dict[str, int] = defaultdict(int)
        capped: list[str] = []
        for tid in track_ids:
            a = artist_of.get(tid, "")
            if per_artist[a] < tracks_per_artist_cap:
                per_artist[a] += 1
                capped.append(tid)
        if verbose:
            print(f"Artist cap ({tracks_per_artist_cap}/artist): "
                  f"{len(track_ids)} → {len(capped)} tracks")
        track_ids = capped

    # Random sample down to sample_n
    if sample_n is not None and sample_n < len(track_ids):
        track_ids = rng.sample(track_ids, sample_n)
        if verbose:
            print(f"Sampled → {len(track_ids)} tracks")

    objs = [mpd_track_to_spotify_obj(first_seen[tid]) for tid in track_ids]
    # Trim membership to the kept tracks only (keeps the returned map small)
    kept = set(track_ids)
    membership = {tid: pls for tid, pls in membership.items() if tid in kept}
    return objs, membership


# ──────────────────────────────────────────────────────────────────────────
# Spotify ISRC enrichment (optional, high-precision matching)
# ──────────────────────────────────────────────────────────────────────────
def get_spotify_token(
    client_id: str | None = None, client_secret: str | None = None
) -> str:
    """
    Client-credentials OAuth token for the Spotify Web API.

    Credentials resolve from args, else env vars SPOTIFY_CLIENT_ID /
    SPOTIFY_CLIENT_SECRET. Create an app at developer.spotify.com to get them.
    Client-credentials scope is enough for GET /tracks (public catalog metadata).
    """
    client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not (client_id and client_secret):
        raise ValueError(
            "Spotify credentials missing. Pass client_id/client_secret or set "
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET env vars."
        )
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {auth}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def enrich_isrc(
    objs: list[dict],
    client_id: str | None = None,
    client_secret: str | None = None,
    token: str | None = None,
    batch_size: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Fill ``external_ids.isrc`` on each Spotify-track object **in place**, using
    batched GET /v1/tracks (up to 50 ids per call).

    Only objects with a Spotify ``id`` and no existing isrc are looked up.
    Handles 429 rate-limits (honors Retry-After) and refreshes an expired token
    once on 401.

    Returns a small stats dict: {'looked_up', 'filled', 'batches'}.
    """
    if token is None:
        token = get_spotify_token(client_id, client_secret)

    todo = [o for o in objs if o.get("id") and not (o.get("external_ids") or {}).get("isrc")]
    by_id = {o["id"]: o for o in todo}
    ids = list(by_id.keys())

    filled = 0
    batches = 0
    i = 0
    while i < len(ids):
        chunk = ids[i : i + batch_size]
        url = f"{SPOTIFY_API}/tracks"
        r = requests.get(
            url,
            params={"ids": ",".join(chunk)},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

        if r.status_code == 401:                      # token expired → refresh once
            token = get_spotify_token(client_id, client_secret)
            continue
        if r.status_code == 429:                      # rate limited → wait and retry
            wait = int(r.headers.get("Retry-After", "2")) + 1
            if verbose:
                print(f"  429 rate-limited; sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()

        for t in r.json().get("tracks", []) or []:
            if not t:
                continue                              # null = unknown/removed id
            isrc = (t.get("external_ids") or {}).get("isrc")
            if isrc:
                obj = by_id.get(t.get("id"))
                if obj is not None:
                    obj.setdefault("external_ids", {})["isrc"] = isrc
                    filled += 1

        batches += 1
        i += batch_size
        if verbose and batches % 20 == 0:
            print(f"  isrc enrich: {i}/{len(ids)} looked up, {filled} filled")

    if verbose:
        pct = (filled / len(ids) * 100) if ids else 0.0
        print(f"ISRC enrich: filled {filled}/{len(ids)} ({pct:.0f}%) over {batches} batches")
    return {"looked_up": len(ids), "filled": filled, "batches": batches}


# ──────────────────────────────────────────────────────────────────────────
# Join playlist membership back onto the built corpus
# ──────────────────────────────────────────────────────────────────────────
def attach_playlist_membership(
    corpus: list[dict], membership: dict[str, list[dict]]
) -> list[dict]:
    """
    Add a ``'playlists'`` field to every corpus row, joining on Spotify id
    (``row['sp_id']``). Rows with no known membership get an empty list.

    This is the metadata addition the EP-to-playlist workflow needs
    (REFERENCE_CORPUS_DESIGN.md §5): a track can belong to several playlists,
    so each row carries a list of {'pid', 'name'} entries.
    """
    for row in corpus:
        row["playlists"] = membership.get(row.get("sp_id"), [])
    return corpus


# ──────────────────────────────────────────────────────────────────────────
# High-level one-call convenience
# ──────────────────────────────────────────────────────────────────────────
def ingest_mpd_corpus(
    mpd_dir: str | Path,
    sample_n: int | None = 8000,
    tracks_per_artist_cap: int | None = 5,
    max_slices: int | None = None,
    clip_seconds: float = 30.0,
    enrich: bool = True,
    spotify_client_id: str | None = None,
    spotify_client_secret: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """
    End-to-end: MPD slices → (optional ISRC enrich) → Deezer previews →
    embed-ready waveforms with playlist tags.

    Returns (corpus, misses, stats). ``corpus`` rows have 'audio' (float32 mono
    @ 24 kHz), the usual spotify_deezer metadata, and a 'playlists' list.
    ``misses`` is the spotify_deezer miss log (audit coverage / tune min_ratio).

    The Deezer fetch + decode lives in spotify_deezer.load_spotify_via_deezer;
    this only prepares its input and re-joins playlist membership afterward.
    """
    # Imported lazily so this module stays importable without the audio stack
    # (librosa/soundfile) when you only need the parsing / enrichment helpers.
    from anther_ml.spotify_deezer import load_spotify_via_deezer

    objs, membership = build_mpd_queries(
        mpd_dir,
        max_slices=max_slices,
        sample_n=sample_n,
        tracks_per_artist_cap=tracks_per_artist_cap,
        seed=seed,
        verbose=verbose,
    )

    enrich_stats = {"looked_up": 0, "filled": 0, "batches": 0}
    if enrich:
        try:
            enrich_stats = enrich_isrc(
                objs,
                client_id=spotify_client_id,
                client_secret=spotify_client_secret,
                verbose=verbose,
            )
        except ValueError as e:
            # No credentials → proceed fuzzy-only rather than failing the build.
            if verbose:
                print(f"Skipping ISRC enrichment ({e}); using fuzzy matching only.")

    corpus, misses = load_spotify_via_deezer(
        objs, clip_seconds=clip_seconds, verbose=verbose
    )
    attach_playlist_membership(corpus, membership)

    stats = {
        "unique_tracks_prepared": len(objs),
        "isrc_enriched": enrich_stats["filled"],
        "matched": len(corpus),
        "missed": len(misses),
        "isrc_matches": sum(1 for c in corpus if c.get("match_method") == "isrc"),
        "fuzzy_matches": sum(1 for c in corpus if c.get("match_method") == "fuzzy"),
    }
    if verbose:
        print(f"\nCorpus built: {stats}")
    return corpus, misses, stats
