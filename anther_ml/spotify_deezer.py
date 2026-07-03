"""
anther_ml.data — Spotify metadata → Deezer preview audio loader.

Workflow this supports (genre-agnostic Phase 2 corpus building):
    1. You pull track + artist info from Spotify playlists (metadata only).
    2. We match each track to Deezer, ISRC-first, fuzzy-title fallback.
    3. We fetch the Deezer 30s preview and decode it to a consistent
       mono waveform at MERT's sample rate, optionally clipped.

Design notes
------------
- ISRC is the clean cross-platform join key. Pass it whenever Spotify
  gives it to you (track['external_ids']['isrc']); fuzzy match is only a
  fallback and is logged separately so you can audit it.
- Deezer preview URLs are signed and expire within hours, so this fetches
  and decodes in one pass and never persists the audio to disk (a temp
  file is used only during decode and deleted immediately). Keep it that
  way — store embeddings, not audio.
- Everything (including your personal files) should go through the SAME
  clip_waveform() so durations and offsets are consistent across sources.
  That's what makes Deezer previews and your full-length personal tracks
  comparable in the embedding space.

Returns embed-ready waveforms (np.float32, mono, target_sr). If your
get_embedding()/embed_batch() take file paths rather than arrays, use
waveform_to_tempfile() as the bridge (see bottom of file).
"""

import io
import time
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import requests
import librosa

DEEZER_API = "https://api.deezer.com"
MERT_SR = 24000  # MERT-v1-330M expects 24 kHz mono


# ──────────────────────────────────────────────────────────────────────────
# Spotify side: normalize playlist items into a flat dict the matcher wants
# ──────────────────────────────────────────────────────────────────────────
def spotify_track_to_query(sp_track):
    """
    Flatten a Spotify *track object* into the minimal dict this module uses.

    Accepts either a full track object or a playlist-item wrapper
    ({'track': {...}}), which is what GET /playlists/{id}/items returns
    (the endpoint formerly known as /tracks before the Feb 2026 rename).

    Returns: {'sp_id', 'isrc', 'title', 'artist', 'duration_ms'}
    """
    t = sp_track.get("track", sp_track)  # unwrap playlist item if present
    if t is None:
        return None
    artists = t.get("artists") or []
    return {
        "sp_id": t.get("id"),
        "isrc": (t.get("external_ids") or {}).get("isrc"),
        "title": t.get("name") or "",
        "artist": artists[0]["name"] if artists else "",
        "duration_ms": t.get("duration_ms"),
    }


def _norm(s):
    """Lowercase, strip accents and common noise for fuzzy comparison."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    # drop feat./parenthetical/bracketed qualifiers that differ across DSPs
    for cut in (" feat.", " feat ", " ft.", " ft ", " (", " ["):
        i = s.find(cut)
        if i != -1:
            s = s[:i]
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()


def _ratio(a, b):
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ──────────────────────────────────────────────────────────────────────────
# Deezer side: HTTP with light throttling + retry, ISRC-first matching
# ──────────────────────────────────────────────────────────────────────────
def _deezer_get(path, params=None, throttle=0.05, retries=3, timeout=15):
    """GET against the Deezer simple API (no key needed). Returns parsed JSON."""
    url = f"{DEEZER_API}/{path}"
    last = None
    for attempt in range(retries):
        time.sleep(throttle)
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(0.5 * (attempt + 1))  # backoff
    return {"error": {"message": last or "unknown"}}


def match_deezer_track(query, min_ratio=0.82, throttle=0.05):
    """
    Resolve one Spotify query dict to a Deezer track.

    Strategy: exact ISRC lookup first; if that misses, advanced text search
    and accept the top hit only if its artist+title similarity clears
    min_ratio. Returns a dict with the preview URL and match provenance, or
    a dict with 'error' explaining the miss.
    """
    isrc = query.get("isrc")

    # 1) ISRC — exact, no ambiguity
    if isrc:
        d = _deezer_get(f"track/isrc:{isrc}", throttle=throttle)
        if "error" not in d and d.get("preview"):
            return {
                "preview": d["preview"],
                "deezer_id": d.get("id"),
                "matched_title": d.get("title"),
                "matched_artist": (d.get("artist") or {}).get("name"),
                "match_method": "isrc",
                "match_score": 1.0,
            }

    # 2) Fuzzy text fallback
    title, artist = query.get("title", ""), query.get("artist", "")
    if not (title and artist):
        return {"error": "no_isrc_match_and_insufficient_text"}

    q = f'artist:"{artist}" track:"{title}"'
    res = _deezer_get("search/track", params={"q": q, "limit": 5}, throttle=throttle)
    hits = res.get("data") or []
    if not hits:
        return {"error": "no_fuzzy_hits"}

    best, best_score = None, 0.0
    for h in hits:
        score = 0.5 * _ratio(title, h.get("title", "")) + \
                0.5 * _ratio(artist, (h.get("artist") or {}).get("name", ""))
        if score > best_score:
            best, best_score = h, score

    if best is None or best_score < min_ratio:
        return {"error": f"low_score:{best_score:.2f}"}
    if not best.get("preview"):
        return {"error": "matched_but_no_preview"}

    return {
        "preview": best["preview"],
        "deezer_id": best.get("id"),
        "matched_title": best.get("title"),
        "matched_artist": (best.get("artist") or {}).get("name"),
        "match_method": "fuzzy",
        "match_score": round(best_score, 3),
    }


# ──────────────────────────────────────────────────────────────────────────
# Audio: fetch preview → decode → clip (one consistent clip path for ALL sources)
# ──────────────────────────────────────────────────────────────────────────
def clip_waveform(wav, sr, clip_seconds=None, offset_seconds=0.0, pad=False):
    """
    Trim a mono waveform to a consistent window. Use this on your personal
    full-length tracks too, with the same clip_seconds you use for previews,
    so every source occupies the same temporal footprint.

    Note: a Deezer preview's *content* may start anywhere in the track, so
    matching clip_seconds matches duration, not musical section. That's fine
    for timbre/texture clustering — just don't over-read cluster structure.
    """
    if clip_seconds is None:
        return wav
    start = int(offset_seconds * sr)
    end = start + int(clip_seconds * sr)
    clip = wav[start:end]
    if pad and len(clip) < int(clip_seconds * sr):
        clip = np.pad(clip, (0, int(clip_seconds * sr) - len(clip)))
    return clip


def fetch_preview_waveform(preview_url, target_sr=MERT_SR,
                           clip_seconds=None, offset_seconds=0.0, timeout=20):
    """
    Download a Deezer preview MP3 and decode to mono float32 at target_sr.
    Audio is held in memory + a short-lived temp file only; nothing persists.
    Returns (waveform, true_duration_sec) or raises on failure.
    """
    r = requests.get(preview_url, timeout=timeout)
    r.raise_for_status()

    # librosa/soundfile won't reliably decode MP3 from a BytesIO, so route
    # through a temp file (deleted on context exit) and let audioread handle it.
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(r.content)
        tmp.flush()
        wav, sr = librosa.load(tmp.name, sr=target_sr, mono=True)

    true_dur = len(wav) / float(sr)
    wav = clip_waveform(wav, sr, clip_seconds, offset_seconds)
    return wav.astype(np.float32), true_dur


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator: Spotify track dicts in → embed-ready corpus out, misses logged
# ──────────────────────────────────────────────────────────────────────────
def load_spotify_via_deezer(sp_tracks, target_sr=MERT_SR, clip_seconds=30.0,
                            offset_seconds=0.0, min_ratio=0.82, throttle=0.05,
                            dedup=True, verbose=True):
    """
    Parameters
    ----------
    sp_tracks : iterable of Spotify track objects or playlist items.
    clip_seconds : window every track is trimmed to (default 30s ≈ preview len).
    dedup : drop duplicate ISRCs across playlists before fetching (curated
            playlists overlap heavily — this is the main efficiency win).

    Returns
    -------
    corpus : list of dicts, each with
        'audio' (np.float32 mono @ target_sr), 'sr',
        'name', 'artist', 'genre', 'source', 'isrc', 'sp_id',
        'deezer_id', 'match_method', 'match_score', 'true_duration'
    misses : list of dicts, each with the Spotify query + 'reason'
             (so you can audit coverage and tune min_ratio).
    """
    queries = [q for q in (spotify_track_to_query(t) for t in sp_tracks) if q]

    if dedup:
        seen, deduped = set(), []
        for q in queries:
            key = q.get("isrc") or (_norm(q["artist"]), _norm(q["title"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)
        if verbose:
            print(f"Deduped {len(queries)} → {len(deduped)} tracks")
        queries = deduped

    corpus, misses = [], []
    for i, q in enumerate(queries, 1):
        m = match_deezer_track(q, min_ratio=min_ratio, throttle=throttle)
        if "error" in m:
            misses.append({**q, "reason": m["error"]})
            continue
        try:
            wav, true_dur = fetch_preview_waveform(
                m["preview"], target_sr=target_sr,
                clip_seconds=clip_seconds, offset_seconds=offset_seconds,
            )
        except Exception as e:                      # network / decode failure
            misses.append({**q, "reason": f"fetch_failed:{type(e).__name__}"})
            continue

        corpus.append({
            "audio": wav,
            "sr": target_sr,
            "name": q["title"],
            "artist": q["artist"],
            "genre": "unknown",                     # Spotify per-track genre is gone post-deprecation
            "source": "spotify_deezer",
            "isrc": q.get("isrc"),
            "sp_id": q.get("sp_id"),
            "deezer_id": m["deezer_id"],
            "match_method": m["match_method"],
            "match_score": m["match_score"],
            "true_duration": round(true_dur, 2),
        })

        if verbose and i % 25 == 0:
            print(f"  {i}/{len(queries)}  matched={len(corpus)}  missed={len(misses)}")

    if verbose:
        hit = len(corpus)
        total = len(queries)
        fuzzy = sum(1 for c in corpus if c["match_method"] == "fuzzy")
        print(f"\nMatched {hit}/{total} ({hit/total*100:.0f}%) — "
              f"{hit - fuzzy} via ISRC, {fuzzy} via fuzzy, {len(misses)} missed")
    return corpus, misses


# ──────────────────────────────────────────────────────────────────────────
# Bridge: if get_embedding()/embed_batch() take PATHS not arrays, use this
# ──────────────────────────────────────────────────────────────────────────
def waveform_to_tempfile(wav, sr, suffix=".wav"):
    """
    Write a waveform to a temp WAV and return its Path. Caller deletes it
    after embedding. Only needed if your embedder is path-based; if
    get_embedding() accepts an array, hand it corpus_item['audio'] directly.
    """
    import soundfile as sf
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    sf.write(f.name, wav, sr)
    return Path(f.name)
