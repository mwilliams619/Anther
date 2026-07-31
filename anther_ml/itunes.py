"""
anther_ml.itunes — Apple/iTunes Search API → 30s preview audio.

Why this exists
---------------
Deezer is the primary preview source (see ``spotify_deezer``), but it has two
distinct failure modes for small/independent artists:

  1. *Search-index gaps.* The track is in Deezer's catalog but its
     ``search/track`` endpoint won't surface it by artist name, returning
     confident-looking noise instead ("Rob Knack" → "Rob & Jack"). The fix for
     that is the Spotify → ISRC → ``track/isrc:`` path in ``atlas.search``,
     not this module.
  2. *Genuine catalog gaps.* The track isn't on Deezer at all, by ISRC or
     otherwise. Measured on the two artists that motivated this module:
     Rob Knack 151/152 tracks resolve on Deezer, but Matt Brade only 3/17.

This module covers (2). The iTunes Search API needs no key and no auth, and
carries the long tail of DistroKid/TuneCore-distributed artists that Deezer
misses. Everything here is metadata + the public 30s preview asset.

Two things differ from the Deezer path and drive the design:

- **Previews are AAC in .m4a.** libsndfile can't decode AAC and librosa's
  audioread fallback needs a system ffmpeg. ``decode_preview`` routes through
  PyAV (bundled FFmpeg in the wheel) and hands back a temp WAV, so callers
  keep using the existing *path-based* ``embed_query``/``embed_query_dual``.
- **No ISRC.** The lookup endpoint returns no ISRC for any row (verified
  across both artists' full discographies), so iTunes rows cannot be joined to
  Spotify/Deezer on an exact key — only fuzzily, on artist+title.

Preview URLs here are static CDN assets, not signed like Deezer's, so they do
not need re-fetching before download.
"""

import logging
import tempfile
import time
from pathlib import Path

import requests

from .spotify_deezer import _norm, _primary_artist

log = logging.getLogger("anther_ml.itunes")

ITUNES_API = "https://itunes.apple.com"
MERT_SR = 24000  # MERT-v1-330M expects 24 kHz mono

# The lookup endpoint caps out at 200 entities per call and returns the artist
# record itself as row 0, so this is "200 including the header row".
LOOKUP_LIMIT = 200


def _get(path, params=None, throttle=0.05, retries=3, timeout=15):
    """GET against the iTunes Search API (no key needed). Returns parsed JSON.

    Apple rate-limits at roughly 20 calls/min per IP and answers with 403 when
    it's unhappy; the backoff here mirrors ``spotify_deezer._deezer_get`` so
    both source modules fail the same way.
    """
    url = f"{ITUNES_API}/{path}"
    last = None
    for attempt in range(retries):
        time.sleep(throttle)
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                # Apple serves JSON as text/javascript; r.json() is still fine,
                # but an empty body on a throttled call is not.
                if not r.content.strip():
                    last = "empty response"
                else:
                    return r.json()
            else:
                last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = str(e)
        time.sleep(0.5 * (attempt + 1))
    return {"error": {"message": last or "unknown"}}


def _track_row(r: dict) -> dict | None:
    """Normalize one iTunes result into this project's track contract, or None
    if it isn't a playable music track."""
    if r.get("wrapperType") != "track" or r.get("kind") != "song":
        return None
    if not r.get("previewUrl") or not r.get("trackId"):
        return None
    return {
        "source":      "itunes",
        "id":          f"itunes:{r['trackId']}",
        "itunes_id":   r["trackId"],
        "artist_id":   r.get("artistId"),
        "title":       r.get("trackName") or "",
        "artist":      r.get("artistName") or "",
        "album":       r.get("collectionName") or "",
        "cover":       r.get("artworkUrl100") or r.get("artworkUrl60") or "",
        "genre":       r.get("primaryGenreName"),   # DISPLAY ONLY — never a model input
        "preview_url": r["previewUrl"],
    }


def dedupe(rows: list[dict]) -> list[dict]:
    """Collapse re-releases of the same recording. First occurrence wins.

    iTunes lists every release separately, so one recording shows up 2-3 times
    under different ``trackId``s — and the *credit string* varies along with
    it, not just the title: "Matt Brade — Dreamweaver (feat. Noturlover)" and
    "Matt Brade & Noturlover — Dreamweaver" are one song. So the key is
    normalized title plus the **primary** artist, which reduces both of those
    to ("matt brade", "dreamweaver"). ``_norm`` already drops the trailing
    "(feat. …)" from titles.

    Keying on the primary artist rather than the full credit also keeps a
    genuine guest appearance distinct: "Rob Knack — Scroll (feat. Matt Brade)"
    keys under "rob knack" and survives a Matt Brade import.
    """
    seen = set()
    out = []
    for r in rows:
        key = (_norm(_primary_artist(r.get("artist", ""))), _norm(r.get("title", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def search_tracks(q: str, limit: int = 25) -> list[dict]:
    """Free-text track search. Returns normalized track dicts (may be empty)."""
    q = (q or "").strip()
    if not q:
        return []
    data = _get("search", params={"term": q, "entity": "musicTrack",
                                  "limit": max(1, min(limit, LOOKUP_LIMIT))})
    if "error" in data:
        log.warning("iTunes search failed for %r: %s", q, data["error"])
        return []
    rows = [row for row in (_track_row(r) for r in data.get("results", [])) if row]
    return dedupe(rows)[:limit]


def resolve_artist(name: str) -> dict | None:
    """Find an artist by name. Returns {"artist_id", "name", "genre"} or None.

    Prefers an exact normalized-name match over Apple's own ranking — a search
    for "Matt Brade" returns "Matt Fradd" as a plausible-looking second hit,
    and taking result[0] blindly is how the Deezer tier ends up showing noise.
    """
    name = (name or "").strip()
    if not name:
        return None
    data = _get("search", params={"term": name, "entity": "musicArtist", "limit": 10})
    if "error" in data:
        log.warning("iTunes artist search failed for %r: %s", name, data["error"])
        return None
    target = _norm(name)
    for r in data.get("results", []):
        if _norm(r.get("artistName", "")) == target:
            return {"artist_id": r.get("artistId"),
                    "name": r.get("artistName", ""),
                    "genre": r.get("primaryGenreName")}
    return None


def artist_tracks(artist_id, include_features: bool = False,
                  limit: int = LOOKUP_LIMIT) -> list[dict]:
    """Every track iTunes lists for an artist id, deduped.

    ``include_features=False`` (default) keeps only rows whose ``artistId`` is
    the queried artist. Apple's lookup leaks collaborators in both directions:
    a Rob Knack lookup returns 2 rows credited to others, and a Matt Brade
    lookup returns 4 Rob Knack tracks he features on. Set True to keep those
    guest appearances.
    """
    if artist_id in (None, ""):
        return []
    data = _get("lookup", params={"id": artist_id, "entity": "song",
                                  "limit": max(1, min(limit, LOOKUP_LIMIT))})
    if "error" in data:
        log.warning("iTunes lookup failed for artist %s: %s", artist_id, data["error"])
        return []
    rows = [row for row in (_track_row(r) for r in data.get("results", [])) if row]
    if not include_features:
        rows = [r for r in rows if str(r.get("artist_id")) == str(artist_id)]
    return dedupe(rows)


# ──────────────────────────────────────────────────────────────────────────
# Audio: AAC/.m4a preview → temp WAV the existing path-based embedders accept
# ──────────────────────────────────────────────────────────────────────────
def decode_preview(src: str | Path, target_sr: int = MERT_SR):
    """Decode an AAC/.m4a preview to a mono ``target_sr`` WAV on disk.

    Returns ``(path, cleanup)`` matching ``atlas._download_url``'s contract, so
    callers can drop this in wherever a temp audio path is expected.

    Goes through PyAV rather than librosa: libsndfile has no AAC support and
    the audioread fallback needs a system ffmpeg binary that isn't assumed to
    exist. PyAV's wheel bundles FFmpeg, so this works on a bare venv.
    """
    import av
    import numpy as np
    import soundfile as sf

    chunks = []
    with av.open(str(src)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream in preview")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):        # flush
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError("preview decoded to zero samples")
    wav = np.concatenate(chunks).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    path = Path(tmp.name)
    sf.write(str(path), wav, target_sr)
    return path, lambda: path.unlink(missing_ok=True)


def download_preview(url: str, timeout: int = 20):
    """Download an iTunes preview and decode it to a temp WAV.

    Returns ``(path, cleanup)``. Unlike Deezer's signed URLs these are plain
    CDN assets, so there's no refresh step.
    """
    if not url:
        raise ValueError("no preview URL available")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if len(r.content) < 1024:                       # error page, not audio
        raise ValueError(f"suspiciously small response ({len(r.content)} bytes)")

    raw = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    raw.write(r.content)
    raw.close()
    raw_path = Path(raw.name)
    try:
        return decode_preview(raw_path)
    finally:
        raw_path.unlink(missing_ok=True)
