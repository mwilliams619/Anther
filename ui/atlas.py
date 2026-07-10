"""
Frozen-corpus atlas: tiered search + placement for the d3 force-graph UI.

A song is resolved by a three-tier search — (1) the local corpus embedding,
(2) Deezer, (3) Spotify (last resort, gated on credentials) — then placed onto
the frozen 100k reference corpus as its own node (a cluster id is assigned but
its nearest corpus neighbors are not added to the graph — that fan-out made
the map too dense to read; see ``_merge_fragment``). Placed songs still link
to each other via query↔query similarity edges. The corpus is loaded once and
never re-fit; the exact UMAP coords are irrelevant here (positions come from
the force sim).

This module owns all corpus/MERT state so ui/app.py stays a thin router.
"""

import os
import json
import time
import sqlite3
import tempfile
import threading
from pathlib import Path

import numpy as np
import requests

from anther_ml import mpd_sql
from anther_ml.corpus.bundle import ReferenceCorpus
from anther_ml.corpus.place import embed_query, place, recommend_from_seeds
from anther_ml.spotify_deezer import _deezer_get, match_deezer_track, _norm, _ratio

# ── Config ───────────────────────────────────────────────────────────────────

CORPUS_DIR      = os.environ.get("ANTHER_CORPUS", "models/corpus_corpus_mpd_100k")
MPD_DB          = os.environ.get(
    "ANTHER_MPD_DB",
    str(Path(__file__).parent.parent / "data" / "mpd_dump" / "spotifydbdumpshare.sqlite"),
)
MPD_PREP_HINT   = ("full-MPD playlist data unavailable — run: "
                   "python -m anther_ml.mpd_sql --db data/mpd_dump/spotifydbdumpshare.sqlite --prepare-ui")
TOP_K           = 8       # neighbors pulled in per placed song
QQ_MAX_PER_NODE = 6       # cap on query↔query edges added per placed song
IMPORT_CAP      = int(os.environ.get("ANTHER_IMPORT_CAP", "100"))  # max songs per playlist/album add
CORPUS_MIN_HITS = 5       # < this many strong corpus hits → fall through to Deezer
STRONG_SCORE    = 0.6     # _ratio threshold for a "strong" corpus match

# MERT vectors sit in a narrow cone: two *random* corpus songs are ~0.96 cosine
# apart, so a raw cosine floor means nothing (a low one links everything into one
# blob). We instead draw a query↔query edge only when the pair is more similar
# than QUERY_LINK_PCTL % of random corpus pairs — the same null-distribution
# framing the corpus placement uses. The concrete cosine cutoff is calibrated
# from the corpus once at load (see _calibrate_qq_threshold).
QUERY_LINK_PCTL = float(os.environ.get("ANTHER_QQ_PCTL", "95"))
_qq_threshold   = 0.981   # replaced at load() with the corpus-calibrated value

SESSION_DIR      = Path(__file__).parent / "session"
GRAPH_PATH       = SESSION_DIR / "graph.json"
EMBED_CACHE_PATH = SESSION_DIR / "embed_cache.sqlite"

# ── Lazy singletons ──────────────────────────────────────────────────────────

_corpus = None
_id_to_idx: dict = {}
_id_to_cluster: dict = {}
_playlist_index: dict = {}   # pid -> {"pid","name","name_norm","n_tracks","indices":[int]}
_playlist_rows: list = []    # _playlist_index values sorted by -n_tracks (search scans)
_corpus_lock = threading.Lock()

_model = _processor = _device = None
_mert_lock = threading.Lock()

_graph = {"nodes": {}, "links": []}          # nodes keyed by id; links is a list
_link_keys: set = set()                      # (source, target) dedupe
_query_vecs: dict = {}                        # placed-song id → index-space unit vec
_groups: dict = {}                            # gid → {"name", "kind"} for imported playlists/albums
_graph_lock = threading.Lock()


# ── Corpus + MERT loading ────────────────────────────────────────────────────

def load() -> ReferenceCorpus:
    """Load the frozen corpus once (idempotent, thread-safe)."""
    global _corpus, _id_to_idx, _id_to_cluster, _playlist_index, _playlist_rows
    with _corpus_lock:
        if _corpus is not None:
            return _corpus
        corpus = ReferenceCorpus.load(CORPUS_DIR)
        labels = corpus.labels
        id_to_idx, id_to_cluster = {}, {}
        for i, m in enumerate(corpus.metadata):
            tid = m.get("id")
            if tid is None:
                continue
            id_to_idx[tid] = i
            id_to_cluster[tid] = int(labels[i])
        _corpus, _id_to_idx, _id_to_cluster = corpus, id_to_idx, id_to_cluster
        _playlist_index, _playlist_rows = _build_playlist_index(corpus)
        _calibrate_qq_threshold(corpus)
        _load_graph()
        return _corpus


def _build_playlist_index(corpus) -> tuple[dict, list]:
    """One pass over metadata → pid-keyed playlist index for name search and
    member lookup. The bundle keeps only sampled tracks, so `indices` is each
    playlist's *in-corpus* subset, not its full original membership."""
    index: dict = {}
    for i, m in enumerate(corpus.metadata):
        for pl in m.get("playlists") or []:
            pid = pl.get("pid")
            if pid is None:
                continue
            entry = index.get(pid)
            if entry is None:
                name = pl.get("name") or ""
                entry = index[pid] = {"pid": pid, "name": name,
                                      "name_norm": _norm(name), "indices": []}
            entry["indices"].append(i)
    for entry in index.values():
        entry["n_tracks"] = len(entry["indices"])
    rows = sorted(index.values(), key=lambda e: -e["n_tracks"])
    return index, rows


def _calibrate_qq_threshold(corpus, n_pairs: int = 200_000) -> None:
    """Set the query↔query cosine cutoff to the QUERY_LINK_PCTL percentile of
    random corpus-pair cosines (index space) — so an edge means "more similar
    than that fraction of released music," not an arbitrary absolute cosine."""
    global _qq_threshold
    E = corpus.index.embeddings                       # standardized + L2-normalized
    n = E.shape[0]
    if n < 2:
        return
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    mask = a != b
    cos = np.einsum("ij,ij->i", E[a[mask]], E[b[mask]])
    _qq_threshold = float(np.percentile(cos, QUERY_LINK_PCTL))


def warm() -> None:
    """Warm corpus (and later MERT) in a background thread at startup."""
    threading.Thread(target=load, daemon=True).start()


def is_ready() -> bool:
    return _corpus is not None


def _mert():
    """Lazily load MERT once (~30 s first call)."""
    global _model, _processor, _device
    with _mert_lock:
        if _model is None:
            from anther_ml.embedding import load_mert
            _model, _processor, _device = load_mert()
    return _model, _processor, _device


# ── Full-MPD playlist DB ─────────────────────────────────────────────────────

_mpd_ready: bool | None = None


def mpd_ready() -> bool:
    """True iff the MPD sqlite DB exists and mpd_sql.prepare_ui() has run."""
    global _mpd_ready
    if _mpd_ready is None:
        _mpd_ready = Path(MPD_DB).exists() and mpd_sql.is_ui_ready(MPD_DB)
    return _mpd_ready


# ── Embedding cache (raw MERT vecs for out-of-corpus tracks) ────────────────
# Keyed by node id ("spotify:<id>", "deezer:<id>", …) and storing the RAW
# embedding — pre-standardization — so cached vecs survive a corpus swap and
# can feed both place() and index.transform_query(). Re-adding a playlist or
# adding overlapping playlists never re-downloads/re-embeds a track.

_cache_lock = threading.Lock()


def cached_vec(track_id: str) -> np.ndarray | None:
    if not track_id or not EMBED_CACHE_PATH.exists():
        return None
    try:
        con = sqlite3.connect(str(EMBED_CACHE_PATH))
        try:
            row = con.execute(
                "SELECT vec FROM embed_cache WHERE track_id = ?", (track_id,)
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def cache_vec(track_id: str, vec, name: str = "", artist: str = "") -> None:
    if not track_id:
        return
    v = np.asarray(vec, dtype=np.float32)
    with _cache_lock:
        SESSION_DIR.mkdir(exist_ok=True)
        con = sqlite3.connect(str(EMBED_CACHE_PATH))
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache ("
                "track_id TEXT PRIMARY KEY, vec BLOB NOT NULL, dim INTEGER NOT NULL, "
                "name TEXT, artist TEXT, created REAL)"
            )
            con.execute(
                "INSERT OR REPLACE INTO embed_cache VALUES (?, ?, ?, ?, ?, ?)",
                (track_id, v.tobytes(), int(v.size), name, artist, time.time()),
            )
            con.commit()
        finally:
            con.close()


# ── Tiered search ────────────────────────────────────────────────────────────

def spotify_configured() -> bool:
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


def search(q: str, limit: int = 25) -> dict:
    """
    Three-tier search. Corpus first; fall through to Deezer only if the corpus
    yields fewer than CORPUS_MIN_HITS strong matches; fall through to Spotify
    only if Deezer is also empty. Returns {results, tiers, spotify_configured}.
    """
    load()
    q = (q or "").strip()
    if not q:
        return {"results": [], "tiers": {}, "spotify_configured": spotify_configured()}

    corpus_hits = _search_corpus(q, limit)
    strong = [h for h in corpus_hits if h["score"] >= STRONG_SCORE]
    results = list(corpus_hits)
    tiers = {"corpus": len(corpus_hits)}

    if len(strong) < CORPUS_MIN_HITS:
        deezer_hits = _search_deezer(q, limit)
        results += deezer_hits
        tiers["deezer"] = len(deezer_hits)
        if not deezer_hits:
            sp = _search_spotify(q, limit)
            results += sp["results"]
            tiers["spotify"] = len(sp["results"])

    return {"results": results, "tiers": tiers, "spotify_configured": spotify_configured()}


def _search_corpus(q: str, limit: int) -> list:
    """Substring pre-filter (all query tokens present) then _ratio ranking."""
    tokens = _norm(q).split()
    if not tokens:
        return []
    scored = []
    for i, m in enumerate(_corpus.metadata):
        name = m.get("name") or ""
        artist = m.get("artist") or ""
        hay = _norm(f"{name} {artist}")
        if not all(t in hay for t in tokens):
            continue
        score = max(_ratio(q, name), _ratio(q, artist), _ratio(q, f"{artist} {name}"))
        scored.append((score, i, m))
    scored.sort(key=lambda t: -t[0])
    out = []
    for score, i, m in scored[:limit]:
        tid = m.get("id")
        out.append({
            "source":  "corpus",
            "id":      tid,
            "idx":     i,
            "title":   m.get("name") or "",
            "artist":  m.get("artist") or "",
            "cluster": _id_to_cluster.get(tid),
            "score":   round(float(score), 3),
        })
    return out


def search_playlists(q: str, limit: int = 20) -> dict:
    """Playlist-name search. Against the full MPD DB (real track counts, any
    of the 1M playlists) when prepared; else the corpus playlist index
    (in-corpus counts only) plus a notice telling the user how to upgrade.
    Returns {"results": [...], "notice": str|None}."""
    load()
    q = (q or "").strip()
    if not q:
        return {"results": [], "notice": None}

    if mpd_ready():
        hits = mpd_sql.search_playlists_db(MPD_DB, q, limit=limit)
        for h in hits:
            entry = _playlist_index.get(h["pid"])
            h["n_in_corpus"] = len(entry["indices"]) if entry else 0
            h["source"] = "mpd"
        return {"results": hits, "notice": None}

    tokens = _norm(q).split()
    scored = []
    for e in _playlist_rows:
        if not all(t in e["name_norm"] for t in tokens):
            continue
        scored.append((_ratio(q, e["name"]), e))
    scored.sort(key=lambda t: (-t[0], -t[1]["n_tracks"]))
    results = [{
        "pid":         e["pid"],
        "name":        e["name"],
        "n_tracks":    e["n_tracks"],
        "n_in_corpus": e["n_tracks"],
        "source":      "corpus",
        "score":       round(float(score), 3),
    } for score, e in scored[:limit]]
    return {"results": results, "notice": MPD_PREP_HINT}


def _search_deezer(q: str, limit: int) -> list:
    data = _deezer_get("search/track", params={"q": q, "limit": limit})
    if "error" in data:
        return []
    out = []
    for h in (data.get("data") or []):
        if not h.get("preview"):
            continue
        out.append({
            "source":      "deezer",
            "id":          f"deezer:{h['id']}",
            "deezer_id":   h["id"],
            "title":       h.get("title", ""),
            "artist":      (h.get("artist") or {}).get("name", ""),
            "album":       (h.get("album") or {}).get("title", ""),
            "cover":       (h.get("album") or {}).get("cover_small", ""),
            "preview_url": h["preview"],
        })
    return out


def _search_spotify(q: str, limit: int) -> dict:
    """
    Last-resort tier: Spotify search → ISRC → Deezer preview (so we still have
    audio to embed). Fully gated on SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET.
    """
    if not spotify_configured():
        return {"configured": False, "results": []}
    try:
        from anther_ml.mpd_ingest import get_spotify_token
        token = get_spotify_token()
        r = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": q, "type": "track", "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        items = ((r.json().get("tracks") or {}).get("items")) or []
    except Exception as exc:                                    # auth / network
        return {"configured": True, "error": str(exc), "results": []}

    out = []
    for t in items:
        artist = (t.get("artists") or [{}])[0].get("name", "")
        title = t.get("name", "")
        m = match_deezer_track({
            "isrc":   (t.get("external_ids") or {}).get("isrc"),
            "title":  title,
            "artist": artist,
        })
        if "error" in m:
            continue                                            # no Deezer preview → skip
        out.append({
            "source":      "spotify",
            "id":          f"spotify:{t.get('id')}",
            "deezer_id":   m.get("deezer_id"),
            "title":       title,
            "artist":      artist,
            "preview_url": m["preview"],
        })
    return {"configured": True, "results": out}


# ── Placement → graph fragment ───────────────────────────────────────────────

# One CUDA consumer at a time: the Flask request thread (place_song) and the
# playlist background worker both embed through this lock.
_embed_lock = threading.Lock()


class PlacementSkip(Exception):
    """A track that can't be placed (no audio resolvable). .reason is a short
    machine-readable slug surfaced to the UI's skipped-tracks report."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _place_corpus_track(idx: int, extra: dict | None = None) -> dict:
    """Place a corpus row by index and merge it into the graph."""
    corpus = load()
    self_id = corpus.metadata[idx].get("id")
    raw_vec = corpus.embeddings[idx]
    node = {
        "id":         self_id,
        "name":       corpus.metadata[idx].get("name", ""),
        "artist":     corpus.metadata[idx].get("artist", ""),
        "cluster":    _id_to_cluster.get(self_id),
        "kind":       "query",
        "confidence": 1.0,
        "source":     "corpus",
        **(extra or {}),
    }
    return _merge_fragment(node, corpus.index.transform_query(raw_vec))


def _place_query_vec(track_id: str, name: str, artist: str, raw_vec,
                     source: str, extra: dict | None = None) -> dict:
    """Place an out-of-corpus track from an already-computed raw MERT vector."""
    corpus = load()
    res = place(corpus, raw_vec, top_k=TOP_K)
    node = {
        "id":         track_id,
        "name":       name,
        "artist":     artist,
        "cluster":    int(res["cluster"]["id"]),
        "kind":       "query",
        "confidence": round(float(res["cluster"]["confidence"]), 3),
        "source":     source,
        # persisted at placement time: these can't be recomputed later without
        # the raw vector (which for non-cached songs isn't kept on disk)
        "tags":          res["tags"],
        "cluster_label": res["cluster"].get("label", ""),
        **(extra or {}),
    }
    return _merge_fragment(node, corpus.index.transform_query(raw_vec))


def place_song(result: dict) -> dict:
    """
    Place one search result onto the frozen corpus and merge it into the graph.
    Returns the fragment {nodes, links} that was newly added (for the frontend
    to splice into the running force simulation).
    """
    corpus = load()
    source = result.get("source")
    # re-placed removed nodes keep their playlist/album ring
    extra = ({"playlist_pid": result["playlist_pid"]}
             if result.get("playlist_pid") is not None else None)

    if source == "corpus":
        idx = result.get("idx")
        if idx is None:
            idx = _id_to_idx.get(result.get("id"))
        if idx is None:
            raise ValueError("corpus result missing idx/id")
        return _place_corpus_track(idx, extra=extra)

    # Cache first: any song embedded before (incl. removed-then-re-placed ones)
    # skips the download + MERT pass entirely.
    raw = cached_vec(result.get("id"))
    if raw is not None:
        return _place_query_vec(result.get("id"), result.get("title", ""),
                                result.get("artist", ""), raw, source, extra=extra)

    if source == "upload":
        path, cleanup = Path(result["path"]), None
    else:
        path, cleanup = _download_preview(result)
    try:
        with _embed_lock:
            model, processor, device = _mert()
            vec = embed_query(path, corpus, model, processor, device)
    finally:
        if cleanup:
            cleanup()
    cache_vec(result.get("id"), vec, result.get("title", ""), result.get("artist", ""))
    return _place_query_vec(result.get("id"), result.get("title", ""),
                            result.get("artist", ""), vec, source, extra=extra)


def resolve_and_embed(track: dict) -> tuple[np.ndarray, str]:
    """
    Audio for one out-of-corpus track → raw MERT vector.

    ``track`` is {"id", "name", "artist", "preview_url"}. Tries the stored
    Spotify preview URL first (many p.scdn.co links are dead — Spotify
    deprecated previews in late 2024), then a Deezer name/artist match.
    Returns (raw_vec, method); raises PlacementSkip when no audio resolves.
    """
    corpus = load()
    path = cleanup = None
    method = None

    url = track.get("preview_url")
    if url:
        try:
            path, cleanup = _download_url(url)
            method = "spotify_preview"
        except Exception:
            path = None                                     # dead link → Deezer

    if path is None:
        m = match_deezer_track({"title": track.get("name", ""),
                                "artist": track.get("artist", "")})
        if "error" in m or not m.get("preview"):
            raise PlacementSkip("deezer_no_match" if url else "no_preview")
        try:
            path, cleanup = _download_preview(
                {"preview_url": m["preview"], "deezer_id": m.get("deezer_id")})
            method = "deezer"
        except Exception as e:
            raise PlacementSkip(f"download_failed:{e}")

    try:
        with _embed_lock:
            model, processor, device = _mert()
            vec = embed_query(path, corpus, model, processor, device)
    except Exception as e:
        raise PlacementSkip(f"embed_failed:{e}")
    finally:
        if cleanup:
            cleanup()
    cache_vec(track["id"], vec, track.get("name", ""), track.get("artist", ""))
    return vec, method


def place_external_track(track: dict, raw_vec, playlist_pid=None) -> dict:
    """Merge one embedded out-of-corpus track into the graph (worker entry)."""
    extra = {"playlist_pid": playlist_pid} if playlist_pid is not None else None
    source = "deezer" if str(track["id"]).startswith("deezer:") else "mpd"
    return _place_query_vec(track["id"], track.get("name", ""),
                            track.get("artist", ""), raw_vec, source, extra=extra)


def _seed_vec(seed_id: str) -> np.ndarray | None:
    """Raw MERT vector for a placed seed id: corpus row if in-corpus, else the
    embed cache (populated when the song was searched/placed). None if neither."""
    idx = _id_to_idx.get(seed_id)
    if idx is not None:
        return load().embeddings[idx]
    return cached_vec(seed_id)


def recommend(seed_ids: list, top_k: int = 20, method: str = "centroid",
              splice: bool = True) -> dict:
    """
    Multi-song recommendation (use-case 2): given the ids of several placed seed
    songs, return corpus tracks similar to the *set* as a whole.

    Every seed must already have a vector available — in-corpus rows and any
    song previously searched/placed (embed-cached) resolve instantly; a seed
    with no cached vector is reported in ``skipped`` rather than silently
    dropped. Seeds are excluded from their own results.

    ``method`` is passed through to ``recommend_from_seeds`` ("centroid" default,
    "topk" fallback for multi-mood seed sets). When ``splice`` is true the
    returned tracks are also merged into the shared graph (as corpus nodes wired
    to the seeds' neighborhood) so the list and the map stay in sync.

    Returns {"results": [...], "n_seeds": int, "skipped": [ids], "method": str}.
    """
    ids = [s for s in (seed_ids or []) if s]
    if not ids:
        raise ValueError("recommend needs at least one seed id")
    load()

    vecs, used, skipped = [], [], []
    for sid in ids:
        v = _seed_vec(sid)
        if v is None:
            skipped.append(sid)
        else:
            vecs.append(v)
            used.append(sid)
    if not vecs:
        raise ValueError("no seed ids resolved to a vector (none cached yet)")

    results = recommend_from_seeds(
        load(), vecs, top_k=top_k, exclude_ids=set(used), method=method
    )

    if splice:
        for r in results:
            idx = _id_to_idx.get(r.get("id"))
            if idx is not None:
                _place_corpus_track(idx, extra={"recommended": True})

    return {"results": results, "n_seeds": len(used),
            "skipped": skipped, "method": method}


def _download_url(url: str):
    """Download an audio URL to a temp mp3; returns (path, cleanup)."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    if len(r.content) < 1024:                               # error page, not audio
        raise ValueError(f"suspiciously small response ({len(r.content)} bytes)")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(r.content)
    tmp.close()
    p = Path(tmp.name)
    return p, lambda: p.unlink(missing_ok=True)


def _download_preview(result: dict):
    """Download a (possibly re-freshed) Deezer preview to a temp mp3."""
    url = result.get("preview_url")
    did = result.get("deezer_id")
    if did:                                                     # signed URLs expire
        fresh = _deezer_get(f"track/{did}").get("preview")
        if fresh:
            url = fresh
    if not url:
        raise ValueError("no preview URL available")
    return _download_url(url)


def _merge_fragment(node: dict, qvec=None) -> dict:
    """Upsert the query node into the graph.

    Placed songs are *not* fanned out to their nearest corpus neighbors
    (that flooded the map with grey context nodes and made it too dense to
    read). Instead this wires the placed song directly to every *other*
    placed song whose index-space cosine clears the corpus-calibrated
    ``_qq_threshold`` — so similar songs you add pull together in the force
    sim regardless of which Leiden cluster each landed in.
    """
    added_nodes, added_links = [], []
    with _graph_lock:
        existing = _graph["nodes"].get(node["id"])
        if existing is None:
            _graph["nodes"][node["id"]] = node
            added_nodes.append(node)
        else:                                                  # re-add → promote to query
            existing.update(node)                              # carries playlist_pid etc.

        # ── query↔query similarity edges (top-QQ_MAX_PER_NODE by score, so a
        # coherent playlist batch can't flood O(m²) links) ──
        if qvec is not None:
            cands = []
            for other_id, ovec in _query_vecs.items():
                if other_id == node["id"]:
                    continue
                score = float(np.dot(qvec, ovec))
                if score < _qq_threshold:
                    continue
                key = (node["id"], other_id)
                rkey = (other_id, node["id"])
                if key in _link_keys or rkey in _link_keys:
                    continue
                cands.append((score, other_id))
            cands.sort(reverse=True)
            for score, other_id in cands[:QQ_MAX_PER_NODE]:
                link = {"source": node["id"], "target": other_id,
                        "value": round(score, 3), "kind": "qq"}
                _graph["links"].append(link)
                _link_keys.add((node["id"], other_id))
                added_links.append(link)
            _query_vecs[node["id"]] = qvec

        _save_graph()
    return {"nodes": added_nodes, "links": added_links}


def _place_collection_rows(rows: list, gid, source: str):
    """Shared playlist/album placement split. ``rows`` are
    {"id","name","artist","preview_url"}; in-corpus and cached tracks merge into
    the returned fragment now, the rest come back as the pending embed list.
    Returns (fragment, n_immediate, pending)."""
    fragment = {"nodes": [], "links": []}
    n_immediate = 0
    pending = []
    for r in rows:
        nid = r["id"]
        idx = _id_to_idx.get(nid)
        if idx is not None:                                # in-corpus → instant
            f = _place_corpus_track(idx, extra={"playlist_pid": gid})
        else:
            raw = cached_vec(nid)
            if raw is not None:                            # embedded before → instant
                f = _place_query_vec(nid, r["name"], r["artist"], raw,
                                     source, extra={"playlist_pid": gid})
            else:
                with _graph_lock:
                    existing = _graph["nodes"].get(nid)
                    if existing is not None and existing.get("kind") == "query":
                        existing["playlist_pid"] = gid     # placed pre-cache: tag only
                        continue
                pending.append({"id": nid, "name": r["name"],
                                "artist": r["artist"],
                                "preview_url": r.get("preview_url")})
                continue
        n_immediate += 1
        fragment["nodes"] += f["nodes"]
        fragment["links"] += f["links"]
    return fragment, n_immediate, pending


def _collection_response(gid, name, rows, n_total, fragment, n_immediate,
                         pending, notice=None) -> dict:
    with _graph_lock:                                     # remember the group's display name
        _groups[str(gid)] = {"name": name,
                             "kind": "album" if str(gid).startswith("album:") else "playlist"}
        _save_graph()
    job_id = None
    if pending:
        import playlist_jobs
        job_id = playlist_jobs.start(gid, name, pending)
    return {
        "playlist": {
            "pid":         gid,
            "name":        name,
            "n_tracks":    len(rows),
            "n_total":     n_total,
            "capped":      n_total > len(rows),
            "n_immediate": n_immediate,
            "n_pending":   len(pending),
        },
        "fragment": fragment,
        "job_id":   job_id,
        "notice":   notice,
    }


def place_playlist(pid) -> dict:
    """
    Place a playlist's songs onto the graph as ordinary query nodes (no
    artificial hub; membership is carried on each node as ``playlist_pid``).

    With the prepared MPD DB, membership is pulled from the full dump, capped at
    the IMPORT_CAP most popular tracks: in-corpus and already-embedded (cached)
    tracks merge into the returned fragment immediately; the rest are handed to
    the background embed worker (``playlist_jobs``) and stream in via
    /api/playlist/status. Without the DB, falls back to the in-corpus subset
    only, with a notice.

    Returns {"playlist": {...}, "fragment": {nodes, links}, "job_id", "notice"}.
    """
    load()

    if mpd_ready():
        pid = str(pid)
        db_rows = mpd_sql.playlist_tracks(MPD_DB, pid)    # popularity DESC
        if not db_rows:
            raise ValueError(f"unknown or empty playlist: {pid!r}")
        name = mpd_sql.playlist_name(MPD_DB, pid) or ""
        n_total = len(db_rows)
        rows = [{"id": f"spotify:{r['track_id']}", "name": r["name"],
                 "artist": r["artist"], "preview_url": r["preview_url"]}
                for r in db_rows[:IMPORT_CAP]]            # top-IMPORT_CAP by popularity
        fragment, n_immediate, pending = _place_collection_rows(rows, pid, "mpd")
        return _collection_response(pid, name, rows, n_total,
                                    fragment, n_immediate, pending)

    entry = _playlist_index.get(pid)
    if entry is None and pid is not None:                 # JSON may flip int/str
        entry = _playlist_index.get(str(pid))
        if entry is None:
            try:
                entry = _playlist_index.get(int(pid))
            except (TypeError, ValueError):
                pass
    if entry is None:
        raise ValueError(f"unknown playlist: {pid!r}")
    pid, name = entry["pid"], entry["name"]
    n_total = len(entry["indices"])
    idxs = entry["indices"][:IMPORT_CAP]
    fragment = {"nodes": [], "links": []}
    for idx in idxs:
        f = _place_corpus_track(idx, extra={"playlist_pid": pid})
        fragment["nodes"] += f["nodes"]
        fragment["links"] += f["links"]
    return _collection_response(pid, name, idxs, n_total, fragment,
                                len(idxs), [], notice=MPD_PREP_HINT)


# ── Album search + placement (Deezer-sourced) ────────────────────────────────

def search_albums(q: str, limit: int = 20) -> dict:
    """Album-name search against the Deezer catalog (no local DB needed).
    Returns {"results": [{album_id, name, artist, cover, n_tracks}]}."""
    q = (q or "").strip()
    if not q:
        return {"results": []}
    data = _deezer_get("search/album", params={"q": q, "limit": limit})
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Deezer error"))
    results = [{
        "album_id": h.get("id"),
        "name":     h.get("title", ""),
        "artist":   (h.get("artist") or {}).get("name", ""),
        "cover":    h.get("cover_small") or "",
        "n_tracks": h.get("nb_tracks"),
    } for h in (data.get("data") or [])]
    return {"results": results}


def place_album(album_id) -> dict:
    """
    Place a Deezer album's tracks onto the graph — same flow as
    ``place_playlist`` (cached tracks instant, the rest background-embedded),
    grouped under ``playlist_pid = "album:<id>"`` so they share an accent ring.
    Every Deezer track carries a fresh preview URL, so placement is reliable.
    """
    load()
    if album_id in (None, ""):
        raise ValueError("missing album id")
    info = _deezer_get(f"album/{album_id}")
    if "error" in info:
        raise ValueError(f"unknown album: {album_id!r}")
    album_artist = (info.get("artist") or {}).get("name", "")
    name = info.get("title", "")

    tr = _deezer_get(f"album/{album_id}/tracks", params={"limit": max(IMPORT_CAP, 100)})
    if "error" in tr:
        raise ValueError(f"album tracks unavailable: {album_id!r}")
    all_rows = [{
        "id":          f"deezer:{t.get('id')}",
        "name":        t.get("title", ""),
        "artist":      (t.get("artist") or {}).get("name", album_artist),
        "preview_url": t.get("preview"),
    } for t in (tr.get("data") or []) if t.get("id")]
    if not all_rows:
        raise ValueError(f"album has no tracks: {album_id!r}")

    n_total = info.get("nb_tracks") or len(all_rows)
    rows = all_rows[:IMPORT_CAP]                          # album order
    gid = f"album:{album_id}"
    fragment, n_immediate, pending = _place_collection_rows(rows, gid, "deezer")
    return _collection_response(gid, f"{album_artist} — {name}", rows, n_total,
                                fragment, n_immediate, pending)


# ── Song detail (click panel) ────────────────────────────────────────────────

def _cluster_label(cid) -> str:
    if cid is None or int(cid) < 0:
        return ""
    try:
        return _corpus.cluster_profile(int(cid)).get("label_final", "")
    except KeyError:
        return ""


def _inherit_tags(corpus, qvec=None, neighbor_ids=None, top_k: int = 3, knn: int = 10) -> list:
    """Micro-genre tags inherited from nearest corpus tracks' track_tags
    (display only). Backfill for query nodes that predate tag persistence:
    rank neighbors by the in-memory query vector when we have it, else fall
    back to the node's stored corpus neighbors on the graph."""
    track_tags = corpus.track_tags
    if track_tags is None:
        return []
    if qvec is not None:
        sims = corpus.index.embeddings @ qvec
        idxs = [int(i) for i in np.argsort(sims)[::-1][:knn]]
    else:
        idxs = [_id_to_idx[nid] for nid in (neighbor_ids or []) if nid in _id_to_idx]
        if not idxs:
            return []
    score: dict = {}
    for i in idxs:
        for t in track_tags[i]["tags"]:
            score[t["genre"]] = score.get(t["genre"], 0.0) + t["score"] / len(idxs)
    top = sorted(score.items(), key=lambda kv: -kv[1])[:top_k]
    return [{"genre": g, "score": round(s, 4), "primary": rank == 0, "source": "neighbors"}
            for rank, (g, s) in enumerate(top)]


_preview_cache: dict[str, str | None] = {}   # song_id → Deezer preview url | None (no match)


def get_preview_url(song_id: str) -> str | None:
    """Resolve a 30s preview URL for the detail popover's play button.

    Neither corpus rows nor placed nodes carry a preview URL (only transient
    search results do), so this matches by title/artist against Deezer on
    demand and caches the result in-process — None means "no match found",
    cached too so a repeat click doesn't re-hit Deezer.
    """
    if song_id in _preview_cache:
        return _preview_cache[song_id]

    idx = _id_to_idx.get(song_id)
    if idx is not None:
        corpus = load()
        name = corpus.metadata[idx].get("name", "")
        artist = corpus.metadata[idx].get("artist", "")
    else:
        with _graph_lock:
            gnode = _graph["nodes"].get(song_id)
        if gnode is None:
            return None
        name, artist = gnode.get("name", ""), gnode.get("artist", "")

    url = None
    if name and artist:
        m = match_deezer_track({"title": name, "artist": artist})
        if "error" not in m:
            url = m.get("preview")
    _preview_cache[song_id] = url
    return url


def song_detail(song_id: str, top_n: int = 10) -> dict | None:
    """
    Full detail payload for the click panel: identity, cluster, micro-genre
    tags, playlist membership, and a top-N similar-songs list.

    Corpus tracks are recomputed on demand from the loaded bundle. Non-corpus
    query nodes (deezer/spotify/upload) use placement-time data — their similar
    list comes from the in-memory query vector when this session placed them,
    else from the node's stored graph edges. Returns None for unknown ids.
    """
    corpus = load()
    idx = _id_to_idx.get(song_id)

    with _graph_lock:
        gnode = _graph["nodes"].get(song_id)
        gnode = dict(gnode) if gnode is not None else None
        node_ids = set(_graph["nodes"])
        qvec = _query_vecs.get(song_id)
        # link-derived fallback rows (resolved here while we hold the lock)
        link_rows = []
        if idx is None and gnode is not None and qvec is None:
            for l in _graph["links"]:
                other = (l["target"] if l["source"] == song_id
                         else l["source"] if l["target"] == song_id else None)
                if other is None:
                    continue
                on = _graph["nodes"].get(other, {})
                link_rows.append({
                    "id":       other,
                    "name":     on.get("name", ""),
                    "artist":   on.get("artist", ""),
                    "score":    l.get("value"),
                    "cluster":  on.get("cluster"),
                    "on_graph": True,
                })

    if idx is None and gnode is None:
        return None

    if idx is not None:                                 # corpus track
        m = corpus.metadata[idx]
        cid = _id_to_cluster.get(song_id)
        track_tags = corpus.track_tags
        tags = track_tags[idx].get("tags", []) if track_tags is not None else []
        rows = corpus.index.query(corpus.embeddings[idx], top_k=top_n + 1)
        rows = [r for r in rows if r.get("id") != song_id][:top_n]
        similar = [{
            "id":       r.get("id"),
            "name":     r.get("name", ""),
            "artist":   r.get("artist", ""),
            "score":    round(float(r.get("score", 0.0)), 3),
            "cluster":  _id_to_cluster.get(r.get("id")),
            "on_graph": r.get("id") in node_ids,
        } for r in rows]
        return {
            "id":        song_id,
            "name":      m.get("name", ""),
            "artist":    m.get("artist", ""),
            "kind":      gnode.get("kind", "corpus") if gnode else "corpus",
            "source":    gnode.get("source", "corpus") if gnode else "corpus",
            "cluster":   {"id": cid,
                          "confidence": gnode.get("confidence") if gnode else None,
                          "label": _cluster_label(cid)},
            "tags":      tags,
            "genre":     m.get("genre"),
            "playlists": m.get("playlists") or [],
            "similar":   similar,
        }

    # non-corpus query node (deezer / spotify / upload / mpd)
    cid = gnode.get("cluster")
    if qvec is not None:
        sims = corpus.index.embeddings @ qvec
        order = np.argsort(sims)[::-1][:top_n]
        similar = []
        for i in order:
            i = int(i)
            nid = corpus.metadata[i].get("id")
            similar.append({
                "id":       nid,
                "name":     corpus.metadata[i].get("name", ""),
                "artist":   corpus.metadata[i].get("artist", ""),
                "score":    round(float(sims[i]), 3),
                "cluster":  int(corpus.labels[i]),
                "on_graph": nid in node_ids,
            })
    else:                                               # pre-session node: stored edges only
        link_rows.sort(key=lambda r: -(r["score"] or 0.0))
        similar = link_rows[:top_n]

    tags = gnode.get("tags") or []
    if not tags:                                        # placed before tags were persisted
        tags = _inherit_tags(corpus, qvec=qvec,
                             neighbor_ids=[r["id"] for r in link_rows])
        if tags:
            with _graph_lock:                           # append the prediction to the track
                n = _graph["nodes"].get(song_id)
                if n is not None and not n.get("tags"):
                    n["tags"] = tags
                    _save_graph()

    return {
        "id":        song_id,
        "name":      gnode.get("name", ""),
        "artist":    gnode.get("artist", ""),
        "kind":      gnode.get("kind", "query"),
        "source":    gnode.get("source", ""),
        "cluster":   {"id": cid,
                      "confidence": gnode.get("confidence"),
                      "label": gnode.get("cluster_label") or _cluster_label(cid)},
        "tags":      tags,
        "genre":     None,
        "playlists": [],
        "similar":   similar,
    }


# ── Graph persistence ────────────────────────────────────────────────────────

def get_graph() -> dict:
    # "ready" lets the frontend tell "corpus still warming up" apart from "empty
    # session" — before load() finishes, _graph hasn't been read from disk yet.
    with _graph_lock:
        return {"ready": is_ready(),
                "nodes": list(_graph["nodes"].values()),
                "links": list(_graph["links"]),
                "groups": dict(_groups)}


def clear_graph() -> None:
    """Wipe the session map (nodes, links, groups). The embed cache is kept, so
    re-importing previously embedded songs stays instant."""
    with _graph_lock:
        _graph["nodes"].clear()
        _graph["links"].clear()
        _link_keys.clear()
        _query_vecs.clear()
        _groups.clear()
        _save_graph()


def remove_node(node_id: str) -> dict | None:
    """Remove one placed song from the map, along with its links and any grey
    corpus-context neighbors that end up with no remaining links. Returns
    {"removed": [ids...], "node": <the popped node>} (node first in the list),
    or None if the id isn't on the graph."""
    with _graph_lock:
        node = _graph["nodes"].pop(node_id, None)
        if node is None:
            return None
        _query_vecs.pop(node_id, None)

        kept = []
        for l in _graph["links"]:
            if l["source"] == node_id or l["target"] == node_id:
                _link_keys.discard((l["source"], l["target"]))
                continue
            kept.append(l)
        _graph["links"][:] = kept

        linked = {l["source"] for l in kept} | {l["target"] for l in kept}
        orphans = [nid for nid, n in _graph["nodes"].items()
                   if n.get("kind") == "corpus" and nid not in linked]
        for nid in orphans:
            del _graph["nodes"][nid]
        _save_graph()
    return {"removed": [node_id, *orphans], "node": node}


def _save_graph() -> None:
    SESSION_DIR.mkdir(exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(
        {"nodes": list(_graph["nodes"].values()), "links": _graph["links"],
         "groups": _groups}))


def _load_graph() -> None:
    global _graph, _link_keys, _query_vecs, _groups
    if not GRAPH_PATH.exists():
        return
    try:
        data = json.loads(GRAPH_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return
    # Migration: playlist hub nodes and hub→member spokes are gone (members are
    # ordinary query nodes now) — strip them from sessions saved by older code.
    nodes = {n["id"]: n for n in data.get("nodes", [])
             if n.get("kind") != "playlist"}
    links = [l for l in data.get("links", [])
             if l.get("kind") != "member"
             and l["source"] in nodes and l["target"] in nodes]
    _graph = {"nodes": nodes, "links": links}
    _link_keys = {(l["source"], l["target"]) for l in links}

    # Groups registry (for the filter UI). Backfill display names for gids
    # recorded before the registry existed: playlist names from the MPD DB;
    # unknowns fall back to showing the raw gid.
    _groups = dict(data.get("groups") or {})
    seen_gids = {str(n["playlist_pid"]) for n in nodes.values()
                 if n.get("playlist_pid") is not None}
    unnamed = {gid for gid, g in _groups.items() if g.get("name") in (None, "", gid)}
    for gid in (seen_gids - set(_groups)) | unnamed:
        if gid.startswith("album:"):
            name = None
            try:
                info = _deezer_get(f"album/{gid[len('album:'):]}")
                if "error" not in info:
                    artist = (info.get("artist") or {}).get("name", "")
                    name = f"{artist} — {info.get('title', '')}".strip(" —")
            except Exception:  # noqa: BLE001 — offline load must not fail
                pass
            _groups[gid] = {"name": name or gid, "kind": "album"}
        else:
            name = mpd_sql.playlist_name(MPD_DB, gid) if mpd_ready() else None
            _groups[gid] = {"name": name or gid, "kind": "playlist"}

    # Best-effort: rebuild query→query similarity vecs so newly placed songs can
    # cross-link against them — from the corpus for corpus-source songs, from the
    # embed cache for external ones. Uncached external nodes (pre-cache-era
    # deezer/upload) keep their saved edges only.
    _query_vecs = {}
    for n in nodes.values():
        if n.get("kind") != "query":
            continue
        idx = _id_to_idx.get(n["id"])
        if idx is not None:
            raw = _corpus.embeddings[idx]
        else:
            raw = cached_vec(n["id"])
            if raw is None:
                continue
        _query_vecs[n["id"]] = _corpus.index.transform_query(raw)
