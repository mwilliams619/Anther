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
import contextlib
import contextvars
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

# ── human-readable similarity score ─────────────────────────────────────────
# Raw cosine is meaningless to a non-technical user (everything sits at 0.96-
# 0.99) and, being a fixed *corpus-wide* calibration rather than a per-map
# rescale, a given pair's displayed score never shifts just because other
# songs were added to or removed from the map. Two fixed anchors, both drawn
# from the same one-time 200k-random-pair corpus sample _calibrate_qq_threshold
# already computes at load():
#   - SCORE_FLOOR_RAW = _qq_threshold (the QUERY_LINK_PCTL percentile of random
#     corpus pairs) maps to SCORE_FLOOR_DISPLAY. Any pair that clears the
#     qq_threshold (i.e. every edge actually drawn on the map) therefore always
#     reads >= SCORE_FLOOR_DISPLAY — "linked" reliably means "a relatively high
#     score" to the user, per product decision.
#   - SCORE_CEILING_RAW = the single highest cosine observed anywhere in that
#     sample (~0.994, not the unreachable theoretical 1.0 of a song matching
#     itself) maps to 100 — using an attainable ceiling means realistic pairs
#     actually spread across most of the 0-100 range instead of bunching near
#     the bottom.
# Both anchors are set once in _calibrate_qq_threshold(); _display_score() below
# does the linear map.
SCORE_FLOOR_DISPLAY = 55.0
_score_ceiling_raw   = 0.994   # replaced at load() with the corpus-calibrated value

SESSION_DIR          = Path(__file__).parent / "session"
DEFAULT_SESSION_ID   = "default"   # single-user / no-cookie fallback
CUSTOM_PLAYLISTS_DIR = Path(__file__).parent / "custom_playlists"

# ── Lazy singletons ──────────────────────────────────────────────────────────

_corpus = None
_id_to_idx: dict = {}
_id_to_cluster: dict = {}
_profile_by_cluster: dict = {}   # cluster_id → profile dict
_playlist_index: dict = {}   # pid -> {"pid","name","name_norm","n_tracks","indices":[int]}
_playlist_rows: list = []    # _playlist_index values sorted by -n_tracks (search scans)
_custom_playlist_index: dict = {}  # pid -> {pid, name, name_norm, description, tracks, n_tracks}
_corpus_lock = threading.Lock()

_model = _processor = _device = None
_mert_lock = threading.Lock()

# ── Per-session state (Option 1: session-scoped isolation) ───────────────────
# Every browser session gets its own map (graph + links + query vectors +
# group registry) and its own on-disk files under session/<sid>/. State is
# selected per-request via a ContextVar bound in app.py's before_request hook;
# code paths that run off-request (the playlist background worker) bind it
# explicitly with `use_session(sid)`. The reserved "default" session is the
# single-user / no-cookie fallback and inherits the legacy flat files
# (session/graph.json, session/uploads/) via a one-time migration.


def _safe_session_id(session_id: str | None) -> str:
    """Filesystem-safe session id (alnum/_/- only) so it can name a subdir
    without traversal. Falls back to DEFAULT_SESSION_ID for empty/None."""
    if not session_id:
        return DEFAULT_SESSION_ID
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in "-_")
    return safe or DEFAULT_SESSION_ID


class _SessionState:
    """One browser session's map. Holds the in-memory graph and the paths to
    its persisted files. `lock` guards all mutation of graph/link_keys/
    query_vecs/groups, mirroring the old module-level _graph_lock."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = SESSION_DIR / session_id
        self.graph_path = self.dir / "graph.json"
        self.embed_cache_path = self.dir / "embed_cache.sqlite"
        self.uploads_dir = self.dir / "uploads"
        self.graph = {"nodes": {}, "links": []}   # nodes keyed by id; links is a list
        self.link_keys: set = set()                # (source, target) dedupe
        self.query_vecs: dict = {}                 # placed-song id → index-space unit vec
        self.groups: dict = {}                     # gid → {"name", "kind"}
        self.lock = threading.Lock()
        self._loaded = False

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def ensure_loaded(self) -> None:
        """Lazily read this session's graph from disk. Deferred until the
        corpus is ready, since _load_graph rebuilds query vectors from it."""
        if self._loaded or not is_ready():
            return
        with self.lock:
            if self._loaded:
                return
            # Set the flag BEFORE loading: _load_graph → cached_vec → get_session
            # re-enters ensure_loaded on this same session, and threading.Lock is
            # not reentrant, so the flag must already be True to short-circuit.
            self._loaded = True
            _load_graph(self)


# Registry of live sessions, and the request-scoped selector.
_sessions: dict[str, _SessionState] = {}
_sessions_lock = threading.Lock()
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "anther_session_id", default=DEFAULT_SESSION_ID)


def _migrate_legacy_default(st: _SessionState) -> None:
    """One-time: fold the pre-multi-session flat files (session/graph.json,
    session/embed_cache.sqlite, session/uploads/) into the default session
    dir so existing maps and uploads survive the upgrade."""
    st.dir.mkdir(parents=True, exist_ok=True)
    moves = [
        (SESSION_DIR / "graph.json",         st.graph_path),
        (SESSION_DIR / "embed_cache.sqlite", st.embed_cache_path),
        (SESSION_DIR / "uploads",            st.uploads_dir),
    ]
    for old, new in moves:
        if old.exists() and not new.exists():
            try:
                old.rename(new)
            except OSError:
                pass


def set_session(session_id: str | None) -> contextvars.Token:
    """Bind the current session for this request/thread. Returns a token that
    `reset_session` restores. Called once per request by app.py."""
    return _session_id_var.set(_safe_session_id(session_id))


def reset_session(token: contextvars.Token) -> None:
    _session_id_var.reset(token)


@contextlib.contextmanager
def use_session(session_id: str | None):
    """Context manager form for off-request threads (background workers)."""
    token = set_session(session_id)
    try:
        yield get_session()
    finally:
        reset_session(token)


def current_session_id() -> str:
    return _session_id_var.get()


def get_session() -> _SessionState:
    """The _SessionState for the current contextvar id, created + loaded on
    first use (thread-safe). Off-request threads must bind via use_session
    first, else they get the default session."""
    sid = current_session_id()
    with _sessions_lock:
        st = _sessions.get(sid)
        if st is None:
            st = _SessionState(sid)
            if sid == DEFAULT_SESSION_ID:
                _migrate_legacy_default(st)
            st.ensure_dirs()
            _sessions[sid] = st
    st.ensure_loaded()
    return st


# ── Corpus + MERT loading ────────────────────────────────────────────────────

def load() -> ReferenceCorpus:
    """Load the frozen corpus once (idempotent, thread-safe)."""
    global _corpus, _id_to_idx, _id_to_cluster, _playlist_index, _playlist_rows, _profile_by_cluster
    global _custom_playlist_index
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
        _profile_by_cluster = {p["cluster_id"]: p for p in (corpus.profiles or [])}
        _playlist_index, _playlist_rows = _build_playlist_index(corpus)
        _custom_playlist_index = _load_custom_playlists()
        _calibrate_qq_threshold(corpus)
        # Per-session graphs are loaded lazily on first access (see
        # _SessionState.ensure_loaded); nothing to load here.
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


def _load_custom_playlists() -> dict:
    """Read all *.json files from ui/custom_playlists/ and return a pid-keyed dict.
    Each file must have at minimum ``pid``, ``name``, and ``tracks``
    (list of {id, name, artist})."""
    index: dict = {}
    if not CUSTOM_PLAYLISTS_DIR.is_dir():
        return index
    for path in sorted(CUSTOM_PLAYLISTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        pid = data.get("pid")
        if not pid:
            continue
        name = data.get("name") or path.stem
        tracks = data.get("tracks") or []
        index[str(pid)] = {
            "pid":         str(pid),
            "name":        name,
            "name_norm":   _norm(name),
            "description": data.get("description", ""),
            "tracks":      tracks,
            "n_tracks":    len(tracks),
        }
    return index


def _calibrate_qq_threshold(corpus, n_pairs: int = 200_000) -> None:
    """Set the query↔query cosine cutoff to the QUERY_LINK_PCTL percentile of
    random corpus-pair cosines (index space) — so an edge means "more similar
    than that fraction of released music," not an arbitrary absolute cosine.

    Reuses the same random-pair sample to also set _score_ceiling_raw (see
    _display_score) — one 200k-pair draw, two calibrated constants."""
    global _qq_threshold, _score_ceiling_raw
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
    _score_ceiling_raw = float(cos.max())


def _display_score(raw_cos: float, clip_low: bool = True) -> float:
    """Map a raw cosine to the 0-100 human-readable similarity score.

    _qq_threshold -> SCORE_FLOOR_DISPLAY, _score_ceiling_raw -> 100, linear
    between. `clip_low=True` (map edges, which structurally can't fall below
    _qq_threshold) floors the result at SCORE_FLOOR_DISPLAY; `clip_low=False`
    (the corpus-wide "show more" search, which can surface pairs that don't
    clear the link threshold) lets the score read honestly below the floor,
    only clipping at 0. Both cases clip at 100 on top."""
    span = _score_ceiling_raw - _qq_threshold
    frac = (raw_cos - _qq_threshold) / span if span > 0 else 1.0
    score = SCORE_FLOOR_DISPLAY + (100.0 - SCORE_FLOOR_DISPLAY) * frac
    lo = SCORE_FLOOR_DISPLAY if clip_low else 0.0
    return float(np.clip(score, lo, 100.0))


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
    cache_path = get_session().embed_cache_path
    if not track_id or not cache_path.exists():
        return None
    try:
        con = sqlite3.connect(str(cache_path))
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
    st = get_session()
    with _cache_lock:
        st.ensure_dirs()
        con = sqlite3.connect(str(st.embed_cache_path))
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
    Custom playlists from ui/custom_playlists/ are always prepended.
    Returns {"results": [...], "notice": str|None}."""
    load()
    q = (q or "").strip()
    if not q:
        return {"results": [], "notice": None}

    # Custom playlists always searched regardless of MPD availability
    tokens = _norm(q).split()
    custom_hits = []
    for e in _custom_playlist_index.values():
        if not all(t in e["name_norm"] for t in tokens):
            continue
        custom_hits.append({
            "pid":         e["pid"],
            "name":        e["name"],
            "n_tracks":    e["n_tracks"],
            "n_in_corpus": e["n_tracks"],
            "source":      "custom",
            "score":       round(_ratio(q, e["name"]), 3),
        })
    custom_hits.sort(key=lambda h: -h["score"])

    if mpd_ready():
        hits = mpd_sql.search_playlists_db(MPD_DB, q, limit=limit)
        for h in hits:
            entry = _playlist_index.get(h["pid"])
            h["n_in_corpus"] = len(entry["indices"]) if entry else 0
            h["source"] = "mpd"
        return {"results": custom_hits + hits, "notice": None}

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
    return {"results": custom_hits + results, "notice": MPD_PREP_HINT}


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
        try:
            with _embed_lock:
                model, processor, device = _mert()
                vec = embed_query(path, corpus, model, processor, device)
        finally:
            if cleanup:
                cleanup()
    else:
        # Non-upload sources (deezer, spotify, etc.) use two-tier fallback:
        # Try the provided preview_url, fall back to Deezer name/artist match
        # if the URL is dead. Same logic as playlist_jobs.py worker.
        try:
            vec, _method = resolve_and_embed({
                "id": result.get("id"),
                "name": result.get("title", ""),
                "artist": result.get("artist", ""),
                "preview_url": result.get("preview_url"),
            })
        except PlacementSkip as e:
            raise ValueError(f"Could not place song: {e.reason}") from e

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
    # Recommendations aren't guaranteed map edges (seeds' centroid/topk score is
    # a different quantity than a pairwise qq cosine) so use the same open
    # floor as the "show more" search — a weak recommendation can honestly
    # read below 55 rather than being floored to look stronger than it is.
    for r in results:
        r["score"] = round(_display_score(float(r["score"]), clip_low=False), 1)

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
    st = get_session()
    added_nodes, added_links = [], []
    with st.lock:
        existing = st.graph["nodes"].get(node["id"])
        if existing is None:
            st.graph["nodes"][node["id"]] = node
        else:                                                  # re-add → promote to query
            existing.update(node)                              # carries playlist_pid etc.
            node = existing
        # Always report the node in this call's fragment — callers (playlist/
        # album/demo placement) rely on getting back every track they asked to
        # place, not just ones that were brand-new to the graph. Omitting
        # "already there" nodes here is what made freshly-cleared-then-reloaded
        # collections render incomplete until a full page refresh.
        added_nodes.append(node)

        # ── query↔query similarity edges (top-QQ_MAX_PER_NODE by score, so a
        # coherent playlist batch can't flood O(m²) links) ──
        if qvec is not None:
            cands = []
            for other_id, ovec in st.query_vecs.items():
                if other_id == node["id"]:
                    continue
                score = float(np.dot(qvec, ovec))
                if score < _qq_threshold:
                    continue
                key = (node["id"], other_id)
                rkey = (other_id, node["id"])
                if key in st.link_keys or rkey in st.link_keys:
                    continue
                cands.append((score, other_id))
            cands.sort(reverse=True)
            for score, other_id in cands[:QQ_MAX_PER_NODE]:
                # "value" = raw cosine (kept for recalibration/debugging); "score"
                # = the fixed-calibration 0-100 human-readable number the UI
                # actually displays (see _display_score) — computed once here so
                # it never needs recomputing per-request or per-render.
                link = {"source": node["id"], "target": other_id,
                        "value": round(score, 3),
                        "score": round(_display_score(score, clip_low=True), 1),
                        "kind": "qq"}
                st.graph["links"].append(link)
                st.link_keys.add((node["id"], other_id))
                added_links.append(link)
            st.query_vecs[node["id"]] = qvec

        _save_graph(st)
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
                st = get_session()
                with st.lock:
                    existing = st.graph["nodes"].get(nid)
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
    st = get_session()
    with st.lock:                                         # remember the group's display name
        st.groups[str(gid)] = {"name": name,
                               "kind": "album" if str(gid).startswith("album:") else "playlist"}
        _save_graph(st)
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


DEMO_PLAYLIST_PID = "demo_top20_2025"


def place_demo_top20() -> dict:
    """
    Demo cluster: place the pre-resolved 2025 year-end top-20 tracks stored in
    ``ui/custom_playlists/demo_top20_2025.json`` (Deezer track ids baked in —
    no live search). Every track was embedded once and is cached in
    session/embed_cache.sqlite, so this is now instant: place_playlist()
    picks the custom-playlist branch, which resolves each id straight from
    the cache with zero network calls.
    """
    return place_playlist(DEMO_PLAYLIST_PID)


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

    # Custom playlists checked before MPD/corpus
    custom = _custom_playlist_index.get(str(pid))
    if custom is not None:
        rows = []
        for t in custom["tracks"][:IMPORT_CAP]:
            tid = str(t.get("id", ""))
            # Prefer bare ID if in corpus; fall back to spotify: prefix
            if _id_to_idx.get(tid) is None and not tid.startswith(("spotify:", "deezer:")):
                nid = f"spotify:{tid}"
            else:
                nid = tid
            rows.append({
                "id":          nid,
                "name":        t.get("name", ""),
                "artist":      t.get("artist", ""),
                "preview_url": t.get("preview_url"),
            })
        fragment, n_immediate, pending = _place_collection_rows(
            rows, str(pid), "custom")
        return _collection_response(str(pid), custom["name"], rows,
                                    custom["n_tracks"], fragment, n_immediate, pending)

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

def _nearest_artists(qvec, n: int = 5, exclude_artist: str = "") -> list:
    """Top-n unique corpus artists nearest to the index-space qvec.
    exclude_artist (normalised comparison) is the track's own artist."""
    sims = _corpus.index.embeddings @ qvec
    order = np.argsort(sims)[::-1][: n * 8]
    seen, artists = set(), []
    excl = _norm(exclude_artist)
    for i in order:
        a = _corpus.metadata[i].get("artist", "")
        a_norm = _norm(a)
        if not a or a_norm == excl or a_norm in seen:
            continue
        seen.add(a_norm)
        artists.append(a)
        if len(artists) >= n:
            break
    return artists


def _pitch_summary(label: str, nearest: list, n: int = 3) -> str:
    """One-line, copy-pasteable positioning sentence for pitching this track
    (playlist submissions, DSP forms, etc.) — cluster label + top-n nearest
    artists. Empty string if there isn't enough to say anything (no label and
    no nearest artists), so the caller can omit the line entirely."""
    def _join(names):
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return " and ".join(names)
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    names = nearest[:n]
    if label and names:
        return f"Sits in {label} territory — sounds closest to {_join(names)}."
    if label:
        return f"Sits in {label} territory."
    if names:
        return f"Sounds closest to {_join(names)}."
    return ""


def _positioning(cluster_id, qvec, artist: str) -> dict:
    """Build the positioning report dict surfaced in the click-detail panel.
    cluster_id: Leiden cluster assignment for this track.
    qvec: index-space vector (corpus.index.transform_query output) or None.
    artist: track's artist string (excluded from nearest-artist list).
    """
    profile = _profile_by_cluster.get(cluster_id) if cluster_id is not None else None
    label = (profile or {}).get("label_final") or (profile or {}).get("label", "")
    size = (profile or {}).get("size", 0)

    nearest = _nearest_artists(qvec, n=5, exclude_artist=artist) if qvec is not None else []

    return {
        "cluster_label": label,
        "cluster_size":  size,
        "nearest_artists": nearest,
        "pitch_summary": _pitch_summary(label, nearest),
    }


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
        st = get_session()
        with st.lock:
            gnode = st.graph["nodes"].get(song_id)
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


_spotify_id_cache: dict[str, str | None] = {}   # song_id → bare Spotify track id | None


def get_spotify_track_id(song_id: str) -> str | None:
    """Resolve a bare Spotify track id for the no-login iframe embed
    (open.spotify.com/embed/track/<id>) — this gives logged-in Spotify
    visitors full-track playback with zero OAuth/app registration, and
    falls back to a 30s preview automatically for everyone else.

    Every MPD-sourced corpus row already carries its native ``spotify:<id>``
    node id — free, no lookup. Deezer/upload-origin tracks have no such id
    on hand; for those we optionally cross-match by title/artist against the
    Spotify Search API (same SPOTIFY_CLIENT_ID/SECRET-gated client-credentials
    tier already used by the search fallback), caching the result — None
    means "no confident match", cached too so a repeat click doesn't re-hit
    the API.
    """
    if song_id in _spotify_id_cache:
        return _spotify_id_cache[song_id]

    if song_id.startswith("spotify:"):
        tid = song_id.split(":", 1)[1]
        _spotify_id_cache[song_id] = tid
        return tid

    if not spotify_configured():
        _spotify_id_cache[song_id] = None
        return None

    idx = _id_to_idx.get(song_id)
    if idx is not None:
        corpus = load()
        name = corpus.metadata[idx].get("name", "")
        artist = corpus.metadata[idx].get("artist", "")
    else:
        st = get_session()
        with st.lock:
            gnode = st.graph["nodes"].get(song_id)
        if gnode is None:
            _spotify_id_cache[song_id] = None
            return None
        name, artist = gnode.get("name", ""), gnode.get("artist", "")

    tid = None
    if name and artist:
        try:
            from anther_ml.mpd_ingest import get_spotify_token
            token = get_spotify_token()
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": f"track:{name} artist:{artist}", "type": "track", "limit": 1},
                timeout=15,
            )
            r.raise_for_status()
            items = ((r.json().get("tracks") or {}).get("items")) or []
            if items:
                tid = items[0].get("id")
        except Exception:
            tid = None
    _spotify_id_cache[song_id] = tid
    return tid


def _map_neighbors(song_id: str) -> list:
    """Songs actually connected to ``song_id`` by a drawn map edge (qq-link),
    using each edge's persisted display score. This is what the click panel
    shows by default — cheap (no corpus search, just an in-memory link scan)
    and guaranteed to match the lines drawn on screen, unlike a fresh
    nearest-neighbor search which can rank differently than what's linked."""
    st = get_session()
    with st.lock:
        rows = []
        for l in st.graph["links"]:
            other = (l["target"] if l["source"] == song_id
                     else l["source"] if l["target"] == song_id else None)
            if other is None:
                continue
            on = st.graph["nodes"].get(other, {})
            score = l.get("score")
            if score is None and l.get("value") is not None:   # pre-migration edge
                score = round(_display_score(l["value"], clip_low=True), 1)
            rows.append({
                "id":       other,
                "name":     on.get("name", ""),
                "artist":   on.get("artist", ""),
                "score":    score,
                "on_graph": True,
            })
    rows.sort(key=lambda r: -(r["score"] or 0.0))
    return rows


def song_detail(song_id: str, top_n: int = 10, expand: bool = False) -> dict | None:
    """
    Full detail payload for the click panel: identity, micro-genre tags,
    playlist membership, and the songs it's connected to on the map.

    ``map_neighbors`` (cheap — an in-memory link scan) is always included.
    The expensive corpus-wide nearest-neighbor search only runs when
    ``expand=True``, returned separately as ``similar`` with map_neighbors'
    ids excluded (the "show more like this" list). Returns None for unknown
    ids.
    """
    corpus = load()
    idx = _id_to_idx.get(song_id)

    st = get_session()
    with st.lock:
        gnode = st.graph["nodes"].get(song_id)
        gnode = dict(gnode) if gnode is not None else None
        node_ids = set(st.graph["nodes"])
        qvec = st.query_vecs.get(song_id)

    if idx is None and gnode is None:
        return None

    map_neighbors = _map_neighbors(song_id)
    seen_ids = {song_id} | {r["id"] for r in map_neighbors}

    if idx is not None:                                 # corpus track
        m = corpus.metadata[idx]
        track_tags = corpus.track_tags
        tags = track_tags[idx].get("tags", []) if track_tags is not None else []
        similar = []
        if expand:
            rows = corpus.index.query(corpus.embeddings[idx], top_k=top_n + 1 + len(seen_ids))
            rows = [r for r in rows if r.get("id") not in seen_ids][:top_n]
            similar = [{
                "id":       r.get("id"),
                "name":     r.get("name", ""),
                "artist":   r.get("artist", ""),
                "score":    round(_display_score(float(r.get("score", 0.0)), clip_low=False), 1),
                "on_graph": r.get("id") in node_ids,
            } for r in rows]
        return {
            "id":            song_id,
            "name":          m.get("name", ""),
            "artist":        m.get("artist", ""),
            "kind":          gnode.get("kind", "corpus") if gnode else "corpus",
            "source":        gnode.get("source", "corpus") if gnode else "corpus",
            "tags":          tags,
            "genre":         m.get("genre"),
            "playlists":     m.get("playlists") or [],
            "map_neighbors": map_neighbors,
            "similar":       similar,
            "positioning":   _positioning(
                _id_to_cluster.get(song_id),
                corpus.index.embeddings[idx],
                m.get("artist", ""),
            ),
        }

    # non-corpus query node (deezer / spotify / upload / mpd)
    similar = []
    if expand:
        if qvec is not None:
            sims = corpus.index.embeddings @ qvec
            order = np.argsort(sims)[::-1]
            for i in order:
                i = int(i)
                nid = corpus.metadata[i].get("id")
                if nid in seen_ids:
                    continue
                similar.append({
                    "id":       nid,
                    "name":     corpus.metadata[i].get("name", ""),
                    "artist":   corpus.metadata[i].get("artist", ""),
                    "score":    round(_display_score(float(sims[i]), clip_low=False), 1),
                    "on_graph": nid in node_ids,
                })
                if len(similar) >= top_n:
                    break
        # pre-session node with no cached vector: no corpus-wide search
        # possible — map_neighbors (already computed above) is all we have.

    tags = gnode.get("tags") or []
    if not tags:                                        # placed before tags were persisted
        tags = _inherit_tags(corpus, qvec=qvec,
                             neighbor_ids=[r["id"] for r in map_neighbors])
        if tags:
            with st.lock:                               # append the prediction to the track
                n = st.graph["nodes"].get(song_id)
                if n is not None and not n.get("tags"):
                    n["tags"] = tags
                    _save_graph(st)

    return {
        "id":            song_id,
        "name":          gnode.get("name", ""),
        "artist":        gnode.get("artist", ""),
        "kind":          gnode.get("kind", "query"),
        "source":        gnode.get("source", ""),
        "tags":          tags,
        "genre":         None,
        "playlists":     [],
        "map_neighbors": map_neighbors,
        "similar":       similar,
        "positioning":   _positioning(
            gnode.get("cluster"),
            qvec,
            gnode.get("artist", ""),
        ),
    }


# ── Graph persistence ────────────────────────────────────────────────────────

def get_graph() -> dict:
    # "ready" lets the frontend tell "corpus still warming up" apart from "empty
    # session" — before load() finishes, _graph hasn't been read from disk yet.
    st = get_session()
    with st.lock:
        return {"ready": is_ready(),
                "nodes": list(st.graph["nodes"].values()),
                "links": list(st.graph["links"]),
                "groups": dict(st.groups)}


def clear_graph() -> None:
    """Wipe the session map (nodes, links, groups). The embed cache is kept, so
    re-importing previously embedded songs stays instant.

    Also forgets all background playlist/album/demo embed jobs: without this,
    a job still marked "running" for a given gid (e.g. the demo's fixed
    "demo_top20_2025" pid) makes the next re-add's ``playlist_jobs.start()``
    return that stale job_id instead of queuing the new pending tracks —
    they're silently dropped rather than placed, so each clear+reload cycle
    comes back with fewer tracks than the last.
    """
    st = get_session()
    with st.lock:
        st.graph["nodes"].clear()
        st.graph["links"].clear()
        st.link_keys.clear()
        st.query_vecs.clear()
        st.groups.clear()
        _save_graph(st)
    import playlist_jobs
    playlist_jobs.forget_all()


def remove_node(node_id: str) -> dict | None:
    """Remove one placed song from the map, along with its links and any grey
    corpus-context neighbors that end up with no remaining links. Returns
    {"removed": [ids...], "node": <the popped node>} (node first in the list),
    or None if the id isn't on the graph."""
    st = get_session()
    with st.lock:
        node = st.graph["nodes"].pop(node_id, None)
        if node is None:
            return None
        st.query_vecs.pop(node_id, None)

        kept = []
        for l in st.graph["links"]:
            if l["source"] == node_id or l["target"] == node_id:
                st.link_keys.discard((l["source"], l["target"]))
                continue
            kept.append(l)
        st.graph["links"][:] = kept

        linked = {l["source"] for l in kept} | {l["target"] for l in kept}
        orphans = [nid for nid, n in st.graph["nodes"].items()
                   if n.get("kind") == "corpus" and nid not in linked]
        for nid in orphans:
            del st.graph["nodes"][nid]
        _save_graph(st)
    return {"removed": [node_id, *orphans], "node": node}


def _save_graph(st: "_SessionState") -> None:
    st.dir.mkdir(parents=True, exist_ok=True)
    st.graph_path.write_text(json.dumps(
        {"nodes": list(st.graph["nodes"].values()), "links": st.graph["links"],
         "groups": st.groups}))


def _load_graph(st: "_SessionState") -> None:
    if not st.graph_path.exists():
        return
    try:
        data = json.loads(st.graph_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    # Migration: playlist hub nodes and hub→member spokes are gone (members are
    # ordinary query nodes now) — strip them from sessions saved by older code.
    nodes = {n["id"]: n for n in data.get("nodes", [])
             if n.get("kind") != "playlist"}
    links = [l for l in data.get("links", [])
             if l.get("kind") != "member"
             and l["source"] in nodes and l["target"] in nodes]
    st.graph = {"nodes": nodes, "links": links}
    st.link_keys = {(l["source"], l["target"]) for l in links}

    # Backfill: sessions saved before the "score" field existed only have the
    # raw cosine ("value"). Compute it once here from the already-calibrated
    # _qq_threshold/_score_ceiling_raw (set by _calibrate_qq_threshold, called
    # just before _load_graph in load()) so old sessions don't need re-placing.
    backfilled = False
    for l in links:
        if l.get("kind") == "qq" and l.get("score") is None and l.get("value") is not None:
            l["score"] = round(_display_score(l["value"], clip_low=True), 1)
            backfilled = True
    if backfilled:
        _save_graph(st)

    # Groups registry (for the filter UI). Backfill display names for gids
    # recorded before the registry existed: playlist names from the MPD DB;
    # unknowns fall back to showing the raw gid.
    st.groups = dict(data.get("groups") or {})
    seen_gids = {str(n["playlist_pid"]) for n in nodes.values()
                 if n.get("playlist_pid") is not None}
    unnamed = {gid for gid, g in st.groups.items() if g.get("name") in (None, "", gid)}
    for gid in (seen_gids - set(st.groups)) | unnamed:
        if gid.startswith("album:"):
            name = None
            try:
                info = _deezer_get(f"album/{gid[len('album:'):]}")
                if "error" not in info:
                    artist = (info.get("artist") or {}).get("name", "")
                    name = f"{artist} — {info.get('title', '')}".strip(" —")
            except Exception:  # noqa: BLE001 — offline load must not fail
                pass
            st.groups[gid] = {"name": name or gid, "kind": "album"}
        else:
            name = mpd_sql.playlist_name(MPD_DB, gid) if mpd_ready() else None
            st.groups[gid] = {"name": name or gid, "kind": "playlist"}

    # Best-effort: rebuild query→query similarity vecs so newly placed songs can
    # cross-link against them — from the corpus for corpus-source songs, from the
    # embed cache for external ones. Uncached external nodes (pre-cache-era
    # deezer/upload) keep their saved edges only.
    st.query_vecs = {}
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
        st.query_vecs[n["id"]] = _corpus.index.transform_query(raw)
