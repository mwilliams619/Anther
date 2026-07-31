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
import uuid
from pathlib import Path

import numpy as np
import requests

from anther_ml import mpd_sql
from anther_ml import merit as merit_mod
from anther_ml import calibration as link_calibration
from anther_ml.cluster import assign_cluster_knn
from anther_ml.corpus.bundle import ReferenceCorpus
from anther_ml.corpus.place import embed_query, embed_query_dual, place, recommend_from_seeds
from anther_ml.corpus.popularity import load_popularity_by_track_id, popularity_percentiles
from anther_ml.spotify_deezer import _deezer_get, match_deezer_track, _norm, _ratio

# ── Config ───────────────────────────────────────────────────────────────────

CORPUS_DIR      = os.environ.get("ANTHER_CORPUS", "models/corpus_mpd_100k_merit_ext_billboard")
MPD_DB          = os.environ.get(
    "ANTHER_MPD_DB",
    str(Path(__file__).parent.parent / "data" / "mpd_dump" / "spotifydbdumpshare.sqlite"),
)
POPULARITY_PATH = os.environ.get(
    "ANTHER_POPULARITY", "data/billboard/popularity_by_track_id.json"
)
POPULARITY_BETA = float(os.environ.get("ANTHER_POPULARITY_BETA", "0.15"))
ARTIST_CLUSTERING_DIR = os.environ.get(
    "ANTHER_ARTIST_CLUSTERING", "data/artist_clustering"
)
ARTIST_MODE_ENABLED = os.environ.get("ANTHER_ARTIST_MODE", "1").lower() not in (
    "0", "false", "no", "off"
)
ARTIST_PROFILES_DB = os.environ.get(
    "ANTHER_ARTIST_PROFILES", "data/artist_profiles.sqlite"
)
ARTIST_MAX_EDGES_PER_NODE = 6
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
QUERY_LINK_PCTL = link_calibration.QUERY_LINK_PCTL
_qq_threshold   = link_calibration.DEFAULT_QQ_THRESHOLD  # replaced at load()
# Full calibrated LinkThresholds (qq/close/near-identical cutoffs + ceiling) —
# the single source of truth; _qq_threshold/_score_ceiling_raw below are kept
# as separate globals only because ~10 call sites already reference them by
# name. Also handed to mentor/anther_service.py via the link_calibration.json
# sidecar (see _calibrate_qq_threshold) so both processes band cosines the
# same way.
_link_thresholds = link_calibration.LinkThresholds.defaults()
# MERIT-aggregate counterpart of _link_thresholds above — calibrated from the
# bundle's link_calibration_merit.json sidecar (see merit_index.py) when the
# corpus carries a MERIT-aggregate index, else left None. None is the signal
# every MERIT-space call site falls back to the MERT-space path (see
# _display_score/_merge_fragment) — this is how a bundle built without
# --capture-merit-backbone keeps working unmodified.
_merit_link_thresholds: link_calibration.LinkThresholds | None = None
# Per-factor (melody/rhythm/timbre) counterpart — each factor calibrated off
# its OWN raw-cosine distribution rather than sharing _merit_link_thresholds'
# aggregate-calibrated scale (see calibration.py's
# CALIBRATION_FILENAME_MERIT_FACTORS docstring for why: e.g. timbre commonly
# runs much hotter than the aggregate, so a shared scale clips timbre's
# display score to 100 far more often than melody/rhythm's). None if the
# bundle predates this sidecar — _breakdown_scores falls back to
# _merit_link_thresholds for every factor in that case.
_merit_factor_thresholds: dict[str, link_calibration.LinkThresholds] | None = None

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
SCORE_FLOOR_DISPLAY = link_calibration.SCORE_FLOOR_DISPLAY
_score_ceiling_raw   = link_calibration.DEFAULT_SCORE_CEILING_RAW  # replaced at load()

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
_popularity_pct: dict = {}  # track_id -> popularity percentile in [0, 1], empty if sidecar missing

# Artist clustering mode (Phase 2) — None until load() is called
_artist_meta: list = []           # list of {artist, n_tracks, sources, sample_track}
_artist_embeddings: np.ndarray = None  # shape (n_artists, 1024)
_artist_labels: np.ndarray = None     # shape (n_artists,) — Leiden cluster IDs
_artist_embedding_2d: np.ndarray = None  # shape (n_artists, 2) — UMAP projection
_artist_id_map: dict = {}         # artist name -> index in _artist_meta (for search)
# Low-confidence pool: corpus artists with 1-4 tracks, excluded from the frozen
# ≥5-track reference bundle but kNN-placeable against it on demand. Built in
# memory from the corpus each load; each row is
# {artist, n_tracks, track_indices, sources, sample_track}. Node id is
# "lowconf:<idx>" where idx indexes _artist_lowconf_rows.
_artist_lowconf_rows: list = []
_artist_lowconf_by_norm: dict = {}  # normalized name -> index in _artist_lowconf_rows
_artist_leiden: dict = None       # loaded Leiden pickle: {labels, scaler, pca, ...}
_artist_space: np.ndarray = None  # L2-normalized Leiden clustering space
_artist_thresholds: link_calibration.LinkThresholds | None = None
_artist_ready = False
_artist_error = "artist data has not loaded"
# Durable enrichment profiles (name/image/following/genres/origin/labels), keyed
# by normalized artist name so they survive a corpus rebuild. Read-only here;
# built offline by `python -m anther_ml.artist_enrichment`. Empty if the DB is
# absent — profile fields simply don't appear. See ARTIST_ENRICHMENT_PLAN.md.
_artist_profiles: dict = {}       # _norm(name) -> compact api_profile dict

ARTIST_CLUSTER_LABELS = {
    0: "Industrial/experimental electronic", 1: "Cinematic/orchestral/ambient",
    2: "Metal", 3: "Hip-hop/rap", 4: "Classic rock/punk", 5: "Trance/EDM",
    6: "Country", 7: "International/world pop", 8: "Pop",
    9: "Trap/modern hip-hop", 10: "Pop-rock/alt-pop", 11: "Alt-rock/punk",
    12: "New age/instrumental", 13: "Jazz", 14: "Latin/salsa",
    15: "Early blues & jazz vocalists", 16: "Film & orchestral score composers",
    17: "Classical choral/early music", 18: "Singer-songwriter/Americana",
    19: "Baroque classical", 20: "Mid-century pop/crooners", 21: "Reggae/ska",
    22: "Euro schlager/adult contemporary", 23: "Turkish/Middle Eastern pop",
    24: "Novelty/comedy/children's", 25: "Soul/funk",
    26: "Classical & flamenco guitar", 27: "1950s-60s rock & roll",
}

ARTIST_DEMO_NAMES = [
    "Drake", "Kendrick Lamar", "Beyoncé", "Taylor Swift", "Metallica",
    "Miles Davis", "Johnny Cash", "Aretha Franklin", "Madonna", "David Bowie",
    "Bad Bunny", "Lady Gaga", "Black Sabbath", "Frank Sinatra", "Willie Nelson",
    "Adele", "Rihanna", "Eminem", "Dolly Parton", "The Cure",
]

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
        self.artist_graph_path = self.dir / "artist_graph.json"
        self.embed_cache_path = self.dir / "embed_cache.sqlite"
        self.uploads_dir = self.dir / "uploads"
        self.graph = {"nodes": {}, "links": []}   # nodes keyed by id; links is a list
        self.link_keys: set = set()                # (source, target) dedupe
        self.query_vecs: dict = {}                 # placed-song id → MERT index-space unit vec
        self.merit_vecs: dict = {}                 # placed-song id → MERIT-aggregate index-space unit vec
        self.groups: dict = {}                     # gid → {"name", "kind"}
        self.artist_graph = {"nodes": {}, "links": []}
        self.artist_vectors: dict[str, np.ndarray] = {}
        self._artist_loaded = False
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
            _load_artist_graph(self)


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
    global _custom_playlist_index, _popularity_pct
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
        pop_path = Path(POPULARITY_PATH)
        if pop_path.exists():
            by_id = load_popularity_by_track_id(pop_path)
            _popularity_pct = popularity_percentiles(by_id)
            print(f"[atlas] loaded popularity sidecar: {len(_popularity_pct)} tracks -> percentiles")
        else:
            _popularity_pct = {}
            print(f"[atlas] no popularity sidecar at {pop_path} — reranking disabled")
        # Load artist clustering data (Phase 2) if available
        _load_artist_clustering()
        _load_artist_profiles()
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


def _load_artist_clustering() -> None:
    """Load and validate the immutable artist clustering bundle."""
    global _artist_meta, _artist_embeddings, _artist_labels, _artist_embedding_2d
    global _artist_id_map, _artist_leiden
    global _artist_lowconf_rows, _artist_lowconf_by_norm
    global _artist_space, _artist_thresholds, _artist_ready, _artist_error

    _artist_ready = False
    _artist_error = "artist mode is disabled"
    if not ARTIST_MODE_ENABLED:
        print("[atlas] artist mode disabled by ANTHER_ARTIST_MODE")
        return
    
    art_dir = Path(ARTIST_CLUSTERING_DIR)
    if not art_dir.exists():
        _artist_error = f"artist clustering directory not found: {art_dir}"
        print(f"[atlas] {_artist_error}")
        return
    
    try:
        import pickle
        required = ["artist_meta.json", "artist_embeddings.npy", "artist_labels.npy",
                    "artist_embedding_2d.npy", "artist_leiden.pkl"]
        missing = [name for name in required if not (art_dir / name).is_file()]
        if missing:
            raise ValueError("missing artist artifacts: " + ", ".join(missing))

        _artist_meta = json.loads((art_dir / "artist_meta.json").read_text())
        _artist_embeddings = np.load(art_dir / "artist_embeddings.npy", allow_pickle=False)
        _artist_labels = np.load(art_dir / "artist_labels.npy", allow_pickle=False)
        _artist_embedding_2d = np.load(art_dir / "artist_embedding_2d.npy", allow_pickle=False)
        with open(art_dir / "artist_leiden.pkl", "rb") as f:
            _artist_leiden = pickle.load(f)

        n = len(_artist_meta)
        lengths = {
            "metadata": n,
            "embeddings": len(_artist_embeddings),
            "labels": len(_artist_labels),
            "coordinates": len(_artist_embedding_2d),
            "clustering_space": len(_artist_leiden.get("clustering_space", [])),
        }
        if n == 0 or len(set(lengths.values())) != 1:
            raise ValueError("artist artifact row mismatch: " +
                             ", ".join(f"{k}={v}" for k, v in lengths.items()))
        if _artist_embeddings.ndim != 2 or _artist_embedding_2d.shape[1] != 2:
            raise ValueError("artist embedding artifacts have invalid dimensions")
        if not np.isfinite(_artist_embeddings).all() or not np.isfinite(_artist_embedding_2d).all():
            raise ValueError("artist embedding artifacts contain non-finite values")

        _artist_id_map = {}
        for i, entry in enumerate(_artist_meta):
            normalized = _norm(entry.get("artist", ""))
            if normalized:
                _artist_id_map.setdefault(normalized, i)

        _build_lowconf_pool()

        space = np.asarray(_artist_leiden["clustering_space"], dtype=np.float32)
        norms = np.linalg.norm(space, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("artist clustering space contains zero vectors")
        _artist_space = space / norms
        artist_index = type("ArtistIndex", (), {"embeddings": _artist_space})()
        _artist_thresholds = link_calibration.calibrate_link_thresholds(
            artist_index, pctl=95.0, seed=0)
        _artist_ready = True
        _artist_error = ""
        print(f"[atlas] loaded artist clustering: {n} artists, "
              f"embeddings shape {_artist_embeddings.shape}, "
              f"clusters {len(np.unique(_artist_labels))}")
    except Exception as e:
        _artist_meta, _artist_id_map = [], {}
        _artist_lowconf_rows, _artist_lowconf_by_norm = [], {}
        _artist_embeddings = _artist_labels = _artist_embedding_2d = None
        _artist_leiden = _artist_space = _artist_thresholds = None
        _artist_error = str(e)
        print(f"[atlas] artist mode unavailable: {_artist_error}")


def _build_lowconf_pool() -> None:
    """Index corpus artists with 1-4 tracks (the low-confidence pool).

    These artists are excluded from the frozen ≥5-track reference bundle (their
    means are too noisy to *build* clusters from) but can still be kNN-assigned
    into it on demand — the artist-level analogue of placing a non-corpus song
    against the frozen song corpus. Same artist grouping as
    scripts/artist_clustering/01_aggregate_artist_embeddings.py, but keeping the
    1-4 track artists instead of discarding them.
    """
    global _artist_lowconf_rows, _artist_lowconf_by_norm
    _artist_lowconf_rows, _artist_lowconf_by_norm = [], {}
    if _corpus is None:
        return
    by_artist_idx: dict[str, list[int]] = {}
    for i, m in enumerate(_corpus.metadata):
        artist = (m.get("artist") or "").strip()
        if not artist or artist in ("???", "Unknown Artist"):
            continue
        by_artist_idx.setdefault(artist, []).append(i)

    rows = []
    for artist, idx in by_artist_idx.items():
        if not (1 <= len(idx) < 5):
            continue
        normalized = _norm(artist)
        # Skip if this name is already a ≥5-track reference artist (defensive —
        # a <5 artist cannot be in the reference bundle, but names can collide).
        if not normalized or normalized in _artist_id_map:
            continue
        sample = _corpus.metadata[idx[0]]
        sources = sorted({_corpus.metadata[i].get("source") for i in idx})
        rows.append({
            "artist": artist,
            "n_tracks": len(idx),
            "track_indices": idx,
            "sources": sources,
            "sample_track": sample.get("name"),
        })

    rows.sort(key=lambda r: r["artist"].casefold())
    _artist_lowconf_rows = rows
    for i, row in enumerate(rows):
        _artist_lowconf_by_norm.setdefault(_norm(row["artist"]), i)
    print(f"[atlas] loaded low-confidence artist pool: {len(rows)} artists (1-4 tracks)")


def _load_artist_profiles() -> None:
    """Load the optional artist-profile enrichment layer into ``_artist_profiles``.

    Read-only, keyed by normalized artist name. A missing/broken DB is not an
    error — profile fields just won't appear (the invariant is that enrichment
    never affects clustering or the map). Built offline via
    ``python -m anther_ml.artist_enrichment``."""
    global _artist_profiles
    _artist_profiles = {}
    try:
        from anther_ml.artist_enrichment.store import open_readonly
        store = open_readonly(ARTIST_PROFILES_DB)
        if store is None:
            print(f"[atlas] no artist profiles at {ARTIST_PROFILES_DB} — "
                  "profile fields disabled")
            return
        _artist_profiles = store.load_all()
        store.close()
        print(f"[atlas] loaded artist profiles: {len(_artist_profiles)} enriched")
    except Exception as e:  # never let enrichment break artist mode
        _artist_profiles = {}
        print(f"[atlas] artist profiles unavailable: {e}")


def _calibrate_qq_threshold(corpus, n_pairs: int = 200_000) -> None:
    """Set the query↔query cosine cutoff to the QUERY_LINK_PCTL percentile of
    random corpus-pair cosines (index space) — so an edge means "more similar
    than that fraction of released music," not an arbitrary absolute cosine.

    Delegates the actual draw to anther_ml.calibration (shared with the
    mentor process — see docs/mentor-graph-aware.md) and reuses the same
    200k-pair sample for _score_ceiling_raw (see _display_score) plus the
    close/near-identical band cutoffs. Also persists the result as a sidecar
    JSON next to the corpus bundle so the mentor process doesn't have to
    redo this 200k-pair draw itself."""
    global _qq_threshold, _score_ceiling_raw, _link_thresholds, _merit_link_thresholds
    global _merit_factor_thresholds
    E = corpus.index.embeddings                       # standardized + L2-normalized
    if E.shape[0] < 2:
        return
    _link_thresholds = link_calibration.calibrate_link_thresholds(
        corpus.index, n_pairs=n_pairs, pctl=QUERY_LINK_PCTL)
    _qq_threshold = _link_thresholds.qq_threshold
    _score_ceiling_raw = _link_thresholds.score_ceiling_raw
    try:
        link_calibration.save_calibration(CORPUS_DIR, _link_thresholds)
    except OSError:
        pass  # sidecar is a nice-to-have; mentor falls back to its own draw

    # MERIT-aggregate thresholds are computed once at build time (see
    # merit_index.build_merit_aggregate_index) and persisted alongside the
    # bundle — reload them here rather than recalibrating, so app startup
    # doesn't pay another 200k-pair draw. If the sidecar is missing (e.g. a
    # build step that skipped/failed the calibration write) fall back to
    # calibrating in-process here, same as build time would have, rather
    # than silently leaving every MERIT call site stuck on MERT-space
    # thresholds (which flattens MERIT cosines to near-0 display scores).
    _merit_link_thresholds = corpus.merit_calibration
    if _merit_link_thresholds is None and corpus.merit_index is not None:
        _merit_link_thresholds = link_calibration.calibrate_link_thresholds(
            corpus.merit_index, n_pairs=n_pairs, seed=0)
        try:
            link_calibration.save_calibration(
                CORPUS_DIR, _merit_link_thresholds,
                filename=link_calibration.CALIBRATION_FILENAME_MERIT)
        except OSError:
            pass

    # Per-factor (melody/rhythm/timbre) thresholds — see
    # calibration.py's CALIBRATION_FILENAME_MERIT_FACTORS docstring for why
    # these can't share the aggregate scale above. Same missing-sidecar
    # fallback as MERIT-aggregate thresholds.
    _merit_factor_thresholds = corpus.merit_factor_calibration
    if _merit_factor_thresholds is None and corpus.merit_factors is not None:
        _merit_factor_thresholds = link_calibration.calibrate_factor_link_thresholds(
            corpus.merit_factors, n_pairs=n_pairs, seed=0)
        try:
            link_calibration.save_factor_calibration(
                CORPUS_DIR, _merit_factor_thresholds,
                filename=link_calibration.CALIBRATION_FILENAME_MERIT_FACTORS)
        except OSError:
            pass

def _display_score(raw_cos: float, clip_low: bool = True,
                    thresholds=None) -> float:
    """Map a raw cosine to the 0-100 human-readable similarity score.

    _qq_threshold -> SCORE_FLOOR_DISPLAY, _score_ceiling_raw -> 100, linear
    between. `clip_low=True` (map edges, which structurally can't fall below
    _qq_threshold) floors the result at SCORE_FLOOR_DISPLAY; `clip_low=False`
    (the corpus-wide "show more" search, which can surface pairs that don't
    clear the link threshold) lets the score read honestly below the floor,
    only clipping at 0. Both cases clip at 100 on top.

    `thresholds` overrides the MERT-space `_link_thresholds` global — pass
    `_merit_link_thresholds` for MERIT-aggregate-space scores."""
    return link_calibration.display_score(
        raw_cos, thresholds if thresholds is not None else _link_thresholds,
        clip_low=clip_low)


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


_merit_heads = None
_merit_heads_lock = threading.Lock()


def _heads():
    """Lazily load the 3 MERIT projection heads once. Raises FileNotFoundError
    (propagated to callers, who treat it the same as "no MERIT support on this
    machine" — see _merit_query_vec) if models/merit_heads isn't populated."""
    global _merit_heads
    with _merit_heads_lock:
        if _merit_heads is None:
            _merit_heads = merit_mod.load_heads(merit_mod.DEFAULT_HEADS_DIR)
    return _merit_heads


def _merit_query_vec(backbone) -> np.ndarray | None:
    """Raw 5120-d MERIT backbone -> 384-d concatenated-factor query vector,
    or None if there's no backbone (track cached before MERIT support) or the
    projection heads aren't available locally."""
    if backbone is None:
        return None
    try:
        heads = _heads()
    except FileNotFoundError:
        return None
    return merit_mod.merit_query_vector(backbone, heads)


def _merit_breakdown(agg_a, agg_b) -> dict | None:
    """Per-factor melody/rhythm/timbre cosines from two 384-d MERIT-aggregate
    vectors (each a concat of 3 *unit* factor sub-vectors, in
    ``merit_mod.FACTORS`` order, then globally L2-renormalized by
    ``SongIndex``/``transform_query`` — a uniform 1/sqrt(3) rescale that
    cancels out of a cosine between two same-form vectors). Splitting the
    concat back into its 3 equal segments and taking each segment's own
    cosine recovers the exact same per-factor similarity used at build time,
    independent of that global rescale. Returns None if either vector is
    missing or the corpus doesn't carry a MERIT index (dim can't be split).

    Also returns ``aggregate`` = mean of the 3 factor cosines, which is
    mathematically identical to the whole-vector cosine of ``agg_a``/``agg_b``
    (equal-weight concat of unit vectors) — recomputed here explicitly so the
    breakdown is internally consistent even if a caller passes vectors that
    aren't already unit-norm.
    """
    if agg_a is None or agg_b is None:
        return None
    a = np.asarray(agg_a, dtype=np.float32)
    b = np.asarray(agg_b, dtype=np.float32)
    n = len(merit_mod.FACTORS)
    if a.shape[0] % n != 0 or a.shape != b.shape:
        return None
    d = a.shape[0] // n
    out = {}
    cosines = []
    for i, f in enumerate(merit_mod.FACTORS):
        sa, sb = a[i * d:(i + 1) * d], b[i * d:(i + 1) * d]
        na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
        c = float(np.dot(sa, sb) / (na * nb)) if na > 0 and nb > 0 else 0.0
        out[merit_mod.FACTOR_NAMES[f]] = c
        cosines.append(c)
    out["aggregate"] = sum(cosines) / len(cosines)
    return out


_FACTOR_NAME_TO_CODE = {v: k for k, v in merit_mod.FACTOR_NAMES.items()}

def _compress_merit_score(score: float, floor: float = 35.0) -> float:
    return floor + (100.0 - floor) * score / 100.0

def _breakdown_scores(agg_a, agg_b) -> dict | None:
    """``_merit_breakdown`` mapped through ``_display_score`` -> 0-100 ints
    for the UI's expandable per-factor rows, or None if unavailable.

    Each factor (melody/rhythm/timbre) is scored against its OWN calibrated
    scale (``_merit_factor_thresholds``) rather than the aggregate's
    (``_merit_link_thresholds``) — the two can differ a lot (e.g. timbre's
    raw cosines run far hotter than the aggregate's), so reusing one scale
    for all three either clips a hot factor to 100 constantly or leaves a
    cold factor never reaching the top of the range. Falls back to the
    shared aggregate scale for a factor whose sidecar isn't available yet.
    clip_low=False so a factor score reads honestly even when it's below
    that factor's own link threshold.

    ``aggregate`` is deliberately NOT scored against its own independently
    calibrated scale anymore — it's defined as the plain mean of the three
    factor display scores above. Averaging raw cosines and then calibrating
    that mean separately (the old approach) has no fixed relationship to
    averaging the three *already-calibrated* display numbers: each factor's
    raw-cosine distribution has a different width, so
    display(mean(raw_i)) can land above every display(raw_i) even though
    mean(raw_i) itself always sits between them. That produced a single
    "similarity score" that could read higher than all three melody/rhythm/
    timbre bars underneath it. Defining aggregate as mean-of-displays
    instead guarantees it always falls within [min, max] of the three
    factors shown in the expanded panel."""
    raw = _merit_breakdown(agg_a, agg_b)
    if raw is None:
        return None
    out = {}
    for k in ("melody", "rhythm", "timbre"):
        v = raw[k]
        code = _FACTOR_NAME_TO_CODE.get(k)
        thresholds = (_merit_factor_thresholds or {}).get(code) or _merit_link_thresholds
        score = _display_score(v, clip_low=False, thresholds=thresholds)
        out[k] = _compress_merit_score(score)
    out["aggregate"] = round((out["melody"] + out["rhythm"] + out["timbre"]) / 3.0, 1)
    out["melody"] = round(out["melody"], 1)
    out["rhythm"] = round(out["rhythm"], 1)
    out["timbre"] = round(out["timbre"], 1)
    return out


def _merit_aggregate_score(agg_a, agg_b) -> float | None:
    """The single top-level similarity number for a MERIT-scored pair —
    identical to ``_breakdown_scores(agg_a, agg_b)['aggregate']`` (the mean
    of melody/rhythm/timbre's own calibrated display scores). Every call
    site that shows a bare score next to (or instead of) a breakdown should
    go through this, so a plain "score" can never contradict the expanded
    per-factor bars for the same pair. None if MERIT vectors aren't
    available for either side."""
    bd = _breakdown_scores(agg_a, agg_b)
    return bd["aggregate"] if bd is not None else None


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


def _ensure_cache_columns(con: sqlite3.Connection) -> None:
    """Migrate an older embed_cache (MERT-only) to also carry the raw MERIT
    backbone, so a track cached before the MERIT integration can still be
    upgraded to a merit_vec on next placement without re-downloading audio."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS embed_cache ("
        "track_id TEXT PRIMARY KEY, vec BLOB NOT NULL, dim INTEGER NOT NULL, "
        "name TEXT, artist TEXT, created REAL)"
    )
    cols = {row[1] for row in con.execute("PRAGMA table_info(embed_cache)")}
    if "backbone" not in cols:
        con.execute("ALTER TABLE embed_cache ADD COLUMN backbone BLOB")
    if "backbone_dim" not in cols:
        con.execute("ALTER TABLE embed_cache ADD COLUMN backbone_dim INTEGER")


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


def cached_backbone(track_id: str) -> np.ndarray | None:
    """Raw 5120-d MERIT backbone for a previously-cached track, or None if
    the track isn't cached or was cached before MERIT support (no backbone
    column populated)."""
    cache_path = get_session().embed_cache_path
    if not track_id or not cache_path.exists():
        return None
    try:
        con = sqlite3.connect(str(cache_path))
        try:
            row = con.execute(
                "SELECT backbone FROM embed_cache WHERE track_id = ?", (track_id,)
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def cache_vec(track_id: str, vec, name: str = "", artist: str = "",
              backbone=None) -> None:
    if not track_id:
        return
    v = np.asarray(vec, dtype=np.float32)
    b = None if backbone is None else np.asarray(backbone, dtype=np.float32)
    st = get_session()
    with _cache_lock:
        st.ensure_dirs()
        con = sqlite3.connect(str(st.embed_cache_path))
        try:
            _ensure_cache_columns(con)
            con.execute(
                "INSERT OR REPLACE INTO embed_cache "
                "(track_id, vec, dim, name, artist, created, backbone, backbone_dim) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (track_id, v.tobytes(), int(v.size), name, artist, time.time(),
                 None if b is None else b.tobytes(),
                 None if b is None else int(b.size)),
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


def _place_corpus_track(idx: int, extra: dict | None = None,
                        kind: str = "query") -> dict:
    """Merge a corpus row into the graph as a placed song or context node."""
    corpus = load()
    self_id = corpus.metadata[idx].get("id")
    raw_vec = corpus.embeddings[idx]
    node = {
        "id":         self_id,
        "name":       corpus.metadata[idx].get("name", ""),
        "artist":     corpus.metadata[idx].get("artist", ""),
        "cluster":    _id_to_cluster.get(self_id),
        "kind":       kind,
        "confidence": 1.0,
        "source":     "corpus",
        **(extra or {}),
    }
    # merit_index.embeddings is row-aligned to corpus.metadata (both derive
    # from the same index.json), so idx indexes it directly — no
    # transform_query() needed, unlike the raw-MERT-vec case below.
    merit_qvec = (corpus.merit_index.embeddings[idx]
                  if corpus.merit_index is not None else None)
    return _merge_fragment(node, corpus.index.transform_query(raw_vec), merit_qvec)


def _place_query_vec(track_id: str, name: str, artist: str, raw_vec,
                     source: str, extra: dict | None = None, merit_vec=None) -> dict:
    """Place an out-of-corpus track from an already-computed raw MERT vector.

    ``merit_vec`` is the raw 384-d MERIT-aggregate query vector (concatenated
    factor projections, from ``_merit_query_vec``) if available — passed
    through to ``place()`` so ranking/cluster-neighbor selection uses MERIT
    space when the corpus supports it, and to ``_merge_fragment`` for edge
    scoring.
    """
    corpus = load()
    res = place(corpus, raw_vec, top_k=TOP_K, merit_vec=merit_vec,
                popularity_pct=_popularity_pct or None, popularity_beta=POPULARITY_BETA)
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
    merit_qvec = (corpus.merit_index.transform_query(merit_vec)
                  if merit_vec is not None and corpus.merit_index is not None else None)
    return _merge_fragment(node, corpus.index.transform_query(raw_vec), merit_qvec)


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
        merit_vec = _merit_query_vec(cached_backbone(result.get("id")))
        return _place_query_vec(result.get("id"), result.get("title", ""),
                                result.get("artist", ""), raw, source, extra=extra,
                                merit_vec=merit_vec)

    backbone = None
    if source == "upload":
        path, cleanup = Path(result["path"]), None
        try:
            with _embed_lock:
                model, processor, device = _mert()
                if corpus.merit_index is not None:
                    vec, backbone = embed_query_dual(path, corpus, model, processor, device)
                else:
                    vec = embed_query(path, corpus, model, processor, device)
        finally:
            if cleanup:
                cleanup()
    else:
        # Non-upload sources (deezer, spotify, etc.) use two-tier fallback:
        # Try the provided preview_url, fall back to Deezer name/artist match
        # if the URL is dead. Same logic as playlist_jobs.py worker.
        try:
            vec, backbone, _method = resolve_and_embed({
                "id": result.get("id"),
                "name": result.get("title", ""),
                "artist": result.get("artist", ""),
                "preview_url": result.get("preview_url"),
            })
        except PlacementSkip as e:
            raise ValueError(f"Could not place song: {e.reason}") from e

    cache_vec(result.get("id"), vec, result.get("title", ""), result.get("artist", ""),
              backbone=backbone)
    merit_vec = _merit_query_vec(backbone)
    return _place_query_vec(result.get("id"), result.get("title", ""),
                            result.get("artist", ""), vec, source, extra=extra,
                            merit_vec=merit_vec)


def resolve_and_embed(track: dict) -> tuple[np.ndarray, np.ndarray | None, str]:
    """
    Audio for one out-of-corpus track → raw MERT vector (+ raw MERIT backbone
    when the corpus supports it).

    ``track`` is {"id", "name", "artist", "preview_url"}. Tries the stored
    Spotify preview URL first (many p.scdn.co links are dead — Spotify
    deprecated previews in late 2024), then a Deezer name/artist match.
    Returns (raw_vec, backbone_or_None, method); raises PlacementSkip when no
    audio resolves.
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

    backbone = None
    try:
        with _embed_lock:
            model, processor, device = _mert()
            if corpus.merit_index is not None:
                vec, backbone = embed_query_dual(path, corpus, model, processor, device)
            else:
                vec = embed_query(path, corpus, model, processor, device)
    except Exception as e:
        raise PlacementSkip(f"embed_failed:{e}")
    finally:
        if cleanup:
            cleanup()
    cache_vec(track["id"], vec, track.get("name", ""), track.get("artist", ""),
              backbone=backbone)
    return vec, backbone, method


def place_external_track(track: dict, raw_vec, playlist_pid=None, backbone=None) -> dict:
    """Merge one embedded out-of-corpus track into the graph (worker entry)."""
    extra = {"playlist_pid": playlist_pid} if playlist_pid is not None else None
    source = "deezer" if str(track["id"]).startswith("deezer:") else "mpd"
    merit_vec = _merit_query_vec(backbone)
    return _place_query_vec(track["id"], track.get("name", ""),
                            track.get("artist", ""), raw_vec, source, extra=extra,
                            merit_vec=merit_vec)


def _seed_vec(seed_id: str, *, use_merit: bool = False) -> np.ndarray | None:
    """Seed vector in the active recommendation space, or ``None``.

    Corpus rows are already aligned to the MERIT sidecar. External songs need
    their cached raw MERIT backbone projected through the local factor heads;
    a legacy MERT-only cache entry is skipped rather than mixed into a MERIT
    recommendation.
    """
    idx = _id_to_idx.get(seed_id)
    if idx is not None:
        corpus = load()
        return (corpus.merit_index.embeddings[idx]
                if use_merit else corpus.embeddings[idx])
    if use_merit:
        return _merit_query_vec(cached_backbone(seed_id))
    return cached_vec(seed_id)


def recommend(seed_ids: list, top_k: int = 20, method: str = "topk",
              splice: bool = True) -> dict:
    """
    Multi-song recommendation (use-case 2): given the ids of several placed seed
    songs, return corpus tracks similar to the *set* as a whole.

    Every seed must already have a vector available — in-corpus rows and any
    song previously searched/placed (embed-cached) resolve instantly; a seed
    with no cached vector is reported in ``skipped`` rather than silently
    dropped. Seeds are excluded from their own results.

    ``method`` is passed through to ``recommend_from_seeds`` ("topk" default;
    "centroid" remains available for an explicitly shared-center query). When ``splice`` is true the
    returned tracks are also merged into the shared graph as corpus context
    nodes, so the list and the map stay in sync without changing the seed set.

    Returns {"results": [...], "n_seeds": int, "skipped": [ids], "method": str}.
    """
    ids = [s for s in (seed_ids or []) if s]
    if not ids:
        raise ValueError("recommend needs at least one seed id")
    corpus = load()
    use_merit = corpus.merit_index is not None

    vecs, used, skipped = [], [], []
    for sid in ids:
        v = _seed_vec(sid, use_merit=use_merit)
        if v is None:
            skipped.append(sid)
        else:
            vecs.append(v)
            used.append(sid)
    if not vecs:
        raise ValueError("no seed ids resolved to a vector (none cached yet)")

    # Exclude the whole current map, not only the submitted seeds. A result
    # should always be a fresh context node rather than duplicate a song the
    # user already placed (including nodes added by another UI action).
    st = get_session()
    with st.lock:
        placed_ids = set(st.graph["nodes"])
    results = recommend_from_seeds(
        corpus, vecs, top_k=top_k, exclude_ids=set(used) | placed_ids, method=method,
        popularity_pct=_popularity_pct or None, popularity_beta=POPULARITY_BETA,
        index=corpus.merit_index if use_merit else corpus.index,
    )
    # Recommendations aren't guaranteed map edges (seeds' centroid/topk score is
    # a different quantity than a pairwise qq cosine) so use the same open
    # floor as the "show more" search — a weak recommendation can honestly
    # read below 55 rather than being floored to look stronger than it is.
    for r in results:
        r["score"] = round(_display_score(
            float(r["score"]), clip_low=False,
            thresholds=_merit_link_thresholds if use_merit else _link_thresholds,
        ), 1)
        # Carries through the API and graph reload so the renderer can give
        # recommendation relationships their distinct hover treatment.
        r["recommended"] = True

    if splice:
        for r in results:
            idx = _id_to_idx.get(r.get("id"))
            if idx is not None:
                # Recommendations are graph context, not newly placed seeds.
                # Keeping this kind aligned with the frontend means reloading
                # the graph cannot silently change the next recommendation.
                _place_corpus_track(idx, extra={"recommended": True}, kind="corpus")

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


def _merge_fragment(node: dict, qvec=None, merit_qvec=None) -> dict:
    """Upsert the query node into the graph.

    Placed songs are *not* fanned out to their nearest corpus neighbors
    (that flooded the map with grey context nodes and made it too dense to
    read). Instead this wires the placed song directly to every *other*
    placed song whose cosine clears the corpus-calibrated query-link
    threshold — so similar songs you add pull together in the force sim
    regardless of which Leiden cluster each landed in.

    When ``merit_qvec`` is given and the corpus carries a MERIT-aggregate
    index (``_merit_link_thresholds is not None``), edges are scored in
    MERIT-aggregate space instead of MERT space — this is the "aggregate
    replaces MERT for edges + ranking" path. ``qvec`` (MERT space) is still
    stored on the session so MERT-only consumers (``_positioning``,
    ``_inherit_tags``, ``recommend``) keep working unmodified, but it no
    longer drives which edges get drawn once MERIT is available.
    """
    st = get_session()
    added_nodes, added_links = [], []
    use_merit = merit_qvec is not None and _merit_link_thresholds is not None
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
        if use_merit:
            threshold = _merit_link_thresholds.qq_threshold
            space_vecs = st.merit_vecs
            this_vec = merit_qvec
            thresholds_for_display = _merit_link_thresholds
        else:
            threshold = _qq_threshold
            space_vecs = st.query_vecs
            this_vec = qvec
            thresholds_for_display = None  # → _display_score's MERT default

        if this_vec is not None:
            cands = []
            for other_id, ovec in space_vecs.items():
                if other_id == node["id"]:
                    continue
                # never link to an id that isn't a live node — the vec space can
                # briefly outlive a removed node; a link to it would be dangling
                # (endpoint absent) and freeze the frontend force sim.
                if other_id not in st.graph["nodes"]:
                    continue
                score = float(np.dot(this_vec, ovec))
                if score < threshold:
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
                #
                # In MERIT space, use the same mean-of-3-factor-displays as
                # _breakdown_scores (via _merit_aggregate_score) rather than
                # _display_score on the raw aggregate cosine directly — those
                # two used to diverge (independently-calibrated aggregate
                # scale vs. per-factor scales), which could persist an edge
                # score higher than every melody/rhythm/timbre bar the click
                # panel shows for the same pair. clip_low doesn't apply here
                # since the mean-of-displays isn't floored/ceilinged again.
                if use_merit:
                    display = _merit_aggregate_score(this_vec, space_vecs[other_id])
                    if display is None:
                        display = _display_score(
                            score, clip_low=True, thresholds=thresholds_for_display)
                else:
                    display = _display_score(
                        score, clip_low=True, thresholds=thresholds_for_display)
                link = {"source": node["id"], "target": other_id,
                        "value": round(score, 3),
                        "score": round(display, 1),
                        "kind": "qq"}
                st.graph["links"].append(link)
                st.link_keys.add((node["id"], other_id))
                added_links.append(link)

        if qvec is not None:
            st.query_vecs[node["id"]] = qvec
        if merit_qvec is not None:
            st.merit_vecs[node["id"]] = merit_qvec

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


def _merit_vec_for(song_id: str, corpus=None) -> np.ndarray | None:
    """The 384-d MERIT-aggregate index-space vector for a placed song — a
    corpus track's own row (``corpus.merit_index.embeddings[idx]``) or a
    query node's cached session vector (``st.merit_vecs``), whichever
    applies. None if the corpus carries no MERIT index, or the vector was
    never computed for this node (cached before MERIT support, or the
    projection heads weren't available locally when it was placed)."""
    corpus = corpus if corpus is not None else load()
    if corpus.merit_index is None:
        return None
    idx = _id_to_idx.get(song_id)
    if idx is not None:
        return corpus.merit_index.embeddings[idx]
    return get_session().merit_vecs.get(song_id)


def _map_neighbors(song_id: str, corpus=None) -> list:
    """Songs actually connected to ``song_id`` by a drawn map edge (qq-link),
    using each edge's persisted display score. This is what the click panel
    shows by default — cheap (no corpus search, just an in-memory link scan)
    and guaranteed to match the lines drawn on screen, unlike a fresh
    nearest-neighbor search which can rank differently than what's linked.

    Each row also carries ``breakdown`` — the melody/rhythm/timbre/aggregate
    MERIT scores between ``song_id`` and that neighbor (see
    ``_breakdown_scores``), or None if MERIT vectors aren't available for
    this pair (older bundle, or one side was cached pre-MERIT). When a
    breakdown IS available, ``score`` is set to ``breakdown["aggregate"]``
    (mean of the three factor display scores) rather than the edge's own
    persisted ``score`` — those used to diverge slightly since the edge was
    scored on an independently-calibrated aggregate scale, which could show
    a top-level number higher than every individual factor underneath it.
    Falls back to the persisted edge score (or a raw-value recompute for a
    pre-migration link) only when no breakdown can be computed, so the
    click panel never has to show a bare score with no bars to back it."""
    self_merit_vec = _merit_vec_for(song_id, corpus)
    st = get_session()
    with st.lock:
        rows = []
        for l in st.graph["links"]:
            other = (l["target"] if l["source"] == song_id
                     else l["source"] if l["target"] == song_id else None)
            if other is None:
                continue
            on = st.graph["nodes"].get(other, {})
            breakdown = _breakdown_scores(self_merit_vec, _merit_vec_for(other, corpus))
            if breakdown is not None:
                score = breakdown["aggregate"]
            else:
                score = l.get("score")
                if score is None and l.get("value") is not None:   # pre-migration edge
                    score = round(_display_score(l["value"], clip_low=True), 1)
            rows.append({
                "id":       other,
                "name":     on.get("name", ""),
                "artist":   on.get("artist", ""),
                "score":    score,
                "breakdown": breakdown,
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

    Both ``map_neighbors`` and (when the corpus carries a MERIT-aggregate
    index) ``similar`` rank/score in MERIT-aggregate space — the same signal
    driving the map's edges (see ``_merge_fragment``) — and each row carries
    a ``breakdown`` (melody/rhythm/timbre/aggregate, see ``_breakdown_scores``)
    for the panel's expandable per-factor view. Falls back to MERT-space
    ranking with no breakdown on bundles without a MERIT index.
    """
    corpus = load()
    idx = _id_to_idx.get(song_id)
    has_merit = corpus.merit_index is not None

    st = get_session()
    with st.lock:
        gnode = st.graph["nodes"].get(song_id)
        gnode = dict(gnode) if gnode is not None else None
        node_ids = set(st.graph["nodes"])
        qvec = st.query_vecs.get(song_id)

    if idx is None and gnode is None:
        return None

    self_merit_vec = _merit_vec_for(song_id, corpus)
    map_neighbors = _map_neighbors(song_id, corpus)
    seen_ids = {song_id} | {r["id"] for r in map_neighbors}

    if idx is not None:                                 # corpus track
        m = corpus.metadata[idx]
        track_tags = corpus.track_tags
        tags = track_tags[idx].get("tags", []) if track_tags is not None else []
        similar = []
        if expand:
            if has_merit and self_merit_vec is not None:
                rows = corpus.merit_index.query(self_merit_vec, top_k=top_n + 1 + len(seen_ids))
            else:
                rows = corpus.index.query(corpus.embeddings[idx], top_k=top_n + 1 + len(seen_ids))
            rows = [r for r in rows if r.get("id") not in seen_ids][:top_n]
            similar = []
            for r in rows:
                breakdown = _breakdown_scores(
                    self_merit_vec, _merit_vec_for(r.get("id"), corpus)) if has_merit else None
                # breakdown["aggregate"] (mean of the 3 calibrated factor scores)
                # is the source of truth for the top-level score whenever a
                # breakdown is available, so it can't read higher than every
                # individual factor underneath it — see _breakdown_scores.
                score = breakdown["aggregate"] if breakdown is not None else round(
                    _display_score(float(r.get("score", 0.0)), clip_low=False,
                                    thresholds=_merit_link_thresholds if has_merit else None), 1)
                similar.append({
                    "id":       r.get("id"),
                    "name":     r.get("name", ""),
                    "artist":   r.get("artist", ""),
                    "score":    score,
                    "breakdown": breakdown,
                    "on_graph": r.get("id") in node_ids,
                })
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
        if has_merit and self_merit_vec is not None:
            sims = corpus.merit_index.embeddings @ self_merit_vec
            order = np.argsort(sims)[::-1]
            for i in order:
                i = int(i)
                nid = corpus.metadata[i].get("id")
                if nid in seen_ids:
                    continue
                breakdown = _breakdown_scores(self_merit_vec, corpus.merit_index.embeddings[i])
                # breakdown["aggregate"] is the source of truth here too (see
                # the corpus-track branch above) — it can't read higher than
                # every individual factor underneath it.
                score = breakdown["aggregate"] if breakdown is not None else round(
                    _display_score(float(sims[i]), clip_low=False,
                                    thresholds=_merit_link_thresholds), 1)
                similar.append({
                    "id":       nid,
                    "name":     corpus.metadata[i].get("name", ""),
                    "artist":   corpus.metadata[i].get("artist", ""),
                    "score":    score,
                    "breakdown": breakdown,
                    "on_graph": nid in node_ids,
                })
                if len(similar) >= top_n:
                    break
        elif qvec is not None:
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
                    "breakdown": None,
                    "on_graph": nid in node_ids,
                })
                if len(similar) >= top_n:
                    break
        # pre-session node with no cached vector in either space: no
        # corpus-wide search possible — map_neighbors (already computed
        # above) is all we have.

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

def artist_status() -> dict:
    return {"enabled": ARTIST_MODE_ENABLED, "available": _artist_ready,
            "error": _artist_error or None}


def _require_artist_ready() -> None:
    if not ARTIST_MODE_ENABLED:
        raise RuntimeError("artist mode is disabled")
    if not _artist_ready:
        raise RuntimeError(_artist_error or "artist clustering is unavailable")


def _ensure_artist_tables(st: "_SessionState") -> None:
    st.ensure_dirs()
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        _ensure_cache_columns(con)
        con.execute(
            "CREATE TABLE IF NOT EXISTS artist_profiles ("
            "profile_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "name_norm TEXT NOT NULL UNIQUE, created REAL NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS artist_track_assignments ("
            "track_id TEXT PRIMARY KEY, artist_id TEXT NOT NULL, created REAL NOT NULL)"
        )
        con.commit()
    finally:
        con.close()


def _session_artist_rows(st: "_SessionState") -> list[dict]:
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT p.profile_id, p.name, COUNT(a.track_id) AS upload_count "
            "FROM artist_profiles p LEFT JOIN artist_track_assignments a "
            "ON a.artist_id = p.profile_id GROUP BY p.profile_id, p.name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def _artist_upload_count(st: "_SessionState", artist_id: str) -> int:
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM artist_track_assignments WHERE artist_id = ?",
            (artist_id,),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def _artist_upload_counts(st: "_SessionState") -> dict[str, int]:
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        rows = con.execute(
            "SELECT artist_id, COUNT(*) FROM artist_track_assignments GROUP BY artist_id"
        ).fetchall()
        return {str(artist_id): int(count) for artist_id, count in rows}
    finally:
        con.close()


def _parse_corpus_artist_id(artist_id: str) -> int | None:
    if not str(artist_id).startswith("corpus:"):
        return None
    try:
        idx = int(str(artist_id).split(":", 1)[1])
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < len(_artist_meta) else None


def _parse_lowconf_artist_id(artist_id: str) -> int | None:
    """Index into _artist_lowconf_rows for a "lowconf:<idx>" id, else None."""
    if not str(artist_id).startswith("lowconf:"):
        return None
    try:
        idx = int(str(artist_id).split(":", 1)[1])
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < len(_artist_lowconf_rows) else None


def _lowconf_session_vectors(st: "_SessionState", artist_id: str) -> list[np.ndarray]:
    """Raw MERT vectors for any supplemented preview tracks assigned to a
    lowconf artist (empty until the easter-egg supplement runs)."""
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        rows = con.execute(
            "SELECT e.vec FROM artist_track_assignments a "
            "JOIN embed_cache e ON e.track_id = a.track_id WHERE a.artist_id = ?",
            (artist_id,),
        ).fetchall()
    finally:
        con.close()
    return [np.frombuffer(r[0], dtype=np.float32) for r in rows]


def _lowconf_raw_mean(st: "_SessionState", artist_id: str) -> tuple[np.ndarray, int] | None:
    """Combined raw mean embedding (base corpus tracks + supplemented previews)
    for a lowconf artist, and the total track count. None if unresolvable."""
    idx = _parse_lowconf_artist_id(artist_id)
    if idx is None or _corpus is None:
        return None
    row = _artist_lowconf_rows[idx]
    base = _corpus.embeddings[row["track_indices"]]
    vectors = [np.asarray(v, dtype=np.float32) for v in base]
    for v in _lowconf_session_vectors(st, artist_id):
        if v.shape == vectors[0].shape:
            vectors.append(v)
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0).astype(np.float32), len(vectors)


def _artist_raw_profile_vector(st: "_SessionState", artist_id: str) -> np.ndarray | None:
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        rows = con.execute(
            "SELECT e.vec FROM artist_track_assignments a "
            "JOIN embed_cache e ON e.track_id = a.track_id WHERE a.artist_id = ?",
            (artist_id,),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return None
    vectors = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    if any(v.shape != vectors[0].shape for v in vectors):
        return None
    return np.mean(np.stack(vectors), axis=0).astype(np.float32)


def _transform_raw_to_space(raw: np.ndarray) -> np.ndarray | None:
    """Put a raw MERT mean vector through the Leiden scaler/PCA and L2-normalize,
    matching the frozen clustering space. Shared by session and lowconf artists."""
    x = np.asarray(raw, dtype=np.float32).reshape(1, -1)
    scaler, pca = _artist_leiden.get("scaler"), _artist_leiden.get("pca")
    if scaler is not None:
        x = scaler.transform(x)
    if pca is not None:
        x = pca.transform(x)
    vec = np.asarray(x[0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else None


def _artist_vector_for_id(st: "_SessionState", artist_id: str) -> np.ndarray | None:
    idx = _parse_corpus_artist_id(artist_id)
    if idx is not None:
        return _artist_space[idx]
    lc = _lowconf_raw_mean(st, artist_id)
    if lc is not None:
        return _transform_raw_to_space(lc[0])
    if not str(artist_id).startswith("session:"):
        return None
    raw = _artist_raw_profile_vector(st, artist_id)
    if raw is None:
        return None
    return _transform_raw_to_space(raw)


def _session_profile(st: "_SessionState", artist_id: str) -> dict | None:
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT profile_id, name FROM artist_profiles WHERE profile_id = ?",
            (artist_id,),
        ).fetchone()
    finally:
        con.close()
    return dict(row) if row else None


def _with_low_confidence_flag(node: dict | None) -> dict | None:
    """Flag an artist placed from fewer than 5 tracks as low-confidence.

    Computed at read time from ``track_count`` — never persisted to the frozen
    bundle (which is row-count-locked)."""
    if node is not None:
        node["low_confidence"] = int(node.get("track_count", 0)) < 5
    return node


def _cluster_id_for_raw(raw: np.ndarray) -> int:
    """kNN-assign a raw MERT mean into the frozen artist clusters."""
    cluster_id, _confidence = assign_cluster_knn(
        raw, _artist_leiden["clustering_space"], _artist_labels,
        scaler=_artist_leiden.get("scaler"), pca=_artist_leiden.get("pca"),
        k=15, metric=_artist_leiden.get("metric", "cosine"),
    )
    return cluster_id


def _artist_node_for_id(st: "_SessionState", artist_id: str) -> dict | None:
    idx = _parse_corpus_artist_id(artist_id)
    if idx is not None:
        meta = _artist_meta[idx]
        uploads = _artist_upload_count(st, artist_id)
        return _with_low_confidence_flag(
            {"id": artist_id, "name": meta.get("artist", ""),
             "track_count": int(meta.get("n_tracks", 0)), "upload_count": uploads,
             "cluster_id": int(_artist_labels[idx]), "source": "corpus",
             "kind": "artist"})
    lc_idx = _parse_lowconf_artist_id(artist_id)
    if lc_idx is not None:
        combined = _lowconf_raw_mean(st, artist_id)
        if combined is None:
            return None
        raw, track_count = combined
        return _with_low_confidence_flag(
            {"id": artist_id, "name": _artist_lowconf_rows[lc_idx]["artist"],
             "track_count": track_count, "upload_count": _artist_upload_count(st, artist_id),
             "cluster_id": _cluster_id_for_raw(raw), "source": "corpus",
             "kind": "artist"})
    profile = _session_profile(st, artist_id)
    if profile is None:
        return None
    raw = _artist_raw_profile_vector(st, artist_id)
    if raw is None:
        return None
    uploads = _artist_upload_count(st, artist_id)
    return _with_low_confidence_flag(
        {"id": artist_id, "name": profile["name"], "track_count": uploads,
         "upload_count": uploads, "cluster_id": _cluster_id_for_raw(raw),
         "source": "session", "kind": "artist"})


def search_artists(query: str, limit: int = 20) -> dict:
    _require_artist_ready()
    q = (query or "").strip()
    if len(q) < 2:
        return {"results": []}
    qn = _norm(q)
    st = get_session()
    upload_counts = _artist_upload_counts(st)
    results = []
    for i, meta in enumerate(_artist_meta):
        name = meta.get("artist", "")
        nn = _norm(name)
        score = 100 if nn == qn else (96 if nn.startswith(qn) else _ratio(qn, nn) * 100)
        if qn in nn:
            score = max(score, 92)
        if score >= 70:
            results.append({"id": f"corpus:{i}", "name": name,
                            "track_count": int(meta.get("n_tracks", 0)),
                            "upload_count": upload_counts.get(f"corpus:{i}", 0),
                            "cluster_id": int(_artist_labels[i]), "source": "corpus",
                            "low_confidence": False, "score": round(float(score), 1)})
    for i, row in enumerate(_artist_lowconf_rows):
        nn = _norm(row["artist"])
        score = 100 if nn == qn else (96 if nn.startswith(qn) else _ratio(qn, nn) * 100)
        if qn in nn:
            score = max(score, 92)
        if score >= 70:
            node = _artist_node_for_id(st, f"lowconf:{i}")
            if node:
                results.append({**node, "score": round(float(score), 1)})
    for row in _session_artist_rows(st):
        nn = _norm(row["name"])
        score = 100 if nn == qn else (96 if nn.startswith(qn) else _ratio(qn, nn) * 100)
        if qn in nn:
            score = max(score, 92)
        if score >= 70:
            node = _artist_node_for_id(st, row["profile_id"])
            if node:
                results.append({**node, "score": round(float(score), 1)})
    results.sort(key=lambda r: (-r["score"], r["name"].casefold(), r["id"]))
    return {"results": results[:max(1, min(int(limit), 50))]}


def _save_artist_graph(st: "_SessionState") -> None:
    st.dir.mkdir(parents=True, exist_ok=True)
    st.artist_graph_path.write_text(json.dumps({
        "nodes": list(st.artist_graph["nodes"].values()),
        "links": st.artist_graph["links"],
    }))


def _load_artist_graph(st: "_SessionState") -> None:
    st.artist_graph = {"nodes": {}, "links": []}
    st.artist_vectors = {}
    st._artist_loaded = False
    if not _artist_ready or not st.artist_graph_path.is_file():
        st._artist_loaded = _artist_ready
        return
    try:
        data = json.loads(st.artist_graph_path.read_text())
    except (OSError, json.JSONDecodeError):
        st._artist_loaded = True
        return
    for saved in data.get("nodes", []):
        artist_id = saved.get("id", "")
        node = _artist_node_for_id(st, artist_id)
        vec = _artist_vector_for_id(st, artist_id)
        if node is not None and vec is not None:
            st.artist_graph["nodes"][artist_id] = node
            st.artist_vectors[artist_id] = vec
    valid = set(st.artist_graph["nodes"])
    st.artist_graph["links"] = [l for l in data.get("links", [])
                                 if l.get("source") in valid and l.get("target") in valid]
    st._artist_loaded = True


def _ensure_artist_graph_loaded(st: "_SessionState") -> None:
    if st._artist_loaded:
        return
    with st.lock:
        if not st._artist_loaded:
            _load_artist_graph(st)


def get_artist_graph() -> dict:
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    with st.lock:
        return {"ready": True, "nodes": list(st.artist_graph["nodes"].values()),
                "links": list(st.artist_graph["links"])}


def place_artist(artist_id: str) -> dict:
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    artist_id = str(artist_id or "")
    # Low-confidence corpus artists (1-4 tracks) are auto-strengthened on
    # placement: pull a few strict-matched previews so the node lands with a
    # ≥5-track position instead of a shaky one. It's cheap (a handful of 30s
    # embeds) and the frontend surfaces the same "Placing…" wait a Deezer song
    # placement shows. Best-effort and done outside the lock — a fetch/embed
    # hiccup just leaves the artist low-confidence.
    if (_parse_lowconf_artist_id(artist_id) is not None
            and artist_id not in st.artist_graph["nodes"]):
        try:
            _supplement_lowconf_tracks(st, artist_id)
        except Exception as exc:                         # noqa: BLE001 — place as-is
            print(f"[atlas] lowconf auto-supplement failed for {artist_id}: {exc}")
    with st.lock:
        fragment = _place_artist_locked(st, artist_id)
        _save_artist_graph(st)
        return fragment


def _place_artist_locked(st: "_SessionState", artist_id: str) -> dict:
    """Add an artist while ``st.lock`` is held; persistence is the caller's job."""
    if artist_id in st.artist_graph["nodes"]:
        return {"nodes": [], "links": [], "existing": True, "id": artist_id}
    node = _artist_node_for_id(st, artist_id)
    vec = _artist_vector_for_id(st, artist_id)
    if node is None or vec is None:
        raise ValueError("unknown artist id")
    candidates = []
    for other_id, other_vec in st.artist_vectors.items():
        cosine = float(np.dot(vec, other_vec))
        if cosine >= _artist_thresholds.qq_threshold:
            candidates.append((cosine, other_id))
    candidates.sort(reverse=True)
    links = [{"source": artist_id, "target": other_id, "value": cosine,
              "score": round(link_calibration.display_score(
                  cosine, _artist_thresholds, clip_low=True), 1), "kind": "artist"}
             for cosine, other_id in candidates[:ARTIST_MAX_EDGES_PER_NODE]]
    st.artist_graph["nodes"][artist_id] = node
    st.artist_vectors[artist_id] = vec
    st.artist_graph["links"].extend(links)
    return {"nodes": [node], "links": links, "existing": False, "id": artist_id}


def _session_artist_from_songs(st: "_SessionState", artist_name: str,
                               track_ids: list[str]) -> str | None:
    """Create (or reuse) a session artist for a name that isn't in the frozen
    corpus, assigning the given on-map song track ids that have cached
    embeddings. Returns the ``session:`` id, or None if no song had a usable
    embedding. Reuses an existing session profile of the same normalized name so
    repeat builds don't mint duplicate identities.
    """
    normalized = _norm(artist_name)
    if not normalized:
        return None
    _ensure_artist_tables(st)
    # Corpus songs keep their raw vector in the frozen corpus, not embed_cache —
    # copy it in (idempotently) so the session artist's vector machinery, which
    # reads embed_cache, can use it just like a Deezer-placed song. Deezer/upload
    # songs are already cached, so they have no _id_to_idx entry and are skipped.
    if _corpus is not None:
        for tid in track_ids:
            idx = _id_to_idx.get(tid)
            if idx is not None:
                meta = _corpus.metadata[idx]
                cache_vec(tid, _corpus.embeddings[idx],
                          meta.get("name", ""), meta.get("artist", ""))
    # Longer busy timeout: a background embed job (e.g. from the demo import)
    # may be writing embed_cache concurrently — wait for the lock, don't fail.
    con = sqlite3.connect(str(st.embed_cache_path), timeout=30.0)
    try:
        usable = [tid for tid in track_ids
                  if con.execute("SELECT 1 FROM embed_cache WHERE track_id = ?",
                                 (tid,)).fetchone() is not None]
        if not usable:
            return None
        row = con.execute("SELECT profile_id FROM artist_profiles WHERE name_norm = ?",
                          (normalized,)).fetchone()
        if row:
            artist_id = str(row[0])
        else:
            artist_id = f"session:{uuid.uuid4().hex}"
            con.execute("INSERT INTO artist_profiles VALUES (?, ?, ?, ?)",
                        (artist_id, artist_name, normalized, time.time()))
        for tid in usable:
            con.execute(
                "INSERT OR REPLACE INTO artist_track_assignments(track_id, artist_id, created) "
                "VALUES (?, ?, ?)", (tid, artist_id, time.time()))
        con.commit()
    finally:
        con.close()
    return artist_id


def build_artist_graph_from_song_graph(mode: str = "append") -> dict:
    """Seed the persisted artist graph from user-added song nodes.

    Artists in the frozen corpus (≥5 tracks) place directly; low-confidence
    corpus artists (1-4 tracks) and artists not in the corpus at all are placed
    too — the latter as session artists built from their on-map songs. Any
    artist landing with fewer than 5 tracks is auto-supplemented with extra
    strict-matched Deezer previews up to 5 total (on-map songs count toward it),
    so newly built artists match the confidence of the search-placed ones.
    """
    _require_artist_ready()
    if mode not in {"append", "replace"}:
        raise ValueError("artist graph mode must be 'append' or 'replace'")
    st = get_session()
    _ensure_artist_graph_loaded(st)
    _ensure_artist_tables(st)

    # Group on-map songs by normalized artist name (node id kept for session
    # artists built from those songs).
    with st.lock:
        songs_by_artist: dict[str, dict] = {}
        for node in st.graph["nodes"].values():
            if node.get("kind") != "query":
                continue
            nn = _norm(node.get("artist", ""))
            if not nn:
                continue
            entry = songs_by_artist.setdefault(
                nn, {"name": str(node.get("artist", "")).strip(), "ids": []})
            entry["ids"].append(str(node.get("id")))

    # Resolve names → placeable ids and supplement anything under the 5-track
    # target. Network/embed work stays OUTSIDE st.lock (same as place_artist).
    artist_ids: list[str] = []
    skipped: list[str] = []
    for nn in sorted(songs_by_artist):
        name, ids = songs_by_artist[nn]["name"], songs_by_artist[nn]["ids"]
        if nn in _artist_id_map:                         # ≥5-track frozen corpus artist
            artist_ids.append(f"corpus:{_artist_id_map[nn]}")
            continue
        if nn in _artist_lowconf_by_norm:                # 1-4 track frozen corpus artist
            aid = f"lowconf:{_artist_lowconf_by_norm[nn]}"
            try:
                _supplement_lowconf_tracks(st, aid)
            except Exception as exc:                     # noqa: BLE001 — place as-is
                print(f"[atlas] lowconf auto-supplement failed for {aid}: {exc}")
            artist_ids.append(aid)
            continue
        try:                                             # not in corpus → build it
            aid = _session_artist_from_songs(st, name, ids)
        except Exception as exc:                          # noqa: BLE001
            print(f"[atlas] could not build session artist for {name!r}: {exc}")
            aid = None
        if aid is None:
            skipped.append(name)                          # no usable embedding / build failed
            continue
        try:
            _supplement_artist_tracks(st, aid, name, _artist_upload_count(st, aid))
        except Exception as exc:                          # noqa: BLE001 — place as-is
            print(f"[atlas] session artist supplement failed for {aid}: {exc}")
        artist_ids.append(aid)

    if not artist_ids:
        return {"nodes": list(st.artist_graph["nodes"].values()),
                "links": list(st.artist_graph["links"]), "artist_count": 0,
                "skipped_count": len(skipped), "skipped_artists": skipped}

    with st.lock:
        if mode == "replace":
            st.artist_graph = {"nodes": {}, "links": []}
            st.artist_vectors = {}
        # Place each resolved artist independently — a single failure (e.g. a
        # transient DB lock while a background embed job runs) must not turn the
        # whole build into a 500.
        added = 0
        for artist_id in artist_ids:
            try:
                fragment = _place_artist_locked(st, artist_id)
            except Exception as exc:                      # noqa: BLE001
                print(f"[atlas] could not place artist {artist_id}: {exc}")
                continue
            if not fragment.get("existing"):
                added += 1
        _save_artist_graph(st)
        return {"nodes": list(st.artist_graph["nodes"].values()),
                "links": list(st.artist_graph["links"]),
                "artist_count": len(st.artist_graph["nodes"]),
                "added": added,
                "skipped_count": len(skipped), "skipped_artists": skipped}


def _refresh_artist_node(st: "_SessionState", artist_id: str) -> dict:
    node = _artist_node_for_id(st, artist_id)
    vec = _artist_vector_for_id(st, artist_id)
    if node is None or vec is None:
        raise ValueError("artist profile has no usable uploaded tracks")
    st.artist_graph["nodes"][artist_id] = node
    st.artist_vectors[artist_id] = vec
    st.artist_graph["links"] = [l for l in st.artist_graph["links"]
                                if artist_id not in (l["source"], l["target"])]
    candidates = []
    for other_id, other_vec in st.artist_vectors.items():
        if other_id == artist_id:
            continue
        cosine = float(np.dot(vec, other_vec))
        if cosine >= _artist_thresholds.qq_threshold:
            candidates.append((cosine, other_id))
    candidates.sort(reverse=True)
    new_links = [{"source": artist_id, "target": other_id, "value": cosine,
                  "score": round(link_calibration.display_score(
                      cosine, _artist_thresholds, clip_low=True), 1), "kind": "artist"}
                 for cosine, other_id in candidates[:ARTIST_MAX_EDGES_PER_NODE]]
    st.artist_graph["links"].extend(new_links)
    _save_artist_graph(st)
    return {"nodes": [node], "links": new_links, "replace": True, "id": artist_id}


ARTIST_CONFIDENCE_TARGET = 5  # track count at/above which placement is "confident"


def _lowconf_assigned_ids(st: "_SessionState", artist_id: str) -> set[str]:
    """Track ids already supplemented for a lowconf artist."""
    _ensure_artist_tables(st)
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        rows = con.execute(
            "SELECT track_id FROM artist_track_assignments WHERE artist_id = ?",
            (artist_id,),
        ).fetchall()
    finally:
        con.close()
    return {str(r[0]) for r in rows}


def _assign_lowconf_track(st: "_SessionState", track_id: str, artist_id: str) -> None:
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        con.execute(
            "INSERT OR REPLACE INTO artist_track_assignments "
            "(track_id, artist_id, created) VALUES (?, ?, ?)",
            (track_id, artist_id, time.time()),
        )
        con.commit()
    finally:
        con.close()


def _supplement_artist_tracks(st: "_SessionState", artist_id: str,
                              artist_name: str, current: int,
                              target: int = ARTIST_CONFIDENCE_TARGET) -> int:
    """Pull strict artist-name-matched Deezer previews for ``artist_id`` until it
    reaches ``target`` tracks (given it currently has ``current``), embedding
    each and recording the assignment in this session. Returns tracks added.

    Network + 30s-preview embed work happens here, so the caller must NOT hold
    ``st.lock``. Session-only: the extra previews live in this session's
    embed_cache / artist_track_assignments and never touch the frozen bundle.
    Reuses the same embed path as placing a song in song view
    (``resolve_and_embed``). Works for both ``lowconf:`` and ``session:`` ids.
    """
    needed = max(0, int(target) - int(current))
    if needed <= 0:
        return 0
    _ensure_artist_tables(st)
    added = 0
    existing = _lowconf_assigned_ids(st, artist_id)
    qn = _norm(artist_name)
    # Over-fetch: the strict artist-name filter discards most text hits.
    hits = _search_deezer(artist_name, limit=max(needed * 5, 20))
    for hit in hits:
        if added >= needed:
            break
        if _norm(hit.get("artist", "")) != qn:
            continue                                    # strict artist-name match
        track_id = hit.get("id")
        if not track_id or track_id in existing:
            continue
        try:
            resolve_and_embed({"id": track_id, "name": hit.get("title", ""),
                               "artist": hit.get("artist", ""),
                               "preview_url": hit.get("preview_url")})
        except PlacementSkip:
            continue                                    # dead preview → skip
        _assign_lowconf_track(st, track_id, artist_id)
        existing.add(track_id)
        added += 1
    return added


def _supplement_lowconf_tracks(st: "_SessionState", artist_id: str,
                               target: int = ARTIST_CONFIDENCE_TARGET) -> int:
    """Supplement a frozen low-confidence corpus artist (``lowconf:`` id) up to
    ``target`` tracks. Base count is the artist's frozen corpus tracks plus any
    already-supplemented previews. See ``_supplement_artist_tracks``.
    """
    lc_idx = _parse_lowconf_artist_id(artist_id)
    if lc_idx is None:
        return 0
    _ensure_artist_tables(st)
    artist_name = _artist_lowconf_rows[lc_idx]["artist"]
    combined = _lowconf_raw_mean(st, artist_id)
    current = combined[1] if combined else _artist_lowconf_rows[lc_idx]["n_tracks"]
    return _supplement_artist_tracks(st, artist_id, artist_name, current, target)


def supplement_artist_placement(artist_id: str,
                                target: int = ARTIST_CONFIDENCE_TARGET) -> dict:
    """Pull extra strict-matched previews for a low-confidence artist until it
    reaches ``target`` tracks, then re-place the node with higher confidence.

    Backs both the manual "strengthen this placement" easter egg and the
    automatic supplement run when a low-confidence artist is first placed (see
    ``place_artist``). Mechanics live in ``_supplement_lowconf_tracks``.
    """
    _require_artist_ready()
    lc_idx = _parse_lowconf_artist_id(artist_id)
    if lc_idx is None:
        raise ValueError("supplement is only available for low-confidence artists")
    st = get_session()
    _ensure_artist_graph_loaded(st)
    _ensure_artist_tables(st)
    added = _supplement_lowconf_tracks(st, artist_id, target)

    combined = _lowconf_raw_mean(st, artist_id)
    current = combined[1] if combined else _artist_lowconf_rows[lc_idx]["n_tracks"]

    with st.lock:
        fragment = _refresh_artist_node(st, artist_id)
    node = fragment["nodes"][0] if fragment.get("nodes") else None
    return {**fragment, "supplemented": True, "added": added,
            "track_count": node.get("track_count") if node else current,
            "low_confidence": node.get("low_confidence", True) if node else True}


def assign_upload_artist(track_id: str, artist_id: str = "",
                         artist_name: str = "") -> dict | None:
    """Associate a successfully embedded upload with a frozen or private artist."""
    artist_id, artist_name = str(artist_id or "").strip(), str(artist_name or "").strip()
    if artist_id and artist_name:
        raise ValueError("provide artist_id or artist_name, not both")
    if not artist_id and not artist_name:
        return None
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    _ensure_artist_tables(st)
    created_profile = False
    previous_assignment = None
    con = sqlite3.connect(str(st.embed_cache_path))
    try:
        if con.execute("SELECT 1 FROM embed_cache WHERE track_id = ?", (track_id,)).fetchone() is None:
            raise ValueError("uploaded track embedding is unavailable")
        if artist_name:
            if len(artist_name) > 255:
                raise ValueError("artist name must be at most 255 characters")
            normalized = _norm(artist_name)
            if not normalized:
                raise ValueError("artist name is required")
            if normalized in _artist_id_map:
                raise ValueError(
                    f"artist already exists; select corpus:{_artist_id_map[normalized]}")
            row = con.execute("SELECT profile_id FROM artist_profiles WHERE name_norm = ?",
                              (normalized,)).fetchone()
            if row:
                raise ValueError(f"artist already exists; select {row[0]}")
            artist_id = f"session:{uuid.uuid4().hex}"
            con.execute("INSERT INTO artist_profiles VALUES (?, ?, ?, ?)",
                        (artist_id, artist_name, normalized, time.time()))
            created_profile = True
        else:
            idx = _parse_corpus_artist_id(artist_id)
            if idx is None and _session_profile(st, artist_id) is None:
                raise ValueError("unknown artist id")
        previous_assignment = con.execute(
            "SELECT artist_id, created FROM artist_track_assignments WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        con.execute(
            "INSERT OR REPLACE INTO artist_track_assignments(track_id, artist_id, created) "
            "VALUES (?, ?, ?)", (track_id, artist_id, time.time()))
        con.commit()
    finally:
        con.close()

    try:
        with st.lock:
            if artist_id in st.artist_graph["nodes"]:
                return _refresh_artist_node(st, artist_id)
        return place_artist(artist_id)
    except Exception:
        # Keep failed graph/profile updates from leaving an orphan identity.
        con = sqlite3.connect(str(st.embed_cache_path))
        try:
            con.execute("DELETE FROM artist_track_assignments WHERE track_id = ?", (track_id,))
            if previous_assignment is not None:
                con.execute(
                    "INSERT INTO artist_track_assignments(track_id, artist_id, created) "
                    "VALUES (?, ?, ?)",
                    (track_id, previous_assignment[0], previous_assignment[1]),
                )
            if created_profile:
                con.execute("DELETE FROM artist_profiles WHERE profile_id = ?", (artist_id,))
            con.commit()
        finally:
            con.close()
        raise


def validate_upload_artist(artist_id: str = "", artist_name: str = "") -> str:
    """Validate an upload assignment without mutating state; return display name."""
    artist_id, artist_name = str(artist_id or "").strip(), str(artist_name or "").strip()
    if artist_id and artist_name:
        raise ValueError("provide artist_id or artist_name, not both")
    if not artist_id and not artist_name:
        return "personal"
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    if artist_id:
        node = _artist_node_for_id(st, artist_id)
        if node is None:
            raise ValueError("unknown artist id")
        return node["name"]
    if len(artist_name) > 255 or not _norm(artist_name):
        raise ValueError("artist name must be between 1 and 255 characters")
    normalized = _norm(artist_name)
    if normalized in _artist_id_map:
        raise ValueError(f"artist already exists; select corpus:{_artist_id_map[normalized]}")
    for row in _session_artist_rows(st):
        if _norm(row["name"]) == normalized:
            raise ValueError(f"artist already exists; select {row['profile_id']}")
    return artist_name


def artist_detail(artist_id: str) -> dict | None:
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    node = _artist_node_for_id(st, artist_id)
    if node is None:
        return None
    idx = _parse_corpus_artist_id(artist_id)
    lc_idx = _parse_lowconf_artist_id(artist_id)
    if idx is not None:
        meta = _artist_meta[idx]
    elif lc_idx is not None:
        meta = _artist_lowconf_rows[lc_idx]
    else:
        meta = {}
    with st.lock:
        connected_ids = []
        for link in st.artist_graph["links"]:
            if link["source"] == artist_id:
                connected_ids.append((link["target"], link.get("score")))
            elif link["target"] == artist_id:
                connected_ids.append((link["source"], link.get("score")))
        connected = [{**st.artist_graph["nodes"][other_id], "score": score}
                     for other_id, score in connected_ids
                     if other_id in st.artist_graph["nodes"]]
    connected.sort(key=lambda row: -(row.get("score") or 0))
    return {**node, "cluster_label": ARTIST_CLUSTER_LABELS.get(
                node["cluster_id"], f"Cluster {node['cluster_id']}"),
            "sources": meta.get("sources", ["user_upload"]),
            "sample_track": meta.get("sample_track", ""),
            "profile": _artist_profiles.get(_norm(node.get("name", ""))),
            "connected_artists": connected}


def remove_artist(artist_id: str) -> dict | None:
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    with st.lock:
        node = st.artist_graph["nodes"].pop(artist_id, None)
        if node is None:
            return None
        st.artist_vectors.pop(artist_id, None)
        st.artist_graph["links"] = [l for l in st.artist_graph["links"]
                                    if artist_id not in (l["source"], l["target"])]
        _save_artist_graph(st)
        return {"removed": [artist_id], "node": node}


def clear_artist_graph() -> None:
    _require_artist_ready()
    st = get_session()
    _ensure_artist_graph_loaded(st)
    with st.lock:
        st.artist_graph = {"nodes": {}, "links": []}
        st.artist_vectors = {}
        _save_artist_graph(st)


def place_artist_demo() -> dict:
    _require_artist_ready()
    fragments = []
    for name in ARTIST_DEMO_NAMES:
        idx = _artist_id_map.get(_norm(name))
        if idx is None:
            raise ValueError(f"demo artist missing from corpus: {name}")
        fragments.append(place_artist(f"corpus:{idx}"))
    graph = get_artist_graph()
    return {"nodes": graph["nodes"], "links": graph["links"],
            "added": sum(1 for f in fragments if not f.get("existing"))}


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
        st.merit_vecs.clear()
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
        # merit_vecs is the MERIT-space edge-building space; leaving a removed
        # node's vec here makes a later placement draw a qq edge to a node that
        # no longer exists (a dangling link that then freezes the frontend sim).
        st.merit_vecs.pop(node_id, None)

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
    # Recommendation nodes used to be persisted as ``query`` even though the
    # frontend rendered them as corpus context. Normalize old sessions too, so
    # a reload cannot turn an earlier recommendation into a future seed.
    migrated_recommendations = False
    for node in nodes.values():
        if node.get("recommended") and node.get("kind") == "query":
            node["kind"] = "corpus"
            migrated_recommendations = True
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
    if backfilled or migrated_recommendations:
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
    # deezer/upload) keep their saved edges only. Also rebuild the MERIT-space
    # vecs (st.merit_vecs) in parallel when the corpus carries a MERIT index —
    # corpus rows use the row-aligned merit_index.embeddings directly; cached
    # external tracks need their raw backbone re-projected through the heads.
    st.query_vecs = {}
    st.merit_vecs = {}
    has_merit = _corpus.merit_index is not None
    for n in nodes.values():
        if n.get("kind") != "query":
            continue
        idx = _id_to_idx.get(n["id"])
        if idx is not None:
            raw = _corpus.embeddings[idx]
            merit_qvec = _corpus.merit_index.embeddings[idx] if has_merit else None
        else:
            raw = cached_vec(n["id"])
            if raw is None:
                continue
            merit_qvec = None
            if has_merit:
                merit_vec = _merit_query_vec(cached_backbone(n["id"]))
                if merit_vec is not None:
                    merit_qvec = _corpus.merit_index.transform_query(merit_vec)
        st.query_vecs[n["id"]] = _corpus.index.transform_query(raw)
        if merit_qvec is not None:
            st.merit_vecs[n["id"]] = merit_qvec
