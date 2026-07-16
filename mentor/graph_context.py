"""Live on-screen session state for the mentor's graph tools.

The UI is multi-session: each browser has its OWN map on disk under
``ui/session/<graph_session_id>/`` —

  * ``ui/session/<sid>/graph.json``          — the placed nodes / links / groups
  * ``ui/session/<sid>/embed_cache.sqlite``  — raw MERT 1024-d vectors by track id

The mentor runs in a *separate process* from the UI and cannot see the UI's
in-memory atlas, nor its request-scoped session ContextVar. So GraphContext is
told which session to read: the browser's ``graph_session_id`` rides in with
every /chat request (app.py → service.py → MusicMentor.chat → here). One
GraphContext instance serves many browsers by ``for_session(sid)`` re-pointing
between turns. It reads ``graph.json`` fresh (mtime-cached) and pulls each
node's vector straight from that session's embed cache — NOT via
``atlas.cached_vec``, which is bound to the wrong session in this process.

``MentorContext`` is the per-session conversational state: the node the user has
selected in the UI (sent with every /chat request), the last anchor the agent
resolved, and the last tool observations (for follow-ups like "which ones?").
The graph UI is the user's memory — every tool receives this context.

Vectors returned here are *raw* 1024-d MERT; downstream tools L2-normalize via
``index.transform_query`` before scoring.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from difflib import get_close_matches

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(os.path.dirname(HERE), "ui", "session")
DEFAULT_SESSION_ID = "default"


def _norm_text(s):
    return " ".join(str(s or "").strip().lower().split())


def _safe_session_id(session_id):
    """Mirror atlas._safe_session_id: keep only path-safe chars, never allow
    traversal, fall back to the shared 'default' session for empty/None."""
    if not session_id:
        return DEFAULT_SESSION_ID
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(session_id))
    return safe or DEFAULT_SESSION_ID


def _session_paths(session_id):
    sid = _safe_session_id(session_id)
    base = os.path.join(SESSION_DIR, sid)
    return sid, os.path.join(base, "graph.json"), os.path.join(base, "embed_cache.sqlite")


def _sqlite_vec_reader(cache_path):
    """Build a vec_reader(track_id) -> raw np.ndarray|None over one session's
    embed_cache.sqlite. Reads the DB directly so we never depend on atlas's
    request-scoped session ContextVar (which is 'default' in this process)."""
    def read(track_id):
        if not track_id or not os.path.exists(cache_path):
            return None
        try:
            con = sqlite3.connect(cache_path)
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
    return read


class GraphContext:
    """One browser session's on-screen graph, read fresh from disk.

    Point it at a session with ``session_id=`` (production) or an explicit
    ``graph_path=`` + ``vec_reader=`` (tests). ``for_session(sid)`` re-points a
    live instance so the mentor process can serve many browsers.
    """

    def __init__(self, session_id=None, graph_path=None, vec_reader=None):
        if graph_path is not None:
            # explicit path (tests) — vec_reader must be supplied too
            self.session_id = _safe_session_id(session_id)
            self.graph_path = graph_path
            self._vec_reader = vec_reader or (lambda _id: None)
        else:
            self.session_id, self.graph_path, cache_path = _session_paths(session_id)
            self._vec_reader = vec_reader or _sqlite_vec_reader(cache_path)
        self._reset_indices()

    def for_session(self, session_id):
        """Re-point this instance at another browser session (in-place). No-op
        when already on that session. Returns self for chaining."""
        sid = _safe_session_id(session_id)
        if sid == self.session_id:
            return self
        self.session_id, self.graph_path, cache_path = _session_paths(sid)
        self._vec_reader = _sqlite_vec_reader(cache_path)
        self._mtime = None
        self._reset_indices()
        return self

    def _reset_indices(self):
        self._mtime = None
        self._nodes = []            # list of node dicts (as stored in graph.json)
        self._by_id = {}            # id -> node
        self._artist_to_ids = {}    # norm(artist) -> [id, ...]
        self._name_to_id = {}       # norm("artist - name") and norm(name) -> id
        self._vec_cache = {}        # id -> raw vec (or None if uncached)
        self._groups = {}

    # ---- freshness ---------------------------------------------------------
    def _reload_if_stale(self):
        try:
            mtime = os.path.getmtime(self.graph_path)
        except OSError:
            mtime = None
        if mtime == self._mtime and self._mtime is not None:
            return
        self._mtime = mtime
        self._load()

    def _load(self):
        self._nodes = []
        self._by_id = {}
        self._artist_to_ids = {}
        self._name_to_id = {}
        self._vec_cache = {}
        data = None
        try:
            with open(self.graph_path) as f:
                data = json.load(f)
        except Exception:
            data = None
        if not data:
            return
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict):                # tolerate id-keyed dict
            nodes = list(nodes.values())
        for n in nodes:
            nid = n.get("id")
            if not nid:
                continue
            self._nodes.append(n)
            self._by_id[nid] = n
            a = _norm_text(n.get("artist", ""))
            nm = _norm_text(n.get("name", ""))
            if a:
                self._artist_to_ids.setdefault(a, []).append(nid)
            if a and nm:
                self._name_to_id.setdefault(f"{a} - {nm}", nid)
            if nm:
                self._name_to_id.setdefault(nm, nid)
        self._groups = data.get("groups", {}) or {}

    # ---- vectors -----------------------------------------------------------
    def vec_for_id(self, nid):
        """Raw 1024-d vector for a node id, or None if not cached."""
        if nid in self._vec_cache:
            return self._vec_cache[nid]
        v = self._vec_reader(nid)
        if v is not None:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
        self._vec_cache[nid] = v
        return v

    def artist_centroid(self, ids):
        vs = [self.vec_for_id(i) for i in ids]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        c = np.mean(np.stack(vs), axis=0).astype(np.float32)
        return c

    # ---- lookups -----------------------------------------------------------
    @property
    def nodes(self):
        self._reload_if_stale()
        return self._nodes

    @property
    def groups(self):
        self._reload_if_stale()
        return getattr(self, "_groups", {})

    def node(self, nid):
        self._reload_if_stale()
        return self._by_id.get(nid)

    def has_nodes(self):
        self._reload_if_stale()
        return bool(self._nodes)

    def artist_ids(self, artist):
        self._reload_if_stale()
        return list(self._artist_to_ids.get(_norm_text(artist), []))

    def mentions_entity(self, text):
        """True if the text contains an on-map artist or song name. Used by the
        router to recognise a bare entity ("what is Big On Big connected to")
        as a graph question even without map-referential phrasing."""
        self._reload_if_stale()
        q = _norm_text(text)
        if not q:
            return False
        for key in self._artist_to_ids:
            if key and len(key) >= 3 and key in q:
                return True
        for key in self._name_to_id:
            if key and len(key) >= 3 and key in q:
                return True
        return False

    def summary(self, max_artists=40):
        """A compact human-readable census of the map for the router prompt:
        node count, distinct artists, and territory labels. Empty string when
        the map is empty."""
        self._reload_if_stale()
        if not self._nodes:
            return ""
        artists, seen = [], set()
        territories = {}
        for n in self._nodes:
            a = n.get("artist", "")
            if a and _norm_text(a) not in seen:
                seen.add(_norm_text(a))
                artists.append(a)
            label = n.get("cluster_label") or ""
            if label:
                territories[label] = territories.get(label, 0) + 1
        parts = [f"{len(self._nodes)} songs on the map"]
        if artists:
            shown = ", ".join(artists[:max_artists])
            more = "" if len(artists) <= max_artists else f", +{len(artists)-max_artists} more"
            parts.append(f"artists: {shown}{more}")
        if territories:
            terr = ", ".join(sorted(territories, key=lambda t: -territories[t])[:8])
            parts.append(f"territories: {terr}")
        return "; ".join(parts)

    def candidate_vectors(self, exclude_ids=None):
        """(ids, matrix, nodes) for every on-screen node with a cached vector."""
        self._reload_if_stale()
        exclude = set(exclude_ids or [])
        ids, mats, nodes = [], [], []
        for n in self._nodes:
            nid = n["id"]
            if nid in exclude:
                continue
            v = self.vec_for_id(nid)
            if v is None:
                continue
            ids.append(nid)
            mats.append(v)
            nodes.append(n)
        if not mats:
            return [], None, []
        return ids, np.stack(mats).astype(np.float32), nodes

    # ---- resolution --------------------------------------------------------
    def resolve(self, spec):
        """On-screen-first anchor resolution.

        Returns (raw_vec, info) or (None, info) if not found on screen. ``info``
        mirrors the corpus resolver's shape with an on-screen ``type``.
        """
        self._reload_if_stale()
        q = _norm_text(spec)
        if not q or not self._nodes:
            return None, {"input": spec, "type": "onscreen", "resolved": False}

        # exact artist -> centroid of that artist's placed nodes
        if q in self._artist_to_ids:
            ids = self._artist_to_ids[q]
            vec = self.artist_centroid(ids)
            if vec is not None:
                return vec, {
                    "input": spec, "type": "onscreen_artist", "resolved": True,
                    "match": q, "count": len(ids), "scope": "onscreen",
                }

        # exact track ("artist - name" or bare name)
        if q in self._name_to_id:
            nid = self._name_to_id[q]
            vec = self.vec_for_id(nid)
            if vec is not None:
                n = self._by_id[nid]
                return vec, {
                    "input": spec, "type": "onscreen_track", "resolved": True,
                    "match": f"{n.get('artist','')} - {n.get('name','')}".strip(" -"),
                    "id": nid, "scope": "onscreen",
                }

        # substring over "artist - name" keys
        for key, nid in self._name_to_id.items():
            if q in key and " - " in key:
                vec = self.vec_for_id(nid)
                if vec is not None:
                    n = self._by_id[nid]
                    return vec, {
                        "input": spec, "type": "onscreen_track", "resolved": True,
                        "match": f"{n.get('artist','')} - {n.get('name','')}".strip(" -"),
                        "id": nid, "scope": "onscreen", "substring": True,
                    }

        # fuzzy artist, then fuzzy track key
        artist_hit = get_close_matches(q, list(self._artist_to_ids.keys()), n=1, cutoff=0.8)
        if artist_hit:
            ids = self._artist_to_ids[artist_hit[0]]
            vec = self.artist_centroid(ids)
            if vec is not None:
                return vec, {
                    "input": spec, "type": "onscreen_artist", "resolved": True,
                    "match": artist_hit[0], "count": len(ids), "scope": "onscreen",
                    "fuzzy": True,
                }
        name_hit = get_close_matches(q, list(self._name_to_id.keys()), n=1, cutoff=0.82)
        if name_hit:
            nid = self._name_to_id[name_hit[0]]
            vec = self.vec_for_id(nid)
            if vec is not None:
                n = self._by_id[nid]
                return vec, {
                    "input": spec, "type": "onscreen_track", "resolved": True,
                    "match": f"{n.get('artist','')} - {n.get('name','')}".strip(" -"),
                    "id": nid, "scope": "onscreen", "fuzzy": True,
                }

        return None, {"input": spec, "type": "onscreen", "resolved": False}


@dataclass
class MentorContext:
    """Per-session conversational state, one instance per browser session.

    ``selected_node_id`` is refreshed on every /chat request from the UI (the
    node the user has clicked/pinned on the map, or None). The graph itself is
    NOT stored here — it lives on disk and is read via ``GraphContext``, which
    the tool layer owns and shares across sessions.
    """
    selected_node_id: str | None = None
    graph_session_id: str | None = None
    last_anchor: str | None = None
    last_intent: str | None = None
    last_observation: dict | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=8))

    def reset(self):
        self.selected_node_id = None
        # graph_session_id is a property of the browser, not the chat — it is
        # refreshed each turn and deliberately survives a /reset.
        self.last_anchor = None
        self.last_intent = None
        self.last_observation = None
        self.history.clear()

    def record(self, role, content):
        if content:
            self.history.append({"role": role, "content": str(content).strip()})

    def recent_block(self, n=4):
        turns = list(self.history)[-n:]
        lines = [f"{t.get('role','user')}: {t.get('content','').strip()}"
                 for t in turns if t.get("content", "").strip()]
        if not lines:
            return ""
        return "\n\n[Recent conversation context]\n" + "\n".join(lines)


# Backward-friendly alias: service.py keyed sessions by ConversationState
# before the refactor; the concept is the same object.
ConversationState = MentorContext
