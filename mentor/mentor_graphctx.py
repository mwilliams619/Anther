"""Live on-screen session state for the mentor's graph tools.

The mentor service runs in a *separate process* from the UI, so ``atlas._graph``
(an in-memory global in the UI process) is empty here. The two process-independent
sources of truth are on disk:

  * ``ui/session/graph.json``          — the placed nodes / links / groups
  * ``ui/session/embed_cache.sqlite``  — raw MERT 1024-d vectors keyed by track id

``GraphContext`` reads ``graph.json`` fresh (mtime-cached so a chat burst doesn't
re-parse it every turn) and reconstructs a query vector per node from the embed
cache. It gives the tool layer an on-screen-first anchor resolver and an on-screen
candidate pool, with the frozen corpus kept as an explicit fallback.

Vectors returned here are *raw* 1024-d MERT (same convention as the corpus
``raw_embeddings`` and ``atlas.cached_vec``); downstream tools L2-normalize via
``index.transform_query`` before scoring.
"""
from __future__ import annotations

import json
import os
import sys
from difflib import get_close_matches

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(HERE), "ui")
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import atlas  # noqa: E402


def _norm_text(s):
    return " ".join(str(s or "").strip().lower().split())


class GraphContext:
    """On-screen session graph, read fresh from disk (mtime-cached)."""

    def __init__(self, graph_path=None, vec_reader=None):
        self.graph_path = graph_path or str(atlas.GRAPH_PATH)
        # vec_reader(track_id) -> raw np.ndarray | None. Defaults to the embed
        # cache; injectable so tests can stub it.
        self._vec_reader = vec_reader or atlas.cached_vec
        self._mtime = None
        self._nodes = []            # list of node dicts (as stored in graph.json)
        self._by_id = {}            # id -> node
        self._artist_to_ids = {}    # norm(artist) -> [id, ...]
        self._name_to_id = {}       # norm("artist - name") and norm(name) -> id
        self._vec_cache = {}        # id -> raw vec (or None if uncached)

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
