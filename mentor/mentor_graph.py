"""Deterministic graph tools over the frozen Anther similarity corpus."""

from __future__ import annotations

import os
import sys
from difflib import get_close_matches

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(HERE), "ui")
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import atlas


class MentorGraphTools:
    def __init__(self, anther):
        self.anther = anther

    def _artist_indices(self, artist):
        self.anther._ensure_lookup()
        q = self.anther._norm_text(artist)
        if not q:
            return [], {"input": artist, "resolved": False, "type": "artist"}
        if q in self.anther._artist_to_indices:
            return list(self.anther._artist_to_indices[q]), {
                "input": artist,
                "resolved": True,
                "type": "artist",
                "match": q,
                "count": len(self.anther._artist_to_indices[q]),
            }
        keys = list(self.anther._artist_to_indices.keys())
        hit = get_close_matches(q, keys, n=1, cutoff=0.72)
        if hit:
            idxs = list(self.anther._artist_to_indices[hit[0]])
            return idxs, {
                "input": artist,
                "resolved": True,
                "type": "artist",
                "match": hit[0],
                "count": len(idxs),
                "fuzzy": True,
            }
        return [], {"input": artist, "resolved": False, "type": "artist"}

    @staticmethod
    def _index_to_result(m, score, idx, rank):
        return {
            "rank": rank,
            "index": int(idx),
            "score": round(float(score), 4),
            "artist": m.get("artist", ""),
            "name": m.get("name", ""),
        }

    def _neighbors_with_indices(self, vec, top_k=8):
        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings @ q
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        out = []
        for rank, idx in enumerate(order, start=1):
            m = self.anther.index.metadata[int(idx)]
            out.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "score": round(float(sims[idx]), 4),
                    "artist": m.get("artist", ""),
                    "name": m.get("name", ""),
                }
            )
        return out

    def resolve(self, name, context=None):
        vec, info = self.anther.resolve_anchor(name, context=context)
        return {
            "tool": "resolve",
            "ok": bool(info.get("resolved")),
            "resolved": info,
            "vector_dim": int(vec.shape[0]) if vec is not None else 0,
        }

    def sounds_like(self, anchor, top_k=8, context=None):
        vec, info = self.anther.resolve_anchor(anchor, context=context)
        if vec is None:
            return {"tool": "sounds_like", "ok": False, "error": "anchor_not_resolved", "resolved": info}
        placed = self.anther.sounds_like_from_vec(vec, top_k=top_k)
        return {
            "tool": "sounds_like",
            "ok": True,
            "resolved": info,
            "neighbors": placed.get("neighbors", [])[:top_k],
            "cluster": placed.get("cluster", {}),
            "tags": placed.get("tags", []),
        }

    def compare(self, anchor_a, anchor_b, top_k=8, context=None):
        (vec_a, info_a), (vec_b, info_b) = self.anther.resolve_two(anchor_a, anchor_b, context=context)
        if vec_a is None or vec_b is None:
            return {
                "tool": "compare",
                "ok": False,
                "error": "anchor_not_resolved",
                "anchor_a": info_a,
                "anchor_b": info_b,
            }

        qa = self.anther.index.transform_query(vec_a)
        qb = self.anther.index.transform_query(vec_b)
        score = float(np.dot(qa, qb))
        left = self._neighbors_with_indices(vec_a, top_k=top_k)
        right = self._neighbors_with_indices(vec_b, top_k=top_k)
        return {
            "tool": "compare",
            "ok": True,
            "anchor_a": info_a,
            "anchor_b": info_b,
            "similarity": round(score, 4),
            "left_neighbors": left,
            "right_neighbors": right,
            "left_cluster": self.anther.cluster_of(vec_a),
            "right_cluster": self.anther.cluster_of(vec_b),
        }

    def bridge(self, anchor_a, anchor_b, k=6, context=None):
        (vec_a, info_a), (vec_b, info_b) = self.anther.resolve_two(anchor_a, anchor_b, context=context)
        if vec_a is None or vec_b is None:
            return {
                "tool": "bridge",
                "ok": False,
                "error": "anchor_not_resolved",
                "anchor_a": info_a,
                "anchor_b": info_b,
            }

        na = vec_a / max(1e-8, np.linalg.norm(vec_a))
        nb = vec_b / max(1e-8, np.linalg.norm(vec_b))
        mid = (na + nb) / 2.0
        mid = mid / max(1e-8, np.linalg.norm(mid))

        neighbors = self._neighbors_with_indices(mid, top_k=k)
        enriched = []
        for n in neighbors:
            idx = n["index"]
            qv = self.anther._vector_for_index(idx)
            sa = float(np.dot(qv, na) / max(1e-8, np.linalg.norm(qv)))
            sb = float(np.dot(qv, nb) / max(1e-8, np.linalg.norm(qv)))
            side = "mid"
            if sa > sb + 0.02:
                side = "a"
            elif sb > sa + 0.02:
                side = "b"
            n2 = dict(n)
            n2["lean"] = side
            if self.anther.labels is not None:
                n2["cluster_id"] = int(self.anther.labels[idx])
            enriched.append(n2)

        return {
            "tool": "bridge",
            "ok": True,
            "anchor_a": info_a,
            "anchor_b": info_b,
            "neighbors": enriched,
            "midpoint_cluster": self.anther.cluster_of(mid),
        }

    def crossover(self, anchor, k=6, context=None):
        vec, info = self.anther.resolve_anchor(anchor, context=context)
        if vec is None:
            return {"tool": "crossover", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        base = self.anther.cluster_of(vec)
        base_id = base.get("cluster_id")
        nbr = self._neighbors_with_indices(vec, top_k=250)
        out = []
        for n in nbr:
            idx = n["index"]
            if self.anther.labels is None:
                continue
            cid = int(self.anther.labels[idx])
            if cid == base_id:
                continue
            n2 = dict(n)
            n2["cluster_id"] = cid
            profile = self.anther.profiles.get(cid, {})
            ex = profile.get("exemplars", [])[:2]
            n2["cluster_label"] = profile.get("label_final") or ", ".join(
                f"{e.get('artist','')} - {e.get('name','')}" for e in ex
            )
            out.append(n2)
            if len(out) >= k:
                break

        return {
            "tool": "crossover",
            "ok": True,
            "resolved": info,
            "anchor_cluster": base,
            "neighbors": out,
        }

    def tagmates(self, anchor, k=8, context=None):
        vec, info = self.anther.resolve_anchor(anchor, context=context)
        if vec is None:
            return {"tool": "tagmates", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        anchor_tags = [t["genre"] for t in self.anther._tags_for(vec, top_k=4, knn=12)]
        tag_set = set(anchor_tags)
        if not tag_set:
            return {
                "tool": "tagmates",
                "ok": True,
                "resolved": info,
                "anchor_tags": [],
                "neighbors": [],
            }

        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings @ q
        order = np.argsort(sims)[::-1][:800]
        out = []
        for idx in order:
            idx = int(idx)
            tags = self.anther.track_tags[idx].get("tags", [])
            genres = {t.get("genre") for t in tags if t.get("genre")}
            shared = sorted(genres & tag_set)
            if not shared:
                continue
            m = self.anther.index.metadata[idx]
            out.append(
                {
                    "index": idx,
                    "artist": m.get("artist", ""),
                    "name": m.get("name", ""),
                    "score": round(float(sims[idx]), 4),
                    "shared_tags": shared,
                }
            )
            if len(out) >= k:
                break

        return {
            "tool": "tagmates",
            "ok": True,
            "resolved": info,
            "anchor_tags": anchor_tags,
            "neighbors": out,
        }

    def artist_tracks(self, anchor, artist, top_k=8, context=None):
        vec, info = self.anther.resolve_anchor(anchor, context=context)
        if vec is None:
            return {"tool": "artist_tracks", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        idxs, artist_info = self._artist_indices(artist)
        if not idxs:
            return {
                "tool": "artist_tracks",
                "ok": False,
                "error": "artist_not_resolved",
                "resolved": info,
                "artist": artist_info,
            }

        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings[idxs] @ q
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        out = []
        for rank, rel_idx in enumerate(order, start=1):
            idx = int(idxs[int(rel_idx)])
            m = self.anther.index.metadata[idx]
            out.append(self._index_to_result(m, sims[int(rel_idx)], idx, rank))

        return {
            "tool": "artist_tracks",
            "ok": True,
            "resolved": info,
            "artist": artist_info,
            "neighbors": out,
        }

    def search(self, query, limit=25):
        return atlas.search(query, limit=limit)

    def search_playlists(self, query, limit=20):
        return atlas.search_playlists(query, limit=limit)

    def search_albums(self, query, limit=20):
        return atlas.search_albums(query, limit=limit)

    def place_song(self, result):
        return atlas.place_song(result)

    def place_playlist(self, pid):
        return atlas.place_playlist(pid)

    def place_album(self, album_id):
        return atlas.place_album(album_id)

    def recommend(self, seed_ids, top_k=20, method="centroid"):
        return atlas.recommend(seed_ids, top_k=top_k, method=method)

    def song_detail(self, song_id, top_n=10):
        return atlas.song_detail(song_id, top_n=top_n)

    def get_graph(self):
        return atlas.get_graph()

    def clear_graph(self):
        atlas.clear_graph()
        return {"tool": "clear_graph", "ok": True, "status": "cleared"}

    def remove_node(self, node_id):
        result = atlas.remove_node(node_id)
        if result is None:
            return {"tool": "remove_node", "ok": False, "error": "unknown_node"}
        result = dict(result)
        result["tool"] = "remove_node"
        result["ok"] = True
        return result


TOOLS = {
    "search": {
        "callable": MentorGraphTools.search,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "search_playlists": {
        "callable": MentorGraphTools.search_playlists,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "search_albums": {
        "callable": MentorGraphTools.search_albums,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "place_song": {
        "callable": MentorGraphTools.place_song,
        "schema": {
            "type": "object",
            "properties": {"result": {"type": "object"}},
            "required": ["result"],
        },
    },
    "place_playlist": {
        "callable": MentorGraphTools.place_playlist,
        "schema": {
            "type": "object",
            "properties": {"pid": {}},
            "required": ["pid"],
        },
    },
    "place_album": {
        "callable": MentorGraphTools.place_album,
        "schema": {
            "type": "object",
            "properties": {"album_id": {}},
            "required": ["album_id"],
        },
    },
    "recommend": {
        "callable": MentorGraphTools.recommend,
        "schema": {
            "type": "object",
            "properties": {
                "seed_ids": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "method": {"type": "string"},
            },
            "required": ["seed_ids"],
        },
    },
    "song_detail": {
        "callable": MentorGraphTools.song_detail,
        "schema": {
            "type": "object",
            "properties": {
                "song_id": {"type": "string"},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["song_id"],
        },
    },
    "graph": {
        "callable": MentorGraphTools.get_graph,
        "schema": {"type": "object", "properties": {}},
    },
    "clear_graph": {
        "callable": MentorGraphTools.clear_graph,
        "schema": {"type": "object", "properties": {}},
    },
    "remove_node": {
        "callable": MentorGraphTools.remove_node,
        "schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    "resolve": {
        "callable": MentorGraphTools.resolve,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "sounds_like": {
        "callable": MentorGraphTools.sounds_like,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
    "compare": {
        "callable": MentorGraphTools.compare,
        "schema": {
            "type": "object",
            "properties": {
                "anchor_a": {"type": "string"},
                "anchor_b": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor_a", "anchor_b"],
        },
    },
    "bridge": {
        "callable": MentorGraphTools.bridge,
        "schema": {
            "type": "object",
            "properties": {
                "anchor_a": {"type": "string"},
                "anchor_b": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor_a", "anchor_b"],
        },
    },
    "crossover": {
        "callable": MentorGraphTools.crossover,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
    "artist_tracks": {
        "callable": MentorGraphTools.artist_tracks,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "artist": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor", "artist"],
        },
    },
    "tagmates": {
        "callable": MentorGraphTools.tagmates,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
}
