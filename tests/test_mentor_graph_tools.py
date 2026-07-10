"""Regression tests for the expanded mentor graph tool suite."""
from types import SimpleNamespace

import numpy as np
import pytest

from mentor import mentor_graph
from mentor.mentor_graph import MentorGraphTools
from mentor.mentor_react import MentorReAct, _fallback_call


class _FakeIndex:
    def __init__(self):
        self.embeddings = np.array(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.15, 0.85],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        self.metadata = [
            {"artist": "XXXTentacion", "name": "Look at Me!"},
            {"artist": "XXXTentacion", "name": "Sad!"},
            {"artist": "Beyonce", "name": "Halo"},
            {"artist": "Beyonce", "name": "Irreplaceable"},
        ]

    def transform_query(self, vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm


class _FakeAnther:
    def __init__(self):
        self.index = _FakeIndex()
        self.labels = np.array([0, 0, 1, 1], dtype=int)
        self.profiles = {
            0: {"label_final": "rap"},
            1: {"label_final": "pop"},
        }
        self.track_tags = [
            {"tags": []},
            {"tags": []},
            {"tags": []},
            {"tags": []},
        ]
        self._artist_to_indices = {}
        self._name_keys = []
        self._ensure_lookup()
        self._anchor_vecs = {
            "Halo by Beyonce": np.array([1.0, 0.0], dtype=np.float32),
            "Take a Step Back": np.array([0.7, 0.7], dtype=np.float32),
            "XXXTentacion": np.array([0.9, 0.1], dtype=np.float32),
            "Beyonce": np.array([1.0, 0.0], dtype=np.float32),
        }

    def _norm_text(self, s):
        return " ".join(str(s or "").strip().lower().split())

    def _ensure_lookup(self):
        self._artist_to_indices = {}
        self._name_keys = []
        for i, m in enumerate(self.index.metadata):
            artist = self._norm_text(m.get("artist", ""))
            name = self._norm_text(m.get("name", ""))
            self._name_keys.append(f"{artist} - {name}".strip(" -"))
            self._artist_to_indices.setdefault(artist, []).append(i)

    def resolve_anchor(self, spec, context=None):
        vec = self._anchor_vecs.get(str(spec))
        if vec is None:
            return None, {"input": spec, "resolved": False}
        return vec, {"input": spec, "resolved": True, "match": str(spec)}

    def resolve_two(self, spec_a, spec_b, context=None):
        return self.resolve_anchor(spec_a, context=context), self.resolve_anchor(spec_b, context=context)

    def cluster_of(self, vec, knn=25):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return {"cluster_id": int(0 if vec[0] >= vec[1] else 1), "label": "stub", "confidence": 1.0}


def test_artist_tracks_filters_artist_and_ranks_by_anchor():
    tools = MentorGraphTools(_FakeAnther())

    result = tools.artist_tracks("Halo by Beyonce", "xxxtentacion", top_k=2)

    assert result["ok"] is True
    assert result["artist"]["resolved"] is True
    assert [n["name"] for n in result["neighbors"]] == ["Look at Me!", "Sad!"]
    assert result["neighbors"][0]["score"] >= result["neighbors"][1]["score"]


def test_compare_returns_similarity_and_both_neighbor_lists():
    tools = MentorGraphTools(_FakeAnther())

    result = tools.compare("Halo by Beyonce", "Take a Step Back", top_k=2)

    assert result["ok"] is True
    assert result["similarity"] == pytest.approx(np.sqrt(2) / 2, abs=1e-4)
    assert result["left_neighbors"]
    assert result["right_neighbors"]
    assert result["anchor_a"]["resolved"] is True
    assert result["anchor_b"]["resolved"] is True


def test_frontend_wrappers_delegate_to_atlas(monkeypatch):
    calls = {}

    def _record(key, value, result):
        calls[key] = value
        return result

    fake_atlas = SimpleNamespace(
        search=lambda query, limit=25: _record("search", (query, limit), {"results": [query, limit]}),
        search_playlists=lambda query, limit=20: _record("search_playlists", (query, limit), {"results": [query, limit]}),
        search_albums=lambda query, limit=20: _record("search_albums", (query, limit), {"results": [query, limit]}),
        place_song=lambda result: _record("place_song", result, {"nodes": [result], "links": []}),
        place_playlist=lambda pid: _record("place_playlist", pid, {"playlist": {"pid": pid}}),
        place_album=lambda album_id: _record("place_album", album_id, {"playlist": {"pid": album_id}}),
        recommend=lambda seed_ids, top_k=20, method="centroid": _record("recommend", (seed_ids, top_k, method), {"results": seed_ids}),
        song_detail=lambda song_id, top_n=10: _record("song_detail", (song_id, top_n), {"id": song_id}),
        get_graph=lambda: _record("get_graph", True, {"nodes": [], "links": [], "groups": {}}),
        clear_graph=lambda: _record("clear_graph", True, {"ok": True}),
        remove_node=lambda node_id: _record("remove_node", node_id, {"removed": [node_id], "node": {"id": node_id}}),
    )
    monkeypatch.setattr(mentor_graph, "atlas", fake_atlas)

    tools = MentorGraphTools(_FakeAnther())
    assert tools.search("halo", limit=3)["results"] == ["halo", 3]
    assert tools.search_playlists("wave", limit=4)["results"] == ["wave", 4]
    assert tools.search_albums("pop", limit=5)["results"] == ["pop", 5]
    assert tools.place_song({"id": "x"}) == {"nodes": [{"id": "x"}], "links": []}
    assert tools.place_playlist(1)["playlist"]["pid"] == 1
    assert tools.place_album("al1")["playlist"]["pid"] == "al1"
    assert tools.recommend(["a"], top_k=2, method="topk")["results"] == ["a"]
    assert tools.song_detail("song:1", top_n=7)["id"] == "song:1"
    assert tools.get_graph()["groups"] == {}
    assert tools.clear_graph()["ok"] is True
    assert tools.remove_node("node:1")["ok"] is True
    assert calls["search"] == ("halo", 3)
    assert calls["remove_node"] == "node:1"


def test_execute_dispatches_compare_and_artist_tracks():
    react = MentorReAct(_FakeAnther(), generate_fn=lambda *a, **k: "")

    compare_obs = react._execute(
        {"tool": "compare", "args": {"anchor_a": "Halo by Beyonce", "anchor_b": "Take a Step Back", "top_k": 2}}
    )
    artist_obs = react._execute(
        {"tool": "artist_tracks", "args": {"anchor": "Halo by Beyonce", "artist": "xxxtentacion", "top_k": 2}}
    )

    assert compare_obs["tool"] == "compare"
    assert compare_obs["ok"] is True
    assert artist_obs["tool"] == "artist_tracks"
    assert artist_obs["ok"] is True
    assert [n["name"] for n in artist_obs["neighbors"]] == ["Look at Me!", "Sad!"]


def test_execute_still_rejects_unsupported_map_mutating_tools():
    react = MentorReAct(_FakeAnther(), generate_fn=lambda *a, **k: "")

    obs = react._execute({"tool": "place_song", "args": {"result": {"id": "x"}}})

    assert obs == {"tool": "place_song", "ok": False, "error": "unknown_tool"}


def test_fallback_call_parses_artist_tracks_and_compare():
    artist_call = _fallback_call("what xxxtentacion songs are the closest to Halo by Beyonce?")
    compare_call = _fallback_call("compare Take a Step Back to Halo by Beyonce")

    assert artist_call["tool"] == "artist_tracks"
    assert artist_call["args"]["artist"] == "xxxtentacion"
    assert artist_call["args"]["anchor"] == "Halo by Beyonce"
    assert compare_call["tool"] == "compare"
    assert compare_call["args"]["anchor_a"] == "Take a Step Back"
    assert compare_call["args"]["anchor_b"] == "Halo by Beyonce"
