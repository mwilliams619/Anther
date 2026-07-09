"""
Map management (ui/atlas.py): clear_graph, remove_node, groups registry, and
cache-first re-placement. Same injected-corpus fixture pattern as
test_atlas_playlist.py — no network, no MERT, no bundle on disk.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
import atlas  # noqa: E402

from anther_ml.corpus import ReferenceCorpus  # noqa: E402
from anther_ml.similarity import SongIndex  # noqa: E402


def _fake_corpus():
    emb = np.eye(4, dtype=np.float32)
    meta = [
        {"id": "spotify:t0", "name": "Blue Monday", "artist": "New Order",
         "playlists": [{"pid": 1, "name": "new wave classics"}]},
        {"id": "spotify:t1", "name": "Bizarre Love Triangle", "artist": "New Order",
         "playlists": [{"pid": 1, "name": "new wave classics"}]},
        {"id": "spotify:t2", "name": "Enjoy the Silence", "artist": "Depeche Mode",
         "playlists": [{"pid": 2, "name": "quiet storm"}]},
        {"id": "spotify:t3", "name": "Personal Jesus", "artist": "Depeche Mode",
         "playlists": [{"pid": 2, "name": "quiet storm"}]},
    ]
    index = SongIndex(emb, meta, standardize=False)
    leiden = {"labels": np.array([0, 0, 1, 1]), "scaler": None, "pca": None,
              "reducer_2d": None, "clustering_space": emb}
    manifest = {"corpus_format_version": 1, "embedding_config": {}}
    profiles = [{"cluster_id": 0, "label_final": "synth pop"},
                {"cluster_id": 1, "label_final": "dark wave"}]
    return ReferenceCorpus(emb, index, leiden, np.zeros((4, 2)), emb[:1],
                           profiles, manifest)


@pytest.fixture
def injected(monkeypatch, tmp_path):
    corpus = _fake_corpus()
    pindex, prows = atlas._build_playlist_index(corpus)
    monkeypatch.setattr(atlas, "_corpus", corpus)
    monkeypatch.setattr(atlas, "_id_to_idx", {m["id"]: i for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_id_to_cluster",
                        {m["id"]: int(corpus.labels[i]) for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_playlist_index", pindex)
    monkeypatch.setattr(atlas, "_playlist_rows", prows)
    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "_query_vecs", {})
    monkeypatch.setattr(atlas, "_groups", {})
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(atlas, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(atlas, "EMBED_CACHE_PATH", tmp_path / "embed_cache.sqlite")
    monkeypatch.setattr(atlas, "load", lambda: corpus)
    monkeypatch.setattr(atlas, "mpd_ready", lambda: False)
    monkeypatch.setattr(atlas, "_mert",
                        lambda: (_ for _ in ()).throw(AssertionError("MERT loaded")))
    return corpus


def _place(q):
    return atlas.place_song(atlas._search_corpus(q, limit=1)[0])


# ── remove_node ──────────────────────────────────────────────────────────────

def test_remove_node_prunes_links_and_orphans(injected):
    _place("blue monday")                    # t0 query + t1..t3 corpus neighbors
    assert len(atlas._graph["nodes"]) == 4

    result = atlas.remove_node("spotify:t0")
    assert result["node"]["id"] == "spotify:t0"
    assert result["removed"][0] == "spotify:t0"
    # every neighbor was only linked to t0 → all pruned
    assert set(result["removed"]) == {"spotify:t0", "spotify:t1",
                                      "spotify:t2", "spotify:t3"}
    assert atlas._graph["nodes"] == {}
    assert atlas._graph["links"] == []
    assert atlas._link_keys == set()
    assert "spotify:t0" not in atlas._query_vecs


def test_remove_node_keeps_shared_neighbors(injected):
    _place("blue monday")                    # t0 query, t1-t3 neighbors
    _place("enjoy the silence")              # t2 promoted to query, has own links

    result = atlas.remove_node("spotify:t0")
    # t2 is a query node with its own links → stays; so do its neighbors
    assert "spotify:t2" in atlas._graph["nodes"]
    assert "spotify:t0" not in atlas._graph["nodes"]
    assert all(l["source"] != "spotify:t0" and l["target"] != "spotify:t0"
               for l in atlas._graph["links"])
    assert "spotify:t0" in result["removed"]


def test_remove_unknown_node_returns_none(injected):
    assert atlas.remove_node("spotify:nope") is None


# ── clear_graph ──────────────────────────────────────────────────────────────

def test_clear_graph_empties_everything(injected):
    atlas.place_playlist(1)
    assert atlas._graph["nodes"] and atlas._groups

    atlas.clear_graph()
    assert atlas._graph["nodes"] == {}
    assert atlas._graph["links"] == []
    assert atlas._link_keys == set()
    assert atlas._query_vecs == {}
    assert atlas._groups == {}
    g = atlas.get_graph()
    assert g["nodes"] == [] and g["links"] == [] and g["groups"] == {}


# ── groups registry ──────────────────────────────────────────────────────────

def test_groups_recorded_and_served(injected):
    atlas.place_playlist(1)
    g = atlas.get_graph()
    assert g["groups"] == {"1": {"name": "new wave classics", "kind": "playlist"}}


def test_groups_survive_reload(injected, monkeypatch):
    atlas.place_playlist(1)
    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "_query_vecs", {})
    monkeypatch.setattr(atlas, "_groups", {})
    atlas._load_graph()
    assert atlas._groups["1"]["name"] == "new wave classics"


# ── cache-first re-place ─────────────────────────────────────────────────────

def test_place_song_uses_cache_before_download(injected, monkeypatch):
    vec = np.zeros(4, dtype=np.float32); vec[3] = 1.0
    monkeypatch.setattr(atlas, "cached_vec",
                        lambda tid: vec if tid == "deezer:77" else None)
    monkeypatch.setattr(atlas, "_download_preview",
                        lambda r: (_ for _ in ()).throw(AssertionError("downloaded")))

    frag = atlas.place_song({"source": "deezer", "id": "deezer:77",
                             "title": "Cached Song", "artist": "Someone",
                             "playlist_pid": "album:al9"})
    node = atlas._graph["nodes"]["deezer:77"]
    assert node["kind"] == "query"
    assert node["playlist_pid"] == "album:al9"     # ring survives re-place
    assert frag["nodes"]
