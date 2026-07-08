"""
Playlist search + one-shot playlist placement (ui/atlas.py), with a tiny
injected corpus carrying playlist membership. No network, no MERT, no bundle
on disk — mirrors the fixture pattern of test_atlas_search.py.
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
    """Four near-orthogonal tracks split across two playlists."""
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
    return ReferenceCorpus(emb, index, leiden, np.zeros((4, 2)), emb[:1], [], manifest)


@pytest.fixture
def injected(monkeypatch, tmp_path):
    """Inject the fake corpus into atlas' module globals; isolate graph I/O."""
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
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(atlas, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(atlas, "load", lambda: corpus)  # don't touch disk
    # MERT must never load for playlist / corpus-tier placement
    monkeypatch.setattr(atlas, "_mert",
                        lambda: (_ for _ in ()).throw(AssertionError("MERT loaded")))
    return corpus


# ── playlist-name search ─────────────────────────────────────────────────────

def test_search_playlists_finds_by_name(injected):
    hits = atlas.search_playlists("new wave")
    assert hits and hits[0]["pid"] == 1
    assert hits[0]["name"] == "new wave classics"
    assert hits[0]["n_tracks"] == 2
    # token prefilter excludes the other playlist entirely
    assert all(h["pid"] != 2 for h in hits)


def test_search_playlists_empty_query(injected):
    assert atlas.search_playlists("") == []
    assert atlas.search_playlists("zzz nonexistent") == []


# ── placement: hub-and-spoke fragment, no neighbor fan-out ───────────────────

def test_place_playlist_builds_hub_fragment(injected):
    frag = atlas.place_playlist(1)

    hub = next(n for n in frag["nodes"] if n["kind"] == "playlist")
    assert hub["id"] == "playlist:1"
    assert hub["name"] == "new wave classics"
    assert hub["n_tracks"] == 2

    members = [n for n in frag["nodes"] if n["kind"] == "query"]
    assert {m["id"] for m in members} == {"spotify:t0", "spotify:t1"}
    assert all(m["cluster"] == 0 and m["playlist_pid"] == 1 for m in members)
    # no corpus-neighbor fan-out nodes
    assert all(n["kind"] in ("playlist", "query") for n in frag["nodes"])

    member_links = [l for l in frag["links"] if l["kind"] == "member"]
    assert len(member_links) == 2
    assert all(l["source"] == "playlist:1" for l in member_links)

    assert frag["playlist"]["hub_id"] == "playlist:1"
    assert frag["playlist"]["added"] == 2
    assert frag["playlist"]["already_on_map"] == 0
    # members registered for future query↔query edges; hub carries no vec
    assert set(atlas._query_vecs) == {"spotify:t0", "spotify:t1"}


def test_place_playlist_accepts_str_pid_and_rejects_unknown(injected):
    frag = atlas.place_playlist("2")
    assert frag["playlist"]["pid"] == 2
    with pytest.raises(ValueError):
        atlas.place_playlist(999)


# ── dedupe / promote / idempotency ───────────────────────────────────────────

def test_place_playlist_promotes_existing_nodes(injected):
    atlas.place_song(atlas._search_corpus("blue monday", limit=1)[0])
    # t0 is now a query node and t1 one of its grey corpus neighbours
    frag = atlas.place_playlist(1)

    graph = atlas.get_graph()
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert atlas._graph["nodes"]["spotify:t0"]["kind"] == "query"
    assert atlas._graph["nodes"]["spotify:t1"]["kind"] == "query"   # promoted
    assert frag["playlist"]["already_on_map"] == 2
    assert frag["playlist"]["added"] == 0

    # reloading the same playlist is a no-op fragment
    frag2 = atlas.place_playlist(1)
    assert frag2["nodes"] == [] and frag2["links"] == []


# ── query↔query edges ────────────────────────────────────────────────────────

def test_place_playlist_qq_edges_skip_intra_batch(injected, monkeypatch):
    monkeypatch.setattr(atlas, "TOP_K", 0)          # keep t2 fan-out off the graph
    monkeypatch.setattr(atlas, "_qq_threshold", -1.0)  # every pair clears the bar
    atlas.place_song(atlas._search_corpus("enjoy the silence", limit=1)[0])

    frag = atlas.place_playlist(1)
    qq = [l for l in frag["links"] if l["kind"] == "qq"]
    # each member links to the pre-existing query, never to its batch sibling
    assert {(l["source"], l["target"]) for l in qq} == {
        ("spotify:t0", "spotify:t2"), ("spotify:t1", "spotify:t2")}


# ── persistence round-trip ───────────────────────────────────────────────────

def test_graph_reload_restores_members_not_hub(injected, monkeypatch):
    atlas.place_playlist(1)

    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "_query_vecs", {})
    atlas._load_graph()

    assert "playlist:1" in atlas._graph["nodes"]
    assert atlas._graph["nodes"]["playlist:1"]["kind"] == "playlist"
    # members regain their similarity vecs; the artificial hub never has one
    assert set(atlas._query_vecs) == {"spotify:t0", "spotify:t1"}
    member_links = [l for l in atlas._graph["links"] if l.get("kind") == "member"]
    assert len(member_links) == 2
