"""
Tiered search + force-graph placement (ui/atlas.py), with a tiny injected
corpus. No network, no MERT, no bundle on disk — corpus-tier placement and the
tier fall-through are exercised directly.
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
    """Four near-orthogonal tracks with searchable titles/artists."""
    emb = np.eye(4, dtype=np.float32)
    meta = [
        {"id": "spotify:t0", "name": "Blue Monday", "artist": "New Order"},
        {"id": "spotify:t1", "name": "Bizarre Love Triangle", "artist": "New Order"},
        {"id": "spotify:t2", "name": "Enjoy the Silence", "artist": "Depeche Mode"},
        {"id": "spotify:t3", "name": "Personal Jesus", "artist": "Depeche Mode"},
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
    monkeypatch.setattr(atlas, "_corpus", corpus)
    monkeypatch.setattr(atlas, "_id_to_idx", {m["id"]: i for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_id_to_cluster",
                        {m["id"]: int(corpus.labels[i]) for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_profile_by_cluster",
                        {p["cluster_id"]: p for p in (corpus.profiles or [])})
    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(atlas, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(atlas, "load", lambda: corpus)  # don't touch disk
    return corpus


# ── tier 1: corpus text search ───────────────────────────────────────────────

def test_corpus_search_ranks_title_match_first(injected):
    hits = atlas._search_corpus("blue monday", limit=10)
    assert hits[0]["id"] == "spotify:t0"
    assert hits[0]["source"] == "corpus"
    assert hits[0]["score"] >= atlas.STRONG_SCORE
    assert hits[0]["cluster"] == 0


def test_corpus_search_substring_prefilter_excludes_nonmatches(injected):
    # "depeche" appears only in the two Depeche Mode artists.
    ids = {h["id"] for h in atlas._search_corpus("depeche", limit=10)}
    assert ids == {"spotify:t2", "spotify:t3"}


def test_search_falls_through_to_deezer_when_corpus_weak(injected, monkeypatch):
    calls = {"deezer": 0}

    def fake_deezer(path, params=None, **kw):
        calls["deezer"] += 1
        return {"data": [{"id": 42, "preview": "http://x/p.mp3",
                          "title": "Obscure Track", "artist": {"name": "Nobody"},
                          "album": {"title": "LP", "cover_small": ""}}]}

    monkeypatch.setattr(atlas, "_deezer_get", fake_deezer)
    out = atlas.search("zzz nonexistent nowhere", limit=10)
    sources = {r["source"] for r in out["results"]}
    assert "deezer" in sources
    assert calls["deezer"] == 1
    deez = next(r for r in out["results"] if r["source"] == "deezer")
    assert deez["id"] == "deezer:42"
    assert deez["preview_url"] == "http://x/p.mp3"


def test_search_skips_deezer_when_corpus_strong(injected, monkeypatch):
    called = {"deezer": False}
    monkeypatch.setattr(atlas, "_deezer_get",
                        lambda *a, **k: called.__setitem__("deezer", True) or {"data": []})
    # Both "New Order" tracks + fuzzy others clear enough matches? Ensure a clear
    # single strong hit still allows fall-through only when < CORPUS_MIN_HITS.
    monkeypatch.setattr(atlas, "CORPUS_MIN_HITS", 1)
    out = atlas.search("new order", limit=10)
    assert not called["deezer"]
    assert all(r["source"] == "corpus" for r in out["results"])


# ── tier 3: spotify gated on credentials ─────────────────────────────────────

def test_spotify_tier_reports_not_configured(injected, monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    assert atlas.spotify_configured() is False
    res = atlas._search_spotify("anything", limit=5)
    assert res == {"configured": False, "results": []}


# ── placement: corpus tier builds a graph fragment without MERT ──────────────

def test_place_corpus_hit_builds_fragment_without_mert(injected, monkeypatch):
    # Guard: MERT must never be loaded for a corpus-tier placement.
    monkeypatch.setattr(atlas, "_mert",
                        lambda: (_ for _ in ()).throw(AssertionError("MERT loaded")))
    hit = atlas._search_corpus("blue monday", limit=1)[0]
    frag = atlas.place_song(hit)

    query_nodes = [n for n in frag["nodes"] if n["kind"] == "query"]
    assert len(query_nodes) == 1
    q = query_nodes[0]
    assert q["id"] == "spotify:t0"
    assert q["cluster"] == 0
    # Neighbours exclude self; links wire the query to them.
    neigh_ids = {n["id"] for n in frag["nodes"] if n["kind"] == "corpus"}
    assert "spotify:t0" not in neigh_ids
    assert len(neigh_ids) >= 1
    assert all(l["source"] == "spotify:t0" for l in frag["links"])
    assert {l["target"] for l in frag["links"]} <= neigh_ids


def test_place_dedupes_shared_neighbor_across_songs(injected, monkeypatch):
    monkeypatch.setattr(atlas, "_mert",
                        lambda: (_ for _ in ()).throw(AssertionError("MERT loaded")))
    atlas.place_song(atlas._search_corpus("blue monday", limit=1)[0])
    frag2 = atlas.place_song(atlas._search_corpus("bizarre love triangle", limit=1)[0])
    # t1 was pulled in as a neighbour of t0; re-adding it as a query promotes it
    # in place rather than duplicating the node.
    graph = atlas.get_graph()
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert atlas._graph["nodes"]["spotify:t1"]["kind"] == "query"
