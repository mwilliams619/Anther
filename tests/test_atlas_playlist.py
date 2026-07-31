"""
Playlist search + placement (ui/atlas.py), with a tiny injected corpus
carrying playlist membership. No network, no MERT, no bundle on disk —
mirrors the fixture pattern of test_atlas_search.py.

Playlists place as ordinary query nodes with per-track corpus-neighbor
fan-out (no artificial hub). Tests run the corpus-fallback path by forcing
mpd_ready() False; the full-MPD path is covered by stubbing mpd_sql +
playlist_jobs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
import atlas  # noqa: E402

from anther_ml.corpus import ReferenceCorpus  # noqa: E402
from anther_ml.similarity import SongIndex  # noqa: E402

# STALE: written against the pre-multi-session atlas. The `injected` fixture and
# test bodies use module globals (`_graph`, `_link_keys`, `_query_vecs`,
# `_groups`, `GRAPH_PATH`, `EMBED_CACHE_PATH`) that the per-session
# `_SessionState` refactor removed, so every test errors at setup. Excluded from
# the default suite; port to the session model to revive.
pytestmark = pytest.mark.stale


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
    profiles = [{"cluster_id": 0, "label_final": "synth pop"},
                {"cluster_id": 1, "label_final": "dark wave"}]
    return ReferenceCorpus(emb, index, leiden, np.zeros((4, 2)), emb[:1],
                           profiles, manifest)


@pytest.fixture
def injected(monkeypatch, tmp_path):
    """Inject the fake corpus into atlas' module globals; isolate graph I/O."""
    corpus = _fake_corpus()
    pindex, prows = atlas._build_playlist_index(corpus)
    monkeypatch.setattr(atlas, "_corpus", corpus)
    monkeypatch.setattr(atlas, "_id_to_idx", {m["id"]: i for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_id_to_cluster",
                        {m["id"]: int(corpus.labels[i]) for i, m in enumerate(corpus.metadata)})
    monkeypatch.setattr(atlas, "_profile_by_cluster",
                        {p["cluster_id"]: p for p in (corpus.profiles or [])})
    monkeypatch.setattr(atlas, "_playlist_index", pindex)
    monkeypatch.setattr(atlas, "_playlist_rows", prows)
    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "_query_vecs", {})
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(atlas, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(atlas, "EMBED_CACHE_PATH", tmp_path / "embed_cache.sqlite")
    monkeypatch.setattr(atlas, "load", lambda: corpus)  # don't touch disk
    monkeypatch.setattr(atlas, "mpd_ready", lambda: False)  # corpus-fallback path
    # MERT must never load for playlist / corpus-tier placement
    monkeypatch.setattr(atlas, "_mert",
                        lambda: (_ for _ in ()).throw(AssertionError("MERT loaded")))
    return corpus


# ── playlist-name search (corpus fallback) ───────────────────────────────────

def test_search_playlists_finds_by_name(injected):
    out = atlas.search_playlists("new wave")
    hits = out["results"]
    assert hits and hits[0]["pid"] == 1
    assert hits[0]["name"] == "new wave classics"
    assert hits[0]["n_tracks"] == 2
    assert hits[0]["source"] == "corpus"
    # token prefilter excludes the other playlist entirely
    assert all(h["pid"] != 2 for h in hits)
    # fallback mode advertises the prepare-ui upgrade path
    assert out["notice"] == atlas.MPD_PREP_HINT


def test_search_playlists_empty_query(injected):
    assert atlas.search_playlists("")["results"] == []
    assert atlas.search_playlists("zzz nonexistent")["results"] == []


# ── placement: query nodes with neighbor fan-out, no hub ─────────────────────

def test_place_playlist_places_members_with_fanout(injected):
    resp = atlas.place_playlist(1)
    frag = resp["fragment"]

    assert not any(n.get("kind") == "playlist" for n in frag["nodes"])
    assert not any(l.get("kind") == "member" for l in frag["links"])

    members = [n for n in frag["nodes"] if n["kind"] == "query"]
    assert {m["id"] for m in members} == {"spotify:t0", "spotify:t1"}
    assert all(m["cluster"] == 0 and m["playlist_pid"] == 1 for m in members)

    # each member got its own corpus-neighbor fan-out (plain links from it)
    for tid in ("spotify:t0", "spotify:t1"):
        fan = [l for l in frag["links"]
               if l.get("kind") is None and l["source"] == tid]
        assert fan, f"{tid} has no neighbor links"

    assert resp["playlist"] == {"pid": 1, "name": "new wave classics",
                                "n_tracks": 2, "n_total": 2, "capped": False,
                                "n_immediate": 2, "n_pending": 0}
    assert resp["job_id"] is None
    assert resp["notice"] == atlas.MPD_PREP_HINT
    # members registered for future query↔query edges
    assert set(atlas._query_vecs) == {"spotify:t0", "spotify:t1"}


def test_place_playlist_accepts_str_pid_and_rejects_unknown(injected):
    resp = atlas.place_playlist("2")
    assert resp["playlist"]["pid"] == 2
    with pytest.raises(ValueError):
        atlas.place_playlist(999)


# ── full-MPD path: immediate in-corpus + pending handed to the job runner ────

def test_place_playlist_mpd_splits_immediate_and_pending(injected, monkeypatch):
    monkeypatch.setattr(atlas, "mpd_ready", lambda: True)
    rows = [
        {"track_id": "t0", "name": "Blue Monday", "artist": "New Order",
         "preview_url": None, "popularity": 60.0},               # in-corpus
        {"track_id": "x9", "name": "Unknown Song", "artist": "Nobody",
         "preview_url": "http://example/x9.mp3", "popularity": 10.0},  # pending
    ]
    monkeypatch.setattr(atlas.mpd_sql, "playlist_tracks", lambda db, pid: rows)
    monkeypatch.setattr(atlas.mpd_sql, "playlist_name", lambda db, pid: "mixed bag")

    started = {}

    class _StubJobs:
        @staticmethod
        def start(pid, name, pending):
            started.update(pid=pid, name=name, pending=pending)
            return "job-123"

    monkeypatch.setitem(sys.modules, "playlist_jobs", _StubJobs)

    resp = atlas.place_playlist("abc123")
    assert resp["playlist"] == {"pid": "abc123", "name": "mixed bag",
                                "n_tracks": 2, "n_total": 2, "capped": False,
                                "n_immediate": 1, "n_pending": 1}
    assert resp["job_id"] == "job-123"
    assert started["pending"] == [{"id": "spotify:x9", "name": "Unknown Song",
                                   "artist": "Nobody",
                                   "preview_url": "http://example/x9.mp3"}]
    placed = [n for n in resp["fragment"]["nodes"] if n["kind"] == "query"]
    assert [n["id"] for n in placed] == ["spotify:t0"]
    assert placed[0]["playlist_pid"] == "abc123"


# ── import cap ───────────────────────────────────────────────────────────────

def test_place_playlist_caps_import(injected, monkeypatch):
    monkeypatch.setattr(atlas, "IMPORT_CAP", 1)
    resp = atlas.place_playlist(1)                        # playlist has 2 tracks
    p = resp["playlist"]
    assert p["capped"] is True
    assert p["n_total"] == 2
    assert p["n_tracks"] == 1
    assert p["n_immediate"] == 1
    members = [n for n in resp["fragment"]["nodes"] if n["kind"] == "query"]
    assert len(members) == 1


# ── album placement (Deezer-sourced, stubbed) ────────────────────────────────

def test_place_album_splits_and_groups(injected, monkeypatch):
    deezer = {
        "album/al1": {"id": "al1", "title": "Discovery",
                      "artist": {"name": "Daft Punk"}, "nb_tracks": 2},
        "album/al1/tracks": {"data": [
            {"id": 91, "title": "One More Time",
             "artist": {"name": "Daft Punk"}, "preview": "http://x/91.mp3"},
            {"id": 92, "title": "Aerodynamic",
             "artist": {"name": "Daft Punk"}, "preview": "http://x/92.mp3"},
        ]},
    }
    monkeypatch.setattr(atlas, "_deezer_get", lambda path, params=None: deezer[path])
    # deezer:91 was embedded before → cached → instant; 92 → pending
    vec = np.zeros(4, dtype=np.float32); vec[2] = 1.0
    monkeypatch.setattr(atlas, "cached_vec",
                        lambda tid: vec if tid == "deezer:91" else None)

    started = {}

    class _StubJobs:
        @staticmethod
        def start(pid, name, pending):
            started.update(pid=pid, name=name, pending=pending)
            return "job-al1"

    monkeypatch.setitem(sys.modules, "playlist_jobs", _StubJobs)

    resp = atlas.place_album("al1")
    p = resp["playlist"]
    assert p["pid"] == "album:al1"
    assert p["name"] == "Daft Punk — Discovery"
    assert (p["n_tracks"], p["n_immediate"], p["n_pending"]) == (2, 1, 1)
    assert resp["job_id"] == "job-al1"
    assert started["pid"] == "album:al1"
    assert started["pending"] == [{"id": "deezer:92", "name": "Aerodynamic",
                                   "artist": "Daft Punk",
                                   "preview_url": "http://x/92.mp3"}]
    placed = [n for n in resp["fragment"]["nodes"] if n["kind"] == "query"]
    assert [n["id"] for n in placed] == ["deezer:91"]
    assert placed[0]["playlist_pid"] == "album:al1"
    assert placed[0]["source"] == "deezer"

    with pytest.raises(ValueError):
        monkeypatch.setattr(atlas, "_deezer_get",
                            lambda path, params=None: {"error": {"message": "nope"}})
        atlas.place_album("zzz")


# ── dedupe / promote / idempotency ───────────────────────────────────────────

def test_place_playlist_promotes_existing_nodes(injected):
    atlas.place_song(atlas._search_corpus("blue monday", limit=1)[0])
    # t0 is now a query node and t1 one of its grey corpus neighbours
    resp = atlas.place_playlist(1)

    graph = atlas.get_graph()
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert atlas._graph["nodes"]["spotify:t0"]["kind"] == "query"
    assert atlas._graph["nodes"]["spotify:t1"]["kind"] == "query"   # promoted
    assert atlas._graph["nodes"]["spotify:t1"]["playlist_pid"] == 1


# ── query↔query edges ────────────────────────────────────────────────────────

def test_place_playlist_qq_edges_capped(injected, monkeypatch):
    monkeypatch.setattr(atlas, "TOP_K", 0)          # keep fan-out off the graph
    monkeypatch.setattr(atlas, "_qq_threshold", -1.0)  # every pair clears the bar
    monkeypatch.setattr(atlas, "QQ_MAX_PER_NODE", 1)
    atlas.place_song(atlas._search_corpus("enjoy the silence", limit=1)[0])

    resp = atlas.place_playlist(1)
    qq = [l for l in resp["fragment"]["links"] if l.get("kind") == "qq"]
    # intra-batch pairs are eligible now (no hub grouping them), but each
    # placed node adds at most QQ_MAX_PER_NODE edges
    assert len(qq) == 2
    per_source = {}
    for l in qq:
        per_source[l["source"]] = per_source.get(l["source"], 0) + 1
    assert all(v <= 1 for v in per_source.values())


# ── persistence round-trip + legacy-hub migration ────────────────────────────

def test_graph_reload_restores_members_and_strips_legacy_hub(injected, monkeypatch):
    atlas.place_playlist(1)

    # simulate a graph saved by the old hub-and-spoke code
    with atlas._graph_lock:
        atlas._graph["nodes"]["playlist:1"] = {
            "id": "playlist:1", "name": "new wave classics",
            "kind": "playlist", "source": "playlist", "pid": 1}
        atlas._graph["links"].append({"source": "playlist:1",
                                      "target": "spotify:t0",
                                      "value": 0.9, "kind": "member"})
        atlas._save_graph()

    monkeypatch.setattr(atlas, "_graph", {"nodes": {}, "links": []})
    monkeypatch.setattr(atlas, "_link_keys", set())
    monkeypatch.setattr(atlas, "_query_vecs", {})
    atlas._load_graph()

    assert "playlist:1" not in atlas._graph["nodes"]
    assert not any(l.get("kind") == "member" for l in atlas._graph["links"])
    # members regain their similarity vecs and their playlist tag
    assert {"spotify:t0", "spotify:t1"} <= set(atlas._query_vecs)
    assert atlas._graph["nodes"]["spotify:t0"]["playlist_pid"] == 1
