"""Live-graph tests: prove the mentor reads the RIGHT browser session's map
off disk, the way production does.

These are the tests that would have caught the shipped bug where
``GraphContext()`` crashed on a removed ``atlas.GRAPH_PATH`` and the mentor ran
blind. Here we place maps under ``ui/session/<sid>/`` (graph.json + a real
embed_cache.sqlite) and drive the pipeline by session id — no injected paths,
no GPU, no LLM.
"""
import json
import os

import numpy as np
import pytest

from mentor import graph_context
from mentor.graph_context import GraphContext, MentorContext
from mentor.graph_tools import MentorGraphTools
from tests import mentor_fixtures as F
from tests.mentor_fixtures import DARK, POP, FakeSimilarity, write_session_on_disk

# A second, deliberately different map so we can prove session isolation.
OTHER_NODES = [
    {"id": "z1", "artist": "Herbie Hancock", "name": "Chameleon", "cluster": 0,
     "cluster_label": "Jazz Funk",
     "tags": [{"genre": "jazz", "score": 0.9}], "vec": [0.6, 0.8]},
    {"id": "z2", "artist": "Weather Report", "name": "Birdland", "cluster": 0,
     "cluster_label": "Jazz Funk",
     "tags": [{"genre": "jazz", "score": 0.8}], "vec": [0.5, 0.86]},
]


@pytest.fixture()
def session_root(tmp_path, monkeypatch):
    """Point GraphContext's SESSION_DIR at a throwaway dir and populate two
    distinct browser sessions on disk."""
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setattr(graph_context, "SESSION_DIR", str(root))
    write_session_on_disk(root, "sess_dark", nodes=F.GRAPH_NODES, groups=F.GRAPH_GROUPS)
    write_session_on_disk(root, "sess_jazz", nodes=OTHER_NODES, groups={})
    return root


# ---- session resolution (the bug that shipped) --------------------------------
def test_graph_context_resolves_session_path(session_root):
    gc = GraphContext(session_id="sess_dark")
    assert gc.has_nodes()
    assert len(gc.nodes) == 5
    assert gc.graph_path.endswith(os.path.join("sess_dark", "graph.json"))


def test_vectors_come_from_that_sessions_embed_cache(session_root):
    gc = GraphContext(session_id="sess_dark")
    v = gc.vec_for_id("n1")                       # Embolo - First Demo
    assert v is not None and v.shape == (2,)      # read straight from sqlite
    assert gc.vec_for_id("does-not-exist") is None


def test_unknown_session_is_empty_not_crash(session_root):
    gc = GraphContext(session_id="never-opened")
    assert not gc.has_nodes()
    assert gc.summary() == ""


def test_default_when_no_session_id(session_root):
    # No sess id -> the shared 'default' session, which we didn't populate.
    gc = GraphContext()
    assert gc.session_id == "default"
    assert not gc.has_nodes()


# ---- session isolation + retargeting ------------------------------------------
def test_for_session_switches_maps(session_root):
    gc = GraphContext(session_id="sess_dark")
    assert {n["artist"] for n in gc.nodes} == {"Embolo", "Burial", "Aphex Twin", "Beyonce"}

    gc.for_session("sess_jazz")
    assert {n["artist"] for n in gc.nodes} == {"Herbie Hancock", "Weather Report"}
    assert gc.vec_for_id("z1") is not None
    assert gc.vec_for_id("n1") is None            # dark-session node is gone

    gc.for_session("sess_dark")                    # and back
    assert gc.has_nodes() and len(gc.nodes) == 5


def test_for_session_same_id_is_noop(session_root):
    gc = GraphContext(session_id="sess_dark")
    before = id(gc._by_id)
    gc.for_session("sess_dark")
    gc._reload_if_stale()
    assert gc.has_nodes()


# ---- full path: session id -> disk -> tools -----------------------------------
def test_inspect_graph_reflects_the_live_session(session_root):
    gc = GraphContext(session_id="sess_dark")
    tools = MentorGraphTools(FakeSimilarity(), graph_ctx=gc)
    obs = tools.inspect_graph(context=MentorContext(graph_session_id="sess_dark"))
    assert obs["ok"] and obs["visible_nodes"] == 5
    territories = {t["territory"] for t in obs["territories"]}
    assert territories == {DARK, POP}
    assert "Burial" in obs["artists"]


def test_neighbors_uses_live_session_vectors(session_root):
    gc = GraphContext(session_id="sess_dark")
    tools = MentorGraphTools(FakeSimilarity(), graph_ctx=gc)
    ctx = MentorContext(selected_node_id="n1", graph_session_id="sess_dark")
    obs = tools.neighbors("me", context=ctx)
    assert obs["ok"] and obs["scope"] == "map"
    assert obs["neighbors"][0]["song"] == "Archangel"   # nearest on the real map


def test_two_sessions_dont_bleed(session_root):
    dark = MentorGraphTools(FakeSimilarity(), graph_ctx=GraphContext(session_id="sess_dark"))
    jazz = MentorGraphTools(FakeSimilarity(), graph_ctx=GraphContext(session_id="sess_jazz"))
    d = dark.inspect_graph(context=MentorContext(graph_session_id="sess_dark"))
    j = jazz.inspect_graph(context=MentorContext(graph_session_id="sess_jazz"))
    assert "Burial" in d["artists"] and "Burial" not in j["artists"]
    assert "Herbie Hancock" in j["artists"] and "Herbie Hancock" not in d["artists"]


# ---- freshness: a re-place mid-conversation is picked up -----------------------
def test_reload_on_graph_change(session_root):
    gc = GraphContext(session_id="sess_dark")
    assert len(gc.nodes) == 5
    # user adds a song → graph.json rewritten with a newer mtime
    import time
    time.sleep(0.01)
    write_session_on_disk(session_root, "sess_dark",
                          nodes=F.GRAPH_NODES + OTHER_NODES, groups=F.GRAPH_GROUPS)
    assert len(gc.nodes) == 7                       # mtime-triggered reload


# ---- summary / entity detection over the live map -----------------------------
def test_summary_and_entity_detection(session_root):
    gc = GraphContext(session_id="sess_dark")
    s = gc.summary()
    assert "5 songs on the map" in s
    assert "Burial" in s
    assert gc.mentions_entity("what is burial connected to")
    assert gc.mentions_entity("tell me about First Demo")
    assert not gc.mentions_entity("what's the weather today")


# ---- opt-in smoke test against the developer's real sessions -------------------
REAL_SESSION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "session")


def _real_sessions_with_nodes():
    out = []
    if not os.path.isdir(REAL_SESSION_DIR):
        return out
    for name in os.listdir(REAL_SESSION_DIR):
        gp = os.path.join(REAL_SESSION_DIR, name, "graph.json")
        if not os.path.isfile(gp):
            continue
        try:
            with open(gp) as f:
                if json.load(f).get("nodes"):
                    out.append(name)
        except Exception:
            pass
    return out


# STALE (marked at the user's request): unlike the pre-multi-session atlas
# tests, this one exercises current production code — it is machine-data
# dependent, not architecture-stale. It only runs where real
# ui/session/<sid>/graph.json files exist, and asserts every populated session
# has a node with a cached vector; it fails when a local session's embed_cache
# was cleared/pruned while its graph.json was kept. Excluded from the default
# run; revisit by loosening the vec assertion or cleaning stray local sessions.
@pytest.mark.stale
@pytest.mark.skipif(not _real_sessions_with_nodes(),
                    reason="no populated ui/session/<sid>/graph.json on this machine")
def test_real_sessions_load_and_summarize():
    """If the developer has real maps on disk, every populated one must load,
    summarize, and expose vectors — the exact things production needs."""
    for sid in _real_sessions_with_nodes():
        gc = GraphContext(session_id=sid)
        assert gc.has_nodes(), sid
        assert gc.summary(), sid
        # at least one node should have a cached vector
        assert any(gc.vec_for_id(n["id"]) is not None for n in gc.nodes[:20]), sid
