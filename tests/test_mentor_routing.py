"""Stage-1 routing (ADVICE / GRAPH / OFFTOPIC) must be graph-aware.

The shipped failure: "what is Big On Big connected to on the map?" routed to
OFFTOPIC ("I see you're into geography...") because the router was blind to the
map. These tests build a MusicMentor WITHOUT loading the 7B model (via
__new__), wire in a real GraphContext + a stub LLM, and prove:

  * explicit map language wins outright, even if the LLM would say OFFTOPIC
  * a bare on-map entity is recognised as GRAPH
  * the LLM classifier is handed a census of what's on the map
  * genuinely off-topic questions still route OFFTOPIC
"""
import pytest

from mentor import graph_context
from mentor.graph_context import GraphContext, MentorContext
from mentor.mentor import MusicMentor
from tests import mentor_fixtures as F
from tests.mentor_fixtures import write_session_on_disk


@pytest.fixture()
def mentor(tmp_path, monkeypatch):
    """A MusicMentor with the model/RAG/agent skipped — just the routing logic,
    a real GraphContext over an on-disk session, and a scripted LLM."""
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setattr(graph_context, "SESSION_DIR", str(root))
    write_session_on_disk(root, "sess_dark", nodes=F.GRAPH_NODES, groups=F.GRAPH_GROUPS)

    m = MusicMentor.__new__(MusicMentor)          # skip __init__ (no model load)
    m.graph_ctx = GraphContext(session_id="sess_dark")
    m._last_intent_prompt = None

    def fake_generate(messages, **kw):
        m._last_intent_prompt = messages[-1]["content"]
        return m._scripted                        # whatever the test wants the LLM to say
    m._generate = fake_generate
    m._scripted = "OFFTOPIC"                        # default: pessimistic LLM
    return m


# ---- explicit map language wins even against a hostile LLM --------------------
def test_connected_to_on_the_map_is_graph(mentor):
    mentor._scripted = "OFFTOPIC"                  # LLM would misfire...
    assert mentor.classify_intent("what is Big On Big connected to on the map?") == "GRAPH"


def test_whats_on_the_map_is_graph(mentor):
    mentor._scripted = "OFFTOPIC"
    assert mentor.classify_intent("what's on the map right now?") == "GRAPH"


def test_why_is_this_node_here_is_graph(mentor):
    mentor._scripted = "ADVICE"
    assert mentor.classify_intent("why is this node here?") == "GRAPH"


def test_what_do_i_sound_like_is_graph(mentor):
    mentor._scripted = "OFFTOPIC"
    assert mentor.classify_intent("what do I sound like?") == "GRAPH"


# ---- bare on-map entity, no map phrase ---------------------------------------
def test_on_map_entity_is_graph(mentor):
    # "Burial" is on the map; no explicit "on the map" phrase here.
    mentor._scripted = "OFFTOPIC"
    assert mentor.classify_intent("tell me about Burial") == "GRAPH"


def test_offmap_entity_defers_to_llm(mentor):
    # Not on the map and no map phrase -> the LLM decides. It says OFFTOPIC.
    mentor._scripted = "OFFTOPIC"
    assert mentor.classify_intent("what's the capital of France?") == "OFFTOPIC"


# ---- the LLM is handed the map census ----------------------------------------
def test_classifier_prompt_carries_the_map_census(mentor):
    mentor._scripted = "ADVICE"
    mentor.classify_intent("how do I promote my next single?")
    assert "On the map right now" in mentor._last_intent_prompt
    assert "Burial" in mentor._last_intent_prompt


# ---- genuine advice / off-topic still route correctly ------------------------
def test_generic_advice_routes_advice(mentor):
    mentor._scripted = "ADVICE"
    assert mentor.classify_intent("how do I stop overproducing my tracks?") == "ADVICE"


def test_generic_offtopic_routes_offtopic(mentor):
    mentor._scripted = "OFFTOPIC"
    assert mentor.classify_intent("write me a python script to rename files") == "OFFTOPIC"


# ---- _graph_signal unit behaviour --------------------------------------------
def test_graph_signal_phrases(mentor):
    assert mentor._graph_signal("what sits between these two on the graph")
    assert mentor._graph_signal("who is this closest to")
    assert not mentor._graph_signal("i feel stuck creatively")


def test_graph_signal_survives_empty_map(tmp_path, monkeypatch):
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setattr(graph_context, "SESSION_DIR", str(root))
    m = MusicMentor.__new__(MusicMentor)
    m.graph_ctx = GraphContext(session_id="empty")
    # No entities to match, but explicit phrases still fire.
    assert m._graph_signal("what's on the map?")
    assert not m._graph_signal("tell me about Burial")   # nothing placed
