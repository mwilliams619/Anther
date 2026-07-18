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


# ---- short graph follow-ups: a hard rule, not an LLM guess -------------------
#
# The shipped failure (screenshot, 2026-07-15): after a GRAPH turn about
# "TV Off", the user asked "what are the scores?" — no _GRAPH_PHRASES match,
# no on-map entity mention, so it fell to the 4-token LLM classifier, which
# read "scores" as sports and answered OFFTOPIC ("I'm here for musicians'
# questions, not sports"). The old "[Previous turn: GRAPH]" hint was only
# ever advisory text inside that same misfiring LLM call.

def test_short_followup_after_graph_stays_graph_even_if_llm_says_offtopic(mentor):
    mentor._scripted = "OFFTOPIC"                  # LLM would misfire, exactly as shipped
    mentor.graph_ctx  # sanity: fixture wired
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("what are the scores?", state=state) == "GRAPH"


def test_short_followup_literal_screenshot_repro(mentor):
    mentor._scripted = "OFFTOPIC"
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("what are the scores?", state=state) == "GRAPH"
    # the LLM is never even asked -- the hard rule short-circuits classify_intent
    assert mentor._last_intent_prompt is None


def test_short_followup_which_ones_after_graph(mentor):
    mentor._scripted = "OFFTOPIC"
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("which ones?", state=state) == "GRAPH"


def test_short_followup_after_advice_does_not_force_graph(mentor):
    # Same short question, but the previous turn was ADVICE, not GRAPH --
    # the hard rule must not fire; it defers to the LLM as before.
    mentor._scripted = "ADVICE"
    state = MentorContext(last_intent="ADVICE")
    assert mentor.classify_intent("what are the scores?", state=state) == "ADVICE"
    assert mentor._last_intent_prompt is not None   # LLM was consulted


def test_short_followup_with_no_previous_intent_defers_to_llm(mentor):
    mentor._scripted = "OFFTOPIC"
    state = MentorContext()                        # last_intent is None
    assert mentor.classify_intent("what are the scores?", state=state) == "OFFTOPIC"
    assert mentor._last_intent_prompt is not None


def test_long_followup_after_graph_still_defers_to_llm(mentor):
    # Longer than SHORT_FOLLOWUP_MAX_WORDS -- the hard rule backs off and
    # the (weaker, advisory) previous-turn hint is left to do its job.
    mentor._scripted = "ADVICE"
    state = MentorContext(last_intent="GRAPH")
    long_q = "actually never mind the map, can you give me general advice on mixing vocals"
    assert mentor.classify_intent(long_q, state=state) == "ADVICE"
    assert mentor._last_intent_prompt is not None


def test_short_followup_backs_off_on_clear_advice_topic_change(mentor):
    # Short, but names a real subject change (release strategy) -- must not
    # be swallowed by the graph-continuity rule.
    mentor._scripted = "ADVICE"
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("how's my release going?", state=state) == "ADVICE"
    assert mentor._last_intent_prompt is not None


def test_short_followup_backs_off_on_clear_offtopic_topic_change(mentor):
    mentor._scripted = "OFFTOPIC"
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("what's today's game score?", state=state) == "OFFTOPIC"
    assert mentor._last_intent_prompt is not None


def test_short_graph_followup_helper_directly(mentor):
    graph_state = MentorContext(last_intent="GRAPH")
    advice_state = MentorContext(last_intent="ADVICE")
    none_state = MentorContext()
    assert mentor._short_graph_followup("what are the scores?", graph_state)
    assert mentor._short_graph_followup("which ones?", graph_state)
    assert not mentor._short_graph_followup("what are the scores?", advice_state)
    assert not mentor._short_graph_followup("what are the scores?", none_state)
    assert not mentor._short_graph_followup("", graph_state)
    long_q = "tell me a whole lot more about this because I am very curious about it"
    assert not mentor._short_graph_followup(long_q, graph_state)


# ---- ADVICE-continuity: a reformat/continuation request, not a topic pivot --
#
# The shipped failure (live session, 2026-07-16): "help me plan my release"
# correctly routed ADVICE, then "can you pu tthst in bullet points for me?/"
# (9 words, no _GRAPH_PHRASES match, no on-map entity) was misread as GRAPH
# by the LLM classifier, and the graph agent then failed anchor resolution
# ("Nothing is selected on your map..."), losing the advice thread. This is
# the mirror-image bug to the GRAPH-continuity one above.

# NOTE: reformat/continuation requests ("bullet points", "shorten that",
# "tldr") used to be forced to ADVICE by a phrase list INSIDE classify_intent.
# That gate has moved: it is now _is_reformat_request, checked in chat()
# BEFORE classify_intent is even called (see tests/test_reformat_request.py).
# classify_intent on its own no longer special-cases these phrases, so a
# bare classify_intent() call on reformat-shaped text now legitimately
# defers to the LLM/soft-hint path like any other ambiguous short follow-up
# -- the interception that actually protects the user happens one level up.
def test_bare_classify_intent_no_longer_special_cases_reformat_phrasing(mentor):
    # Without the chat()-level interception, classify_intent alone treats
    # this like any other short follow-up: no hard rule fires (last_intent
    # is ADVICE, not GRAPH, so _short_graph_followup doesn't apply either),
    # so it defers to the (scripted) LLM.
    mentor._scripted = "GRAPH"
    state = MentorContext(last_intent="ADVICE")
    q = "can you pu tthst in bullet points for me?/"
    assert mentor.classify_intent(q, state=state) == "GRAPH"
    assert mentor._last_intent_prompt is not None


def test_continuation_phrasing_after_graph_stays_graph_via_short_graph_rule(mentor):
    # Reformat phrasing right after a GRAPH turn is still correctly routed
    # GRAPH -- not via any ADVICE-continuity rule (that lived in chat()'s
    # _is_reformat_request now, and only fires when last_intent == "ADVICE"),
    # but via the pre-existing _short_graph_followup rule, since this reads
    # as a short continuation of the GRAPH thread.
    mentor._scripted = "ADVICE"                    # LLM would say ADVICE if consulted...
    state = MentorContext(last_intent="GRAPH")
    assert mentor.classify_intent("put that in bullet points", state=state) == "GRAPH"
    assert mentor._last_intent_prompt is None      # ...but the hard rule short-circuits first


def test_continuation_phrasing_with_no_previous_intent_defers_to_llm(mentor):
    mentor._scripted = "OFFTOPIC"
    state = MentorContext()                        # last_intent is None
    assert mentor.classify_intent("can you shorten that?", state=state) == "OFFTOPIC"
    assert mentor._last_intent_prompt is not None


def test_non_continuation_short_question_after_advice_still_defers_to_llm(mentor):
    # A short question after ADVICE must still defer to the LLM as before --
    # classify_intent has no ADVICE-continuity rule of its own; that logic
    # now lives one level up in chat()'s _is_reformat_request.
    mentor._scripted = "OFFTOPIC"
    state = MentorContext(last_intent="ADVICE")
    assert mentor.classify_intent("what about tuesday?", state=state) == "OFFTOPIC"
    assert mentor._last_intent_prompt is not None
