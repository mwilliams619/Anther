"""Reformat/continuation requests on the previous ADVICE answer.

Two shipped failures, both from the same live transcript, after a "help me
plan my release" ADVICE turn:

  1. "format that in bullet points please" -> the bot repeated the exact
     same paragraph back verbatim, not reformatted at all. Root cause: the
     GROUNDED/SYNTHESIS/NO_RETRIEVAL branches all re-derive an answer from
     RAG passages or the mentor persona; none of them has any concept of
     "restyle what I already told you", so a formatting request just
     regenerates from the same source material.

  2. "reformat your answer please. But good answer!" -> came back with a
     totally unrelated GRAPH-mode answer about a track never mentioned in
     the conversation. Root cause: this exact phrasing wasn't on the
     phrase list from the FIRST graph-continuity fix (which forced ADVICE
     continuity only for a fixed set of literal phrases), so it fell
     through to the LLM classifier and got misread as GRAPH.

Both are really one design gap: there was no dedicated "meta request about
the prior turn" pathway. The fix adds one -- _is_reformat_request, checked
in chat() BEFORE classify_intent ever runs, and REFORMAT_SYSTEM, which
rewrites the ACTUAL STORED previous answer instead of re-deriving content.
The detector is combinatorial (style-verb + reference-token, or a small set
of strong standalone phrases) rather than an enumerated phrase list, so it
is not tied to exact wording the way the first attempt was.
"""
import pytest

from mentor import graph_context
from mentor.graph_context import GraphContext, MentorContext
from mentor.mentor import MusicMentor, REFORMAT_SYSTEM, GROUNDED_SYSTEM
from tests import mentor_fixtures as F
from tests.mentor_fixtures import write_session_on_disk


class FakeRAG:
    def __init__(self, hits=None, max_score=0.0):
        self.hits = hits if hits is not None else []
        self.max_score = max_score
        self.calls = []

    def retrieve(self, query, top_k=3):
        self.calls.append(query)
        return list(self.hits[:top_k]), self.max_score


@pytest.fixture()
def mentor(tmp_path, monkeypatch):
    """A MusicMentor with model/agent/similarity skipped -- just the
    reformat-interception logic, a real (empty) GraphContext so GRAPH
    never fires from map content, a scripted LLM, and a swappable FakeRAG.
    """
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setattr(graph_context, "SESSION_DIR", str(root))
    write_session_on_disk(root, "sess_empty", nodes=[], groups={})

    m = MusicMentor.__new__(MusicMentor)
    m.graph_ctx = GraphContext(session_id="sess_empty")
    m.agent = None
    m.similarity = None
    m._session_state = MentorContext()
    m.rag = FakeRAG()
    m._last_messages = None

    def fake_generate(messages, **kw):
        m._last_messages = messages
        return m._scripted
    m._generate = fake_generate
    m._scripted_intent = "ADVICE"
    m._scripted_mode = "GROUNDED"
    m._scripted = "restyled answer"

    def fake_classify_intent(question, state=None):
        # Should never be reached for a genuine reformat request -- the
        # interception must happen upstream in chat(). Return something
        # obviously wrong (GRAPH) so a bug that skips the interception is
        # loudly visible in the test failure instead of silently passing.
        return "GRAPH"
    m.classify_intent = fake_classify_intent

    def fake_classify_retrieval_mode(question, max_score, hits):
        return m._scripted_mode
    m.classify_retrieval_mode = fake_classify_retrieval_mode

    return m


def _seed_prior_advice_turn(mentor, state, answer="Release on Friday and pitch playlists a week ahead."):
    """Simulate a completed ADVICE turn so a reformat request has something
    to restyle."""
    state.last_intent = "ADVICE"
    state.record("user", "help me plan my release")
    state.record("assistant", answer)


# ---- _is_reformat_request: unit-level ----------------------------------------

def test_bullet_points_request_after_advice_is_reformat(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert mentor._is_reformat_request("format that in bullet points please", state)


def test_reformat_your_answer_after_advice_is_reformat(mentor):
    # The literal phrasing that broke the FIRST fix's phrase list.
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert mentor._is_reformat_request("reformat your answer please. But good answer!", state)


def test_shorten_that_is_reformat(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert mentor._is_reformat_request("can you shorten that for me?", state)


def test_tldr_is_reformat(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert mentor._is_reformat_request("tldr?", state)


def test_various_unlisted_reformat_phrasings_still_detected(mentor):
    # None of these exact strings appears in any fixed phrase list -- they
    # combine a style-verb with a reference token, which is the point.
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    for q in (
        "could you clean up your response a bit?",
        "polish that answer for me",
        "can you make your response clearer?",
        "please restate the answer more simply",
        "make what you said punchier",
    ):
        assert mentor._is_reformat_request(q, state), f"expected reformat: {q!r}"


def test_reformat_backs_off_when_no_prior_intent_was_advice(mentor):
    state = MentorContext(last_intent="GRAPH")
    state.record("user", "what's connected to tv off?")
    state.record("assistant", "Squabble Up and Not Like Us are closest.")
    assert not mentor._is_reformat_request("put that in bullet points", state)


def test_reformat_backs_off_with_no_history(mentor):
    state = MentorContext(last_intent="ADVICE")  # no recorded turns
    assert not mentor._is_reformat_request("shorten that", state)


def test_reformat_backs_off_on_explicit_map_pivot(mentor):
    # Continuation phrasing, but the question also names the map -- a real
    # pivot to GRAPH must still win.
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    q = "actually can you just summarize that -- what's on the map right now?"
    assert not mentor._is_reformat_request(q, state)


def test_non_reformat_question_after_advice_is_not_reformat(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert not mentor._is_reformat_request("what about pitching to blogs too?", state)


def test_empty_question_is_not_reformat(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    assert not mentor._is_reformat_request("", state)


# ---- chat(): full interception behavior --------------------------------------

def test_chat_bullet_points_followup_never_reaches_classify_intent(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    mentor._scripted = "- Release on Friday\n- Pitch playlists a week ahead"
    out = mentor.chat("format that in bullet points please", conversation_context=state)
    assert out == mentor._scripted
    assert mentor._last_messages[0]["content"] == REFORMAT_SYSTEM
    assert mentor._last_messages[0]["content"] != GROUNDED_SYSTEM


def test_chat_reformat_your_answer_literal_repro_never_becomes_graph(mentor):
    # Literal reproduction of the second shipped failure: this phrasing
    # must land on the reformat path, never on the GRAPH branch (which
    # would have required agent.run() and produced an unrelated answer).
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    mentor._scripted = "Restyled: release Friday, pitch playlists early."
    out = mentor.chat("reformat your answer please. But good answer!",
                       conversation_context=state)
    assert out == mentor._scripted
    assert mentor._last_messages[0]["content"] == REFORMAT_SYSTEM
    assert state.last_intent == "ADVICE"


def test_chat_reformat_prompt_includes_actual_previous_answer(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state, answer="Ship Friday. Pre-save link two weeks out.")
    mentor.chat("shorten that", conversation_context=state)
    user_msg = mentor._last_messages[1]["content"]
    assert "Ship Friday. Pre-save link two weeks out." in user_msg


def test_chat_reformat_sets_last_intent_advice_for_further_continuity(mentor):
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)
    mentor.chat("tldr?", conversation_context=state)
    assert state.last_intent == "ADVICE"


def test_chat_map_pivot_after_advice_still_reaches_graph_agent(mentor):
    # A genuine topic change to the map must NOT be swallowed by the
    # reformat interception, even with continuation-shaped wording.
    state = MentorContext()
    _seed_prior_advice_turn(mentor, state)

    class FakeAgent:
        def run(self, question, context=None):
            # The real MentorGraphTools agent sets context.last_intent
            # itself (mentor/agent.py); replicate that here.
            if context is not None:
                context.last_intent = "GRAPH"
            return {"answer": "TV Off sits near Squabble Up.", "intent": "neighbors",
                     "status": "ok", "observation": {"ok": True}}
    mentor.agent = FakeAgent()

    q = "actually summarize that -- what's connected to tv off on the map?"
    out = mentor.chat(q, conversation_context=state)
    assert out == "TV Off sits near Squabble Up."
    assert state.last_intent == "GRAPH"
