"""The 3-mode retrieval router (GROUNDED / SYNTHESIS / NO_RETRIEVAL).

The shipped failure this replaces: "ask for advice on release" got grounded
advice (RAG scored well), but a brainstorm/plan-drafting follow-up on the
SAME on-topic subject either got a verbatim knowledgebase quote (if some
passage happened to score just high enough) or an "I'm here for musicians'
questions" redirect (if nothing did) — because the old gate was a single
`in_scope = intent=="ADVICE" and max_score >= SCOPE_THRESHOLD` check with
only one failure branch, shared with genuinely off-topic questions.

These tests build a MusicMentor without loading the 7B model or the RAG
index (via __new__), wire in a scripted LLM and a stub retriever, and prove:

  * explicit brainstorm/plan/draft language forces SYNTHESIS outright, even
    if the LLM classifier would say something else
  * a direct, settleable question with a strong passage match is GROUNDED
  * an on-topic question with only a weak/no passage match still gets
    answered (NO_RETRIEVAL) rather than redirected off-topic
  * chat()'s ADVICE branch dispatches each mode to a genuinely different
    system prompt / answer shape, and none of them is the OFFTOPIC redirect
"""
import pytest

from mentor import graph_context
from mentor.graph_context import GraphContext, MentorContext
from mentor.mentor import (
    MusicMentor,
    GROUNDED_SYSTEM,
    SYNTHESIS_SYSTEM,
    NO_RETRIEVAL_SYSTEM,
    UNGROUNDED_SYSTEM,
    SCOPE_THRESHOLD,
)
from tests import mentor_fixtures as F
from tests.mentor_fixtures import write_session_on_disk


class FakeRAG:
    """Stub retriever: returns whatever the test pre-loads, regardless of query."""

    def __init__(self, hits=None, max_score=0.0):
        self.hits = hits if hits is not None else []
        self.max_score = max_score
        self.calls = []

    def retrieve(self, query, top_k=3):
        self.calls.append(query)
        return list(self.hits[:top_k]), self.max_score


def _hit(score, advice="Ship the single on a Friday and pitch playlists a week early."):
    return {"score": score, "question": "", "advice": advice, "source": "test"}


@pytest.fixture()
def mentor(tmp_path, monkeypatch):
    """A MusicMentor with model/agent/similarity skipped — just the router,
    a real GraphContext (empty map, so GRAPH never fires), a scripted LLM,
    and a swappable FakeRAG."""
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

    def fake_classify_intent(question, state=None):
        return m._scripted_intent
    m.classify_intent = fake_classify_intent

    # classify_retrieval_mode is the thing under test in most cases below,
    # so leave the REAL implementation wired -- only _generate is faked, and
    # the deterministic phrase pre-filter runs for real.
    orig_classify_mode = MusicMentor.classify_retrieval_mode.__get__(m)
    m._scripted = "GROUNDED"  # what the retrieval-mode LLM call returns by default

    return m


# ---- classify_retrieval_mode: deterministic pre-filter -----------------------

def test_explicit_brainstorm_language_is_synthesis_even_if_llm_disagrees(mentor):
    mentor._scripted = "GROUNDED"     # LLM would say otherwise
    mode = mentor.classify_retrieval_mode(
        "Help me brainstorm ideas for my album rollout.", max_score=0.9, hits=[_hit(0.9)])
    assert mode == "SYNTHESIS"


def test_draft_a_plan_is_synthesis(mentor):
    mentor._scripted = "GROUNDED"
    mode = mentor.classify_retrieval_mode(
        "Draft a release plan for my next single.", max_score=0.4, hits=[])
    assert mode == "SYNTHESIS"


def test_help_me_plan_out_next_months_is_synthesis(mentor):
    mentor._scripted = "NO_RETRIEVAL"
    mode = mentor.classify_retrieval_mode(
        "Can you help me plan out my next few months as an artist?",
        max_score=0.2, hits=[])
    assert mode == "SYNTHESIS"


# ---- classify_retrieval_mode: LLM decides the non-obvious cases --------------

def test_direct_settleable_question_is_grounded(mentor):
    mentor._scripted = "GROUNDED"
    mode = mentor.classify_retrieval_mode(
        "How do I stop clipping when I master my tracks?",
        max_score=0.8, hits=[_hit(0.8)])
    assert mode == "GROUNDED"


def test_open_ended_personal_question_is_no_retrieval(mentor):
    mentor._scripted = "NO_RETRIEVAL"
    mode = mentor.classify_retrieval_mode(
        "Should I keep making music even though I'm not blowing up yet?",
        max_score=0.1, hits=[])
    assert mode == "NO_RETRIEVAL"


def test_retrieval_mode_llm_gets_a_dedicated_prompt_not_the_intent_prompt(mentor):
    mentor._scripted = "GROUNDED"
    mentor.classify_retrieval_mode("How do I master a track?", max_score=0.7, hits=[_hit(0.7)])
    assert mentor._last_messages is not None
    assert mentor._last_messages[0]["role"] == "system"
    assert "GROUNDED" in mentor._last_messages[0]["content"]
    assert "SYNTHESIS" in mentor._last_messages[0]["content"]
    assert "NO_RETRIEVAL" in mentor._last_messages[0]["content"]


def test_retrieval_mode_falls_back_safely_never_to_a_redirect(mentor):
    # LLM returns garbage both tries -> safety-net fallback, never crashes,
    # never returns anything outside the valid mode set.
    mentor._scripted = "gibberish, not a real label"
    mode = mentor.classify_retrieval_mode("How do I improve my mixing?",
                                          max_score=0.9, hits=[_hit(0.9)])
    assert mode in ("GROUNDED", "SYNTHESIS", "NO_RETRIEVAL")
    mode2 = mentor.classify_retrieval_mode("How do I improve my mixing?",
                                           max_score=0.1, hits=[])
    assert mode2 in ("GROUNDED", "SYNTHESIS", "NO_RETRIEVAL")


# ---- chat(): end-to-end dispatch, three genuinely different answer shapes ----

def test_chat_grounded_defers_to_strong_passage(mentor):
    mentor.rag = FakeRAG(hits=[_hit(0.85, "Release on a Friday and pitch playlists early.")],
                         max_score=0.85)
    mentor._scripted_intent = "ADVICE"
    mentor._scripted = "GROUNDED"
    out = mentor.chat("What's the best day of the week to release a single?")
    assert mentor._last_messages[0]["content"] == GROUNDED_SYSTEM
    assert "Release on a Friday" in mentor._last_messages[-1]["content"]
    assert out == "GROUNDED"   # fake_generate echoes m._scripted


def test_chat_synthesis_never_quotes_and_never_redirects(mentor):
    mentor.rag = FakeRAG(hits=[_hit(0.55, "Some tangentially related tip.")], max_score=0.55)
    mentor._scripted_intent = "ADVICE"
    out = mentor.chat("Help me brainstorm ideas for my album rollout.")
    assert mentor._last_messages[0]["content"] == SYNTHESIS_SYSTEM
    assert mentor._last_messages[0]["content"] != UNGROUNDED_SYSTEM
    # passages are offered as inspiration only, never as a "defer to this" quote
    assert "defer to this" not in mentor._last_messages[-1]["content"]


def test_chat_synthesis_works_even_with_zero_hits(mentor):
    mentor.rag = FakeRAG(hits=[], max_score=0.05)
    mentor._scripted_intent = "ADVICE"
    out = mentor.chat("Draft a 4-week plan for releasing my EP.")
    assert mentor._last_messages[0]["content"] == SYNTHESIS_SYSTEM
    assert out != UNGROUNDED_SYSTEM


def test_chat_no_retrieval_answers_instead_of_redirecting(mentor):
    # On-topic ADVICE, but nothing in the corpus is a good match -- this
    # must NOT fall through to the "that's not my lane" redirect.
    mentor.rag = FakeRAG(hits=[], max_score=0.1)
    mentor._scripted_intent = "ADVICE"
    mentor._scripted = "NO_RETRIEVAL"
    out = mentor.chat("Do you think my sound has matured this year?")
    assert mentor._last_messages[0]["content"] == NO_RETRIEVAL_SYSTEM
    assert mentor._last_messages[0]["content"] != UNGROUNDED_SYSTEM
    assert out == "NO_RETRIEVAL"


def test_chat_offtopic_still_redirects_as_before(mentor):
    mentor._scripted_intent = "OFFTOPIC"
    mentor._scripted = "not my lane, whatever the model says"
    out = mentor.chat("what's the weather tomorrow")
    assert mentor._last_messages[0]["content"] == UNGROUNDED_SYSTEM


def test_chat_grounded_mode_with_weak_hits_falls_back_to_no_retrieval_not_redirect(mentor):
    # classify_retrieval_mode said GROUNDED, but none of the hits actually
    # clear SCOPE_THRESHOLD (e.g. the LLM misjudged) -- chat() must still
    # answer rather than quote a weak, irrelevant-scoring passage.
    mentor.rag = FakeRAG(hits=[_hit(SCOPE_THRESHOLD - 0.2, "weak match")],
                         max_score=SCOPE_THRESHOLD - 0.2)
    mentor._scripted_intent = "ADVICE"
    mentor._scripted = "GROUNDED"
    out = mentor.chat("How should I think about my sound evolving?")
    assert mentor._last_messages[0]["content"] == NO_RETRIEVAL_SYSTEM


# ---- SYNTHESIS takes priority over a coincidentally-strong passage match -----

def test_explicit_brainstorm_wins_even_with_a_strong_passage_match(mentor):
    # The user explicitly asked to brainstorm; even if some passage happens
    # to score above SCOPE_THRESHOLD, they get synthesis, not a single
    # quoted tip standing in for a real plan.
    mentor.rag = FakeRAG(hits=[_hit(0.95, "Some strongly-matching single tip.")],
                         max_score=0.95)
    mentor._scripted_intent = "ADVICE"
    mentor._scripted = "GROUNDED"     # LLM/legacy signal would have said GROUNDED
    out = mentor.chat("Help me brainstorm ideas for my next release.")
    assert mentor._last_messages[0]["content"] == SYNTHESIS_SYSTEM


def test_draft_a_plan_with_strong_hits_still_synthesizes(mentor):
    mentor.rag = FakeRAG(hits=[_hit(0.9, "One specific tip.")], max_score=0.9)
    mentor._scripted_intent = "ADVICE"
    mode = mentor.classify_retrieval_mode(
        "Draft a plan for promoting my new EP over the next month.",
        max_score=0.9, hits=[_hit(0.9)])
    assert mode == "SYNTHESIS"


# ---- conversational state: all three ADVICE modes record last_intent -------

@pytest.mark.parametrize("mode,scripted_intent,rag_hits,rag_max", [
    ("GROUNDED", "ADVICE", [_hit(0.9)], 0.9),
    ("NO_RETRIEVAL", "ADVICE", [], 0.05),
])
def test_advice_modes_set_last_intent_advice_for_followup_continuity(
        mentor, mode, scripted_intent, rag_hits, rag_max):
    mentor.rag = FakeRAG(hits=rag_hits, max_score=rag_max)
    mentor._scripted_intent = scripted_intent
    mentor._scripted = mode
    state = MentorContext()
    mentor.chat("How do I get more plays on my new track?", conversation_context=state)
    assert state.last_intent == "ADVICE"


def test_synthesis_mode_sets_last_intent_advice_for_followup_continuity(mentor):
    mentor.rag = FakeRAG(hits=[], max_score=0.0)
    mentor._scripted_intent = "ADVICE"
    state = MentorContext()
    mentor.chat("Brainstorm some ideas for my next music video.", conversation_context=state)
    assert state.last_intent == "ADVICE"


# ---- the literal scenario from the user's report -----------------------------

def test_release_advice_then_brainstorm_followup_both_land_on_topic(mentor):
    # "if you ask for advice on release you get advice from knowledgebase but
    # then if you try to brainstorm or draft a plan you either get direct
    # knowledgebase quotes or redirection messages" -- reproduce both turns.
    mentor.rag = FakeRAG(hits=[_hit(0.88, "Release on Friday, pitch playlists a week ahead.")],
                         max_score=0.88)
    mentor._scripted_intent = "ADVICE"
    mentor._scripted = "GROUNDED"
    state = MentorContext()
    out1 = mentor.chat("What's the best strategy for releasing my next single?",
                       conversation_context=state)
    assert mentor._last_messages[0]["content"] == GROUNDED_SYSTEM

    # Second turn: brainstorm/plan follow-up on the same topic. Must not be
    # a verbatim quote and must not be an off-topic redirect.
    out2 = mentor.chat("Ok, can you help me brainstorm a full release plan for it?",
                       conversation_context=state)
    assert mentor._last_messages[0]["content"] == SYNTHESIS_SYSTEM
    assert mentor._last_messages[0]["content"] != UNGROUNDED_SYSTEM
