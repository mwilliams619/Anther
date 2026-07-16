"""MentorAgent pipeline: LLM interprets intent -> tools observe -> LLM narrates.

The LLM is a scripted stand-in, so these tests pin the *harness* behaviour:
context resolution against the selected node, follow-ups over the previous
observation, honest answers for unknown anchors, and the hallucination guard
on the narrated answer.
"""
import pytest

from mentor.agent import MentorAgent
from tests.mentor_fixtures import make_context, make_tools


class ScriptedLLM:
    """Returns canned intent JSON for classify calls and (optionally) a canned
    narration for compose calls; records every call for assertions."""

    def __init__(self, intent_json, narration=None):
        self.intent_json = intent_json
        self.narration = narration
        self.classify_calls = []
        self.narrate_calls = []

    def __call__(self, messages, **kwargs):
        last = messages[-1]["content"]
        if "FACTS:" in last:
            self.narrate_calls.append(messages)
            return self.narration or ""
        self.classify_calls.append(messages)
        return self.intent_json


@pytest.fixture()
def tools(tmp_path):
    return make_tools(tmp_path)


# ---- context resolution -------------------------------------------------------
def test_what_do_i_sound_like_uses_selected_node(tools):
    llm = ScriptedLLM('{"intent":"neighbors","args":{"anchor":"me"}}')
    agent = MentorAgent(tools, llm)
    ctx = make_context(selected_node_id="n1")
    r = agent.run("what do I sound like?", context=ctx)
    assert r["status"] == "ok" and r["intent"] == "neighbors"
    assert r["observation"]["anchor"] == "Embolo - First Demo"
    assert "Archangel" in r["answer"]


def test_classifier_sees_the_selection(tools):
    llm = ScriptedLLM('{"intent":"explain","args":{"anchor":"me"}}')
    agent = MentorAgent(tools, llm)
    ctx = make_context(selected_node_id="n2")
    agent.run("why is this node here?", context=ctx)
    prompt = llm.classify_calls[0][-1]["content"]
    assert "[Selected node: Burial - Archangel]" in prompt


# ---- bridge ----------------------------------------------------------------------
def test_bridge_between_two_named_anchors(tools):
    llm = ScriptedLLM('{"intent":"bridge","args":{"anchor_a":"Burial","anchor_b":"Halo"}}')
    agent = MentorAgent(tools, llm)
    r = agent.run("what sits between Burial and Halo?", context=make_context())
    assert r["status"] == "ok" and r["intent"] == "bridge"
    assert r["observation"]["bridge_tracks"]
    assert all(t["why"] for t in r["observation"]["bridge_tracks"])


# ---- follow-up ---------------------------------------------------------------------
def test_followup_reuses_previous_observation(tools):
    ctx = make_context(selected_node_id="n1")
    first = MentorAgent(tools, ScriptedLLM('{"intent":"neighbors","args":{"anchor":"me"}}'))
    r1 = first.run("what do I sound like?", context=ctx)
    assert r1["status"] == "ok"

    followup_llm = ScriptedLLM('{"intent":"followup","args":{}}')
    second = MentorAgent(tools, followup_llm)
    r2 = second.run("which ones?", context=ctx)
    assert r2["status"] == "ok"
    assert r2["observation"] is r1["observation"]     # no new tool call
    assert "Archangel" in r2["answer"]


def test_followup_with_no_history_is_honest(tools):
    agent = MentorAgent(tools, ScriptedLLM('{"intent":"followup","args":{}}'))
    r = agent.run("which ones?", context=make_context())
    assert r["status"] == "no_prior"
    assert "haven't mapped anything" in r["answer"]


# ---- unknown artist -----------------------------------------------------------------
def test_unknown_artist_gets_honest_unresolved_answer(tools):
    llm = ScriptedLLM('{"intent":"neighbors","args":{"anchor":"Taylor Swift"}}')
    agent = MentorAgent(tools, llm)
    r = agent.run("what about Taylor Swift?", context=make_context())
    assert r["status"] == "anchor_not_resolved"
    assert "isn't on your map" in r["answer"]
    assert "Taylor Swift" in r["answer"]


# ---- selection questions are deterministic + honest (never hallucinated) -------------
def _sel_tools(tmp_path):
    from tests.mentor_fixtures import make_tools
    return make_tools(tmp_path)


def test_selected_node_question_identifies_node(tmp_path):
    tools = _sel_tools(tmp_path)
    # LLM would misclassify to inspect, but the selection guard runs first.
    llm = ScriptedLLM('{"intent":"inspect","args":{}}')
    agent = MentorAgent(tools, llm)
    ctx = make_context(selected_node_id="n2")
    r = agent.run("what is the selected node?", context=ctx)
    assert r["intent"] == "explain" and r["status"] == "ok"
    assert r["observation"]["artist"] == "Burial"
    assert len(llm.classify_calls) == 0            # LLM never consulted


def test_what_song_am_i_selecting_no_selection_is_honest(tmp_path):
    tools = _sel_tools(tmp_path)
    # Hostile LLM that would happily hallucinate a song — must not get the chance.
    llm = ScriptedLLM('{"intent":"neighbors","args":{"anchor":"me"}}',
                      narration="You're listening to Imagine Dragons - Radioactive!")
    agent = MentorAgent(tools, llm)
    r = agent.run("what song am I selecting on the map?", context=make_context())
    assert r["status"] == "no_selection"
    assert "Imagine Dragons" not in r["answer"]
    assert "selected" in r["answer"].lower()
    assert len(llm.narrate_calls) == 0


# ---- terse connections: no per-track genre lecture -----------------------------------
def test_neighbors_relationship_is_terse(tmp_path):
    tools = _sel_tools(tmp_path)
    obs = tools.neighbors("me", context=make_context(selected_node_id="n1"))
    for n in obs["neighbors"]:
        rel = n["relationship"]
        # a short shared-trait note or nothing — never a 3-part cluster path
        assert "/" not in rel and "territory" not in rel
        assert len(rel) < 40


# ---- UI state ------------------------------------------------------------------------
def test_why_is_this_node_here_uses_selection(tools):
    llm = ScriptedLLM('{"intent":"explain","args":{"anchor":"me"}}')
    agent = MentorAgent(tools, llm)
    ctx = make_context(selected_node_id="n2")
    r = agent.run("why is this node here?", context=ctx)
    assert r["status"] == "ok" and r["intent"] == "explain"
    assert r["observation"]["song"] == "Archangel"
    assert "It sits here because" in r["answer"]


# ---- hallucination guard ----------------------------------------------------------------
def test_narration_with_invented_name_is_discarded(tools):
    llm = ScriptedLLM(
        '{"intent":"neighbors","args":{"anchor":"me"}}',
        narration="You sound just like Imagine Dragons Radioactive — lean into that!",
    )
    agent = MentorAgent(tools, llm)
    r = agent.run("what do I sound like?", context=make_context(selected_node_id="n1"))
    assert "Imagine Dragons" not in r["answer"]       # guard rejected the voiced text
    assert "Archangel" in r["answer"]                 # fell back to the facts readout


def test_grounded_narration_is_kept(tools):
    llm = ScriptedLLM(
        '{"intent":"neighbors","args":{"anchor":"me"}}',
        narration="Your closest company is Burial's Archangel — that dark lane is yours.",
    )
    agent = MentorAgent(tools, llm)
    r = agent.run("what do I sound like?", context=make_context(selected_node_id="n1"))
    assert r["answer"].startswith("Your closest company is Burial")


# ---- classifier robustness ------------------------------------------------------------------
def test_junk_classifier_output_gets_honest_clarify(tools):
    llm = ScriptedLLM("I think you should use the neighbors tool maybe?")
    agent = MentorAgent(tools, llm)
    r = agent.run("hmm?", context=make_context())
    assert r["status"] == "unclassified"
    assert len(llm.classify_calls) == 2               # one retry, then honesty


def test_classifier_retry_recovers(tools):
    class FlakyLLM(ScriptedLLM):
        def __call__(self, messages, **kwargs):
            if not self.classify_calls:
                self.classify_calls.append(messages)
                return "not json"
            return super().__call__(messages, **kwargs)

    llm = FlakyLLM('{"intent":"inspect","args":{}}')
    agent = MentorAgent(tools, llm)
    r = agent.run("what's on my map?", context=make_context())
    assert r["status"] == "ok" and r["intent"] == "inspect"


# ---- session state ----------------------------------------------------------------------------
def test_last_anchor_survives_deselection(tools):
    ctx = make_context(selected_node_id="n1")
    agent = MentorAgent(tools, ScriptedLLM('{"intent":"neighbors","args":{"anchor":"me"}}'))
    agent.run("what do I sound like?", context=ctx)
    assert ctx.last_anchor == "Embolo - First Demo"

    ctx.selected_node_id = None                        # user clicks the background
    agent2 = MentorAgent(tools, ScriptedLLM('{"intent":"explain","args":{"anchor":"me"}}'))
    r = agent2.run("why am I placed there?", context=ctx)
    assert r["status"] == "ok"
    assert r["observation"]["song"] == "First Demo"


def test_context_reset_clears_everything(tools):
    ctx = make_context(selected_node_id="n1", last_anchor="x", last_intent="neighbors")
    ctx.record("user", "hello")
    ctx.reset()
    assert ctx.selected_node_id is None and ctx.last_anchor is None
    assert ctx.last_observation is None and not ctx.history
