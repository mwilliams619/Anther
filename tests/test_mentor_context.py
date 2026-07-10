"""Regression tests for mentor chat context continuity."""
from collections import deque

import torch

from mentor.mentor import ConversationState, MusicMentor


class _FakeEncoding(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None, return_dict=None):
        return _FakeEncoding()

    def decode(self, tokens, skip_special_tokens=True):
        return "stub response"


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


class _FakeReact:
    def __init__(self):
        self.calls = []

    def run(self, question, max_steps=3, verbose=False, context=None):
        self.calls.append({"question": question, "context": context, "verbose": verbose})
        return {
            "answer": "bridge answer",
            "calls": [{"tool": "bridge", "args": {"anchor_a": "me", "anchor_b": "Neroptik"}}],
            "observations": [
                {
                    "tool": "bridge",
                    "ok": True,
                    "neighbors": [
                        {"artist": "Artist A", "name": "Track A"},
                        {"artist": "Artist B", "name": "Track B"},
                    ],
                }
            ],
            "graph_state": {"anchor": "Neroptik"},
        }


def _fake_mentor():
    mentor = MusicMentor.__new__(MusicMentor)
    mentor.device = "cpu"
    mentor.tok = _FakeTokenizer()
    mentor.model = _FakeModel()
    mentor.rag = None
    mentor.anther = None
    mentor.react = _FakeReact()
    mentor._session_state = ConversationState()
    return mentor


def test_graph_followup_uses_cached_song_list():
    mentor = _fake_mentor()
    state = ConversationState()
    state.last_intent = "GRAPH"
    state.last_graph_result = {
        "observations": [
            {
                "tool": "bridge",
                "ok": True,
                "neighbors": [
                    {"artist": "Artist A", "name": "Track A"},
                    {"artist": "Artist B", "name": "Track B"},
                ],
            }
        ]
    }

    mentor.classify_intent = lambda question: "OFFTOPIC"

    out = mentor.chat("what specific songs?", conversation_context=state)

    assert "Specific songs to check next" in out
    assert "Artist A - Track A" in out
    assert "Artist B - Track B" in out
    assert state.last_intent == "GRAPH"
    assert len(state.history) == 2


def test_short_followup_carries_last_intent():
    mentor = _fake_mentor()
    state = ConversationState()
    state.last_intent = "GRAPH"
    state.last_graph_result = {"observations": []}

    mentor.classify_intent = lambda question: "OFFTOPIC"

    out = mentor.chat("what now", conversation_context=state)

    assert out == "bridge answer"
    assert mentor.react.calls[-1]["question"] == "what now"
    assert mentor.react.calls[-1]["context"] is state
    assert state.last_intent == "GRAPH"


def test_reset_clears_session_state():
    mentor = _fake_mentor()
    state = ConversationState(
        history=deque([{"role": "user", "content": "hello"}]),
        last_intent="GRAPH",
        last_graph_result={"observations": [{"tool": "bridge"}]},
        last_graph_observations=[{"tool": "bridge"}],
        last_anchor="Neroptik",
    )

    out = mentor.chat("/reset", conversation_context=state)

    assert "Session reset" in out
    assert list(state.history) == []
    assert state.last_intent is None
    assert state.last_graph_result is None
    assert state.last_graph_observations == []
    assert state.last_anchor is None


def test_generate_omits_sampling_args_when_deterministic():
    mentor = _fake_mentor()

    out = mentor._generate([{"role": "user", "content": "hello"}], temperature=0.0)

    assert out == "stub response"
    kwargs = mentor.model.calls[-1]
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["max_new_tokens"] == 320
