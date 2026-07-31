"""The Evidence envelope: every one of the 8 intents (7 pre-existing +
connections) must return the same outer shape —
``{intent, subject, scope, findings, contrasts, provenance}`` — layered
additively on top of whatever fields that tool already returned. This file
asserts the shape holds across the whole surface, on both the ok=True and
ok=False paths, and that the 4-tier band vocabulary
(near-identical/close/related/distant) is used consistently wherever a tool
cites a concrete similarity relationship.
"""
import pytest

from mentor.graph_tools import SIMILARITY_BANDS
from tests.mentor_fixtures import make_context, make_tools

ENVELOPE_KEYS = {"intent", "subject", "scope", "findings", "contrasts", "provenance"}

LINKS = [
    {"source": "n1", "target": "n2", "score": 91.0, "value": 0.99, "kind": "qq"},
    {"source": "n4", "target": "n5", "score": 85.0, "value": 0.98, "kind": "qq"},
]


@pytest.fixture()
def tools(tmp_path):
    return make_tools(tmp_path, links=LINKS)


def _assert_envelope(obs, expected_intent):
    missing = ENVELOPE_KEYS - set(obs.keys())
    assert not missing, f"missing envelope keys for {expected_intent}: {missing}"
    assert obs["intent"] == expected_intent
    assert isinstance(obs["findings"], list)
    assert isinstance(obs["contrasts"], list)
    assert isinstance(obs["provenance"], dict)
    assert obs["scope"] in (None, "map", "library", "persisted", "corpus")


# ---- envelope present on every ok=True call across all 8 intents --------------
def test_envelope_shape_holds_on_success_across_all_intents(tools):
    ctx = make_context(selected_node_id="n1")
    calls = {
        "inspect": lambda: tools.inspect_graph(context=ctx),
        "resolve": lambda: tools.resolve_anchor("me", context=ctx),
        "connections": lambda: tools.connections("me", context=ctx),
        "neighbors": lambda: tools.neighbors("me", context=ctx),
        "compare": lambda: tools.compare("Halo", "Burial", context=make_context()),
        "bridge": lambda: tools.bridge("me", "Halo", context=ctx),
        "explore": lambda: tools.explore_cluster("me", context=ctx),
        "explain": lambda: tools.explain_node("me", context=make_context(selected_node_id="n2")),
    }
    for intent, call in calls.items():
        obs = call()
        assert obs["ok"], f"{intent} unexpectedly failed: {obs}"
        _assert_envelope(obs, intent)


# ---- envelope present on every ok=False (honest-failure) path too -------------
def test_envelope_shape_holds_on_failure_across_all_intents(tools):
    empty_ctx = make_context()  # nothing selected, no history -> every anchor-based
                                 # tool takes its "me" honest-failure path
    calls = {
        "resolve": lambda: tools.resolve_anchor("me", context=empty_ctx),
        "connections": lambda: tools.connections("me", context=empty_ctx),
        "neighbors": lambda: tools.neighbors("me", context=empty_ctx),
        "compare": lambda: tools.compare("me", "Halo", context=empty_ctx),
        "bridge": lambda: tools.bridge("me", "Halo", context=empty_ctx),
        "explore": lambda: tools.explore_cluster("me", context=empty_ctx),
        "explain": lambda: tools.explain_node("me", context=empty_ctx),
    }
    for intent, call in calls.items():
        obs = call()
        assert not obs["ok"], f"{intent} unexpectedly succeeded: {obs}"
        _assert_envelope(obs, intent)
        assert obs["subject"] is None


# ---- subject shape: single dict for one-anchor tools, pair for two-anchor -----
def test_subject_is_single_dict_for_one_anchor_tools(tools):
    ctx = make_context(selected_node_id="n1")
    for obs in (tools.resolve_anchor("me", context=ctx),
                tools.connections("me", context=ctx),
                tools.neighbors("me", context=ctx),
                tools.explore_cluster("me", context=ctx),
                tools.explain_node("me", context=ctx)):
        assert obs["ok"]
        assert isinstance(obs["subject"], dict)
        assert {"label", "kind", "id", "on_map"} <= set(obs["subject"].keys())


def test_subject_is_a_pair_for_two_anchor_tools(tools):
    ctx = make_context(selected_node_id="n1")
    compare_obs = tools.compare("Halo", "Burial", context=make_context())
    bridge_obs = tools.bridge("me", "Halo", context=ctx)
    for obs in (compare_obs, bridge_obs):
        assert obs["ok"]
        assert isinstance(obs["subject"], list) and len(obs["subject"]) == 2
        for s in obs["subject"]:
            assert {"label", "kind", "id", "on_map"} <= set(s.keys())


# ---- band vocabulary is exactly the 4-tier scheme, applied everywhere a ------
# ---- tool scores a concrete pair (connections/neighbors/compare/bridge/explain)
@pytest.mark.parametrize("intent_call", [
    lambda tools, ctx: tools.connections("me", context=ctx),
    lambda tools, ctx: tools.neighbors("me", context=ctx),
    lambda tools, ctx: tools.explain_node("me", context=ctx),
])
def test_bands_are_from_the_4tier_vocabulary(tools, intent_call):
    ctx = make_context(selected_node_id="n1")
    obs = intent_call(tools, ctx)
    assert obs["ok"]
    assert obs["findings"], "fixture should produce at least one finding"
    for f in obs["findings"]:
        assert f["band"] in SIMILARITY_BANDS
        assert isinstance(f["score"], float)
        assert 0.0 <= f["score"] <= 100.0


def test_compare_and_bridge_also_use_the_4tier_vocabulary(tools):
    ctx = make_context(selected_node_id="n1")
    compare_obs = tools.compare("Halo", "Burial", context=make_context())
    bridge_obs = tools.bridge("me", "Halo", context=ctx)
    assert compare_obs["findings"][0]["band"] in SIMILARITY_BANDS
    for f in bridge_obs["findings"]:
        assert f["band"] in SIMILARITY_BANDS


def test_similarity_bands_constant_is_the_4_agreed_tiers():
    assert SIMILARITY_BANDS == ("near-identical", "close", "related", "distant")


# ---- additive guarantee: pre-existing fields survive the envelope wrap --------
def test_envelope_is_additive_preserves_preexisting_fields(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.neighbors("me", context=ctx)
    # the tool's own pre-envelope fields must still be present, unchanged in
    # shape, alongside the new envelope keys
    assert "neighbors" in obs and "anchor" in obs and "territory" in obs and "traits" in obs
    assert obs["findings"] == [
        {"artist": n["artist"], "song": n["song"], "relationship": n["relationship"],
         "band": n["band"], "score": n["score"]}
        for n in obs["neighbors"]
    ]
