"""The mentor's 7-tool graph surface, exercised the way the UI uses it:
songs placed on a synthetic map, a node selected (or not), questions asked.

No GPU, no LLM, no corpus — see tests/mentor_fixtures.py.
"""
import re

import pytest

from tests.mentor_fixtures import DARK, POP, make_context, make_tools


@pytest.fixture()
def tools(tmp_path):
    return make_tools(tmp_path)


# ---- inspect_graph -----------------------------------------------------------
def test_inspect_reports_map_state(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.inspect_graph(context=ctx)
    assert obs["ok"] and obs["visible_nodes"] == 5
    territories = {t["territory"]: t["songs"] for t in obs["territories"]}
    assert territories == {DARK: 3, POP: 2}
    assert obs["selected_node"]["name"] == "First Demo"
    assert obs["selected_node"]["traits"] == ["dark", "bass"]
    assert "Late Night Drive" in obs["groups"]


def test_inspect_empty_map(tmp_path):
    tools = make_tools(tmp_path, nodes=[], groups={})
    obs = tools.inspect_graph(context=make_context())
    assert obs["ok"] and obs["visible_nodes"] == 0
    assert obs["selected_node"] is None


# ---- resolve_anchor: selected node > visible nodes > corpus -------------------
def test_me_resolves_to_selected_node(tools):
    ctx = make_context(selected_node_id="n2")
    obs = tools.resolve_anchor("me", context=ctx)
    assert obs["ok"] and obs["type"] == "graph_node"
    assert obs["artist"] == "Burial" and obs["name"] == "Burial - Archangel"


def test_this_song_without_selection_falls_back_to_last_anchor(tools):
    ctx = make_context(selected_node_id=None, last_anchor="Halo")
    obs = tools.resolve_anchor("this song", context=ctx)
    assert obs["ok"] and obs["artist"] == "Beyonce"


def test_no_selection_and_no_history_is_honest(tools):
    obs = tools.resolve_anchor("me", context=make_context())
    assert not obs["ok"]
    assert "selected" in obs["message"].lower() or "click a node" in obs["message"].lower()


def test_visible_node_beats_corpus(tools):
    # "Halo" exists on the map; resolution must prefer the graph node.
    obs = tools.resolve_anchor("Halo", context=make_context())
    assert obs["ok"] and obs["type"] == "graph_node"


def test_corpus_fallback_for_offscreen_name(tools):
    obs = tools.resolve_anchor("Rihanna", context=make_context())
    assert obs["ok"] and obs["type"] == "library_track"


def test_unknown_artist_is_honest(tools):
    obs = tools.resolve_anchor("Taylor Swift", context=make_context())
    assert not obs["ok"]
    assert "isn't on your map" in obs["message"]


# ---- neighbors -----------------------------------------------------------------
def test_neighbors_uses_selection_and_explains_why(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.neighbors("me", context=ctx)
    assert obs["ok"] and obs["scope"] == "map"
    names = [n["song"] for n in obs["neighbors"]]
    assert "First Demo" not in names          # never its own neighbour
    assert names[0] == "Archangel"            # nearest dark-territory mate
    top = obs["neighbors"][0]
    assert top["relationship"]                # a reason, not a number
    assert "dark" in top["relationship"]      # shared micro-genre tag
    assert not re.search(r"\d\.\d", top["relationship"])  # no raw cosines


def test_neighbors_falls_back_to_library_when_map_empty(tmp_path):
    tools = make_tools(tmp_path, nodes=[], groups={})
    obs = tools.neighbors("Rihanna", context=make_context())
    assert obs["ok"] and obs["scope"] == "library"
    assert obs["neighbors"]


# ---- compare ---------------------------------------------------------------------
def test_compare_names_both_territories(tools):
    obs = tools.compare("Halo", "Burial", context=make_context())
    assert obs["ok"]
    assert DARK in obs["similarity_summary"] and POP in obs["similarity_summary"]
    diffs = obs["differences"]
    assert any("pop" in v for v in diffs.values())
    assert any("dark" in v for v in diffs.values())


def test_compare_same_territory_shares_traits(tools):
    obs = tools.compare("First Demo", "Archangel", context=make_context())
    assert obs["ok"]
    assert "same" in obs["similarity_summary"]
    assert "dark" in obs["shared_traits"]


# ---- bridge -----------------------------------------------------------------------
def test_bridge_tracks_carry_a_why(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.bridge("me", "Halo", context=ctx)
    assert obs["ok"] and obs["bridge_tracks"]
    for t in obs["bridge_tracks"]:
        assert t["why"]
    anchors = {"First Demo", "Halo"}
    assert not anchors & {t["song"] for t in obs["bridge_tracks"]}


# ---- explore_cluster ------------------------------------------------------------------
def test_explore_points_at_the_adjacent_territory(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.explore_cluster("me", context=ctx)
    assert obs["ok"]
    assert obs["cluster"] == DARK
    assert obs["adjacent_territory"] == POP
    assert POP in obs["recommended_direction"]
    assert obs["nearby_artists"]


# ---- explain_node ----------------------------------------------------------------------
def test_explain_selected_node(tools):
    ctx = make_context(selected_node_id="n2")
    obs = tools.explain_node("me", context=ctx)
    assert obs["ok"]
    assert obs["song"] and obs["artist"] == "Burial"
    assert obs["territory"] == DARK
    assert obs["traits"] == ["dark", "garage"]
    assert obs["neighbors"]
    assert obs["position_summary"].startswith("It sits here because")


# ---- dispatcher -------------------------------------------------------------------------
def test_execute_validates_intent_and_clamps_k(tools):
    ctx = make_context(selected_node_id="n1")
    obs = tools.execute("neighbors", {"anchor": "me", "k": 999}, context=ctx)
    assert obs["ok"] and len(obs["neighbors"]) <= 20
    bad = tools.execute("place_song", {}, context=ctx)
    assert not bad["ok"] and bad["error"] == "unknown_intent"
