"""``connections`` reads persisted map edges only; ``neighbors`` computes a
fresh embedding kNN. They answer different questions and can legitimately
disagree — that's the whole point of splitting them (see graph_tools.py's
module docstring and docs/mentor-graph-aware.md). This file proves the split
actually holds on a live-shaped fixture, and that connections' honest-failure
paths never silently fall back to a live-similarity guess.
"""
import pytest

from tests.mentor_fixtures import GRAPH_NODES, make_context, make_tools

# n1 (Embolo - First Demo) is embedding-nearest to n2 (Burial - Archangel) in
# the fixture's vecs (see test_neighbors_uses_selection_and_explains_why in
# test_mentor_graph_tools.py — "Archangel" is neighbors' #1 pick for n1).
# We persist n1's ONLY drawn edge to n3 (Aphex Twin) instead, to mirror a
# real map where the UI's placement-time drawn edges don't always match a
# node's current nearest neighbours (different candidate pool, different
# scoring pass, or simply drawn before other songs were added).
DIVERGENT_LINKS = [
    {"source": "n1", "target": "n3", "score": 88.0, "value": 0.93, "kind": "qq"},
    {"source": "n4", "target": "n5", "score": 91.0, "value": 0.97, "kind": "qq"},
]


@pytest.fixture()
def tools_with_links(tmp_path):
    return make_tools(tmp_path, links=DIVERGENT_LINKS)


@pytest.fixture()
def tools_no_links(tmp_path):
    return make_tools(tmp_path, links=[])


# ---- the core divergence claim -------------------------------------------------
def test_connections_and_neighbors_diverge_on_live_shaped_fixture(tools_with_links):
    ctx = make_context(selected_node_id="n1")
    conn = tools_with_links.connections("me", context=ctx)
    neigh = tools_with_links.neighbors("me", context=ctx)

    assert conn["ok"] and neigh["ok"]
    # connections: exactly the one persisted edge, nothing else
    assert [f["artist"] for f in conn["findings"]] == ["Aphex Twin"]
    # neighbors: the live-kNN ranking, topped by the embedding-nearest node
    assert neigh["findings"][0]["artist"] == "Burial"
    # the two tools disagree on the #1 answer for the identical anchor —
    # this is the bug (both intents collapsing to the same kNN answer) the
    # split exists to fix.
    assert conn["findings"][0]["artist"] != neigh["findings"][0]["artist"]
    # and they're provenance-distinguishable, not just accidentally different
    assert conn["provenance"]["source"] == "persisted_links"
    assert neigh["provenance"]["source"] == "live_knn"
    assert conn["scope"] == "persisted"
    assert neigh["scope"] == "map"


def test_connections_never_falls_back_to_live_similarity(tools_no_links):
    """A node WITH zero drawn edges must report "no connections", never a
    live-kNN substitute — that silent substitution is exactly what made
    "connected to" and "most similar" collapse to the same wrong answer."""
    ctx = make_context(selected_node_id="n1")
    conn = tools_no_links.connections("me", context=ctx)
    assert conn["ok"]
    assert conn["connections"] == []
    assert conn["findings"] == []
    assert "no drawn connections" in conn["message"]
    # never silently substitutes a kNN-derived neighbour name
    neigh_names = {n["song"] for n in tools_no_links.neighbors("me", context=ctx)["neighbors"]}
    conn_names = {c["song"] for c in conn["connections"]}
    assert not (conn_names & neigh_names)


def test_connections_ranks_by_persisted_value_descending(tools_with_links):
    """Multiple persisted edges must come back ranked by the persisted raw
    cosine (``value``), not insertion order or a fresh recompute."""
    links = [
        {"source": "n1", "target": "n2", "score": 60.0, "value": 0.80, "kind": "qq"},
        {"source": "n1", "target": "n3", "score": 90.0, "value": 0.97, "kind": "qq"},
    ]
    import tempfile
    from pathlib import Path
    tools = make_tools(Path(tempfile.mkdtemp()), links=links)
    ctx = make_context(selected_node_id="n1")
    conn = tools.connections("me", context=ctx)
    assert [f["artist"] for f in conn["findings"]] == ["Aphex Twin", "Burial"]
    assert [f["rank"] for f in conn["findings"]] == [1, 2]


def test_connections_off_map_anchor_is_honest_not_a_fallback(tools_with_links):
    """An anchor that resolves to the corpus/library (not a placed graph
    node) has no map edges to read at all — connections must say so plainly,
    not silently answer with a corpus-similarity guess."""
    obs = tools_with_links.connections("Rihanna", context=make_context())
    assert not obs["ok"]
    assert obs["error"] == "not_on_map"
    assert "isn't placed on your map" in obs["message"]


def test_connections_unresolved_anchor_is_honest(tools_with_links):
    obs = tools_with_links.connections("NoSuchArtistAtAll", context=make_context())
    assert not obs["ok"]
    assert obs["error"] == "anchor_not_resolved"


def test_connections_respects_k(tools_with_links):
    ctx = make_context(selected_node_id="n1")
    obs = tools_with_links.connections("me", k=1, context=ctx)
    assert obs["ok"] and len(obs["findings"]) <= 1


def test_execute_dispatches_connections_intent(tools_with_links):
    ctx = make_context(selected_node_id="n1")
    obs = tools_with_links.execute("connections", {"anchor": "me", "k": 5}, context=ctx)
    assert obs["ok"] and obs["intent"] == "connections"
