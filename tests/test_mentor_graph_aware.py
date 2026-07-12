"""No-GPU regression harness for the graph-aware mentor.

Runs the full reasoning loop (generate_fn=None -> deterministic composer) against
the LIVE on-screen session at ui/session/graph.json + embed_cache.sqlite. These
are the ten questions the mentor used to fail on: five deflected, Q9 hallucinated
artist names. Each test asserts the question now routes to the right tool, returns
ok:True against the real graph, and surfaces the expected grounded fact.

The harness skips (does not fail) when the live session artifacts are absent, so
it is safe to run on a machine that has never opened the UI. No model, no GPU, no
network — the whole thing is deterministic reads over the on-screen vectors.
"""
import json
import os

import pytest

from mentor.mentor_anther import AntherSoundsLike
from mentor.mentor_graphctx import GraphContext
from mentor.mentor_react import MentorReAct, classify_question

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GRAPH_JSON = os.path.join(REPO, "ui", "session", "graph.json")
CORPUS_DIR = os.path.join(REPO, "models", "corpus_corpus_mpd_100k")
CORPUS_OK = os.path.isdir(CORPUS_DIR)


def _expected_n_clusters() -> int:
    """Read the live cluster count from cluster_profiles.json rather than
    hardcoding it — reclustering (13 -> 84, and whatever comes next) should
    never require touching this test."""
    profiles_path = os.path.join(CORPUS_DIR, "cluster_profiles.json")
    with open(profiles_path) as f:
        return len(json.load(f))

pytestmark = pytest.mark.skipif(
    not (os.path.exists(GRAPH_JSON) and CORPUS_OK),
    reason="needs live ui/session/graph.json and the frozen corpus bundle",
)


@pytest.fixture(scope="module")
def react():
    ant = AntherSoundsLike()
    gc = GraphContext()
    if not gc.has_nodes():
        pytest.skip("on-screen graph has no placed nodes")
    # generate_fn=None -> the loop composes deterministically from observations,
    # so the assertions below test the TOOLS + LOOP, not a language model.
    return MentorReAct(ant, generate_fn=None, graph_ctx=gc)


def _run(react, q):
    r = react.run(q)
    ok_tools = [o["tool"] for o in r["observations"] if o.get("ok")]
    return r, ok_tools


# ---- the motivating question ------------------------------------------------
def test_within_bridge_umo_returns_meshuggah(react):
    """What song from UMO bridges the Rae Sremmurd and Jimi Hendrix clusters?"""
    q = ("what song from unknown mortal orchestra bridges the rae sremmurd "
         "and jimi hendrix clusters?")
    assert classify_question(q)["shape"] == "within_bridge"
    r, ok = _run(react, q)
    assert "bridge" in ok, r["answer"]
    obs = next(o for o in r["observations"] if o.get("tool") == "bridge" and o.get("ok"))
    names = [n["name"] for n in obs["neighbors"]]
    # Meshuggah is the strongest bridge (min-sim ~0.970) on the current graph.
    assert names[0] == "Meshuggah", names
    assert obs.get("most_balanced", {}).get("name") == "The Widow"


# ---- Q1: sounds-closest-to an uploaded demo --------------------------------
def test_q1_sounds_like_parenthetical_anchor(react):
    q = "What established artists and tracks does my uploaded demo sound closest to? (embolo)"
    assert classify_question(q)["args"]["anchor"] == "embolo"
    r, ok = _run(react, q)
    assert "sounds_like" in ok, r["answer"]
    assert r.get("status") != "absent_anchor"


# ---- Q2: micro-genres from on-screen probe tags ----------------------------
def test_q2_micro_genres_from_probe(react):
    q = "Which micro-genres does embolo actually fall into?"
    assert classify_question(q)["shape"] == "micro_genres"
    r, ok = _run(react, q)
    assert "micro_genres" in ok, r["answer"]
    obs = next(o for o in r["observations"] if o.get("tool") == "micro_genres")
    assert obs["tags"], "expected at least one micro-genre tag"


# ---- Q3: artists between two on-screen anchors -----------------------------
def test_q3_bridge_between_two_anchors(react):
    q = "What artists sit sonically between jimi hendrix and justin beiber?"
    assert classify_question(q)["shape"] == "bridge"
    r, ok = _run(react, q)
    assert "bridge" in ok, r["answer"]


# ---- Q4: which of artist X's songs matches a reference ---------------------
def test_q4_artist_tracks(react):
    q = "Which of Jimi Hendrix's songs is the closest match to Embolo?"
    cls = classify_question(q)
    assert cls["shape"] == "artist_tracks"
    assert cls["args"]["artist"].lower() == "jimi hendrix"
    r, ok = _run(react, q)
    assert "artist_tracks" in ok, r["answer"]


# ---- Q5: adjacent cluster to explore ---------------------------------------
def test_q5_crossover(react):
    # Needs a concrete anchor to answer "which adjacent cluster"; a bare "should
    # I study..." with no upload in context honestly deflects (there is no anchor
    # to be adjacent TO). The real use names an on-screen artist.
    q = "Which adjacent cluster should Rae Sremmurd cross into?"
    assert classify_question(q)["shape"] == "crossover"
    r, ok = _run(react, q)
    assert "crossover" in ok, r["answer"]


# ---- Q6: album coherence + outliers ----------------------------------------
def test_q6_coherence_album_purpose(react):
    q = "How coherent is the album Purpose, and what are the outliers?"
    cls = classify_question(q)
    assert cls["shape"] == "coherence"
    assert cls["args"]["target"] == "Purpose"
    r, ok = _run(react, q)
    assert "coherence" in ok, r["answer"]
    obs = next(o for o in r["observations"] if o.get("tool") == "coherence")
    assert obs["n"] >= 2
    assert obs["outliers"], "expected at least one flagged outlier"
    assert "Life Is Worth Living" in {m["name"] for m in obs["outliers"]}


# ---- Q7: tagmates -----------------------------------------------------------
def test_q7_tagmates(react):
    q = "What tagmates does jimi hendrix have?"
    assert classify_question(q)["shape"] == "tagmates"
    r, ok = _run(react, q)
    assert "tagmates" in ok, r["answer"]


# ---- Q8: compare two artists -----------------------------------------------
def test_q8_compare(react):
    q = "Compare rae sremmurd to jimi hendrix"
    assert classify_question(q)["shape"] == "compare"
    r, ok = _run(react, q)
    assert "compare" in ok, r["answer"]
    obs = next(o for o in r["observations"] if o.get("tool") == "compare")
    assert 0.0 < obs["similarity"] <= 1.0


# ---- Q9: sonic territories = cluster_profiles.json's clusters (was hallucinating) --
def test_q9_cluster_summary_all_territories(react):
    q = "What are the major sonic territories on the map?"
    assert classify_question(q)["shape"] == "cluster_summary"
    r, ok = _run(react, q)
    assert "cluster_summary" in ok, r["answer"]
    obs = next(o for o in r["observations"] if o.get("tool") == "cluster_summary")
    # Compared against the live cluster_profiles.json length, not a hardcoded
    # number, so a future reclustering (this repo has already gone 13 -> 84)
    # can't silently desync this assertion from reality again.
    assert obs["n_clusters"] == _expected_n_clusters()
    # every territory has a real label read from cluster_profiles.json
    assert all(c["label"] for c in obs["clusters"])


# ---- Q10: multi-seed recommend ---------------------------------------------
def test_q10_recommend_from_seeds(react):
    q = "Recommend something based on rae sremmurd and jimi hendrix"
    cls = classify_question(q)
    assert cls["tool"] in ("bridge", "sounds_like")
    r, ok = _run(react, q)
    assert ok, r["answer"]
    assert r.get("status") != "absent_anchor"


# ---- honest deflection: a genuinely absent anchor --------------------------
def test_absent_anchor_is_named_not_generic(react):
    # A name that is neither on-screen nor in the frozen corpus, so it can't
    # resolve at any tier. The loop should say so BY NAME, not emit the old
    # generic "I need a concrete anchor" line.
    q = 'What does "Zxqwvbn Nonexistentartist" sound closest to?'
    r, ok = _run(react, q)
    assert not ok
    assert r.get("status") == "absent_anchor"
    assert "Zxqwvbn Nonexistentartist" in r["answer"]
    assert "I need a concrete anchor" not in r["answer"]
