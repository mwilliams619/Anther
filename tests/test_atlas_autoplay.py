"""
Graph autoplay's two server-side pieces (ui/atlas.py):

  * ``nearest_unplayed``    — the island jump the frontend can't compute itself
  * ``resolve_spotify_batch`` — batched Spotify-id resolve-ahead

Both are exercised against a hand-built ``_SessionState`` rather than the
injected-corpus fixture the older test_atlas_*.py files use (those are marked
stale against the pre-multi-session atlas). ``nearest_unplayed`` takes its
state explicitly for exactly this reason, so no corpus, no MERT and no network
are involved here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
import atlas  # noqa: E402


def _unit(*xs) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def st(tmp_path, monkeypatch):
    """A bare session with four query nodes on a unit circle, so cosine order
    is obvious by construction: a is nearest b, then c, then d."""
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    state = atlas._SessionState("test")
    nodes = {
        "spotify:a": _unit(1.0, 0.0),
        "spotify:b": _unit(1.0, 0.1),    # closest to a
        "spotify:c": _unit(1.0, 1.0),    # 45 degrees off
        "spotify:d": _unit(-1.0, 0.0),   # opposite a
    }
    for nid, vec in nodes.items():
        state.graph["nodes"][nid] = {"id": nid, "kind": "query", "name": nid}
        state.query_vecs[nid] = vec
    return state


# ── nearest_unplayed ─────────────────────────────────────────────────────────

def test_picks_the_highest_cosine_candidate(st):
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit["id"] == "spotify:b"
    assert hit["value"] == pytest.approx(float(np.dot(st.query_vecs["spotify:a"],
                                                      st.query_vecs["spotify:b"])), abs=1e-3)


def test_excludes_played_ids(st):
    hit = atlas.nearest_unplayed("spotify:a", ["spotify:b", "spotify:c"], st=st)
    assert hit["id"] == "spotify:d"


def test_never_returns_the_anchor_itself(st):
    # 'a' is not in exclude, but jumping to the song we just played is a bug.
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit["id"] != "spotify:a"


def test_corpus_nodes_are_not_jump_targets(st):
    st.graph["nodes"]["spotify:ctx"] = {"id": "spotify:ctx", "kind": "corpus", "name": "ctx"}
    st.query_vecs["spotify:ctx"] = _unit(1.0, 0.01)      # nearer a than b is
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit["id"] == "spotify:b", "context nodes are never played, so never jumped to"


def test_returns_none_when_everything_is_played(st):
    played = [n for n in st.graph["nodes"]]
    assert atlas.nearest_unplayed("spotify:a", played, st=st) is None


def test_empty_map_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path)
    assert atlas.nearest_unplayed("spotify:a", [], st=atlas._SessionState("empty")) is None


def test_anchor_without_a_vector_still_yields_a_candidate(st):
    """A tour must never stall because one node was cached before MERIT
    support — map order is the honest fallback."""
    hit = atlas.nearest_unplayed("spotify:ghost", [], st=st)
    assert hit is not None and hit["id"] in st.graph["nodes"]
    assert hit["score"] is None


def test_no_anchor_at_all_yields_a_candidate(st):
    hit = atlas.nearest_unplayed(None, [], st=st)
    assert hit is not None and hit["score"] is None


def test_candidates_without_vectors_are_skipped_not_fatal(st):
    st.graph["nodes"]["spotify:novec"] = {"id": "spotify:novec", "kind": "query", "name": "nv"}
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit["id"] == "spotify:b"


def test_all_candidates_lack_vectors_falls_back_to_map_order(st):
    st.query_vecs.clear()
    st.query_vecs["spotify:a"] = _unit(1.0, 0.0)         # anchor only
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit is not None and hit["score"] is None


def test_jump_score_reads_below_the_link_floor(st):
    """A jump is by definition below the edge threshold; clip_low=False lets
    that read honestly instead of flattening to SCORE_FLOOR_DISPLAY."""
    hit = atlas.nearest_unplayed("spotify:a", ["spotify:b", "spotify:c"], st=st)
    assert hit["id"] == "spotify:d"
    assert hit["score"] < 55.0, "opposite vectors must not display as a floor-level match"


def test_uses_merit_space_when_the_bundle_has_one(st, monkeypatch):
    from anther_ml import calibration as link_calibration
    monkeypatch.setattr(atlas, "_merit_link_thresholds",
                        link_calibration.LinkThresholds.defaults())
    # MERIT space disagrees with MERT space: here 'd' is the nearest neighbour.
    st.merit_vecs.update({
        "spotify:a": _unit(1.0, 0.0),
        "spotify:b": _unit(-1.0, 0.0),
        "spotify:c": _unit(0.0, 1.0),
        "spotify:d": _unit(1.0, 0.05),
    })
    hit = atlas.nearest_unplayed("spotify:a", [], st=st)
    assert hit["id"] == "spotify:d", "must rank in the same space the edges were drawn in"


# ── resolve_spotify_batch ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_spotify_cache():
    atlas._spotify_id_cache.clear()
    yield
    atlas._spotify_id_cache.clear()


def test_spotify_prefixed_ids_resolve_without_any_api_call(monkeypatch):
    monkeypatch.setattr(atlas, "spotify_configured", lambda: True)
    def _boom(*a, **k):
        raise AssertionError("a spotify: id must never hit the Search API")
    monkeypatch.setattr(atlas.requests, "get", _boom)
    out = atlas.resolve_spotify_batch(["spotify:abc123", "spotify:def456"])
    assert out == {"spotify:abc123": "abc123", "spotify:def456": "def456"}


def test_unresolvable_ids_map_to_none(monkeypatch):
    monkeypatch.setattr(atlas, "spotify_configured", lambda: False)
    out = atlas.resolve_spotify_batch(["upload:mysong.mp3", "deezer:99"])
    assert out == {"upload:mysong.mp3": None, "deezer:99": None}


def test_one_bad_id_does_not_fail_the_batch(monkeypatch):
    real = atlas.get_spotify_track_id
    def _flaky(song_id):
        if song_id == "deezer:boom":
            raise RuntimeError("network died")
        return real(song_id)
    monkeypatch.setattr(atlas, "get_spotify_track_id", _flaky)
    out = atlas.resolve_spotify_batch(["spotify:ok", "deezer:boom"])
    assert out == {"spotify:ok": "ok", "deezer:boom": None}


def test_duplicates_and_empties_are_collapsed(monkeypatch):
    monkeypatch.setattr(atlas, "spotify_configured", lambda: False)
    out = atlas.resolve_spotify_batch(["spotify:a", "spotify:a", "", None])
    assert out == {"spotify:a": "a"}
