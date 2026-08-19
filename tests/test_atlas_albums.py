"""
Tiered album search (ui/atlas.py): Deezer title search → Deezer artist
discography → Spotify → iTunes.

No network: every tier is exercised through a stubbed ``_deezer_get`` plus
stubbed name tiers. The corpus is never loaded — album search doesn't touch it.

The fixture catalog is the real failure this tiering was built for: King
Krule's "The OOZ" is in Deezer's catalog but its album *search* index matches
titles only, so "king krule the ooz" returns nothing and "the ooz" returns an
unrelated band called The Oozes.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
import atlas  # noqa: E402


OOZES_ALBUMS = [
    {"id": 1, "title": "Bitchboy", "artist": {"name": "The Oozes"}, "nb_tracks": 1},
    {"id": 2, "title": "Gelatinous Man", "artist": {"name": "The Oozes"}, "nb_tracks": 10},
]
KRULE_ALBUMS = [
    {"id": 100, "title": "The OOZ"},
    {"id": 101, "title": "Man Alive!"},
    {"id": 102, "title": "6 Feet Beneath The Moon"},
]


def _fake_deezer(catalog_search=None):
    """Deezer stub. ``catalog_search`` is what search/album returns for a plain
    query; the artist endpoints always know King Krule and The Oozes."""
    calls = []

    def get(path, params=None, **kw):
        calls.append((path, dict(params or {})))
        q = (params or {}).get("q", "")
        if path == "search/album":
            if q.startswith("artist:"):                     # structured resolve
                return {"data": KRULE_ALBUMS[:1] if "OOZ" in q else []}
            return {"data": catalog_search if catalog_search is not None else []}
        if path == "search/artist":
            if "king krule" in q.lower():
                return {"data": [{"id": 7, "name": "King Krule", "nb_album": 3}]}
            if "ooz" in q.lower():
                return {"data": [{"id": 8, "name": "The Oozes", "nb_album": 2}]}
            return {"data": []}
        if path == "artist/7/albums":
            return {"data": KRULE_ALBUMS}
        if path == "artist/8/albums":
            return {"data": OOZES_ALBUMS}
        if path.startswith("album/"):
            return {"id": path.split("/")[1], "nb_tracks": 19}
        return {"data": []}

    get.calls = calls
    return get


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Empty the artist-discography cache and silence the name tiers; each
    test opts back into the ones it is exercising."""
    atlas._artist_albums_cache.clear()
    monkeypatch.setattr(atlas, "_spotify_album_names", lambda q, limit: [])
    monkeypatch.setattr(atlas, "_itunes_album_names", lambda q, limit: [])
    monkeypatch.setattr(atlas, "spotify_configured", lambda: False)
    yield
    atlas._artist_albums_cache.clear()


# ── relevance filters ────────────────────────────────────────────────────────

def test_relevant_albums_drops_confident_noise():
    """"the ooz" must not be answered with a band called The Oozes: "ooz" is
    a substring of "oozes", not a word in it."""
    rows = [{"artist": "The Oozes", "name": "Bitchboy"},
            {"artist": "King Krule", "name": "The OOZ"}]
    assert atlas._relevant_albums("the ooz", rows) == [rows[1]]


def test_recall_filter_keeps_partial_token_matches():
    """The name tiers use recall, not all-tokens: "sgt pepper" survives the
    possessive on "Sgt. Pepper's", nonsense does not survive at all."""
    beatles = {"artist": "The Beatles", "name": "Sgt. Pepper's Lonely Hearts Club Band"}
    junk = {"artist": "Unexist", "name": "Don't exist"}
    kept = atlas._albums_by_recall("the beatles sgt pepper", [beatles, junk])
    assert kept == [beatles]


def test_usable_half_rejects_stopwords_and_stubs():
    assert atlas._usable_half("king krule")
    assert not atlas._usable_half("the")
    assert not atlas._usable_half("a")


# ── tier 2: artist discography ───────────────────────────────────────────────

def test_artist_album_query_resolves_through_discography(monkeypatch):
    """"king krule the ooz" finds nothing in Deezer's album index; splitting
    it into artist + album and matching the discography finds the record."""
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    out = atlas.search_albums("king krule the ooz", limit=8)
    assert out["results"][0]["album_id"] == 100
    assert out["results"][0]["name"] == "The OOZ"
    assert out["tiers"]["deezer"] == 0 and out["tiers"]["deezer_artist"] > 1
    # the rest of the discography rides along, and never the exact hit twice
    assert [r["album_id"] for r in out["results"]] == [100, 101, 102]


def test_reversed_query_order_resolves_too(monkeypatch):
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    out = atlas.search_albums("the ooz king krule", limit=8)
    assert out["results"][0]["album_id"] == 100


def test_bare_artist_query_browses_discography(monkeypatch):
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    out = atlas.search_albums("king krule", limit=8)
    assert {r["album_id"] for r in out["results"]} == {100, 101, 102}


def test_near_miss_artist_does_not_hijack_the_query(monkeypatch):
    """"the ooz" is an 0.87 name match for "The Oozes" — close enough for the
    split path, not close enough to answer the query with their discography."""
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    out = atlas.search_albums("the ooz", limit=8)
    assert out["tiers"]["deezer_artist"] == 0
    assert out["results"] == []


def test_ghost_artist_loses_to_the_real_catalog(monkeypatch):
    """Deezer ranks an empty duplicate artist above the real one; the entry
    with albums must win, or the discography route comes back empty."""
    def get(path, params=None, **kw):
        if path == "search/artist":
            return {"data": [{"id": 900, "name": "King Krule", "nb_album": 0},
                             {"id": 7, "name": "King Krule", "nb_album": 3}]}
        if path == "artist/7/albums":
            return {"data": KRULE_ALBUMS}
        if path == "artist/900/albums":
            return {"data": []}
        if path.startswith("album/"):
            return {"nb_tracks": 19}
        return {"data": []}
    monkeypatch.setattr(atlas, "_deezer_get", get)
    assert [r["album_id"] for r in atlas._deezer_artist_albums("king krule")] == [100, 101, 102]


# ── tiers 3/4: name resolvers ────────────────────────────────────────────────

def test_spotify_names_resolve_back_onto_deezer(monkeypatch):
    """Spotify only supplies a canonical (artist, title) pair — the result the
    UI places is still a Deezer album id, so place_album is unchanged."""
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    monkeypatch.setattr(atlas, "_spotify_album_names",
                        lambda q, limit: [("King Krule", "The OOZ")])
    out = atlas.search_albums("the ooz album", limit=8)
    assert [r["album_id"] for r in out["results"]] == [100]
    assert out["tiers"]["spotify"] == 1


def test_itunes_tier_runs_when_spotify_is_short(monkeypatch):
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    monkeypatch.setattr(atlas, "_itunes_album_names",
                        lambda q, limit: [("King Krule", "The OOZ")])
    out = atlas.search_albums("the ooz album", limit=8)
    assert out["tiers"]["itunes"] == 1
    assert [r["album_id"] for r in out["results"]] == [100]


def test_track_counts_filled_for_discography_rows(monkeypatch):
    """artist/<id>/albums carries no nb_tracks; the UI would show "? tracks"."""
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    out = atlas.search_albums("king krule the ooz", limit=8)
    assert out["results"][0]["n_tracks"] == 19


# ── failure modes ────────────────────────────────────────────────────────────

def test_deezer_outage_still_raises_when_nothing_else_answers(monkeypatch):
    monkeypatch.setattr(atlas, "_deezer_get",
                        lambda path, params=None, **kw: {"error": {"message": "HTTP 503"}})
    with pytest.raises(RuntimeError, match="503"):
        atlas.search_albums("king krule the ooz")


def test_empty_query_is_not_a_search(monkeypatch):
    monkeypatch.setattr(atlas, "_deezer_get", _fake_deezer())
    assert atlas.search_albums("   ") == {"results": [], "tiers": {},
                                          "spotify_configured": False}
