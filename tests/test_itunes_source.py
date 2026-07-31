"""
iTunes preview tier (anther_ml/itunes.py) + the search relevance filter and
tier fall-through it hangs off (ui/atlas.py).

No network: the iTunes HTTP layer is monkeypatched with recorded response
shapes. The one real-audio test synthesizes its own AAC file through PyAV
rather than committing a binary fixture.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
import atlas  # noqa: E402

from anther_ml import itunes  # noqa: E402


# ── Fixtures mirroring real iTunes response shapes ───────────────────────────

def _result(track_id, artist_id, artist, title, *, preview=True,
            kind="song", wrapper="track", album="Some Album"):
    return {
        "wrapperType": wrapper,
        "kind": kind,
        "trackId": track_id,
        "artistId": artist_id,
        "artistName": artist,
        "trackName": title,
        "collectionName": album,
        "artworkUrl100": "https://example.invalid/art.jpg",
        "primaryGenreName": "Hip-Hop/Rap",
        **({"previewUrl": f"https://example.invalid/{track_id}.m4a"} if preview else {}),
    }


ARTIST_ID = 1598196562


def _patch_get(monkeypatch, payload):
    monkeypatch.setattr(itunes, "_get", lambda *a, **k: payload)


# ── Row normalization ────────────────────────────────────────────────────────

def test_track_row_maps_to_the_project_contract():
    row = itunes._track_row(_result(1, ARTIST_ID, "Matt Brade", "Pour Down"))
    assert row["id"] == "itunes:1"
    assert row["source"] == "itunes"
    assert row["itunes_id"] == 1
    assert row["artist"] == "Matt Brade"
    assert row["title"] == "Pour Down"
    assert row["preview_url"].endswith(".m4a")


@pytest.mark.parametrize("bad", [
    {"wrapper": "collection"},          # the artist/album header row
    {"kind": "music-video"},            # not an audio track
    {"preview": False},                 # nothing to embed
])
def test_track_row_rejects_unplayable_rows(bad):
    assert itunes._track_row(_result(1, ARTIST_ID, "A", "B", **bad)) is None


# ── Dedup: same song on both a single and an album ───────────────────────────

def test_dedupe_collapses_rereleases_across_collections():
    rows = [
        itunes._track_row(_result(10, ARTIST_ID, "Matt Brade", "Dreamweaver")),
        itunes._track_row(_result(11, ARTIST_ID, "Matt Brade",
                                  "Dreamweaver (feat. Noturlover)")),
        itunes._track_row(_result(12, ARTIST_ID, "Matt Brade", "Pour Down")),
    ]
    out = itunes.dedupe(rows)
    # _norm strips the parenthetical, so the two Dreamweavers collapse; the
    # first-seen trackId wins so repeated calls stay stable.
    assert [r["id"] for r in out] == ["itunes:10", "itunes:12"]


def test_dedupe_collapses_when_the_credit_string_differs_too():
    # A real pair from the live catalog: same recording, listed once under the
    # solo credit with the guest in the title, once under the joint credit.
    rows = [
        itunes._track_row(_result(20, ARTIST_ID, "Matt Brade & Noturlover",
                                  "Dreamweaver")),
        itunes._track_row(_result(21, ARTIST_ID, "Matt Brade",
                                  "Dreamweaver (feat. Noturlover)")),
    ]
    assert [r["id"] for r in itunes.dedupe(rows)] == ["itunes:20"]


def test_dedupe_keeps_a_real_guest_appearance_distinct():
    # Same title, genuinely different recording by a different primary artist —
    # keying on the primary credit must not collapse these.
    rows = [
        itunes._track_row(_result(30, ARTIST_ID, "Matt Brade", "Scroll")),
        itunes._track_row(_result(31, 1847129711, "Rob Knack",
                                  "Scroll (feat. Matt Brade)")),
    ]
    assert [r["id"] for r in itunes.dedupe(rows)] == ["itunes:30", "itunes:31"]


# ── Artist lookup: exact-name preference and collaborator leakage ────────────

def test_resolve_artist_prefers_exact_name_over_apple_ranking(monkeypatch):
    # Apple ranks a near-miss first; taking results[0] is how the Deezer tier
    # ends up showing "Matt Fradd" for "Matt Brade".
    _patch_get(monkeypatch, {"results": [
        {"artistId": 807748600, "artistName": "Matt Fradd",
         "primaryGenreName": "Spoken Word"},
        {"artistId": ARTIST_ID, "artistName": "Matt Brade",
         "primaryGenreName": "Hip-Hop/Rap"},
    ]})
    assert itunes.resolve_artist("Matt Brade")["artist_id"] == ARTIST_ID


def test_resolve_artist_returns_none_when_no_exact_match(monkeypatch):
    _patch_get(monkeypatch, {"results": [
        {"artistId": 1, "artistName": "Matt Fradd"},
    ]})
    assert itunes.resolve_artist("Matt Brade") is None


def test_artist_tracks_drops_other_artists_rows_by_default(monkeypatch):
    # A real lookup leaks collaborators: a Matt Brade lookup returns Rob Knack
    # tracks he features on.
    _patch_get(monkeypatch, {"results": [
        _result(1, ARTIST_ID, "Matt Brade", "Pour Down"),
        _result(2, 1847129711, "Rob Knack", "Facts (feat. Matt Brade)"),
    ]})
    own = itunes.artist_tracks(ARTIST_ID)
    assert [r["id"] for r in own] == ["itunes:1"]

    both = itunes.artist_tracks(ARTIST_ID, include_features=True)
    assert [r["id"] for r in both] == ["itunes:1", "itunes:2"]


def test_artist_tracks_survives_an_api_error(monkeypatch):
    _patch_get(monkeypatch, {"error": {"message": "HTTP 403"}})
    assert itunes.artist_tracks(ARTIST_ID) == []


# ── Relevance filter (the reason the fall-through was unreachable) ───────────

def _hit(artist, title, source="deezer"):
    return {"source": source, "artist": artist, "title": title}


def test_relevant_drops_confident_looking_catalog_noise():
    # Verbatim from Deezer's live response for "Rob Knack" (it reports 300 hits).
    noise = [_hit("Rob & Jack", "Bashment Ting"),
             _hit("Rob Black", "Influencer"),
             _hit("Sebastien Drums", "He's a Dream (Radio Mix)")]
    assert atlas._relevant("Rob Knack", noise) == []


def test_relevant_keeps_genuine_hits():
    hits = [_hit("Rob Knack", "Strippers"), _hit("Rob Knack", "Come Thru")]
    assert atlas._relevant("Rob Knack", hits) == hits


def test_relevant_matches_whole_words_not_substrings():
    # Spotify really returns this for "Rob Knack": substring containment passes
    # it ("knack" in "knackered", "rob" in "robe"), word matching rejects it.
    sneaky = [_hit("The Knackered Ramblers", "Wait 'til I Get on My Robe")]
    assert atlas._relevant("Rob Knack", sneaky) == []


def test_relevant_spans_artist_and_title_together():
    hits = [_hit("Kendrick Lamar", "Alright")]
    assert atlas._relevant("Kendrick Lamar Alright", hits) == hits


def test_relevant_passes_everything_through_on_an_empty_query():
    hits = [_hit("Anyone", "Anything")]
    assert atlas._relevant("   ", hits) == hits


# ── Tier fall-through ────────────────────────────────────────────────────────

@pytest.fixture
def tiers(monkeypatch):
    """Stub every tier; record which ones actually ran."""
    called = []

    def stub(name, results):
        def _f(q, limit):
            called.append(name)
            return list(results)
        return _f

    monkeypatch.setattr(atlas, "load", lambda: None)
    monkeypatch.setattr(atlas, "_search_corpus", stub("corpus", []))
    monkeypatch.setattr(atlas, "spotify_configured", lambda: True)
    return called


def test_noise_only_deezer_no_longer_blocks_the_later_tiers(monkeypatch, tiers):
    # The original bug: Deezer returns irrelevant-but-non-empty results, so the
    # `if not deezer_hits` gate never opened and Spotify/iTunes never ran.
    monkeypatch.setattr(atlas, "_search_deezer",
                        lambda q, l: [_hit("Rob & Jack", "Bashment Ting")])
    monkeypatch.setattr(atlas, "_search_spotify",
                        lambda q, l: {"configured": True, "results": []})
    monkeypatch.setattr(atlas, "_search_itunes",
                        lambda q, l: [_hit("Rob Knack", "Snake eyes", "itunes")])

    out = atlas.search("Rob Knack")
    assert out["tiers"] == {"corpus": 0, "deezer": 0, "spotify": 0, "itunes": 1}
    assert [h["source"] for h in out["results"]] == ["itunes"]


def test_a_thin_spotify_tier_still_tops_up_from_itunes(monkeypatch, tiers):
    """The gap is per-track, not per-query. Spotify surfaces exactly one Matt
    Brade track Deezer can resolve; stopping there hides the 14 only iTunes
    has, which is the whole point of adding the tier."""
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, l: [])
    monkeypatch.setattr(atlas, "_search_spotify",
                        lambda q, l: {"configured": True,
                                      "results": [_hit("Matt Brade", "Esquina", "spotify")]})
    monkeypatch.setattr(atlas, "_search_itunes",
                        lambda q, l: [_hit("Matt Brade", "Pour Down", "itunes"),
                                      _hit("Matt Brade", "Longtime", "itunes")])

    out = atlas.search("Matt Brade")
    assert out["tiers"] == {"corpus": 0, "deezer": 0, "spotify": 1, "itunes": 2}
    assert [h["title"] for h in out["results"]] == ["Esquina", "Pour Down", "Longtime"]


def test_topping_up_does_not_duplicate_a_track_an_earlier_tier_found(monkeypatch, tiers):
    # No shared id space (iTunes rows carry no ISRC), so dedup is on artist+title.
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, l: [])
    monkeypatch.setattr(atlas, "_search_spotify",
                        lambda q, l: {"configured": True,
                                      "results": [_hit("Matt Brade", "Esquina", "spotify")]})
    monkeypatch.setattr(atlas, "_search_itunes",
                        lambda q, l: [_hit("Matt Brade", "Esquina (feat. Blvck Svm)", "itunes"),
                                      _hit("Matt Brade", "Pour Down", "itunes")])

    out = atlas.search("Matt Brade")
    assert [(h["source"], h["title"]) for h in out["results"]] == [
        ("spotify", "Esquina"), ("itunes", "Pour Down")]
    assert out["tiers"]["itunes"] == 1


def test_a_full_deezer_tier_stops_the_chain(monkeypatch, tiers):
    # A well-covered artist must not pay for two extra API round-trips.
    ran = []
    monkeypatch.setattr(atlas, "_search_deezer",
                        lambda q, l: [_hit("Radiohead", f"Song {i}") for i in range(12)])
    monkeypatch.setattr(atlas, "_search_spotify",
                        lambda q, l: ran.append("spotify") or {"results": []})
    monkeypatch.setattr(atlas, "_search_itunes", lambda q, l: ran.append("itunes") or [])

    out = atlas.search("Radiohead")
    assert out["tiers"] == {"corpus": 0, "deezer": 12}
    assert ran == []


# ── Node-id routing ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("node_id,expected", [
    ("itunes:1641560176", "itunes"),
    ("deezer:12345", "deezer"),
    ("spotify:4uLU6hMCjMI75M1A2tKUQC", "spotify"),
    ("upload:my_song.mp3", "upload"),
    ("0UaMYEvWZi0ZqiDOoHU3YI", "mpd"),
])
def test_source_for_id(node_id, expected):
    assert atlas._source_for_id(node_id) == expected


def test_itunes_track_never_falls_back_to_a_deezer_fuzzy_match(monkeypatch):
    """An itunes: track is on the map *because* Deezer lacks it, so the Deezer
    fallback could only match a different song and embed the wrong audio."""
    monkeypatch.setattr(atlas, "load", lambda: None)

    def boom(*a, **k):
        raise AssertionError("Deezer fallback must not run for an itunes: track")

    monkeypatch.setattr(atlas, "match_deezer_track", boom)
    monkeypatch.setattr(atlas, "_download_url", boom)
    monkeypatch.setattr(itunes, "download_preview",
                        lambda url, **k: (_ for _ in ()).throw(RuntimeError("404")))

    with pytest.raises(atlas.PlacementSkip) as exc:
        atlas.resolve_and_embed({"id": "itunes:1", "name": "Pour Down",
                                 "artist": "Matt Brade",
                                 "preview_url": "https://example.invalid/1.m4a"})
    assert "download_failed" in str(exc.value.reason)


# ── Real AAC decode ──────────────────────────────────────────────────────────

def test_decode_preview_turns_aac_into_a_loadable_mono_wav(tmp_path):
    """The whole reason PyAV is a dependency: libsndfile can't read AAC and
    librosa's audioread fallback needs a system ffmpeg that isn't assumed."""
    av = pytest.importorskip("av")
    import soundfile as sf

    src = tmp_path / "preview.m4a"
    sr = 44100
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype("float32")
    with av.open(str(src), "w") as container:
        stream = container.add_stream("aac", rate=sr, layout="mono")
        frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1),
                                           format="fltp", layout="mono")
        frame.sample_rate = sr
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    # Precondition: the format really is unreadable without the transcode.
    with pytest.raises(Exception):
        sf.read(str(src))

    path, cleanup = itunes.decode_preview(src, target_sr=24000)
    try:
        wav, out_sr = sf.read(str(path))
        assert out_sr == 24000
        assert wav.ndim == 1                              # mono
        assert 0.8 < len(wav) / out_sr < 1.3              # ~1s, AAC pads a little
        assert float(np.sqrt((wav ** 2).mean())) > 0.05   # real signal, not silence
    finally:
        cleanup()
    assert not path.exists()


def test_decode_preview_rejects_a_file_with_no_audio_stream(tmp_path):
    junk = tmp_path / "notaudio.m4a"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(Exception):
        itunes.decode_preview(junk)
