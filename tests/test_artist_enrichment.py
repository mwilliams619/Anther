"""
anther_ml.artist_enrichment — schema, store, normalization, resolver and the
orchestrator, all exercised offline (source clients mocked, no network).
"""
import json

import pytest

from anther_ml.artist_enrichment import enrich, normalize
import anther_ml.artist_enrichment.resolve as resolve_mod
from anther_ml.artist_enrichment.schema import (
    ArtistProfile, STATUS_COMPLETE, STATUS_PARTIAL, STATUS_REVIEW, STATUS_UNMATCHED)
from anther_ml.artist_enrichment.store import ArtistProfileStore, open_readonly


# --- schema ---------------------------------------------------------------
def test_profile_roundtrip_and_api_shape():
    p = ArtistProfile(
        artist_key="weird al yankovic", name='"Weird Al" Yankovic',
        identities={"musicbrainz_id": "abc", "deezer_id": 12},
        profile_image={"url": "https://img", "source": "deezer", "updated_at": "2026-07-22"},
        following={"count": 1000, "source": "deezer", "updated_at": "2026-07-22"},
        genres=[{"name": "comedy", "source": "musicbrainz"}],
        hometown={"name": "California, US", "type": "area", "source": "musicbrainz"},
        labels=[{"name": "Volcano", "source": "musicbrainz", "scope": "release-derived"}],
        match_confidence=0.95, status=STATUS_COMPLETE, updated_at="2026-07-22")
    again = ArtistProfile.from_dict(p.to_dict())
    assert again == p
    api = p.to_api_profile()
    assert api["image_url"] == "https://img"
    assert api["following"] == 1000
    assert api["origin"] == "California, US"
    assert api["genres"] == ["comedy"]
    assert api["labels"] == ["Volcano"]
    assert api["enrichment_status"] == STATUS_COMPLETE


def test_missing_fields():
    p = ArtistProfile(artist_key="k", name="n", following={"count": 3})
    missing = p.missing_fields()
    assert "profile_image" in missing and "genres" in missing
    assert "following" not in missing


def test_from_dict_defaults_for_absent_keys():
    p = ArtistProfile.from_dict({"artist_key": "k", "name": "n"})
    assert p.genres == [] and p.identities == {} and p.status == STATUS_UNMATCHED


# --- store ----------------------------------------------------------------
def test_store_upsert_get_and_checkpoint(tmp_path):
    db = tmp_path / "p.sqlite"
    with ArtistProfileStore(db) as store:
        p = ArtistProfile(artist_key="k", name="Name", status=STATUS_COMPLETE,
                          match_confidence=0.9, updated_at="2026-07-22")
        store.upsert(p)
        assert store.has_complete("k")
        got = store.get("k")
        assert got.name == "Name" and got.status == STATUS_COMPLETE
        # idempotent upsert overwrites, no duplicate row
        store.upsert(ArtistProfile(artist_key="k", name="Renamed"))
        assert store.count() == 1
        assert store.get("k").name == "Renamed"


def test_store_raw_cache_and_review(tmp_path):
    db = tmp_path / "p.sqlite"
    with ArtistProfileStore(db) as store:
        store.cache_raw("deezer", "42", "search", {"data": [1]}, "2026-07-22")
        assert store.get_raw("deezer", "42", "search") == {"data": [1]}
        assert store.get_raw("deezer", "99", "search") is None
        store.queue_review("k", "Name", "ambiguous", [{"mbid": "x"}], "2026-07-22")
        rows = store.review_rows()
        assert rows[0]["name"] == "Name" and rows[0]["candidates"][0]["mbid"] == "x"
        assert store.review_count() == 1
        store.dequeue_review("k")            # rescued ⇒ no longer in the queue
        assert store.review_count() == 0
        store.dequeue_review("absent")       # idempotent, no error
        assert store.review_count() == 0


def test_open_readonly_missing(tmp_path):
    assert open_readonly(tmp_path / "nope.sqlite") is None


def test_load_all_api_join(tmp_path):
    db = tmp_path / "p.sqlite"
    with ArtistProfileStore(db) as store:
        store.upsert(ArtistProfile(artist_key="k", name="N",
                                   following={"count": 5, "source": "deezer"}))
        joined = store.load_all()
        assert joined["k"]["following"] == 5


# --- normalization --------------------------------------------------------
def test_normalize_genres_alias_dedupe_cap():
    raw = [{"name": "Hip Hop", "source": "spotify", "weight": 2},
           {"name": "hip-hop", "source": "musicbrainz", "weight": 9},
           {"name": "R&B", "source": "mb"}, {"name": "rock", "source": "mb"},
           {"name": "pop", "source": "mb"}, {"name": "jazz", "source": "mb"},
           {"name": "funk", "source": "mb"}]
    out = normalize.normalize_genres(raw)
    names = [g["name"] for g in out]
    assert names.count("hip-hop") == 1           # aliased + deduped
    assert out[0]["name"] == "hip-hop"           # highest weight sorts first
    assert len(out) == normalize.MAX_GENRES      # capped


def test_genres_from_musicbrainz_prefers_genres_over_tags():
    doc = {"genres": [{"name": "indie rock", "count": 5}],
           "tags": [{"name": "seen live", "count": 99}]}
    out = normalize.genres_from_musicbrainz(doc)
    assert [g["name"] for g in out] == ["indie rock"]
    # falls back to tags only when genres absent
    out2 = normalize.genres_from_musicbrainz({"tags": [{"name": "shoegaze", "count": 3}]})
    assert out2[0]["name"] == "shoegaze"


def test_structure_hometown_precedence():
    assert normalize.structure_hometown(
        {"begin_area": "LA", "area": "US", "country": "US"})["type"] == "begin_area"
    assert normalize.structure_hometown({"area": "US"})["type"] == "area"
    assert normalize.structure_hometown({"country": "GB"})["type"] == "country"
    assert normalize.structure_hometown({}) is None


def test_normalize_labels_scope_and_cap():
    raw = [{"name": f"L{i}", "release_count": 10 - i} for i in range(6)]
    out = normalize.normalize_labels(raw, top_n=3)
    assert len(out) == 3
    assert all(l["scope"] == "release-derived" for l in out)


def test_clean_image_url():
    assert normalize.clean_image_url("https://x/y.jpg") == "https://x/y.jpg"
    assert normalize.clean_image_url("ftp://x") is None
    assert normalize.clean_image_url(None) is None


# --- resolver -------------------------------------------------------------
def _mb(name, ratio, score=90, **kw):
    return {"mbid": kw.get("mbid", "mb-" + name), "name": name,
            "name_ratio": ratio, "mb_score": score,
            "begin_area": kw.get("begin_area"), "area": kw.get("area"),
            "country": kw.get("country")}


def _dz(name, ratio, deezer_id=1, nb_fan=100, image="https://i"):
    return {"deezer_id": deezer_id, "name": name, "name_ratio": ratio,
            "nb_fan": nb_fan, "image_url": image}


def test_resolve_complete_both_sources():
    r = resolve_mod.resolve("Radiohead", [_mb("Radiohead", 1.0, begin_area="Abingdon")],
                        [_dz("Radiohead", 1.0)])
    assert r.status == STATUS_COMPLETE
    assert r.mb["mbid"].startswith("mb-") and r.deezer["deezer_id"] == 1
    assert r.confidence > 0.9


def test_resolve_partial_deezer_only():
    r = resolve_mod.resolve("Obscure", [_mb("Totally Different", 0.4)], [_dz("Obscure", 1.0)])
    assert r.status == STATUS_PARTIAL
    assert r.mb is None and r.deezer is not None


def test_resolve_partial_musicbrainz_only():
    r = resolve_mod.resolve("Obscure", [_mb("Obscure", 0.98)], [_dz("Nope", 0.3)])
    assert r.status == STATUS_PARTIAL
    assert r.mb is not None and r.deezer is None


def test_resolve_ambiguous_goes_to_review():
    # two near-tied strong MB names — the short/common-name failure mode
    r = resolve_mod.resolve("John Smith",
                        [_mb("John Smith", 0.99, mbid="a"),
                         _mb("John Smith", 0.98, mbid="b")],
                        [])
    assert r.status == STATUS_REVIEW
    assert r.mb is None                      # MB fields withheld
    assert len(r.review_candidates) >= 2


def test_resolve_famous_artist_not_ambiguous_despite_namesakes():
    # MusicBrainz returns the canonical artist (score 100) plus obscure same-named
    # acts (score ≤80). The relevance gap means it is NOT ambiguous — pick the top.
    r = resolve_mod.resolve("Adele",
                        [_mb("Adele", 1.0, score=100, mbid="canonical"),
                         _mb("Adele", 1.0, score=80, mbid="namesake")],
                        [_dz("Adele", 1.0)])
    assert r.status == STATUS_COMPLETE
    assert r.mb["mbid"] == "canonical"


def test_resolve_ambiguous_keeps_confident_deezer():
    r = resolve_mod.resolve("John Smith",
                        [_mb("John Smith", 0.99, mbid="a"), _mb("John Smith", 0.985, mbid="b")],
                        [_dz("John Smith", 1.0)])
    assert r.status == STATUS_REVIEW
    assert r.deezer is not None              # independent Deezer match still kept


def test_resolve_unmatched():
    r = resolve_mod.resolve("Nobody", [_mb("Someone Else", 0.3)], [_dz("Different", 0.2)])
    assert r.status == STATUS_UNMATCHED
    assert r.mb is None and r.deezer is None


# --- orchestrator (mocked clients) ---------------------------------------
class _FakeMB:
    source = "musicbrainz"

    def search_artist(self, name, limit=8):
        return [_mb(name, 1.0, begin_area="London")]

    def lookup_artist(self, mbid):
        return {"genres": [{"name": "Trip Hop", "count": 4}]}

    def artist_release_labels(self, mbid, limit=100):
        return [{"name": "Wild Bunch", "mbid": "l1", "release_count": 3}]


class _FakeDZ:
    source = "deezer"

    def search_artist(self, name, limit=8):
        return [_dz(name, 1.0, nb_fan=555, image="https://pic")]


def _write_meta(tmp_path):
    meta = [{"artist": "Massive Attack", "n_tracks": 5, "sample_track": "Teardrop"},
            {"artist": "Massive Attack", "n_tracks": 5, "sample_track": "dup"},  # deduped
            {"artist": "Portishead", "n_tracks": 4, "sample_track": "Roads"}]
    path = tmp_path / "artist_meta.json"
    path.write_text(json.dumps(meta))
    return path


def test_enrich_one_builds_full_profile(tmp_path):
    artists = enrich.load_corpus_artists(_write_meta(tmp_path))
    assert len(artists) == 2  # dup normalized name collapsed
    prof, res = enrich.enrich_one(artists[0], mb=_FakeMB(), dz=_FakeDZ(),
                                  store=None, now="2026-07-22")
    assert prof.status == STATUS_COMPLETE
    assert prof.profile_image["url"] == "https://pic"
    assert prof.following["count"] == 555
    assert prof.genres[0]["name"] == "trip hop"          # normalized/lowercased
    assert prof.hometown["name"] == "London"
    assert prof.labels[0]["name"] == "Wild Bunch"
    assert prof.identities["deezer_id"] == 1


def test_run_writes_store_and_reports_and_resumes(tmp_path, monkeypatch):
    meta = _write_meta(tmp_path)
    db = tmp_path / "profiles.sqlite"
    monkeypatch.setattr(enrich, "MusicBrainzClient", lambda **kw: _FakeMB())
    monkeypatch.setattr(enrich, "DeezerArtistClient", lambda **kw: _FakeDZ())

    report = enrich.run(meta_path=meta, db_path=db)
    assert report.processed == 2 and report.matched_complete == 2
    assert report.field_coverage["genres"] == 2

    # resume: everything complete ⇒ all skipped, nothing reprocessed
    report2 = enrich.run(meta_path=meta, db_path=db)
    assert report2.processed == 0 and report2.skipped_complete == 2

    # the store now serves the API join
    with ArtistProfileStore(db) as store:
        joined = store.load_all()
        assert joined["massive attack"]["following"] == 555


def test_run_dry_run_does_not_write(tmp_path, monkeypatch):
    meta = _write_meta(tmp_path)
    db = tmp_path / "profiles.sqlite"
    monkeypatch.setattr(enrich, "MusicBrainzClient", lambda **kw: _FakeMB())
    monkeypatch.setattr(enrich, "DeezerArtistClient", lambda **kw: _FakeDZ())
    report = enrich.run(meta_path=meta, db_path=db, dry_run=True)
    assert report.processed == 2
    assert not db.exists()
