import json
import io
import pickle
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def atlas_module():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
    import atlas
    atlas._load_artist_clustering()
    assert atlas.artist_status()["available"] is True
    return atlas


@pytest.fixture()
def isolated_artist_sessions(atlas_module, monkeypatch, tmp_path):
    atlas = atlas_module
    monkeypatch.setattr(atlas, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(atlas, "is_ready", lambda: True)
    atlas._sessions.clear()
    yield atlas
    atlas._sessions.clear()


def test_canonical_artist_artifacts_are_aligned(atlas_module):
    atlas = atlas_module
    assert len(atlas._artist_meta) == 2163
    assert atlas._artist_embeddings.shape == (2163, 1024)
    assert atlas._artist_labels.shape == (2163,)
    assert atlas._artist_embedding_2d.shape == (2163, 2)


def test_mismatched_bundle_is_controlled_unavailable(atlas_module, monkeypatch, tmp_path):
    atlas = atlas_module
    art_dir = tmp_path / "bad-artists"
    art_dir.mkdir()
    (art_dir / "artist_meta.json").write_text(json.dumps([
        {"artist": "One"}, {"artist": "Two"},
    ]))
    np.save(art_dir / "artist_embeddings.npy", np.ones((1, 4), dtype=np.float32))
    np.save(art_dir / "artist_labels.npy", np.zeros(1, dtype=np.int64))
    np.save(art_dir / "artist_embedding_2d.npy", np.ones((1, 2), dtype=np.float32))
    with open(art_dir / "artist_leiden.pkl", "wb") as handle:
        pickle.dump({"clustering_space": np.ones((1, 2), dtype=np.float32)}, handle)

    original = atlas.ARTIST_CLUSTERING_DIR
    monkeypatch.setattr(atlas, "ARTIST_CLUSTERING_DIR", str(art_dir))
    atlas._load_artist_clustering()
    assert atlas.artist_status()["available"] is False
    assert "row mismatch" in atlas.artist_status()["error"]
    with pytest.raises(RuntimeError, match="row mismatch"):
        atlas.get_artist_graph()

    monkeypatch.setattr(atlas, "ARTIST_CLUSTERING_DIR", original)
    atlas._load_artist_clustering()


def test_incremental_graph_duplicate_remove_clear_and_demo(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    with atlas.use_session("one"):
        search = atlas.search_artists("Drake")["results"]
        assert search[0]["id"] == "corpus:546"
        first = atlas.place_artist("corpus:546")
        assert len(first["nodes"]) == 1
        assert atlas.place_artist("corpus:546")["existing"] is True
        atlas.place_artist("corpus:1039")
        graph = atlas.get_artist_graph()
        assert len(graph["nodes"]) == 2
        assert all(link["score"] >= 55 for link in graph["links"])
        assert atlas.remove_artist("corpus:546")["removed"] == ["corpus:546"]
        atlas.clear_artist_graph()
        assert atlas.get_artist_graph()["nodes"] == []
        demo = atlas.place_artist_demo()
        assert len(demo["nodes"]) == 20
        assert demo["added"] == 20
        assert atlas.place_artist_demo()["added"] == 0


def test_artist_graphs_are_session_isolated(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    with atlas.use_session("alpha"):
        atlas.place_artist("corpus:546")
        assert len(atlas.get_artist_graph()["nodes"]) == 1
    with atlas.use_session("beta"):
        assert atlas.get_artist_graph()["nodes"] == []


def test_build_artist_graph_from_added_song_nodes(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    with atlas.use_session("song-seeds"):
        st = atlas.get_session()
        st.graph["nodes"] = {
            "added": {"id": "added", "artist": "Drake", "kind": "query"},
            # Context nodes must not influence the artist graph.
            "context": {"id": "context", "artist": "The Beatles", "kind": "corpus"},
            "unknown": {"id": "unknown", "artist": "Uncatalogued Artist", "kind": "query"},
        }

        result = atlas.build_artist_graph_from_song_graph("append")
        assert result["artist_count"] == 1
        assert result["skipped_artists"] == ["Uncatalogued Artist"]
        assert [node["id"] for node in result["nodes"]] == ["corpus:546"]

        atlas.place_artist("corpus:1039")
        replaced = atlas.build_artist_graph_from_song_graph("replace")
        assert [node["id"] for node in replaced["nodes"]] == ["corpus:546"]


def test_build_from_songs_creates_and_supplements_noncorpus_artist(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    hits = [{"source": "deezer", "id": f"deezer:{i}", "deezer_id": i,
             "title": f"Extra {i}", "artist": "Brand New Artist", "album": "",
             "cover": "", "preview_url": f"http://x/{i}.mp3"} for i in range(6)]
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: hits)

    extra = [atlas._artist_embeddings[100 + i] for i in range(6)]

    def fake_resolve(track):
        idx = int(track["id"].split(":")[1])
        atlas.cache_vec(track["id"], extra[idx], track["name"], track["artist"])
        return extra[idx], None, "deezer"

    monkeypatch.setattr(atlas, "resolve_and_embed", fake_resolve)

    with atlas.use_session("build-noncorpus"):
        st = atlas.get_session()
        # One on-map song by an artist not in the corpus, with a cached embedding.
        atlas.cache_vec("song1", atlas._artist_embeddings[200], "My Song",
                        "Brand New Artist")
        st.graph["nodes"] = {
            "song1": {"id": "song1", "artist": "Brand New Artist", "kind": "query"},
        }
        result = atlas.build_artist_graph_from_song_graph("append")
        assert result["artist_count"] == 1
        assert result["skipped_artists"] == []
        node = result["nodes"][0]
        assert node["id"].startswith("session:")
        assert node["source"] == "session"
        # 1 on-map song + 4 supplemented previews = 5 total → no longer low-conf.
        assert node["track_count"] == 5
        assert node["low_confidence"] is False


def test_build_from_songs_uses_corpus_song_embeddings(
        isolated_artist_sessions, monkeypatch):
    # A corpus song by an artist absent from the artist bundle: its vector lives
    # in the frozen corpus (not embed_cache), yet the session artist must still
    # be built from it. No Deezer supplement available → stays low-confidence.
    atlas = isolated_artist_sessions
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: [])
    emb = np.stack([atlas._artist_embeddings[42]])
    meta = [{"artist": "Corpus-Only Artist", "name": "Deep Cut",
             "source": "corpus", "id": "song-x"}]
    monkeypatch.setattr(atlas, "_corpus", _FakeCorpus(meta, emb))
    monkeypatch.setattr(atlas, "_id_to_idx", {"song-x": 0})

    with atlas.use_session("build-corpus-song"):
        st = atlas.get_session()
        st.graph["nodes"] = {
            "song-x": {"id": "song-x", "artist": "Corpus-Only Artist",
                       "kind": "query", "source": "corpus"},
        }
        result = atlas.build_artist_graph_from_song_graph("append")
        assert result["artist_count"] == 1
        assert result["skipped_artists"] == []
        node = result["nodes"][0]
        assert node["id"].startswith("session:")
        assert node["track_count"] == 1
        assert node["low_confidence"] is True


def test_build_from_songs_survives_placement_error(
        isolated_artist_sessions, monkeypatch):
    # A single artist that fails to place (e.g. a transient DB lock) must not
    # turn the whole build into an unhandled 500 — it's skipped, others survive.
    atlas = isolated_artist_sessions
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: [])
    emb = np.stack([atlas._artist_embeddings[42]])
    meta = [{"artist": "Corpus-Only Artist", "name": "Deep Cut",
             "source": "corpus", "id": "song-x"}]
    monkeypatch.setattr(atlas, "_corpus", _FakeCorpus(meta, emb))
    monkeypatch.setattr(atlas, "_id_to_idx", {"song-x": 0})

    real_place = atlas._place_artist_locked

    def boom(st, artist_id):
        if str(artist_id).startswith("session:"):
            raise Exception("simulated placement failure")
        return real_place(st, artist_id)

    monkeypatch.setattr(atlas, "_place_artist_locked", boom)

    with atlas.use_session("build-place-error"):
        st = atlas.get_session()
        st.graph["nodes"] = {
            "song-x": {"id": "song-x", "artist": "Corpus-Only Artist",
                       "kind": "query", "source": "corpus"},
        }
        result = atlas.build_artist_graph_from_song_graph("append")
        assert result["artist_count"] == 0        # nothing landed, but no crash
        assert result["added"] == 0


def test_build_from_songs_skips_noncorpus_artist_without_embedding(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: [])
    with atlas.use_session("build-noembed"):
        st = atlas.get_session()
        # Song node with no cached embedding → cannot build an artist from it.
        st.graph["nodes"] = {
            "ghost": {"id": "ghost", "artist": "Nobody Here", "kind": "query"},
        }
        result = atlas.build_artist_graph_from_song_graph("append")
        assert result["artist_count"] == 0
        assert result["skipped_artists"] == ["Nobody Here"]


# STALE: seeds the song graph on the fixture's default session, then POSTs via a
# fresh test client that gets its own cookie-scoped session — so the route reads
# an empty graph and returns artist_count 0. Needs rewriting to share one session
# id between the seeding and the request. Excluded from the default run.
@pytest.mark.stale
def test_artist_graph_from_song_map_route(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    st = atlas.get_session()
    st.graph["nodes"] = {"added": {"id": "added", "artist": "Drake", "kind": "query"}}
    import app
    response = app.app.test_client().post('/api/artist/from-song-graph', json={'mode': 'append'})
    assert response.status_code == 200
    assert response.get_json()["artist_count"] == 1


def test_private_profile_uses_cached_upload_and_survives_reload(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    with atlas.use_session("profile"):
        raw = np.asarray(atlas._artist_embeddings[546], dtype=np.float32)
        atlas.cache_vec("upload:one.wav", raw, "one", "Private Persona")
        fragment = atlas.assign_upload_artist(
            "upload:one.wav", artist_name="Private Persona")
        artist_id = fragment["id"]
        assert artist_id.startswith("session:")
        assert atlas.artist_detail(artist_id)["upload_count"] == 1
        atlas.cache_vec("upload:two.wav", atlas._artist_embeddings[1039],
                        "two", "Private Persona")
        updated = atlas.assign_upload_artist("upload:two.wav", artist_id=artist_id)
        assert updated["replace"] is True
        assert atlas.artist_detail(artist_id)["upload_count"] == 2
        with pytest.raises(ValueError, match="already exists"):
            atlas.validate_upload_artist(artist_name="Private Persona")

    atlas._sessions.pop("profile")
    with atlas.use_session("profile"):
        graph = atlas.get_artist_graph()
        assert [node["id"] for node in graph["nodes"]] == [artist_id]
        assert atlas.artist_detail(artist_id)["name"] == "Private Persona"


def test_frozen_artist_assignment_does_not_change_bundle(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    before = atlas._artist_embeddings[546].copy()
    with atlas.use_session("frozen-upload"):
        atlas.cache_vec("upload:two.wav", before, "two", "Drake")
        fragment = atlas.assign_upload_artist("upload:two.wav", artist_id="corpus:546")
        assert fragment["id"] == "corpus:546"
        assert atlas.artist_detail("corpus:546")["upload_count"] == 1
    np.testing.assert_array_equal(atlas._artist_embeddings[546], before)


def test_failed_private_artist_placement_rolls_back_profile(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    with atlas.use_session("rollback"):
        atlas.cache_vec("upload:failed.wav", atlas._artist_embeddings[546],
                        "failed", "Rollback Artist")
        monkeypatch.setattr(atlas, "place_artist",
                            lambda _artist_id: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            atlas.assign_upload_artist("upload:failed.wav", artist_name="Rollback Artist")
        assert atlas._session_artist_rows(atlas.get_session()) == []


def test_incremental_artist_routes_and_client_isolation(isolated_artist_sessions):
    atlas = isolated_artist_sessions
    import app
    first = app.app.test_client()
    second = app.app.test_client()
    search = first.get('/api/artist/search?q=Drake')
    assert search.status_code == 200
    assert search.get_json()['results'][0]['id'] == 'corpus:546'
    assert first.post('/api/artist/place', json={'artist_id': 'corpus:546'}).status_code == 200
    assert len(first.get('/api/artist/graph').get_json()['nodes']) == 1
    assert second.get('/api/artist/graph').get_json()['nodes'] == []
    assert first.get('/api/artist/corpus:546').get_json()['name'] == 'Drake'
    assert first.post('/api/artist/create', json={'name': 'unsafe'}).status_code == 405


def test_upload_creates_private_profile_only_after_embedding(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    import app
    raw = np.asarray(atlas._artist_embeddings[546], dtype=np.float32)

    def fake_place(result):
        atlas.cache_vec(result['id'], raw, result['title'], result['artist'])
        return {'nodes': [{'id': result['id']}], 'links': []}

    monkeypatch.setattr(atlas, 'place_song', fake_place)
    client = app.app.test_client()
    response = client.post('/api/upload', data={
        'file': (io.BytesIO(b'fake audio'), 'private.wav'),
        'artist_name': 'Private Upload Artist',
    }, content_type='multipart/form-data')
    assert response.status_code == 200
    data = response.get_json()
    assert data['artist_fragment']['id'].startswith('session:')
    assert len(client.get('/api/artist/graph').get_json()['nodes']) == 1


def test_upload_without_artist_still_works_when_artist_mode_is_unavailable(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    import app
    monkeypatch.setattr(atlas, '_artist_ready', False)
    monkeypatch.setattr(atlas, '_artist_error', 'test unavailable')
    monkeypatch.setattr(atlas, 'place_song', lambda result: {'nodes': [], 'links': []})
    response = app.app.test_client().post('/api/upload', data={
        'file': (io.BytesIO(b'fake audio'), 'personal.wav'),
    }, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json()['artist_fragment'] is None


def test_same_upload_filename_gets_distinct_identity(
        isolated_artist_sessions, monkeypatch):
    atlas = isolated_artist_sessions
    import app
    ids = []

    def fake_place(result):
        ids.append(result['id'])
        return {'nodes': [{'id': result['id']}], 'links': []}

    monkeypatch.setattr(atlas, 'place_song', fake_place)
    client = app.app.test_client()
    for payload in (b'first audio', b'second audio'):
        response = client.post('/api/upload', data={
            'file': (io.BytesIO(payload), 'same-name.wav'),
        }, content_type='multipart/form-data')
        assert response.status_code == 200

    assert len(ids) == 2
    assert ids[0].startswith('upload:')
    assert ids[0] != ids[1]


class _FakeCorpus:
    """Minimal stand-in for ReferenceCorpus so the low-confidence pool can be
    built without loading the 100k-track bundle."""
    def __init__(self, metadata, embeddings):
        self.metadata = metadata
        self.embeddings = embeddings


@pytest.fixture()
def lowconf_pool(isolated_artist_sessions, monkeypatch):
    """A 2-track corpus artist ('Lowconf Artist') pushed into the pool, reusing
    real 1024-d embedding rows so the Leiden scaler/PCA transform is valid."""
    atlas = isolated_artist_sessions
    emb = np.stack([atlas._artist_embeddings[546], atlas._artist_embeddings[1039]])
    meta = [
        {"artist": "Lowconf Artist", "name": "Track One", "source": "test", "id": "lc1"},
        {"artist": "Lowconf Artist", "name": "Track Two", "source": "test", "id": "lc2"},
    ]
    monkeypatch.setattr(atlas, "_corpus", _FakeCorpus(meta, emb))
    atlas._build_lowconf_pool()
    yield atlas
    atlas._build_lowconf_pool()  # reset (real _corpus is None → empty)


def test_lowconf_pool_is_searchable_and_flagged(lowconf_pool):
    atlas = lowconf_pool
    assert len(atlas._artist_lowconf_rows) == 1
    with atlas.use_session("lc-search"):
        results = atlas.search_artists("Lowconf Artist")["results"]
        hit = next(r for r in results if r["id"] == "lowconf:0")
        assert hit["low_confidence"] is True
        assert hit["track_count"] == 2
        assert isinstance(hit["cluster_id"], int)


def test_lowconf_artist_places_against_frozen_reference(lowconf_pool, monkeypatch):
    atlas = lowconf_pool
    # No supplementary previews available → placement gracefully lands the
    # artist as low-confidence rather than failing.
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: [])
    with atlas.use_session("lc-place"):
        fragment = atlas.place_artist("lowconf:0")
        assert fragment["nodes"][0]["low_confidence"] is True
        detail = atlas.artist_detail("lowconf:0")
        assert detail["low_confidence"] is True
        assert detail["sample_track"] == "Track One"


def test_place_lowconf_artist_auto_supplements(lowconf_pool, monkeypatch):
    atlas = lowconf_pool
    hits = [{"source": "deezer", "id": f"deezer:{i}", "deezer_id": i,
             "title": f"Extra {i}", "artist": "Lowconf Artist", "album": "",
             "cover": "", "preview_url": f"http://x/{i}.mp3"} for i in range(3)]
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: hits)

    extra = [atlas._artist_embeddings[100], atlas._artist_embeddings[200],
             atlas._artist_embeddings[300]]

    def fake_resolve(track):
        idx = int(track["id"].split(":")[1])
        atlas.cache_vec(track["id"], extra[idx], track["name"], track["artist"])
        return extra[idx], None, "deezer"

    monkeypatch.setattr(atlas, "resolve_and_embed", fake_resolve)

    with atlas.use_session("lc-autoplace"):
        fragment = atlas.place_artist("lowconf:0")
        node = fragment["nodes"][0]
        # Placement pulled the strict-matched previews up to the 5-track target.
        assert node["low_confidence"] is False
        assert node["track_count"] == 5
        assert atlas.artist_detail("lowconf:0")["low_confidence"] is False
        # Session-only: the frozen pool row is untouched.
        assert atlas._artist_lowconf_rows[0]["n_tracks"] == 2


def test_supplement_promotes_lowconf_artist_out_of_low_confidence(
        lowconf_pool, monkeypatch):
    atlas = lowconf_pool
    hits = [{"source": "deezer", "id": f"deezer:{i}", "deezer_id": i,
             "title": f"Extra {i}", "artist": "Lowconf Artist", "album": "",
             "cover": "", "preview_url": f"http://x/{i}.mp3"} for i in range(3)]
    monkeypatch.setattr(atlas, "_search_deezer", lambda q, limit: hits)

    extra = [atlas._artist_embeddings[100], atlas._artist_embeddings[200],
             atlas._artist_embeddings[300]]

    def fake_resolve(track):
        idx = int(track["id"].split(":")[1])
        atlas.cache_vec(track["id"], extra[idx], track["name"], track["artist"])
        return extra[idx], None, "deezer"

    monkeypatch.setattr(atlas, "resolve_and_embed", fake_resolve)

    with atlas.use_session("lc-supplement"):
        # Manual easter-egg path: supplement_artist_placement both fetches the
        # previews and places the strengthened node.
        result = atlas.supplement_artist_placement("lowconf:0")
        assert result["added"] == 3
        assert result["track_count"] == 5
        assert result["low_confidence"] is False
        # Session-only: the frozen pool row is unchanged.
        assert atlas._artist_lowconf_rows[0]["n_tracks"] == 2
        assert atlas.artist_detail("lowconf:0")["low_confidence"] is False


def test_supplement_rejects_non_lowconf_ids(lowconf_pool):
    atlas = lowconf_pool
    with atlas.use_session("lc-reject"):
        with pytest.raises(ValueError, match="low-confidence"):
            atlas.supplement_artist_placement("corpus:546")


def test_static_ui_keeps_song_and_artist_renderers_separate():
    root = Path(__file__).parent.parent
    html = (root / 'ui/static/index.html').read_text()
    app_js = (root / 'ui/static/app.js').read_text()
    artist_js = (root / 'ui/static/artist-graph.js').read_text()
    assert '<svg id="graph"></svg>' in html
    assert '<svg id="artist-graph" hidden></svg>' in html
    assert '/static/artist-graph.js' in html
    assert "state.viewMode" in app_js
    assert "getElementById('graph').toggleAttribute('hidden', artists)" in app_js
    assert "getElementById('artist-graph').toggleAttribute('hidden', !artists)" in app_js
    assert "fetch('/api/artist/place'" in app_js
    assert "const ArtistGraph" in artist_js
    assert "fetch('/api/artist/graph')" in artist_js
