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
