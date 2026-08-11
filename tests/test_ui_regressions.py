"""Focused regressions for the mobile graph shell and Spotify resolver."""

from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_mobile_graphs_disable_hover_affordances():
    graph = (ROOT / "ui/static/graph.js").read_text()
    artist = (ROOT / "ui/static/artist-graph.js").read_text()
    mobile = (ROOT / "ui/static/mobile.js").read_text()
    css = (ROOT / "ui/static/style.css").read_text()

    assert "max-width: 768px" in graph
    assert "if (!hoverEnabled()) return;" in graph
    assert "max-width: 768px" in artist
    assert "if (!hoverEnabled()) return;" in artist
    assert "tip.style.display = 'none'" in mobile
    assert "body.mshell #graph-tooltip" in css


@pytest.fixture
def atlas_module(monkeypatch):
    pytest.importorskip("numpy")
    import sys
    sys.path.insert(0, str(ROOT / "ui"))
    import atlas

    monkeypatch.setattr(atlas, "_spotify_id_cache", {})
    monkeypatch.setattr(atlas, "_spotify_status_cache", {})
    return atlas


def test_native_spotify_id_does_not_need_api_credentials(atlas_module, monkeypatch):
    atlas = atlas_module
    monkeypatch.setattr(atlas, "spotify_configured", lambda: False)

    result = atlas.resolve_spotify_track("spotify:abc123")

    assert result == {"track_id": "abc123", "status": "resolved"}


def test_missing_spotify_credentials_are_reported(atlas_module, monkeypatch):
    atlas = atlas_module
    monkeypatch.setattr(atlas, "spotify_configured", lambda: False)

    result = atlas.resolve_spotify_track("deezer:42")

    assert result == {"track_id": None, "status": "not_configured"}


def test_spotify_auth_failure_is_reported(atlas_module, monkeypatch):
    atlas = atlas_module
    monkeypatch.setattr(atlas, "spotify_configured", lambda: True)
    monkeypatch.setattr(atlas, "track_name_artist", lambda _id: ("Song", "Artist"))

    class Response:
        status_code = 401

        def raise_for_status(self):
            import requests
            error = requests.HTTPError("unauthorized")
            error.response = self
            raise error

    monkeypatch.setattr(atlas.requests, "get", lambda *args, **kwargs: Response())

    result = atlas.resolve_spotify_track("deezer:42")

    assert result == {"track_id": None, "status": "auth_failed"}
