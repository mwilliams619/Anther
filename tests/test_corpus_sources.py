"""
anther_ml.corpus.sources.sql_source_oversampled — oversample-and-substitute
fetch semantics, exercised with everything below the SQL sample mocked out
(no real DB, no network).
"""
import numpy as np
import pytest

from anther_ml.corpus import sources


def _patch_common(monkeypatch, pool, fail_ids=frozenset(), deezer_ok=frozenset()):
    """
    ``pool``: list of {"track_id","name","preview_url","artist_name"} dicts,
    in the deterministic order sample_tracks would return them.
    ``fail_ids``: track_ids whose Spotify preview fetch raises.
    ``deezer_ok``: subset of fail_ids that succeed via the Deezer fallback.
    """
    monkeypatch.setattr(sources, "log", sources.log)  # keep logger, no-op
    monkeypatch.setattr(
        "anther_ml.mpd_sql.ensure_db", lambda **kw: "fake.sqlite"
    )
    monkeypatch.setattr(
        "anther_ml.mpd_sql.sample_tracks",
        lambda db_path, **kw: (pool, {}),
    )

    def fake_fetch(url, target_sr=None, clip_seconds=None):
        # url encodes the track_id as "spotify_preview:<id>" or "deezer:<id>"
        kind, tid = url.split(":", 1)
        if kind == "spotify_preview" and tid in fail_ids:
            raise RuntimeError("dead preview url")
        if kind == "deezer" and tid not in deezer_ok:
            raise RuntimeError("deezer fetch failed")
        return np.zeros(24000, dtype=np.float32), 1.0

    def fake_match(track, min_ratio=0.82):
        tid = track["sp_id"]
        if tid in deezer_ok:
            return {"preview": f"deezer:{tid}"}
        return {"error": "no match"}

    monkeypatch.setattr(
        "anther_ml.spotify_deezer.fetch_preview_waveform", fake_fetch
    )
    monkeypatch.setattr(
        "anther_ml.spotify_deezer.match_deezer_track", fake_match
    )


def _pool(n: int, dead: frozenset = frozenset()) -> list[dict]:
    return [
        {
            "track_id": f"t{i:04d}",
            "name": f"song {i}",
            "preview_url": f"spotify_preview:t{i:04d}",
            "popularity": float(i),
            "artist_name": f"artist{i % 7}",
        }
        for i in range(n)
    ]


def test_oversampled_reaches_exact_target_with_no_failures(monkeypatch):
    pool = _pool(20)
    _patch_common(monkeypatch, pool)
    items = list(
        sources.sql_source_oversampled(
            target_n=10, oversample_ratio=1.5, n_workers=4,
        )
    )
    assert len(items) == 10
    ids = [it["id"] for it in items]
    assert len(set(ids)) == 10  # no duplicates
    # deterministic prefix: same pool order as a plain sample would give
    assert ids[0] == "spotify:t0000"


def test_oversampled_substitutes_past_dead_urls(monkeypatch):
    pool = _pool(20)
    dead = frozenset(f"t{i:04d}" for i in (0, 1, 2))  # first 3 candidates dead
    _patch_common(monkeypatch, pool, fail_ids=dead)
    items = list(
        sources.sql_source_oversampled(
            target_n=10, oversample_ratio=2.0, n_workers=4,
        )
    )
    assert len(items) == 10  # exact count despite 3 dead candidates
    ids = {it["id"] for it in items}
    assert not (ids & {f"spotify:{tid}" for tid in dead})  # dead ones excluded


def test_oversampled_uses_deezer_fallback(monkeypatch):
    pool = _pool(10)
    dead = frozenset({"t0000"})
    _patch_common(monkeypatch, pool, fail_ids=dead, deezer_ok=dead)
    items = list(
        sources.sql_source_oversampled(
            target_n=10, oversample_ratio=1.0, n_workers=4,
        )
    )
    assert len(items) == 10  # t0000 recovered via Deezer, not skipped


def test_oversampled_raises_if_pool_exhausted(monkeypatch):
    pool = _pool(10)
    dead = frozenset(r["track_id"] for r in pool)  # everything dead
    _patch_common(monkeypatch, pool, fail_ids=dead)
    with pytest.raises(RuntimeError, match="only 0/10"):
        list(
            sources.sql_source_oversampled(
                target_n=10, oversample_ratio=1.0, n_workers=4,
            )
        )


def test_oversampled_pool_exhaustion_blames_the_db_not_the_ratio(monkeypatch):
    # Pool smaller than target_n * oversample_ratio → SQLite ran out of rows.
    _patch_common(monkeypatch, _pool(4))
    with pytest.raises(RuntimeError, match="DB is exhausted") as exc:
        list(
            sources.sql_source_oversampled(
                target_n=10, oversample_ratio=2.0, n_workers=4,
                min_popularity=31, max_popularity=70,
            )
        )
    assert "will not help" in str(exc.value)
    assert "popularity 31-70" in str(exc.value)


def test_oversampled_allow_short_yields_what_exists(monkeypatch):
    _patch_common(monkeypatch, _pool(4))
    items = list(
        sources.sql_source_oversampled(
            target_n=10, oversample_ratio=2.0, n_workers=4, allow_short=True,
        )
    )
    assert len(items) == 4


def test_oversampled_allow_short_still_raises_on_dead_urls(monkeypatch):
    # Full pool available, but every preview is dead → the buffer is the problem,
    # so allow_short must not paper over it.
    pool = _pool(20)
    _patch_common(monkeypatch, pool, fail_ids=frozenset(r["track_id"] for r in pool))
    with pytest.raises(RuntimeError, match="Raise oversample_ratio"):
        list(
            sources.sql_source_oversampled(
                target_n=10, oversample_ratio=2.0, n_workers=4, allow_short=True,
            )
        )


def test_oversampled_items_match_source_contract(monkeypatch):
    pool = _pool(5)
    _patch_common(monkeypatch, pool)
    items = list(
        sources.sql_source_oversampled(target_n=5, oversample_ratio=1.2)
    )
    for it in items:
        assert set(it) == {
            "id", "name", "artist", "source", "genre", "playlists",
            "audio", "sr",
            "popularity",
        }
        assert it["source"] == "mpd_sql"
        assert it["genre"] is None  # display-only invariant
        assert it["audio"].shape == (24000,)
