# Notes — pruning the MPD playlist DB for hosting (not implemented)

**Status**: reference only. Not implemented. The hosted UI currently ships
without `ANTHER_MPD_DB` set at all — full-MPD playlist-name search is simply
absent (`mpd_ready()` in `ui/atlas.py` returns false and every call site
already checks it first, so this degrades cleanly). The underlying MPD
dataset is years old, so pruning it wasn't judged worth the effort yet. This
doc exists so the approach isn't lost if that changes (a dataset refresh, or
deciding the stale data is fine after all).

## The problem

`data/mpd_dump/spotifydbdumpshare.sqlite` is **27GB** — by far the largest
piece of storage the app could need, and the only reason a hosted deploy
would need more than ~2.5GB (corpus bundle + MERT weights). Everything else
the app touches at request time is small (see
[REFERENCE_CORPUS_DESIGN.md](REFERENCE_CORPUS_DESIGN.md) for what a corpus
bundle actually contains).

## Why a simple "keep the top N playlists" isn't as simple as it sounds

The obvious approach — keep only the N most popular playlists — needs a
popularity signal. `anther_ml/mpd_sql.py`'s `TABLE_SPEC` for `playlist` only
keeps `(id, name)` from the source MySQL dump:

```python
"playlist": (
    "CREATE TABLE IF NOT EXISTS playlist (id TEXT PRIMARY KEY, name TEXT)",
    [(0, "id"), (1, "name")],
),
```

But the module's own docstring records the *source* dump's real schema:

```
playlist(id, name, followers, uri, total_tracks)
```

The `followers` column — real Spotify follower counts, the actual popularity
signal — is dropped on ingest and was never captured into the SQLite DB.

**Ranking by track count instead (the only signal currently in the DB) is
the wrong proxy.** Verified live against the existing 27GB DB: the top 5,000
playlists by track count reference **6.5M distinct tracks out of 13.28M
total** (half the entire catalog) and sum to 18.1M membership rows — because
sorting by size surfaces sprawling dump/mega playlists, not popular ones.
Any future pruning attempt should rank by `followers`, not `n_tracks`, and
should re-run the same distinct-track-count sanity check shown below before
trusting a size estimate — don't assume popular/editorial playlists are
small just because that's typical; verify.

```sql
-- sanity check to run against whatever ranking is used, before committing to it
CREATE TEMP TABLE topN AS
SELECT playlist_id, COUNT(*) AS n
FROM track_playlist1
GROUP BY playlist_id
ORDER BY n DESC          -- replace with the real ranking (e.g. by followers)
LIMIT 5000;

SELECT COUNT(DISTINCT tp.track_id)
FROM track_playlist1 tp
JOIN topN ON tp.playlist_id = topN.playlist_id;
```

## The approach, if revisited

1. **Add `followers` to `TABLE_SPEC["playlist"]`** in `anther_ml/mpd_sql.py`
   (source column index 2 — `playlist(id, name, followers, uri, total_tracks)`).

2. **Backfill, don't full-reload.** `track`, `track_artist1`, and
   `track_playlist1` are already correctly populated in the existing 27GB
   DB — re-streaming those (the expensive part of the original ~16 min
   ingest, 125M rows) isn't necessary just to add one column to `playlist`
   (~1M rows). Add a small `backfill_playlist_followers(dump_path, db_path)`
   function: re-scan only the `playlist` INSERT lines from the raw dump
   (`data/mpd_dump/spotifydbdumpshare.sql`, still on disk), reusing the
   existing `parse_insert_line`/`_iter_tuples` parsing already in the
   module, and `UPDATE playlist SET followers = ? WHERE id = ?` per row
   (or bulk via a temp table + join-update for speed).

3. **Prune to a minimal DB.** Given the top-N playlist ids ranked by
   `followers DESC`, copy into a fresh, small SQLite file only:
   - the `playlist` rows for those N ids
   - their `track_playlist1` rows
   - the `track` rows those reference
   - the `track_artist1` / `artist` rows for those tracks

   That's exactly the set of tables `search_playlists_db`, `playlist_name`,
   and `playlist_tracks` read (see `anther_ml/mpd_sql.py`) — nothing more is
   needed to serve playlist search and playlist-track placement.

4. **Run the existing `prepare_ui()`** on the new, small file to materialize
   `playlist_search` (name/name_norm/n_tracks) and its indices — it already
   handles this idempotently for any DB matching the schema above.

5. **Point `ANTHER_MPD_DB`** at the new pruned file. No other code changes
   needed — `ui/atlas.py` and `ui/app.py` already read the path from that
   env var.

6. **Verify before shipping**: re-run the distinct-track-count sanity query
   above against the real `followers`-ranked top-N, and report the actual
   resulting `.sqlite` file size — don't assume it'll be small without
   checking (the track-count-ranked test above shows how wrong a plausible-
   sounding assumption can be).

## Suggested starting point

`--top-n 5000` was the number discussed, but it's a free parameter — rerun
step 6 for whatever N is chosen before committing to it.
