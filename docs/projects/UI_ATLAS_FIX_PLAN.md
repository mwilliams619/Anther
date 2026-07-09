# Atlas UI: fix regressions, Tracks/Playlists/Albums modes, 100-song cap, de-emphasize clusters

## Context

Follow-up to the full-MPD playlist placement work (landed, uncommitted). Four items:

1. **Regression reports**: "playlist search gone, graph/sidebar dead, map changed." Diagnosis: (a) the saved session graph got ~380 test nodes from my end-to-end testing — user chose to **keep** them; (b) a real warm-up race: `/api/graph` returns an empty graph while the corpus is still loading (~1–2 min after `python ui/app.py`), so a page opened too early shows a blank, non-interactive map with no way to tell it's still loading — and my test server held port 5000 during much of the prior session, which would break a concurrently started instance; (c) no cache-busting on static assets, so a stale `app.js` can pair with new HTML. The Tracks/Playlists toggle itself was never removed — code is intact ([ui/static/index.html:26-29](../../ui/static/index.html#L26-L29)); fix the race + cache busting and verify at runtime.
2. **Album search mode** (user decision: **Deezer API** — zero DB prep, any album, previews guaranteed; no MPD album table needed).
3. **Cap imports at 100 songs** (user decision: top 100 by popularity).
4. **De-emphasize clusters** (new request): stop coloring map nodes by cluster — the force layout itself shows song relationships; cluster info appears **only in the click popover** (detail panel already shows it).

**Does this restore the interactive cluster UI?** Yes. The d3 force graph, hover highlight/tooltip, and click popover were never removed — they stopped working at runtime because of the warm-up race and stale-cache issues fixed in section A. After this plan: hover shows the tooltip (name/artist + import-group badge) and neighbor highlighting; click pins the node and opens the detail popover with cluster id/label/confidence, micro-genre tags, playlists, and the similar-songs list — same as before. The single intentional difference (per the follow-up request): map nodes are no longer fill-colored by cluster; cluster info lives only in the click popover.

## A. Regression hardening

- **Warm-up gating** — [ui/atlas.py](../../ui/atlas.py) `get_graph()`: return `{"ready": is_ready(), "nodes": [...], "links": [...]}` (empty lists when not ready). [ui/static/graph.js](../../ui/static/graph.js) `load()`: if `!data.ready`, show a "loading corpus…" hint (small overlay div in `.graph-wrap`) and retry every 2 s until ready, then render. Kills the blank-map-on-fresh-open race.
- **Cache busting** — [ui/static/index.html](../../ui/static/index.html): `graph.js?v=2`, `app.js?v=2`, `style.css?v=2` (bump on future changes).
- **Runtime verification** of the toggle + playlist search (they were never removed; confirm after hard refresh in the checklist below).

## B. De-emphasize clusters (map = pure force relationships)

- [ui/static/graph.js](../../ui/static/graph.js): drop `clusterColor`/PALETTE fills — uniform colors: query nodes one neutral accent fill, corpus nodes the existing grey (desaturation CSS already handles rest state). Playlist/album accent **rings stay** (they're import-group badges, not clusters). Remove the `cluster N · conf X%` line from the hover tooltip.
- Click popover ([ui/static/app.js](../../ui/static/app.js) `renderDetail`) already shows cluster id/confidence/label — unchanged; that's now the only place clusters appear.
- No backend change; cluster is still computed and stored on nodes.

## C. Albums via Deezer

Backend ([ui/atlas.py](../../ui/atlas.py), reusing `_deezer_get`, `cached_vec`, `_place_query_vec`, `playlist_jobs`):
- `search_albums(q, limit=20)` → `_deezer_get("search/album")` → `[{album_id, title, artist, cover, n_tracks}]`.
- `place_album(album_id)` → `_deezer_get(f"album/{album_id}/tracks", params={"limit": 100})` → rows `{id: f"deezer:{tid}", name, artist, preview_url}`; apply `IMPORT_CAP`; split: `cached_vec` hits place instantly via `_place_query_vec(source="deezer", extra={"playlist_pid": f"album:{album_id}"})`, rest → `playlist_jobs.start(f"album:{album_id}", album_name, pending)`. Same response shape as `place_playlist` (`playlist` key holds the group info — one shape, one frontend path). Deezer signed preview URLs expire in hours — fine, jobs start immediately; `resolve_and_embed`'s Deezer re-match is the fallback.
- [ui/app.py](../../ui/app.py): `GET /api/albums/search`, `POST /api/album/place` (mirror the playlist routes).

Frontend:
- [ui/static/index.html](../../ui/static/index.html): third mode button `mode-albums`.
- [ui/static/app.js](../../ui/static/app.js): `SEARCH_MODES.albums`; `doSearch` routes to `/api/albums/search`; `renderAlbumResults` (cover thumb + title/artist + track count + Load); `loadAlbum` = same flow as `loadPlaylist` (shared progress line + `startPlaylistPoll`).
- [ui/static/graph.js](../../ui/static/graph.js) tooltip group line: `pid.startsWith('album:') ? '● from album' : '● from playlist'`.

## D. 100-song import cap

- [ui/atlas.py](../../ui/atlas.py): `IMPORT_CAP = int(os.environ.get("ANTHER_IMPORT_CAP", "100"))`. In `place_playlist` (both branches) and `place_album`: `rows = rows[:IMPORT_CAP]` (playlist rows are already popularity DESC from `playlist_tracks`; album rows keep Deezer track order). Response gains `n_total` (uncapped size) + `capped: bool`; `n_tracks` stays the capped count.
- [ui/static/app.js](../../ui/static/app.js): when `capped`, progress copy reads "placing top 100 of 4,270 tracks…".

## E. Tests & docs

- [tests/test_atlas_playlist.py](../../tests/test_atlas_playlist.py): add cap test (monkeypatch `IMPORT_CAP=1`, assert 1 immediate + 0 pending + `capped`/`n_total` fields); add `place_album` test with a stubbed `_deezer_get` (cached/pending split, cap, `album:` group id, job start).
- [docs/architecture.md](../architecture.md): update `ui/atlas.py` row (albums via Deezer, import cap, cluster display only in detail panel). Update memory file after landing.

## Verification

1. `python ui/app.py`; immediately curl `/api/graph` → `ready: false`, then flips true with nodes after corpus load; browser shows "loading corpus…" then the map (no more silent blank).
2. Hard-refresh: all three mode buttons render and switch (placeholder/hint change); playlist search returns results with real counts; **confirm the Tracks/Playlists regression is gone**.
3. Albums e2e: search an album (e.g. Daft Punk "Discovery"), Load → tracks stream in with an album-colored ring, progress line ticks, done summary.
4. Cap: load a >100-track playlist → progress says "top 100 of N", job total = 100 minus instants.
5. Cluster de-emphasis: map nodes uniform (no cluster colors), hover tooltip has no cluster line, click popover still shows cluster id/label/confidence.
6. `pytest` — full suite green except the 2 pre-existing `test_features.py` failures.
