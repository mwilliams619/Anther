# Web UI (`ui/`)

Flask backend + d3 force-graph frontend for the song atlas: search, place
songs/playlists/albums onto the frozen reference corpus, and browse the
resulting map. `ui/app.py` is a thin router; `ui/atlas.py` owns all
corpus/MERT state and does the real work.

```bash
python ui/app.py   # port 5000
```

## Shape

| Piece | Responsibility |
|---|---|
| `ui/app.py` | Flask routes only — no business logic, delegates to `atlas` |
| `ui/atlas.py` | Corpus warm-up; three-tier song search (local corpus → Deezer → Spotify); `place()` onto the frozen corpus; full-MPD playlist/album search + placement; raw-MERT embed cache |
| `ui/playlist_jobs.py` | Single background worker thread (one GPU consumer) that embeds and places queued tracks asynchronously, polled via `/api/playlist/status/<job_id>` |
| `ui/static/graph.js` | d3 force graph — hover highlight/tooltip, click-to-pin, warm-up polling |
| `ui/static/app.js` | Search UI, mode switching (tracks/playlists/albums), detail popover (`renderDetail`) |

## Routes (`ui/app.py`)

| Route | Purpose |
|---|---|
| `/` | Serves the app shell |
| `GET /api/search` | Song search (three-tier) |
| `GET /api/playlists/search` | Full-MPD playlist search |
| `POST /api/playlist/place` | Place a playlist's tracks onto the map |
| `GET /api/playlist/status/<job_id>` | Poll async placement progress |
| `GET /api/albums/search` | Deezer album search |
| `POST /api/album/place` | Place an album's tracks onto the map |
| `POST /api/place` | Place a single song (cache-first: previously embedded ids skip download+MERT) |
| `GET /api/graph` | Current session graph (nodes/links/groups); `ready: false` during warm-up |
| `POST /api/graph/clear` | Wipe the map — nodes, links, groups (embed cache kept) |
| `DELETE /api/node/<id>` | Remove one placed song + its now-orphaned corpus neighbors |
| `GET /api/song/<id>` | Song detail (cluster, tags, similar songs) |
| `POST /api/upload` | Upload a personal track for placement |

## Runtime facts worth knowing

- **Warm-up race**: the corpus takes ~1–2 min to load after `python ui/app.py`
  starts. `/api/graph` returns `ready: is_ready()` so `graph.js` can show a
  "loading corpus…" hint and poll until ready, instead of rendering a blank
  map.
- **Env vars**: `ANTHER_CORPUS` (corpus bundle dir), `ANTHER_MPD_DB` (MPD
  SQLite DB path), `ANTHER_IMPORT_CAP` (playlist/album placement cap, default
  100), `ANTHER_DEBUG` (Flask debug mode, default off — leave off on any
  internet-facing host).
- **Session state** lives under `ui/session/`: `embed_cache.sqlite` (raw
  MERT vectors, avoids re-embedding on repeat placement) and the saved graph
  JSON (persists the map across restarts).
- **Clusters are display-only in the UI too**: computed and stored on every
  node, but only surfaced in the click-detail popover (`renderDetail`) —
  the map itself is not fill-colored by cluster. Full rationale:
  [projects/UI_ATLAS_FIX_PLAN.md](projects/UI_ATLAS_FIX_PLAN.md).
- **Map panel** (left side, `renderMapPanel` in `app.js`): lists every placed
  song with click-to-zoom, per-node remove (`DELETE /api/node/<id>` — also
  prunes orphaned grey neighbors), a session-only recently-removed list with
  one-click re-place (instant via the embed cache), a clear-map button, and an
  artist/playlist/album filter that dims non-matching nodes
  (`AtlasGraph.setFilter`). Group display names come from a `groups` registry
  persisted inside `graph.json` (backfilled at load from the MPD DB /
  Deezer for pre-registry imports).
