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
| `ui/static/app.js` | Search UI, mode switching (tracks/playlists/albums), detail popover (`renderDetail`), recommend-from-map, mentor chat panel |
| `mentor/service.py` | Separate warm process hosting `MusicMentor` (chat/ReAct) over HTTP; `ui/app.py` forwards `/api/mentor/*` to it (see "Mentor chat" below) |

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
| `POST /api/recommend` | Multi-song recommendation: body `{seed_ids, top_k?, method?}` → similar corpus tracks, spliced into the graph server-side |
| `POST /api/mentor/chat` | Body `{question}` → `{answer}`; forwards to `mentor/service.py` using a per-browser session cookie. `503` if the mentor service isn't running |
| `POST /api/mentor/reset` | Clears the caller's mentor conversation state |
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
  internet-facing host), `ANTHER_UI_SECRET_KEY` (Flask session-cookie signing
  key; a random one is generated per restart if unset, which just resets
  mentor chat sessions), `ANTHER_MENTOR_HOST`/`ANTHER_MENTOR_PORT` (where
  `ui/app.py` reaches the mentor service, default `127.0.0.1:5100`),
  `ANTHER_MENTOR_TIMEOUT` (request timeout in seconds, default 30).
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
- **Recommend from map**: the "Recommend similar" button (`doRecommend` in
  `app.js`) seeds `/api/recommend` with every `kind: 'query'` node currently
  on the map (i.e. what's shown in the Map panel list, not corpus-neighbor
  context nodes). Results are already spliced into the shared graph
  server-side (`atlas.recommend`'s `splice=True` default); the frontend just
  mirrors that into the local d3 model as `kind: 'corpus'` nodes and lists
  them for click-to-zoom.

## Mentor chat

`mentor/service.py` loads `MusicMentor` once (base model + LoRA + RAG +
Anther graph tools — ~10s, ~2.6GB VRAM resident per
[`mentor/README.md`](../mentor/README.md)) and stays warm as its own process,
independent of `ui/app.py`'s lifecycle:

```bash
python -m mentor.service   # port 5100, separate process — start alongside ui/app.py
```

`ui/app.py` never loads the model itself; it forwards `/api/mentor/chat` and
`/api/mentor/reset` to the service over localhost, using a signed session
cookie (`ANTHER_UI_SECRET_KEY`) to key each browser's `MentorContext`
inside the service. If the service isn't running, both routes return `503`
and the chat panel shows an inline "unavailable" state — the rest of the
atlas UI keeps working normally.

**The mentor reads the caller's live map.** The service runs in a separate
process and cannot see the UI's request-scoped atlas session, so `ui/app.py`
sends two extra fields on every `/api/mentor/chat` turn:

- `graph_session_id` — the browser's `graph_session_id` cookie, i.e. which
  `ui/session/<sid>/graph.json` map this chat is about. The mentor's
  `GraphContext` re-points at that session per turn and reads the graph +
  `embed_cache.sqlite` straight off disk (see `mentor/graph_context.py`).
- `selected_node_id` — the node pinned in the force graph (`AtlasGraph.getSelectedId()`),
  so "this song" / "why is this here" / "what do I sound like" resolve to it.

Without `graph_session_id` the mentor would read the shared `default` session
(or nothing) and answer as if the map were empty. The full chat body is
`{question, selected_node_id, graph_session_id}` → `{answer}`.

The chat pipeline (`mentor/agent.py`) is: **LLM classifies intent → deterministic
graph tool → LLM narrates the observation**, with a graph-aware stage-1 router
(`mentor/mentor.py`) that recognises map language and on-map entities so map
questions never leak to the generic advice/off-topic branches. The seven
read-only tools (`inspect, resolve, neighbors, compare, bridge, explore,
explain` — see `mentor/graph_tools.py`) cannot place, remove, or mutate the
map; use the atlas UI controls for that. Set `ANTHER_MENTOR_TRACE=1` on the
service to log the full QUESTION → CLASSIFICATION → TOOL → OBSERVATION →
ANSWER trace per turn.
