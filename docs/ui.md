# Web UI (`ui/`)

Flask backend + d3 force-graph frontend for the song atlas: search, place
songs/playlists/albums onto the frozen reference corpus, and browse the
resulting map. `ui/app.py` is a thin router; `ui/atlas.py` owns all
corpus/MERT state and does the real work.

```bash
source .venv/bin/activate
python ui/app.py   # port 5000; use the corpus-compatible Python 3.11 env
```

## Shape

| Piece | Responsibility |
|---|---|
| `ui/app.py` | Flask routes only — no business logic, delegates to `atlas` |
| `ui/atlas.py` | Corpus warm-up; three-tier song search (local corpus → Deezer → Spotify); `place()` onto the frozen corpus; full-MPD playlist/album search + placement; raw-MERT embed cache |
| `ui/playlist_jobs.py` | Single background worker thread (one GPU consumer) that embeds and places queued tracks asynchronously, polled via `/api/playlist/status/<job_id>` |
| `ui/static/graph.js` | Isolated song force graph — hover highlight/tooltip, click-to-pin, warm-up polling |
| `ui/static/artist-graph.js` | Isolated, incremental artist force graph; only explicitly-added artists are rendered |
| `ui/static/app.js` | Song/artist view switching, search, detail popovers, uploads, recommend-from-map, mentor chat panel |
| `mentor/service.py` | Separate warm process hosting `MusicMentor` (chat/ReAct) over HTTP; `ui/app.py` forwards `/api/mentor/*` to it (see "Mentor chat" below) |

## Routes (`ui/app.py`)

| Route | Purpose |
|---|---|
| `/` | Serves the app shell |
| `GET /api/search` | Song search (three-tier) |
| `GET /api/playlists/search` | Full-MPD playlist search |
| `POST /api/playlist/place` | Place a playlist's tracks onto the map |
| `GET /api/playlist/status/<job_id>` | Poll async placement progress, including stop availability |
| `POST /api/playlist/stop/<job_id>` | Gracefully stop a running playlist/album embedding job after the current track |
| `GET /api/albums/search` | Deezer album search |
| `POST /api/album/place` | Place an album's tracks onto the map |
| `POST /api/place` | Place a single song (cache-first: previously embedded ids skip download+MERT) |
| `POST /api/recommend` | Multi-song recommendation: body `{seed_ids, top_k?, method?}` → similar corpus tracks, spliced into the graph server-side |
| `POST /api/mentor/chat` | Body `{question}` → `{answer}`; forwards to `mentor/service.py` using a per-browser session cookie. `503` if the mentor service isn't running |
| `POST /api/mentor/reset` | Clears the caller's mentor conversation state |
| `GET /api/graph` | Current session graph (nodes/links/groups); `ready: false` during warm-up |
| `POST /api/graph/clear` | Wipe the map — nodes, links, groups (embed cache kept) |
| `DELETE /api/node/<id>` | Remove one placed song + its now-orphaned corpus neighbors |
| `GET /api/song/<id>` | Song detail (cluster, tags, similar songs — see "Similarity scoring" below for the aggregate/breakdown split) |
| `POST /api/upload` | Upload a personal track for placement |
| `GET /api/artist/search` | Search immutable ≥5-track corpus artists, the 1-4 track low-confidence pool, plus this session's private artist profiles |
| `GET /api/artist/graph` | Current session's incremental artist graph |
| `POST /api/artist/place` | Add one explicitly-selected artist and threshold-clearing links; a `lowconf:` artist is auto-supplemented to 5 tracks on placement (see below) |
| `POST /api/artist/<id>/supplement` | Manual re-trigger of the same supplement pass for a `lowconf:` artist (pull extra strict-artist-matched Deezer previews up to 5 tracks, embed, re-place at higher confidence; session-only) — backs the "less confident" detail-panel note |
| `GET /api/artist/<id>` | Artist detail and currently connected artists; includes `low_confidence` (placed from <5 tracks) plus an optional `profile` object (image/following/genres/origin/labels) when the artist has been enriched — see [artist-enrichment.md](artist-enrichment.md) |
| `DELETE /api/artist/node/<id>` | Remove an artist from this session's artist graph |
| `POST /api/artist/graph/clear` | Clear only the artist graph |
| `POST /api/artist/demo` | Idempotently add the curated 20-artist demo |
| `GET /api/artist/status` | Report whether Artist View artifacts are available |
| `POST /api/artist/from-song-graph` | Build or append an artist graph from the current song map; artists not in the frozen corpus are built as session artists from their on-map songs, and any artist with <5 tracks is auto-supplemented to 5 total (see below) |
| `POST /api/demo/load` | Load the curated demo playlist |
| `GET /api/song/<id>/preview` | Resolve a playable preview URL when available |
| `GET /api/song/<id>/spotify` | Resolve a Spotify track link when credentials/metadata permit |
| `GET /api/upload-audio/<name>` | Serve an uploaded audio file for playback |

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
- **Artist profiles**: `ANTHER_ARTIST_PROFILES` (default `data/artist_profiles.sqlite`)
  points at the optional enrichment DB loaded read-only at warm-up into
  `_artist_profiles`. Absent DB → no profile fields, no error. Built offline via
  `python -m anther_ml.artist_enrichment` (see [artist-enrichment.md](artist-enrichment.md)).
- **Popularity reranking**: `ANTHER_POPULARITY` optionally points to the
  track-percentile sidecar used for Billboard-aware reranking. The UI's
  popularity blend is currently 0.15; if the sidecar is absent, reranking is
  disabled and normal similarity ordering is used.
- **Artist mode**: `ANTHER_ARTIST_MODE=0` is an emergency kill switch. Artist
  corpus artifacts are immutable and must have identical row counts; invalid
  artifacts disable Artist View without affecting Song View. Artist graphs,
  uploaded-track associations, and newly-created artist profiles are private
  per browser session under `ui/session/<sid>/`.
- **Low-confidence artist pool** (`lowconf:<idx>` ids): the frozen bundle only
  aggregates artists with ≥5 corpus tracks (the clustering *reference frame*).
  Artists with 1-4 tracks are indexed in memory at load (`_build_lowconf_pool`
  in `atlas.py`, one pass over `corpus.metadata`) and kNN-assigned into that
  frozen reference on demand — exactly how Song View places a non-corpus song,
  and how private/session artists already work. They render with a dashed,
  translucent node and a `low_confidence` flag; the frozen bundle and its Leiden
  labels are never recomputed. Placing a `lowconf:` artist (via
  `POST /api/artist/place`, or as part of a from-song-graph build) auto-runs the
  supplement pass first: it pulls extra strict-artist-matched Deezer previews
  (up to 5 tracks total, counting the artist's existing tracks) through the same
  30s-preview embed path as a song placement, so the node usually lands at full
  confidence. The supplement is best-effort — a fetch/embed failure just leaves
  the artist low-confidence, and clicking the "less confident" detail-panel note
  re-triggers it via `POST /api/artist/<id>/supplement`. All session-only, never
  written back to the bundle. The frontend shows the same "Placing…" wait a
  Deezer song placement does while this runs.
- **Building artists from the song map** (`POST /api/artist/from-song-graph`):
  each on-map song's artist is resolved to a ≥5-track corpus artist, a 1-4 track
  `lowconf:` artist, or — when the name is absent from the corpus entirely — a
  new **session artist** built from that artist's on-map songs
  (`_session_artist_from_songs` in `atlas.py`). Both corpus-placed songs (whose
  raw vector is copied out of the frozen corpus into embed_cache) and
  Deezer/upload/MPD songs (already cached) contribute. Session and low-confidence
  artists are auto-supplemented to 5 tracks the same way as above (on-map songs
  count toward the target). An artist is only skipped when none of its on-map
  songs have a usable embedding yet.
- **Session state** lives under `ui/session/`: `embed_cache.sqlite` (raw
  MERT vectors, avoids re-embedding on repeat placement) and the saved graph
  JSON (persists the map across restarts).
- **Clusters are display-only in the UI too**: computed and stored on every
  node, but only surfaced in the click-detail popover (`renderDetail`) —
  the map itself is not fill-colored by cluster. Full rationale:
  The historical implementation rationale is retained in
  `private/implemented_archive/UI_ATLAS_FIX_PLAN.md` (local-only).
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
  context nodes). It uses MERIT-aggregate `topk` ranking by default: a result
  is rewarded for fitting the strongest subset of the seeds, which preserves
  distinct moods in a varied map. Every existing map node is excluded. Results
  are spliced into the shared graph as `kind: 'corpus'` context nodes, and the
  frontend mirrors that same kind, so recommendations never become future seeds
  merely because the graph reloads.

## Similarity scoring — aggregate score + expandable breakdown

Map edges, corpus-wide search, and `/api/song/<id>`'s ranking are all driven by
the MERIT-aggregate `SongIndex` (see [docs/similarity.md](similarity.md)),
not the plain MERT-1024 index — a link on the map means "high melody+rhythm+
timbre agreement," not just "nearby in one opaque embedding."

- **Connection rows default to one number.** Every similar-song row in the
  detail panel (`similarRowHtml` in `app.js`) shows a single 0–100 aggregate
  similarity score, calibrated per-corpus via `link_calibration_merit.json`
  (see `_display_score` in `atlas.py`). This is deliberate: most of the time
  a user just wants to know "how similar," not "similar along which axis."
- **Expand for the "why."** When the corpus carries a MERIT-aggregate index,
  `song_detail()` also returns a `breakdown: {melody, rhythm, timbre,
  aggregate}` dict per row (`_breakdown_scores`/`_merit_breakdown` in
  `atlas.py` — each factor is its own 128-d cosine, mapped through the same
  calibrated display scale but *without* the map's score floor, so a low
  factor score still reads honestly). The row renders a caret button
  (`.btn-expand`) that toggles a hidden `.breakdown-panel` showing three
  labeled bars — melody, rhythm, timbre — so a user can see *why* two songs
  matched (e.g. "same rhythm, different timbre") instead of just a number.
  Rows without breakdown data (older MERT-only bundles) render the plain
  score with no expand affordance — the UI degrades gracefully rather than
  showing a dead button.
- **Corpora built before this integration** (no `index_merit_agg.npy`) fall
  back to the legacy MERT-1024 scoring path automatically — `atlas.py` checks
  `corpus.merit_index is not None` before routing to MERIT space, so nothing
  breaks on an old bundle; it just won't have per-factor breakdowns.

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
