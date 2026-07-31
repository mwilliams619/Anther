# Graph Autoplay — Traverse the Map as a Playlist

**Status: Phase 0 PASSED (measured 2026-07-31). Phases 2–3 built and tested.
Phases 4–5 in progress.**

Make the song graph playable. A tour starts at a song, plays it, then walks an
edge to an unplayed neighbor and plays that, continuing until every reachable
song is played — then jumps to the nearest unplayed island and resumes, until
nothing is left unplayed.

---

## Decisions locked

| Decision | Choice |
|---|---|
| Node set | **Query nodes only** (`kind: 'query'` — what the Map panel lists). Grey `kind: 'corpus'` context nodes are never played and never routed through. |
| Traversal | **DFS with backtracking**, greedy on edge score. |
| Audio | **Spotify IFrame API**, one persistent controller, chained via `loadEntity`. |
| UI | **Now-playing transport bar** + **camera follows the playing node**. No queue panel, no per-node "play from here" (both are cheap to add later — see Deferred). |

## Relationship to FULL_SONG_PLAYBACK_NOTES.md

That doc reverted **YouTube** full-song playback. It explicitly *keeps* the
Spotify iframe as the full-song path: "gives a full track to visitors already
logged into Spotify and a 30s preview to everyone else." This plan chains that
existing, already-sanctioned embed. It does not reopen YouTube, and does not
proxy any audio through Flask.

Two constraints from that doc still bind and are honored here:

- Full-length audio must never reach the embedder. Autoplay is playback only;
  it never feeds `clip_waveform()` or any embedding path.
- No audio bytes transit the Flask host. The Spotify iframe streams directly
  from Spotify to the visitor's browser.

## What "full song" actually means here — state this in the UI

The Spotify embed plays a **full track only for a visitor already logged into
Spotify in that browser**. Everyone else gets a 30s preview. That is per-visitor
and we cannot detect it up front. The tour therefore behaves identically in both
cases except for hop length, and the transport bar carries a one-line note:
*"Log in to Spotify in this browser for full tracks."*

---

## Phase 0 — Gating spike (build nothing else until this passes)

Two unknowns can each kill the feature. Both are cheap to measure and neither is
tunable if it fails. A throwaway HTML page against a handful of hardcoded track
ids answers both in well under an hour.

**Spike A — does chained autoplay work without a per-track gesture?**
`IFrameAPI.createController()` builds its own iframe; we do not control its
`allow` attribute, so we cannot assume `allow="autoplay"` is delegated. Test:
one controller, press play once by hand, then `loadEntity(nextUri)` +
`resume()` on a timer for 3 tracks. If track 2 requires a click, the whole
feature degrades to a "next" button and the user should be told before we build
a transport bar around it.

**Spike B — can we detect end-of-track reliably?**
There is **no `ended` event**. The API exposes only `ready`, `playback_started`,
and `playback_update` (`{playingURI, isPaused, isBuffering, duration,
position}`). Worse, for an anonymous visitor `duration` reports the *full track
length* while audio stops at ~30s — so `position >= duration` never fires and
the chain would stall forever. Log raw `playback_update` frames across the last
5s of a track, in both a logged-in and a logged-out browser, and confirm the
stall signature below is real.

**Exit criteria:** A passes outright, or we accept a manual-advance fallback. B
yields a rule that fires exactly once per track in both auth states.

### Phase 0 results — measured 2026-07-31, logged-in Chrome

**Spike A: PASS.** Three tracks chained from a single user gesture, no manual
advance. The API-built iframe carries
`allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"`
— autoplay *is* delegated, so no manual-advance fallback is needed.

**Spike B: PASS.** The payload matches the docs exactly:
`{isPaused, isBuffering, duration, position, playingURI}`, all present.

Five findings from the raw log that change the Phase 4 implementation:

1. **`reset-to-zero` is the real end signal; `stall` never fired.** Every track
   ended with the identical two-frame signature: `position === duration` while
   `isPaused: false`, then immediately `position: 0, isPaused: true`. Make
   reset-to-zero the primary rule and keep stall only as a backstop.
2. **The `hasAdvanced` guard is load-bearing, not defensive.** Initial buffering
   holds `position: 0` for **~2.6–3.4s**, which is longer than `STALL_MS`
   (1500). Without the "never evaluate stall before the track has actually
   advanced" guard, every track would instantly report its own end at position
   0. This is the single easiest way to get Phase 4 wrong.
3. **`duration` jitters ±50ms between frames** (222973 / 223020 / 222974 for one
   track). Never test `position >= duration` against a cached duration, and
   never compare the two for equality.
4. **`ready` fires on every `loadUri`, not once per controller.** Track 2 and 3
   each emitted it. Any one-time initialisation hung off `ready` needs its own
   guard.
5. **`playback_started` arrives *after* the first `playback_update`.** Don't
   treat it as a precondition for handling position frames.

**Still unmeasured: the logged-out / 30s-preview regime.** This run was
logged-in, so all three tracks were full length. The claim that anonymous
visitors get 30s — and that the same reset-to-zero signature fires at the
preview cutoff — has not been verified. Phase 4 keeps both rules for exactly
this reason; re-run the spike in a private window to close it.

---

## End-of-track detection (the spec Spike B validates)

Do **not** compare `position` to `duration`. Use a stall detector, which is
correct in both the full-track and 30s-preview cases:

```
on playback_update(e):
  if e.position > lastPosition: lastAdvanceAt = now()   // still playing
  lastPosition = e.position
  ended if  playbackStarted
        and e.isPaused
        and (e.position === 0 && lastPosition > 0        // reset to top
             || now() - lastAdvanceAt > STALL_MS)        // stopped advancing
```

- `STALL_MS ≈ 1500`. Guard with a `hasAdvanced` flag so the paused-at-zero state
  before the first play never counts as an end.
- **Advance exactly once per track**: latch on `playingURI`. Ignore any `ended`
  verdict whose `playingURI` is not the track we are currently tracking. This is
  the main bug risk — `loadEntity` emits updates for both old and new URIs
  across the swap.
- **User-pause must not advance.** Set an `userPaused` flag in the transport
  bar's pause handler and suppress the stall rule while it is set.
- **Buffering watchdog**: `isBuffering` continuously for >20s, or no `ready`
  within 15s of `loadEntity`, ⇒ treat the track as failed, mark it played, and
  advance. Prevents a dead track from ending the tour silently.

## Player lifecycle

One controller for the whole session, created lazily on first Play, mounted in
the transport bar. Reuse it via `loadEntity(uri)` for every hop. Never
`destroy()` and recreate mid-tour — recreating risks losing the user-activation
that makes chained autoplay work, which is precisely what Spike A measures.

This is separate from the existing detail-panel embed (`#spotify-embed-wrap`,
[`renderSpotifyEmbed`](../../ui/static/app.js)). Starting a tour must close that
embed, and opening that embed must pause the tour — two players must never sound
at once. Same rule for the existing preview `Audio` in `state.audio`.

---

## Traversal algorithm

Operates on the **query-node subgraph**: nodes with `kind === 'query'`, and
`kind: 'qq'` links where both endpoints are query nodes.

Edges are real: [`atlas.py:1409-1469`](../../ui/atlas.py#L1409) creates
query↔query links above a percentile threshold (`QUERY_LINK_PCTL`, default 95),
capped at `QQ_MAX_PER_NODE = 6` per placed song, each carrying a display
`score`. A 95th-percentile cutoff means **isolated query nodes are normal, not
exceptional** — the island-jump path is a main code path, not an edge case.

```
played   = Set()          // ordered; drives history + "already played" styling
stack    = [startId]      // DFS spine, enables backtracking

next():
  while stack:
    cur = stack.top
    cand = unplayed query-neighbors of cur, sorted by link.score desc
    if cand: pick cand[0]; stack.push(it); return it
    stack.pop()                                  // dead end → backtrack
  return jump()                                  // spine exhausted → new island

jump():
  rest = query nodes not in played
  if rest empty: tour complete
  target = nearest(rest) to the last-played node    // server call
  stack = [target]; return target
```

- **Start node**: the pinned node if one is selected (`AtlasGraph.getSelectedId()`),
  else the first Map-panel node.
- **Ordering by `score`** (the calibrated 0–100 display value already on each
  link), so each hop is the most similar unplayed neighbor.
- **Backtracking is silent** — popping the stack plays nothing; only the
  eventual new node plays.
- **`played` is authoritative over the stack.** A node reached twice by
  different routes is played once.

### Island jumps

The client only knows *thresholded* edges, so it cannot rank non-adjacent pairs
itself. One new endpoint answers "nearest unplayed":

`POST /api/autoplay/jump` → body `{from: <song_id>, exclude: [<song_id>…]}` →
`{id, score}` or `{id: null}` when nothing is left.

Server-side this is a cosine over `st.merit_vecs` (falling back to
`st.query_vecs` when the bundle has no MERIT index, mirroring the `use_merit`
branch at [`atlas.py:1394`](../../ui/atlas.py#L1394)) restricted to live query
nodes not in `exclude`. No threshold applies — a jump is by definition below the
link cutoff. Cheap: a dot product against at most a few hundred session vectors.

**Fallback if the call fails:** jump to the next unplayed node in Map-panel
order. The tour must never stop because of a failed request.

---

## Spotify id resolution and skips

`get_spotify_track_id` ([`atlas.py:1827`](../../ui/atlas.py#L1827)) is free for
`spotify:`-prefixed ids (every MPD-sourced row carries one) and otherwise costs
a Spotify Search API call, cached per id. Coverage is therefore excellent for
corpus/MPD tracks, unreliable for Deezer-origin ones, and **always null for
uploads**.

- **New endpoint** `POST /api/autoplay/resolve` → body `{ids: […]}` →
  `{id: track_id|null}`. Batches what `/api/song/<id>/spotify` does per-song, so
  the tour can pre-check without N round trips.
- **Resolve-ahead**: on tour start, resolve the first ~10 hops; keep a rolling
  lookahead of 3 during playback. Resolution latency then hides under the
  previous track instead of gapping between them.
- **Skip policy**: a node with no Spotify id is marked played, flagged
  `unplayable` in the transport bar's history, and traversal continues **through
  it** — its neighbors stay reachable. It is a routing node, not a wall.
- **All-unplayable guard**: if the resolver returns null for every node in the
  map, stop and surface "No tracks on this map are available on Spotify"
  instead of silently walking a dead graph. Likely for an upload-only map.

---

## Camera and graph feedback

- On each hop, `AtlasGraph.zoomTo(id)` and select the node so the existing
  highlight/detail behavior applies.
- **Playing node**: a `.playing` class — pulsing ring, distinct from `.selected`.
- **Traversed edge**: animate the edge from previous → current so the walk is
  visible. Backtracking should *not* animate (nothing is playing during it).
- **Played nodes** get a `.played` class (muted fill), so the map doubles as
  tour progress.
- **Camera-follow is a toggle** in the transport bar, on by default. A user who
  is dragging the map should not be yanked around; disable follow automatically
  on manual pan/zoom during a tour and let the toggle re-enable it.

## Live-map mutations during a tour

The map is not static — playlist/album imports stream nodes in
(`startPlaylistPoll`), and nodes can be removed.

- **Nodes added mid-tour** are naturally picked up: candidates are recomputed
  from `AtlasGraph` at each hop, never from a precomputed queue. This is the
  main reason not to materialize the whole path up front.
- **Current node removed** ⇒ advance immediately.
- **Node removed while on the stack** ⇒ filter it out of the stack lazily at
  `next()`.
- **Map cleared** (`/api/graph/clear`) ⇒ stop the tour, reset `played`.

---

## Files touched

| File | Change |
|---|---|
| `ui/static/autoplay.js` | **New.** Self-contained `AtlasAutoplay` module: DFS state, Spotify controller lifecycle, end-detection, resolve-ahead. Mirrors the isolation style of `graph.js`. |
| `ui/static/graph.js` | Export `getLinks()`, `neighborsOf(id)` (the internal `adjacency` map is already maintained by `rebuildAdjacency`), `setPlayingNode(id)`, `setPlayed(ids)`, and an `onManualPan` hook for the follow toggle. No behavior change to existing exports. |
| `ui/static/index.html` | Transport bar markup; `<script src="/static/autoplay.js">`; bump the `?v=` on `graph.js`/`app.js`. |
| `ui/static/style.css` | Transport bar, `.playing` pulse, `.played` muted node, edge-walk animation. |
| `ui/static/app.js` | Wire the Map panel's play control to `AtlasAutoplay.start()`; mutual exclusion with `state.audio` and `#spotify-embed-wrap`. |
| `ui/app.py` | Two routes: `POST /api/autoplay/resolve`, `POST /api/autoplay/jump`. Thin, per the "routes only" rule. |
| `ui/atlas.py` | `resolve_spotify_batch(ids)` and `nearest_unplayed(from_id, exclude)`. |
| `docs/ui.md` | Route table + a "Graph autoplay" section. |
| `CLAUDE.md` | Point the Playback row at this doc alongside the notes. |

## Build order

1. **Phase 0** — the spike. Gate.
2. **Traversal, headless** — DFS + backtrack + jump against a fixture graph,
   with no audio at all. Pure function over `{nodes, links}` ⇒ unit-testable,
   and the part most likely to have subtle bugs.
3. **Endpoints** — `resolve` + `jump`, with tests in `tests/`.
4. **Player** — controller lifecycle, end-detection, resolve-ahead, mutual
   exclusion with the two existing players.
5. **UI** — transport bar, camera follow, node/edge states.
6. **Docs** — `docs/ui.md`, `CLAUDE.md`.

## Test plan

Traversal is deterministic and gets real unit tests; the player is manual.

- **Unit (JS fixture graph):** a path graph plays end to end in order; a star
  graph backtracks through the hub exactly once; two disconnected components
  produce exactly one jump; a fully isolated set produces N−1 jumps; every node
  plays exactly once in every case; an unplayable node is skipped but still
  routed through.
- **Server (`tests/`):** `nearest_unplayed` excludes played ids, ignores corpus
  nodes, returns null on an exhausted map; `resolve_spotify_batch` returns null
  for uploads and passes `spotify:`-prefixed ids through without an API call.
- **Manual:** logged-in and logged-out browsers (the two `duration` regimes);
  pause/resume mid-tour; remove the playing node; import a playlist mid-tour;
  drag the map with follow on.

## Open risks

1. **Spike A fails** (autoplay needs a gesture per track) ⇒ the feature becomes
   a manual "next" button. Decide with the user at that point; do not silently
   ship a degraded tour.
2. **Anonymous visitors get a 30s hop.** Unavoidable, per-visitor, disclosed in
   the bar. This is the same trade the 🎵 button already makes today.
3. **Sparse query subgraph.** At a 95th-percentile threshold with ≤6 edges per
   node, a varied map may be mostly islands, making the tour feel like shuffle
   rather than a walk. Worth measuring on a real map early — if islands
   dominate, the honest fix is exposing the jump as "next nearest" rather than
   pretending the graph was traversed.
4. **`playingURI` latching** across `loadEntity` is the likeliest source of
   double-advance bugs. Build it in Phase 4 with the spike's raw event log open.

## Deferred (cheap to add once the core works)

Upcoming-path queue panel; shuffle/repeat; persisting `played` across reloads;
preview-audio fallback for tracks with no Spotify id (the skip path already has
the hook).

**Since built:**

- **Per-node "play from here"** — the `⇉` button in the detail panel
  (`playFromHere` in `app.js`). Shown only on `kind: 'query'` nodes, since the
  traversal walks the query subgraph and a corpus node would silently seed the
  tour somewhere else. Starting from a node **replaces** any running tour, so
  the played history always matches what the map shows.
- **Shift-click to steer** — redirects a *running* tour while **keeping**
  history (`tour.steerTo`). It pushes onto the DFS spine rather than resetting
  it, so steering to a dead end unwinds back and still covers what the previous
  branch could reach.
- **Settings popover** (⚙, left-panel header) — chooses the single-song player
  (Deezer preview vs Spotify embed) and toggles tour auto-advance. Tours stay
  Spotify-only: it's the only source that can play a full track.
- **"▶ Play map" removed** as redundant once `⇉` and shift-click existed; a
  tour is now always seeded from a specific song rather than from map order.
- **Manual stepping** — with auto-advance off, a finished song parks the tour in
  a `waiting` state instead of advancing, and the transport bar prompts for ⏭.
