/* Graph autoplay — player, queue and camera.
 *
 * Walks the query-node subgraph as a playlist (see autoplay-traversal.js for
 * the DFS/backtrack/jump policy) and plays each song through ONE persistent
 * Spotify IFrame controller.
 *
 * Everything about end-of-track detection below is driven by what Phase 0
 * actually measured (docs/projects/GRAPH_AUTOPLAY_PLAN.md), not by the docs:
 *
 *   - There is no 'ended' event. The real signal is a two-frame signature:
 *     position === duration while isPaused:false, then position:0 isPaused:true.
 *     That "reset-to-zero" is the primary rule.
 *   - `stall` is only a backstop, and it is dangerous on its own: initial
 *     buffering holds position at 0 for ~2.6-3.4s, well past STALL_MS. Nothing
 *     may be judged an ending before the track has actually advanced once —
 *     that is what `hasAdvanced` is for.
 *   - `duration` jitters +-50ms between frames, so it is never compared for
 *     equality and never cached as a target.
 *   - 'ready' fires on EVERY loadUri, not once per controller.
 *   - 'playback_started' arrives AFTER the first playback_update.
 *
 * A visitor logged into Spotify in this browser hears full tracks; everyone
 * else hears the same ~30s preview the detail panel's Spotify button already
 * gives them. We can't detect which, and don't need to: both regimes end with
 * the same reset-to-zero frame.
 */
const AtlasAutoplay = (() => {
  'use strict';

  const STALL_MS       = 1500;   // backstop only; never evaluated pre-advance
  const BUFFER_LIMIT   = 20000;  // continuous buffering ⇒ treat track as failed
  const READY_LIMIT    = 15000;  // no playback after loadUri ⇒ skip the track
  const LOOKAHEAD      = 3;      // hops to resolve ahead of the current song
  const PRIME_COUNT    = 10;     // hops resolved when a tour starts
  const SILENT_LIMIT   = 15;     // consecutive unresolvable songs ⇒ give up

  let controller = null, apiPending = false;
  let tour = null, running = false, userPaused = false;
  let currentId = null, currentUri = null;
  let lastPosition = 0, lastAdvanceAt = 0, hasAdvanced = false;
  let bufferingSince = 0, advancedThisTrack = false, consecutiveSilent = 0;
  let autoAdvance = true;        // off ⇒ stop after each song and wait for ⏭
  let waiting = false;           // a song has ended and auto-advance is off
  let watchdog = null;
  let followCamera = true;
  const trackIds = new Map();    // song id → Spotify track id | null (resolved)
  let onChange = () => {};

  /* ── Spotify controller ──────────────────────────────────────────────────
   * One controller for the whole session, created lazily on the first Play and
   * reused via loadUri. Never destroyed mid-tour: re-creating it risks losing
   * the user activation that makes chained autoplay work at all. */
  function ensureApi() {
    return new Promise((resolve, reject) => {
      if (window.__spotifyIframeApi) return resolve(window.__spotifyIframeApi);
      const prev = window.onSpotifyIframeApiReady;
      window.onSpotifyIframeApiReady = api => {
        window.__spotifyIframeApi = api;
        if (typeof prev === 'function') { try { prev(api); } catch (e) {} }
        resolve(api);
      };
      if (apiPending) return;
      apiPending = true;
      const s = document.createElement('script');
      // No `.js` extension — /embed/iframe-api/v1.js 404s. This serves a loader
      // stub that injects the real bundle from embed-cdn.spotifycdn.com.
      s.src = 'https://open.spotify.com/embed/iframe-api/v1';
      s.async = true;
      s.onerror = () => reject(new Error('Spotify embed API failed to load'));
      document.head.appendChild(s);
      setTimeout(() => reject(new Error('Spotify embed API timed out')), 10000);
    });
  }

  function ensureController() {
    if (controller) return Promise.resolve(controller);
    return ensureApi().then(api => new Promise(resolve => {
      const host = document.getElementById('autoplay-embed');
      // createController REPLACES the element it is given, so #autoplay-embed
      // will not exist afterwards — never query it again, use its wrapper.
      api.createController(host, { width: '100%', height: 80 }, c => {
        controller = c;
        c.addListener('playback_update', onUpdate);
        c.addListener('error', () => skipCurrent('playback error'));
        resolve(c);
      });
    }));
  }

  /* ── end-of-track detection ── */
  function onUpdate(e) {
    const d = (e && e.data) || e || {};
    if (!running || d.position === undefined) return;

    // loadUri emits frames for both the outgoing and incoming track across a
    // swap; without this latch a single ending advances the tour twice.
    if (d.playingURI && currentUri && d.playingURI !== currentUri) return;

    if (d.isBuffering) {
      if (!bufferingSince) bufferingSince = Date.now();
      if (Date.now() - bufferingSince > BUFFER_LIMIT) return skipCurrent('stuck buffering');
    } else {
      bufferingSince = 0;
    }

    if (d.position > lastPosition) { lastAdvanceAt = Date.now(); hasAdvanced = true; }
    const prev = lastPosition;
    lastPosition = d.position;

    if (advancedThisTrack || !hasAdvanced) return;
    if (userPaused) return;              // a manual pause is not an ending

    let ended = false;
    if (d.isPaused && d.position === 0 && prev > 0) ended = true;               // primary
    else if (d.isPaused && Date.now() - lastAdvanceAt > STALL_MS) ended = true; // backstop
    if (!ended) return;

    advancedThisTrack = true;
    if (!autoAdvance) {
      // Song over, but the user wants to step manually. Hold position and let
      // the transport's ⏭ (or a shift-click steer) decide what happens next.
      waiting = true;
      clearTimeout(watchdog);
      onChange();
      return;
    }
    advance();
  }

  /* ── queue ── */
  function graphView() {
    return { nodes: AtlasGraph.getNodes(), links: AtlasGraph.getLinks() };
  }

  async function resolve(ids) {
    const need = ids.filter(id => !trackIds.has(id));
    if (!need.length) return;
    try {
      const r = await fetch('/api/autoplay/resolve', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: need }),
      }).then(r => r.json());
      Object.keys(r.tracks || {}).forEach(id => trackIds.set(id, r.tracks[id]));
      need.forEach(id => { if (!trackIds.has(id)) trackIds.set(id, null); });
    } catch (err) {
      need.forEach(id => trackIds.set(id, null));
    }
  }

  async function askJump(from, exclude) {
    try {
      const r = await fetch('/api/autoplay/jump', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from: from, exclude: exclude }),
      }).then(r => r.json());
      return r.id || null;
    } catch (err) {
      return null;                      // planner falls back to map order
    }
  }

  // Resolve likely next hops so lookup latency hides under the current song
  // instead of gapping between songs. The planner can't be run ahead without
  // consuming its state, so this approximates: the current node's unplayed
  // neighbours are where DFS goes next, and map order fills the rest for the
  // eventual island jump. Over-resolving is harmless — results are cached and
  // a spotify:-prefixed id costs no API call at all.
  async function prefetch(n) {
    const played = new Set(tour.getPlayed());
    const unplayed = id => !played.has(id);
    const ahead = [];
    if (currentId) {
      AtlasGraph.neighborsOf(currentId).filter(unplayed).forEach(id => ahead.push(id));
    }
    graphView().nodes
      .filter(x => x.kind === 'query' && unplayed(x.id) && ahead.indexOf(x.id) === -1)
      .forEach(x => ahead.push(x.id));
    if (ahead.length) await resolve(ahead.slice(0, n));
  }

  /* ── transport ── */
  async function step(res) {
    if (!res || res.type === 'done') return finish();

    if (res.type === 'need-jump') {
      const id = await askJump(res.from, res.exclude);
      return step(tour.resumeWithJump(id));
    }

    waiting = false;
    currentId = res.id;
    await resolve([currentId]);
    const tid = trackIds.get(currentId);

    if (!tid) {
      // No Spotify id (uploads never resolve). Mark it silent and keep going —
      // the node still routes the tour to its neighbours, it just doesn't sound.
      tour.markUnplayable(currentId);
      consecutiveSilent++;
      onChange();
      // Give up only after a long silent run. Counting *consecutive* skips
      // rather than a ratio matters: a map that opens with a handful of
      // uploads would otherwise end the tour before reaching its real songs.
      if (consecutiveSilent >= SILENT_LIMIT) {
        return finish('No playable Spotify tracks found on this map.');
      }
      return step(tour.next());
    }
    consecutiveSilent = 0;

    currentUri = 'spotify:track:' + tid;
    lastPosition = 0; hasAdvanced = false; advancedThisTrack = false;
    bufferingSince = 0; lastAdvanceAt = Date.now();

    AtlasGraph.setPlayingNode(currentId);
    if (followCamera) AtlasGraph.zoomTo(currentId);
    onChange();

    // An ad blocker on open.spotify.com rejects here. Without this catch the
    // rejection is unhandled and the tour just stops with no explanation.
    let c;
    try {
      c = await ensureController();
    } catch (err) {
      return finish('Spotify player unavailable: ' + err.message);
    }
    if (!running) return;              // stopped while the API was loading

    c.loadUri(currentUri);
    // Deliberately no user gesture here — Phase 0 confirmed the API-built
    // iframe carries allow="autoplay", so chained playback needs none.
    setTimeout(() => { if (running && !userPaused) c.resume(); }, 400);

    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      if (running && currentId === res.id && !hasAdvanced) skipCurrent('no playback');
    }, READY_LIMIT);

    prefetch(LOOKAHEAD);
  }

  function skipCurrent(why) {
    if (!running) return;
    tour.markUnplayable(currentId);
    advancedThisTrack = true;
    onChange();
    step(tour.next());
  }

  function advance() {
    clearTimeout(watchdog);
    step(tour.next());
  }

  function finish(message) {
    running = false;
    waiting = false;
    clearTimeout(watchdog);
    if (controller) { try { controller.pause(); } catch (e) {} }
    AtlasGraph.setPlayingNode(null);
    onChange(message || null);
  }

  /* ── public ── */
  return {
    /* Start a tour. Defaults to the pinned node, else the first map node. */
    async start(startId) {
      const seed = startId || AtlasGraph.getSelectedId() || null;
      tour = AutoplayTraversal.createTour(graphView);
      const first = tour.start(seed);
      if (first.type === 'done') { onChange('No songs on the map to play.'); return; }
      running = true; userPaused = false; consecutiveSilent = 0;
      // Set currentId before priming so the prefetch reaches for the seed's
      // neighbours — where DFS actually goes next — rather than map order.
      currentId = first.id;
      await prefetch(PRIME_COUNT);
      // Route the seed through the same path as every other hop so an
      // unplayable first song is skipped rather than stalling the tour.
      await step(first);
    },
    /* Redirect a running tour to one song, keeping the played history. The
     * counterpart to start(), which wipes it — see playFromHere in app.js. */
    steerTo(id) {
      if (!running || !tour) return false;
      const res = tour.steerTo(id);
      if (!res) return false;
      clearTimeout(watchdog);
      step(res);
      return true;
    },
    stop() { finish(); },
    pause() {
      if (!controller || !running) return;
      userPaused = true;
      try { controller.pause(); } catch (e) {}
      onChange();
    },
    resume() {
      if (!controller || !running) return;
      userPaused = false;
      lastAdvanceAt = Date.now();
      try { controller.resume(); } catch (e) {}
      onChange();
    },
    next() { if (running) advance(); },
    setAutoAdvance(on) {
      autoAdvance = !!on;
      // Turning it back on while parked at the end of a song should move along
      // rather than wait for a click that no longer has a visible affordance.
      if (autoAdvance && waiting && running) advance();
    },
    autoAdvances: () => autoAdvance,
    isWaiting: () => waiting,
    isRunning: () => running,
    isPaused:  () => userPaused,
    currentNodeId: () => currentId,
    played:    () => (tour ? tour.getPlayed() : []),
    unplayable: id => (tour ? tour.isUnplayable(id) : false),
    remaining: () => (tour ? tour.remaining() : 0),
    setFollowCamera(on) { followCamera = !!on; },
    followsCamera: () => followCamera,
    onChange(cb) { onChange = cb || (() => {}); },
  };
})();
