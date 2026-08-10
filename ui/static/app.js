/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  viewMode:   'songs',   // song and artist maps are independent
  artistAvailable: false,
  audio:      null,
  audioUrl:   null,
  audioBtn:   null,
  detailId:   null,   // song currently shown in the detail panel
  similar:    [],     // similar-song rows backing the panel's list (map-connected
                      // rows first, then any expanded "show more" rows appended)
  mapCount:   0,      // how many leading rows of state.similar are map-connected
  expanded:   false,  // whether "Show more like this" has been fetched for detailId
  searchMode: 'tracks',   // 'tracks' | 'playlists' | 'albums'
  playlistPolls: {},  // group id (pid / album:<id>) → interval id (active polls)
  removed: [],        // recently removed nodes (newest first, session-only)
  activeFilters: [],  // [{label, kind, ids:Set}] — combined as a union
  recommendRows: [],  // rows backing the recommend-results list
  mentorBusy: false,
  detailType: 'song',
};

const REMOVED_MAX = 20;

/* ── Settings ─────────────────────────────────────────────────────────────
 * User playback preferences, persisted across reloads. `player` chooses what
 * the detail panel's single ▶ button does for ONE song; graph tours always use
 * the Spotify embed regardless, since that's the only source that can play a
 * full track. `autoAdvance` decides whether a tour steps on by itself. */
const SETTINGS_KEY = 'anther-settings';
const SETTINGS_DEFAULTS = { player: 'deezer', autoAdvance: true };
const settings = { ...SETTINGS_DEFAULTS };

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    if (saved.player === 'deezer' || saved.player === 'spotify') settings.player = saved.player;
    if (typeof saved.autoAdvance === 'boolean') settings.autoAdvance = saved.autoAdvance;
  } catch (err) { /* corrupt or unavailable storage → defaults */ }
}

function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }
  catch (err) { /* private mode / quota — settings just won't persist */ }
}

const SEARCH_MODES = {
  tracks: {
    placeholder: 'Artist, track…',
    hint: 'Corpus → Deezer → Spotify. Click a result to place it on the graph.',
    empty: 'Search for a song to add it to the atlas',
  },
  playlists: {
    placeholder: 'Playlist name…',
    hint: 'All 1M MPD playlists. Load one to place its top tracks — out-of-corpus songs embed in the background.',
    empty: 'Search for a playlist to load its tracks onto the atlas',
  },
  albums: {
    placeholder: 'Album name…',
    hint: 'Deezer albums — any release. Tracks embed in the background as they download.',
    empty: 'Search for an album to load its tracks onto the atlas',
  },
};

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  console.log("starting graph");
  await AtlasGraph.init('#graph');
  console.log("graph finished");
  await initArtistMode();
  initViewMode();
  initSearch();
  initArtistSearch();
  initUpload();
  initDetail();
  initMapPanel();
  initSettings();
  initAutoplay();
  initRecommend();
  initMentor();
  initMobileSidebar();
  initHelp();
  renderMapPanel();
  renderArtistMapPanel();
}

async function initArtistMode() {
  try {
    const status = await fetch('/api/artist/status').then(r => r.json());
    state.artistAvailable = !!status.available;
    if (state.artistAvailable) await ArtistGraph.init('#artist-graph');
    else {
      const btn = document.getElementById('view-artists');
      btn.disabled = true;
      btn.title = status.error || 'Artist mode unavailable';
      document.getElementById('build-artist-graph').disabled = true;
    }
  } catch (err) {
    state.artistAvailable = false;
    document.getElementById('view-artists').disabled = true;
    document.getElementById('build-artist-graph').disabled = true;
  }
}

function initViewMode() {
  document.getElementById('view-songs').addEventListener('click', () => setViewMode('songs'));
  document.getElementById('view-artists').addEventListener('click', () => setViewMode('artists'));
  document.getElementById('clear-artist-map').addEventListener('click', clearArtistMap);
  // Apply the state on first render as well as on later tab clicks.  Without
  // this, startup relies on the HTML's initial hidden attributes and can show
  // artist controls while the active tab and state still say "songs".
  setViewMode(state.viewMode);
}

function setViewMode(mode) {
  if (mode === 'artists' && !state.artistAvailable) return;
  state.viewMode = mode;
  const artists = mode === 'artists';
  document.getElementById('view-songs').classList.toggle('active', !artists);
  document.getElementById('view-artists').classList.toggle('active', artists);
  document.getElementById('view-songs').setAttribute('aria-selected', String(!artists));
  document.getElementById('view-artists').setAttribute('aria-selected', String(artists));
  document.getElementById('song-search-section').hidden = artists;
  document.getElementById('artist-search-section').hidden = !artists;
  document.getElementById('song-map-section').hidden = artists;
  document.getElementById('artist-map-section').hidden = !artists;
  // SVGElement does not consistently expose HTMLElement's `hidden` property.
  // Toggle the actual attribute so the `[hidden]` CSS rule applies in every
  // browser (and so switching back to songs removes it again).
  document.getElementById('graph').toggleAttribute('hidden', artists);
  document.getElementById('artist-graph').toggleAttribute('hidden', !artists);
  document.getElementById('artist-graph-empty').hidden = !artists || ArtistGraph.getNodes().length > 0;
  document.querySelector('.panel-left .subtitle').textContent = artists ? 'artist atlas' : 'song atlas';
  document.querySelector('.graph-hint').textContent = artists
    ? 'drag artists · scroll to zoom · hover for connections · click for details'
    : 'drag nodes · scroll to zoom · hover to explore · click for details · ♦ = your songs';
  if (artists) {
    ArtistGraph.activate();
    const id = ArtistGraph.getSelectedId();
    if (id) openArtistDetail(id); else document.getElementById('detail-panel').hidden = true;
    document.getElementById('artist-search-input').focus();
  } else {
    const id = AtlasGraph.getSelectedId();
    if (id) openDetail(id); else document.getElementById('detail-panel').hidden = true;
  }
}

/* ── Search (tracks: corpus → deezer → spotify · playlists: corpus) ──────── */
function initSearch() {
  let timer;
  const input = document.getElementById('search-input');
  input.addEventListener('input', e => {
    clearTimeout(timer);
    const q = e.target.value.trim();
    if (!q) { clearSearchResults(); return; }
    timer = setTimeout(() => doSearch(q), 420);
  });

  Object.keys(SEARCH_MODES).forEach(mode => {
    document.getElementById(`mode-${mode}`)
      .addEventListener('click', () => setSearchMode(mode));
  });
}

function setSearchMode(mode) {
  if (state.searchMode === mode) return;
  state.searchMode = mode;
  Object.keys(SEARCH_MODES).forEach(m => {
    document.getElementById(`mode-${m}`).classList.toggle('active', m === mode);
  });
  const input = document.getElementById('search-input');
  input.placeholder = SEARCH_MODES[mode].placeholder;
  document.getElementById('search-hint').textContent = SEARCH_MODES[mode].hint;
  const q = input.value.trim();
  if (q) doSearch(q); else clearSearchResults();
}

function clearSearchResults() {
  document.getElementById('search-results').innerHTML =
    `<div class="empty">${SEARCH_MODES[state.searchMode].empty}</div>`;
}

const SEARCH_URLS = {
  tracks:    q => `/api/search?q=${encodeURIComponent(q)}`,
  playlists: q => `/api/playlists/search?q=${encodeURIComponent(q)}`,
  albums:    q => `/api/albums/search?q=${encodeURIComponent(q)}`,
};

async function doSearch(q) {
  const container = document.getElementById('search-results');
  container.innerHTML = '<div class="empty">Searching…</div>';
  const mode = state.searchMode;
  try {
    const data = await fetch(SEARCH_URLS[mode](q)).then(r => r.json());
    if (data.error) { showError(data.error); clearSearchResults(); return; }
    if (mode === 'playlists') renderPlaylistResults(data);
    else if (mode === 'albums') renderAlbumResults(data);
    else renderSearchResults(data);
  } catch (err) {
    showError('Search failed: ' + err.message);
    clearSearchResults();
  }
}

const SOURCE_LABEL = { corpus: 'corpus', deezer: 'deezer', spotify: 'spotify',
                       itunes: 'itunes' };

function renderSearchResults(data) {
  const el = document.getElementById('search-results');
  const hits = data.results || [];
  if (!hits.length) {
    el.innerHTML = '<div class="empty">No results</div>';
    return;
  }
  el.innerHTML = hits.map((h, i) => {
    const preview = h.preview_url
      ? `<button class="btn-icon" title="Preview"
           onclick="togglePreview('${esc(h.preview_url)}', this)">▶</button>` : '';
    const badge = `<span class="src-badge src-${h.source}">${SOURCE_LABEL[h.source] || h.source}</span>`;
    return `
    <div class="track-row" data-i="${i}">
      <div class="track-info">
        <div class="track-title">${esc(h.title)}</div>
        <div class="track-artist">${esc(h.artist)} ${badge}</div>
      </div>
      <div class="track-actions">
        ${preview}
        <button class="btn-add" onclick='placeHit(${JSON.stringify(h)}, this)'>Add</button>
      </div>
    </div>`;
  }).join('');
}

function renderPlaylistResults(data) {
  const el = document.getElementById('search-results');
  const hits = data.results || [];
  const notice = data.notice
    ? `<div class="hint-text">${esc(data.notice)}</div>` : '';
  if (!hits.length) {
    el.innerHTML = notice + '<div class="empty">No playlists found</div>';
    return;
  }
  el.innerHTML = notice + hits.map(h => {
    const instant = h.n_in_corpus ? ` · ${h.n_in_corpus} instant` : '';
    const count = h.source === 'corpus'
      ? `${h.n_tracks} track${h.n_tracks === 1 ? '' : 's'} in corpus`
      : `${h.n_tracks} track${h.n_tracks === 1 ? '' : 's'}${instant}`;
    return `
    <div class="track-row">
      <div class="track-info">
        <div class="track-title">${esc(h.name)}</div>
        <div class="playlist-count">${count}</div>
      </div>
      <div class="track-actions">
        <button class="btn-add" onclick='loadPlaylist(${JSON.stringify(h)}, this)'>Load</button>
      </div>
    </div>`;
  }).join('');
}

function renderAlbumResults(data) {
  const el = document.getElementById('search-results');
  const hits = data.results || [];
  if (!hits.length) {
    el.innerHTML = '<div class="empty">No albums found</div>';
    return;
  }
  el.innerHTML = hits.map(h => `
    <div class="track-row">
      ${h.cover ? `<img class="album-cover" src="${esc(h.cover)}" alt="" />` : ''}
      <div class="track-info">
        <div class="track-title">${esc(h.name)}</div>
        <div class="track-artist">${esc(h.artist)}
          <span class="playlist-count">· ${h.n_tracks || '?'} tracks</span></div>
      </div>
      <div class="track-actions">
        <button class="btn-add" onclick='loadAlbum(${JSON.stringify(h)}, this)'>Load</button>
      </div>
  </div>`).join('');
}

/* ── Artist search and incremental placement ─────────────────────────────── */
function initArtistSearch() {
  let timer;
  const input = document.getElementById('artist-search-input');
  input.addEventListener('input', e => {
    clearTimeout(timer);
    const q = e.target.value.trim();
    if (!q) {
      document.getElementById('artist-search-results').innerHTML =
        '<div class="empty">Search for an artist to begin</div>';
      return;
    }
    timer = setTimeout(() => doArtistSearch(q), 350);
  });
}

async function doArtistSearch(q) {
  const el = document.getElementById('artist-search-results');
  el.innerHTML = '<div class="empty">Searching…</div>';
  try {
    const response = await fetch(`/api/artist/search?q=${encodeURIComponent(q)}`);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Artist search failed');
    renderArtistResults(data.results || []);
  } catch (err) {
    showError(err.message);
    el.innerHTML = '<div class="empty">Artist search unavailable</div>';
  }
}

function renderArtistResults(results) {
  const el = document.getElementById('artist-search-results');
  if (!results.length) { el.innerHTML = '<div class="empty">No artists found</div>'; return; }
  el.innerHTML = results.map(row => `
    <div class="track-row">
      <div class="track-info">
        <div class="track-title">${esc(row.name)}</div>
        <div class="track-artist">${row.track_count || 0} tracks · cluster ${row.cluster_id}${row.source === 'session' ? ' · private' : ''}</div>
      </div>
      <button class="btn-add" onclick='placeArtist(${JSON.stringify(row.id)}, this, ${String(row.id).startsWith('lowconf:')})'>${ArtistGraph.hasNode(row.id) ? 'On map ✓' : 'Add'}</button>
    </div>`).join('');
}

async function placeArtist(artistId, btn, lowConfidence) {
  if (ArtistGraph.hasNode(artistId)) {
    ArtistGraph.selectNode(artistId);
    return;
  }
  btn.disabled = true;
  // Low-confidence artists auto-supplement with a few extra song previews on
  // placement (fetch + embed), so surface the same "Placing…" wait a Deezer
  // song placement shows — otherwise the longer pause looks like a hang.
  btn.textContent = lowConfidence ? 'Placing…' : '…';
  try {
    const response = await fetch('/api/artist/place', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({artist_id: artistId}),
    });
    const fragment = await response.json();
    if (!response.ok || fragment.error) throw new Error(fragment.error || 'Could not add artist');
    ArtistGraph.mergeFragment(fragment);
    renderArtistMapPanel();
    btn.textContent = 'Added ✓';
  } catch (err) {
    showError(err.message); btn.disabled = false; btn.textContent = 'Add';
  }
}

function renderArtistMapPanel() {
  const list = document.getElementById('artist-map-list');
  const nodes = state.artistAvailable ? ArtistGraph.getNodes() : [];
  document.getElementById('artist-map-count').textContent =
    `${nodes.length} artist${nodes.length === 1 ? '' : 's'}`;
  list.innerHTML = nodes.length ? nodes.map(node => `
    <div class="map-row" onclick='ArtistGraph.selectNode(${JSON.stringify(node.id)})'>
      <div class="track-info">
        <div class="track-title">${esc(node.name)}</div>
        <div class="track-artist">${node.track_count || 0} tracks${node.source === 'session' ? ' · private' : ''}</div>
      </div>
      <button class="btn-icon" title="Remove" onclick='event.stopPropagation(); removeArtist(${JSON.stringify(node.id)})'>×</button>
    </div>`).join('') : '<div class="empty">No artists added yet</div>';
}

async function removeArtist(artistId) {
  try {
    const response = await fetch('/api/artist/node/' + encodeURIComponent(artistId), {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Could not remove artist');
    ArtistGraph.removeNodes(data.removed || [artistId]);
    renderArtistMapPanel();
  } catch (err) { showError(err.message); }
}

async function clearArtistMap() {
  try {
    const response = await fetch('/api/artist/graph/clear', {method: 'POST'});
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Could not clear artist map');
    ArtistGraph.reset(); renderArtistMapPanel(); closeDetail();
  } catch (err) { showError(err.message); }
}

/* ── Place onto the graph ───────────────────────────────────────────────── */
async function placeHit(hit, btn) {
  if (hit.id && AtlasGraph.hasNode(hit.id)) {
    AtlasGraph.zoomTo(hit.id);
    btn.textContent = 'On graph ✓';
    setTimeout(() => { btn.textContent = 'Add'; }, 1800);
    return;
  }
  btn.disabled = true;
  btn.textContent = hit.source === 'corpus' ? '…' : 'Placing…';
  try {
    const frag = await fetch('/api/place', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(hit),
    }).then(r => r.json());
    if (frag.error) { showError(frag.error); btn.disabled = false; btn.textContent = 'Add'; return; }
    AtlasGraph.mergeFragment(frag);
    renderMapPanel();
    btn.textContent = 'Added ✓';
  } catch (err) {
    showError('Placement failed: ' + err.message);
    btn.textContent = 'Error';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Add'; }, 2000);
}

async function loadPlaylist(hit, btn) {
  await loadCollection('/api/playlist/place', { pid: hit.pid }, hit.pid, btn);
}

async function loadAlbum(hit, btn) {
  await loadCollection('/api/album/place', { album_id: hit.album_id },
                       `album:${hit.album_id}`, btn);
}

/* Shared playlist/album import: place instants, then stream the background
 * embeds in via the status poll. `gid` is the group id ("<pid>" / "album:<id>"). */
async function loadCollection(url, body, gid, btn) {
  if (state.playlistPolls[gid]) {               // already streaming this group
    btn.textContent = 'Loading…';
    setTimeout(() => { btn.textContent = 'Load'; }, 1800);
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json());
    if (resp.error) { showError(resp.error); btn.disabled = false; btn.textContent = 'Load'; return; }
    if (resp.notice) showError(resp.notice);
    AtlasGraph.mergeFragment(resp.fragment, { focus: false });
    const p = resp.playlist;
    AtlasGraph.registerGroup(p.pid, {
      name: p.name,
      kind: String(p.pid).startsWith('album:') ? 'album' : 'playlist',
    });
    renderMapPanel();
    const capped = p.capped ? ` (top ${p.n_tracks} of ${p.n_total})` : '';
    if (resp.job_id) {
      setPlaylistProgress(`${p.name} — ${p.n_immediate} of ${p.n_tracks} placed${capped} · embedding ${p.n_pending}…`);
      startPlaylistPoll(resp.job_id, p);
      btn.textContent = 'Streaming…';
    } else {
      setPlaylistProgress(`${p.name} — ${p.n_immediate} of ${p.n_tracks} placed${capped} ✓`, 4000);
      btn.textContent = 'Loaded ✓';
    }
  } catch (err) {
    showError('Import failed: ' + err.message);
    btn.textContent = 'Error';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Load'; }, 2000);
}

/* Background-embed progress: poll /api/playlist/status, splice fragments in
 * as they arrive (no camera jumps), tick the progress line. */
function startPlaylistPoll(jobId, playlist) {
  let cursor = 0;
  const nImmediate = playlist.n_immediate || 0;
  const capped = playlist.capped ? ` (top ${playlist.n_tracks} of ${playlist.n_total})` : '';
  const tick = async () => {
    let s;
    try {
      s = await fetch(`/api/playlist/status/${jobId}?cursor=${cursor}`).then(r => r.json());
    } catch (_) { return; }                      // transient network error: retry next tick
    if (s.state === 'not_found') {
      stopPlaylistPoll(playlist.pid);
      setPlaylistProgress(`${playlist.name} — job lost (server restarted?) · re-load to resume`, 8000);
      return;
    }
    if ((s.fragments || []).length) {
      s.fragments.forEach(f => AtlasGraph.mergeFragment(f, { focus: false }));
      renderMapPanel();
    }
    cursor = s.cursor;
    const placed = nImmediate + s.placed;
    const skipped = s.failed ? ` · ${s.failed} skipped` : '';
    if (s.state === 'done') {
      stopPlaylistPoll(playlist.pid);
      setPlaylistProgress(`${playlist.name} — ${placed} of ${playlist.n_tracks} placed${capped}${skipped} ✓`, 8000);
    } else {
      const canStop = s.can_stop ? ` [<a href="#" onclick="stopPlaylistJob('${jobId}', event)">stop</a>]` : '';
      setPlaylistProgress(`${playlist.name} — ${placed} of ${playlist.n_tracks} placed${capped}${skipped} · ${s.message || ''}${canStop}`);
    }
  };
  state.playlistPolls[playlist.pid] = setInterval(tick, 1500);
  state.playlistJobIds = state.playlistJobIds || {};
  state.playlistJobIds[playlist.pid] = jobId;
  tick();
}

function stopPlaylistPoll(pid) {
  clearInterval(state.playlistPolls[pid]);
  delete state.playlistPolls[pid];
  if (state.playlistJobIds) delete state.playlistJobIds[pid];
}

let progressTimer;
function setPlaylistProgress(text, clearAfterMs) {
  const el = document.getElementById('playlist-progress');
  el.innerHTML = text;  // Changed from textContent to innerHTML to support links
  clearTimeout(progressTimer);
  if (clearAfterMs) progressTimer = setTimeout(() => { el.innerHTML = ''; }, clearAfterMs);
}

async function stopPlaylistJob(jobId, event) {
  event.preventDefault();
  event.stopPropagation();
  try {
    const resp = await fetch(`/api/playlist/stop/${jobId}`, { method: 'POST' }).then(r => r.json());
    if (resp.error) {
      showError(`Stop failed: ${resp.error}`);
    } else if (resp.can_stop === false) {
      // Already stopped, poll will update shortly
    }
  } catch (err) {
    showError(`Stop failed: ${err.message}`);
  }
}

/* ── Map panel: filter/highlight, node list, remove / re-place, clear ────── */

function initMapPanel() {
  const input = document.getElementById('filter-input');
  let timer;
  input.addEventListener('input', e => {
    clearTimeout(timer);
    timer = setTimeout(() => renderFilterSuggest(e.target.value.trim()), 150);
  });
  input.addEventListener('blur', () => {         // let a suggestion click land first
    setTimeout(() => { document.getElementById('filter-suggest').style.display = 'none'; }, 200);
  });
  input.addEventListener('focus', e => {
    if (e.target.value.trim()) renderFilterSuggest(e.target.value.trim());
  });

  document.getElementById('clear-map').addEventListener('click', async () => {
    const n = AtlasGraph.getNodes().length;
    if (!n) return;
    if (!confirm(`Remove all ${n} nodes from the map? This cannot be undone.`)) return;
    try {
      await fetch('/api/graph/clear', { method: 'POST' });
    } catch (err) { showError('Clear failed: ' + err.message); return; }
    AtlasAutoplay.stop();          // a tour over a map that no longer exists
    AtlasGraph.reset();
    clearFilter();
    renderMapPanel();
  });

  document.getElementById('build-artist-graph').addEventListener('click', buildArtistGraphFromSongMap);
}

/* ── Graph autoplay ───────────────────────────────────────────────────────
 * Walks the map as a playlist. See ui/static/autoplay.js for the player and
 * autoplay-traversal.js for the DFS/backtrack/jump policy. */
function initSettings() {
  loadSettings();
  const btn = document.getElementById('settings-toggle');
  const pop = document.getElementById('settings-popover');
  const auto = document.getElementById('setting-autoadvance');

  document.querySelectorAll('input[name="player"]').forEach(r => {
    r.checked = (r.value === settings.player);
    r.addEventListener('change', () => {
      if (!r.checked) return;
      settings.player = r.value;
      saveSettings();
      stopOtherPlayers();          // the old player shouldn't keep sounding
      if (state.detailId) refreshDetailButtons();
    });
  });

  auto.checked = settings.autoAdvance;
  auto.addEventListener('change', () => {
    settings.autoAdvance = auto.checked;
    saveSettings();
    AtlasAutoplay.setAutoAdvance(auto.checked);
    renderAutoplayBar();
  });
  AtlasAutoplay.setAutoAdvance(settings.autoAdvance);

  // On mobile the popover is not a popover: mobile.js re-hosts it as the
  // sheet's Settings tab, where an outside click means "I tapped something
  // else in the panel", not "dismiss".
  const close = () => {
    if (typeof AtlasMobile !== 'undefined' && AtlasMobile.isActive()) return;
    pop.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  };
  btn.addEventListener('click', e => {
    e.stopPropagation();
    const open = pop.hidden;
    pop.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  });
  pop.addEventListener('click', e => e.stopPropagation());
  document.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

/* The single ▶ swaps behaviour with the player setting, so a panel that's
 * already open has to be re-rendered when the setting changes under it. */
function refreshDetailButtons() {
  const wrap = document.getElementById('spotify-embed-wrap');
  if (wrap) { wrap.hidden = true; wrap.innerHTML = ''; wrap.dataset.forId = ''; }
  const btn = document.querySelector('.detail-btn-group .btn-play');
  if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); delete btn.dataset.url; }
}

function initAutoplay() {
  const toggle = document.getElementById('autoplay-toggle');
  const follow = document.getElementById('autoplay-follow');

  toggle.addEventListener('click', () => {
    if (AtlasAutoplay.isPaused()) AtlasAutoplay.resume(); else AtlasAutoplay.pause();
  });
  document.getElementById('autoplay-next').addEventListener('click', () => AtlasAutoplay.next());
  document.getElementById('autoplay-stop').addEventListener('click', () => AtlasAutoplay.stop());
  follow.addEventListener('change', () => AtlasAutoplay.setFollowCamera(follow.checked));

  // Grabbing the map means "I want to look at something" — stop the camera
  // fighting the user. The tour keeps playing; only the auto-zoom stands down.
  AtlasGraph.onManualPan(() => {
    if (!AtlasAutoplay.isRunning() || !AtlasAutoplay.followsCamera()) return;
    AtlasAutoplay.setFollowCamera(false);
    follow.checked = false;
  });

  // Shift-click a node to steer a running tour there, keeping the played
  // history. With nothing playing it just starts a tour from that node, which
  // is the same thing ⇉ does — no reason for the gesture to be inert.
  AtlasGraph.onShiftSelect(id => {
    const n = AtlasGraph.getNodes().find(x => x.id === id);
    if (!n || n.kind !== 'query') {
      showError('Only songs you placed can start a tour.');
      return;
    }
    if (AtlasAutoplay.isRunning()) AtlasAutoplay.steerTo(id);
    else playFromHere(id);
  });

  AtlasAutoplay.onChange(message => renderAutoplayBar(message));
}

function renderAutoplayBar(message) {
  const bar    = document.getElementById('autoplay-bar');
  const toggle = document.getElementById('autoplay-toggle');
  const title  = document.getElementById('autoplay-title');
  const sub    = document.getElementById('autoplay-sub');

  const running = AtlasAutoplay.isRunning();
  if (!running) {
    bar.hidden = true;
    AtlasGraph.setPlayingNode(null);
    if (message) showError(message);
    renderMapPanel();
    return;
  }

  bar.hidden = false;
  toggle.textContent = AtlasAutoplay.isPaused() ? '▶' : '⏸';
  toggle.title = AtlasAutoplay.isPaused() ? 'Resume' : 'Pause';

  const id = AtlasAutoplay.currentNodeId();
  const n  = AtlasGraph.getNodes().find(x => x.id === id);
  title.textContent = n ? (n.name || id) : '…';

  const played  = AtlasAutoplay.played().length;
  const left    = AtlasAutoplay.remaining();
  const silent  = AtlasAutoplay.unplayable(id);
  const waiting = AtlasAutoplay.isWaiting();
  sub.className = 'autoplay-sub' + (silent || waiting ? ' autoplay-warn' : '');
  if (waiting)     sub.textContent = `finished · ⏭ for the next song · ${left} left`;
  else if (silent) sub.textContent = 'not on Spotify — skipping';
  else sub.textContent =
    `${(n && n.artist) || ''}${n && n.artist ? ' · ' : ''}${played} played · ${left} left`;
}

/* Start a tour seeded on one song. Any running tour is replaced rather than
 * redirected: "play from here" reads as a fresh start, and keeping the old
 * history would silently skip songs the user can see are unplayed. */
function playFromHere(id) {
  if (AtlasAutoplay.isRunning()) AtlasAutoplay.stop();
  stopOtherPlayers();
  document.getElementById('autoplay-bar').hidden = false;
  AtlasAutoplay.start(id);
}

/* Only one thing may sound at a time: the preview <audio>, the detail panel's
 * Spotify embed, and the tour's own controller. */
function stopOtherPlayers() {
  if (state.audio) {
    state.audio.pause();
    if (state.audioBtn) { state.audioBtn.textContent = '▶'; state.audioBtn.classList.remove('playing'); }
    state.audio = null; state.audioUrl = null; state.audioBtn = null;
  }
  const wrap = document.getElementById('spotify-embed-wrap');
  if (wrap && !wrap.hidden) { wrap.hidden = true; wrap.innerHTML = ''; wrap.dataset.forId = ''; }
}

function chooseArtistGraphMode() {
  const modal = document.getElementById('artist-graph-choice-modal');
  const overlay = modal.querySelector('.modal-overlay');
  const close = document.getElementById('artist-graph-choice-close');
  const cancel = document.getElementById('artist-graph-choice-cancel');
  const replace = document.getElementById('artist-graph-choice-replace');
  const append = document.getElementById('artist-graph-choice-append');

  return new Promise(resolve => {
    const finish = choice => {
      modal.setAttribute('hidden', '');
      overlay.removeEventListener('click', dismiss);
      close.removeEventListener('click', dismiss);
      cancel.removeEventListener('click', dismiss);
      replace.removeEventListener('click', startNew);
      append.removeEventListener('click', addExisting);
      document.removeEventListener('keydown', onKeydown);
      resolve(choice);
    };
    const dismiss = () => finish(null);
    const startNew = () => finish('replace');
    const addExisting = () => finish('append');
    const onKeydown = event => { if (event.key === 'Escape') dismiss(); };
    overlay.addEventListener('click', dismiss);
    close.addEventListener('click', dismiss);
    cancel.addEventListener('click', dismiss);
    replace.addEventListener('click', startNew);
    append.addEventListener('click', addExisting);
    document.addEventListener('keydown', onKeydown);
    modal.removeAttribute('hidden');
    append.focus();
  });
}

async function buildArtistGraphFromSongMap() {
  if (!state.artistAvailable) {
    showError('Artist graph is unavailable.');
    return;
  }
  const seedCount = AtlasGraph.getNodes().filter(node => node.kind === 'query').length;
  if (!seedCount) {
    showError('Add a song to the map first.');
    return;
  }
  const mode = ArtistGraph.getNodes().length ? await chooseArtistGraphMode() : 'append';
  if (!mode) return;

  const button = document.getElementById('build-artist-graph');
  button.disabled = true;
  // Artists with fewer than 5 on-map songs are auto-supplemented with extra
  // previews (fetch + embed), so this can take a moment for new artists.
  button.textContent = 'Placing…';
  try {
    const response = await fetch('/api/artist/from-song-graph', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Could not build artist graph');
    if (!data.artist_count) {
      showError('None of your added songs could be turned into artists yet.');
      return;
    }
    ArtistGraph.replaceData(data);
    renderArtistMapPanel();
    setViewMode('artists');
    if (data.skipped_count) {
      showError(`${data.skipped_count} artist${data.skipped_count === 1 ? '' : 's'} could not be added because none of their songs have a usable embedding yet.`);
    }
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Artist graph';
  }
}

/* Candidates: artists on the map + imported playlist/album groups. */
function filterCandidates(q) {
  const needle = q.toLowerCase();
  const nodes = AtlasGraph.getNodes();
  const artists = new Map();                     // artist → count
  nodes.forEach(n => {
    const a = (n.artist || '').trim();
    if (a && a.toLowerCase().includes(needle)) artists.set(a, (artists.get(a) || 0) + 1);
  });
  const out = [];
  artists.forEach((count, a) =>
    out.push({ kind: 'artist', label: a, count,
               ids: new Set(nodes.filter(n => (n.artist || '').trim() === a).map(n => n.id)) }));
  const groups = AtlasGraph.getGroups();
  Object.keys(groups).forEach(gid => {
    const g = groups[gid];
    if (!(g.name || gid).toLowerCase().includes(needle)) return;
    const ids = new Set(nodes.filter(n => String(n.playlist_pid) === gid).map(n => n.id));
    if (ids.size) out.push({ kind: g.kind || 'playlist', label: g.name || gid, count: ids.size, ids });
  });
  out.sort((a, b) => b.count - a.count);
  return out.slice(0, 12);
}

const FILTER_KIND_ICON = { artist: '♪', playlist: '▤', album: '◉' };

function renderFilterSuggest(q) {
  const box = document.getElementById('filter-suggest');
  if (!q) { box.style.display = 'none'; return; }
  const cands = filterCandidates(q);
  if (!cands.length) { box.style.display = 'none'; return; }
  state.filterCands = cands;
  box.innerHTML = cands.map((c, i) => `
    <div class="suggest-row" onmousedown="applyFilterCand(${i})">
      <span class="suggest-kind">${FILTER_KIND_ICON[c.kind] || ''} ${c.kind}</span>
      <span>${esc(c.label)}</span>
      <span class="suggest-kind" style="margin-left:auto">${c.count}</span>
    </div>`).join('');
  box.style.display = 'block';
}

/* Union of node ids across every active filter (null when none are active). */
function activeFilterIds() {
  if (!state.activeFilters.length) return null;
  const ids = new Set();
  state.activeFilters.forEach(f => f.ids.forEach(id => ids.add(id)));
  return ids;
}

function renderActiveFilters() {
  const box = document.getElementById('filter-active');
  box.innerHTML = state.activeFilters.map((c, i) => `
    <span class="filter-chip">${FILTER_KIND_ICON[c.kind] || ''} ${esc(c.label)}
      <span class="suggest-kind">${c.count}</span>
      <button onclick="removeFilter(${i})" title="Remove filter">×</button>
    </span>`).join('');
}

function applyFilterCand(i) {
  const c = state.filterCands && state.filterCands[i];
  if (!c) return;
  if (!state.activeFilters.some(f => f.kind === c.kind && f.label === c.label)) {
    state.activeFilters.push(c);
  }
  AtlasGraph.setFilter(activeFilterIds());
  const input = document.getElementById('filter-input');
  input.value = '';
  document.getElementById('filter-suggest').style.display = 'none';
  renderActiveFilters();
  renderMapPanel();
}

function removeFilter(i) {
  state.activeFilters.splice(i, 1);
  AtlasGraph.setFilter(activeFilterIds());
  renderActiveFilters();
  renderMapPanel();
}

function clearFilter() {
  state.activeFilters = [];
  AtlasGraph.setFilter(null);
  document.getElementById('filter-active').innerHTML = '';
  renderMapPanel();
}

function renderMapPanel() {
  const nodes = AtlasGraph.getNodes();
  const filterIds = activeFilterIds();
  let queries = nodes.filter(n => n.kind === 'query');
  if (filterIds) queries = queries.filter(n => filterIds.has(n.id));

  document.getElementById('map-count').textContent =
    filterIds
      ? `${queries.length} matching · ${nodes.length} nodes total`
      : `${queries.length} songs · ${nodes.length} nodes`;

  state.mapRows = queries;
  document.getElementById('map-list').innerHTML = queries.map((n, i) => {
    const ring = n.playlist_pid
      ? `<span class="ring-dot" style="border-color:${AtlasGraph.groupColor(n.playlist_pid)}"></span>`
      : '<span class="ring-dot"></span>';
    return `
    <div class="map-row" onclick="gotoMapRow(${i})">
      ${ring}
      <div class="track-info">
        <div class="track-title">${esc(n.name)}</div>
        <div class="track-artist">${esc(n.artist || '')}</div>
      </div>
      <button class="btn-tiny" title="Remove from map"
              onclick="event.stopPropagation(); removeMapRow(${i})">✕</button>
    </div>`;
  }).join('') || '<div class="empty">Nothing on the map yet</div>';

  document.getElementById('removed-header').hidden = !state.removed.length;
  document.getElementById('removed-list').innerHTML = state.removed.map((n, i) => `
    <div class="map-row">
      <span class="ring-dot"></span>
      <div class="track-info">
        <div class="track-title">${esc(n.name)}</div>
        <div class="track-artist">${esc(n.artist || '')}</div>
      </div>
      <button class="btn-tiny" title="Re-place on map"
              onclick="replaceRemoved(${i})">↩</button>
    </div>`).join('');
}

function gotoMapRow(i) {
  const n = state.mapRows[i];
  if (n) AtlasGraph.selectNode(n.id, { zoom: true });
}

async function removeMapRow(i) {
  const n = state.mapRows[i];
  if (!n) return;
  try {
    const resp = await fetch('/api/node/' + encodeURIComponent(n.id), { method: 'DELETE' })
      .then(r => r.json());
    if (resp.error) { showError(resp.error); return; }
    AtlasGraph.removeNodes(resp.removed);
    state.removed.unshift(resp.node);
    state.removed.length = Math.min(state.removed.length, REMOVED_MAX);
  } catch (err) { showError('Remove failed: ' + err.message); }
  renderMapPanel();
}

async function replaceRemoved(i) {
  const n = state.removed[i];
  if (!n) return;
  try {
    const frag = await fetch('/api/place', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: n.source, id: n.id, title: n.name,
                             artist: n.artist, playlist_pid: n.playlist_pid }),
    }).then(r => r.json());
    if (frag.error) { showError(frag.error); return; }
    AtlasGraph.mergeFragment(frag);
    state.removed.splice(i, 1);
  } catch (err) { showError('Re-place failed: ' + err.message); }
  renderMapPanel();
}

/* ── Recommend from map (seeds = every song currently placed on the map) ─ */
function initRecommend() {
  document.getElementById('recommend-btn').addEventListener('click', doRecommend);
}

async function doRecommend() {
  const btn = document.getElementById('recommend-btn');
  const seedIds = AtlasGraph.getNodes().filter(n => n.kind === 'query').map(n => n.id);
  if (!seedIds.length) { showError('Add a few songs to the map first.'); return; }
  btn.disabled = true;
  btn.textContent = 'Recommending…';
  try {
    const data = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed_ids: seedIds }),
    }).then(r => r.json());
    if (data.error) { showError(data.error); return; }
    renderRecommendResults(data);
  } catch (err) {
    showError('Recommend failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Recommend similar';
  }
}

function renderRecommendResults(data) {
  const el = document.getElementById('recommend-results');
  const results = data.results || [];
  if (!results.length) {
    el.innerHTML = '<div class="empty">No recommendations found</div>';
    return;
  }
  // The backend already splices these into the shared graph (recommend()'s
  // splice=True) — mirror that locally so the map and this list agree.
  AtlasGraph.mergeFragment({
    nodes: results.map(r => ({
      id: r.id, name: r.name, artist: r.artist, kind: 'corpus',
      recommended: r.recommended === true,
    })),
    links: [],
  }, { focus: false });
  state.recommendRows = results;
  el.innerHTML = results.map((r, i) => `
    <div class="map-row" onclick="gotoRecommendRow(${i})">
      <span class="ring-dot"></span>
      <div class="track-info">
        <div class="track-title">${esc(r.name)}</div>
        <div class="track-artist">${esc(r.artist || '')}</div>
      </div>
      <span class="similar-score">${r.score != null ? 'Similarity score: ' + Math.round(r.score) : ''}</span>
    </div>`).join('');
}

function gotoRecommendRow(i) {
  const r = state.recommendRows[i];
  if (r) AtlasGraph.selectNode(r.id, { zoom: true });
}

/* ── Mentor chat ──────────────────────────────────────────────────────── */
function initMentor() {
  document.getElementById('mentor-send').addEventListener('click', sendMentorMessage);
  document.getElementById('mentor-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendMentorMessage();
  });
  document.getElementById('mentor-reset').addEventListener('click', resetMentor);
  
  // Toggle button to open/close mentor panel
  document.getElementById('mentor-toggle').addEventListener('click', toggleMentorPanel);
  document.getElementById('mentor-close').addEventListener('click', toggleMentorPanel);
}

function appendMentorMessage(role, text) {
  const el = document.getElementById('mentor-messages');
  if (el.querySelector('.empty')) el.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'mentor-msg mentor-msg-' + role;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function setMentorStatus(text) {
  const el = document.getElementById('mentor-status');
  el.textContent = text || '';
  el.hidden = !text;
}

async function sendMentorMessage() {
  if (state.mentorBusy) return;
  const input = document.getElementById('mentor-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  appendMentorMessage('user', question);
  state.mentorBusy = true;
  setMentorStatus('Mentor is thinking…');
  try {
    const data = await fetch('/api/mentor/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // The pinned node (or null) rides along so the mentor can resolve
      // "this song" / "what do I sound like" against the user's selection.
      body: JSON.stringify({ question, selected_node_id: AtlasGraph.getSelectedId() }),
    }).then(r => r.json());
    if (data.error) {
      appendMentorMessage('error', data.error);
    } else {
      appendMentorMessage('mentor', data.answer);
    }
  } catch (err) {
    appendMentorMessage('error', 'Mentor is currently unavailable.');
  } finally {
    state.mentorBusy = false;
    setMentorStatus('');
  }
}

async function resetMentor() {
  try {
    await fetch('/api/mentor/reset', { method: 'POST' });
  } catch (err) { /* best-effort — a stale session just times out server-side */ }
  document.getElementById('mentor-messages').innerHTML =
    '<div class="empty">Ask what you sound like, who you\'re close to, or for music advice.</div>';
  setMentorStatus('');
}

/* ── Detail panel (click a node → pinned side pop-over) ─────────────────── */
function initDetail() {
  AtlasGraph.onSelect(openDetail);
  AtlasGraph.onDeselect(closeDetail);
  if (state.artistAvailable) {
    ArtistGraph.onSelect(openArtistDetail);
    ArtistGraph.onDeselect(closeDetail);
  }
  document.getElementById('detail-close')
    .addEventListener('click', () => {
      if (state.viewMode === 'artists') ArtistGraph.clearSelection();
      else AtlasGraph.clearSelection();
    });
}

async function openDetail(id) {
  const panel = document.getElementById('detail-panel');
  const body  = document.getElementById('detail-body');
  state.detailId = id;
  state.detailType = 'song';
  state.expanded = false;
  panel.hidden = false;
  body.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const r = await fetch('/api/song/' + encodeURIComponent(id));
    const data = await r.json();
    if (state.detailId !== id) return;              // superseded by a newer click
    if (!r.ok || data.error) {
      showError(data.error || 'Failed to load song details');
      AtlasGraph.clearSelection();
      return;
    }
    renderDetail(data);
  } catch (err) {
    if (state.detailId !== id) return;
    showError('Failed to load song details: ' + err.message);
    AtlasGraph.clearSelection();
  }
}

async function openArtistDetail(id) {
  if (state.viewMode !== 'artists') return;
  const panel = document.getElementById('detail-panel');
  const body = document.getElementById('detail-body');
  state.detailId = id;
  state.detailType = 'artist';
  panel.hidden = false;
  body.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const response = await fetch('/api/artist/' + encodeURIComponent(id));
    const data = await response.json();
    if (state.detailId !== id || state.detailType !== 'artist') return;
    if (!response.ok || data.error) throw new Error(data.error || 'Failed to load artist');
    const connected = data.connected_artists || [];
    body.innerHTML = `
      <div class="detail-title">${esc(data.name)}</div>
      <div class="detail-artist">${esc(data.cluster_label || `Cluster ${data.cluster_id}`)}</div>
      ${artistProfileHtml(data.profile)}
      <div class="detail-section">
        <div class="positioning-size">${data.track_count || 0} corpus/private track${data.track_count === 1 ? '' : 's'}${data.upload_count ? ` · ${data.upload_count} uploaded` : ''}</div>
        ${data.low_confidence ? `<div class="low-confidence-note" id="lowconf-note" role="button" tabindex="0" title="Click to strengthen this placement with more song previews">Placed from fewer than 5 tracks, so this clustering is less confident.</div>` : ''}
        ${data.sample_track ? `<div class="positioning-pitch">Representative track: ${esc(data.sample_track)}</div>` : ''}
      </div>
      <div class="detail-section"><h3>Connected artists</h3>
        ${connected.length ? connected.map(row => `
          <div class="similar-row on-graph" onclick='ArtistGraph.selectNode(${JSON.stringify(row.id)})'>
            <div class="similar-row-main"><div class="track-info">
              <div class="track-title">${esc(row.name)}</div>
              <div class="track-artist">${row.track_count || 0} tracks</div>
            </div><span class="similar-score">${Math.round(row.score || 0)}</span></div>
          </div>`).join('') : '<div class="empty-hint">No threshold-clearing connections on this map yet.</div>'}
      </div>`;
    if (data.low_confidence) {
      const note = document.getElementById('lowconf-note');
      if (note) {
        const trigger = () => supplementArtist(id, note);
        note.addEventListener('click', trigger);
        note.addEventListener('keydown', ev => {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); trigger(); }
        });
      }
    }
  } catch (err) {
    showError(err.message);
    ArtistGraph.clearSelection();
  }
}

// Easter egg: clicking the low-confidence note pulls extra strict-matched song
// previews for the artist (up to 5 tracks) and re-places the node with higher
// confidence. Session-only — the frozen bundle is never touched.
async function supplementArtist(id, note) {
  if (note.dataset.busy) return;
  note.dataset.busy = '1';
  note.classList.add('working');
  note.textContent = 'Finding more previews to strengthen this placement…';
  try {
    const response = await fetch('/api/artist/' + encodeURIComponent(id) + '/supplement',
      { method: 'POST' });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Could not supplement artist');
    ArtistGraph.mergeFragment(data);
    if (!data.added) {
      note.classList.remove('working');
      note.textContent = 'No additional previews found for this artist.';
      return;
    }
    // Re-open detail so the note/track count reflect the strengthened placement.
    if (state.detailId === id && state.detailType === 'artist') openArtistDetail(id);
  } catch (err) {
    note.classList.remove('working');
    delete note.dataset.busy;
    note.textContent = 'Placed from fewer than 5 tracks, so this clustering is less confident.';
    showError(err.message);
  }
}

// Compact enrichment profile shown in the artist detail pane (see
// ARTIST_ENRICHMENT_PLAN.md §7). Every field is optional; absent data is simply
// omitted rather than faked. `profile` is null when the artist hasn't been
// enriched yet — we show a quiet hint instead of a broken section.
function fmtFollowers(n) {
  if (n == null) return '';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'K';
  return String(n);
}

function artistProfileHtml(p) {
  if (!p) {
    return '<div class="artist-profile-empty">No enrichment profile yet — '
         + 'run artist enrichment to add image, origin, genres &amp; labels.</div>';
  }
  const photo = p.image_url
    ? `<img class="artist-photo" src="${esc(p.image_url)}" alt="" loading="lazy"
         onerror="this.remove()">`
    : '';
  const fans = (p.following != null)
    ? `<div class="artist-fans"><strong>${fmtFollowers(p.following)}</strong>
         ${esc(p.following_source || 'Deezer')} fans${p.following_as_of
           ? ` <span class="as-of">as of ${esc(p.following_as_of)}</span>` : ''}</div>`
    : '';
  const genres = (p.genres && p.genres.length)
    ? `<div class="genre-chips">${p.genres.map(g =>
         `<span class="genre-chip">${esc(g)}</span>`).join('')}</div>`
    : '';
  const origin = p.origin
    ? `<div class="artist-field"><span class="field-label">Origin</span>${esc(p.origin)}</div>`
    : '';
  const labels = (p.labels && p.labels.length)
    ? `<div class="artist-field"><span class="field-label">Associated labels</span>${p.labels.map(esc).join(', ')}</div>`
    : '';
  // Provenance disclosure — sources + freshness, so the data is auditable and
  // never reads as an authoritative "current label" / cross-platform total.
  const srcs = [];
  if (p.image_source || p.following_source) srcs.push('Deezer');
  if (p.origin_source === 'musicbrainz' || (p.genres && p.genres.length) || (p.labels && p.labels.length)) srcs.push('MusicBrainz');
  const prov = `<details class="artist-prov"><summary>Sources &amp; freshness</summary>
      <div>${srcs.length ? esc(srcs.join(' · ')) : 'no external sources'}${
        p.match_confidence != null ? ` · match ${Math.round(p.match_confidence * 100)}%` : ''}${
        p.updated_at ? ` · updated ${esc(p.updated_at)}` : ''}${
        p.enrichment_status && p.enrichment_status !== 'complete'
          ? ` · ${esc(p.enrichment_status)}` : ''}</div>
    </details>`;
  const hasBody = photo || fans || genres || origin || labels;
  return `<div class="artist-profile">
      ${photo}
      <div class="artist-profile-body">
        ${fans}${origin}${labels}${genres}
        ${hasBody ? '' : '<div class="empty-hint">Matched, but no profile fields available.</div>'}
        ${prov}
      </div>
    </div>`;
}

function closeDetail() {
  document.getElementById('detail-panel').hidden = true;
  state.detailId = null;
  state.similar = [];
  state.expanded = false;
  state.detailType = state.viewMode === 'artists' ? 'artist' : 'song';
}

// Melody/rhythm/timbre in the same order MERIT emits them (see anther_ml/merit.py
// FACTORS); "aggregate" is deliberately excluded here — it's already the headline
// score shown before expansion, not one of the three expanded rows.
const BREAKDOWN_FACTORS = [
  { key: 'melody', label: 'Melody', cls: 'melody' },
  { key: 'rhythm', label: 'Rhythm', cls: 'rhythm' },
  { key: 'timbre', label: 'Timbre', cls: 'timbre' },
];

function breakdownHtml(bd) {
  const rows = BREAKDOWN_FACTORS.map(f => {
    const v = bd[f.key];
    if (v == null) return '';
    const pct = Math.max(0, Math.min(100, v));
    return `
      <div class="breakdown-row">
        <span class="breakdown-label">${f.label}</span>
        <div class="breakdown-bar"><div class="breakdown-fill breakdown-${f.cls}" style="width:${pct}%"></div></div>
        <span class="breakdown-val">${Math.round(v)}</span>
      </div>`;
  }).join('');
  return `<div class="breakdown-panel" hidden>${rows}</div>`;
}

function toggleBreakdown(i, btn) {
  const row = btn.closest('.similar-row');
  const panel = row && row.querySelector('.breakdown-panel');
  if (!panel) return;
  const opening = panel.hidden;
  panel.hidden = !opening;
  btn.classList.toggle('open', opening);
  btn.setAttribute('aria-expanded', String(opening));
}

function similarRowHtml(s, i) {
  // Single aggregate score by default; when the corpus carries a MERIT-aggregate
  // index, `s.breakdown` also gives melody/rhythm/timbre — expand to compare
  // *why* two songs are similar, not just how much.
  const hasBreakdown = s.breakdown && typeof s.breakdown === 'object';
  const score = s.score != null
    ? `<span class="similar-score" title="${hasBreakdown ? 'Aggregate similarity' : 'Similarity score'}">${Math.round(s.score)}</span>`
    : '';
  const expandBtn = hasBreakdown
    ? `<button class="btn-expand" title="Show melody / rhythm / timbre breakdown"
         aria-expanded="false" onclick="event.stopPropagation(); toggleBreakdown(${i}, this)">▾</button>`
    : '';
  const onGraph = s.on_graph || AtlasGraph.hasNode(s.id);
  const add = onGraph ? '' : `<button class="btn-add" onclick="event.stopPropagation(); addSimilar(${i}, this)">Add</button>`;
  // Same player setting as the panel's own ▶ — these rows live inside the
  // detail panel and share its #spotify-embed-wrap, so both sources work here.
  const preview = `<button class="btn-icon" title="Play"
       onclick='event.stopPropagation(); playSong(${JSON.stringify(s.id)}, this)'>▶</button>`;
  return `
  <div class="similar-row${onGraph ? ' on-graph' : ''}">
    <div class="similar-row-main" ${onGraph ? `onclick="gotoSimilar(${i})"` : ''}>
      <div class="track-info">
        <div class="track-title">${esc(s.name)}</div>
        <div class="track-artist">${esc(s.artist)}</div>
      </div>
      <span class="similar-score-group">${score}${expandBtn}</span>
      ${preview}${add}
    </div>
    ${hasBreakdown ? breakdownHtml(s.breakdown) : ''}
  </div>`;
}

function renderDetail(d) {
  // Only query nodes can seed a tour — the traversal walks the query subgraph,
  // so offering this on a grey corpus node would silently start somewhere else.
  const gnode = AtlasGraph.getNodes().find(n => n.id === d.id);
  const tourBtn = (gnode && gnode.kind === 'query')
    ? `<button class="btn-icon" title="Play the map starting here — walks edges to similar songs"
               onclick='playFromHere(${JSON.stringify(d.id)})'>⇉</button>`
    : '';
  let html = `
    <div class="detail-title-row">
      <div>
        <div class="detail-title">${esc(d.name)}</div>
        <div class="detail-artist">${esc(d.artist)}</div>
      </div>
      <div class="detail-btn-group">
        <button class="btn-icon btn-play" title="Play this song (${settings.player === 'spotify'
          ? 'Spotify — full track if you\'re logged in' : 'Deezer 30s preview'}) — change in Settings"
                onclick='playSong(${JSON.stringify(d.id)}, this)'>▶</button>
        ${tourBtn}
      </div>
    </div>
    <div id="spotify-embed-wrap" class="spotify-embed-wrap" hidden></div>
    ${d.genre ? `<div class="detail-genre">${esc(d.genre)}</div>` : ''}`;

  // ── Neighborhood label ──────────────────────────────────────────────────
  const pos = d.positioning;
  if (pos?.cluster_label) {
    html += `<div class="detail-section">`;
    html += `<div class="positioning-label">${esc(pos.cluster_label)}</div>`;
    if (pos.cluster_size) {
      html += `<div class="positioning-size">${pos.cluster_size.toLocaleString()} tracks in this corner of the map</div>`;
    }
    if (pos.pitch_summary) {
      html += `<div class="positioning-pitch">${esc(pos.pitch_summary)}</div>`;
    }
    html += `</div>`;
  }

  // Primary list: songs actually connected to this one on the map (map_neighbors).
  // A separate "Show more" fetch (see showMoreSimilar) appends corpus-wide
  // results not already on the map — kept out of the initial payload since
  // that search is the expensive part of song_detail.
  const mapNeighbors = d.map_neighbors || [];
  state.similar = mapNeighbors.slice();
  state.mapCount = mapNeighbors.length;

  const listTitle = d.kind === 'playlist' ? 'Tracks' : 'Connected on map';
  html += `<div class="detail-section" id="similar-section"><h3>${listTitle}</h3>`
    + `<div id="similar-list">`
    + (mapNeighbors.length
        ? mapNeighbors.map((s, i) => similarRowHtml(s, i)).join('')
        : `<div class="empty-hint">Not connected to any songs on your map yet.</div>`)
    + `</div>`
    + `<button id="show-more-btn" class="btn-show-more" onclick="showMoreSimilar()">Show more like this</button>`
    + `</div>`;



  document.getElementById('detail-body').innerHTML = html;
}

async function showMoreSimilar() {
  const btn = document.getElementById('show-more-btn');
  const id = state.detailId;
  if (!id || state.expanded) return;
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const r = await fetch('/api/song/' + encodeURIComponent(id) + '?expand=1');
    const data = await r.json();
    if (state.detailId !== id) return;          // superseded by a newer click
    if (!r.ok || data.error) { showError(data.error || 'Failed to load more songs'); return; }
    state.expanded = true;
    const extra = data.similar || [];
    const list = document.getElementById('similar-list');
    if (extra.length) {
      const startIdx = state.similar.length;
      state.similar = state.similar.concat(extra);
      if (startIdx === 0) {
        // primary list was empty ("Not connected..." placeholder) — replace it
        list.innerHTML = '';
      }
      list.insertAdjacentHTML('beforeend',
        `<div class="more-divider">More like this</div>`
        + extra.map((s, j) => similarRowHtml(s, startIdx + j)).join(''));
    }
    btn.remove();
  } catch (err) {
    showError('Failed to load more songs: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Show more like this';
  }
}

function gotoSimilar(i) {
  const s = state.similar[i];
  if (s) AtlasGraph.selectNode(s.id, { zoom: true });
}

async function addSimilar(i, btn) {
  const s = state.similar[i];
  if (!s) return;
  if (AtlasGraph.hasNode(s.id)) { AtlasGraph.selectNode(s.id, { zoom: true }); return; }
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const frag = await fetch('/api/place', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'corpus', id: s.id, title: s.name, artist: s.artist }),
    }).then(r => r.json());
    if (frag.error) { showError(frag.error); btn.disabled = false; btn.textContent = 'Add'; return; }
    AtlasGraph.mergeFragment(frag);
    renderMapPanel();
    AtlasGraph.selectNode(s.id, { zoom: true });
  } catch (err) {
    showError('Placement failed: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Add';
  }
}

/* The detail panel's single ▶. Which source it uses is a user setting, not a
 * second button — see initSettings. Tours are unaffected: they always use the
 * Spotify embed, the only source that can play a full track. */
function playSong(id, btn) {
  if (settings.player === 'spotify') return toggleSpotifyEmbed(id, btn);
  return playPreview(id, btn);
}

/* ── Preview audio ──────────────────────────────────────────────────────── */
async function playPreview(id, btn) {
  if (btn.dataset.url) { togglePreview(btn.dataset.url, btn); return; }
  if (btn.disabled) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const data = await fetch('/api/song/' + encodeURIComponent(id) + '/preview').then(r => r.json());
    btn.disabled = false;
    if (!data.preview_url) {
      btn.textContent = '✕';
      btn.title = 'No preview available';
      return;
    }
    btn.dataset.url = data.preview_url;
    btn.textContent = original;
    togglePreview(data.preview_url, btn);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function togglePreview(url, btn) {
  // A preview and a running tour must not sound together.
  if (AtlasAutoplay.isRunning() && !AtlasAutoplay.isPaused()) AtlasAutoplay.pause();
  if (state.audio && state.audioUrl === url) {
    state.audio.pause();
    state.audio = null; state.audioUrl = null;
    btn.textContent = '▶'; btn.classList.remove('playing');
    return;
  }
  if (state.audio) {
    state.audio.pause();
    if (state.audioBtn) { state.audioBtn.textContent = '▶'; state.audioBtn.classList.remove('playing'); }
  }
  state.audio = new Audio(url);
  state.audioUrl = url; state.audioBtn = btn;
  state.audio.play().catch(() => {});
  btn.textContent = '⏸'; btn.classList.add('playing');
  state.audio.onended = () => {
    btn.textContent = '▶'; btn.classList.remove('playing');
    state.audio = null; state.audioUrl = null; state.audioBtn = null;
  };
}

/* ── Spotify embed (no login required on our side) ────────────────────────
 * Uses Spotify's public open.spotify.com/embed/track/<id> iframe — no OAuth,
 * no app registration, no Development Mode 5-account cap. A visitor already
 * logged into Spotify in that browser gets full-track playback inside the
 * iframe off their own session; everyone else gets Spotify's normal 30s
 * preview. We just need the bare Spotify track id, resolved server-side. */
async function toggleSpotifyEmbed(id, btn) {
  const wrap = document.getElementById('spotify-embed-wrap');
  if (!wrap) return;

  // Toggle closed if this button's embed is already showing.
  if (!wrap.hidden && wrap.dataset.forId === id) {
    wrap.hidden = true;
    wrap.innerHTML = '';
    wrap.dataset.forId = '';
    return;
  }

  if (state.audio) { state.audio.pause(); }   // don't double up with preview audio
  if (AtlasAutoplay.isRunning() && !AtlasAutoplay.isPaused()) AtlasAutoplay.pause();

  if (btn.dataset.trackId) {
    renderSpotifyEmbed(wrap, id, btn.dataset.trackId);
    return;
  }

  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = '…';
  wrap.hidden = false;
  wrap.dataset.forId = id;
  wrap.innerHTML = '<div class="empty spotify-embed-status">Looking up on Spotify…</div>';
  try {
    const data = await fetch('/api/song/' + encodeURIComponent(id) + '/spotify').then(r => r.json());
    btn.disabled = false;
    btn.textContent = original;
    if (state.detailId !== id && wrap.dataset.forId !== id) return;  // superseded
    if (!data.track_id) {
      btn.title = 'Not found on Spotify';
      wrap.innerHTML = '<div class="empty spotify-embed-status">Not available on Spotify.</div>';
      return;
    }
    btn.dataset.trackId = data.track_id;
    renderSpotifyEmbed(wrap, id, data.track_id);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = original;
    wrap.innerHTML = '<div class="empty spotify-embed-status">Lookup failed — try again.</div>';
  }
}

function renderSpotifyEmbed(wrap, id, trackId) {
  wrap.innerHTML = `<iframe
      style="border-radius:12px" src="https://open.spotify.com/embed/track/${trackId}"
      width="100%" height="152" frameBorder="0"
      allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
      loading="lazy"></iframe>`;
  wrap.hidden = false;
  wrap.dataset.forId = id;
}

/* ── Upload → place ─────────────────────────────────────────────────────── */
function initUpload() {
  initArtistAssignment();
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('upload-input');

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', async e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    for (const file of e.dataTransfer.files) await uploadFile(file);
  });
  zone.addEventListener('click', e => { if (e.target !== input) input.click(); });
  input.addEventListener('change', async e => {
    for (const file of e.target.files) await uploadFile(file);
    input.value = '';
  });
}

async function uploadFile(file) {
  const assignment = await chooseArtistForUpload(file);
  if (assignment === null) return;
  const status = document.getElementById('place-status');
  status.textContent = `Placing ${file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  if (assignment.artist_id) fd.append('artist_id', assignment.artist_id);
  if (assignment.artist_name) fd.append('artist_name', assignment.artist_name);
  try {
    const data = await fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());
    if (data.error) { showError(data.error); status.textContent = ''; return; }
    if (data.fragment) { AtlasGraph.mergeFragment(data.fragment); renderMapPanel(); }
    if (data.artist_fragment && state.artistAvailable) {
      ArtistGraph.mergeFragment(data.artist_fragment);
      renderArtistMapPanel();
    }
    status.textContent = `Placed ${data.title} ✓`;
    setTimeout(() => { status.textContent = ''; }, 2500);
  } catch (err) {
    showError('Upload failed: ' + err.message);
    status.textContent = '';
  }
}

let artistAssignmentResolve = null;
let artistAssignmentTimer = null;

function initArtistAssignment() {
  const modal = document.getElementById('artist-assignment-modal');
  const input = document.getElementById('artist-assignment-input');
  document.getElementById('artist-assignment-close').addEventListener('click', () => settleArtistAssignment(null));
  modal.querySelector('.modal-overlay').addEventListener('click', () => settleArtistAssignment(null));
  document.getElementById('artist-assignment-personal').addEventListener('click', () => settleArtistAssignment({}));
  document.getElementById('artist-assignment-create').addEventListener('click', () => {
    const name = input.value.trim();
    if (name) settleArtistAssignment({artist_name: name});
  });
  input.addEventListener('input', () => {
    clearTimeout(artistAssignmentTimer);
    const q = input.value.trim();
    const create = document.getElementById('artist-assignment-create');
    create.disabled = !q;
    create.textContent = q ? `Create private “${q}”` : 'Create private artist';
    if (!q) {
      document.getElementById('artist-assignment-results').innerHTML = '';
      return;
    }
    artistAssignmentTimer = setTimeout(() => searchUploadArtists(q), 300);
  });
}

function chooseArtistForUpload(file) {
  if (!state.artistAvailable) return Promise.resolve({});
  const modal = document.getElementById('artist-assignment-modal');
  const input = document.getElementById('artist-assignment-input');
  document.getElementById('artist-assignment-file').textContent = `Choose an artist for ${file.name}`;
  document.getElementById('artist-assignment-results').innerHTML = '';
  document.getElementById('artist-assignment-create').disabled = true;
  document.getElementById('artist-assignment-create').textContent = 'Create private artist';
  input.value = '';
  modal.hidden = false;
  setTimeout(() => input.focus(), 0);
  return new Promise(resolve => { artistAssignmentResolve = resolve; });
}

function settleArtistAssignment(value) {
  if (!artistAssignmentResolve) return;
  const resolve = artistAssignmentResolve;
  artistAssignmentResolve = null;
  document.getElementById('artist-assignment-modal').hidden = true;
  resolve(value);
}

async function searchUploadArtists(q) {
  const el = document.getElementById('artist-assignment-results');
  el.innerHTML = '<div class="empty">Searching…</div>';
  try {
    const response = await fetch(`/api/artist/search?q=${encodeURIComponent(q)}&limit=8`);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Artist search failed');
    const rows = data.results || [];
    el.innerHTML = rows.length ? rows.map(row => `
      <button class="artist-assignment-row" onclick='settleArtistAssignment({artist_id:${JSON.stringify(row.id)}})'>
        <span>${esc(row.name)}</span><small>${row.track_count || 0} tracks${row.source === 'session' ? ' · private' : ''}</small>
      </button>`).join('') : '<div class="empty">No existing artist found</div>';
    const exact = rows.some(row => row.score === 100);
    const create = document.getElementById('artist-assignment-create');
    create.disabled = exact;
    if (exact) create.textContent = 'Select the existing artist above';
  } catch (err) {
    el.innerHTML = '<div class="empty">Search unavailable</div>';
  }
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                        .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

let toastTimer;
function showError(msg) {
  const el = document.getElementById('error-toast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

/* ── Boot ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);

function toggleMentorPanel() {
  const app = document.querySelector('.app');
  const mentorPanel = document.querySelector('.mentor-panel');
  const isOpen = app.classList.toggle('mentor-open');
  mentorPanel.hidden = !isOpen;
  if (isOpen) {
    // Focus input when opening
    setTimeout(() => document.getElementById('mentor-input').focus(), 100);
  }
}

/* Legacy left-drawer shell. The bottom-sheet shell in mobile.js replaces it
 * wherever it activates; this stays for the no-JS-module path and for the
 * 768px band a rotation can drop out of. */
function initMobileSidebar() {
  if (typeof AtlasMobile !== 'undefined' && AtlasMobile.isActive()) return;
  const app = document.querySelector('.app');
  const toggle = document.querySelector('.sidebar-toggle');
  const panelLeft = document.querySelector('.panel-left');
  const graphWrap = document.querySelector('.graph-wrap');
  
  if (!toggle) return; // Desktop only has toggle hidden
  
  // Toggle sidebar on button click
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    app.classList.toggle('sidebar-open');
  });
  
  // Close sidebar when clicking on the graph
  graphWrap.addEventListener('click', () => {
    app.classList.remove('sidebar-open');
  });
  
  // Close sidebar when clicking on a panel item (search result, recommendation, etc.)
  panelLeft.addEventListener('click', (e) => {
    // Don't close if clicking on interactive elements like inputs
    if (e.target.closest('input, .btn-primary, .btn-tiny, .mode-btn')) return;
    // Close on track results, playlist results, etc.
    if (e.target.closest('.track-row, #filter-active, #map-list .node-row, #removed-list .node-row')) {
      setTimeout(() => app.classList.remove('sidebar-open'), 100);
    }
  });
  
  // Close sidebar when clicking the semi-transparent backdrop
  app.addEventListener('click', (e) => {
    if (e.target === app && app.classList.contains('sidebar-open')) {
      app.classList.remove('sidebar-open');
    }
  });
}

/* ── Help Modal ─────────────────────────────────────────────────────────────── */
function initHelp() {
  const modal = document.getElementById('help-modal');
  const helpToggle = document.getElementById('help-toggle');
  const helpClose = document.getElementById('help-close');
  const demoBtnEl = document.getElementById('try-demo-btn');
  const artistDemoBtn = document.getElementById('try-artist-demo-btn');
  const tabs = document.querySelectorAll('.modal-tab');
  const tabContents = document.querySelectorAll('.modal-tab-content');

  if (!sessionStorage.getItem('atlas-help-seen')) {
    modal.removeAttribute('hidden');
    sessionStorage.setItem('atlas-help-seen', 'true');
  }
  
  // Open modal
  helpToggle.addEventListener('click', () => {
    modal.removeAttribute('hidden');
  });
  
  // Close modal on close button
  helpClose.addEventListener('click', () => {
    modal.setAttribute('hidden', '');
  });
  
  // Close modal on overlay click
  const overlay = modal.querySelector('.modal-overlay');
  overlay.addEventListener('click', () => {
    modal.setAttribute('hidden', '');
  });
  
  // Close modal on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.hasAttribute('hidden')) {
      modal.setAttribute('hidden', '');
    }
  });
  
  // Tab switching
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabName = tab.getAttribute('data-tab');
      
      // Update active tab button
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      // Update active tab content
      tabContents.forEach(content => content.classList.remove('active'));
      document.querySelector(`.modal-tab-content[data-tab="${tabName}"]`).classList.add('active');
    });
  });
  
  // Try demo button: load the 2025 year-end top 20 chart, via the same
  // loadCollection() path albums/playlists use (instant + streaming fragments,
  // no page refresh needed).
  demoBtnEl.addEventListener('click', async () => {
    modal.setAttribute('hidden', '');
    // gid must match the pid loadCollection()/startPlaylistPoll() actually key
    // polling state on (playlist.pid from the response, i.e. the demo's fixed
    // custom-playlist id) — a mismatched guard key here let duplicate clicks
    // fire overlapping loads instead of being coalesced.
    await loadCollection('/api/demo/load', {}, 'demo_top20_2025', demoBtnEl);
  });

  artistDemoBtn.disabled = !state.artistAvailable;
  artistDemoBtn.addEventListener('click', async () => {
    if (!state.artistAvailable) return;
    artistDemoBtn.disabled = true;
    artistDemoBtn.textContent = 'Loading artists…';
    try {
      const response = await fetch('/api/artist/demo', {method: 'POST'});
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Artist demo failed');
      ArtistGraph.replaceData(data);
      renderArtistMapPanel();
      modal.setAttribute('hidden', '');
      setViewMode('artists');
    } catch (err) { showError(err.message); }
    finally {
      artistDemoBtn.disabled = false;
      artistDemoBtn.textContent = 'Load 20-artist demo →';
    }
  });
}
