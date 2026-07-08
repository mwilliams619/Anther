/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  audio:      null,
  audioUrl:   null,
  audioBtn:   null,
  detailId:   null,   // song currently shown in the detail panel
  similar:    [],     // similar-song rows backing the panel's list
  searchMode: 'tracks',   // 'tracks' | 'playlists'
};

const SEARCH_MODES = {
  tracks: {
    placeholder: 'Artist, track…',
    hint: 'Corpus → Deezer → Spotify. Click a result to place it on the graph.',
    empty: 'Search for a song to add it to the atlas',
  },
  playlists: {
    placeholder: 'Playlist name…',
    hint: 'MPD playlists. Load one to place all its in-corpus tracks at once.',
    empty: 'Search for a playlist to load its tracks onto the atlas',
  },
};

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  await AtlasGraph.init('#graph');
  initSearch();
  initUpload();
  initDetail();
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

  document.getElementById('mode-tracks')
    .addEventListener('click', () => setSearchMode('tracks'));
  document.getElementById('mode-playlists')
    .addEventListener('click', () => setSearchMode('playlists'));
}

function setSearchMode(mode) {
  if (state.searchMode === mode) return;
  state.searchMode = mode;
  document.getElementById('mode-tracks').classList.toggle('active', mode === 'tracks');
  document.getElementById('mode-playlists').classList.toggle('active', mode === 'playlists');
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

async function doSearch(q) {
  const container = document.getElementById('search-results');
  container.innerHTML = '<div class="empty">Searching…</div>';
  const playlistMode = state.searchMode === 'playlists';
  const url = playlistMode
    ? `/api/playlists/search?q=${encodeURIComponent(q)}`
    : `/api/search?q=${encodeURIComponent(q)}`;
  try {
    const data = await fetch(url).then(r => r.json());
    if (data.error) { showError(data.error); clearSearchResults(); return; }
    if (playlistMode) renderPlaylistResults(data); else renderSearchResults(data);
  } catch (err) {
    showError('Search failed: ' + err.message);
    clearSearchResults();
  }
}

const SOURCE_LABEL = { corpus: 'corpus', deezer: 'deezer', spotify: 'spotify' };

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
  if (!hits.length) {
    el.innerHTML = '<div class="empty">No playlists found</div>';
    return;
  }
  el.innerHTML = hits.map(h => `
    <div class="track-row">
      <div class="track-info">
        <div class="track-title">${esc(h.name)}</div>
        <div class="playlist-count">${h.n_tracks} track${h.n_tracks === 1 ? '' : 's'} in corpus</div>
      </div>
      <div class="track-actions">
        <button class="btn-add" onclick='loadPlaylist(${JSON.stringify(h)}, this)'>Load</button>
      </div>
    </div>`).join('');
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
    btn.textContent = 'Added ✓';
  } catch (err) {
    showError('Placement failed: ' + err.message);
    btn.textContent = 'Error';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Add'; }, 2000);
}

async function loadPlaylist(hit, btn) {
  const hubId = 'playlist:' + hit.pid;
  if (AtlasGraph.hasNode(hubId)) {
    AtlasGraph.zoomTo(hubId, 0.9);
    btn.textContent = 'On graph ✓';
    setTimeout(() => { btn.textContent = 'Load'; }, 1800);
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const frag = await fetch('/api/playlist/place', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pid: hit.pid }),
    }).then(r => r.json());
    if (frag.error) { showError(frag.error); btn.disabled = false; btn.textContent = 'Load'; return; }
    AtlasGraph.mergeFragment(frag, { focusId: frag.playlist.hub_id, zoomScale: 0.9 });
    btn.textContent = 'Loaded ✓';
  } catch (err) {
    showError('Playlist load failed: ' + err.message);
    btn.textContent = 'Error';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Load'; }, 2000);
}

/* ── Detail panel (click a node → pinned side pop-over) ─────────────────── */
function initDetail() {
  AtlasGraph.onSelect(openDetail);
  AtlasGraph.onDeselect(closeDetail);
  document.getElementById('detail-close')
    .addEventListener('click', () => AtlasGraph.clearSelection());
}

async function openDetail(id) {
  const panel = document.getElementById('detail-panel');
  const body  = document.getElementById('detail-body');
  state.detailId = id;
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

function closeDetail() {
  document.getElementById('detail-panel').hidden = true;
  state.detailId = null;
  state.similar = [];
}

function renderDetail(d) {
  const c = d.cluster || {};
  const meta = [];
  if (c.id != null) meta.push(`cluster ${c.id}`);
  if (c.confidence != null) meta.push(`conf ${Math.round(c.confidence * 100)}%`);

  let html = `
    <div class="detail-title">${esc(d.name)}</div>
    <div class="detail-artist">${esc(d.artist)}</div>
    ${meta.length ? `<div class="detail-meta">${meta.join(' · ')}</div>` : ''}
    ${d.genre ? `<div class="detail-genre">${esc(d.genre)}</div>` : ''}`;

  const tags = d.tags || [];
  if (tags.length) {
    html += `<div class="detail-section"><h3>Micro-genres</h3><div class="tag-chips">`
      + tags.map(t => `<span class="tag-chip${t.primary ? ' primary' : ''}">${esc(t.genre)}</span>`).join('')
      + `</div></div>`;
  }

  const pls = d.playlists || [];
  if (pls.length) {
    const items = pls.slice(0, 8).map(p => `<li>${esc(p.name)}</li>`).join('')
      + (pls.length > 8 ? `<li class="more">+${pls.length - 8} more</li>` : '');
    html += `<div class="detail-section"><h3>Playlists</h3><ul class="detail-playlists">${items}</ul></div>`;
  }

  state.similar = d.similar || [];
  if (state.similar.length) {
    const listTitle = d.kind === 'playlist' ? 'Tracks' : 'Similar songs';
    html += `<div class="detail-section"><h3>${listTitle}</h3>` + state.similar.map((s, i) => {
      const score = s.score != null
        ? `<span class="similar-score">${Number(s.score).toFixed(3)}</span>` : '';
      const onGraph = s.on_graph || AtlasGraph.hasNode(s.id);
      const add = onGraph ? '' : `<button class="btn-add" onclick="addSimilar(${i}, this)">Add</button>`;
      return `
      <div class="similar-row${onGraph ? ' on-graph' : ''}"
           ${onGraph ? `onclick="gotoSimilar(${i})"` : ''}>
        <div class="track-info">
          <div class="track-title">${esc(s.name)}</div>
          <div class="track-artist">${esc(s.artist)}</div>
        </div>
        ${score}${add}
      </div>`;
    }).join('') + `</div>`;
  }

  document.getElementById('detail-body').innerHTML = html;
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
    AtlasGraph.selectNode(s.id, { zoom: true });
  } catch (err) {
    showError('Placement failed: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Add';
  }
}

/* ── Preview audio ──────────────────────────────────────────────────────── */
function togglePreview(url, btn) {
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

/* ── Upload → place ─────────────────────────────────────────────────────── */
function initUpload() {
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
  const status = document.getElementById('place-status');
  status.textContent = `Placing ${file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());
    if (data.error) { showError(data.error); status.textContent = ''; return; }
    if (data.fragment) AtlasGraph.mergeFragment(data.fragment);
    status.textContent = `Placed ${data.title} ✓`;
    setTimeout(() => { status.textContent = ''; }, 2500);
  } catch (err) {
    showError('Upload failed: ' + err.message);
    status.textContent = '';
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
