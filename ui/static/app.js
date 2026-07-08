/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  audio:    null,
  audioUrl: null,
  audioBtn: null,
};

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  await AtlasGraph.init('#graph');
  initSearch();
  initUpload();
}

/* ── Search (tiered: corpus → deezer → spotify) ─────────────────────────── */
function initSearch() {
  let timer;
  document.getElementById('search-input').addEventListener('input', e => {
    clearTimeout(timer);
    const q = e.target.value.trim();
    if (!q) { renderSearchResults({ results: [] }); return; }
    timer = setTimeout(() => doSearch(q), 420);
  });
}

async function doSearch(q) {
  const container = document.getElementById('search-results');
  container.innerHTML = '<div class="empty">Searching…</div>';
  try {
    const data = await fetch(`/api/search?q=${encodeURIComponent(q)}`).then(r => r.json());
    if (data.error) { showError(data.error); renderSearchResults({ results: [] }); return; }
    renderSearchResults(data);
  } catch (err) {
    showError('Search failed: ' + err.message);
    renderSearchResults({ results: [] });
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
