/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  staging:     [],
  jobId:       null,
  pollTimer:   null,
  audio:       null,
  audioBtn:    null,
};

/* ── Cluster palette (matches jobs.py / song_view.html) ─────────────────── */
const PALETTE = [
  '#7b5ea7','#4e9a8a','#c46b3a','#4a7bbf','#a0516a',
  '#6b9e3c','#9b6b3c','#3c7b9e','#9e3c6b','#5ea74a',
];
function clusterColor(id) {
  if (id === -1) return '#555577';
  return PALETTE[id % PALETTE.length];
}

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  await refreshStaging();
  initUpload();
  initSearch();
  document.getElementById('cluster-btn').addEventListener('click', startCluster);

  // Load any existing results from a previous run
  try {
    const res = await fetch('/api/results');
    if (res.ok) renderResults(await res.json());
  } catch (_) { /* no prior results — that's fine */ }
}

/* ── Search ─────────────────────────────────────────────────────────────── */
function initSearch() {
  let timer;
  document.getElementById('search-input').addEventListener('input', e => {
    clearTimeout(timer);
    const q = e.target.value.trim();
    if (!q) { renderSearchResults([]); return; }
    timer = setTimeout(() => doSearch(q), 420);
  });
}

async function doSearch(q) {
  const container = document.getElementById('search-results');
  container.innerHTML = '<div class="empty">Searching…</div>';
  try {
    const res = await fetch(`/api/deezer/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (data.error) { showError('Deezer: ' + data.error); renderSearchResults([]); return; }
    renderSearchResults(data);
  } catch (err) {
    showError('Search failed: ' + err.message);
    renderSearchResults([]);
  }
}

function renderSearchResults(hits) {
  const el = document.getElementById('search-results');
  if (!hits.length) {
    el.innerHTML = '<div class="empty">No results</div>';
    return;
  }
  el.innerHTML = hits.map(h => `
    <div class="track-row" data-id="${h.deezer_id}">
      <img class="cover" src="${esc(h.cover)}" alt="" loading="lazy" />
      <div class="track-info">
        <div class="track-title">${esc(h.title)}</div>
        <div class="track-artist">${esc(h.artist)}</div>
      </div>
      <div class="track-actions">
        <button class="btn-icon" title="Preview"
          onclick="togglePreview('${esc(h.preview_url)}', this)">▶</button>
        <button class="btn-add"
          onclick='addHit(${JSON.stringify(h)}, this)'>Add</button>
      </div>
    </div>
  `).join('');
}

/* ── Preview audio ───────────────────────────────────────────────────────── */
function togglePreview(url, btn) {
  if (state.audio && state.audioUrl === url) {
    state.audio.pause();
    state.audio = null;
    state.audioUrl = null;
    btn.textContent = '▶';
    btn.classList.remove('playing');
    return;
  }
  // stop whatever is playing
  if (state.audio) {
    state.audio.pause();
    if (state.audioBtn) { state.audioBtn.textContent = '▶'; state.audioBtn.classList.remove('playing'); }
  }
  state.audio    = new Audio(url);
  state.audioUrl = url;
  state.audioBtn = btn;
  state.audio.play().catch(() => {});
  btn.textContent = '⏸';
  btn.classList.add('playing');
  state.audio.onended = () => {
    btn.textContent = '▶';
    btn.classList.remove('playing');
    state.audio = null; state.audioUrl = null; state.audioBtn = null;
  };
}

/* ── Stage add ───────────────────────────────────────────────────────────── */
async function addHit(hit, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const res  = await fetch('/api/stage', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(hit),
    });
    const data = await res.json();
    btn.textContent = data.status === 'duplicate' ? 'Already added' : 'Added ✓';
    await refreshStaging();
  } catch (err) {
    btn.textContent = 'Error';
    showError(err.message);
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Add'; }, 2200);
}

/* ── Stage remove ────────────────────────────────────────────────────────── */
async function removeFromStage(id) {
  await fetch(`/api/stage/${encodeURIComponent(id)}`, { method: 'DELETE' });
  await refreshStaging();
}

async function refreshStaging() {
  const data = await fetch('/api/stage').then(r => r.json());
  state.staging = data;
  renderStaging(data);
}

function renderStaging(items) {
  const list  = document.getElementById('staging-list');
  const count = document.getElementById('staging-count');
  const btn   = document.getElementById('cluster-btn');
  const hint  = document.getElementById('cluster-hint');

  count.textContent = items.length;

  if (items.length === 0) {
    list.innerHTML = '<div class="empty">Add tracks from search or upload</div>';
  } else {
    list.innerHTML = items.map(m => `
      <div class="staged-item">
        <span class="badge badge-${m.type}">${m.type === 'deezer' ? 'deezer' : 'upload'}</span>
        <div class="staged-info">
          <div class="staged-title">${esc(m.title || m.filename || '')}</div>
          <div class="staged-artist">${esc(m.artist || '')}</div>
        </div>
        <button class="btn-remove" onclick="removeFromStage('${esc(m.id)}')" title="Remove">✕</button>
      </div>
    `).join('');
  }

  const need = Math.max(0, 15 - items.length);
  if (need > 0) {
    btn.disabled    = true;
    hint.textContent = `Add ${need} more track${need !== 1 ? 's' : ''} to cluster`;
  } else {
    btn.disabled    = false;
    hint.textContent = `${items.length} tracks ready`;
  }
}

/* ── Upload ──────────────────────────────────────────────────────────────── */
function initUpload() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('upload-input');

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()=> zone.classList.remove('drag-over'));
  zone.addEventListener('drop', async e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    for (const file of e.dataTransfer.files) await uploadFile(file);
  });
  zone.addEventListener('click', e => {
    if (e.target !== input) input.click();
  });
  input.addEventListener('change', async e => {
    for (const file of e.target.files) await uploadFile(file);
    input.value = '';
  });
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    await refreshStaging();
  } catch (err) {
    showError('Upload failed: ' + err.message);
  }
}

/* ── Cluster ─────────────────────────────────────────────────────────────── */
async function startCluster() {
  document.getElementById('cluster-btn').disabled = true;
  showProgress(true, 'Starting…', 0);

  try {
    const res  = await fetch('/api/cluster', { method: 'POST' });
    const data = await res.json();
    if (data.error) {
      showError(data.error);
      document.getElementById('cluster-btn').disabled = false;
      showProgress(false);
      return;
    }
    state.jobId = data.job_id;
    startPolling(data.job_id);
  } catch (err) {
    showError('Failed to start cluster: ' + err.message);
    document.getElementById('cluster-btn').disabled = false;
    showProgress(false);
  }
}

function startPolling(jobId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const status = await fetch(`/api/cluster/status/${jobId}`).then(r => r.json());
      const pct = status.total > 0 ? status.progress / status.total : 0;
      showProgress(true, status.message || '…', pct);

      if (status.state === 'done') {
        clearInterval(state.pollTimer);
        showProgress(false);
        document.getElementById('cluster-btn').disabled = false;
        const results = await fetch('/api/results').then(r => r.json());
        renderResults(results);
        // Refresh staging to reflect new cluster assignments
        await refreshStaging();
      } else if (status.state === 'error') {
        clearInterval(state.pollTimer);
        showProgress(false);
        document.getElementById('cluster-btn').disabled = false;
        showError('Cluster failed: ' + status.message);
      }
    } catch (_) { /* transient network hiccup — keep polling */ }
  }, 1500);
}

function showProgress(visible, msg, pct) {
  const wrap = document.getElementById('progress-wrap');
  wrap.style.display = visible ? 'flex' : 'none';
  if (visible) {
    document.getElementById('progress-msg').textContent  = msg || '';
    document.getElementById('progress-fill').style.width = Math.round((pct || 0) * 100) + '%';
  }
}

/* ── Results ─────────────────────────────────────────────────────────────── */
function renderResults(data) {
  const section = document.getElementById('results-section');
  section.style.display = 'block';

  const clusterIds = [...new Set(data.labels)].sort((a, b) => a - b);
  const traces = clusterIds.map(cid => {
    const idx  = data.labels.reduce((acc, l, i) => { if (l === cid) acc.push(i); return acc; }, []);
    const x    = idx.map(i => data.embedding_2d[i][0]);
    const y    = idx.map(i => data.embedding_2d[i][1]);
    const text = idx.map(i => `${data.metadata[i].name}<br>${data.metadata[i].artist}`);
    return {
      type:  'scattergl',
      mode:  'markers',
      name:  cid === -1 ? 'noise' : `cluster ${cid}`,
      x, y, text,
      hovertemplate: '%{text}<extra></extra>',
      marker: {
        size:    cid === -1 ? 7 : 9,
        color:   clusterColor(cid),
        opacity: cid === -1 ? 0.35 : 0.85,
        line:    { width: 0 },
      },
    };
  });

  Plotly.newPlot('scatter', traces, {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font:          { color: '#7068a0', size: 12 },
    showlegend:    true,
    legend:        { bgcolor: 'transparent', bordercolor: 'transparent' },
    margin:        { l: 20, r: 20, t: 20, b: 20 },
    xaxis:         { showgrid: false, zeroline: false, showticklabels: false },
    yaxis:         { showgrid: false, zeroline: false, showticklabels: false },
  }, { responsive: true, displayModeBar: false });

  const nClusters = clusterIds.filter(c => c !== -1).length;
  const noisePct  = (data.labels.filter(l => l === -1).length / data.labels.length * 100).toFixed(0);
  document.getElementById('result-summary').textContent =
    `${data.labels.length} tracks · ${nClusters} clusters · ${noisePct}% noise`;

  renderTable(data);
}

function renderTable(data) {
  const rows = data.metadata
    .map((m, i) => ({ ...m, cluster: data.labels[i] }))
    .sort((a, b) => a.cluster - b.cluster);

  document.querySelector('#assignments-table tbody').innerHTML = rows.map(r => {
    const dot   = `<span class="cluster-badge" style="background:${clusterColor(r.cluster)}"></span>`;
    const label = r.cluster === -1 ? 'noise' : r.cluster;
    return `<tr>
      <td>${dot}${label}</td>
      <td>${esc(r.name)}</td>
      <td>${esc(r.artist)}</td>
      <td><span class="badge badge-${r.source}">${r.source}</span></td>
    </tr>`;
  }).join('');
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
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
