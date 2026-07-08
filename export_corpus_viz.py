"""
Generate a self-contained scatter-plot viewer for a corpus bundle
(models/corpus_<name>/), e.g. the 25k mpd_val corpus.

Unlike export_viz.py (d3 force graph, meant for ~100s of Phase 2 tracks),
this renders a canvas scatter plot colored by cluster, suited to tens of
thousands of points: pan/zoom + hover-to-inspect, no force simulation.

Run from the repo root:
    python export_corpus_viz.py models/corpus_mpd_val_25k
"""

import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np


def _load_overlay(path):
    """Load placed query songs (from place_local's JSON or a CSV) as overlay
    points: [{x, y, c, f, nn, conf}]. Rows without 2D coords are skipped."""
    path = Path(path)
    if path.suffix == '.json':
        rows = json.loads(path.read_text()).get('rows', [])
    else:
        with open(path, newline='') as fh:
            rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        x, y = r.get('x'), r.get('y')
        if x in (None, '', 'None') or y in (None, '', 'None'):
            continue
        out.append({
            'x': float(x), 'y': float(y), 'c': int(r['cluster']),
            'f': r.get('file', ''), 'nn': r.get('top_neighbor', ''),
            'conf': float(r.get('confidence', 0) or 0),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('corpus_dir', nargs='?', default='models/corpus_mpd_val_25k')
    ap.add_argument('out_path', nargs='?', default=None)
    ap.add_argument('--overlay', default=None,
                    help='placements JSON/CSV to overlay as query songs')
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_path = Path(args.out_path) if args.out_path else Path(f'{corpus_dir.name}_view.html')
    overlay = _load_overlay(args.overlay) if args.overlay else []

    manifest = json.loads((corpus_dir / 'manifest.json').read_text())
    index = json.loads((corpus_dir / 'index.json').read_text())
    metadata = index['metadata']
    emb2d = np.load(corpus_dir / 'embedding_2d.npy')
    labels = np.load(corpus_dir / 'labels.npy')

    profiles = {}
    profiles_path = corpus_dir / 'cluster_profiles.json'
    if profiles_path.exists():
        for p in json.loads(profiles_path.read_text()):
            profiles[p['cluster_id']] = p

    points = []
    for i, m in enumerate(metadata):
        points.append({
            'x': float(emb2d[i, 0]),
            'y': float(emb2d[i, 1]),
            'c': int(labels[i]),
            'n': m.get('name', ''),
            'a': m.get('artist', ''),
        })

    n_clusters = int(labels.max()) + 1
    cluster_labels = []
    for cid in range(n_clusters):
        prof = profiles.get(cid, {})
        exemplars = ', '.join(e['name'] for e in prof.get('exemplars', [])[:3])
        cluster_labels.append(f"Cluster {cid} (n={prof.get('size', '?')}): {exemplars}")

    html = TEMPLATE.replace('__POINTS__', json.dumps(points)) \
                    .replace('__CLUSTER_LABELS__', json.dumps(cluster_labels)) \
                    .replace('__OVERLAY__', json.dumps(overlay)) \
                    .replace('__TITLE__', manifest.get('name', corpus_dir.name)) \
                    .replace('__N__', str(len(points)))

    out_path.write_text(html)
    print(f'Wrote {out_path} ({len(points)} tracks, {n_clusters} clusters, '
          f'{len(overlay)} overlay songs)')


TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__ corpus view</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }
  #canvas { display: block; cursor: grab; }
  #canvas.grabbing { cursor: grabbing; }
  #tooltip {
    position: fixed; pointer-events: none; background: rgba(20,20,20,0.95);
    border: 1px solid #444; border-radius: 6px; padding: 6px 10px; font-size: 13px;
    display: none; max-width: 280px; z-index: 10;
  }
  #legend {
    position: fixed; top: 12px; left: 12px; background: rgba(20,20,20,0.85);
    border: 1px solid #333; border-radius: 8px; padding: 10px 14px; font-size: 12px;
    max-width: 320px; max-height: 80vh; overflow-y: auto;
  }
  #legend h3 { margin: 0 0 8px; font-size: 14px; }
  .legend-item { display: flex; align-items: flex-start; gap: 6px; margin: 3px 0; cursor: pointer; }
  .legend-item.dim { opacity: 0.35; }
  .swatch { width: 10px; height: 10px; border-radius: 50%; margin-top: 3px; flex-shrink: 0; }
  #info { position: fixed; bottom: 12px; left: 12px; font-size: 11px; color: #888; }
  #search { position: fixed; top: 12px; right: 12px; }
  #search input {
    background: #1a1a1a; border: 1px solid #444; color: #eee; padding: 6px 10px;
    border-radius: 6px; font-size: 13px; width: 220px;
  }
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="tooltip"></div>
<div id="legend"><h3>__TITLE__ (__N__ tracks)</h3><div id="legend-items"></div></div>
<div id="search"><input id="search-input" placeholder="Search title or artist..."></div>
<div id="info">Scroll to zoom, drag to pan, click a cluster to isolate &nbsp;·&nbsp; <b>♦ diamonds = your songs</b> &nbsp;·&nbsp; press <b>d</b> to dim corpus</div>
<script>
const points = __POINTS__;
const overlay = __OVERLAY__;
const clusterLabels = __CLUSTER_LABELS__;
const nClusters = clusterLabels.length;
let dimCorpus = false;

const colors = [
  '#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f',
  '#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac',
  '#86bcb6','#f1ce63','#d37295','#a0cbe8','#ffbe7d'
];

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

let dpr = window.devicePixelRatio || 1;
function resize() {
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
}
resize();
window.addEventListener('resize', () => { resize(); draw(); });

const xs = points.map(p => p.x).concat(overlay.map(p => p.x));
const ys = points.map(p => p.y).concat(overlay.map(p => p.y));
const minX = Math.min(...xs), maxX = Math.max(...xs);
const minY = Math.min(...ys), maxY = Math.max(...ys);
const dataW = maxX - minX || 1, dataH = maxY - minY || 1;

let scale, offX, offY;
function fit() {
  const pad = 60 * dpr;
  const availW = canvas.width - 2 * pad, availH = canvas.height - 2 * pad;
  scale = Math.min(availW / dataW, availH / dataH);
  offX = pad + (availW - dataW * scale) / 2 - minX * scale;
  offY = pad + (availH - dataH * scale) / 2 - minY * scale;
}
fit();

let hiddenClusters = new Set();
let hoverIdx = -1;
let overlayHoverIdx = -1;
let searchQuery = '';

function toScreen(p) { return [p.x * scale + offX, p.y * scale + offY]; }
function toData(sx, sy) { return [(sx - offX) / scale, (sy - offY) / scale]; }

function draw() {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const r = Math.max(1.2, 2.2 * Math.sqrt(scale) * dpr * 0.15) * dpr * 0.8;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (hiddenClusters.has(p.c)) continue;
    const matchesSearch = searchQuery && (
      p.n.toLowerCase().includes(searchQuery) || p.a.toLowerCase().includes(searchQuery)
    );
    const [sx, sy] = toScreen(p);
    if (sx < -10 || sx > canvas.width + 10 || sy < -10 || sy > canvas.height + 10) continue;
    ctx.beginPath();
    const rad = (searchQuery && matchesSearch) ? r * 2.5 : r;
    ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    ctx.fillStyle = colors[p.c % colors.length];
    ctx.globalAlpha = searchQuery ? (matchesSearch ? 1.0 : 0.08) : (dimCorpus ? 0.12 : 0.85);
    ctx.fill();
  }
  ctx.globalAlpha = 1.0;
  if (hoverIdx >= 0) {
    const p = points[hoverIdx];
    const [sx, sy] = toScreen(p);
    ctx.beginPath();
    ctx.arc(sx, sy, r * 3, 0, Math.PI * 2);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();
  }
  // overlay: query songs drawn on top as large diamonds, colored by assigned
  // cluster with a white outline so they stand out against the corpus cloud.
  const orad = Math.max(5 * dpr, r * 3.5);
  for (let i = 0; i < overlay.length; i++) {
    const p = overlay[i];
    if (hiddenClusters.has(p.c)) continue;
    const [sx, sy] = toScreen(p);
    if (sx < -20 || sx > canvas.width + 20 || sy < -20 || sy > canvas.height + 20) continue;
    const hovered = (i === overlayHoverIdx);
    const rr = hovered ? orad * 1.5 : orad;
    ctx.beginPath();
    ctx.moveTo(sx, sy - rr); ctx.lineTo(sx + rr, sy);
    ctx.lineTo(sx, sy + rr); ctx.lineTo(sx - rr, sy); ctx.closePath();
    ctx.fillStyle = colors[p.c % colors.length];
    ctx.globalAlpha = 1.0;
    ctx.fill();
    ctx.strokeStyle = hovered ? '#fff' : '#000';
    ctx.lineWidth = 2 * dpr;
    ctx.stroke();
  }
}
draw();

function buildLegend() {
  const container = document.getElementById('legend-items');
  container.innerHTML = '';
  for (let c = 0; c < nClusters; c++) {
    const div = document.createElement('div');
    div.className = 'legend-item' + (hiddenClusters.has(c) ? ' dim' : '');
    div.innerHTML = `<span class="swatch" style="background:${colors[c % colors.length]}"></span><span>${clusterLabels[c]}</span>`;
    div.onclick = () => {
      if (hiddenClusters.has(c)) hiddenClusters.delete(c); else hiddenClusters.add(c);
      buildLegend();
      draw();
    };
    container.appendChild(div);
  }
}
buildLegend();

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', e => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.classList.add('grabbing');
});
window.addEventListener('mouseup', () => { dragging = false; canvas.classList.remove('grabbing'); });
window.addEventListener('mousemove', e => {
  if (dragging) {
    offX += (e.clientX - lastX) * dpr;
    offY += (e.clientY - lastY) * dpr;
    lastX = e.clientX; lastY = e.clientY;
    draw();
    return;
  }
  const mx = e.clientX * dpr, my = e.clientY * dpr;
  // overlay songs have priority — they sit on top and have a larger hit radius
  let oBest = -1, oBestD = 14 * dpr;
  for (let i = 0; i < overlay.length; i++) {
    if (hiddenClusters.has(overlay[i].c)) continue;
    const [sx, sy] = toScreen(overlay[i]);
    const d = Math.hypot(sx - mx, sy - my);
    if (d < oBestD) { oBestD = d; oBest = i; }
  }
  overlayHoverIdx = oBest;
  if (oBest >= 0) {
    const p = overlay[oBest];
    hoverIdx = -1;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
    tooltip.innerHTML = `<b>♦ ${p.f}</b><br>` +
      `<span style="color:${colors[p.c % colors.length]}">Cluster ${p.c}</span> ` +
      `(conf ${Math.round(p.conf * 100)}%)<br>` +
      `<span style="color:#aaa">nearest: ${p.nn}</span>`;
    draw();
    return;
  }
  let best = -1, bestD = 8 * dpr;
  for (let i = 0; i < points.length; i++) {
    if (hiddenClusters.has(points[i].c)) continue;
    const [sx, sy] = toScreen(points[i]);
    const d = Math.hypot(sx - mx, sy - my);
    if (d < bestD) { bestD = d; best = i; }
  }
  hoverIdx = best;
  if (best >= 0) {
    const p = points[best];
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
    tooltip.innerHTML = `<b>${p.n}</b><br>${p.a}<br><span style="color:${colors[p.c % colors.length]}">Cluster ${p.c}</span>`;
  } else {
    tooltip.style.display = 'none';
  }
  draw();
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const mx = e.clientX * dpr, my = e.clientY * dpr;
  const [dx, dy] = toData(mx, my);
  const factor = Math.exp(-e.deltaY * 0.001);
  scale *= factor;
  const [nsx, nsy] = toScreen({x: dx, y: dy});
  offX += mx - nsx;
  offY += my - nsy;
  draw();
}, { passive: false });

document.getElementById('search-input').addEventListener('input', e => {
  searchQuery = e.target.value.trim().toLowerCase();
  draw();
});

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'd' || e.key === 'D') { dimCorpus = !dimCorpus; draw(); }
});
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
