"""
Generate a self-contained song visualizer HTML file.

Run from the repo root:
    python export_viz.py

Opens song_view.html in your default browser when done.
No server or Django required.
"""

import sys, json, webbrowser
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from anther_ml.similarity import SongIndex

PHASE   = 2
TOP_N   = 5
OUT     = Path('song_view.html')

embedding_2d = np.load(f'models/embedding_2d_phase{PHASE}.npy')
index        = SongIndex.load(f'models/index_phase{PHASE}')

sims = index.embeddings @ index.embeddings.T
np.fill_diagonal(sims, -1)

nodes = []
for i, m in enumerate(index.metadata):
    top_idx   = np.argsort(sims[i])[::-1][:TOP_N]
    neighbors = [index.metadata[j].get('name', f'track_{j}') for j in top_idx]
    scores    = [float(sims[i, j]) for j in top_idx]
    nodes.append({
        'id':        m.get('name', f'track_{i}'),
        'artist':    m.get('artist', ''),
        'cluster':   int(m.get('cluster', -1)),
        'x':         float(embedding_2d[i, 0]),
        'y':         float(embedding_2d[i, 1]),
        'neighbors': neighbors,
        'scores':    scores,
    })

name_to_node = {n['id']: n for n in nodes}

links, seen = [], set()
for node in nodes:
    for neighbor_name, score in zip(node['neighbors'], node['scores']):
        if neighbor_name not in name_to_node:
            continue
        key = tuple(sorted([node['id'], neighbor_name]))
        if key not in seen:
            seen.add(key)
            links.append({'source': node['id'], 'target': neighbor_name, 'value': score})

graph_json = json.dumps({'nodes': nodes, 'links': links})

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Song Map</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{
      width: 100%; height: 100%;
      margin: 0; padding: 0;
      overflow: hidden;
      font-family: 'Helvetica Neue', sans-serif;
      background: #0e0b1a;
      color: #ddd;
    }}
    svg {{ width: 100%; height: 100%; display: block; }}

    .link {{
      stroke: #3a3560;
      stroke-opacity: 0.35;
      stroke-width: 1px;
    }}
    .link.neighbor {{
      stroke: #ffaa33;
      stroke-opacity: 0.9;
      stroke-width: 2px;
    }}
    .node {{ cursor: grab; }}
    .node:active {{ cursor: grabbing; }}
    .node.dimmed {{ opacity: 0.15; }}
    .node.neighbor-node {{
      stroke: #ffaa33 !important;
      stroke-width: 2.5px !important;
    }}

    #tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(14, 11, 28, 0.95);
      border: 1px solid #5a3a9a;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 13px;
      line-height: 1.7;
      max-width: 250px;
      display: none;
      z-index: 100;
      box-shadow: 0 4px 24px rgba(0,0,0,0.7);
    }}
    .tip-title {{
      font-weight: 700;
      font-size: 14px;
      color: #fff;
      margin-bottom: 7px;
      word-break: break-word;
    }}
    .tip-label {{
      font-size: 11px;
      color: #9988cc;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 4px;
    }}
    .tip-neighbor {{
      color: #ffcc88;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .tip-score {{
      color: #7766aa;
      font-size: 11px;
      margin-left: 5px;
    }}
    #hint {{
      position: fixed;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 12px;
      color: #4a4070;
      pointer-events: none;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <div id="tooltip"></div>
  <svg></svg>
  <div id="hint">drag nodes &nbsp;·&nbsp; scroll to zoom &nbsp;·&nbsp; hover to explore</div>

  <script src="https://d3js.org/d3.v4.js"></script>
  <script>
  const graph = {graph_json};

  const W = window.innerWidth;
  const H = window.innerHeight;

  const svg = d3.select('svg');
  const g   = svg.append('g');

  svg.call(d3.zoom().scaleExtent([0.15, 10])
    .on('zoom', () => g.attr('transform', d3.event.transform)));

  // Scale UMAP coords to fill viewport
  const pad  = 60;
  const xExt = d3.extent(graph.nodes, d => d.x);
  const yExt = d3.extent(graph.nodes, d => d.y);
  const xSc  = d3.scaleLinear().domain(xExt).range([pad, W - pad]);
  const ySc  = d3.scaleLinear().domain(yExt).range([pad, H - pad]);
  graph.nodes.forEach(n => {{ n.x = xSc(n.x); n.y = ySc(n.y); }});

  // Cluster colours
  const clusters = Array.from(new Set(graph.nodes.map(d => d.cluster))).sort((a,b)=>a-b);
  const palette  = ['#7b5ea7','#4e9a8a','#c46b3a','#4a7bbf','#a0516a',
                    '#6b9e3c','#9b6b3c','#3c7b9e','#9e3c6b','#5ea74a'];
  const clusterColor = d => {{
    if (d.cluster === -1) return '#444466';
    const idx = clusters.filter(c => c !== -1).indexOf(d.cluster);
    return palette[idx % palette.length];
  }};

  // Neighbor lookup
  const neighborSet = {{}};
  graph.nodes.forEach(n => {{ neighborSet[n.id] = new Set(n.neighbors || []); }});

  // Force simulation — weak forces so UMAP layout is preserved
  const sim = d3.forceSimulation(graph.nodes)
    .force('link',    d3.forceLink(graph.links).id(d => d.id).distance(45).strength(0.06))
    .force('charge',  d3.forceManyBody().strength(-20))
    .force('collide', d3.forceCollide(10))
    .alphaDecay(0.03);

  const link = g.append('g').selectAll('line')
    .data(graph.links).enter().append('line').attr('class', 'link');

  const node = g.append('g').selectAll('circle')
    .data(graph.nodes).enter().append('circle')
      .attr('class', 'node')
      .attr('r', 6)
      .attr('fill', clusterColor)
      .attr('stroke', d => d3.color(clusterColor(d)).brighter(0.7))
      .attr('stroke-width', 1.5)
      .on('mouseover', onHover)
      .on('mousemove', onMove)
      .on('mouseout',  onOut)
      .call(d3.drag()
        .on('start', dragStart)
        .on('drag',  dragged)
        .on('end',   dragEnd));

  sim.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
  }});

  const tooltip = d3.select('#tooltip');

  function onHover(d) {{
    link.classed('neighbor', l => l.source.id === d.id || l.target.id === d.id);
    node.classed('neighbor-node', n => neighborSet[d.id].has(n.id))
        .classed('dimmed', n => n.id !== d.id && !neighborSet[d.id].has(n.id));
    d3.select(this).attr('r', 9);

    const rows = (d.neighbors || []).map((name, i) => {{
      const pct = d.scores && d.scores[i] != null
        ? `<span class="tip-score">${{(d.scores[i]*100).toFixed(0)}}%</span>` : '';
      return `<div class="tip-neighbor">${{name}}${{pct}}</div>`;
    }}).join('');

    tooltip.style('display', 'block').html(
      `<div class="tip-title">${{d.id}}</div>` +
      `<div class="tip-label">Similar tracks</div>${{rows}}`
    );
    onMove.call(this, d);
  }}

  function onMove() {{
    const [mx, my] = d3.mouse(document.body);
    const tw = 270, th = 170;
    tooltip
      .style('left', (mx + 14 + tw > W ? mx - tw - 14 : mx + 14) + 'px')
      .style('top',  (my + th > H ? H - th - 8 : my + 8) + 'px');
  }}

  function onOut() {{
    link.classed('neighbor', false);
    node.classed('neighbor-node', false).classed('dimmed', false);
    d3.select(this).attr('r', 6);
    tooltip.style('display', 'none');
  }}

  function dragStart(d) {{
    if (!d3.event.active) sim.alphaTarget(0.15).restart();
    d.fx = d.x; d.fy = d.y;
    tooltip.style('display', 'none');
  }}
  function dragged(d)  {{ d.fx = d3.event.x; d.fy = d3.event.y; }}
  function dragEnd(d)  {{
    if (!d3.event.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
  }}
  </script>
</body>
</html>"""

OUT.write_text(html)
print(f'Written → {OUT.resolve()}')
webbrowser.open(OUT.resolve().as_uri())
