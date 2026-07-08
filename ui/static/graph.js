/* ── Incremental d3-v4 force graph ──────────────────────────────────────────
 * Ported from export_viz.py, adapted to grow as songs are placed. Nodes are
 * songs (kind "query" = a song you added, "corpus" = a nearest neighbour pulled
 * in for context); links are cosine-similarity edges. Positions come from the
 * force sim — the exact corpus embedding is intentionally not used.
 */
const AtlasGraph = (() => {
  const PALETTE = [
    '#7b5ea7','#4e9a8a','#c46b3a','#4a7bbf','#a0516a',
    '#6b9e3c','#9b6b3c','#3c7b9e','#9e3c6b','#5ea74a',
  ];
  const clusterColor = c =>
    (c == null || c === -1) ? '#555577' : PALETTE[c % PALETTE.length];

  let svg, g, gLink, gNode, sim, zoom;
  let nodes = [], links = [];
  const byId = new Map();
  let linkSel, nodeSel, tooltip;

  function init(containerSel) {
    svg = d3.select(containerSel);
    svg.selectAll('*').remove();
    g      = svg.append('g');
    gLink  = g.append('g').attr('class', 'links');
    gNode  = g.append('g').attr('class', 'nodes');
    tooltip = d3.select('#graph-tooltip');

    zoom = d3.zoom().scaleExtent([0.1, 8])
      .on('zoom', () => g.attr('transform', d3.event.transform));
    svg.call(zoom);

    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    sim = d3.forceSimulation(nodes)
      .force('link',   d3.forceLink(links).id(d => d.id)
                          // closer edge = more-similar songs sit tighter
                          .distance(d => 30 + 60 * (1 - (d.value == null ? 0.5 : d.value)))
                          .strength(d => d.kind === 'qq' ? 0.35 : 0.15))
      .force('charge', d3.forceManyBody().strength(-90))
      .force('collide', d3.forceCollide(14))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .alphaDecay(0.035)
      .on('tick', ticked);

    linkSel = gLink.selectAll('line');
    nodeSel = gNode.selectAll('g');
    restart();
  }

  function ticked() {
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
  }

  function restart() {
    // ── links ──
    linkSel = gLink.selectAll('line').data(links, d => `${idOf(d.source)}->${idOf(d.target)}`);
    linkSel.exit().remove();
    linkSel = linkSel.enter().append('line')
      .attr('class', d => 'glink' + (d.kind === 'qq' ? ' glink-qq' : '')).merge(linkSel);

    // ── nodes (a <g> per node: circle + label) ──
    nodeSel = gNode.selectAll('g.gnode').data(nodes, d => d.id);
    nodeSel.exit().remove();

    const enter = nodeSel.enter().append('g')
      .attr('class', 'gnode')
      .call(d3.drag().on('start', dragStart).on('drag', dragged).on('end', dragEnd))
      .on('mouseover', onHover).on('mousemove', onMove).on('mouseout', onOut);

    enter.append('circle');
    enter.append('text')
      .attr('class', 'glabel')
      .attr('x', 10).attr('dy', '0.32em');

    nodeSel = enter.merge(nodeSel);

    nodeSel.select('circle')
      .attr('r', d => d.kind === 'query' ? 9 : 5)
      .attr('fill', d => clusterColor(d.cluster))
      .attr('class', d => d.kind === 'query' ? 'query-node' : 'corpus-node');

    nodeSel.select('text.glabel')
      .text(d => d.kind === 'query' ? d.name : '')
      .attr('class', d => 'glabel ' + (d.kind === 'query' ? 'glabel-query' : ''));

    sim.nodes(nodes);
    sim.force('link').links(links);
    sim.alpha(0.7).restart();
  }

  const idOf = e => (typeof e === 'object' ? e.id : e);

  function mergeFragment(frag) {
    if (!frag) return;
    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    (frag.nodes || []).forEach(n => {
      if (byId.has(n.id)) {                       // promote existing corpus → query
        Object.assign(byId.get(n.id), n);
        return;
      }
      // seed near centre so it animates outward instead of flying from (0,0)
      n.x = W / 2 + (Math.random() - 0.5) * 80;
      n.y = H / 2 + (Math.random() - 0.5) * 80;
      byId.set(n.id, n);
      nodes.push(n);
    });
    (frag.links || []).forEach(l => {
      if (byId.has(l.source) && byId.has(l.target)) links.push(l);
    });
    restart();
    // gently recentre the view on the newest query node
    const q = (frag.nodes || []).find(n => n.kind === 'query');
    if (q) setTimeout(() => zoomTo(q.id), 700);
  }

  function zoomTo(id) {
    const n = byId.get(id);
    if (!n || n.x == null) return;
    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    const k = 1.4;
    svg.transition().duration(600).call(
      zoom.transform,
      d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-n.x, -n.y)
    );
  }

  async function load() {
    try {
      const data = await fetch('/api/graph').then(r => r.json());
      nodes = (data.nodes || []).map(n => ({ ...n }));
      links = (data.links || []);
      nodes.forEach(n => byId.set(n.id, n));
    } catch (_) { nodes = []; links = []; }
  }

  /* ── hover / drag ── */
  function neighborIds(d) {
    const s = new Set([d.id]);
    links.forEach(l => {
      if (idOf(l.source) === d.id) s.add(idOf(l.target));
      if (idOf(l.target) === d.id) s.add(idOf(l.source));
    });
    return s;
  }

  function onHover(d) {
    const near = neighborIds(d);
    linkSel.classed('neighbor', l => idOf(l.source) === d.id || idOf(l.target) === d.id);
    nodeSel.classed('dimmed', n => !near.has(n.id));
    const conf = d.confidence != null
      ? `<div class="tip-label">cluster ${d.cluster} · conf ${Math.round(d.confidence * 100)}%</div>` : '';
    tooltip.style('display', 'block').html(
      `<div class="tip-title">${esc(d.name)}</div>` +
      `<div class="tip-artist">${esc(d.artist || '')}</div>` + conf
    );
    onMove.call(this, d);
  }
  function onMove() {
    const [mx, my] = d3.mouse(document.body);
    tooltip.style('left', (mx + 14) + 'px').style('top', (my + 14) + 'px');
  }
  function onOut() {
    linkSel.classed('neighbor', false);
    nodeSel.classed('dimmed', false);
    tooltip.style('display', 'none');
  }
  function dragStart(d) { if (!d3.event.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; }
  function dragged(d)   { d.fx = d3.event.x; d.fy = d3.event.y; }
  function dragEnd(d)   { if (!d3.event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }

  function esc(s) {
    return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                          .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  return {
    async init(sel) { await load(); init(sel); },
    mergeFragment,
    zoomTo,
    hasNode: id => byId.has(id),
  };
})();
