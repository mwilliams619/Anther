/* ── Incremental d3-v4 force graph ──────────────────────────────────────────
 * Ported from export_viz.py, adapted to grow as songs are placed. Nodes are
 * songs (kind "query" = a song you added, "corpus" = a nearest neighbour pulled
 * in for context); links are cosine-similarity edges. Positions come from the
 * force sim — the exact corpus embedding is intentionally not used.
 */
const AtlasGraph = (() => {
  // Nodes are NOT colored by cluster — the force layout itself shows song
  // relationships; cluster info lives only in the click popover. Fills come
  // from CSS (query = --primary, corpus = grey).

  // Accent ring per playlist/album import: hues chosen to read against the
  // dark node fills. Assigned in first-seen order.
  const PLAYLIST_ACCENTS = [
    '#f2c14e','#5ad1c8','#f27d72','#8fd15a','#d98cf2','#6aa9f2','#f2a25a','#e05a8c',
  ];
  const playlistAccent = new Map();
  const playlistColor = pid => {
    if (!playlistAccent.has(pid)) {
      playlistAccent.set(pid, PLAYLIST_ACCENTS[playlistAccent.size % PLAYLIST_ACCENTS.length]);
    }
    return playlistAccent.get(pid);
  };

  let svg, g, gLink, gNode, sim, zoom;
  let nodes = [], links = [];
  const byId = new Map();
  let linkSel, nodeSel, tooltip;
  let pinnedId = null;                  // clicked node: highlight locked until deselect
  let filterIds = null;                 // active filter: Set of node ids to keep lit
  let groups = {};                      // gid → {name, kind} for imported playlists/albums
  let selectCb = null, deselectCb = null;

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
    svg.on('click', () => {             // background click unpins (pans don't: d3
      if (d3.event.defaultPrevented) return;   // suppresses the click after a drag)
      clearSelection();
    });

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
      .on('mouseover', onHover).on('mousemove', onMove).on('mouseout', onOut)
      .on('click', onClick);

    enter.append('circle');
    enter.append('text')
      .attr('class', 'glabel')
      .attr('x', 10).attr('dy', '0.32em');

    nodeSel = enter.merge(nodeSel);

    nodeSel.select('circle')
      .attr('r', d => d.kind === 'query' ? 9 : 5)
      // import-group badge: accent-colored ring (via CSS var so the
      // pinned-selection ring still overrides it)
      .attr('class', d => (d.kind === 'query' ? 'query-node' : 'corpus-node')
                        + (d.playlist_pid ? ' in-playlist' : ''))
      .style('--pl-accent', d => d.playlist_pid ? playlistColor(d.playlist_pid) : null);

    nodeSel.select('text.glabel')
      .text(d => d.kind === 'query' ? d.name : '')
      .attr('class', d => 'glabel ' + (d.kind === 'query' ? 'glabel-query' : ''));

    sim.nodes(nodes);
    sim.force('link').links(links);
    sim.alpha(0.7).restart();

    // keep the pinned highlight / active filter correct across entered elements
    if (pinnedId !== null) {
      const p = byId.get(pinnedId);
      if (p) {
        applyHighlight(p);
        nodeSel.classed('selected', n => n.id === pinnedId);
      }
    } else if (filterIds !== null) {
      applyFilterClasses();
    }
  }

  const idOf = e => (typeof e === 'object' ? e.id : e);

  function mergeFragment(frag, opts) {
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
    if (opts && opts.focus === false) return;   // streamed merges: no camera jumps
    // gently recentre the view on the focus node (or the newest query node)
    const fid = (opts && opts.focusId)
      || ((frag.nodes || []).find(n => n.kind === 'query') || {}).id;
    if (fid) setTimeout(() => zoomTo(fid, opts && opts.zoomScale), 700);
  }

  function zoomTo(id, scale) {
    const n = byId.get(id);
    if (!n || n.x == null) return;
    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    const k = scale || 1.4;
    svg.transition().duration(600).call(
      zoom.transform,
      d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-n.x, -n.y)
    );
  }

  // Load the saved graph, waiting out the corpus warm-up (~1-2 min after
  // server start): /api/graph reports ready:false until the corpus is loaded,
  // and rendering an "empty" graph during that window looks like a dead UI.
  async function load() {
    const hint = document.getElementById('graph-loading');
    for (;;) {
      try {
        const data = await fetch('/api/graph').then(r => r.json());
        if (data.ready !== false) {
          nodes = (data.nodes || []).map(n => ({ ...n }));
          links = (data.links || []);
          groups = data.groups || {};
          nodes.forEach(n => byId.set(n.id, n));
          break;
        }
      } catch (_) { /* server not up yet — keep retrying */ }
      if (hint) hint.style.display = 'block';
      await new Promise(r => setTimeout(r, 2000));
    }
    if (hint) hint.style.display = 'none';
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

  function applyHighlight(d) {
    const near = neighborIds(d);
    linkSel.classed('neighbor', l => idOf(l.source) === d.id || idOf(l.target) === d.id);
    nodeSel.classed('dimmed', n => !near.has(n.id));
    nodeSel.classed('lit',    n => near.has(n.id));   // un-greys corpus neighbors
  }
  function applyFilterClasses() {
    linkSel.classed('neighbor', false);
    linkSel.classed('filter-out',
      l => !(filterIds.has(idOf(l.source)) && filterIds.has(idOf(l.target))));
    nodeSel.classed('dimmed', n => !filterIds.has(n.id));
    nodeSel.classed('lit',    n => filterIds.has(n.id));
  }
  function clearHighlight() {
    if (filterIds !== null) { applyFilterClasses(); return; }   // filter is the rest state
    linkSel.classed('neighbor', false);
    linkSel.classed('filter-out', false);
    nodeSel.classed('dimmed', false);
    nodeSel.classed('lit', false);
  }

  /* ── filter / bulk management (left-panel Map section) ── */
  function setFilter(ids) {
    filterIds = ids ? new Set(ids) : null;
    if (pinnedId !== null) clearSelection();          // filter becomes the rest state
    if (filterIds === null) {
      linkSel.classed('filter-out', false);
      nodeSel.classed('dimmed', false).classed('lit', false);
    } else {
      applyFilterClasses();
    }
  }

  function removeNodes(ids) {
    const rm = new Set(ids);
    if (pinnedId !== null && rm.has(pinnedId)) clearSelection();
    nodes = nodes.filter(n => !rm.has(n.id));
    links = links.filter(l => !rm.has(idOf(l.source)) && !rm.has(idOf(l.target)));
    rm.forEach(id => { byId.delete(id); if (filterIds) filterIds.delete(id); });
    restart();
  }

  function reset() {
    if (pinnedId !== null) clearSelection();
    nodes = [];
    links = [];
    byId.clear();
    filterIds = null;
    groups = {};
    restart();
  }

  /* ── click / pin selection ── */
  function select(id, opts) {
    const d = byId.get(id);
    if (!d) return;
    pinnedId = id;
    applyHighlight(d);
    nodeSel.classed('selected', n => n.id === id);
    if (opts && opts.zoom) zoomTo(id);
    if (selectCb) selectCb(id);
  }
  function clearSelection() {
    if (pinnedId === null) return;
    pinnedId = null;
    clearHighlight();
    nodeSel.classed('selected', false);
    if (deselectCb) deselectCb();
  }
  function onClick(d) {
    if (d3.event.defaultPrevented) return;   // drag gesture, not a click
    d3.event.stopPropagation();
    select(d.id);
  }

  function onHover(d) {
    if (pinnedId === null) applyHighlight(d);   // hover previews only when unpinned
    const pl = d.playlist_pid
      ? `<div class="tip-label" style="color:${playlistColor(d.playlist_pid)}">● from ${String(d.playlist_pid).startsWith('album:') ? 'album' : 'playlist'}</div>` : '';
    tooltip.style('display', 'block').html(
      `<div class="tip-title">${esc(d.name)}</div>` +
      `<div class="tip-artist">${esc(d.artist || '')}</div>` + pl
    );
    onMove.call(this, d);
  }
  function onMove() {
    const [mx, my] = d3.mouse(document.body);
    tooltip.style('left', (mx + 14) + 'px').style('top', (my + 14) + 'px');
  }
  function onOut() {
    if (pinnedId === null) clearHighlight();
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
    getNodes: () => nodes.slice(),
    getGroups: () => ({ ...groups }),
    registerGroup: (gid, info) => { groups[String(gid)] = info; },
    groupColor: pid => playlistColor(pid),
    selectNode: select,
    clearSelection,
    setFilter,
    removeNodes,
    reset,
    onSelect:   cb => { selectCb = cb; },
    onDeselect: cb => { deselectCb = cb; },
  };
})();
