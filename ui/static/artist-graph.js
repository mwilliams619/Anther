/* Isolated incremental force graph for explicitly-added artists. */
const ArtistGraph = (() => {
  // Touch input may synthesize mouse events. Artist hover labels are a
  // desktop affordance and should not compete with the mobile detail sheet.
  const hoverEnabled = () => !window.matchMedia('(max-width: 768px)').matches
    && !window.matchMedia('(pointer: coarse)').matches;
  let svg, g, gLink, gNode, sim, zoom;
  let nodes = [], links = [];
  const byId = new Map();
  let linkSel, nodeSel, tooltip;
  let selectedId = null;
  let selectCb = null, deselectCb = null;

  const idOf = value => typeof value === 'object' ? value.id : value;
  const escArtist = value => String(value || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function dimensions() {
    const element = svg && svg.node();
    const parent = element && element.parentNode;
    return {
      width: (element && element.clientWidth) || (parent && parent.clientWidth) || 800,
      height: (element && element.clientHeight) || (parent && parent.clientHeight) || 600,
    };
  }

  function init(containerSel) {
    svg = d3.select(containerSel);
    svg.selectAll('*').remove();
    g = svg.append('g');
    gLink = g.append('g').attr('class', 'artist-links');
    gNode = g.append('g').attr('class', 'artist-nodes');
    tooltip = d3.select('#graph-tooltip');

    zoom = d3.zoom().scaleExtent([0.2, 6])
      .on('zoom', () => g.attr('transform', d3.event.transform));
    svg.call(zoom).on('click', () => {
      if (!d3.event.defaultPrevented) clearSelection();
    });

    const {width: W, height: H} = dimensions();
    sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id)
        .distance(d => 45 + 75 * (1 - (d.score == null ? 70 : d.score) / 100))
        .strength(0.35))
      .force('charge', d3.forceManyBody().strength(-230).distanceMax(500))
      .force('collide', d3.forceCollide(34))
      .force('x', d3.forceX(W / 2).strength(0.035))
      .force('y', d3.forceY(H / 2).strength(0.035))
      .alphaDecay(0.04)
      .on('tick', ticked);
    restart();
  }

  function ticked() {
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
  }

  function restart() {
    linkSel = gLink.selectAll('line.artist-link')
      .data(links, d => `${idOf(d.source)}->${idOf(d.target)}`);
    linkSel.exit().remove();
    linkSel = linkSel.enter().append('line').attr('class', 'artist-link')
      .on('mouseover', d => {
        if (!hoverEnabled()) return;
        tooltip.style('display', 'block').html(
          `<div class="tip-title">Artist similarity: ${Math.round(d.score || 0)}</div>`);
      })
      .on('mousemove', moveTooltip).on('mouseout', hideTooltip)
      .merge(linkSel);

    nodeSel = gNode.selectAll('g.artist-gnode').data(nodes, d => d.id);
    nodeSel.exit().remove();
    const entered = nodeSel.enter().append('g').attr('class', 'artist-gnode')
      .call(d3.drag().on('start', dragStart).on('drag', dragged).on('end', dragEnd))
      .on('click', d => {
        if (d3.event.defaultPrevented) return;
        d3.event.stopPropagation();
        select(d.id);
      })
      .on('mouseover', d => {
        highlight(d.id);
        tooltip.style('display', 'block').html(
          `<div class="tip-title">${escArtist(d.name)}</div>` +
          `<div class="tip-artist">${d.track_count || 0} track${d.track_count === 1 ? '' : 's'}</div>`);
      })
      .on('mousemove', moveTooltip)
      .on('mouseout', () => { if (!selectedId) clearHighlight(); hideTooltip(); });
    entered.append('circle').attr('r', 10);
    entered.append('text').attr('class', 'artist-label').attr('x', 14).attr('dy', '0.32em');
    nodeSel = entered.merge(nodeSel);
    nodeSel.select('circle').attr('class', d =>
      `artist-node artist-cluster-${d.cluster_id}${d.source === 'session' ? ' session-artist' : ''}`
      + (d.low_confidence ? ' low-confidence' : ''));
    nodeSel.select('text').text(d => d.name);
    nodeSel.classed('selected', d => d.id === selectedId);

    sim.nodes(nodes);
    sim.force('link').links(links);
    sim.alpha(0.75).restart();
    showEmptyState();
  }

  function moveTooltip() {
    if (!hoverEnabled()) return;
    const point = d3.mouse(document.body);
    tooltip.style('left', `${point[0] + 14}px`).style('top', `${point[1] + 14}px`);
  }
  function hideTooltip() { tooltip.style('display', 'none'); }

  function neighborsOf(id) {
    const out = new Set([id]);
    links.forEach(link => {
      const s = idOf(link.source), t = idOf(link.target);
      if (s === id) out.add(t);
      if (t === id) out.add(s);
    });
    return out;
  }
  function highlight(id) {
    const near = neighborsOf(id);
    nodeSel.classed('dimmed', d => !near.has(d.id));
    linkSel.classed('neighbor', d => idOf(d.source) === id || idOf(d.target) === id)
      .classed('dimmed', d => idOf(d.source) !== id && idOf(d.target) !== id);
  }
  function clearHighlight() {
    nodeSel.classed('dimmed', false);
    linkSel.classed('neighbor', false).classed('dimmed', false);
  }

  function select(id, opts) {
    if (!byId.has(id)) return;
    selectedId = id;
    nodeSel.classed('selected', d => d.id === id);
    highlight(id);
    if (!opts || opts.zoom !== false) zoomTo(id);
    if (selectCb) selectCb(id);
  }
  function clearSelection() {
    if (!selectedId) return;
    selectedId = null;
    nodeSel.classed('selected', false);
    clearHighlight();
    if (deselectCb) deselectCb();
  }

  function zoomTo(id, scale) {
    const node = byId.get(id);
    if (!node || node.x == null) return;
    const {width: W, height: H} = dimensions();
    const k = scale || 1.5;
    svg.transition().duration(500).call(
      zoom.transform,
      d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-node.x, -node.y)
    );
  }

  function dragStart(d) {
    if (!d3.event.active) sim.alphaTarget(0.2).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(d) { d.fx = d3.event.x; d.fy = d3.event.y; }
  function dragEnd(d) {
    if (!d3.event.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  function replaceData(data) {
    const {width: W, height: H} = dimensions();
    byId.clear();
    nodes = (data.nodes || []).map(node => ({ ...node,
      x: W / 2 + (Math.random() - 0.5) * 120,
      y: H / 2 + (Math.random() - 0.5) * 120,
    }));
    nodes.forEach(node => byId.set(node.id, node));
    links = (data.links || []).filter(link => byId.has(idOf(link.source)) && byId.has(idOf(link.target)));
    if (selectedId && !byId.has(selectedId)) selectedId = null;
    restart();
  }

  async function load() {
    const response = await fetch('/api/artist/graph');
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Artist graph unavailable');
    replaceData(data);
    return data;
  }

  function mergeFragment(fragment) {
    if (!fragment) return;
    if (fragment.replace) { load().catch(() => {}); return; }
    const {width: W, height: H} = dimensions();
    (fragment.nodes || []).forEach(node => {
      if (byId.has(node.id)) Object.assign(byId.get(node.id), node);
      else {
        node.x = W / 2 + (Math.random() - 0.5) * 80;
        node.y = H / 2 + (Math.random() - 0.5) * 80;
        byId.set(node.id, node); nodes.push(node);
      }
    });
    (fragment.links || []).forEach(link => {
      // only keep edges whose endpoints are both on the map — a dangling edge
      // stays a bare string id in d3.forceLink and crashes the sim ("Cannot
      // create property 'vx' on string …"). replaceData filters the same way.
      if (byId.has(idOf(link.source)) && byId.has(idOf(link.target))) links.push(link);
    });
    restart();
    if (fragment.id) setTimeout(() => zoomTo(fragment.id), 450);
  }

  function removeNodes(ids) {
    const removed = new Set(ids || []);
    nodes = nodes.filter(node => !removed.has(node.id));
    links = links.filter(link => !removed.has(idOf(link.source)) && !removed.has(idOf(link.target)));
    removed.forEach(id => byId.delete(id));
    if (selectedId && removed.has(selectedId)) clearSelection();
    restart();
  }

  function reset() {
    nodes = []; links = []; byId.clear(); selectedId = null;
    restart();
  }

  function showEmptyState() {
    const empty = document.getElementById('artist-graph-empty');
    if (empty) empty.hidden = nodes.length > 0;
  }

  function activate() {
    const {width: W, height: H} = dimensions();
    sim.force('x', d3.forceX(W / 2).strength(0.035));
    sim.force('y', d3.forceY(H / 2).strength(0.035));
    sim.alpha(0.25).restart();
  }

  return {
    async init(sel) { init(sel); await load(); },
    load, replaceData, mergeFragment, removeNodes, reset, zoomTo, activate,
    hasNode: id => byId.has(id),
    getNodes: () => nodes.slice(),
    getSelectedId: () => selectedId,
    selectNode: select,
    clearSelection,
    onSelect: cb => { selectCb = cb; },
    onDeselect: cb => { deselectCb = cb; },
  };
})();
