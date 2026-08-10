/* ── Incremental d3-v4 force graph ──────────────────────────────────────────
 * Ported from export_viz.py, adapted to grow as songs are placed. Nodes are
 * songs (kind "query" = a song you added, "corpus" = a nearest neighbour pulled
 * in for context); links are cosine-similarity edges. Positions come from the
 * force sim — the exact corpus embedding is intentionally not used.
 */
const loaderStart = performance.now();
const AtlasGraph = (() => {
  // Nodes are NOT colored by cluster — the force layout itself shows song
  // relationships; cluster info lives only in the click popover. Fills come
  // from CSS (query = --primary, corpus = grey).

  // Accent ring per playlist/album import: hues chosen to read against the
  // dark node fills. Assigned in first-seen order.
  const PLAYLIST_ACCENTS = [
    '#f2c14e','#5ad1c8','#f27d72','#8fd15a','#d98cf2','#6aa9f2','#f2a25a','#e05a8c',
  ];
  // Inward-pull strength for the forceX/forceY gravity (see init()). Higher
  // bounds the map tighter (favors zooming in to see local detail); lower
  // lets it spread further before settling. 0.03-0.10 is the useful range.
  const GRAVITY_STRENGTH = 0.03;

  // A corpus node is 5px across — fine for a cursor, far under the ~44px a
  // fingertip needs. On touch devices each node gets an invisible box to catch
  // the tap; on a mouse it is zero-sized, so hover targeting stays exactly as
  // tight as it always was.
  const TOUCH_HIT_R = window.matchMedia('(pointer: coarse)').matches ? 18 : 0;

  // Each qq-edge's human-readable similarity score (0-100) is computed once on
  // the backend (atlas.py: _display_score, a fixed calibration against the
  // corpus null-distribution — not an on-map rescale) and persisted on the
  // link as `d.score`. That means a given pair's score never changes just
  // because other songs were added to or removed from the map; it's read
  // directly here, never recomputed client-side.

  const playlistAccent = new Map();
  const playlistColor = pid => {
    if (!playlistAccent.has(pid)) {
      playlistAccent.set(pid, PLAYLIST_ACCENTS[playlistAccent.size % PLAYLIST_ACCENTS.length]);
    }
    return playlistAccent.get(pid);
  };

  let svg, g, gLink, gHoverLink, gNode, sim, zoom;
  let nodes = [], links = [];
  const byId = new Map();
  let adjacency = new Map();            // id → Set(neighbor ids), rebuilt in restart()
  let linkSel, hitLinkSel, hoverLinkSel, nodeSel, tooltip;
  let hoverLinks = [];                 // transient corpus-result → seed links
  let pinnedId = null;                  // clicked node: highlight locked until deselect
  let filterIds = null;                 // active filter: Set of node ids to keep lit
  let groups = {};                      // gid → {name, kind} for imported playlists/albums
  let selectCb = null, deselectCb = null;
  // ── graph autoplay (see ui/static/autoplay.js) ──
  let playingId = null;                 // node whose track is sounding right now
  let playedIds = new Set();            // tour history, rendered as muted nodes
  let panCb = null;                     // fires on a USER pan/zoom, not zoomTo()
  let shiftSelectCb = null;             // shift-click: steer a running tour

  // ── ambient idle motion ──────────────────────────────────────────────────
  // After IDLE_MS of no pointer activity, the node layer gets .idle-breathing
  // (see the @keyframes gnode-breathe rule in style.css) — a tiny, staggered
  // scale pulse meant to be almost subliminal. Any interaction cancels it and
  // restarts the timer. Deliberately just a CSS transform animation, not a
  // second physics/animation loop, so it costs nothing beyond what the
  // browser already spends compositing a CSS transform.
  const IDLE_MS = 10000;
  let idleTimer = null;
  function resetIdle() {
    if (idleTimer) clearTimeout(idleTimer);
    if (gNode) gNode.classed('idle-breathing', false);
    idleTimer = setTimeout(() => { if (gNode) gNode.classed('idle-breathing', true); }, IDLE_MS);
  }

  function hideLoader() {
  const loader = document.getElementById("loader");
  if (!loader) return;

  const elapsed = performance.now() - loaderStart;
  const minDuration = 1000; // 1 seconds

  setTimeout(() => {
    loader.classList.add("hidden");

    loader.addEventListener("transitionend", () => {
      loader.remove();
    }, { once: true });

  }, Math.max(0, minDuration - elapsed));
}

  function init(containerSel) {
    svg = d3.select(containerSel);
    svg.selectAll('*').remove();

    // static film-grain texture (see .grain-rect in style.css): one SVG
    // filter, rendered once and never re-evaluated per frame — sits behind
    // the pan/zoom group so it doesn't move with the graph.
    const defs = svg.append('defs');
    const grain = defs.append('filter').attr('id', 'grain-filter');
    grain.append('feTurbulence')
      .attr('type', 'fractalNoise').attr('baseFrequency', 0.85)
      .attr('numOctaves', 2).attr('stitchTiles', 'stitch').attr('result', 'noise');
    grain.append('feColorMatrix')
      .attr('in', 'noise').attr('type', 'matrix')
      .attr('values', '0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.05 0');
    svg.append('rect').attr('class', 'grain-rect')
      .attr('width', '100%').attr('height', '100%')
      .attr('filter', 'url(#grain-filter)');

    g      = svg.append('g');
    gLink  = g.append('g').attr('class', 'links');
    // Hover links are not persisted graph structure: they only explain which
    // placed songs a muted similarity/result node is being shown in relation to.
    gHoverLink = g.append('g').attr('class', 'hover-links');
    gNode  = g.append('g').attr('class', 'nodes');
    tooltip = d3.select('#graph-tooltip');

    zoom = d3.zoom().scaleExtent([0.1, 8])
      .on('zoom', () => g.attr('transform', d3.event.transform))
      // A user grabbing the map should stop autoplay's camera from yanking it
      // back. zoomTo() drives the same zoom behaviour through a programmatic
      // transition, which carries no sourceEvent — that's what separates
      // "the user panned" from "we panned".
      .on('start', () => { if (d3.event.sourceEvent && panCb) panCb(); });
    svg.call(zoom);
    svg.on('click', () => {             // background click unpins (pans don't: d3
      if (d3.event.defaultPrevented) return;   // suppresses the click after a drag)
      clearSelection();
    });
    // any pointer activity defers the idle-breathing pulse
    svg.on('mousemove.idle', resetIdle).on('mousedown.idle', resetIdle).on('wheel.idle', resetIdle);
    resetIdle();

    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    sim = d3.forceSimulation(nodes)
      .force('link',   d3.forceLink(links).id(d => d.id)
                          // closer edge = more-similar songs sit tighter. Uses the
                          // backend-computed 0-100 score (d.score), not the raw
                          // cosine (d.value), so the layout actually spreads by
                          // similarity instead of bunching within a fraction of a
                          // pixel. Falls back to a neutral mid-range distance for
                          // any edge that somehow lacks a score (pre-migration data).
                          .distance(d => 30 + 60 * (1 - (d.score == null ? 70 : d.score) / 100))
                          .strength(d => d.kind === 'qq' ? 0.35 : 0.15))
      // capped-range repulsion: a far-flung dissimilar node no longer shoves
      // the rest of the graph outward when it arrives (distanceMax bounds it)
      .force('charge', d3.forceManyBody().strength(-90).distanceMax(350))
      .force('collide', d3.forceCollide(14))
      // GRAVITY, not re-centering: forceCenter only recenters the (growing)
      // centroid and applies zero inward pull, so an outlier node with weak
      // links has nothing holding it in — it drifts out and the whole view
      // has to zoom out to keep it in frame. forceX/forceY are springs: every
      // node is pulled toward the middle with force proportional to distance,
      // so the cloud settles at a bounded radius instead of growing without
      // limit. GRAVITY_STRENGTH is the one dial to tune: higher = tighter map
      // (zoom in for detail); lower = looser spread. 0.03-0.10 is the useful range.
      .force('x', d3.forceX(W / 2).strength(GRAVITY_STRENGTH))
      .force('y', d3.forceY(H / 2).strength(GRAVITY_STRENGTH))
      .alphaDecay(0.035)
      .on('tick', ticked);

    linkSel = gLink.selectAll('line');
    nodeSel = gNode.selectAll('g');
    restart();
    hideLoader();
  }

  function ticked() {
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    hitLinkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
              .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    hoverLinkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
  }

  function restart() {
    // ── links ──
    linkSel = gLink.selectAll('line.glink').data(links, d => `${idOf(d.source)}->${idOf(d.target)}`);
    linkSel.exit().remove();
    linkSel = linkSel.enter().append('line')
      .attr('class', d => 'glink' + (d.kind === 'qq' ? ' glink-qq' : ''))
      .merge(linkSel);

    // wider, invisible line under each visible one — the visible stroke is
    // only 1-1.5px, too thin to reliably hover, so the score tooltip listens
    // on this fatter transparent twin instead
    hitLinkSel = gLink.selectAll('line.glink-hit').data(links, d => `${idOf(d.source)}->${idOf(d.target)}`);
    hitLinkSel.exit().remove();
    hitLinkSel = hitLinkSel.enter().append('line')
      .attr('class', 'glink-hit')
      .on('mouseover', onLinkHover).on('mousemove', onMove)
      .on('mouseout', () => tooltip.style('display', 'none'))
      .merge(hitLinkSel);

    hoverLinkSel = gHoverLink.selectAll('line.hover-similarity-link')
      .data(hoverLinks, d => `${d.source.id}->${d.target.id}`);
    hoverLinkSel.exit().remove();
    hoverLinkSel = hoverLinkSel.enter().append('line')
      .attr('class', 'hover-similarity-link')
      .merge(hoverLinkSel)
      .attr('class', d => 'hover-similarity-link'
        + (d.source.recommended === true ? ' recommendation-link' : ''));

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
    // A <rect>, not a <circle>: the blanket `.gnode circle` rules in style.css
    // (selected ring, breathing, autoplay pulse) would otherwise paint this
    // invisible target as a giant one. Appended last so it sits over the label.
    enter.append('rect')
      .attr('class', 'hit-target')
      .attr('x', -TOUCH_HIT_R).attr('y', -TOUCH_HIT_R)
      .attr('width', TOUCH_HIT_R * 2).attr('height', TOUCH_HIT_R * 2)
      .attr('fill', 'transparent');

    nodeSel = enter.merge(nodeSel);

    nodeSel.select('circle')
      .attr('r', d => d.kind === 'query' ? 9 : 5)
      // import-group badge: accent-colored ring (via CSS var so the
      // pinned-selection ring still overrides it)
      .attr('class', d => (d.kind === 'query' ? 'query-node' : 'corpus-node')
                        + (d.playlist_pid ? ' in-playlist' : ''))
      .style('--pl-accent', d => d.playlist_pid ? playlistColor(d.playlist_pid) : null)
      // stagger the idle-breathing pulse (see .idle-breathing in style.css)
      // so the whole graph doesn't visibly pulse in lockstep — assigned
      // once per node id, stable across re-renders, purely decorative.
      .style('--breathe-delay', d => breatheDelay(d.id) + 's');

    nodeSel.select('text.glabel')
      .text(d => d.name || '')
      .attr('class', d => 'glabel ' + (d.kind === 'query' ? 'glabel-query' : 'glabel-corpus'));

    sim.nodes(nodes);
    sim.force('link').links(links);
    sim.alpha(0.7).restart();

    rebuildAdjacency();

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
    applyPlaybackClasses();   // .playing/.played must survive a re-render too
  }

  // ── graph autoplay rendering ──
  // Applied here (not only at call time) because a playlist streaming in mid-
  // tour re-enters nodes, and a freshly-entered <g> would otherwise lose the
  // playing/played styling of a tour already in progress.
  function applyPlaybackClasses() {
    if (!nodeSel) return;
    nodeSel.classed('playing', n => n.id === playingId)
           .classed('played',  n => playedIds.has(n.id) && n.id !== playingId);
  }

  function setPlayingNode(id) {
    playingId = id;
    if (id !== null && id !== undefined) playedIds.add(id);
    applyPlaybackClasses();
    // raise so the pulsing node isn't buried in a dense cluster
    if (nodeSel && id) nodeSel.filter(n => n.id === id).raise();
  }

  function setPlayed(ids) {
    playedIds = new Set(ids || []);
    applyPlaybackClasses();
  }

  function clearPlayback() {
    playingId = null;
    playedIds = new Set();
    applyPlaybackClasses();
  }

  // deterministic per-node stagger for the breathing keyframe: hashes the
  // id into [0, 4.5) so it lines up with the 4.5s animation duration and
  // stays stable across restart() calls (no re-randomizing on every render).
  function breatheDelay(id) {
    let h = 0;
    const s = String(id);
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return (h % 450) / 100;
  }

  // adjacency map used by applyHighlight() for graduated (1-hop / 2-hop)
  // hop-distance dimming. Rebuilt whenever the link set changes (restart()),
  // not per animation frame or per hover — a couple hundred edges is trivial.
  function rebuildAdjacency() {
    adjacency = new Map();
    nodes.forEach(n => adjacency.set(n.id, new Set()));
    links.forEach(l => {
      const s = idOf(l.source), t = idOf(l.target);
      if (adjacency.has(s)) adjacency.get(s).add(t);
      if (adjacency.has(t)) adjacency.get(t).add(s);
    });
  }

  const idOf = e => (typeof e === 'object' ? e.id : e);

  function setHoverSimilarityLinks(d) {
    // Corpus nodes are recommendations/context. Their permanent graph links
    // are intentionally omitted to keep the force map sparse; reveal their
    // relationship to every placed seed only while the result is hovered.
    hoverLinks = d.kind === 'query'
      ? []
      : nodes.filter(n => n.kind === 'query' && n.id !== d.id)
        .map(n => ({ source: d, target: n }));
    const relatedIds = new Set(hoverLinks.map(l => l.target.id));
    // A recommendation is related to the placed seed set as a whole. Keep
    // those seed songs visible (rather than letting generic hover dimming hide
    // them) so the temporary spokes identify their endpoints.
    nodeSel.classed('hover-related-query', n => relatedIds.has(n.id))
      .classed('hover-related-recommendation',
        n => d.recommended === true && relatedIds.has(n.id));
    restartHoverLinks();
  }

  function restartHoverLinks() {
    hoverLinkSel = gHoverLink.selectAll('line.hover-similarity-link')
      .data(hoverLinks, d => `${d.source.id}->${d.target.id}`);
    hoverLinkSel.exit().remove();
    hoverLinkSel = hoverLinkSel.enter().append('line')
      .merge(hoverLinkSel)
      .attr('class', d => 'hover-similarity-link'
        + (d.source.recommended === true ? ' recommendation-link' : ''));
    ticked();
  }

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
          const W = svg.node().clientWidth;
          const H = svg.node().clientHeight;

          nodes = (data.nodes || []).map(n => ({
            ...n,
            x: W / 2 + (Math.random() - 0.5) * 100,
            y: H / 2 + (Math.random() - 0.5) * 100
          }));
          groups = data.groups || {};
          nodes.forEach(n => byId.set(n.id, n));
          // Drop any dangling link (endpoint not on the map). A stale edge whose
          // source/target node is missing stays a bare string id through
          // d3.forceLink — the sim then does `str.vx += …` every tick, throwing
          // "Cannot create property 'vx' on string …" thousands of times and
          // freezing the graph. mergeFragment already filters the same way.
          links = (data.links || []).filter(l => byId.has(idOf(l.source)) && byId.has(idOf(l.target)));
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
    return adjacency.has(d.id) ? new Set(adjacency.get(d.id)) : new Set();
  }

  // graduated depth relative to the selected/hovered node d: d itself and its
  // 1-hop neighbors stay at full brightness ("near"); 2-hop neighbors get
  // .dimmed-mid (partial fade — see style.css); everything further recedes
  // to the existing .dimmed rest state. Edges follow the same graduated
  // dimming: 1-hop edges (.neighbor, full bright), 2-hop edges (.neighbor-mid,
  // partial fade), everything else (default, near-invisible).
  // Computed once here via a 2-level BFS over the prebuilt adjacency map,
  // not per animation frame.
  function applyHighlight(d) {
    const near = neighborIds(d);
    near.add(d.id);
    const mid = new Set();
    near.forEach(id => {
      (adjacency.get(id) || []).forEach(n2 => { if (!near.has(n2)) mid.add(n2); });
    });

    // 1-hop edges: direct connections from d
    // 2-hop edges: connecting two nodes within the near+mid set, but not 1-hop
    const twoHopNodes = new Set([...near, ...mid]);
    linkSel.classed('neighbor', l => {
      const s = idOf(l.source), t = idOf(l.target);
      return s === d.id || t === d.id;
    });
    linkSel.classed('neighbor-mid', l => {
      const s = idOf(l.source), t = idOf(l.target);
      return twoHopNodes.has(s) && twoHopNodes.has(t) && s !== d.id && t !== d.id;
    });

    nodeSel.classed('dimmed-mid', n => mid.has(n.id));
    nodeSel.classed('dimmed', n => !near.has(n.id) && !mid.has(n.id));
    nodeSel.classed('lit',    n => near.has(n.id));   // un-greys corpus neighbors
  }
  function applyFilterClasses() {
    linkSel.classed('neighbor', false);
    linkSel.classed('neighbor-mid', false);
    linkSel.classed('filter-out',
      l => !(filterIds.has(idOf(l.source)) && filterIds.has(idOf(l.target))));
    nodeSel.classed('dimmed-mid', false);
    nodeSel.classed('dimmed', n => !filterIds.has(n.id));
    nodeSel.classed('lit',    n => filterIds.has(n.id));
  }
  function clearHighlight() {
    if (filterIds !== null) { applyFilterClasses(); return; }   // filter is the rest state
    linkSel.classed('neighbor', false);
    linkSel.classed('neighbor-mid', false);
    linkSel.classed('filter-out', false);
    nodeSel.classed('dimmed-mid', false);
    nodeSel.classed('dimmed', false);
    nodeSel.classed('lit', false);
    nodeSel.classed('selected-neighbor', false);
  }

  /* ── filter / bulk management (left-panel Map section) ── */
  function setFilter(ids) {
    filterIds = ids ? new Set(ids) : null;
    if (pinnedId !== null) clearSelection();          // filter becomes the rest state
    if (filterIds === null) {
      linkSel.classed('filter-out', false);
      nodeSel.classed('dimmed', false).classed('dimmed-mid', false).classed('lit', false);
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
    playingId = null;
    playedIds = new Set();
    restart();
  }

  /* ── click / pin selection ── */
  function select(id, opts) {
    const d = byId.get(id);
    if (!d) return;
    pinnedId = id;
    applyHighlight(d);
    nodeSel.classed('selected', n => n.id === id);
    
    // dim the selected node's 1-hop neighbors to 0.8 so they don't crowd
    // the selected node (which is growing and being pushed away). The
    // neighbors themselves will be pushed away via increased force repulsion.
    const neighbors = neighborIds(d);
    nodeSel.classed('selected-neighbor', n => neighbors.has(n.id));
    // also gray out their labels so they don't compete
    nodeSel.select('text').classed('selected-neighbor-label', n => neighbors.has(n.id));
    
    // raise the entire selected node group (circle + label) so it's on top
    // in dense clusters. The group stays raised until deselected.
    const selected = nodeSel.filter(n => n.id === id);
    selected.raise();
    // add subtle drop shadow to the selected label
    selected.select('text').classed('highlighted-label', true);
    
    // increase charge (repulsion) so neighbors are pushed away from the
    // selected node, giving it breathing room. Restore to normal when deselected.
    sim.force('charge').strength(-250);
    sim.alpha(0.3).restart();
    
    if (opts && opts.zoom) zoomTo(id);
    if (selectCb) selectCb(id);
  }
  function clearSelection() {
    if (pinnedId === null) return;
    // lower the entire node group before clearing the selection id, so we can find it
    const deselected = nodeSel.filter(n => n.id === pinnedId);
    deselected.lower();
    // remove shadow and gray label classes
    deselected.select('text').classed('highlighted-label', false);
    pinnedId = null;
    clearHighlight();
    nodeSel.classed('selected', false);
    nodeSel.classed('selected-neighbor', false);
    nodeSel.select('text').classed('selected-neighbor-label', false);
    
    // restore normal repulsion strength when deselecting
    sim.force('charge').strength(-90);
    sim.alpha(0.3).restart();
    
    if (deselectCb) deselectCb();
  }
  function onClick(d) {
    if (d3.event.defaultPrevented) return;   // drag gesture, not a click
    d3.event.stopPropagation();
    // Shift-click steers a running autoplay tour here (see app.js). It still
    // selects, so the detail panel follows along and the gesture reads as a
    // richer click rather than a different one.
    select(d.id);
    if (d3.event.shiftKey && shiftSelectCb) shiftSelectCb(d.id);
  }

  function onHover(d) {
    if (pinnedId === null) applyHighlight(d);   // hover previews only when unpinned
    setHoverSimilarityLinks(d);
    // raise the entire node group (circle + label) so they're on top of all
    // other nodes and labels in the graph. The group contains both elements so
    // they come to the front together.
    const hovered = nodeSel.filter(n => n.id === d.id);
    hovered.raise();
    // add subtle drop shadow to the hovered label
    hovered.select('text').classed('highlighted-label', true);
    const pl = d.playlist_pid
      ? `<div class="tip-label" style="color:${playlistColor(d.playlist_pid)}">● from ${String(d.playlist_pid).startsWith('album:') ? 'album' : 'playlist'}</div>` : '';
    tooltip.style('display', 'block').html(
      `<div class="tip-title">${esc(d.name)}</div>` +
      `<div class="tip-artist">${esc(d.artist || '')}</div>` + pl
    );
    onMove.call(this, d);
  }

  function onLinkHover(d) {
    if (d.score == null) return;
    tooltip.style('display', 'block').html(
      `<div class="tip-title">Similarity score: ${Math.round(d.score)}</div>`
    );
    onMove.call(this, d);
  }
  function onMove() {
    const [mx, my] = d3.mouse(document.body);
    tooltip.style('left', (mx + 14) + 'px').style('top', (my + 14) + 'px');
  }
  function onOut() {
    if (pinnedId === null) clearHighlight();
    hoverLinks = [];
    nodeSel.classed('hover-related-query', false)
      .classed('hover-related-recommendation', false);
    restartHoverLinks();
    // lower the entire node group back to its natural position if the node is not selected
    // (selected nodes keep their group raised until deselected)
    const d = d3.select(this).datum();
    if (d && d.id !== pinnedId) {
      const hovered = nodeSel.filter(n => n.id === d.id);
      hovered.lower();
      // remove shadow from label
      hovered.select('text').classed('highlighted-label', false);
    }
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
    async init(sel) {
    init(sel);      // create svg + simulation first
    await load();   // then fetch graph
    restart();      // render loaded nodes
  },
    mergeFragment,
    zoomTo,
    hasNode: id => byId.has(id),
    getNodes: () => nodes.slice(),
    getLinks: () => links.slice(),
    neighborsOf: id => Array.from(adjacency.get(id) || []),
    setPlayingNode,
    setPlayed,
    clearPlayback,
    onManualPan: cb => { panCb = cb; },
    onShiftSelect: cb => { shiftSelectCb = cb; },
    getGroups: () => ({ ...groups }),
    registerGroup: (gid, info) => { groups[String(gid)] = info; },
    groupColor: pid => playlistColor(pid),
    selectNode: select,
    getSelectedId: () => pinnedId,
    clearSelection,
    setFilter,
    removeNodes,
    reset,
    onSelect:   cb => { selectCb = cb; },
    onDeselect: cb => { deselectCb = cb; },
  };
})();
