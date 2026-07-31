/* Graph autoplay — traversal planner (pure, no DOM, no network).
 *
 * DFS-with-backtracking over the QUERY-node subgraph: play a song, walk the
 * highest-scoring edge to an unplayed neighbor, and when a branch dead-ends,
 * walk back up the spine to the most recent node that still has one. When the
 * whole spine is exhausted the reachable component is done, and the planner
 * asks its caller to resolve a jump to the nearest unplayed island.
 *
 * Kept deliberately free of audio and DOM so the part most likely to have
 * subtle bugs is unit-testable: see tests/js/autoplay-traversal.test.js.
 * Jumps are the one thing the planner cannot decide alone — the client only
 * knows edges that cleared the link threshold, so it cannot rank non-adjacent
 * pairs — hence next() returns a 'need-jump' request and the caller supplies
 * the answer via resumeWithJump().
 *
 * Loaded as a plain browser script (window.AutoplayTraversal) and as a node
 * module under test; there is no build step in this repo.
 */
(function (root) {
  'use strict';

  // d3's force simulation rewrites link.source/target from ids to node objects
  // once the graph is live, so every read has to tolerate both shapes.
  function idOf(end) {
    return (end && typeof end === 'object') ? end.id : end;
  }

  /* getGraph: () => {nodes, links} — called fresh on every hop rather than
   * snapshotted, so songs streamed in mid-tour by a playlist import are picked
   * up automatically and removed ones fall out. */
  function createTour(getGraph) {
    const played = [];              // ordered history, drives .played styling
    const playedSet = new Set();
    const unplayable = new Set();   // played-but-silent: no Spotify id
    let stack = [];                 // DFS spine
    let lastId = null;              // last node handed out, for jump anchoring

    function queryView() {
      const g = getGraph() || {};
      const nodes = (g.nodes || []).filter(n => n && n.kind === 'query');
      const live = new Set(nodes.map(n => n.id));
      const adj = new Map();
      nodes.forEach(n => adj.set(n.id, []));
      (g.links || []).forEach(l => {
        const s = idOf(l.source), t = idOf(l.target);
        if (!live.has(s) || !live.has(t) || s === t) return;
        const score = (l.score == null) ? 0 : l.score;
        adj.get(s).push({ id: t, score });
        adj.get(t).push({ id: s, score });
      });
      return { nodes, live, adj };
    }

    // Highest score first; id as a stable tie-break so a tour over equal-score
    // edges is reproducible rather than dependent on link insertion order.
    function unplayedNeighbors(id, view) {
      const seen = new Set();
      return (view.adj.get(id) || [])
        .filter(e => {
          if (playedSet.has(e.id) || seen.has(e.id)) return false;
          seen.add(e.id);
          return true;
        })
        .sort((a, b) => (b.score - a.score) || (a.id < b.id ? -1 : 1));
    }

    function take(id, via) {
      played.push(id);
      playedSet.add(id);
      lastId = id;
      return { type: 'node', id: id, via: via == null ? null : via };
    }

    /* Returns one of:
     *   {type:'node', id, via}            play this next
     *   {type:'need-jump', from, exclude} component exhausted; caller resolves
     *   {type:'done'}                     nothing playable is left
     */
    function next() {
      const view = queryView();
      while (stack.length) {
        const cur = stack[stack.length - 1];
        if (!view.live.has(cur)) { stack.pop(); continue; }  // removed mid-tour
        const cand = unplayedNeighbors(cur, view);
        if (cand.length) {
          stack.push(cand[0].id);
          return take(cand[0].id, cur);
        }
        stack.pop();                                          // dead end
      }
      const remaining = view.nodes.filter(n => !playedSet.has(n.id));
      if (!remaining.length) return { type: 'done' };
      return { type: 'need-jump', from: lastId, exclude: played.slice() };
    }

    /* Answer to a 'need-jump'. id === null means the server found nothing —
     * fall back to map order so a failed lookup can never end a tour early. */
    function resumeWithJump(id) {
      const view = queryView();
      let target = (id && view.live.has(id) && !playedSet.has(id)) ? id : null;
      if (!target) {
        const fallback = view.nodes.find(n => !playedSet.has(n.id));
        if (!fallback) return { type: 'done' };
        target = fallback.id;
      }
      stack = [target];
      return take(target, null);
    }

    /* Redirect a running tour to `id`, keeping everything already played.
     * Distinct from start(), which wipes history: steering says "go here next",
     * not "begin again". The node becomes the new head of the DFS spine, so the
     * walk continues outward from there. Re-steering to an already-played song
     * replays it without duplicating it in the history. Returns null if the
     * node isn't a live query node. */
    function steerTo(id) {
      const view = queryView();
      if (!id || !view.live.has(id)) return null;
      stack.push(id);
      if (playedSet.has(id)) { lastId = id; return { type: 'node', id: id, via: null }; }
      return take(id, null);
    }

    function start(startId) {
      const view = queryView();
      let id = (startId && view.live.has(startId)) ? startId : null;
      if (!id) {
        const first = view.nodes[0];
        if (!first) return { type: 'done' };
        id = first.id;
      }
      played.length = 0;
      playedSet.clear();
      unplayable.clear();
      stack = [id];
      lastId = null;
      return take(id, null);
    }

    return {
      start,
      next,
      steerTo,
      resumeWithJump,
      // A track with no Spotify id is still marked played (next() did that when
      // it handed the node out) and stays on the stack, so the tour routes
      // THROUGH it to its neighbors rather than treating it as a wall.
      markUnplayable: id => { unplayable.add(id); },
      isUnplayable: id => unplayable.has(id),
      isPlayed: id => playedSet.has(id),
      getPlayed: () => played.slice(),
      getUnplayable: () => Array.from(unplayable),
      remaining: () => queryView().nodes.filter(n => !playedSet.has(n.id)).length,
      reset() { played.length = 0; playedSet.clear(); unplayable.clear(); stack = []; lastId = null; },
    };
  }

  const api = { createTour, idOf };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AutoplayTraversal = api;
})(typeof self !== 'undefined' ? self : this);
