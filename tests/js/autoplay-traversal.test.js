/* Unit tests for the graph-autoplay traversal planner.
 *
 * Plain node + assert — this repo has no JS test tooling and adding a runner
 * for one file is not worth it. Run directly:
 *
 *     node tests/js/autoplay-traversal.test.js
 */
'use strict';
const assert = require('assert');
const { createTour } = require('../../ui/static/autoplay-traversal.js');

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ok   ' + name); }
  catch (err) { console.error('  FAIL ' + name + '\n       ' + err.message); process.exitCode = 1; }
}

/* ── fixtures ───────────────────────────────────────────────────────────── */

function q(id) { return { id: id, kind: 'query', name: id }; }
function corpus(id) { return { id: id, kind: 'corpus', name: id }; }
function link(s, t, score) { return { source: s, target: t, score: score == null ? 50 : score }; }

function graphOf(nodes, links) {
  return () => ({ nodes: nodes, links: links });
}

/* Drive a tour to completion. The jump resolver stands in for the server's
 * /api/autoplay/jump; by default it picks the first remaining node. Returns the
 * play order plus a per-step trace so tests can count jumps. */
function runTour(getGraph, opts) {
  opts = opts || {};
  const tour = createTour(getGraph);
  const order = [], steps = [];
  const jump = opts.jump || (rem => rem[0]);
  let step = opts.startId === undefined ? tour.start(undefined) : tour.start(opts.startId);
  let guard = 0;
  while (step.type !== 'done') {
    if (++guard > 500) throw new Error('traversal did not terminate');
    if (step.type === 'need-jump') {
      const g = getGraph();
      const done = new Set(tour.getPlayed());
      const rem = g.nodes.filter(n => n.kind === 'query' && !done.has(n.id)).map(n => n.id);
      steps.push({ kind: 'jump', from: step.from });
      step = tour.resumeWithJump(jump(rem, step.from));
      continue;
    }
    order.push(step.id);
    steps.push({ kind: step.via === null ? 'seed' : 'edge', id: step.id, via: step.via });
    if (opts.onNode) opts.onNode(step.id, tour);
    step = tour.next();
  }
  return { order, steps, tour, jumps: steps.filter(s => s.kind === 'jump').length };
}

/* ── tests ──────────────────────────────────────────────────────────────── */

console.log('autoplay traversal');

test('path graph plays end to end in order', () => {
  const nodes = ['a', 'b', 'c', 'd'].map(q);
  const links = [link('a', 'b'), link('b', 'c'), link('c', 'd')];
  const { order, jumps } = runTour(graphOf(nodes, links), { startId: 'a' });
  assert.deepStrictEqual(order, ['a', 'b', 'c', 'd']);
  assert.strictEqual(jumps, 0);
});

test('greedy: highest-scoring edge wins at each hop', () => {
  const nodes = ['a', 'b', 'c', 'd'].map(q);
  const links = [link('a', 'b', 10), link('a', 'c', 90), link('a', 'd', 50)];
  const { order } = runTour(graphOf(nodes, links), { startId: 'a' });
  // star centred on a: greedy takes c (90), dead-ends, backtracks to a,
  // takes d (50), backtracks, takes b (10).
  assert.deepStrictEqual(order, ['a', 'c', 'd', 'b']);
});

test('star graph backtracks through the hub and plays each leaf once', () => {
  const nodes = ['hub', 'l1', 'l2', 'l3'].map(q);
  const links = [link('hub', 'l1', 30), link('hub', 'l2', 20), link('hub', 'l3', 10)];
  const { order, jumps } = runTour(graphOf(nodes, links), { startId: 'hub' });
  assert.deepStrictEqual(order, ['hub', 'l1', 'l2', 'l3']);
  assert.strictEqual(jumps, 0, 'a connected star needs no jumps');
});

test('two disconnected components produce exactly one jump', () => {
  const nodes = ['a', 'b', 'x', 'y'].map(q);
  const links = [link('a', 'b'), link('x', 'y')];
  const { order, jumps } = runTour(graphOf(nodes, links), { startId: 'a' });
  assert.strictEqual(jumps, 1);
  assert.deepStrictEqual(order.slice(0, 2), ['a', 'b']);
  assert.deepStrictEqual(order.slice(2).sort(), ['x', 'y']);
});

test('fully isolated nodes produce N-1 jumps', () => {
  const nodes = ['a', 'b', 'c', 'd'].map(q);
  const { order, jumps } = runTour(graphOf(nodes, []), { startId: 'a' });
  assert.strictEqual(order.length, 4);
  assert.strictEqual(jumps, 3);
});

test('every node plays exactly once across shapes', () => {
  const shapes = [
    [['a', 'b', 'c', 'd', 'e'], [link('a', 'b'), link('b', 'c'), link('c', 'd'), link('d', 'e')]],
    [['a', 'b', 'c', 'd'], [link('a', 'b'), link('a', 'c'), link('a', 'd')]],
    [['a', 'b', 'c'], []],
    // cycle — the classic double-play trap
    [['a', 'b', 'c'], [link('a', 'b'), link('b', 'c'), link('c', 'a')]],
    // dense clique
    [['a', 'b', 'c', 'd'], [link('a', 'b'), link('a', 'c'), link('a', 'd'),
                            link('b', 'c'), link('b', 'd'), link('c', 'd')]],
  ];
  shapes.forEach(([ids, links], i) => {
    const { order } = runTour(graphOf(ids.map(q), links), { startId: ids[0] });
    assert.deepStrictEqual(order.slice().sort(), ids.slice().sort(), 'shape ' + i + ' coverage');
    assert.strictEqual(new Set(order).size, order.length, 'shape ' + i + ' has a repeat');
  });
});

test('duplicate links between the same pair do not cause a replay', () => {
  const nodes = ['a', 'b'].map(q);
  const links = [link('a', 'b', 40), link('b', 'a', 40), link('a', 'b', 40)];
  const { order } = runTour(graphOf(nodes, links), { startId: 'a' });
  assert.deepStrictEqual(order, ['a', 'b']);
});

test('corpus nodes are never played and never routed through', () => {
  const nodes = [q('a'), corpus('ctx'), q('b')];
  // a and b are connected ONLY via the corpus node
  const links = [link('a', 'ctx'), link('ctx', 'b')];
  const { order, jumps } = runTour(graphOf(nodes, links), { startId: 'a' });
  assert.deepStrictEqual(order, ['a', 'b']);
  assert.strictEqual(jumps, 1, 'corpus node must not bridge, so this is an island jump');
});

test('unplayable node is skipped but still routed through', () => {
  const nodes = ['a', 'dead', 'c'].map(q);
  const links = [link('a', 'dead'), link('dead', 'c')];
  const { order, tour } = runTour(graphOf(nodes, links), {
    startId: 'a',
    onNode: (id, t) => { if (id === 'dead') t.markUnplayable(id); },
  });
  assert.deepStrictEqual(order, ['a', 'dead', 'c']);
  assert.deepStrictEqual(tour.getUnplayable(), ['dead']);
  assert.ok(tour.isUnplayable('dead') && !tour.isUnplayable('c'));
});

test('d3-style object link endpoints are understood', () => {
  const a = q('a'), b = q('b');
  // d3's forceLink rewrites source/target into node references in place
  const { order } = runTour(graphOf([a, b], [{ source: a, target: b, score: 50 }]), { startId: 'a' });
  assert.deepStrictEqual(order, ['a', 'b']);
});

test('nodes added mid-tour are picked up', () => {
  const nodes = [q('a'), q('b')];
  const links = [link('a', 'b')];
  const { order } = runTour(graphOf(nodes, links), {
    startId: 'a',
    onNode: id => {
      if (id === 'b' && nodes.length === 2) {         // a playlist streams in
        nodes.push(q('c'));
        links.push(link('b', 'c'));
      }
    },
  });
  assert.deepStrictEqual(order, ['a', 'b', 'c']);
});

test('a node removed mid-tour is dropped from the spine', () => {
  const nodes = [q('a'), q('b'), q('c')];
  const links = [link('a', 'b', 90), link('a', 'c', 10)];
  const { order } = runTour(graphOf(nodes, links), {
    startId: 'a',
    onNode: id => {
      if (id === 'b') {                                // user deletes the hub
        const i = nodes.findIndex(n => n.id === 'a');
        nodes.splice(i, 1);
      }
    },
  });
  // a played, then b; a is gone so the spine unwinds to nothing and c is a jump
  assert.deepStrictEqual(order, ['a', 'b', 'c']);
});

test('start falls back to the first query node when the seed is absent', () => {
  const nodes = [corpus('ctx'), q('a'), q('b')];
  const { order } = runTour(graphOf(nodes, [link('a', 'b')]), { startId: 'nope' });
  assert.deepStrictEqual(order, ['a', 'b']);
});

test('an empty map is done immediately', () => {
  const tour = createTour(graphOf([], []));
  assert.strictEqual(tour.start(undefined).type, 'done');
});

test('a corpus-only map is done immediately', () => {
  const tour = createTour(graphOf([corpus('x'), corpus('y')], []));
  assert.strictEqual(tour.start(undefined).type, 'done');
});

test('a failed jump lookup falls back to map order instead of ending the tour', () => {
  const nodes = ['a', 'x', 'y'].map(q);
  const { order } = runTour(graphOf(nodes, [link('x', 'y')]), {
    startId: 'a',
    jump: () => null,                                  // server call failed
  });
  assert.strictEqual(order.length, 3, 'tour must not end early on a failed jump');
  assert.strictEqual(new Set(order).size, 3);
});

test('jump target that is stale or already played is ignored', () => {
  const nodes = ['a', 'x'].map(q);
  const { order } = runTour(graphOf(nodes, []), {
    startId: 'a',
    jump: () => 'a',                                   // server returned a played id
  });
  assert.deepStrictEqual(order, ['a', 'x']);
});

test('need-jump reports the anchor and the exclude set', () => {
  const tour = createTour(graphOf(['a', 'b'].map(q), []));
  tour.start('a');
  const step = tour.next();
  assert.strictEqual(step.type, 'need-jump');
  assert.strictEqual(step.from, 'a', 'jump anchors on the last played node');
  assert.deepStrictEqual(step.exclude, ['a']);
});

test('steerTo redirects the walk but keeps history', () => {
  const nodes = ['a', 'b', 'far'].map(q);
  const tour = createTour(graphOf(nodes, [link('a', 'b')]));
  tour.start('a');
  assert.deepStrictEqual(tour.steerTo('far'), { type: 'node', id: 'far', via: null });
  assert.deepStrictEqual(tour.getPlayed(), ['a', 'far'], 'history is kept, not wiped');
  // 'far' is isolated, so the spine unwinds back to 'a' — steering redirects
  // the walk without abandoning what the old branch could still reach.
  assert.deepStrictEqual(tour.next(), { type: 'node', id: 'b', via: 'a' });
});

test('steerTo continues outward from the new node', () => {
  const nodes = ['a', 'x', 'y'].map(q);
  const tour = createTour(graphOf(nodes, [link('x', 'y')]));
  tour.start('a');
  tour.steerTo('x');
  assert.deepStrictEqual(tour.next(), { type: 'node', id: 'y', via: 'x' });
});

test('steerTo an already-played song replays it without duplicating history', () => {
  const nodes = ['a', 'b'].map(q);
  const tour = createTour(graphOf(nodes, [link('a', 'b')]));
  tour.start('a');
  tour.next();                                   // plays b
  const step = tour.steerTo('a');
  assert.deepStrictEqual(step, { type: 'node', id: 'a', via: null });
  assert.deepStrictEqual(tour.getPlayed(), ['a', 'b'], 'no duplicate entry');
});

test('steerTo rejects unknown and corpus nodes', () => {
  const tour = createTour(graphOf([q('a'), corpus('ctx')], []));
  tour.start('a');
  assert.strictEqual(tour.steerTo('ctx'), null);
  assert.strictEqual(tour.steerTo('nope'), null);
  assert.strictEqual(tour.steerTo(null), null);
});

test('reset clears history', () => {
  const tour = createTour(graphOf(['a', 'b'].map(q), [link('a', 'b')]));
  tour.start('a');
  tour.next();
  assert.strictEqual(tour.getPlayed().length, 2);
  tour.reset();
  assert.deepStrictEqual(tour.getPlayed(), []);
  assert.strictEqual(tour.isPlayed('a'), false);
});

console.log('\n' + passed + ' passed' + (process.exitCode ? ', FAILURES above' : ''));
