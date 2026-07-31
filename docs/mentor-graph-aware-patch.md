# Mentor graph-aware rework — patch summary

The mentor chatbot could not answer questions about artists on the user's
screen. Ten target questions failed: five returned a generic deflection
("I can map this, but I need a concrete anchor"), and the "sonic territories"
question hallucinated artist names that aren't in the corpus. All ten now
answer correctly against the live on-screen graph, with no new model.

## Root cause

The graph tools resolved every anchor against the **frozen 99k-track corpus**
and never consulted the **live on-screen session**. The artists in the
motivating question (Rae Sremmurd, Jimi Hendrix, Unknown Mortal Orchestra)
were all on screen — placed from Deezer previews — but Rae Sremmurd isn't in
the corpus, so the anchor resolved to `None` and the whole loop deflected.

The fix threads the live session (`ui/session/graph.json` +
`ui/session/embed_cache.sqlite`) into the tool layer, defaults candidate
scope to on-screen, adds three tools the questions needed, and rebuilds the
reasoning loop so it composes answers from tool observations only.

## What changed, by file

| File | Change |
|---|---|
| `mentor/mentor_graphctx.py` *(new, 229 lines)* | `GraphContext`: reads `graph.json` (mtime-cached), reconstructs raw MERT vectors from `embed_cache.sqlite`, and resolves anchors **on-screen first** (exact artist centroid → exact track → substring → fuzzy). Runs in the mentor's own process, reading session state from disk. |
| `mentor/mentor_anther.py` *(+35)* | `resolve_anchor`/`resolve_two` take `graph_ctx=None`; a new `_resolve_spec` tries the on-screen graph before the corpus. `graph_ctx=None` reproduces the old corpus-only behavior (backward compatible). |
| `mentor/mentor_graph.py` *(+460)* | `MentorGraphTools.__init__(anther, graph_ctx=None)` auto-loads a `GraphContext`. All candidate tools gained a `scope` argument defaulting to `"onscreen"` (corpus fallback when nothing is placed). `bridge` gained a `within=<artist>` shape. Three new tools: `cluster_summary`/`get_cluster_profiles`, `micro_genres`, `coherence`. |
| `mentor/mentor_react.py` *(+522)* | Rewrote the ReAct loop with fable-method discipline: deterministic `classify_question` seeds the call, surprise-routing on failure, an answer **composed from observations only** (hallucination-proof), and honest anchor-specific deflection. Registers all 10 tools. |
| `mentor/mentor.py` *(+34)* | Builds and threads `GraphContext`; `_fallback_intent` recognizes the new question shapes; the GRAPH branch runs `max_steps=4` and surfaces the loop's honest deflection instead of the old generic line. |
| `tests/test_mentor_graph_aware.py` *(new, 176 lines)* | No-GPU harness: 12 tests running the full loop against the live graph. |
| `tests/test_mentor_graph_tools.py` *(+26)* | Legacy fixtures updated for the `graph_ctx` kwarg (`_EmptyGraphCtx` forces corpus scope). |

## Design decisions (measured, not assumed)

The MERT embedding cone is tight: corpus random-pair cosine averages **0.953**
(p5–p95 0.906–0.981); on-screen between-artist centroids span 0.950–0.987;
within-artist coherence is 0.985–0.992 (std 0.003–0.009). Two consequences:

- **Bridge lean is relative** with a scaled tie-band (~0.005), not the old
  absolute ±0.02 — which mislabelled everything at on-screen scale.
- **Coherence outliers are z-scores** (node-to-centroid cosine >~2σ below the
  group mean), not an absolute cutoff — all absolute cosines sit near 0.98, so
  a fixed threshold flags all-or-nothing.

Full rationale in [`docs/mentor-graph-aware.md`](mentor-graph-aware.md).

## Reasoning loop (fable-method discipline)

1. **Classify the question shape deterministically** — this seeds the first
   tool call, so the loop is correct even when the model emits nothing usable
   (the old loop fell to a single-guess regex on any bad parse).
2. **Surprise-route on failure** — a `within` artist that isn't on screen
   drops to a plain bridge; an unresolved artist pool drops to `sounds_like`.
   The loop only stops when an anchor is genuinely absent.
3. **Compose from observations only** — artist/track names come from tool
   results. The LLM (when present) only rephrases in voice, and its output is
   discarded if it introduces a name that wasn't observed. This is why Q9 can
   no longer hallucinate.
4. **Honest deflection** — "'Rae Sremmurd' isn't on your map yet" names the
   missing input, instead of one generic line for every failure.

## Test results

`tests/test_mentor_graph_aware.py` — **12 passed** (no GPU, no model, no
network; runs the deterministic composer against live `graph.json`).
Full mentor suite (`test_mentor_graph_aware` + `test_mentor_graph_tools` +
`test_mentor_context`) — **22 passed**.

Live answers now produced (deterministic composer, `generate_fn=None`):

| # | Question | Tool | Grounded answer |
|---|---|---|---|
| 1 | demo sounds closest to (embolo) | `sounds_like` | Justin Bieber – No Sense (0.972), Rae Sremmurd – Set The Roof |
| 2 | micro-genres of embolo | `micro_genres` | chillwave, boom bap, chillhop *(from on-screen probe tags)* |
| 3 | artists between Hendrix & Bieber | `bridge` | Justin Bieber – Company, Skrillex – Where Are Ü Now |
| 4 | which Hendrix song matches embolo | `artist_tracks` | May This Be Love (0.943), Are You Experienced? |
| 5 | adjacent cluster for Rae Sremmurd | `crossover` | Drum and Bass / Chillhop neighbours |
| 6 | coherence of album *Purpose* | `coherence` | mean 0.988, 4 clusters, outlier **Life Is Worth Living** (z −2.0) |
| 7 | tagmates of Jimi Hendrix | `tagmates` | UMO – The Garden (new romantic, progressive rock), … |
| 8 | compare Rae Sremmurd vs Hendrix | `compare` | similarity 0.950 |
| 9 | major sonic territories | `cluster_summary` | **84 real territories** w/ labels, sizes, exemplars (no hallucination) |
| 10 | recommend from Rae + Hendrix | `bridge` | May This Be Love, Take It Or Leave It, … |
| ★ | UMO song bridging Rae↔Hendrix | `bridge within` | **Meshuggah** (0.970); most balanced **The Widow** |

## What does not change

- **No new or bigger model.** The 3B stays; the quality is in the loop
  structure, the on-screen evidence, and the grounded composer.
- **Backward compatible.** `graph_ctx=None` and `scope="corpus"` reproduce the
  original corpus-only behavior; the frozen corpus is still the fallback when
  nothing is on screen.
