# Mentor: graph-aware, on-screen-first reasoning

Design note for the change that makes the chat mentor answer graph / song-similarity
questions from the **live on-screen session**, not just the frozen 99 k corpus.

## The bug, in one sentence

`MentorReAct(anther)` → `MentorGraphTools(anther)` only ever received the frozen-corpus
`AntherSoundsLike`; every anchor resolved against the 99 k corpus metadata, so an artist
the user had **placed on the map** (Deezer preview / uploaded audio) resolved to `None`,
the tool returned `ok:False`, and chat fell to the generic *"I need a concrete anchor"*
deflection — even with the artist sitting on screen. `cluster_summary`-type questions had
no tool at all, so the 3B free-styled and hallucinated artist names.

## Cross-process reality (load-bearing)

The mentor service (`:5100`) runs in a **separate process** from the UI (`:5000`).
`atlas._graph` / `atlas._query_vecs` are module globals in the UI process; in the mentor
process they are empty. Two things *are* process-independent because they live on disk:

- `ui/session/graph.json` — the placed nodes (78–79 of them), rewritten by the UI on every
  placement.
- `ui/session/embed_cache.sqlite` — raw MERT-v1-330M 1024-d vectors keyed by track id,
  readable via `atlas.cached_vec(track_id)`.

**Decision:** graph-awareness reads `graph.json` fresh per graph-turn (mtime-cached) and
pulls vectors via `atlas.cached_vec`. We do **not** rely on `atlas._graph` in-memory. This
sidesteps cross-process state entirely.

## Vector consistency

- Corpus `_resolve_name` returns **raw** 1024-d embeddings (`raw_embeddings[idx]`).
- `atlas.cached_vec(id)` returns **raw** MERT 1024-d.
- Every tool calls `index.transform_query(vec)` before scoring; with `standardize=False`
  (Phase-2 default) that is just L2-normalize.

So an on-screen resolver that returns raw cached vectors is drop-in consistent with the
corpus path. Artist anchors → mean of the artist's on-screen node vectors (centroid).

## Anchor resolution tiers (PR-1)

`resolve_anchor(spec, context, graph_ctx)` tries, in order:

| Tier | Source | `type` | Notes |
|---|---|---|---|
| 0 | context ref ("me", "my track", "those artists") | `context_ref` | unchanged |
| 1 | audio path / `{audio_path}` dict | `audio` | unchanged; embeds via MERT |
| 2 | **on-screen exact artist** | `onscreen_artist` | centroid of placed nodes |
| 3 | **on-screen exact / substring track name** | `onscreen_track` | single node vec |
| 4 | **on-screen fuzzy** (artist then name, cutoff 0.8/0.82) | `onscreen_*` `fuzzy` | |
| 5 | corpus exact/fuzzy (`_resolve_name`) | `artist`/`track` | fallback, always allowed |

Resolution always allows the corpus fallback (so a corpus-only artist still resolves).
The **on-screen-first** rule is what makes placed artists win when both have them.

## Candidate scope (PR-2) — "answer from on-screen, not random corpus songs"

`bridge`, `compare`, `crossover`, `tagmates`, `sounds_like` take `scope` (default
`"onscreen"`). When `onscreen`, the neighbour candidate pool is the **current graph nodes'
vectors**, not `index.embeddings`. `scope="corpus"` is explicit opt-in ("find anything in
the library that…"). This kills the failure mode where the bot answers with obscure corpus
tracks nobody placed, and it removes most of the noisy-centroid caveat (we compare a handful
of user-chosen artists, not 84 near-collinear global clusters).

`bridge` also gains an optional `within=<artist>` arg: candidate pool = that artist's
on-screen songs, scored by min-similarity to both anchors. This is the shape of the original
motivating question ("what song from **UMO** bridges the Rae Sremmurd and Jimi Hendrix
clusters"), and returns **"Meshuggah"** on the live graph.

## New tools (PR-3)

- `cluster_summary()` / `get_cluster_profiles()` — reads `cluster_profiles.json` (already in
  `AntherSoundsLike.profiles`): the 84 sonic territories with label, exemplars, top tags.
  Fixes Q9's hallucination by giving the model real data to read.
- `micro_genres(anchor)` — on-screen node → reads the **probe `tags`** already stored on the
  node (all 59 vec-bearing nodes carry them); off-screen → computes via neighbour tag
  inheritance. Answers Q2.
- `coherence(scope)` — over a placed album / playlist / artist's on-screen nodes: centroid,
  each node's cosine to it, distinct-cluster spread, and **z-score outliers**. Answers Q6.

## Threshold re-tuning (measured on the live graph)

The MERT space is tight and compressed:

- corpus random-pair cosine: mean **0.953**, p5–p95 **0.906–0.981**
- on-screen between-artist centroids: **0.950–0.987**
- on-screen within-artist coherence: mean **0.985–0.992**, std **0.003–0.009**

Consequences:

- `bridge` lean was an absolute `±0.02` margin (corpus-tuned). At on-screen scale, between-
  artist gaps are ~0.03–0.04 and within-artist std ~0.005, so `±0.02` mislabels. **Change to
  relative lean**: compare `sa` vs `sb` directly with a scaled tie-band (`0.005`), else the
  larger side wins.
- `coherence` outliers use a **z-score** (node cosine to centroid > ~2σ below the group mean),
  not an absolute cutoff — absolute cosines are all ~0.98 and an absolute threshold would flag
  everything or nothing.

## Reasoning-mode loop (PR-4, fable-method discipline)

The old loop was greedy: first parseable JSON or a single-guess `_fallback_call`,
`max_steps=3`, and on all-`ok:False` it hit the generic deflection. Rewrite per fable-method
(borrow the *loop discipline*, not the subagent orchestration):

1. **Classify the question shape** deterministically (single-anchor / two-anchor / three-anchor
   `within` / subset-stat / profile-lookup / recommend) → pick the matching tool + args. This
   seeds the call and **backs up** an unusable LLM emission (so the loop is correct even with a
   weak/stub generator — fable-method's thesis: structure over model capacity).
2. **A surprise routes the loop.** `ok:False` is not a terminal state: re-resolve, and if an
   anchor is genuinely absent, stop with `status="absent_anchor"` naming the missing anchor.
3. **Verify by observation.** The final answer is composed **deterministically from the
   observations** (names only from tool results); the LLM is used only to phrase it in voice
   when a GPU model is present. A hallucinated cluster answer can't pass because the composer
   never emits an unobserved name.
4. **Honest deflection** distinguishes *absent* ("Rae Sremmurd isn't on your map — want me to
   place it, or pick another anchor?") from *ambiguous* (needs disambiguation), instead of one
   generic line.

## What does NOT change

- No new model; the 3B stays. The upgrade is loop structure + data wiring.
- Corpus path and existing tool signatures stay backward-compatible (`graph_ctx=None` and
  `scope="corpus"` reproduce old behaviour); existing callers keep working.
- Read-only: no tool places, removes, or mutates the graph.
