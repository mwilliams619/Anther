# Multi-Song Recommendation ("find songs like this feeling") — implementation notes

**Use case 2.** A user searches several songs, then gets back songs similar to
the *set* as a whole — not to any one of them. Distinct from the existing
per-song placement (use case 1) and from calibrated playlist-fit (which scores
*one query against a fixed playlist*).

**Executor:** Claude Code, in `/home/matt/Dev/Anther`.
**Status:** core + endpoint + tests landed and green. UX decisions locked
(2026-07-09, see below). Frontend (modal, "Recommend from map" action, auto
centroid/topk switch, dedup/skip/empty handling) not yet built.

---

## What was decided (in conversation)

- **Output:** *both* a distinct ranked recommendation list **and** the existing
  force-graph view (recommended tracks are spliced into the graph as well).
- **Aggregation:** **centroid** — average the seed vectors, rank corpus tracks
  by cosine to that shared center. `topk` (mean of top-k per-seed cosines) is
  implemented as a drop-in fallback for the known centroid failure mode (see
  Caveats), selectable per request; centroid is the default.

## Design finding: the ML core did not need to change

Embedding recipe, `SongIndex`, Leiden partition, the frozen corpus bundle,
`eval.py`, and every invariant are untouched. Multi-song recommendation is an
**aggregation layer** on top of primitives that already existed:

- `SongIndex.transform_query(vec)` applies the corpus's frozen standardize + L2
  to a raw seed, so seeds and corpus rows live in one space (the same guarantee
  `place()` relies on — raw in, index applies its own transform).
- `index.embeddings @ q` is the exact cosine search already used everywhere.
- `_topk_mean` (used by `playlist_fit`) is reused verbatim for the `topk` mode.
- Seed vectors are **already available** with no re-embedding: in-corpus songs
  are corpus rows; anything the user searched/placed is in the embed cache
  (`ui/session/embed_cache.sqlite`, via `cache_vec`).

## What was added

1. **`anther_ml/corpus/place.py` → `recommend_from_seeds(corpus, seed_vecs,
   top_k=20, exclude_ids=None, method="centroid", per_seed_k=3)`**
   - Transforms each raw seed via `index.transform_query`, then:
     - `centroid`: mean of unit seeds → renormalize → `embeddings @ centroid`.
     - `topk`: `embeddings @ Q.T` → `_topk_mean(sims, per_seed_k)`.
   - Drops `exclude_ids` (caller passes the seed ids so a seed can't recommend
     itself), returns `{"rank","score",**metadata}` rows — identical shape to
     `SongIndex.query`, so the frontend can render it like any neighbor list.
   - Raises on empty seeds, unknown method, and a degenerate (zero) centroid
     (exactly-antipodal seeds).
   - Exported from `anther_ml.corpus` (`__init__.py` + `__all__`).

2. **`ui/atlas.py` → `recommend(seed_ids, top_k=20, method="centroid",
   splice=True)`** and **`_seed_vec(seed_id)`**
   - `_seed_vec` resolves a placed node id to a raw vector: corpus row
     (`_id_to_idx`) if in-corpus, else the embed cache (`cached_vec`).
   - `recommend` collects seed vectors, reports any id with no cached vector in
     `skipped` (rather than silently dropping it), calls `recommend_from_seeds`
     with the seeds excluded, and — when `splice=True` — merges each in-corpus
     recommendation into the shared graph via `_place_corpus_track(...,
     extra={"recommended": True})` so the list and the map stay in sync.
   - Returns `{"results", "n_seeds", "skipped", "method"}`.

3. **`ui/app.py` → `POST /api/recommend`**
   - Body `{seed_ids: [...], top_k?: int, method?: "centroid"|"topk"}`.
   - `400` on bad input (empty/unresolvable seeds), `500` otherwise; mirrors the
     existing `/api/place` and `/api/playlist/place` handlers.

4. **`tests/test_corpus_recommend.py`** — 8 tests, all green. Centroid ranks the
   shared center first on a hand-checkable orthonormal corpus; `exclude_ids`
   drops seeds; rank/score shape matches `query`; empty/unknown-method/antipodal
   error paths; `topk` recovers a two-mood seed set the centroid strands; real
   fixture-corpus recommendations stay in-cluster. Existing
   `test_corpus_place`, `test_atlas_*`, `test_similarity` still pass.

## Caveats / known tradeoffs

- **Centroid strands multi-mood seed sets.** If the seeds span two distinct
  moods, the average lands in the sparse valley between them and top results can
  feel like neither. Mitigation is already in place: pass `method="topk"` (no
  interface change) to score by nearest-subset instead. `test_topk_recovers_
  both_modes_when_centroid_strands` demonstrates this.
- **Seeds must be embedded first.** A seed id only resolves if it's in-corpus or
  was previously searched/placed (embed cache). Ids with no vector come back in
  `skipped`; the frontend should surface that ("2 of 5 seeds not yet embedded").
- **`splice=True` only splices in-corpus recommendations** into the graph.
  Recommendations are drawn from the corpus, so in practice all of them splice;
  the guard is defensive.

---

## UX decisions (2026-07-09)

1. **Seed collection.** All currently-placed nodes on the map are the seed set.
   No separate pin/select UI — a "Recommend from map" action calls
   `/api/recommend` with every placed node id. Re-adding/removing map nodes and
   re-running naturally changes the seed set.

2. **Where the list lives.** Modal/overlay, opened by "Recommend from map."
   Clicking a result places it (`/api/place`, splice already handles the graph
   side) — closing the modal returns to the graph view. (No auto-promote-to-seed
   round-trip for v1; re-running recommend after placing more nodes achieves
   the same "refine the feeling" loop since seeds = placed nodes.)

3. **Centroid vs. topk.** Hidden auto-switch. Compute mean pairwise cosine
   across seeds; below a threshold (start at 0.3, tune empirically), call
   `recommend_from_seeds(..., method="topk")` instead of centroid. No visible
   toggle in v1.

4. **De-duplication.** Exclude already-placed songs from results outright (in
   addition to the existing seed exclusion) — filter recommended ids against
   the current map's placed-id set before returning. `top_k` stays default 20.

5. **Skipped seeds.** No blocking, no background job. Recommend immediately
   from whatever seeds resolve; response's `skipped` list drives a small note
   in the modal ("2 of 5 seeds not yet embedded").

6. **Empty/degenerate state.** Plain message in the modal, no silent fallback:
   "no seeds could be embedded" when everything is skipped, "these songs point
   in very different directions — try fewer or more similar songs" when
   centroid is degenerate. User adjusts seeds manually; no auto-retry with
   topk (the auto mean-pairwise-cosine switch in #3 already prevents most
   degenerate-centroid cases before they'd reach this path).
