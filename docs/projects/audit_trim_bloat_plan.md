## Context

This document originally was a first-pass audit that ranked 12 files by
"bloat" and gave one-line justifications, but it was written without reading
full file bodies — several of its claims didn't survive contact with the
actual code. Before running fix passes, the claims needed to be checked
against the real functions/line numbers, otherwise a "fix" could target code
that isn't actually duplicated (wasted effort) or miss the real duplication in
a file (missed value). All 12 files were read in full and checked
function-by-function. This revision corrects the record and turns the
survivors into an ordered backlog with exact extraction targets, so each
future pass is a known, mechanical, low-ambiguity diff — not a fresh
investigation.

Two additional facts change the risk posture from the original pass:
- Every target file was touched **2–7 days ago** ("ml development first
  commit" through "adding genre metadata labeling") — this is active
  development, not settled legacy code. Passes must be small and independently
  revertable.
- `mentor/` is entirely **untracked** (`git status`: `?? mentor/`) — it has
  never been committed. Refactoring it is editing in-flight, unreviewed work,
  not a stable module — extra reason to keep each pass small and test-backed.

## Claim validation summary

| # | File | Original claim | Verdict | Real finding |
|---|---|---|---|---|
| 1 | `corpus/build.py` | checkpoint/dedupe/profile-gen are the bloat | **Partially accurate** | Those are necessary complexity, not bloat. The real hotspot is `build_corpus` itself (368-558, 191 lines) — two near-duplicate embed loops + a long finalize tail, uncalled-out. |
| 2 | `cluster.py` | duplicated clustering paths, repeated diagnostics | **Accurate** | `fit_clusters` vs `fit_clusters_leiden` duplicate scale→PCA (66-75 vs 298-305) and 2D-UMAP-for-viz (89-96 vs 320-325) blocks near-verbatim. |
| 3 | `mpd_sql.py` | extended-INSERT parsing split across near-overlapping helpers | **Accurate** | `_iter_tuples` (137-152) and `_split_fields` (169-184) hand-roll near-identical quote/escape scanners. |
| 4 | `spotify_deezer.py` | matching/retry/waveform interleaved | **Wrong** | File is cleanly section-organized; retry lives only in `_deezer_get`, waveform only in `clip_waveform`/`fetch_preview_waveform`. No structural extraction warranted — **drop from backlog**. |
| 5 | `embedding.py` | window planning/waveform prep split across "several" call paths | **Partially accurate** | Only 2 paths, not several. `get_embedding` (223-226) and `embed_batch`'s inner loop (272-275) share a byte-identical 4-line block. |
| 6 | `corpus/sources.py` | repeated sampling/capping logic across sources | **Wrong** | Capping (`_cap_by_artist`) is called exactly once. Real duplication (missed by original claim): the "deezer match → fetch preview → mark missed" block is repeated in `mpd_source` (174-189) and `sql_source.fetch_one` (272-284). |
| 7 | `eval.py` | title-stem normalization, ranking, HTML export somewhat duplicated | **Partially accurate** | `title_stem` is NOT duplicated (single function, reused correctly). Real issues: `neighbor_audit` recomputes a similarity matrix `_cosine_rankings` already built; `write_audit`'s field list is written once for CSV and again hand-rolled for HTML. |
| 8 | `corpus/labels.py` | normalization and profile mutation are a moderate cleanup target | **Partially accurate** | Normalization is NOT duplicated (single call site). Profile mutation duplication is real: `label_final` resolution logic and the JSON-write-back are each duplicated in two places. |
| 9 | `mentor/mentor.py` | intent routing / chat branch handling concentrated in one place | **Accurate** | `chat()` (301-417, 117 lines) is the hotspot. Missed by the original claim: `self._record_turn(...); self._record_turn(...); return out` appears **verbatim 7 times** — the single biggest mechanical win in the whole backlog. Also: `_is_graph_followup` here duplicates `MentorReAct._is_followup` cross-file (deferred, see Deferred section). |
| 10 | `mentor/mentor_graph.py` | repeated tool schemas and passthrough wrappers dominate the file | **Partially accurate** | They're 43% of the file, not dominant — the neighbor-search logic (78-285) is the largest chunk and is real logic. Real duplication: a schema property fragment repeated across 6 tool defs, and an `anchor_not_resolved` guard repeated in every graph method. |
| 11 | `mentor/mentor_react.py` | fallback dispatch and prompt wiring are the main complexity | **Accurate** | Confirmed. Missed by original claim: the 7-tool name list is declared independently in 3 places (`REACT_SYSTEM` docstring, `_execute` if-chain, `mentor_graph.TOOLS`) with no single source of truth (deferred, cross-file). |
| 12 | `mentor/mentor_anther.py` | anchor resolution complex-but-necessary; wrappers/lookup are the small targets | **Accurate** | Confirmed, and the original plan's "small targets" list is right. Missed: inside `_resolve_name`, the artist-centroid computation is duplicated near-verbatim for the exact-match path (112-122) and fuzzy-match path (151-165). |

## Ordered fix backlog

Ordered lowest-risk/highest-confidence first. Each item names the exact
extraction, its location, and its test safety net. One item = one pass = one
commit, in this order, so a regression is trivially bisectable.

**Tier 1 — mechanical, single-file, direct test coverage, near-zero behavior risk**

1. **`anther_ml/embedding.py`** — `embed_batch`'s try-block (lines 272-275) reimplements `get_embedding` (213-227). Make it call `get_embedding(model, processor, p, device, layer_aggregation, normalize)` instead. Safety net: `tests/test_embedding.py`.
2. **`anther_ml/corpus/labels.py`** — extract `_resolve_label_final(profile)` (used at line 211 and 272) and `_save_profiles(dir_path, profiles)` (used at 241-242 and 273-274). Safety net: `tests/test_corpus_labels.py`.
3. **`anther_ml/mpd_sql.py`** — extract a shared `_scan_quoted(s, i, stop_chars)` primitive used by both `_iter_tuples` (121-158) and `_split_fields` (161-193); they differ only in the terminator char set. Safety net: `tests/test_mpd_sql.py`.
4. **`mentor/mentor.py`** — extract `_respond(state, question, out, intent=None)` to collapse the 7 verbatim `_record_turn` + `_record_turn` + `return` sites (330-331, 339-340, 356-357, 360-361, 388-389, 401-402, 416-417). Safety net: `tests/test_mentor_context.py` — note this file covers the followup/intent-carry/reset helpers `chat()` calls, but not every branch of `chat()` itself; if any of the 7 sites has no covering test, add one assertion (call `chat()`, assert the two-turn history) for that branch before refactoring.

**Tier 2 — numeric/algorithmic or multi-call-site, still test-covered, moderate risk**

5. **`anther_ml/cluster.py`** — extract `_scale_and_pca(X, n_pca_components, random_state, whiten)` and `_fit_2d_umap(X, random_state, n_neighbors)`, shared by `fit_clusters` (31-106) and `fit_clusters_leiden` (262-338). Also hoist the 3x locally-reimported `Counter` to the module-level import. Safety net: `tests/test_cluster.py`. Because this is numeric, additionally diff the produced label/embedding arrays before/after on the same fixture input (not just pass/fail) — see Verification.
6. **`anther_ml/eval.py`** — have `_cosine_rankings` optionally return `sims` alongside `order` so `neighbor_audit` (line 234) reuses it instead of recomputing `embeddings @ embeddings.T`; generate `write_audit`'s HTML rows from the same fieldnames list already used for the CSV (247-266). Safety net: `tests/test_eval.py::test_neighbor_audit_export`.
7. **`mentor/mentor_graph.py`** — factor the repeated `{"anchor_a": {...}, "top_k": {...}}`-style schema property fragments (used across `sounds_like`/`compare`/`bridge`/`crossover`/`artist_tracks`/`tagmates`, 433-502) into shared dict fragments merged into each tool's schema. Extract the repeated `if vec is None: return {"tool":..., "ok": False, "error": "anchor_not_resolved", ...}` guard (90, 103-110, 131-138, 175, 210, 258) into one helper. Safety net: `tests/test_mentor_graph_tools.py`.
8. **`mentor/mentor_react.py`** — rewrite `_execute`'s if/elif chain (137-154) to dispatch via a `{tool_name: bound_method}` dict. Safety net: `tests/test_mentor_graph_tools.py::test_execute_dispatches_compare_and_artist_tracks` and `::test_execute_still_rejects_unsupported_map_mutating_tools`.
9. **`mentor/mentor_anther.py`** — extract `_artist_centroid(key, fuzzy=False)` from the two duplicated centroid-computation blocks inside `_resolve_name` (112-122 exact-match, 151-165 fuzzy-match). **No dedicated test file exists for this module.** Before refactoring: write a small characterization test exercising `resolve_anchor` through both the exact-key and fuzzy-key paths against a fixture index (reuse the fixture pattern in `tests/corpus_fixtures.py` if it fits), confirming identical output dicts before vs. after the extraction.

**Tier 3 — larger orchestration, higher blast radius, needs a characterization step first**

10. **`anther_ml/corpus/build.py`** — inside `build_corpus` (368-558): extract the two parallel embed loops (421-478: custom `embed_fn` path vs batched MERT path, both "skip if done → embed → checkpoint.add") into `_run_custom_embed(...)`/`_run_mert_embed(...)`, and extract the post-embedding tail (482-557: dedupe → index → leiden → profiles → manifest → save) into `_finalize_and_save(...)`. Target: shrink `build_corpus`'s body from 191 lines to ~40 lines of orchestration. Safety net: `tests/test_corpus_build.py` already has strong coverage (checkpoint resume, dedupe, end-to-end, failed-track skip) — run the full file before and after, not just a subset, since this function is the file's single largest and riskiest target.
11. **`anther_ml/corpus/sources.py`** — extract `_fetch_via_deezer(query, min_ratio, clip_seconds) -> np.ndarray | None`, shared by `mpd_source` (174-189) and `sql_source.fetch_one` (272-284); they differ only in miss-signaling (`continue` vs `wav = None`), so the helper must return `None` on miss and let each caller keep its own signaling. **No dedicated test file exists for `sources.py`.** Before refactoring: write characterization tests that mock `match_deezer_track`/`fetch_preview_waveform` and assert both call sites behave identically (miss path and hit path) before vs. after.

**Tier 4 — drop or optional, original claim did not hold**

12. **`anther_ml/spotify_deezer.py`** — no structural refactor warranted (see claim validation). If touched at all, the only defensible micro-change is a `_miss(q, reason)` helper for the two `misses.append({**q, "reason": ...})` call sites (~3 lines saved) — optional, skip unless doing a pass through this file for other reasons.

## Deferred (cross-file, out of scope for "small, module-local" passes)

Per the original constraint ("keep refactors file-local first; avoid
cross-package rewrites unless duplication is clearly repeated in multiple
places") — these are real findings but multi-file, so they're recorded here
rather than scheduled:
- `mentor/mentor.py:_is_graph_followup` duplicates `mentor/mentor_react.py:MentorReAct._is_followup` (near-identical marker lists, two files).
- The 7-tool name list is declared independently in three places: `mentor_react.py`'s `REACT_SYSTEM` docstring, its `_execute` if-chain, and `mentor_graph.py`'s `TOOLS` dict — no single source of truth. A real fix requires a shared constant imported by both modules.

## Verification (applies to every pass above)

1. Before editing a file, run its safety-net test file and confirm current
   baseline is green: `pytest tests/test_<name>.py -v`.
2. For the two files with no dedicated test file (`mentor_anther.py` item 9,
   `corpus/sources.py` item 11), write the characterization test **first**,
   confirm it passes against the *current* (pre-refactor) code, then refactor.
3. Make only the scoped extraction — no adjacent renames, no behavior changes,
   no touching code outside the named line ranges.
4. Re-run the same test file; it must pass identically (same tests, same
   assertions, no skips added).
5. For the two numeric/algorithmic items (5: `cluster.py`, 10: `corpus/build.py`),
   additionally run the relevant function on a fixed small fixture
   before and after the change and diff the numeric output (label array
   equality for `cluster.py`; embedding/index/profile equality for
   `corpus/build.py`) — unit test pass/fail alone doesn't guarantee bit-identical
   numeric output.
6. One commit per backlog item, in the Tier 1 → 2 → 3 order above, so any
   regression can be isolated to a single small diff.
