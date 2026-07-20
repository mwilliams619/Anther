# Anther Project Status Review — July 17-18 Breakage (RESOLVED July 20)

## Current State

**Branch:** `graph-dual-mode-revert` (HEAD at commit 64418058)
**Uncommitted changes:** Deleted `.claude` agent specs (not critical, pre-existing),
  one-line fix in `ui/app.py` (see below), restored `data/billboard/`
**Environment:** `.venv` at project root, all deps installed, registered
  as Claude Science env `anther-dev`

---

## Root Cause of App Breakage (found + fixed)

Commit `281323ee` ("Revert dual-mode D3 changes ... - kept artist backend")
changed `ui/app.py`'s import from `import atlas` to `from . import atlas`.
This is unrelated to the D3/artist revert it was bundled with — `ui/` has
no `__init__.py` and the module is meant to run as a standalone script
(`python ui/app.py`, documented in README.md/docs/ui.md). The relative
import crashes on startup with
`ImportError: attempted relative import with no known parent package`,
which is why **the whole app failed to boot**, not just artist mode.

**Fix applied:** reverted to `import atlas` in `ui/app.py` (1-line diff).
Confirmed the Flask app now imports cleanly, all routes register
(including `/api/artist/*` — the backend endpoints were never actually
removed, only the frontend D3 dual-mode toggle was reverted), and the dev
server boots and serves real search/place/song-detail requests against
the restored corpus.

The Billboard data loss (`data/billboard/` accidentally deleted) was a
separate, second problem — see Feature 3 below.

---

## Three Features: Implementation Status

### Feature 1: Merit Granular Score + Aggregated Similarity Score ✓ Code Present

**What was added:**
- `anther_ml/merit.py` — Factor projection heads (melody, rhythm, timbre)
- `anther_ml/corpus/merit_index.py` — MERIT-aggregate index building
- `ui/atlas.py` — Merit integration: `_merit_query_vec()`, `_merit_breakdown()`, merit link thresholds
- New corpus: `models/corpus_mpd_100k_merit_ext_billboard` (MERIT + Billboard)
- Imports added: `embed_query_dual`, `link_calibration`, new similarity functions

**Status:** ✅ VERIFIED WORKING
- `pytest tests/test_merit.py tests/test_corpus_merit_index.py` → 14/14 passed
- Full corpus/embedding/similarity suite (69 tests across `test_corpus_build.py`,
  `test_corpus_bundle.py`, `test_corpus_place.py`, `test_corpus_recommend.py`,
  `test_corpus_sources.py`, `test_embedding.py`, `test_similarity.py`) → all passed
- Live smoke test: `GET /api/song/<id>?expand=true` returns a `similar` list
  where each neighbor carries a `breakdown` with distinct `melody`/`rhythm`/
  `timbre`/`aggregate` scores — confirmed against the running dev server

---

### Feature 2: Musician Search & Graph UI ⚠️ PARTIALLY REVERTED

**What was done:**
- Backend: `ui/atlas.py` — Artist clustering loaded, search functions, persistence
  - `_load_artist_clustering()` — loads from `data/artist_clustering/`
  - `persist_new_artist()` — saves new artists to disk
  - `_nearest_artists()` — kNN search in artist space
  - Artist data: 2,164 artists with embeddings/labels/2D coords
  
- Frontend: **REVERTED** (commit 281323ee "Revert dual-mode D3 changes")
  - Removed: `ui/static/graph.js` dual-mode rendering
  - Removed: `ui/static/app.js` mode toggle (Song/Artist)
  - Removed: `ui/static/index.html` artist search UI
  - Removed: `/api/artist/*` endpoints from Flask

**What's left:**
- Backend artist data structures are still in `atlas.py` but not exposed via Flask
- Artist functions exist but are orphaned — no endpoints to call them
- Frontend can't toggle artist mode because the buttons were removed

**Status:** ⚠️ DEFERRED TO NEW BRANCH (per user decision — more complex, out
of scope for this fix pass)
**Note:** Backend endpoints (`/api/artist/search`, `/api/artist/<id>`,
`/api/artist/create`, `/api/artist/graph`) ARE still registered in the
running Flask app — they were never removed, contrary to this doc's
original read. Only the frontend D3 dual-mode toggle was reverted.
Also noted: `artist_meta` has 2,165 entries vs `artist_embeddings` shape
(2163, 1024) — a pre-existing shape mismatch to resolve on the artist branch.

---

### Feature 3: Pop Music (Billboard) Expansion ✓ Code Present, Data Unclear

**What was added:**
- `data/billboard/` — Popularity data (all_hot100.json)
- `scripts/billboard_expansion/` — 4-stage pipeline:
  1. `01_build_popularity_table.py` — fetch Billboard Hot 100 metadata
  2. `02_match_against_corpus.py` — fuzzy-match to existing corpus tracks
  3. `03_fetch_and_embed_missing.py` — MERT-embed unmatched songs
  4. `04_append_to_index.py` — add to corpus index
  5. `run_pipeline.py` — orchestrator

- `ui/atlas.py` additions:
  - `POPULARITY_PATH` config (defaults to `data/billboard/popularity_by_track_id.json`)
  - `POPULARITY_BETA` weight (0.15)
  - `_popularity_pct` global (track_id → percentile)
  - Load at boot: `load_popularity_by_track_id()`, `popularity_percentiles()`

**Corpus state:**
- Name: `corpus_mpd_100k_merit_ext_billboard` 
- Size: 100k base + Billboard hits
- Last modified: Jul 19 00:01
- Has MERIT index? Yes (implied by name `merit_ext_*`)

**Status:** ✅ RESTORED AND VERIFIED
- `data/billboard/` was accidentally deleted; corpus model itself
  (`models/corpus_mpd_100k_merit_ext_billboard/`) was untouched — its
  manifest already records the Billboard extension (7,654 tracks added
  from source `billboard_hot100` on 2026-07-18, 9 duplicates dropped)
- Rebuilt via pipeline steps 1–2 (`01_build_popularity_table.py`,
  `02_match_against_corpus.py`): re-downloaded Billboard Hot 100 history
  (10,249 unique songs, last 20 years), matched 7,735/10,249 (75.5%)
  against the corpus by normalized (artist, title), wrote
  `popularity_by_track_id.json` (7,735 entries) and `popularity.json`
  (10,249 entries)
- Live smoke test: `atlas.load()` loads the popularity sidecar
  successfully (`[atlas] loaded popularity sidecar: 7735 tracks ->
  percentiles`) and `GET /api/search?q=blinding lights` returns a
  Billboard-sourced track (`id: billboard:deezer:908604612`)

---

## Environment Status

**Python:** 3.11.15, project's own `.venv` at repo root (already existed,
just wasn't registered/activated)
**Dependencies:** ✅ ALL INSTALLED — flask 3.1.3, torch 2.6.0+cu124,
transformers 5.13.0, librosa 0.11.0, numpy 2.4.6, pandas 2.3.3,
scikit-learn 1.9.0, umap-learn 0.5.12, leidenalg 0.12.0, igraph 1.0.0,
requests 2.34.2, pytest 9.1.1 — confirmed via editable install
(`anther-ml`, Location: `.venv/lib/python3.11/site-packages`)

**Data dependencies:**
- ✓ Artist clustering dir exists: `data/artist_clustering/` (2,165 artists)
- ✅ Billboard data restored: `data/billboard/` (see Feature 3 above)
- ✓ MERT/MERIT heads present: `models/merit_heads/head_{mel,rhy,tim}/best_head.pt`
- ✓ MPD database present: `data/mpd_dump/spotifydbdumpshare.sqlite`

---

## Root Cause Summary (both fixed)

1. **App-wide startup crash:** `ui/app.py` line 40 changed from
   `import atlas` to `from . import atlas` in commit `281323ee` — broke
   `python ui/app.py` for every feature, not just artist mode. **Fixed.**
2. **Billboard data loss:** `data/billboard/` was accidentally deleted
   (user action, unrelated to any commit). The corpus model itself was
   never affected. **Rebuilt from source via the existing pipeline scripts.**

Merit and Billboard are both confirmed working end-to-end against the
live dev server. Artist mode is deferred to a new branch per your
instruction — its backend is intact (not orphaned, as originally assessed),
only the frontend D3 dual-mode toggle remains reverted.

---

## Remaining Pre-Existing Issues (out of scope, NOT touched)

These test failures predate the Merit/Billboard/Artist work and were
traced via `git log` to unrelated earlier commits — left alone:
- 24 `AttributeError: module 'atlas' has no attribute '_graph'` errors in
  `test_atlas_graph.py`/`test_atlas_playlist.py`/`test_atlas_search.py` —
  stale tests referencing a module-level `_graph` global removed in commit
  `f4ef8bcb` (session-based state refactor, predates Merit/Billboard)
- 2 failures in `test_features.py` (Phase 1 librosa features) — predate
  Merit/MERT entirely, trace to the first commit and `5b125fb5`
- 1 failure in `test_mentor_live_graph.py` — mentor refactor
  (commit `c03c5176`), unrelated to Merit/Billboard/Artist
- `artist_meta` (2,165 entries) vs `artist_embeddings` shape (2163, 1024)
  mismatch — pre-existing, to resolve on the artist-mode branch

---

## Next Steps

### Ready now:
1. Review and commit the `ui/app.py` fix + restored `data/billboard/` (see
   below for suggested commit message)
2. Start a new branch off this one for the artist-mode UI work

### For the artist-mode branch (future):
1. Restore D3 dual-mode toggle UI (`graph.js` `rerenderMode`, `app.js`
   `setViewMode`) — backend endpoints already work, just need the frontend
2. Resolve the `artist_meta`/`artist_embeddings` count mismatch (2,165 vs 2,163)
3. Wire artist search bar in `index.html`

### Optional cleanup (unrelated to the three features, no action needed unless desired):
1. The 24 stale `_graph` test failures could be updated to match the
   current per-session state model
2. `test_features.py`/`test_mentor_live_graph.py` failures are separate,
   pre-existing issues

---

## Update: third Merit calibration bug — top-level score could exceed all 3 factors (fixed)

**Symptom** (user-reported): the single top-level similarity score shown on a similar-song
row sometimes read *higher* than every one of the melody/rhythm/timbre bars in the expanded
breakdown underneath it — e.g. a pair where all three factors read 0 but the aggregate showed
12.7. Measured empirically: **34.5% of 4,000 random corpus pairs** had `breakdown.aggregate >
max(melody, rhythm, timbre)`.

**Root cause.** This was a continuation of the same scale-mismatch family as the timbre-100 bug,
but structural rather than incidental. The aggregate's raw value is the *mean* of the three
factor cosines — averaging compresses variance, so the aggregate's own raw-cosine distribution
across random pairs is narrower than any individual factor's. Its own independently-calibrated
floor/ceiling (from `calibrate_link_thresholds` on the whole 384-d concatenated vector) sit at
different points than the per-factor calibrations. So `display(mean(raw_i))` had no guaranteed
relationship to `display(raw_i)` per factor, even though `mean(raw_i)` itself always sits between
the raw cosines. There were also four separate call sites computing a top-level "score"
independently of the breakdown (`_merge_fragment`'s edge scoring, `_map_neighbors`'s click-panel
default view, and both branches of `song_detail`'s expanded similar list) — each could diverge
from `breakdown.aggregate` in its own way.

**Fix** (user's explicit choice: plain unweighted mean, not weighted, not min-based):
- `ui/atlas.py` — `_breakdown_scores`'s `"aggregate"` key is no longer computed against its own
  calibrated scale. It's now defined as the plain mean of the three *already-calibrated*
  melody/rhythm/timbre display scores: `round((melody + rhythm + timbre) / 3.0, 1)`. This
  mathematically guarantees aggregate always falls within `[min, max]` of the three factors.
- New helper `_merit_aggregate_score(agg_a, agg_b)` wraps `_breakdown_scores(...)["aggregate"]`
  as the single source of truth for a bare top-level score.
- Updated all four call sites that show a bare score to derive it from the breakdown whenever
  one is available (falling back to the old raw-cosine `_display_score` path only when a
  breakdown can't be computed — e.g. pre-MERIT cached vectors):
  - `_merge_fragment`'s MERIT-space edge scoring (drives both the drawn link's persisted `score`
    and the D3 tooltip)
  - `_map_neighbors` (the click-panel's default "songs actually connected on the map" view)
  - `song_detail`'s corpus-track branch (expanded similar-list)
  - `song_detail`'s non-corpus query-node branch (expanded similar-list)

**Verification.**
- 4,000 random corpus pairs (seed=7): `aggregate > max(factor)` count dropped from 1,382 (34.5%)
  to **0**; `aggregate < min(factor)` also **0**. Aggregate matches the plain mean exactly in
  every sampled case.
- Full merit/corpus test suite: 69 passed, no failures (unchanged).
- Full project suite: 270 passed, 3 failed, 24 errors — identical to the established
  pre-existing baseline, no new regressions.
- Live HTTP verification via `/api/song/<id>?expand=true` on "Timeless" (The Weeknd & Playboi
  Carti): all 6 top neighbors now show `score == breakdown.aggregate` exactly, always ≤ the max
  factor bar (e.g. "She Moves (Far Away) - Club Mix" score=68.0, bd melody=86.6/rhythm=86.7/
  timbre=30.7 — aggregate correctly pulled down by the weak timbre match).
- Live HTTP verification of the drawn-edge path: placed "Timeless" then "Luther" via
  `/api/place`, confirmed the returned edge's `score` (33.8) exactly matches
  `_merit_aggregate_score` for that pair offline, and `/api/song/<id>` (non-expanded,
  `map_neighbors` path) shows the identical score/breakdown for the same edge.

Files modified (uncommitted, same pattern as prior fixes — left for review/commit together):
`ui/atlas.py` only for this fix (the per-factor-calibration fix from the prior update already
touched `calibration.py`/`merit_index.py`/`bundle.py`).

## Update: second Merit calibration bug — per-factor scale mismatch (fixed)

After the missing-sidecar fix above, the breakdown panel populated but
**timbre pinned near 100 far more often than melody/rhythm**. Root cause:
`_breakdown_scores` mapped all three factors (melody/rhythm/timbre) through
the *same* calibrated scale — one derived from the **aggregate** vector
(mean of the three factor cosines). But each factor has its own raw-cosine
distribution over random corpus pairs, and they differ a lot on this corpus:

| factor    | mean   | p95    | max    |
|-----------|--------|--------|--------|
| melody    | 0.4075 | 0.7170 | 0.9145 |
| rhythm    | 0.2410 | 0.6660 | 0.9451 |
| timbre    | 0.4134 | 0.9483 | 0.9942 |
| aggregate | 0.3540 | 0.6378 | 0.8347 |

Timbre's p95 (0.948) sits almost exactly at the aggregate-calibrated
ceiling (0.952), so a huge fraction of ordinary pairs displayed as
near-100 on timbre alone, while melody/rhythm (whose distributions never
get that high) spread out normally on the same scale.

**Fix** (source-level, so future corpus builds don't hit this again):
- `anther_ml/calibration.py`: added `calibrate_factor_link_thresholds()`,
  `save_factor_calibration()`, `load_factor_calibration()`, and the
  `link_calibration_merit_factors.json` sidecar filename constant —
  calibrates melody/rhythm/timbre independently off the same 200k-pair
  draw (seed=0), rather than reusing the aggregate's scale.
- `anther_ml/corpus/merit_index.py`: `build_merit_aggregate_index` now
  calls the new per-factor calibration and persists the sidecar
  alongside the existing aggregate one, so this is automatic on future
  builds.
- `anther_ml/corpus/bundle.py`: added `ReferenceCorpus.merit_factor_calibration`
  lazy-load property (mirrors `merit_calibration`).
- `ui/atlas.py`: added `_merit_factor_thresholds` global; `_calibrate_qq_threshold`
  now loads it (with an in-process recalibration fallback if the sidecar
  is missing, generalizing the fix from the first bug so a missing sidecar
  self-heals rather than silently degrading); `_breakdown_scores` now looks
  up each factor's own thresholds instead of sharing `_merit_link_thresholds`
  for all three (the "aggregate" key still uses the aggregate scale).

**Verified**: regenerated `link_calibration_merit_factors.json` for the
current corpus; on 3,000 random pairs, 0% now hit the 100 ceiling on any
factor (previously frequent on timbre). On real near-neighbors of "Timeless"
(The Weeknd & Playboi Carti), timbre now reads 61.0/76.7/30.7/78.2/83.0/59.3
across its top 6 neighbors instead of pinning near 100 — melody and rhythm
show similarly independent spread. Confirmed live via
`/api/song/<id>?expand=true`. Full test suite re-run: still 270 passed, 3
failed, 24 errors — same pre-existing/out-of-scope failures as before,
no new regressions.

**Files changed** (all on disk, not yet committed): `anther_ml/calibration.py`,
`anther_ml/corpus/merit_index.py`, `anther_ml/corpus/bundle.py`, `ui/atlas.py`.
