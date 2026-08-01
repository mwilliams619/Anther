# Plan — Running the Corpus Build Pipeline Efficiently

**For:** `anther-ml` corpus builder (`anther_ml/corpus/build.py`, `embedding.py`,
`sources.py`). **Executor:** Claude Code, on the local RTX 4080 SUPER (16 GB VRAM,
24 CPU, 47 GB RAM). **Goal:** make the 25k–100k reference-corpus build fast and
correct without a re-architecture, and document the genuine escape hatch to
millions.

Every recommendation is grounded in a single-cell / large-scale-retrieval
precedent, because the data-engineering problem the corpus builder faces —
embed once, freeze a coordinate system, map new items in cheaply — is the
same one single-cell genomics solved when it went from per-dataset analysis to
70-million-cell atlases. This plan is the **data-engineering** companion to the
existing historical review in `private/implemented_archive/SCRNA_METHODS_REVIEW.md`,
which already covered the clustering
*algorithm* (Leiden, cluster-the-embedding, standardize-before-cosine,
production-as-batch-effect).

---

## ✅ Completed (Tier 1 + SQL ingest + the 25k validation build)

Tier 1 is **done and validated**; its detailed section has been removed from
this plan (this changelog is the record). What shipped:

- **SQL ingest (`anther_ml/mpd_sql.py`).** Streams the 10 GB MySQL dump
  (`spotifydbdumpshare.sql`) → SQLite (13.3M tracks, one-time ~16 min), then
  samples deterministically (artist-capped, `--min-popularity`) inside SQLite.
  Wired as `--source sql` in the corpus CLI. `sample_tracks` returns 25k in ~26 s.
- **Tier 1A — batched fp16 GPU inference** (`embedding.py::embed_tracks_batched`).
  Equal-length-window bucketing so batching introduces no padding. Validated on
  real MERT: batched-fp32 vs per-track **cos = 1.000000** (invariant holds),
  fp16 vs fp32 **cos = 1.000000**, **~11.7× throughput** (0.51 → 0.044 s/track).
- **Tier 1B — parallel fetch + prefetch** (`sources.py::sql_source` 12-worker
  ordered pool; `build.py::_prefetch` producer→consumer). GPU runs at ~100 % util.
- **Tier 1C — append-only sharded checkpoint** (`build.py::BuildCheckpoint`).
  Per-flush `emb_<k>.npy`/`rows_<k>.jsonl` + atomic `shards.json` commit; peak RAM
  flat in unflushed vectors. Resume validated (crash-mid-run → no dup/loss).
- **25k validation corpus — built & validated.**
  `models/corpus_mpd_val_25k/` (24,965 tracks). Genre-free eval: **cluster
  stability ARI = 0.796**, 10 clusters / 0 % noise / 17 % largest, same-artist
  retrieval ~50× chance, and recognizable genres emerged with zero genre input
  (punk / hip-hop / jazz / electronic / classical / latin clusters).
- **Tier 2A — ANN dedupe** (`build.py::dedupe_near_identical`, `method="auto"`).
  pynndescent approximate-kNN replaces the O(N²) matrix above `_DEDUPE_EXACT_MAX`
  (30k); exact stays the reference below. Validated: ann == exact keep-mask on
  planted dupes and on 40 dupes injected into the real 25k vectors (0
  disagreements); ran on 100k in ~5 s where the exact matrix would be 40 GB.

> **Note for production:** the validation DB was built `--no-membership` (playlist
> tags aren't needed for the genre-free eval). Rebuild the SQLite DB *with*
> membership before the 100k build so playlist-fit (design §5) is available.

---

## Hard ceilings: what the dump can actually supply per popularity band

Measured on `data/mpd_dump/spotifydbdumpshare.sqlite` (2026-07-27). The
`popularity` column is brutally skewed — **12.7M of the 13.3M tracks sit at
popularity ≤ 30**, and popular music barely exists in this dump at all:

| band | tracks | with preview | **capped ceiling** (`artist_cap=5`) |
|---|---|---|---|
| 0–30 (long tail) | 12,675,853 | 8,830,594 | **2,435,985** |
| 31–70 (middle) | 603,324 | 471,819 | **207,425** (273,325 at cap 10) |
| 71–100 (popular) | 2,832 | 1,639 | **1,341** |

So any stratified request is capped hard: a nominally sensible 25/50/25 split at
600k tracks asks for 300k middle-band and 150k popular tracks that **do not
exist** — the popular ask is off by ~100×. `sample_tracks(require_preview=True)`
also filters *before* the Deezer fallback in `sources.py` ever runs, so the
fallback cannot recover a popular track that lacks a Spotify preview.

Consequences for anyone extending a corpus from SQL:

- **Pre-flight, always.** `mpd_sql.count_candidates()` gives the exact ceiling
  for a band/cap/exclusion set in seconds. `scripts/corpus_enrichment/extend_sql.py`
  calls it per stratum and prints a plan (`--plan-only` to stop there), clamping
  each stratum to what exists and spilling the shortfall into bands with
  headroom unless `--no-spill`. Before this check, exhaustion surfaced as a
  `RuntimeError` *hours* into a fetch, one stratum at a time.
- **Raising `--oversample-ratio` does not help** when the pool is DB-limited;
  the ratio only buys headroom against dead preview URLs (which run ~0.005% —
  199,102/199,111 fetched on the run that exposed this). `sql_source_oversampled`
  now distinguishes the two failures in its error message.
- **Pass `--exclude-checkpoint` for every prior checkpoint.** Omitting them
  re-embeds tracks already done: the 200k/600k/remaining checkpoints hold
  696,576 rows but only ~430k distinct tracks.
- Excluding tracks *frees artist-cap slots*, promoting previously-capped tracks
  of the same artist into the candidate set — so the remaining count after
  excluding N tracks is not simply `ceiling − N`.

---

## 0. The reframing that makes everything else cheap

The full dump is ~13.3M tracks; ~70% have a usable preview → ~9M embeddable →
**~200+ days on MPS / ~2 weeks on a GPU fleet.** That is a non-starter, and
chasing it is the wrong instinct.

The single-cell field's hard-won lesson is that the reference is a **prior over
the space, not a census of it** — a well-spread, deduped, artist-capped sample
defines the same coordinate system for a fraction of the cost, and the
"embed-once-map-many" reference paradigm (CELLxGENE Discover reference mapping)
means everything outside the sample is served by cheap projection, never
re-embedding. This is already the placement regime in `REFERENCE_CORPUS_DESIGN.md §2`.

**Decision that gates the whole plan:** target **25k for validation, 100k for
production**, sampled via the `--source sql` path (deterministic, artist-capped,
`--min-popularity`-biased). The millions-scale path is Section 3 — real, but
deferred, and only worth building if a specific product need appears.

---

## Tier 2 — Needed once you cross ~50–100k

**2A (ANN dedupe) is done** — see the changelog above. The remaining Tier-2 item
is optional even at 100k:

### 2B. (Optional at 100k) chunked, memory-mappable embedding store
**File:** new `bundle.py` storage path; `SongIndex`.

**Reality check:** 100k × 1024 float32 = **410 MB** — this fits in RAM, and exact
numpy cosine per query over 100k is fast (~400 MB scan). So a chunked store is
**not required at 100k**; the design doc's own note ("exact numpy cosine fine
under ~100k, no FAISS needed") holds. List this as a *readiness* item, not a
blocker: adopt it only when you commit to Tier 3.

**Precedent when you do:** TileDB-SOMA / CELLxGENE Census interact with data
**out-of-core via iterable streaming — queried and processed in fixed-size
chunks** — which is how they serve 70M cells to laptop RAM; scanpy's equivalent
is dask-backed AnnData with lazy chunked reads (`read_elem_lazy`). Store
embeddings in **zarr or TileDB** chunked arrays and stream them.

---

## Tier 3 — The millions-scale re-architecture (deferred; build only on demand)

Only if a product need forces embedding ≫100k. Each item swaps a whole-batch
step for an online/approximate one; none is a config change.

| Step | Whole-batch today | Millions-scale replacement | Precedent |
|---|---|---|---|
| **Query index** | `SongIndex` exact cosine over full matrix (54 GB/query at 13M) | FAISS **IVF-PQ** (or HNSW) ANN index; PCA→~100-D first | FAISS billion-scale cheat-sheet; IVF-PQ standard for ≥1M |
| **kNN graph for Leiden** | `sc.pp.neighbors` exact | approximate kNN (pynndescent — already scanpy's default; or LSH) | dropClust LSH graph; PARC/Secuer approximate-kNN to millions |
| **Whitening fit** | in-memory PCA over full matrix | `sklearn.IncrementalPCA.partial_fit` over chunks | Census incremental mean/variance; online algorithms |
| **Embedding store** | single `.npy`, all in RAM | sharded zarr/TileDB, memory-mapped | TileDB-SOMA out-of-core streaming |
| **Embedding compute** | single-GPU loop | shard track list → embed shards in parallel (multi-GPU/host) → concat; fit once | embarrassingly-parallel; atlas fan-out across samples |
| **Dedupe** | ANN blocked (Tier 2A) already covers this | same, over sharded store | dropClust LSH |

The key property that makes the fan-out trivial: **MERT embedding is
deterministic per track**, so embedding is embarrassingly parallel — only the
*fit* (whitening + graph + Leiden) is global, and it runs once on the assembled
matrix. Sharding is the highest-leverage Tier 3 item if a second GPU/host ever
appears. (The append-only sharded checkpoint from Tier 1C is already the on-disk
shape this wants.)

---

## Remaining execution order (for Claude Code)

1. **Rebuild the SQLite DB with membership** (`python -m anther_ml.mpd_sql
   --dump …` *without* `--no-membership`), so the 100k corpus supports
   playlist-fit (design §5).
2. **Build 100k** (`--source sql --sample-n 100000`); auto-dispatch already uses
   the Tier-2A ANN dedupe above 30k. Re-run the genre-free eval; compare cluster
   stability (ARI) vs. the 25k corpus to confirm the coordinate system has
   stabilized (design §3B — diminishing returns past the point where shape
   stabilizes).
3. Stop. Tier 2B / Tier 3 only if a concrete need to exceed 100k appears.

## Things to preserve (already correct — don't regress)
- **Recipe invariance corpus↔query.** `plan_windows`, layer aggregation,
  loudnorm, 24 kHz, and the `embedding_config` stamp must stay byte-identical
  across corpus and query (`docs/invariants.md`). Batching must not change the
  per-track result — enforced by equal-length-window bucketing and locked by
  `test_embed_tracks_batched_matches_per_track`.
- **Artist-cap before embedding** (`sources.py::_cap_by_artist`, and the SQL
  window-function cap in `mpd_sql.sample_tracks`) — capped tracks cost zero GPU.
- **Stream one waveform at a time** for network sources — don't materialize all
  previews in RAM. The prefetch queue and fetch window are *bounded* for this reason.
- **Resumable checkpoint** — the sharded checkpoint keeps crash-safe resume; the
  commit point is the atomic `shards.json` rewrite.
- **Fit once, freeze, map many** — never re-fit the transform per upload
  (`place.py` already respects this).

---

## Literature grounding (retrieved this session)

**Single-cell / atlas data-engineering**
- CELLxGENE Discover / Census (Nucleic Acids Research 2025) — reference mapping
  ("embed once, map many"); TileDB-SOMA out-of-core iterable streaming in
  fixed-size chunks; incremental mean/variance; SOMA/Census PyTorch data loaders.
- scanpy (scverse) — scales to >100M cells; dask-backed out-of-core AnnData,
  `read_elem_lazy`; approximate kNN (pynndescent) as the default neighbor graph.
- rapids-singlecell (sc-best-practices) — 50k–1M single-GPU sweet spot.

**Approximate neighbors / scalable clustering (the dedupe + graph precedents)**
- dropClust (bioRxiv) — **LSH for logarithmic-time approximate neighborhoods**,
  then graph community detection. Direct precedent for Tier 2A.
- PARC (bioRxiv), Secuer (PLoS Comput Biol) — approximate-kNN graph clustering to
  millions of cells; anchor-based bipartite graphs.
- minicore (bioRxiv) — mini-batch k-means clustering a 4M-cell dataset in minutes
  under 10 GiB RAM — the online-algorithm discipline.

**ANN indexing (Tier 3 query index)**
- FAISS billion-scale ANN cheat-sheet — IVF-PQ / HNSW selection; PCA to ~100-D
  first; PQ compression math.
- ParlayANN (arXiv 2305.04359) — graph-based ANN reaches >0.9 recall at
  billion-scale where IVF struggles; informs HNSW-over-IVF at high recall.

_Clustering-algorithm precedents (Leiden, cluster-the-embedding, batch-effect
correction) are in the historical `private/implemented_archive/SCRNA_METHODS_REVIEW.md`
and not repeated here._
