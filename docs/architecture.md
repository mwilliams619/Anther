# Architecture & script index

Two-phase music similarity pipeline. Both phases share the same `anther_ml`
clustering and similarity API — only the input embeddings differ.

**Phase 1 — Pre-computed librosa features (fast, no GPU)**
- Input: FMA `features.csv` (518-dim, 3-level MultiIndex header — always load via `load_fma_features()`)
- Clustering (primary): `StandardScaler → PCA(100d) → k-NN graph → Leiden → 2D-UMAP (viz only)` via `fit_clusters_leiden`
- Query path for a personal song: `extract_librosa_features(path)` (labeled Series) → `align_to_corpus()` → `assign_cluster_knn()`

**Phase 2 — MERT neural embeddings (GPU recommended)**
- Model: `m-a-p/MERT-v1-330M` (~1.3 GB HuggingFace download on first use)
- Same clustering API, 1024-dim input. All-layer + multi-window pooling, optional loudness normalization. Embedding 8k FMA tracks takes ~2–3 hrs on Apple MPS.
- Query path: `get_embedding(model, processor, path, device)` → `assign_cluster_knn()`

## `anther_ml/` — core similarity/clustering pipeline

| Module | Responsibility | Details |
|---|---|---|
| `data.py` | FMA CSV loading; ordered-categorical subset filter | [phase1-features.md](phase1-features.md) |
| `features.py` | Phase 1 librosa features → labeled `pandas.Series`; loudness-feature weighting | [phase1-features.md](phase1-features.md) |
| `embedding.py` | Phase 2 MERT inference at 24000 Hz — all-layer + multi-window | [phase2-embeddings.md](phase2-embeddings.md) |
| `audio.py` | Audio loading + EBU R128 loudness normalization | [phase2-embeddings.md](phase2-embeddings.md) |
| `cluster.py` | Leiden clustering + diagnostics + k-NN assign (legacy HDBSCAN kept for comparison) | [clustering.md](clustering.md) |
| `similarity.py` | `SongIndex` — standardized cosine search, versioned index | [similarity.md](similarity.md) |
| `eval.py` | Genre-free eval harness (`python -m anther_ml.eval`) | [evaluation.md](evaluation.md) |

## `anther_ml/` — corpus-building tools (not part of the two-phase pipeline)

| Module | Responsibility |
|---|---|
| `mpd_sql.py` | **MySQL MPD dump** (`spotifydbdumpshare.sql`) → SQLite → deterministic artist-capped sample (`load_dump_to_sqlite`, `ensure_db`, `sample_tracks`). Streaming, resumable, logged. This is the supported path for our dataset. |
| `mpd_ingest.py` | Legacy: RecSys `mpd.slice.*.json` files → corpus queries (`build_mpd_queries`, `enrich_isrc`). Kept for JSON-format MPD downloads |
| `spotify_deezer.py` | Spotify metadata → Deezer preview audio (`load_spotify_via_deezer`); preview fetch/decode reused by both MPD paths |

The corpus `sql_source` (in `corpus/sources.py`) reads the SQL dump via `mpd_sql`
and fetches audio from each track's Spotify `preview_url` directly (Deezer
fallback for dead URLs). Build the DB once, then sample many times:

```bash
python -m anther_ml.mpd_sql --dump data/mpd_dump/spotifydbdumpshare.sql   # one-time
python -m anther_ml.corpus build --source sql \
    --sql-dump data/mpd_dump/spotifydbdumpshare.sql --name mpd_25k --sample-n 25000
```

The full dump is ~13.3M tracks — far past the design's 10k–100k range — so `sql_source`
always *samples down* (artist-capped, deduped post-embed); it is never used to embed
the whole dump.

## `anther_ml/corpus/` — frozen reference-corpus bundles

A MERT-space reference corpus that new songs are *placed onto*, never re-clustered
from scratch. Design rationale in [REFERENCE_CORPUS_DESIGN.md](../REFERENCE_CORPUS_DESIGN.md).
CLI: `python -m anther_ml.corpus build …` / `… place song.mp3 …`.

| Module | Responsibility |
|---|---|
| `sources.py` | Track sources under one item contract (`fma_source`, `local_source`, `mpd_source`) |
| `build.py` | `build_corpus` — embed (checkpointed/resumable) → dedupe → fit `SongIndex` + Leiden → per-cluster profiles → freeze bundle |
| `bundle.py` | `ReferenceCorpus` — the frozen, versioned bundle (`save`/`load`, config stamp) |
| `place.py` | Placement regime — `place`, `embed_query`, playlist-fit / `rank_playlists` |
| `__main__.py` | `build` / `place` CLI |

Bundles are written to `models/corpus_<name>/`. Tests: `tests/test_corpus_{build,bundle,place}.py`.

## Top-level scripts & `ui/`

| Script | Responsibility |
|---|---|
| `export_viz.py` | Bakes an index + 2D embedding into the standalone `song_view.html` d3 map (`PHASE` set at top; re-run after any index rebuild) — see [notebooks.md](notebooks.md) |
| `ui/app.py` | Flask backend for the song staging + clustering UI |
| `ui/jobs.py` | Background cluster-job runner (one job at a time; MERT loaded once) |

## Tests

`tests/test_{audio,cluster,data,embedding,eval,features,similarity}.py`, one per
core module. Run with `pytest`.
