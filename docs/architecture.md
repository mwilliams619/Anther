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
| `mpd_ingest.py` | Spotify Million Playlist Dataset → corpus queries (`build_mpd_queries`, `enrich_isrc`, `ingest_mpd_corpus`) |
| `spotify_deezer.py` | Spotify metadata → Deezer preview audio (`load_spotify_via_deezer`) |

> A `corpus/` subpackage is under construction and intentionally undocumented for now.

## Top-level scripts & `ui/`

| Script | Responsibility |
|---|---|
| `export_viz.py` | Bakes an index + 2D embedding into the standalone `song_view.html` d3 map (`PHASE` set at top; re-run after any index rebuild) — see [notebooks.md](notebooks.md) |
| `ui/app.py` | Flask backend for the song staging + clustering UI |
| `ui/jobs.py` | Background cluster-job runner (one job at a time; MERT loaded once) |

## Tests

`tests/test_{audio,cluster,data,embedding,eval,features,similarity}.py`, one per
core module. Run with `pytest`.
