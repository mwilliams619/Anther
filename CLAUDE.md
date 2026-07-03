# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
source venv/bin/activate
pip install -e .          # editable install (pyproject.toml) — preferred
pip install -e '.[dev]'   # + pytest
```

The package is now an editable install, so `import anther_ml` works from anywhere and the `sys.path.insert(0, '..')` hack in the notebooks is no longer required (harmless if left). Run the test suite with `pytest` from the repo root.

## Architecture

Two-phase music similarity pipeline. Both phases share the same `anther_ml` clustering and similarity API — only the input embeddings differ.

**Phase 1 — Pre-computed librosa features (fast, no GPU)**
- Input: FMA `features.csv` (518-dim, 3-level MultiIndex header — always load via `load_fma_features()`)
- Clustering (primary): `StandardScaler → PCA(100d) → k-NN graph → Leiden → 2D-UMAP (viz only)` via `fit_clusters_leiden`
- Query path for a personal song: `extract_librosa_features(path)` (labeled Series) → `align_to_corpus()` → `assign_cluster_knn()`

**Phase 2 — MERT neural embeddings (GPU recommended)**
- Model: `m-a-p/MERT-v1-330M` (~1.3 GB HuggingFace download on first use)
- Same clustering API, 1024-dim input. All-layer + multi-window pooling, optional loudness normalization. Embedding 8k FMA tracks takes ~2-3 hrs on Apple MPS.
- Query path: `get_embedding(model, processor, path, device)` → `assign_cluster_knn()`

## Script index

**`anther_ml/` — core similarity/clustering pipeline**

| Module | Responsibility |
|---|---|
| `data.py` | FMA CSV loading (`load_fma_tracks`, `load_fma_features`, `get_audio_path`); `load_fma_features` uses an ordered-categorical subset filter |
| `features.py` | Phase 1 feature extraction — returns a labeled `pandas.Series` on FMA's 518-dim MultiIndex; loudness feature weighting (`drop_loudness_features`) |
| `embedding.py` | Phase 2 MERT loading and inference at 24000 Hz — all-layer + multi-window, `embedding_config` |
| `audio.py` | Audio loading + EBU R128 loudness normalization (`loudness_normalize`, `load_audio`) |
| `cluster.py` | Leiden path (`fit_clusters_leiden`, `leiden_labels`, `assign_cluster_knn`, `cluster_diagnostics`, `resolution_sweep`); legacy `fit_clusters`/`assign_cluster` kept for comparison |
| `similarity.py` | `SongIndex` — standardize + L2 + cosine search; versioned, self-describing index |
| `eval.py` | Genre-free eval harness (`python -m anther_ml.eval`) |

**`anther_ml/` — corpus-building tools (not part of the two-phase pipeline)**

| Module | Responsibility |
|---|---|
| `mpd_ingest.py` | Spotify Million Playlist Dataset → corpus queries (`build_mpd_queries`, `enrich_isrc`, `ingest_mpd_corpus`) |
| `spotify_deezer.py` | Spotify metadata → Deezer preview audio (`load_spotify_via_deezer`) |

**Top-level scripts & `ui/`**

| Script | Responsibility |
|---|---|
| `export_viz.py` | Bakes an index + 2D embedding into the standalone `song_view.html` d3 map (`PHASE` set at top; re-run after any index rebuild) |
| `ui/app.py` | Flask backend for the song staging + clustering UI |
| `ui/jobs.py` | Background cluster-job runner (one job at a time; MERT loaded once) |

**Tests** — `tests/test_{audio,cluster,data,embedding,eval,features,similarity}.py`, one per core module. Run with `pytest`.

## Clustering API signatures

**Primary path — Leiden (`fit_clusters_leiden`) returns a dict**, not a tuple:

```python
out = fit_clusters_leiden(X, resolution=1.0, n_pca_components=100, standardize=True)
# keys: labels, embedding_2d, scaler, pca, reducer_2d, clustering_space,
#       n_neighbors, resolution, metric, diagnostics
save_leiden('models/pipeline_phase1.pkl', out)
out = load_leiden('models/pipeline_phase1.pkl')

# Assign a new song by k-NN vote against the labeled corpus (same scale/PCA):
cluster_id, confidence = assign_cluster_knn(
    vec, out['clustering_space'], out['labels'], scaler=out['scaler'], pca=out['pca'])
```

`resolution` is the one interpretable knob (low → few broad clusters, high → many fine); sweep it with `resolution_sweep` and pick via the eval harness. `cluster_diagnostics` prints n_clusters / noise% / largest-share on every fit and warns if any cluster exceeds 60% of tracks.

**Legacy path — HDBSCAN-on-UMAP (`fit_clusters`)** is retained only for comparison and returns 7 values (`labels, embedding_2d, scaler, pca, reducer, clusterer, reducer_2d`); `load_pipeline` returns 5. It clusters the UMAP projection, which is the wrong space (see invariants) — do not use it for new work.

`SongIndex` saves and loads as a pair — always omit the extension:
```python
# standardize=True is the Phase-1 default (fixes scale domination); pass a
# config dict so the index is self-describing and incompatible indices are caught.
index = SongIndex(X, metadata, standardize=True, config=embedding_config(...))
index.save('models/index_phase1')      # writes .npy + .json (dict schema, versioned)
SongIndex.load('models/index_phase1')  # reads both; still loads legacy v1 (bare-list json)
```

## Notebook pipeline order

Each notebook saves artifacts the next one loads:

| Notebook | Reads | Writes |
|---|---|---|
| `01_explore_fma` | `data/fma_metadata/` | `models/fma_small_features.pkl` |
| `02_cluster_precomputed` | `fma_small_features.pkl` | `models/pipeline_phase1.pkl`, `index_phase1.*`, `embedding_2d_phase1.npy`, `labels_phase1.npy` |
| `03_spectrogram_extraction` | personal MP3s | visualization only (educational) |
| `04_embedding_mert` | personal MP3s + optionally `data/audio/fma_small/` | `models/pipeline_phase2.pkl`, `index_phase2.*`, `embedding_2d_phase2.npy`, `labels_phase2.npy` |
| `05_test_new_song` | pipeline + index (either phase) | `models/test_song_phase{N}.png` |

`05_test_new_song` is the production-like entry point — it supports both phases via a `PHASE = 1 | 2` toggle.

> **Status:** notebooks `01`, `02`, `04`, `05` are migrated to the `fit_clusters_leiden` + `save_leiden`/`load_leiden` + standardized/config-tagged `SongIndex` + genre-free-eval stack. `03` is educational (spectrograms) and unchanged. The on-disk indices have been rebuilt in v2 format: Phase 1 = 8,000 fma_small tracks (standardized), Phase 2 = 107 personal tracks (all-layer / multi-window / loudness-normalized). Re-run a notebook to regenerate its artifacts, then re-run `export_viz.py` to refresh `song_view.html`.

## Visualization

`song_view.html` is a self-contained d3 map generated by `export_viz.py` (`PHASE = 2` hardcoded at the top). It bakes the index + 2D embedding in at export time, so **re-run `python export_viz.py` after rebuilding an index** — it does not read `models/` live. Open the resulting file in a real browser. The separate `ui/` Flask app builds its own per-session index and picks up code changes automatically.

## Key invariants

**Cluster the embedding, never the UMAP/t-SNE coordinates.** Leiden builds its k-NN graph on the PCA/whitened embedding (`clustering_space`); 2D UMAP is strictly for the picture. 2D projections distort density and inter-cluster distance — clustering them fuses or splits groups that aren't. (The legacy `fit_clusters` violated this by clustering the 32D UMAP output; that's why it's comparison-only.)

**Standardize before cosine similarity (Phase 1).** Raw librosa feature families span a ~8,000× magnitude range, so row-L2+cosine alone is dominated by Hz-scale spectral dims. `SongIndex(standardize=True)` z-scores columns (stats fit on the corpus, persisted with the index, applied identically to queries) before L2. Default True for Phase-1; evaluate for Phase-2.

**Feature ordering is explicit, not positional.** `extract_librosa_features` returns a labeled `pandas.Series` on FMA's `(feature, statistic, number)` MultiIndex; always `align_to_corpus(query, corpus.columns)` before comparing — it asserts the axes match. A silent reorder (same length, wrong mapping) is what made Phase-1 uploads meaningless.

**Never introduce genre** as a clustering input, training signal, or eval metric. It may be displayed as metadata only. Quality is judged by the genre-free eval harness.

**FMA audio path convention**: `audio_dir/AAA/AAAAAA.mp3` — use `get_audio_path(dir, track_id)`, don't construct manually.

**Sample rates**: Phase 1 matches FMA's recipe — `extract_librosa_features` loads at native sample rate (`sr=None`) over the full track, since FMA's `features.csv` was computed that way; passing `sr=22050` would introduce param drift. Phase 2 uses 24000 Hz (MERT requirement). The `SR` constant in `embedding.py`/`audio.py` encodes the Phase-2 rate.

## Evaluation (genre-free)

`python -m anther_ml.eval --index models/index_phase{N}` prints a scorecard: known-related-pairs self-retrieval (recall@1/5/10, MRR), and — with `--labels models/labels_phase{N}.npy` — cluster size distribution + silhouette. `--audit-out <path>` dumps a neighbor CSV/HTML for spot-listening. This is the arbiter for every "compare/evaluate" design choice (standardize on/off, resolution, all-layer vs last-layer, loudness on/off): run it before and after each change.

Current numbers (8 auto-detected related groups in the 107-track personal corpus):
- **Phase 2, rebuilt** (all-layer / multi-window / loudness): recall@1 = 0.625, recall@10 = 0.875, MRR = 0.666; 5 clusters, 0 noise, silhouette 0.202.
- **Phase 2, earlier** (single-layer, single-window): recall@1 = 0.875.

⚠️ The richer embedding **lowered** top-1 self-retrieval on this small corpus (0.875 → 0.625), though recall@10 held. Per Workstream E, "keep the change only if F improves" — so this is an open keep-or-revert decision that needs more data (embed fma_small, add a hand-labeled `related_pairs.json`) before committing. Toggle `LAYER_AGG='last'` / `NORMALIZE=False` in notebook 04 to A/B it.

## Jupyter module caching

Modifying `anther_ml/*.py` while a kernel is live leaves old function objects in memory. Restart the kernel, or force a reload:

```python
import importlib, anther_ml.cluster
importlib.reload(anther_ml.cluster)
from anther_ml.cluster import load_leiden, assign_cluster_knn  # re-bind after reload
```

`importlib.reload` updates the module object but does **not** update names already bound in other cells via `from anther_ml.cluster import ...`. Only names re-imported after the reload pick up the new definition. `anther_ml/__init__.py` re-exports cluster functions — this is another source of stale references after a reload.
