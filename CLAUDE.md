# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
source venv/bin/activate
```

Notebooks use `sys.path.insert(0, '..')` and must be launched from the repo root or the `notebooks/` directory. There is no `setup.py` or `pyproject.toml` — the package is not installed, just path-inserted.

## Architecture

Two-phase music similarity pipeline. Both phases share the same `anther_ml` clustering and similarity API — only the input embeddings differ.

**Phase 1 — Pre-computed librosa features (fast, no GPU)**
- Input: FMA `features.csv` (518-dim, 3-level MultiIndex header — always load via `load_fma_features()`)
- Pipeline: `StandardScaler → PCA(100d) → UMAP(32d) → HDBSCAN → 2D-UMAP viz`
- Query path for a personal song: `extract_librosa_features(path)` → `assign_cluster()`

**Phase 2 — MERT neural embeddings (GPU recommended)**
- Model: `m-a-p/MERT-v1-330M` (~1.3 GB HuggingFace download on first use)
- Same clustering API, 1024-dim input. Embedding 8k FMA tracks takes ~2-3 hrs on Apple MPS.
- Query path: `get_embedding(model, processor, path, device)` → `assign_cluster()`

**`anther_ml/` module map**

| Module | Responsibility |
|---|---|
| `data.py` | FMA CSV loading (`load_fma_tracks`, `load_fma_features`, `get_audio_path`) |
| `features.py` | Phase 1 feature extraction — mirrors FMA's 518-dim schema at 22050 Hz |
| `embedding.py` | Phase 2 MERT loading and inference at 24000 Hz |
| `cluster.py` | `fit_clusters`, `assign_cluster`, `save_pipeline`, `load_pipeline` |
| `similarity.py` | `SongIndex` — in-memory cosine search (L2-normalised numpy dot product) |

## Clustering API signatures

These return-value counts matter — getting them wrong causes silent unpacking errors:

```python
# 6 return values (pca may be None if n_pca_components=None)
labels, embedding_2d, scaler, pca, reducer, clusterer = fit_clusters(X, ...)

# 4 return values (pca=None for pipelines saved before PCA was added)
scaler, pca, reducer, clusterer = load_pipeline(path)

# pca keyword is required if the pipeline has one
cluster_id, strength = assign_cluster(vec, scaler, reducer, clusterer, pca=pca)
```

`SongIndex` saves and loads as a pair — always omit the extension:
```python
index.save('../models/index_phase1')      # writes .npy + .json
SongIndex.load('../models/index_phase1')  # reads both
```

## Notebook pipeline order

Each notebook saves artifacts the next one loads:

| Notebook | Reads | Writes |
|---|---|---|
| `01_explore_fma` | `data/fma_metadata/` | `models/fma_small_features.pkl` |
| `02_cluster_precomputed` | `fma_small_features.pkl` | `models/pipeline_phase1.pkl`, `index_phase1.*`, `embedding_2d_phase1.npy`, `labels_phase1.npy` |
| `03_spectrogram_extraction` | personal MP3s | visualization only (educational) |
| `04_embedding_mert` | personal MP3s + optionally `data/audio/fma_small/` | `models/pipeline_phase2.pkl`, `index_phase2.*` |
| `05_test_new_song` | pipeline + index (either phase) | `models/test_song_phase{N}.png` |

`05_test_new_song` is the production-like entry point — it supports both phases via a `PHASE = 1 | 2` toggle.

## Key invariants

**2D UMAP must be derived from 32D UMAP output, not raw features.** The 2D visualization embedding in `fit_clusters` is computed from `X_reduced` (the 32D UMAP output), not from `X_scaled`. This ensures cluster boundaries visible in the plot correspond to HDBSCAN's actual decisions. Never feed raw features to the 2D reducer.

**FMA audio path convention**: `audio_dir/AAA/AAAAAA.mp3` — use `get_audio_path(dir, track_id)`, don't construct manually.

**Sample rates are not interchangeable**: Phase 1 uses 22050 Hz (librosa default, matches FMA). Phase 2 uses 24000 Hz (MERT requirement). The constants `SR` in each module encode this.

## Jupyter module caching

Modifying `anther_ml/*.py` while a kernel is live leaves old function objects in memory. Restart the kernel, or force a reload:

```python
import importlib, anther_ml.cluster
importlib.reload(anther_ml.cluster)
from anther_ml.cluster import load_pipeline, assign_cluster  # re-bind after reload
```

`importlib.reload` updates the module object but does **not** update names already bound in other cells via `from anther_ml.cluster import ...`. Only names re-imported after the reload pick up the new definition. `anther_ml/__init__.py` re-exports cluster functions — this is another source of stale references after a reload.
