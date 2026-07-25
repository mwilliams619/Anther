# ANTHER APP — COMPLETE FUNCTION & ANALYSIS INVENTORY

## LAYER 1: CORE AUDIO ANALYSIS (anther_ml/)

### Data Loading & Management
- `data.py` — FMA metadata loading
  • `load_fma_features(subset='small'|'medium'|'large')` — load 518-dim librosa features from FMA CSV
  • `fma_feature_columns()` — get canonical feature column ordering
  • Supports track subset filtering by ordered-categorical (small < medium < large)

- `audio.py` — Audio I/O & loudness normalization
  • `load_audio(path, sr)` — load audio from disk at specified sample rate
  • `loudness_normalize(waveform, sr)` — EBU R128 loudness normalization
  • Phase 1 uses native SR; Phase 2 uses 24000 Hz fixed

### PHASE 1: Pre-computed Librosa Features (Fast, CPU-only)

- `features.py` — Feature extraction from audio files
  • `extract_librosa_features(path)` → labeled pandas.Series (518 dims on FMA's MultiIndex)
  • `align_to_corpus(query_series, corpus_columns)` → array aligned to corpus order (asserts match)
  • `drop_loudness_features(vec)` / `feature_weight_vector(...)` — optional down-weighting of loudness-driven features
  • Features: chroma, MFCC, RMS, spectral (bandwidth, centroid, contrast, rolloff), tonnetz, zero-crossing rate
  • Stats per feature: kurtosis, max, mean, median, min, skew, std
  • **Critical**: Always align query to corpus before comparing (silent reorder bug eliminated)

### PHASE 2: MERT Neural Embeddings (GPU-friendly, ~1.3 GB download)

- `embedding.py` — MERT model inference (m-a-p/MERT-v1-330M)
  • `get_embedding(model, processor, path, device, config)` → 1024-dim vector for a single track
  • `embed_batch(model, processor, paths, device, config)` → batch embed multiple tracks
  • `embedding_config(model, sr, layer_aggregation, window_seconds, n_windows, normalize)` → versioned config dict (stored in index)
  • **All-layer aggregation**: mean-pool all 25 MERT hidden states over time
  • **Multi-window**: embed fixed-length windows across track; mean-pool results
  • **Loudness control**: optional EBU R128 normalization before embedding
  • `aggregate_layers='all'|'last'` — toggle for A/B testing
  • Deterministic, reproducible embeddings per corpus+config pair

---

## LAYER 2: CLUSTERING & SIMILARITY SEARCH

### Clustering (anther_ml/cluster.py)

**Primary path — Leiden community detection:**
- `fit_clusters_leiden(X, resolution=1.0, n_pca_components=100, standardize=True)` → dict
  • Returns: `labels`, `embedding_2d`, `scaler`, `pca`, `reducer_2d`, `clustering_space`, diagnostics, config
  • Builds k-NN graph in PCA/embedding space, partitions with Leiden
  • No noise bucket — every node gets a cluster
  • `resolution` tuning: low → broad clusters, high → fine-grained
  • Auto-scales `n_neighbors` by corpus size
  • Cosine metric throughout (eliminates old cosine→euclidean mismatch)
  • 2D UMAP is for visualization only, never for clustering
  
- `save_leiden(path, pipeline_dict)` → pickle (versioned)
- `load_leiden(path)` → restore full pipeline state
- `assign_cluster_knn(query_vec, clustering_space, labels, scaler, pca)` → (cluster_id, confidence)
  • Place a new song onto the corpus via k-NN voting in the learned PCA space

- `resolution_sweep(X, resolutions, ...)` → metrics across resolution range (for tuning)
- `cluster_diagnostics(labels)` → print summary (n_clusters, noise%, degenerate checks)

**Legacy path (comparison only):**
- `fit_clusters(X, ...)` — HDBSCAN on UMAP projection (wrong space, kept for reference)
- `load_pipeline(path)` — legacy 5-tuple return

### Similarity Search (anther_ml/similarity.py)

- `SongIndex` — in-memory exact cosine search
  • `__init__(embeddings, metadata, standardize=True, config=None)`
  • `query(raw_vec, top_k=10)` → top-k neighbors (standardizes + L2-norms internally)
  • `transform_query(raw_vec)` → frozen transform (for `embeddings @ q`)
  • `save(path)` → `.npy` + `.json` pair (config, stats, version)
  • `load(path)` → restore (auto-detects format_version, backward-compatible)
  • `assert_compatible(other_config)` → raise on mismatch (prevents silent incompatibilities)

**Standardization (Phase 1 fix):**
- Raw librosa features span ~8000× range; raw cosine dominated by Hz-scale dims
- Column z-score before L2-norm, using corpus stats only
- `standardize=True` (default Phase 1), `False` (default Phase 2 MERT)
- Query standardized against corpus stats, never its own

---

## LAYER 3: REFERENCE CORPUS & PLACEMENT

### Corpus Building (anther_ml/corpus/)

- `sources.py` — Track source contracts
  • `fma_source` — FMA tracks with built-in features
  • `local_source` — local MP3 files
  • `mpd_source` — Spotify tracks via MySQL dump or RecSys JSON

- `build.py` — Corpus building orchestration
  • `build_corpus(source, name, output_dir, ...)` — embed (resumable/checkpointed) → dedupe → fit clustering → fit index → bundle
  • Full end-to-end: pulls tracks, embeds, builds Leiden clusters, fits `SongIndex`, stores frozen artifacts

- `bundle.py` — Frozen reference corpus
  • `ReferenceCorpus` — immutable snapshot of: embeddings, index, clusters, metadata, config
  • `save(path)` → versioned bundle
  • `load(path)` → restore with integrity checking
  • Bundles live in `models/corpus_<name>/`

### Placement Onto Corpus (anther_ml/corpus/place.py)

- `place(query_vec, reference_corpus)` → placement metadata
  • Finds cluster assignment and nearest corpus neighbors
  • Checks compatibility (config match)
  • Returns: cluster_id, confidence, neighbors, tags (if probe is present)

- `embed_query(audio_path, reference_corpus)` → query embedding
  • Embed a song using the corpus's MERT config
  
- `rank_playlists(playlist_tracks, reference_corpus)` → ranked playlists by fit
  • Score how well a playlist clusters on the corpus

### Micro-genre Tagging (anther_ml/corpus/tagging/)

**Walled-off display layer** — never feeds back to clustering/embedding:

- `vocab.py` — everynoise vocabulary (2000+ micro-genres)
  • Load & match playlist names to tags
  
- `weak_labels.py` — Stage A: playlist-name → seed labels (noisy but in-distribution)

- `probe.py` — Stage B: multi-label probe (frozen)
  • Train on confident seeds, predict for all tracks

- `build_tags.py` — Full pipeline
  • Fit probe → predict → k-NN propagate in MERT space → store tag artifacts in bundle

- `evaluate.py` — Held-out eval (FMA → everynoise crosswalk)
  • Evaluation metrics **never** tune the main map

---

## LAYER 4: EVALUATION

### Scorecard (anther_ml/eval.py)

- `python -m anther_ml.eval --index models/index_phase{N} [--labels labels_phase{N}.npy] [--audit-out path]`
  • Genre-free evaluation of a built index + clustering
  • **Self-retrieval** (primary): Recall@1/5/10 for known-related track pairs
  • **Cluster quality**: silhouette score, size distribution
  • **Neighbor audit**: exportable CSV/HTML for spot-listening
  • **No genre metric ever** — invariant enforced

- `build_scorecard(index, labels, related_pairs)` → metrics dict
- `audit_neighbors(index, related_pairs, top_k)` → HTML export for listening

---

## LAYER 5: CORPUS-BUILDING SOURCES

### Spotify/Deezer Integration (anther_ml/spotify_deezer.py)

- `load_spotify_via_deezer(spotify_id, preview_fallback)` → audio bytes
  • Fetch Spotify track metadata, get preview URL, fall back to Deezer

### MPD Ingestion

**SQL path (recommended):**
- `anther_ml/mpd_sql.py`
  • `python -m anther_ml.mpd_sql --dump <spotifydbdumpshare.sql>` → SQLite (one-time)
  • `python -m anther_ml.mpd_sql --db <.sqlite> --prepare-ui` → playlist index for UI
  • `sample_tracks(db, n, artist_cap)` → deterministic artist-capped sampling
  • UI backend: `search_playlists_db`, `playlist_tracks`

**JSON path (legacy):**
- `anther_ml/mpd_ingest.py` — RecSys JSON MPD slices → corpus queries

---

## LAYER 6: WEB UI

### Atlas Search & Placement (ui/atlas.py)

- **Three-tier song search**:
  1. Local corpus
  2. Deezer (fallback)
  3. Spotify (fallback)

- `place_song(url_or_path)` → embed + place on corpus → add to graph
- `place_playlist(playlist_id)` → async job that embeds & places all tracks
- `place_album(album_id)` → similar for albums
- `recommend(seed_ids, top_k)` → k-NN recommend from selected songs
- **Embed cache** (`embed_cache.sqlite`) — avoids re-embedding repeat songs

### Flask Routes (ui/app.py)

| Route | Function |
|---|---|
| `GET /api/search` | Song search (3-tier) |
| `GET /api/playlists/search` | Full-MPD playlist search (DB) |
| `POST /api/playlist/place` | Async place playlist |
| `GET /api/playlist/status/<job_id>` | Poll placement progress |
| `POST /api/playlist/stop/<job_id>` | Gracefully stop a running placement job |
| `GET /api/albums/search` | Deezer album search |
| `POST /api/album/place` | Async place album |
| `POST /api/place` | Place single song |
| `POST /api/recommend` | Recommend from seed songs |
| `GET /api/song/<id>` | Song detail + cluster + tags |
| `POST /api/upload` | Upload personal track for placement |
| `GET /api/graph` | Current map state (JSON) |
| `POST /api/graph/clear` | Wipe the map |
| `DELETE /api/node/<id>` | Remove one placed song |
| `GET /api/song/<id>/preview` | Resolve a playable preview URL |
| `GET /api/song/<id>/spotify` | Resolve a Spotify track link when available |
| `GET /api/upload-audio/<name>` | Serve uploaded audio |
| `POST /api/demo/load` | Load the curated demo playlist |
| `POST /api/mentor/chat` | Chat with mentor (ReAct) |
| `POST /api/mentor/reset` | Clear mentor session |
| `GET /api/artist/status` | Report Artist View availability |
| `GET /api/artist/search` | Search artists |
| `GET /api/artist/<id>` | Artist detail and optional enrichment profile |
| `GET /api/artist/graph` | Current session artist graph |
| `POST /api/artist/place` | Place an artist |
| `POST /api/artist/from-song-graph` | Build an artist graph from the song map |
| `DELETE /api/artist/node/<id>` | Remove an artist from the session graph |
| `POST /api/artist/graph/clear` | Clear the session artist graph |
| `POST /api/artist/demo` | Load the curated artist demo |

### Graph Visualization (ui/static/)

- `graph.js` — d3 force-directed graph
  • Hover highlight/tooltip, click-to-pin, warm-up polling
  
- `app.js` — Search UI, mode switching (tracks/playlists/albums)
  • Detail popover, recommend-from-map, filter panel
  • Click-to-zoom, artist/album filter
  • Recently-removed with one-click re-place

### Background Jobs (ui/playlist_jobs.py)

- Single async worker (one GPU consumer)
  • Queues playlist/album placements
  • `/api/playlist/status/<job_id>` for polling

### Mentor Chat (mentor/service.py)

- Separate warm process (independent lifecycle)
  • `MusicMentor` — base model + LoRA + RAG + ReAct graph tools
  • Read-only tools: `resolve`, `sounds_like`, `compare`, `bridge`, `crossover`, `artist_tracks`, `tagmates`
  • Cannot mutate map (all mutations via UI controls)
  • Per-browser session state

---

## LAYER 7: VISUALIZATION & EXPORT

### Static Visualization Exports

- `export_viz.py` — Bake index + 2D embedding into standalone HTML
  • Generates `song_view.html` (d3 interactive map)
  • Re-run after index rebuild to refresh

- `export_corpus_viz.py` — Canvas scatter plot for entire corpus bundle
  • Tens of thousands of points, pan/zoom, no force simulation
  • Generates `corpus_<name>_view.html`

### Notebooks

| Notebook | Reads | Writes | Use |
|---|---|---|---|
| `01_explore_fma` | FMA metadata | `fma_small_features.pkl` | Understand FMA structure |
| `02_cluster_precomputed` | features | Phase 1 artifacts | Build Phase 1 pipeline |
| `03_spectrogram_extraction` | personal MP3s | visualization only | Learn spectrograms |
| `04_embedding_mert` | personal MP3s + FMA audio | Phase 2 artifacts | Build Phase 2 pipeline |
| `05_test_new_song` | either pipeline | test viz | Query new songs |

---

## LAYER 8: DATA SOURCES & CORPUS SAMPLING

### FMA

- 106k tracks with pre-computed librosa features
- `subset='small'` (8000 tracks, standard)
- Audio available at native sample rates
- Features ordered on strict MultiIndex (enforced via `align_to_corpus`)

### Spotify Playlist Dataset (MPD)

- **~13.3M tracks** (never embedded whole; always sampled)
- Build once: `python -m anther_ml.mpd_sql --dump ...` → SQLite
- Sample: `python -m anther_ml.corpus build --source sql --sample-n 25000`
- Query: 1M playlists, full metadata via `search_playlists_db`, `playlist_tracks`

### Personal Audio

- Any MP3 file locally
- Librosa extraction or MERT embedding on-demand

### Deezer/Spotify

- Live search & preview URL fetch
- Used as fallback in UI three-tier search

---

## LAYER 9: CONFIGURATION & INTROSPECTION

### Phase Configuration

- `embedding_config(model, sr, layer_aggregation, window_seconds, n_windows, normalize)` → dict
  • Stored in `SongIndex`, read on query
  • Ensures query/corpus use identical settings
  • Prevents silent incompatibilities

- `assert_compatible(config1, config2)` → raise on mismatch

### Checkpoints & Versioning

- Leiden pipeline: `save_leiden` / `load_leiden` (pickle, versioned)
- Index: `SongIndex.save` / `.load` (`.npy` + `.json` pair)
  • Backward-compatible with v1 (bare-list JSON)
- Corpus bundle: `ReferenceCorpus.save` / `.load` (config stamp, integrity check)

---

## KEY INVARIANTS (Do Not Break)

1. **Never genre** → never feeds clustering/embedding/eval (only display layer)
2. **Cluster embeddings, not UMAP** → k-NN graph & Leiden on PCA/full embedding
3. **Standardize Phase 1** → z-score columns before L2 (corpus stats only)
4. **Query = Corpus transform** → read config, apply identical standardization/embedding
5. **Feature ordering explicit** → always `align_to_corpus` before Phase 1 compare
6. **Phase-specific SR** → Phase 1 native, Phase 2 fixed 24kHz
7. **Audio path convention** → `audio_dir/AAA/AAAAAA.mp3` (use `get_audio_path`)

---

## CURRENT STATE / KNOWN LIMITATIONS

### What's Working
- **Phase 1 (librosa)** — fully functional, evaluated on 8000 FMA tracks
- **Phase 2 (MERT)** — builds, embeds, clusters; evaluated on 107 personal tracks
- **UI atlas** — place songs/playlists/albums, search 3-tier, force graph
- **Corpus bundles** — freeze, load, place onto
- **Tagging probe** — builds and predicts micro-genre tags (display-only)
- **Mentor chat** — read-only graph introspection tools (LLM is broken per note)

### Known Issues / Decisions Pending
- **Phase 2 all-layer vs single-layer**: all-layer embedding lowered recall@1 (0.875 → 0.625) on 107-track corpus but held recall@10. Needs larger eval (embed fma_small + hand-labeled pairs) to decide keep-or-revert. Toggle `LAYER_AGG` in notebook 04.
- **Phase 2 standardization**: evaluate whether column z-score helps (current: `standardize=False`).
- **Mentor LLM**: currently broken; chat UI available but responses unreliable.

---

## QUICK REFERENCE: COMMAND-LINE ENTRY POINTS

```bash
# Evaluation
python -m anther_ml.eval --index models/index_phase{N} [--labels models/labels_phase{N}.npy] [--audit-out path.html]

# Corpus building
python -m anther_ml.corpus build --source sql|fma|local \
    [--sql-dump path/spotifydbdumpshare.sql] \
    [--name corpus_name] [--sample-n 25000]

# Corpus placement
python -m anther_ml.corpus place song.mp3 --bundle models/corpus_mpd_100k

# Tagging
python -m anther_ml.corpus.tagging seed|fit|predict|evaluate

# MPD ingestion
python -m anther_ml.mpd_sql --dump data/mpd_dump/spotifydbdumpshare.sql  # one-time
python -m anther_ml.mpd_sql --db data/mpd_dump/spotifydbdumpshare.sqlite --prepare-ui

# UI
python ui/app.py  # port 5000
python -m mentor.service  # port 5100 (separate process)

# Visualization
python export_viz.py  # creates song_view.html
python export_corpus_viz.py  # creates corpus_*_view.html
```

---

## SUGGESTED NEXT STEPS FOR DEEP INVENTORY

If you want to map every callable function:

```bash
# All module-level functions
grep -r "^def " anther_ml/*.py | grep -v test | grep -v "__"

# All classes
grep -r "^class " anther_ml/*.py

# All @click.command() CLI entry points
grep -r "@click.command" anther_ml/ ui/

# All test functions (inverse discovery of what's exercised)
pytest --collect-only -q
```
