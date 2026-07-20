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
| `mpd_sql.py` | **MySQL MPD dump** (`spotifydbdumpshare.sql`) → SQLite → deterministic artist-capped sample (`load_dump_to_sqlite`, `ensure_db`, `sample_tracks`). Streaming, resumable, logged. This is the supported path for our dataset. Also the web UI's full-MPD playlist backend: `--prepare-ui` (one-time index + `playlist_search` table), then `search_playlists_db` / `playlist_tracks` serve any of the 1M playlists |
| `mpd_ingest.py` | Legacy: RecSys `mpd.slice.*.json` files → corpus queries (`build_mpd_queries`, `enrich_isrc`). Kept for JSON-format MPD downloads |
| `spotify_deezer.py` | Spotify metadata → Deezer preview audio (`load_spotify_via_deezer`); preview fetch/decode reused by both MPD paths |

The corpus `sql_source` (in `corpus/sources.py`) reads the SQL dump via `mpd_sql`
and fetches audio from each track's Spotify `preview_url` directly (Deezer
fallback for dead URLs). Build the DB once, then sample many times:

```bash
python -m anther_ml.mpd_sql --dump data/mpd_dump/spotifydbdumpshare.sql   # one-time
python -m anther_ml.corpus build --source sql \
    --sql-dump data/mpd_dump/spotifydbdumpshare.sql --name mpd_25k --sample-n 25000
# one-time UI prep (playlist_id index + playlist_search table; minutes, ~4 GB growth)
python -m anther_ml.mpd_sql --db data/mpd_dump/spotifydbdumpshare.sqlite --prepare-ui
```

The full dump is ~13.3M tracks — far past the design's 10k–100k range — so `sql_source`
always *samples down* (artist-capped, deduped post-embed); it is never used to embed
the whole dump.

## `anther_ml/corpus/` — frozen reference-corpus bundles

A MERT-space reference corpus that new songs are *placed onto*, never re-clustered
from scratch. Design rationale in [projects/REFERENCE_CORPUS_DESIGN.md](projects/REFERENCE_CORPUS_DESIGN.md).
CLI: `python -m anther_ml.corpus build …` / `… place song.mp3 …`.

| Module | Responsibility |
|---|---|
| `sources.py` | Track sources under one item contract (`fma_source`, `local_source`, `mpd_source`) |
| `build.py` | `build_corpus` — embed (checkpointed/resumable) → dedupe → fit `SongIndex` + Leiden → per-cluster profiles → freeze bundle |
| `bundle.py` | `ReferenceCorpus` — the frozen, versioned bundle (`save`/`load`, config stamp); lazily loads the MERIT-aggregate sidecar via `.merit_index`/`.merit_factors`/`.merit_calibration` when present |
| `place.py` | Placement regime — `place`, `embed_query`/`embed_query_dual`, playlist-fit / `rank_playlists`; `place()` routes to the MERIT-aggregate index when `merit_vec=` is given and the bundle has one, else falls back to the legacy MERT index |
| `merit_index.py` | Builds the MERIT-aggregate `SongIndex` sidecar (`index_merit_agg.npy/.json`) + per-factor cosine sidecars (`factor_mel/rhy/tim.npy`) + its own calibration (`link_calibration_merit.json`) from a bundle's `merit_backbone.npy` — additive, never touches the MERT index. See [similarity.md](similarity.md)'s "MERIT-aggregate index" section |
| `labels.py` | Playlist-name cluster labels — post-processing over a built bundle ([projects/PLAYLIST_LABELS_BUILD_PLAN.md](projects/PLAYLIST_LABELS_BUILD_PLAN.md)) |
| `tagging/` | Micro-genre tag probe — vocab, weak seeds, probe fit/predict, FMA held-out eval (`python -m anther_ml.corpus.tagging`; see [tagging.md](tagging.md)) |
| `__main__.py` | `build` / `place` / `label` CLI |

Bundles are written to `models/corpus_<name>/` (primary:
`corpus_mpd_100k_merit`, the first bundle built with
`--capture-merit-backbone` and a MERIT-aggregate sidecar). Tests:
`tests/test_corpus_{build,bundle,place,labels,tagging,merit_index}.py`.

## Top-level scripts & `ui/`

| Script | Responsibility |
|---|---|
| `export_viz.py` | Bakes an index + 2D embedding into the standalone `song_view.html` d3 map (`PHASE` set at top; re-run after any index rebuild) — see [notebooks.md](notebooks.md) |
| `export_corpus_viz.py` | Canvas scatter-plot viewer for a whole corpus bundle (tens of thousands of points; pan/zoom, no force sim) → `corpus_*_view.html` |
| `ui/` | Flask + d3 song atlas UI — search, place songs/playlists/albums onto the frozen corpus, browse the map. `python ui/app.py`, port 5000. Details: [ui.md](ui.md) |

## Tests

`tests/test_{audio,cluster,data,embedding,eval,features,similarity,mpd_sql}.py`
(one per core module), `tests/test_corpus_*.py` (corpus subpackage), and
`tests/test_atlas_search.py` (UI atlas search tiers), and
`tests/test_atlas_playlist.py` (playlist search + placement). Run with `pytest`.
