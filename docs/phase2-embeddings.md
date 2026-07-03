# Phase 2 — MERT embeddings (`embedding.py`, `audio.py`)

## Embedding design

`get_embedding` / `embed_batch` produce 1024-dim MERT vectors with:

- **All-layer aggregation.** MERT is a 24-layer transformer (25 hidden states); different layers encode different musical facets. We mean-pool each layer over time, then mean across layers. Layer count is read at runtime (`aggregate_layers`), so a model swap can't break it. `layer_aggregation='last'` reproduces the old single-layer behavior for A/B comparison.
- **Multi-window.** Several evenly-spaced fixed-length windows are embedded and mean-pooled (`plan_windows`), so the vector represents the whole track, not just its opening. Deterministic — corpus and query get identical plans.
- **Loudness normalization.** When `normalize=True`, audio is EBU R128 loudness-normalized before inference (`audio.loudness_normalize`), removing the mastering-loudness production confound. This is the music-similarity analogue of a single-cell batch effect.

`embedding_config(...)` returns a self-describing metadata dict (model, sr,
layer_aggregation, window_seconds, n_windows, loudness settings) — store it in
the `SongIndex` config so incompatible indices can't be mixed
([similarity.md](similarity.md)).

## Sample rate

Phase 2 uses **24000 Hz** (MERT requirement). The `SR` constant in
`embedding.py` / `audio.py` encodes this. (Phase 1 differs — see
[phase1-features.md](phase1-features.md).)

## Consistency requirement

A query must be embedded with the **same config as the corpus**. Notebook 05
reads `index.config` and passes matching `layer_aggregation` / `normalize` to
`get_embedding`, so the query and corpus are produced identically.
