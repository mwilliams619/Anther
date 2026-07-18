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

## Dual-space encoding (MERIT backbone)

`embed_tracks_batched_dual` / `_embed_windows_dual` / `get_embedding_dual`
extract **both** representations from one shared MERT forward pass per
track/window — no second inference pass:

- the standard 1024-d MERT vector above (`aggregate_layers(..., "mean")`),
  and
- a 5120-d MERIT backbone (`aggregate_layers(..., "merit_concat")`, 5 layers
  × 1024-d each — `MERIT_LAYERS = (3, 4, 5, 6, 23)`) that downstream projects
  through the 3 frozen MERIT heads (melody/rhythm/timbre — see
  `anther_ml/merit.py` and [similarity.md](similarity.md)'s "MERIT-aggregate
  index" section for how that becomes the driving similarity signal).

A corpus build captures the backbone with `--capture-merit-backbone`
(writes `merit_backbone.npy`, row-aligned to `index.json`); this is what
`anther_ml/corpus/merit_index.py` requires as input.

## `HF_HOME` gotcha (GPU-load hang)

When invoking MERT/MERIT loading from a subprocess or a non-interactive
shell, **set `HF_HOME` explicitly** (e.g.
`HF_HOME=/home/matt/.cache/huggingface`). A relative or unset `XDG_CACHE_HOME`
can cause `transformers`/`huggingface_hub` to resolve the cache to an
unexpected path, which manifests as the GPU-loading step (`_mert()` /
`load_mert`) hanging indefinitely rather than failing fast — it looks like a
CUDA/driver problem but isn't. Any long-running re-embed or batch-encode
script should pin `HF_HOME` in its own environment rather than relying on the
shell's ambient value.
