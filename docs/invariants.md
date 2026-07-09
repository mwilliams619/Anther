# Key invariants

Break these and results silently become meaningless.

**Never introduce genre** as a clustering input, training signal, or eval
metric *for the map itself*. Genre never feeds the MERT embedding, the
`SongIndex`, the Leiden partition, or `anther_ml.eval` — quality of the map is
judged by the genre-free eval harness ([evaluation.md](evaluation.md)).
One carve-out: genre may train **display-only artifacts downstream of the
frozen map** (the micro-genre tag probe in `anther_ml/corpus/tagging/`,
[tagging.md](tagging.md)). Those artifacts are read-outs, like
cluster labels: their metrics live in `tagging/evaluate.py`, walled off from
`build_scorecard`, and must never be used to tune embedding, index, clustering,
or placement hyperparameters. Two scoreboards, one wall.

**Cluster the embedding, never the UMAP/t-SNE coordinates.** Leiden builds its
k-NN graph on the PCA/whitened embedding (`clustering_space`); 2D UMAP is
strictly for the picture. 2D projections distort density and inter-cluster
distance — clustering them fuses or splits groups that aren't. (The legacy
`fit_clusters` violated this by clustering the 32D UMAP output; that's why it's
comparison-only.) See [clustering.md](clustering.md).

**Standardize before cosine similarity (Phase 1).** Raw librosa feature families
span a ~8,000× magnitude range, so row-L2+cosine alone is dominated by Hz-scale
spectral dims. `SongIndex(standardize=True)` z-scores columns (stats fit on the
corpus, persisted, applied identically to queries) before L2. Default True for
Phase-1; evaluate for Phase-2. See [similarity.md](similarity.md).

**Feature ordering is explicit, not positional.** `extract_librosa_features`
returns a labeled `pandas.Series` on FMA's `(feature, statistic, number)`
MultiIndex; always `align_to_corpus(query, corpus.columns)` before comparing —
it asserts the axes match. See [phase1-features.md](phase1-features.md).

**Query and corpus must use the same transform.** Phase-2 queries read
`index.config` to match embedding settings; the index applies its own
standardization to raw query vectors. Never pre-standardize a query or mix
indices built with different configs (`assert_compatible`).

**FMA audio path convention**: `audio_dir/AAA/AAAAAA.mp3` — use
`get_audio_path(dir, track_id)`, don't construct manually.

**Sample rates are phase-specific**: Phase 1 loads at native rate (`sr=None`,
full track) to match FMA's recipe; Phase 2 uses 24000 Hz (MERT). See
[phase1-features.md](phase1-features.md) and [phase2-embeddings.md](phase2-embeddings.md).
