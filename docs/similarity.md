# Similarity index (`similarity.py`)

`SongIndex` is an in-memory exact cosine search over <10k tracks (numpy dot
product; add FAISS only if scaling beyond that — correctness, not scale, is the
concern). A single bundle now carries **two** `SongIndex` instances built on
different embedding spaces — see "MERIT-aggregate index" below for which one
actually drives the UI.

## Standardization (the Phase-1 scale-domination fix)

Raw librosa feature families span a ~8,000× magnitude range (spectral
rolloff/centroid in Hz ≈ 10³ vs. chroma/tonnetz ≈ 10⁻¹). L2-normalizing rows
alone lets the few Hz-scale dimensions dominate cosine, so "most similar"
degrades to "nearest spectral rolloff." `SongIndex` therefore column-standardizes
(z-score) **before** L2-normalizing, using stats fit on the corpus only and
persisted with the index. A query is standardized against the *corpus's* stats,
never its own.

- `standardize=True` — default for Phase-1 (librosa).
- For Phase-2 (MERT/MERIT), evaluate: L2+cosine alone is often best; both the
  MERT-1024 index and the MERIT-aggregate index below use `standardize=False`.

## MERIT-aggregate index — the driving similarity signal

**As of this integration, map edges, corpus-wide search, and ranking are all
driven by the MERIT-aggregate `SongIndex`, not the plain MERT-1024 one.**
MERIT (`anther_ml/merit.py`) projects the same MERT backbone activations
through three frozen, independently-trained heads — melody, rhythm, timbre
(`FACTORS = ("mel", "rhy", "tim")`) — each producing a 128-d L2-unit vector.
The aggregate index is the 384-d concatenation of the three unit vectors:
because each sub-vector has unit norm, **cosine similarity of the concat
equals the equal-weight mean of the three per-factor cosines** — no separate
weighted-scoring code path is needed at query time.

```
index.npy / index.json              MERT-1024 (kept — Leiden clustering was
                                     fit on this space; see anther_ml/cluster.py.
                                     No longer drives edges/search/ranking.)
index_merit_agg.npy / .json         MERIT-384 aggregate (mel+rhy+tim concat) —
                                     drives map edges, corpus search, ranking
factor_mel.npy / factor_rhy.npy /
factor_tim.npy                      (N, 128) unit vectors, row-aligned to
                                     index.json's metadata — the per-factor
                                     breakdown for the song-detail panel
link_calibration_merit.json         LinkThresholds calibrated on
                                     index_merit_agg (separate file/space from
                                     link_calibration.json, which still
                                     calibrates the legacy MERT index)
```

Built via `anther_ml/corpus/merit_index.py` as an **additive sidecar** inside
an existing corpus bundle (never overwrites the MERT `index.npy`/`.json`):

```bash
python -m anther_ml.corpus.merit_index --bundle models/corpus_mpd_100k_merit
```

This requires the bundle to have been built with `--capture-merit-backbone`
(a `merit_backbone.npy`, 5-layer/5120-d, row-aligned to `index.json`). Loading
is lazy via `ReferenceCorpus.merit_index` / `.merit_factors` /
`.merit_calibration` properties (`anther_ml/corpus/bundle.py`) — a bundle
built before this integration simply reports `None` for all three, and
callers fall back to the legacy MERT path (see `place.py`'s `merit_vec=`
fan-out and `ui/atlas.py`'s `_merit_vec_for`).

**Why this is a real improvement, not just a re-scaling**: MERT-1024 vectors
sit in a narrow cone (random corpus pairs cosine ≈0.95–0.98), which forces the
link threshold up near 0.98 and leaves almost no headroom to separate "very
similar" from "somewhat similar." The MERIT-aggregate space spreads pairwise
cosine over a much wider range, so the calibrated query-query link threshold
on the real 100k corpus drops to **`qq_threshold≈0.641`** (vs. MERT's
`≈0.981`) with a same-artist-vs-random-pair discrimination AUC of **0.804**
(vs. MERT's 0.771) — see
[merit_edge_comparison.png](../merit_edge_comparison.png) and the
before/after numbers in `merit_edge_comparison.csv` for the full comparison
on a real 8,000-track sample of the corpus.

## Usage

```python
# Pass a config dict so the index is self-describing and incompatible indices
# can be caught (assert_compatible).
index = SongIndex(X, metadata, standardize=True, config=embedding_config(...))
index.save('models/index_phase1')      # writes .npy + .json (dict schema, versioned) — omit the extension
index = SongIndex.load('models/index_phase1')   # reads both; still loads legacy v1 (bare-list json)

results = index.query(raw_vec, top_k=10)   # standardizes + L2-normalizes internally
q = index.transform_query(raw_vec)         # same frozen transform, for `index.embeddings @ q`
```

- Save/load is a `(.npy, .json)` **pair** — always omit the extension.
- The JSON stores `format_version`, `standardize`, `config`, and the corpus `mean`/`scale`.
- `assert_compatible(other_config)` raises on a config mismatch (e.g. clip length, layers) so a query and corpus can't be silently compared under different settings.
- Pass **raw** vectors to `query`/`transform_query`; the index applies its own transform. Do not pre-standardize.
