# Similarity index (`similarity.py`)

`SongIndex` is an in-memory exact cosine search over <10k tracks (numpy dot
product; add FAISS only if scaling beyond that — correctness, not scale, is the
concern).

## Standardization (the Phase-1 scale-domination fix)

Raw librosa feature families span a ~8,000× magnitude range (spectral
rolloff/centroid in Hz ≈ 10³ vs. chroma/tonnetz ≈ 10⁻¹). L2-normalizing rows
alone lets the few Hz-scale dimensions dominate cosine, so "most similar"
degrades to "nearest spectral rolloff." `SongIndex` therefore column-standardizes
(z-score) **before** L2-normalizing, using stats fit on the corpus only and
persisted with the index. A query is standardized against the *corpus's* stats,
never its own.

- `standardize=True` — default for Phase-1 (librosa).
- For Phase-2 (MERT), evaluate: L2+cosine alone is often best; the current Phase-2 index uses `standardize=False`.

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
