# Phase 1 — librosa features (`features.py`, `data.py`)

## Feature-ordering contract

`extract_librosa_features` returns a labeled `pandas.Series` on FMA's exact
`(feature, statistic, number)` MultiIndex (518 dims). FMA blocks features
alphabetically (chroma_cens, chroma_cqt, chroma_stft, mfcc, rmse,
spectral_bandwidth, spectral_centroid, spectral_contrast, spectral_rolloff,
tonnetz, zcr) with stats also alphabetical (kurtosis, max, mean, median, min,
skew, std).

**Always `align_to_corpus(query, corpus.columns)` before comparing** — it
reindexes to the corpus column order and asserts the axes match. A silent
reorder (same 518 length, wrong mapping) is exactly what made Phase-1 uploads
meaningless before this contract existed.

```python
series    = extract_librosa_features(path)               # labeled Series (FMA order)
query_vec = align_to_corpus(series, fma_feature_columns())  # aligned array; asserts
```

## Sample rate

`extract_librosa_features` loads at **native sample rate (`sr=None`) over the
full track**, matching FMA's official recipe (FMA's `features.csv` was computed
that way). Passing `sr=22050` would introduce parameter drift.

## Corpus sizing

`load_fma_features(subset=...)` uses an **ordered-categorical** subset filter
(`small < medium < large`). A raw string `<= "small"` compares alphabetically
(large < medium < small) and silently matches all 106k tracks — the bug that
inflated the "fma_small" corpus 13×. fma_small is the 8,000 tracks labeled
`small`.

## Loudness-feature weighting

`drop_loudness_features` / `feature_weight_vector` down-weight or drop
production-driven blocks (RMS energy, absolute spectral magnitude) in favor of
scale-invariant descriptors (chroma, MFCC shape, tonnetz). Use only if the
neighbor audit still shows loudness-driven neighbors after standardization.
