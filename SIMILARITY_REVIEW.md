# Similarity Data — Inventory & Review

_Review of `anther-ml` generated similarity artifacts and current approach._

## 1. Inventory of generated data (`models/`)

| Artifact | Shape / size | What it actually contains |
|---|---|---|
| `index_phase1.npy` + `.json` | **106,574 × 518** | Reference corpus embeddings + metadata (track_id, name, artist, genre, cluster). **Raw librosa features, row-L2-normalized only.** |
| `labels_phase1.npy` | 106,574 | HDBSCAN cluster IDs. **Degenerate: cluster 1 = 100,373 (94%), cluster 0 = 503, noise = 5,698 (5%).** Only 2 clusters total. |
| `embedding_2d_phase1.npy` | 106,574 × 2 | 2D UMAP viz coords. |
| `pipeline_phase1.pkl` | 316 MB | Fitted scaler / PCA / UMAP / HDBSCAN. |
| `fma_small_features.pkl` | 224 MB | Cached FMA features. |
| `index_phase2.npy` + `.json` | **107 × 1024** | MERT embeddings. **All 107 are the user's own uploads** — `artist: personal`, `genre: unknown`. No reference corpus. |
| `labels_phase2.npy` | 107 | 5 clusters + noise. **43% noise (46/107).** |
| `personal_embeddings.npy` / `personal_metadata.pkl` | 107 × 1024 | Same 107 personal tracks. |
| `data/audio/personal/` | **3 mp3 files** | Only 3 audio files actually on disk (Mic Check, Tropic Juice, icy_). The 107 embeddings were generated from files no longer present. |

**Headline:** the two phases are not comparable products. Phase 1 has a large reference library but broken similarity math and failed clustering. Phase 2 has good embeddings but **nothing to compare against** — it can only compare the user's songs to each other, which is the opposite of the stated goal ("see how new songs compare to what's out there").

---

## 2. Weaknesses in the approach

### Critical — these make the similarity output wrong, not just suboptimal

**W1. Phase-1 similarity runs on unscaled features → dominated by loudness/brightness in Hz.**
`SongIndex` L2-normalizes each row but never standardizes columns. Raw FMA feature families span a **~8,000× magnitude range** (spectral_rolloff ≈ 2174, spectral_centroid ≈ 1237, spectral_bandwidth ≈ 1012 vs. tonnetz ≈ 0.27, chroma ≈ 0.4). After row-normalization the three Hz-scale spectral features swamp the cosine score; harmony (chroma, tonnetz), timbre (MFCC), and rhythm (ZCR) contribute almost nothing. "Most similar song" currently means "closest spectral rolloff frequency," not perceptually similar. The `StandardScaler` exists but is only applied *inside* the clustering pipeline, never on the similarity path.

**W2. Personal songs are projected into Phase 1 with the wrong feature ordering.**
`extract_librosa_features()` emits blocks in order `zcr, chroma_stft, chroma_cqt, chroma_cens, tonnetz, mfcc, ...` with per-feature stats ordered `mean, std, skew, kurtosis, median, min, max`. FMA's `features.csv` orders blocks `chroma_cens, chroma_cqt, chroma_stft, mfcc, ..., tonnetz, zcr` with stats `kurtosis, max, mean, median, min, skew, std`. Both produce 518 dims so nothing errors — but **every dimension is misaligned**. A personal song fed to the FMA-trained scaler/index compares chroma against MFCC, kurtosis against mean, etc. Phase-1 query results for uploaded songs are meaningless.

**W3. Phase-1 clustering collapsed.** 94% of tracks fell into one cluster that mixes Rock, Experimental, Electronic, Hip-Hop, Pop indiscriminately. `min_cluster_size=100` on cosine-UMAP of un-de-correlated features gave essentially one blob + noise. There are no usable "neighborhoods" to place a song into.

**W4. `load_fma_features(subset="small")` loaded the full 106k corpus, not the 8k small set.** The filter `tracks[("set","subset")] <= "small"` is a *string* comparison; alphabetically `large < medium < small`, so `<= "small"` matches everything. The index is 106,574 tracks — the entire FMA large set — not the intended 8,000. (Not fatal, but it means Phase-1 ran on 13× the intended data with the intended hyperparameters.)

**W5. Phase 2 has no reference universe.** All 107 vectors are the user's own material. There is no commercial/FMA corpus embedded with MERT, so "how does my song compare to what's out there" cannot be answered in the good-embedding phase. The playlist-placement use case is not yet servable by either phase.

### Moderate — quality/robustness

**W6. MERT embedding uses only the last hidden layer, mean-pooled.** MERT's own guidance is to take a weighted sum (or concatenation) across all 13 hidden layers — lower layers carry pitch/timbre, higher layers carry semantic/genre content. Using only `last_hidden_state` discards most of what makes MERT strong for similarity.

**W7. 30-second, single-clip embeddings.** Both phases load `duration=30.0` from the file start. Intros/outros bias the vector; a song's characteristic section is often missed. No multi-window aggregation.

**W8. Deezer/upload previews vs. full files aren't length-matched on the Phase-2 index.** The UI plan clips to 30s for comparability, but the existing `index_phase2` was built from full personal files at varying lengths — mixing clip lengths shifts mean-pooled vectors.

**W9. No evaluation or ground truth anywhere.** No held-out check, no "do genre neighbors actually rank near each other," no silhouette/ARI on labels. There's no signal that similarity is working beyond eyeballing the UMAP.

**W10. Metadata hygiene.** Phase-2 metadata has duplicates (`Mic Check` ×3 variants, `hang v1` ×2), every `genre` is `unknown`, and the 107 embeddings are orphaned from their source audio (only 3 mp3s on disk) — not reproducible.

---

## 3. Weaknesses in project setup — immediately addressable

These are cheap fixes with high leverage, roughly in priority order.

1. **Standardize before similarity (W1).** Persist the fitted `StandardScaler` (or z-score stats) with each index and apply it to both corpus and query vectors before building/querying `SongIndex`. This alone changes Phase-1 similarity from "loudness match" to something musically meaningful. ~20 lines in `similarity.py` + a rebuild.
2. **Fix the feature-ordering contract (W2).** Make `extract_librosa_features()` return a `pd.Series`/`DataFrame` indexed by the exact FMA `(feature, statistic, number)` MultiIndex and reindex to the FMA column order before use — never rely on positional concatenation. Add an assertion that query columns equal index columns.
3. **Fix the subset filter (W4).** Replace the string `<=` with an explicit ordered category or membership set (`subset.isin(["small"])`), so "small" means 8,000 tracks. Decide corpus size deliberately.
4. **Re-tune Phase-1 clustering (W3).** Lower `min_cluster_size` (e.g. 30–50), add the missing decorrelation, and report cluster count + noise% + per-cluster genre purity every refit so a degenerate run is caught immediately.
5. **Add a tiny evaluation harness (W9).** Using FMA `genre_top` as weak labels: for a sample of tracks, measure whether top-k neighbors share the genre (precision@10) and compute silhouette/ARI on labels. One script, run after every rebuild — turns "looks fine" into a number.
6. **De-dup + re-link Phase-2 metadata (W10).** Dedup by name, and either restore the source audio for the 107 embeddings or regenerate from a known folder so the index is reproducible.
7. **Repo packaging.** There's no `pyproject.toml`/`setup.py` (everything is `sys.path` hacks), models are 1+ GB of un-versioned pickles, and `mac-arm64.dmg` / `JR-sp-17-01.pdf` / `cellranger-benchmark-analysis.md` are unrelated files sitting in `anther_ml/`. Add a minimal `pyproject.toml` (editable install kills the reload/caching gotchas in CLAUDE.md), and move stray binaries out of the package dir.

### The bigger, non-immediate item
**Build a real reference corpus in the MERT space (W5).** Phase 2 is the right engine but needs a library of "what's out there" — embed the FMA small/medium set (or a Deezer-preview corpus) with MERT so uploaded songs have something to be compared against. This is the difference between "cluster my own tracks" and the actual product ("where does my song fit among released music / playlists"). It's the highest-value next build, just not a same-day fix.

---

## 4. One-line verdict
The MERT embedding engine (Phase 2) is the sound foundation; the **similarity computation and the reference data around it are the weak points.** Fixing feature scaling (W1) and query alignment (W2) makes existing results correct today; building a MERT reference corpus (W5) is what unlocks the stated playlist-placement goal.
