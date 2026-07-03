# Plan — Fix the Clustering & Similarity Foundation

**For:** Claude Code, working in `anther-ml`.
**Goal:** Make musical similarity and clustering *correct* — fix the technical bugs and design decisions that currently make results meaningless or degenerate. This is engineering cleanup, not a new feature.

**Basis:** Workstreams D and I draw on the single-cell RNA-seq (scRNA-seq) clustering literature, which solves the same core problem — meaningful, label-free clustering of noisy high-dimensional embeddings. The methods and DOIs are documented in `SCRNA_METHODS_REVIEW.md` (same repo); this plan is self-contained, but that file has the citations and the reasoning behind each borrowed technique.

## Guiding principle (read first)
**Do not use genre as a clustering target, a training signal, or an evaluation metric.** Genre is an editorial label that cuts across real acoustic similarity (a stripped-back "Rock" ballad is closer to a "Folk" track than to a "Rock" wall-of-noise track). Clusters must emerge from the audio embeddings alone. Where the old code or docs reach for `genre_top` to judge quality, replace that with the genre-free evaluation in **Workstream F**. Genre may still be *displayed* as metadata; it must never *drive* the math.

## Definition of done
1. Similarity queries return acoustically sensible neighbors (verified by the known-related-pairs test in F).
2. Clustering produces >2 non-degenerate clusters with no single cluster holding >~60% of tracks, on a deliberately-sized corpus.
3. A personal upload is embedded and placed through the *exact* same transform as the reference corpus, with a runtime assertion that dimensions align.
4. `python -m anther_ml.eval` prints a scorecard (self-retrieval, cluster stability, size distribution) that can be re-run after any change.
5. The repo installs as a package; no stray binaries in the source dir.

---

## Workstream A — Fix the similarity math (highest priority, smallest change)

**Problem.** `SongIndex` only L2-normalizes rows; it never standardizes columns. Raw librosa feature families span a ~8,000× magnitude range (spectral rolloff/centroid/bandwidth in Hz ≈ 10³ vs. chroma/tonnetz ≈ 10⁻¹). After row-normalization, cosine similarity is dominated by the few Hz-scale spectral dimensions — "most similar" currently means "nearest spectral rolloff," not musically similar. `StandardScaler` is fit inside `fit_clusters` but never applied on the similarity/query path.

**Fix.**
- Standardize features **before** they enter `SongIndex`, using stats fit on the reference corpus only. Persist those stats (mean/scale) *with* the index, not just inside the clustering pipeline.
- Change `SongIndex` to store the fitted scaler (or the mean/scale vectors) and apply it in both `__init__` (corpus) and `query()` (incoming vector) before L2-normalization. A query must never be normalized against un-standardized corpus stats.
- Keep L2-normalize + dot product as the cosine step *after* standardization.
- For Phase-2 (MERT) embeddings, standardization is a design choice, not an obvious win — neural embeddings are often best with L2+cosine alone. **Make it a constructor flag** (`standardize: bool`) and let Workstream F decide per-phase which wins. Default: `True` for Phase-1 (librosa), evaluate for Phase-2.

**Acceptance.** After the fix, re-running a query on any Phase-1 track returns neighbors whose *chroma/MFCC* profiles are close, not just their loudness. Add a unit test asserting that two tracks identical except for a 10× gain on one spectral band are NOT ranked as near-duplicates (they were, before).

---

## Workstream B — Fix the feature-ordering contract (silent correctness bug)

**Problem.** `extract_librosa_features()` emits feature blocks in order `zcr, chroma_stft, chroma_cqt, chroma_cens, tonnetz, mfcc, …` with per-feature stats `mean, std, skew, kurtosis, median, min, max`. FMA's `features.csv` uses block order `chroma_cens, chroma_cqt, chroma_stft, mfcc, …, tonnetz, zcr` and stat order `kurtosis, max, mean, median, min, skew, std`. Both yield 518 dims, so nothing errors — but **every dimension is misaligned**. A personal song projected into the FMA index compares chroma-against-MFCC, kurtosis-against-mean. Phase-1 upload results are meaningless.

**Fix.**
- Make `extract_librosa_features()` return a **labeled `pandas.Series`** indexed by the exact FMA `(feature, statistic, number)` MultiIndex — not a bare positional array.
- Before use, `reindex()` to the FMA `features.csv` column order (load the reference columns once and cache them). This makes ordering explicit and self-correcting.
- Add a hard assertion at the query boundary: `assert list(query.index) == list(corpus.columns)`, failing loudly on any mismatch.
- Add a round-trip test: extract features for one FMA track whose audio you have, reindex, and confirm the vector matches that track's row in `features.csv` within numerical tolerance (this simultaneously validates ordering *and* that your librosa params match FMA's).

**Note.** The tolerance test may reveal small param drift (window, hop, n_mfcc). If it does, either align the params to FMA's documented settings or accept Phase-1 personal-song queries as approximate and lean on Phase-2 for upload accuracy. Document whichever you choose.

---

## Workstream C — Fix corpus sizing (string-comparison bug)

**Problem.** `load_fma_features(subset="small")` filters with `tracks[("set","subset")] <= "small"`. `subset` is a *string* column; alphabetically `large < medium < small`, so `<= "small"` matches **everything**. The Phase-1 index is 106,574 tracks (full FMA large), not the intended 8,000 — the clustering hyperparameters (`min_cluster_size=100`) were tuned for a corpus 13× smaller.

**Fix.**
- Cast `("set","subset")` to an **ordered categorical** `["small","medium","large"]` and filter with membership or ordered comparison against that category — never string `<=`.
- Make corpus size an explicit, logged decision. For iteration speed and cleaner clusters, start from **fma_small (8k)**; scale up only after the pipeline is validated.
- Log the resulting track count at load time so a wrong subset is immediately visible.

---

## Workstream D — Replace the clustering design with graph community detection (the core "fundamental" issue)

**Problem.** Phase-1 clustering collapsed: 94% of tracks in one cluster (100,373), a 503-track second cluster, 5% noise, only 2 clusters total. Root causes compound: (i) it ran on 13× the intended data with small-corpus hyperparameters (C); (ii) it clustered features whose scale wasn't controlled (A); and (iii) the design runs **HDBSCAN on a UMAP projection** — the wrong algorithm on the wrong space.

**The fix comes from single-cell RNA-seq**, which faced the identical problem (unsupervised grouping of noisy, high-dim, unlabeled data) and converged on a standard pipeline: `k-NN graph → Leiden community detection`, with UMAP demoted to visualization only. See `SCRNA_METHODS_REVIEW.md` for citations (Traag 2019 Leiden; Wolf 2018 SCANPY; Becht 2018 UMAP-for-viz).

**Fixes (do A + C first, then re-evaluate — they may resolve much of the degeneracy on their own):**
1. **Swap HDBSCAN → Leiden community detection on a k-NN graph.** This is the highest-leverage change. Build a k-nearest-neighbor graph in the embedding space, then partition it with the Leiden algorithm. Reference implementation: SCANPY's `sc.pp.neighbors(adata, n_neighbors=…, metric="cosine")` + `sc.tl.leiden(adata, resolution=…)`; standalone: `leidenalg` + `igraph`. Why this fixes the specific failure:
   - **No giant catch-all cluster and no noise bucket.** Graph partitioning assigns *every* node to a community — it structurally cannot produce HDBSCAN's one-mega-cluster-plus-outliers result. The 5,698 "noise" tracks get placed.
   - **One interpretable knob — `resolution`.** Low → few broad clusters, high → many fine ones. Far more controllable than HDBSCAN's `min_cluster_size`/`min_samples` interacting with UMAP hyperparameters. Sweep resolution (e.g. 0.2–2.0) and pick using the Workstream F stability metric.
   - **Leiden, not Louvain.** Louvain can return internally *disconnected* communities; Leiden guarantees well-connected ones, converges faster, and gives better partitions. Use Leiden.
2. **Cluster the graph/embedding, never the UMAP coordinates.** The single-cell field is unanimous that UMAP/t-SNE are visualization-only — 2D projections distort local density and inter-cluster distance, fusing or splitting groups that aren't. Build the k-NN graph on the PCA/whitened embedding (Phase-1) or the MERT latent (Phase-2), and keep 2D-UMAP strictly as the picture. This also **eliminates the old cosine-UMAP → euclidean-HDBSCAN metric mismatch** entirely, since the graph is built directly in the embedding with one chosen metric (cosine).
3. **Set `n_neighbors` relative to corpus size**, not a hardcoded constant. Typical single-cell defaults are 10–30; for Phase-2 (n=107) use a smaller value (e.g. 10–15), for a large Phase-1 corpus 15–30. Log it.
4. **Expose cluster diagnostics on every fit.** After clustering, always print: n_clusters, largest-cluster share, size histogram (and noise % if any method still produces it). A degenerate run (one cluster >~60%) must be obvious without manual inspection — log a warning automatically.

**Note on HDBSCAN.** You may keep HDBSCAN as an optional comparison method, but run it **on the embedding/PCA space, not on UMAP output**, and let Workstream F decide. The primary recommendation is Leiden.

**Acceptance.** On fma_small (8k) with fixes A+C+D via Leiden: >2 clusters, largest cluster <~60% of tracks, stable under resampling (Workstream F2). If clean clusters genuinely don't exist in the data, that's a real finding — report the size/stability distribution rather than forcing clusters.

---

## Workstream E — Fix the MERT embedding quality (design decisions)

**Problem.** `get_embedding` / `embed_batch` use only `outputs.last_hidden_state`, mean-pooled over a single 30s clip taken from file start. Two issues:
- **Single-layer.** MERT is a stacked transformer; different layers encode different musical facets (lower ≈ pitch/timbre, higher ≈ higher-level structure). MERT's own usage guidance is to aggregate across *all* hidden layers, not use only the last. Using one layer discards most of the model's representational content.
- **Single 30s window from the start.** Intros/outros bias the vector; the characteristic section is often missed.

**Fixes.**
1. Request `output_hidden_states=True` (already done) and aggregate across **all** hidden states. Derive the layer count at runtime — `n_layers = len(outputs.hidden_states)` (do not hardcode; the 330M model has 24 transformer layers → 25 hidden states, but read it from the tensor so a model swap can't break it). Start with a simple **mean across layers** of the time-mean-pooled per-layer vectors; optionally expose a learned/weighted-layer option later. Compare single-last-layer vs all-layer-mean using Workstream F.
2. **Aggregate multiple windows.** Embed several fixed-length windows (e.g. 3× overlapping 10–30s windows, or evenly spaced) and mean-pool the resulting vectors, so the embedding represents the whole track rather than its opening.
3. **Enforce one consistent clip length** across the whole index — the existing Phase-2 index mixed full-length personal files with what the UI clips to 30s. Pick one `CLIP_SECONDS`, apply it identically to corpus and query, and record it in the index metadata so incompatible indices can't be silently mixed.

**Acceptance.** The all-layer / multi-window embedding scores at least as well as the current single-layer/single-window on the F metrics; if it doesn't, keep the simpler one and document why.

---

## Workstream F — Genre-free evaluation harness (replaces the missing eval, no genre labels)

**Problem.** There is currently no way to tell whether similarity works — only eyeballing a UMAP. The obvious fix (genre-precision@k) is explicitly rejected here: genre is not the target.

**Fix — build `anther_ml/eval.py`, runnable as `python -m anther_ml.eval --index <path>`.** Use ground truth that requires no genre and already exists in the data:

1. **Known-related-pairs self-retrieval (primary metric).** The personal corpus contains near-duplicate/related items — alternate mixes, versions, and stems: `hang v1` / `hang v1-1`, `Mic Check` / `Mic Check vocals` / `Mic Check vocals_1`, `colture rough mix 1` / `colture rough mix 2`, `The Marías` ×2, etc. Define these related groups (start with a small hand-labeled `related_pairs.json`; detect obvious ones by title-stem matching to seed it). Metric: for each track with a known relative, is that relative in its top-k neighbors? Report **recall@1 / @5 / @10 and mean reciprocal rank**. A correct similarity model ranks a song's own alternate mix near the top — with zero genre involved.
2. **Cluster stability under resampling.** Refit clustering on bootstrap subsamples (e.g. 80% of tracks, 5–10 seeds) and measure label agreement (Adjusted Rand Index) on the shared tracks. Stable clusters reproduce; degenerate/noise-driven ones don't. Report mean ± std ARI.
3. **Internal cluster quality.** Silhouette score (on the clustering-space embedding, not the 2D viz) and the size distribution (n_clusters, noise %, largest-cluster share). These need no labels.
4. **Neighbor-audit export.** For a handful of seed tracks, dump top-10 neighbors to a small HTML/CSV so a human can spot-listen — the ultimate genre-free check is "do these sound alike."

Make F the arbiter for every design choice flagged "compare / evaluate" in A, D, and E. Run it before and after each change and keep the scorecard in the repo.

---

## Workstream G — Data hygiene & reproducibility

- **De-duplicate Phase-2 metadata** (exact-name and near-name duplicates like the `Mic Check`/`hang v1` variants) — but keep genuine alternate mixes, since F depends on them; dedup only true duplicates.
- **Re-link embeddings to source audio.** Only 3 of the 107 personal mp3s are on disk; the 107 embeddings are orphaned and not reproducible. Either restore the source folder or regenerate the index from a known, committed-to-a-manifest audio location. Record the source path + `CLIP_SECONDS` + embedding config (layers, windows) in the index JSON so any index is self-describing.
- **Version the index format.** Add a `format_version` and the embedding config to the saved metadata; `SongIndex.load` should refuse to mix indices built with different configs.

## Workstream H — Packaging & repo cleanup (removes a whole class of bugs)

- Add a minimal `pyproject.toml` and make `anther_ml` an editable install (`pip install -e .`). This eliminates the `sys.path.insert` hacks and the Jupyter stale-module reload hazards documented in `CLAUDE.md`.
- Move unrelated files out of the package dir: `anther_ml/mac-arm64.dmg`, `anther_ml/JR-sp-17-01.pdf`, `anther_ml/cellranger-benchmark-analysis.md` do not belong in source.
- Keep the 1 GB+ of model pickles out of git (they're gitignored already) but document how to regenerate them from the notebooks/scripts.
- **New dependencies** introduced by this plan (add to `requirements.txt` / `pyproject.toml`): `scanpy` (or `leidenalg` + `python-igraph`) for Workstream D's Leiden clustering; `pyloudnorm` (EBU R128) for Workstream I's loudness normalization; `scikit-learn` (silhouette/ARI) for the Workstream F metrics if not already present.

---

## Workstream I — Treat production/loudness as a "batch effect" (new — borrowed from scRNA-seq integration)

**Problem.** Beyond the raw feature-scale issue (Workstream A), there is a deeper confound: two songs can land near each other in embedding space because they share **production characteristics** — mastering loudness, mix brightness, codec/bitrate, lo-fi vs. studio, recording era — rather than because they are *musically* similar. This is exactly what the Phase-1 scale analysis surfaced (loudness/brightness dominating the cosine score), and it will affect MERT embeddings too, since MERT sees the audio as-is.

**The scRNA-seq analogy.** Single-cell data has "batch effects" — technical variation (which machine, day, lab, chemistry) that makes cells cluster by *batch* instead of *biology*. The field's standard response is **integration/batch correction**: learn to factor a nominated nuisance variable out of the embedding so the remaining structure reflects the signal of interest (Harmony — Korsunsky 2019; benchmarks — Tran 2020, Luecken 2021; see `SCRNA_METHODS_REVIEW.md`). Production/loudness is the music-similarity batch effect, and the same playbook applies.

**Fixes (in increasing order of effort — do #1 regardless):**
1. **Loudness-normalize audio before embedding.** Apply EBU R128 / ReplayGain (or simple peak/RMS normalization) to every file — corpus and query alike — before feature extraction or MERT inference. This removes the most blatant production confound (mastering loudness) at near-zero cost, and directly attacks the loudness-domination failure. Record the normalization method in the index metadata so corpus and query stay consistent.
2. **Down-weight or drop pure-loudness features (Phase-1).** RMS energy and absolute spectral-magnitude stats encode production more than musical content. Consider excluding or down-weighting them in the librosa feature path, and rely on scale-invariant descriptors (chroma, MFCC shape, tonnetz).
3. **Harmony-style correction over a production covariate (optional, higher effort).** If you can tag tracks with a nuisance factor — bitrate/codec, or a coarse lo-fi/studio/live label — apply a Harmony-style correction (or scVI's conditional variant) to factor it out of the embedding. Only worth it if #1 and #2 leave a visible production-driven structure in the Workstream F neighbor audits.

**Acceptance.** After loudness normalization (#1), the neighbor-audit export (Workstream F4) should show fewer "these are just both loud/quiet" neighbors and more musically-coherent ones. If a production covariate is available, cluster stability (F2) should not be explained by that covariate (check that clusters don't simply recover bitrate/era).

---

## Suggested execution order
1. **H** (packaging) — do first so everything else imports cleanly and tests are runnable.
2. **A + B + C** — the correctness bugs. Small, high-leverage, and they likely fix much of D on their own.
3. **I #1** (loudness normalization) — do this early too: it's cheap, and every embedding built afterward benefits. Rebuilding indices later just to add it is wasteful.
4. **F** — build the eval harness now, so D, E, and I are measured, not guessed.
5. **D** — replace HDBSCAN-on-UMAP with Leiden graph clustering on the corrected, correctly-sized corpus; sweep `resolution` using F.
6. **E** — improve MERT embeddings; keep changes only if F scores improve.
7. **I #2–#3** — down-weight loudness features / optional Harmony-style correction, only if F neighbor audits still show production-driven structure.
8. **G** — data hygiene / reproducibility once the pipeline is stable.

## What NOT to do
- Do not introduce genre (or any editorial tag) as a clustering input, loss, or eval metric.
- Do not add FAISS/approximate search yet — at <10k tracks, exact numpy cosine is fine; scale is not the current problem, correctness is.
- Do not force a target number of clusters. If the data doesn't cluster cleanly, report that honestly via F rather than tuning until arbitrary clusters appear.
- Do not skip the assertions (B) — silent dimension misalignment is exactly the class of bug that made Phase-1 uploads meaningless.
- Do not cluster on UMAP/t-SNE coordinates (D2) — those are for visualization only. Cluster the graph/embedding.
- Do not use Louvain where Leiden is available (D1) — Louvain can produce disconnected communities.
