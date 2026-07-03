# Design — A Reference Corpus in MERT Space

**For:** `anther-ml`. **Slots into:** Workstream W5 / "the bigger non-immediate item" in `SIMILARITY_REVIEW.md` and `CLUSTERING_FIX_PLAN.md`.
**Goal:** Define what a MERT-space reference corpus *is*, why it is the thing that makes clustering of small subsets and new uploads meaningful, and the smartest way to build one.

---

## 1. What "a reference corpus in MERT space" actually means

Today Phase 2 has good embeddings but the index is 107 vectors, **all the user's own tracks**. There is no "out there." A reference corpus fixes that by being a **large, fixed, representative set of released music, each embedded once with MERT, that serves as the coordinate system every new song is placed into.**

Crucially, the corpus is **not just a pile of embedding vectors.** It is a frozen *bundle*:

| Component | What it is | Why it must be frozen with the corpus |
|---|---|---|
| `corpus_embeddings.npy` (N×D) | MERT vectors, final embedding recipe | the reference point cloud |
| `corpus_metadata.json` | track_id, title, artist, source (+ genre as **display only**) | lets neighbors be named / traced to playlists |
| `scaler` / `pca` (whitening) | fit on corpus **only** | the transform a query must reuse *exactly* |
| `reducer` (UMAP, kNN graph) | fit on corpus | the manifold new songs are projected into |
| `clusterer` (Leiden partition + kNN assign model) | fit on corpus | the **named neighborhoods** new songs are assigned to |
| `reducer_2d` | fit on 32D output | the picture, aligned to the clusters |
| `embedding_config` stamp | model, layers, clip_seconds, n_windows, loudnorm, SR, `format_version` | so a query can NEVER be mixed into an incompatibly-built map |
| cluster profiles | per-cluster centroid, size, human "sound" label | turns cluster IDs into placements a musician understands |

The corpus **is** the fitted map. New songs are *mapped onto it*, never clustered from scratch.

---

## 2. Why this improves clustering of small subsets — and of all new songs

This is the core ML point. Clustering (UMAP + HDBSCAN/Leiden) is a **transductive, whole-batch** operation: its output depends on the density and geometry of *every* point you feed it. Cluster 107 songs and the manifold, the graph, the density estimates are all computed from those 107 points. That is why Phase 2 gives **43% noise** and Phase 1 collapsed to one 94% blob — not enough mass, or the wrong mass, to define structure.

A reference corpus removes four failure modes at once:

1. **It defines the axes of variation.** Clustering only the user's catalog discovers the axes along which *their* songs vary (maybe just "energetic vs. chill"). The axes that matter for playlist placement — genre, instrumentation, mood, era, production — only appear when you sample released music broadly. The corpus defines the **shape of the whole space**; a subset merely occupies a region of it.

2. **It gives density-based methods enough mass.** HDBSCAN estimates density; Leiden builds a kNN graph. Both need enough points per region to know where the real "sounds" are. 100 points spread across all of music look like noise or one lump. 10–50k populate the manifold so neighborhoods are real.

3. **It makes cluster identity stable and named.** Re-cluster 30 songs and "cluster 3" means something different every run — you can't persist "cluster 3 = dreamy lo-fi." Fit the corpus **once**, freeze it, and every upload is *assigned into* the same, named neighborhoods. Two songs uploaded a month apart are directly comparable because the map never moved.

4. **It regularizes / reduces variance.** Small-n clustering overfits sampling noise. Anchoring to a large fixed corpus is a strong prior: a subset (an EP, a batch) becomes a minority perturbation on a well-populated manifold instead of the entire dataset. Add five songs and the map doesn't shift.

**This is the single-cell reference-mapping paradigm your own docs already lean on.** scRNA-seq stopped re-integrating every dataset from scratch and instead built a reference atlas *once*, then mapped query cells onto it (ingest / scArches / scANVI label transfer). Same move here: a MERT reference atlas of released music, with uploads mapped in.

**Two regimes — pick per use case:**

- **Placement regime (default, recommended).** Freeze the reference; map new songs in via `reducer.transform` → kNN/`approximate_predict`. Stable, comparable, O(1)-ish per upload. This is what serves playlist placement.
- **Discovery regime (occasional, deliberate).** To find genuinely new structure the reference lacks (a new microgenre), co-embed new songs *with* the reference and re-cluster — then re-version the map. Even here the reference helps: the new songs perturb a stable manifold rather than defining it. Do this on a cadence, never per upload.

---

## 3. The smartest way to build it — design axes in priority order

### A. Composition/coverage — the single most important decision
The corpus is a **prior over musical space**; its composition *is* the clusters you'll get.
- **Match the deployment target.** "What's out there" for playlist placement = released commercial music across the genres/moods your artists actually make and target. FMA is free and CC-licensed but skews indie/experimental/electronic and under-represents mainstream pop/hip-hop/country — clusters will inherit that skew. State it honestly; don't let FMA's editorial bias masquerade as the shape of music.
- **Coverage beats raw size.** A well-spread 20k across genres/eras/moods beats a lopsided 200k that's 60% lo-fi hip-hop.
- **Cap tracks-per-artist and dedupe near-identical tracks.** Otherwise remixes/one prolific artist create fake dense clusters — a density-method artifact, not a real neighborhood.
- **Densify where your queries live.** If your artists make electronic/pop, over-sample the corpus there so placement resolution is high in the region they care about. Sparse corpus regions give coarse, unreliable placement.

### B. Size
- **10k–50k** is the sweet spot on a laptop/MPS: enough mass for stable structure, exact numpy cosine still fine (per your SIMILARITY note — no FAISS needed under ~100k). Start **fma_small (8k)** to validate the pipeline end-to-end, then scale to medium (~25k).
- Diminishing returns: past some point the space's *shape* stabilizes and more tracks only add density.

### C. Embedding protocol must be byte-for-byte identical, corpus vs. query
This is the class of bug that already bit you (W2 feature-ordering). The corpus **defines** the transform; the query must pass through the *exact same* one: same MERT model, same layer aggregation, same clip length, same windowing, same loudness normalization, same 24 kHz. **Build the corpus with the *final* recipe** (Workstreams E + I) so you never have to re-embed 30k tracks — this is why loudness-norm (I#1) must be decided *before* the corpus build, not bolted on later. Stamp all of it into `embedding_config` and have `SongIndex.load` refuse to mix mismatched configs.

### D. Representation choices specific to MERT
- **Aggregate across layers, not last-layer-only.** MERT's 25 hidden states encode different facets (lower ≈ pitch/timbre, higher ≈ structure/semantics). Building the *foundation* on `last_hidden_state` alone bakes a weaker representation into everything downstream. Use all-layer mean (or a task-weighted sum) — read `n_layers` at runtime (W6/E).
- **Freeze a corpus-fit whitening.** Fit PCA-whitening (e.g. 1024→128) on the corpus, freeze it, apply to queries. It decorrelates and denoises before the graph is built, and gives queries a transform to reuse. For MERT, L2+cosine is often strong on its own — make whitening a flag and let the Workstream F eval decide.
- **Treat production as a batch effect (Workstream I).** MERT sees production. If the corpus is lo-fi FMA and queries are studio-mastered demos, the dominant axis separating query from corpus may be **production, not music** — a demo lands in "lo-fi" because FMA is lo-fi, not because the song is. Loudness-normalize corpus *and* query identically (I#1, cheap, do it always); consider a Harmony-style correction over a production covariate only if F's neighbor audits still show production-driven structure.

### E. Assignment mechanics (the payoff)
- **Similarity:** query MERT vec → same transform → cosine kNN over corpus → "your song sits nearest these *released* tracks." Playlist placement = inspect which clusters/playlists those neighbors belong to.
- **Cluster assignment:** project into the frozen UMAP/graph, assign to the nearest reference community (kNN vote or `approximate_predict`) → "your song belongs to the dreamy-electronic neighborhood," stable across uploads.

### F. Maintenance
The corpus is a snapshot of "what's out there," and music drifts. **Additive extension** (embed new tracks, append, transform frozen) is safe and keeps cluster IDs stable. **Refitting the transform** re-labels clusters, so do it on a cadence and bump `format_version` — never per upload. Anchor-stability vs. freshness is the tradeoff.

---

## 4. Concrete build path in this repo

1. **Source audio.** Two good options:
   - **FMA (small→medium→large)** — already downloaded and path-conventioned (`get_audio_path`). Fast to start; inherits FMA's indie skew.
   - **Deezer 30 s previews** (`anther_ml/spotify_deezer.py` already exists) — *smarter for the product*: curatable to match target playlists/mainstream music, and previews are uniformly 30 s, which solves the clip-length consistency issue (W8) for free.
2. **Embed once with the final recipe** (all-layer aggregation + multi-window + loudness-norm, 24 kHz). 8k on MPS ≈ 2–3 hrs; a 25k medium corpus is a real compute job — candidate for remote GPU.
3. **Fit + freeze the bundle:** scaler/PCA-whiten → kNN graph → Leiden (sweep `resolution` per Workstream D) → 2D UMAP; compute per-cluster profiles.
4. **Run the Workstream F eval** (known-related-pairs recall, cluster stability ARI, silhouette, neighbor-audit export) *before and after* so the corpus's quality is a number, not a vibe.
5. **Save as a versioned, self-describing index** with the `embedding_config` stamp; wire the Phase-2 query path to standardize/transform through the frozen corpus stats.

### Sequencing note
Build the corpus with the **final** embedding + loudness recipe (E + I#1) already in place. Re-embedding tens of thousands of tracks later just to add loudness normalization or all-layer aggregation is the expensive mistake this doc exists to prevent.

---

## 5. Workflow: EP-to-playlist fit (the primary product loop)

This is the concrete workflow the corpus is built to serve. **It is the placement regime (§2), applied to a playlist.**

**The steps:**
1. Corpus of ~100k across time/genre/fidelity → defines the coordinate system + frozen whitening/graph/cluster model. (At 100k, exact numpy cosine is still fine — ~400 MB float32, fast per query. Top edge before FAISS would help; not needed yet.)
2. A playlist, all of whose songs are already embedded in the corpus → the playlist is a **labeled region** of the frozen map, not a separate thing.
3. An EP, embedded through the **same frozen transform** → EP songs get coordinates in the identical space.
4. Compare EP ↔ playlist over just those ~50 songs → **subset the frozen map and measure similarity there.**

### The critical distinction — "examine only these ~50" must mean *filter*, not *re-cluster*
- **Correct (filter the frozen map):** EP + playlist songs are embedded through the corpus's frozen transform, so they already have fixed coordinates in the 100k-defined space. Subset the *view* to those ~50 points and read similarities/positions off it. The coordinates were set by the corpus and do not move.
- **Wrong (re-cluster the subset):** feeding only the ~50 songs into a fresh UMAP/HDBSCAN/Leiden discards the corpus and returns to the unstable small-n regime (the 43%-noise failure). **Never re-fit on the subset.**

The corpus contextualizes the difference (it computes the map from 100k); you only *examine* a corner of it. That corner-view is a filter over frozen coordinates.

### Metadata addition required
The corpus metadata (§1) needs **playlist membership** — a multi-valued tag per track (a track can be on several playlists). This is the only schema change the playlist workflow adds.

### "Match well" — a calibrated fit score, not a raw cosine
A raw cosine of 0.72 means nothing on its own. The corpus turns it into a calibrated statement:
- **Raw fit:** `fit(ep_song → playlist)` = mean (or top-k mean, e.g. k=5) cosine from the EP song to the playlist members, in the whitened corpus space.
- **Calibrate against the corpus as a null distribution:** compute that same score for a few thousand *random corpus tracks* → the distribution of "how well an arbitrary released song matches this playlist." The EP song's **percentile** against that distribution is the headline metric: *"fits this playlist better than 94% of released music."* This sentence is only possible because the 100k corpus defines "typical" — that is the contextualization, made numeric.
- **Cluster co-location (coarser sanity layer):** which frozen communities the playlist occupies, and whether the EP songs assign into those same ones.

Without the corpus you can still compute the 50×50 cosines, but you cannot say whether any of them are *good* — there is no reference for "close." The corpus **is** that reference distribution.

### Visualization — a literal picture of the workflow
Plot the frozen 2D UMAP of the whole ~100k corpus as a grey backdrop; color the playlist points; overlay the EP points. "Does my EP land inside the playlist's cloud?" becomes visible. The grey backdrop **is** "the corpus contextualizes the difference"; the colored overlay **is** "examine only these few." One figure — the honest version of the pitch.

### Generalization (free, and it's the real product)
Since every playlist lives in the same frozen corpus, score an EP against *all* playlists and rank them — "which playlist does this song fit best" is the same computation run N times. The EP-to-one-playlist loop is just the N=1 case.

### Caveat to state, not hide
This measures whether the EP is **acoustically consistent** with the playlist's sound. Real placement also depends on editorial/curatorial factors MERT cannot see (lyrics, artist stage, release timing). The corpus answers "does it sonically belong" — the part that is actually computable. Don't let it silently stand in for the whole placement decision.

---

## 6. One-line verdict
A MERT reference corpus is a **frozen map of released music** — embeddings + the fitted whitening/graph/cluster model + a config stamp — fit once on a broad, deduped, deployment-matched sample. It turns clustering from an unstable, catalog-relative operation on 100 songs into stable *placement onto a fixed coordinate system*, which is exactly what "see how my song compares to what's out there / where it fits on a playlist" requires.
