# Literature Review — What scRNA-seq Clustering Can Teach the Music-Similarity Pipeline

_Scope: canonical single-cell RNA-seq (scRNA-seq) clustering/integration methods, and which of their techniques transfer to `anther-ml`'s problem — meaningful, label-free clustering of noisy high-dimensional audio embeddings. All papers below were retrieved and DOI-verified via OpenAlex; citation counts are as of retrieval._

## Why this field is the right analogy

Your instinct is correct, and the analogy is unusually tight. scRNA-seq measures ~20,000 gene counts per cell across thousands–millions of cells. The data is **noisy, extremely high-dimensional, has no ground-truth labels, and the goal is to recover meaningful groups (cell types/states) that emerge from the data rather than from pre-assigned categories.** That last point matters for you specifically: single-cell clustering is *definitionally unsupervised* — cell populations are discovered from expression similarity, then characterized after the fact. It never clusters "to a label." That is exactly the genre-free philosophy you asked for, and the field has spent a decade hardening the machinery to do it well.

Two community "best-practices" syntheses define the consensus pipeline and are worth reading in full: Luecken & Theis 2019 (`10.15252/msb.20188746`, 2,367 cites) and its updated successor Heumos et al. 2023, *Nature Reviews Genetics* (`10.1038/s41576-023-00586-w`, 1,056 cites).

**The consensus scRNA-seq pipeline:**
`QC → normalize + variance-stabilize → select informative features → linear reduction (PCA) → build a k-NN graph → graph community detection (Leiden) → non-linear embedding (UMAP) for visualization only`

Compare to your current chain: `scale → PCA → UMAP(32d) → HDBSCAN → 2D-UMAP`. The two differ at precisely the steps that are failing for you. The rest of this document walks the transferable techniques in priority order.

---

## Tier 1 — Directly fixes your degenerate-clustering problem

### 1. Graph-based community detection (Leiden) instead of HDBSCAN on a UMAP
**Papers:** Leiden — Traag, Waltman & van Eck 2019, *Scientific Reports* (`10.1038/s41598-019-41695-z`, 5,246 cites). Implemented as the default clusterer in SCANPY (Wolf et al. 2018, *Genome Biology*, `10.1186/s13059-017-1382-0`, 9,365 cites) and Seurat (Hao et al. 2023, *Nature Biotechnology*, `10.1038/s41587-023-01767-y`, 4,880 cites).

**The single most important transfer.** scRNA-seq abandoned both density clustering *and* k-means in favor of **community detection on a k-nearest-neighbor graph**. You build a graph where each point links to its k nearest neighbors in embedding space, then partition the graph into communities that are internally dense and externally sparse. This is now the default in every major single-cell toolkit.

Why it fixes your specific failure (94% of tracks in one cluster, 5% "noise"):
- **No giant catch-all cluster and no noise bucket.** Graph partitioning assigns every node to a community; it does not have HDBSCAN's failure mode of declaring one mega-cluster plus outliers. Your 5,698 "noise" tracks would instead be placed.
- **A single, interpretable knob: resolution.** The resolution parameter directly controls granularity — low resolution → few broad clusters, high → many fine ones. This is far more controllable than juggling HDBSCAN's `min_cluster_size`/`min_samples` on a UMAP projection.
- **Leiden specifically over Louvain.** Louvain (the older method) can produce internally *disconnected* "communities" — a real defect. Leiden's guarantee is in its title: it guarantees well-connected communities, converges faster, and gives better partitions. If you use graph clustering, use Leiden, not Louvain.

**Action for `anther-ml`:** replace the HDBSCAN step with Leiden on a k-NN graph built from the MERT (or scaled-librosa) embedding. `scanpy`'s `sc.pp.neighbors` + `sc.tl.leiden` is the reference implementation; `leidenalg` + `igraph` is the standalone. This also removes the metric-mismatch issue I flagged in the fix plan (UMAP-cosine feeding HDBSCAN-euclidean).

### 2. Cluster the graph, never the UMAP coordinates
**Paper:** Becht et al. 2018, *Nature Biotechnology* — "Dimensionality reduction for visualizing single-cell data using UMAP" (`10.1038/nbt.4314`, 5,765 cites); cautionary counterpart for t-SNE, Kobak & Berens 2019, *Nature Communications* (`10.1038/s41467-019-13056-x`, 1,138 cites).

The field is unanimous and explicit: **UMAP/t-SNE are for visualization only.** Clustering is done on the high-dimensional embedding (or its PCA/latent reduction), never on the 2D coordinates, because 2D projections distort local density and inter-cluster distances — they can fuse or split groups that aren't really fused or split. Your pipeline currently runs HDBSCAN on a 32-d UMAP, which is the middle-ground version of this mistake. Move clustering onto the k-NN graph in the embedding space and keep 2D-UMAP strictly as the picture. (This is the same conclusion as Workstream D1 in your fix plan, now with the field's citation weight behind it.)

### 3. Consensus / stability as the correctness criterion for small corpora
**Paper:** SC3 — Kiselev et al. 2017, *Nature Methods* (`10.1038/nmeth.4236`, 1,661 cites).

SC3 established **consensus clustering** for scRNA-seq: cluster many times across parameter settings / subsamples and keep the partition the runs agree on. It was designed for *small-n* datasets — which is exactly your Phase-2 situation (107 tracks). The practical takeaways: (a) a single clustering run is not trustworthy on small data; (b) the agreement across runs (Adjusted Rand Index across bootstraps/seeds) is itself the quality signal. This is the rigorous version of the "cluster stability" metric in your fix plan's evaluation harness — and it's fully label-free.

---

## Tier 2 — Fixes your feature-scaling problem (and validates the MERT direction)

### 4. Normalization + variance stabilization (your Hz-scale problem, solved upstream)
**Paper:** sctransform — Hafemeister & Satija 2019, *Genome Biology* (`10.1186/s13059-019-1874-1`, 4,967 cites).

scRNA-seq has the identical pathology I measured in your Phase-1 index: a handful of features (highly expressed genes / your Hz-scale spectral dimensions) dominate by raw magnitude and swamp the informative low-magnitude ones. The field's answer is **variance stabilization** — a principled transform (regularized negative-binomial regression, in sctransform's case) so that no feature dominates the distance metric purely by scale. This is the same fix as Workstream A (standardize before similarity), but the scRNA-seq literature frames it as a first-class, must-do pipeline stage rather than an afterthought — reinforcing that it belongs *before* both clustering and similarity, on a persisted transform shared by corpus and query.

### 5. Learned latent embeddings (scVI) — MERT is already your "scVI"
**Paper:** scVI — Lopez et al. 2018, *Nature Methods*, "Deep generative modeling for single-cell transcriptomics" (`10.1038/s41592-018-0229-2`, 2,662 cites); denoising autoencoder precedent, Eraslan et al. 2019 (`10.1038/s41467-018-07931-2`, 1,160 cites).

scVI trains a deep generative model to compress noisy 20k-dim counts into a smooth ~10–30-dim latent space, then does **all downstream clustering on that latent space** rather than on raw features. This is the strongest conceptual endorsement of your Phase-2 design: **MERT embeddings are the audio analogue of the scVI latent space** — a learned, denoised, semantically meaningful representation. The lesson is directional: invest in clustering the neural embedding well (Tier 1), rather than over-engineering the hand-crafted librosa feature path. The single-cell field largely moved *from* hand-crafted-feature clustering *to* learned-latent clustering over exactly this period.

---

## Tier 3 — The non-obvious insight: production is your "batch effect"

### 6. Batch-effect correction (Harmony) — remove the nuisance variable you *don't* want to cluster on
**Papers:** Harmony — Korsunsky et al. 2019, *Nature Methods* (`10.1038/s41592-019-0619-0`, 10,591 cites); benchmarks establishing why it matters — Tran et al. 2020, *Genome Biology* (`10.1186/s13059-019-1850-9`, 1,209 cites) and Luecken et al. 2021, *Nature Methods*, "Benchmarking atlas-level data integration" (`10.1038/s41592-021-01336-8`, 1,429 cites).

This is the technique most worth stealing that isn't already on your radar. In scRNA-seq, a **batch effect** is technical variation — which machine, day, lab, or chemistry produced a sample — that makes cells cluster by *batch* instead of by *biology*. Harmony (and scVI's conditional variant) learn to remove a nominated nuisance factor from the embedding so the remaining structure reflects the signal you care about.

**The music parallel is direct and, I think, a genuine finding for this project.** Your audio has its own batch effects: mastering loudness, mix style, codec/bitrate, recording era, lo-fi vs. studio production. Two songs can land near each other because they were both loudly mastered or both low-bitrate MP3s — not because they are musically similar. That is precisely the failure mode my Phase-1 scale analysis surfaced (loudness/brightness dominating the cosine score). The Harmony framing gives a principled response: **identify production/loudness as a nuisance variable and correct the embedding to factor it out**, so clusters reflect musical content rather than production polish. Concretely, this could mean loudness-normalizing audio before embedding (EBU R128 / ReplayGain), and/or a Harmony-style correction over a "production" covariate if you can tag one. For a tool whose job is placing songs by musical fit, removing the production confound is high-value.

### 7. Feature selection (highly variable genes) — prune uninformative dimensions
Covered in the best-practices tutorials above. scRNA-seq selects the top ~2,000 most variable genes before clustering, discarding dimensions that carry no discriminative signal. For your **librosa** path this maps to dropping low-variance/redundant features before building the graph; for the **MERT** path it's less critical (learned dims are already informative), though a variance filter or PCA whitening before graph construction is still standard and cheap.

---

## Methods considered and de-prioritized

- **PhenoGraph** (Levine et al. 2015, *Cell*, `10.1016/j.cell.2015.05.047`, 2,566 cites) and **PARC** (Stassen et al. 2020, *Bioinformatics*, `10.1093/bioinformatics/btaa042`, 152 cites) are k-NN-graph + community-detection clusterers — same family as recommendation #1. PARC is engineered for millions of points; at your <10k scale, plain Leiden in SCANPY is simpler and sufficient. Worth knowing if you scale to a large reference corpus later.
- **t-SNE** (Kobak & Berens): use UMAP over t-SNE for the 2D view — UMAP better preserves global structure, and the t-SNE paper is cited here mainly as evidence that projection choice materially changes the picture.
- **Doublet detection, cell-type marker annotation, RNA velocity, trajectory inference**: single-cell-specific and not transferable.

---

## Concrete recommendations for `anther-ml`, in priority order

1. **Swap HDBSCAN → Leiden community detection on a k-NN graph** built in the embedding space (SCANPY `sc.pp.neighbors` + `sc.tl.leiden`, or `leidenalg`+`igraph`). Tune the single `resolution` parameter. This is the highest-leverage change and directly targets the 94%-in-one-cluster degeneracy. *(Extends Workstream D of the fix plan.)*
2. **Cluster the graph, never the UMAP.** Demote UMAP to visualization only. *(Confirms Workstream D1.)*
3. **Variance-stabilize / standardize features on a persisted transform** shared by corpus and query, before both clustering and similarity. *(This is Workstream A, with the scRNA-seq literature as backing.)*
4. **Adopt consensus/stability (ARI across bootstraps) as the clustering quality metric** — essential at Phase-2's n=107. *(This is Workstream F2.)*
5. **Treat production/loudness as a batch effect**: loudness-normalize audio before embedding, and consider a Harmony-style correction over a production covariate. *(New — not in the prior fix plan.)*
6. **Keep leaning on MERT as your learned latent space** (the scVI lesson); prune the librosa feature path rather than expand it.

Every one of these is unsupervised and label-free — no genre anywhere — which is exactly why the single-cell playbook fits your goal.

## References (DOI-verified via OpenAlex)
| # | Method | First author, year | Venue | DOI |
|---|---|---|---|---|
| 1 | Leiden community detection | Traag 2019 | Sci. Rep. | 10.1038/s41598-019-41695-z |
| 2 | SCANPY (toolkit, Leiden default) | Wolf 2018 | Genome Biol. | 10.1186/s13059-017-1382-0 |
| 3 | Seurat v5 (dictionary learning) | Hao 2023 | Nat. Biotechnol. | 10.1038/s41587-023-01767-y |
| 4 | Best-practices tutorial | Luecken & Theis 2019 | Mol. Syst. Biol. | 10.15252/msb.20188746 |
| 5 | Best practices (update) | Heumos 2023 | Nat. Rev. Genet. | 10.1038/s41576-023-00586-w |
| 6 | UMAP for single-cell viz | Becht 2018 | Nat. Biotechnol. | 10.1038/nbt.4314 |
| 7 | t-SNE cautions | Kobak & Berens 2019 | Nat. Commun. | 10.1038/s41467-019-13056-x |
| 8 | SC3 consensus clustering | Kiselev 2017 | Nat. Methods | 10.1038/nmeth.4236 |
| 9 | sctransform normalization | Hafemeister 2019 | Genome Biol. | 10.1186/s13059-019-1874-1 |
| 10 | scVI latent embedding | Lopez 2018 | Nat. Methods | 10.1038/s41592-018-0229-2 |
| 11 | DCA denoising | Eraslan 2019 | Nat. Commun. | 10.1038/s41467-018-07931-2 |
| 12 | Harmony batch integration | Korsunsky 2019 | Nat. Methods | 10.1038/s41592-019-0619-0 |
| 13 | Batch-correction benchmark | Tran 2020 | Genome Biol. | 10.1186/s13059-019-1850-9 |
| 14 | Atlas integration benchmark | Luecken 2021 | Nat. Methods | 10.1038/s41592-021-01336-8 |
| 15 | PhenoGraph | Levine 2015 | Cell | 10.1016/j.cell.2015.05.047 |
| 16 | PARC (scalable graph clustering) | Stassen 2020 | Bioinformatics | 10.1093/bioinformatics/btaa042 |
