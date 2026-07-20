# CLAUDE.md

Guidance for Claude Code working in this repository. This file is a router —
detailed guidance lives in topic files under `docs/`. Read the relevant one
before working in that area. New to this repo? Start at
[docs/first-time.md](docs/first-time.md).

## Model routing

Before starting non-trivial work, dispatch to `triager` (Haiku) to classify complexity:
- **trivial** → handle directly or dispatch `implementer` (Sonnet)
- **moderate** → dispatch `implementer` (Sonnet)  
- **complex** → dispatch `architect` (Opus)

This minimizes token cost by routing simple tasks to cheaper models and reserving expensive models for genuinely complex work.

## Environment

```bash
source venv/bin/activate
pip install -e .          # editable install (pyproject.toml) — preferred
pip install -e '.[dev]'   # + pytest
```

`import anther_ml` works from anywhere after the editable install; the
`sys.path.insert(0, '..')` in the notebooks is no longer required (harmless if
left). Run the test suite with `pytest` from the repo root.

## Documentation map

| Topic | Read when… | File |
|---|---|---|
| Architecture & script index | you need the big picture or "where does X live" | [docs/architecture.md](docs/architecture.md) |
| **Key invariants** | **before changing anything** — the silent-correctness rules | [docs/invariants.md](docs/invariants.md) |
| Clustering (Leiden) | touching `cluster.py` or cluster assignment | [docs/clustering.md](docs/clustering.md) |
| Similarity index | touching `similarity.py` / `SongIndex` / index files | [docs/similarity.md](docs/similarity.md) |
| Phase 1 features | touching `features.py` / `data.py` / FMA feature vectors | [docs/phase1-features.md](docs/phase1-features.md) |
| Phase 2 embeddings | touching `embedding.py` / `audio.py` / MERT | [docs/phase2-embeddings.md](docs/phase2-embeddings.md) |
| Reference corpus | touching `anther_ml/corpus/` (build/place/sources/bundle) or MPD ingest | [docs/projects/REFERENCE_CORPUS_DESIGN.md](docs/projects/REFERENCE_CORPUS_DESIGN.md), perf: [docs/projects/CORPUS_BUILD_EFFICIENCY_PLAN.md](docs/projects/CORPUS_BUILD_EFFICIENCY_PLAN.md) |
| MPD playlist DB pruning | touching `anther_ml/mpd_sql.py` or shrinking the 27GB MPD DB for hosting | [docs/projects/MPD_PLAYLIST_PRUNING_NOTES.md](docs/projects/MPD_PLAYLIST_PRUNING_NOTES.md) (not implemented — reference only) |
| Cluster labels | touching `anther_ml/corpus/labels.py` or the `corpus label` CLI | [docs/projects/PLAYLIST_LABELS_BUILD_PLAN.md](docs/projects/PLAYLIST_LABELS_BUILD_PLAN.md) |
| Micro-genre tagging | touching `anther_ml/corpus/tagging/` (per-track genre tags) | [docs/tagging.md](docs/tagging.md); build plan: [docs/projects/MICROGENRE_TAGGING_BUILD_PLAN.md](docs/projects/MICROGENRE_TAGGING_BUILD_PLAN.md), status: [docs/projects/MICROGENRE_TAGGING_NEXT_STEPS.md](docs/projects/MICROGENRE_TAGGING_NEXT_STEPS.md) |
| Web UI | touching `ui/` (Flask app, corpus atlas, force-graph frontend) | [docs/ui.md](docs/ui.md) |
| Evaluation | measuring a change or picking a hyperparameter | [docs/evaluation.md](docs/evaluation.md) |
| Notebooks, viz & Jupyter | running the pipeline notebooks or `export_viz.py` | [docs/notebooks.md](docs/notebooks.md) |

## Critical rules (details in [docs/invariants.md](docs/invariants.md))

- **Never use genre** as a clustering input, training signal, or eval metric for the map — display only. (Sole carve-out: the display-only tag probe in `corpus/tagging/`; its metrics never tune the map.)
- **Cluster the embedding, not UMAP/t-SNE coordinates** — 2D is viz only.
- **Standardize before cosine** (Phase 1); **align features by label** (`align_to_corpus`) before comparing.
- **Query and corpus must share the same transform/config**; don't mix indices built differently.

> The `anther_ml/corpus/` subpackage (frozen reference-corpus bundles) is
> functional — see [docs/projects/REFERENCE_CORPUS_DESIGN.md](docs/projects/REFERENCE_CORPUS_DESIGN.md) for
> the design and `python -m anther_ml.corpus --help` for the build/place/label
> CLI. The primary bundle is the frozen 100k MPD corpus at
> `models/corpus_corpus_mpd_100k/` (also: `corpus_mpd_val_25k`, smoke bundles).
> Micro-genre tagging (`python -m anther_ml.corpus.tagging`) has seed/fit/predict
> done on the 100k bundle; held-out FMA eval is pending the FMA download — see
> [docs/projects/MICROGENRE_TAGGING_NEXT_STEPS.md](docs/projects/MICROGENRE_TAGGING_NEXT_STEPS.md).
