# CLAUDE.md

Guidance for Claude Code working in this repository. This file is a router —
detailed guidance lives in topic files under `docs/`. Read the relevant one
before working in that area.

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
| Evaluation | measuring a change or picking a hyperparameter | [docs/evaluation.md](docs/evaluation.md) |
| Notebooks, viz & Jupyter | running the pipeline notebooks or `export_viz.py` | [docs/notebooks.md](docs/notebooks.md) |

## Critical rules (details in [docs/invariants.md](docs/invariants.md))

- **Never use genre** as a clustering input, training signal, or eval metric — display only.
- **Cluster the embedding, not UMAP/t-SNE coordinates** — 2D is viz only.
- **Standardize before cosine** (Phase 1); **align features by label** (`align_to_corpus`) before comparing.
- **Query and corpus must share the same transform/config**; don't mix indices built differently.

> A `corpus/` subpackage is under construction and intentionally undocumented.
