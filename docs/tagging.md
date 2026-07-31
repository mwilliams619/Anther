# Micro-genre tagging (`anther_ml/corpus/tagging/`)

Per-track micro-genre tags on top of an **already-built** frozen corpus
bundle — display only, never fed back into the map (see
[invariants.md](invariants.md)). Multi-label: each track can carry several
tags drawn from the everynoise top-2000 vocabulary, each with a score and a
primary/secondary distinction.

## Two-stage supervision

- **Stage A — playlist weak labels.** Match each track's own playlist names
  against the vocabulary → noisy but in-vocab, in-distribution per-track seed
  labels (`vocab.py`, `weak_labels.py`).
- **Stage B — MERT probe + propagation.** A frozen multi-label probe
  (`probe.py`) trains on the confident seed subset, predicts tags for every
  track, then kNN-propagates in MERT space (`build_tags.py`) to fill tracks
  with no playlist match and to denoise Stage A.

## Modules

| Module | Responsibility |
|---|---|
| `vocab.py` | everynoise vocabulary loading + matching helpers |
| `weak_labels.py` | Stage A — playlist-name → seed tag matching |
| `probe.py` | Stage B — frozen multi-label probe (own `StandardScaler`, fit/predict) |
| `build_tags.py` | Orchestrates fit → predict → kNN propagation → writes bundle tag artifacts |
| `crosswalk.py` | FMA genre → everynoise vocab mapping, used only by held-out eval |
| `evaluate.py` | Held-out FMA evaluation — walled off from `build_scorecard` |
| `__main__.py` | `python -m anther_ml.corpus.tagging` CLI (`seed`/`fit`/`predict`/`embed-eval`/`evaluate`) |

## The wall

Tag-probe metrics (`evaluate.py`) judge the probe against itself; they must
never tune the MERT embedding, `SongIndex`, the Leiden partition, or
`anther_ml.eval`'s scorecard — that's the repo's genre carve-out, spelled out
in [invariants.md](invariants.md). Two scoreboards, one wall.

## Where things stand / full design

- Build rationale, two-stage design, constraints: [projects/MICROGENRE_TAGGING_BUILD_PLAN.md](projects/MICROGENRE_TAGGING_BUILD_PLAN.md)
- Live status, what's done vs. blocked, next steps: [projects/MICROGENRE_TAGGING_NEXT_STEPS.md](projects/MICROGENRE_TAGGING_NEXT_STEPS.md)
