# anther-ml

Two-phase music similarity and clustering pipeline.

- **Phase 1** — pre-computed librosa features (fast, no GPU), matching FMA's 518-dim schema.
- **Phase 2** — MERT neural embeddings (`m-a-p/MERT-v1-330M`, 1024-dim).

Both phases share the same clustering and similarity API in `anther_ml`; only the
input embeddings differ. See `CLAUDE.md` for the full architecture and notebook
pipeline order.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .            # editable install — no more sys.path hacks
pip install -e '.[dev]'     # + pytest for the test suite
```

The editable install replaces the `sys.path.insert(0, '..')` used by the notebooks.

`data/` and `models/` are gitignored and large (tens of GB once populated —
raw audio, FMA metadata, reference-corpus bundles). Cloning this repo gets
you code only, not a working corpus; see
[docs/projects/REFERENCE_CORPUS_DESIGN.md](docs/projects/REFERENCE_CORPUS_DESIGN.md)
for how a bundle is built. There is currently no shortcut path to a
pre-built bundle for a new collaborator — ask the repo owner.

## Web UI

Flask + d3 song atlas — search, place songs/playlists/albums onto a frozen
reference corpus, browse the resulting map:

```bash
python ui/app.py   # http://localhost:5000
```

Requires a built corpus bundle (`ANTHER_CORPUS`, default
`models/corpus_corpus_mpd_100k`) and, for full-MPD playlist/album search, an
MPD SQLite DB (`ANTHER_MPD_DB`). See [docs/ui.md](docs/ui.md) for routes,
env vars, and session-state details.

## Tests

```bash
pytest
```

## Evaluation

Every design choice is measured by the genre-free eval harness (Workstream F):

```bash
python -m anther_ml.eval --index models/index_phase1
```

## Visualize

`export_viz.py` bakes the current index + 2D embedding into a self-contained
`song_view.html` (open it in a browser):

```bash
python export_viz.py    # PHASE is set at the top of the file
```

Re-run it after rebuilding an index — the HTML does not read `models/` live.

## Regenerating artifacts

Model pickles and indices in `models/` are derived (gitignored). Regenerate them
by running the notebooks in order (`01` → `05`); see the table in `CLAUDE.md`.
The clustering/similarity notebooks (`01`, `02`, `04`, `05`) use the Leiden +
standardized-index stack; `03` is an educational spectrogram demo.
