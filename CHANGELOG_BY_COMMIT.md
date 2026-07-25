# Change log by commit

This is a feature-oriented record of the current Anther ML/UI development
lineage, from 2026-06-27 onward. Each entry is tied to one Git commit and is
limited to three sentences; older 2023–2024 commits belong to the superseded
legacy web application and are not repeated here.

## 5b125fb5 — 2026-06-27 — Add ML pipeline: Phase 1 (librosa) + Phase 2 (MERT) clustering

Established the core `anther_ml` pipeline: labeled Phase 1 librosa features,
Phase 2 MERT embeddings, clustering, similarity search, notebooks, and initial
developer commands. Added the first repository-level architecture and
invariant guidance.

## b66e8330 — 2026-07-03 — ml development first commit

Expanded the ML architecture with audio loading, improved clustering, MERT
embedding support, evaluation direction, and reference-corpus design. The
documentation began formalizing the genre-free and corpus-frozen workflow.

## 1f6fca74 — 2026-07-03 — adde corpus building work

Added the frozen reference-corpus package, including corpus sources, bundle
serialization, build tooling, placement, and the corpus CLI. This introduced a
stable coordinate system for mapping new songs instead of reclustering each
query set.

## 62d2c724 — 2026-07-08 — adding genre metadata labeling

Added the playlist-name cluster-labeling plan and micro-genre tagging plan,
along with corpus CLI and build-path changes needed to support display-only
metadata. The design explicitly kept labels and tags outside clustering and
evaluation inputs.

## 2d19cc6a — 2026-07-08 — updated ui and did unsupervised genre labeling

Implemented the corpus tagging package: vocabulary handling, weak labels,
probe fitting, prediction, propagation, crosswalks, evaluation, and CLI
commands. Placement began returning display-only tags while preserving the
frozen embedding and clustering regime.

## c9707852 — 2026-07-09 — Remove dead staging/clustering backend — atlas is now the sole UI

Removed the obsolete staging/clustering UI backend and consolidated the web
application around `ui/atlas.py`. This reduced duplicate UI paths and made the
atlas the single application surface.

## e88f34ff — 2026-07-09 — Atlas UI: full-MPD playlist/album placement, warm-up gating, import cap

Added full-MPD playlist search and asynchronous playlist/album placement,
including the background job worker and a configurable import cap. Added corpus
warm-up readiness handling, graph/UI tests, and the supporting MPD preparation
path.

## e9c2c373 — 2026-07-09 — Tag Deezer-sourced tracks correctly in place_external_track

Corrected source tagging for externally placed Deezer tracks so downstream
metadata and graph behavior identify their origin correctly.

## 8d0aaa05 — 2026-07-09 — Reorganize docs: tagging.md, ui.md, first-time.md, group plans under docs/projects/

Reorganized the documentation into topic guides and project plans, and added
the repository guidance/router structure. This established the current docs
navigation model used by contributors.

## 71eddbd5 — 2026-07-09 — Repo hygiene: fix collaborator setup gaps, stop tracking derived/session files

Removed tracked derived artifacts, generated views, and session state from the
repository. This made the source tree reproducible without committing local
runtime output.

## a7e97635 — 2026-07-09 — Complete previous hygiene commit (pyproject/README/gitignore/debug-flag)

Completed repository setup with editable-install metadata, README startup
instructions, ignore rules, and the Flask debug configuration. The documented
development workflow became `pip install -e .` plus the repository test suite.

## 9dc861e5 — 2026-07-10 — added chat interface and ui overhaul

Added the mentor service documentation and initial web-chat integration plan,
along with multi-song recommendation, MPD-pruning, mentor-v3, and cleanup
planning documents. The UI documentation was expanded to describe the broader
atlas and mentor architecture.

## f4ef8bcb — 2026-07-12 — big ui update with test and reclustering to higher resolution

Delivered a major atlas/UI update with higher-resolution reclustering, tests,
mentor graph documentation, demo/alpha materials, and the playlist stop-button
implementation. It also added cluster-label application tooling and expanded
the project inventory.

## 85a56902 — 2026-07-15 — new gitignore loading animation

Added the loading animation and improved warm-up behavior across the Flask app,
atlas, playlist jobs, force graph, and static shell. The UI now communicates
corpus initialization instead of presenting an unexplained blank state.

## 32016569 — 2026-07-15 — demo playlists added

Added the curated demo playlist artifact and its supporting upload/session
handling. This made the atlas demo flow reproducible without manually finding a
playlist first.

## d3692e24 — 2026-07-15 — Refactor mentor agent: LLM intent classifier replaces deterministic regex

Replaced the earlier mentor routing structure with an LLM-backed intent
classifier, a service-oriented agent, graph tools, and updated mentor demos and
documentation. The refactor removed the older ReAct/graph modules in favor of
the current agent and tool organization.

## c03c5176 — 2026-07-16 — Fix mentor graph blindness: session-aware GraphContext + graph-aware routing

Added session-aware graph context and graph-aware mentor routing so follow-up
questions can use the current atlas state. Added mentor routing/live-graph tests,
service forwarding changes, and corresponding UI documentation.

## f2d8f7b0 — 2026-07-16 — Mentor: deterministic selection answers, terser connections, stricter narrator

Improved mentor answers for selected graph entities, shortened connection
descriptions, and tightened narrator behavior. Added focused agent tests and
exposed the related service behavior through the UI layer.

## 281323ee — 2026-07-18 — Revert dual-mode D3 changes (graph.js/app.js/index.html) - kept artist backend

Reverted the original full-corpus dual-mode artist frontend while retaining the
artist backend and associated experimental artifacts. The later incremental
Artist View superseded this intermediate state.

## 0857652c — 2026-07-19 — untracked files on labeling-try-ml-dev: f2d8f7b0 Mentor: deterministic selection answers, terser connections, stricter narrator

Captured the in-progress branch state containing the alternate UI module and
static application file. This is repository-history bookkeeping rather than a
separate user-facing feature.

## 87ebc418 — 2026-07-19 — index on labeling-try-ml-dev: f2d8f7b0 Mentor: deterministic selection answers, terser connections, stricter narrator

Recorded the companion branch index state with additional atlas and static-style
changes. This is an integration snapshot supporting the subsequent MERIT/UI
work rather than a standalone feature release.

## 64418058 — 2026-07-19 — On labeling-try-ml-dev: merit changes

Integrated the first MERIT-oriented atlas changes, including score display and
static UI support. This commit is an intermediate branch snapshot; the scoring
behavior was refined by the following MERIT commits.

## 51330bc2 — 2026-07-20 — fixed_merit_scoring

Added MERIT factor calibration and aggregate-score handling across the corpus
bundle, atlas, API, and UI. Melody, rhythm, timbre, and aggregate scores now
share explicit calibrated display semantics and are covered by expanded tests
and runtime verification.

## 4f00dab3 — 2026-07-19 — Merge pull request #2 from mwilliams619/graph-dual-mode-revert

Merged the graph-dual-mode revert line and its associated repository guidance,
status material, inventories, and deployment-era documentation. This merge
preserved the then-current song atlas while leaving artist work available for a
later implementation path.

## 84dd1244 — 2026-07-20 — ci/cd setup

Added staging and stable deployment workflows, shell deployment scripts,
systemd service definitions, and CI/CD plus rollout documentation. The project
now has documented operational paths for both staging and stable environments.

## a23a74b0 — 2026-07-21 — added artist clustering view

Implemented the incremental per-session Artist View with artist search, artist
graph rendering, artist placement, upload associations, and graph tests. The
feature uses stable corpus/session IDs and keeps the song atlas as a separate
view.

## a517cebd — 2026-07-21 — increase score floor to 35

Raised the UI similarity-link score floor to reduce weak visual connections and
updated the related display text. This was a presentation/calibration threshold
change, not a change to the underlying embedding space.

## 012d73f2 — 2026-07-21 — Merge pull request #3 from mwilliams619/add-artist-clustering

Merged the incremental artist-clustering implementation and its tests,
frontend graph, backend routes, and documentation. This merge represents the
current Artist View baseline rather than the older reverted dual-mode design.

## bd773da9 — 2026-07-22 — aesthetic updates

Refined atlas presentation and interaction details, including artist-view
polish, recommendation display behavior, graph styling, and artist enrichment
integration. Added the first durable artist-profile layer backed by Deezer and
MusicBrainz data and exposed optional profiles through artist detail responses.

## Working tree — uncommitted as of 2026-07-25

The current working tree contains additional MPD/source compatibility changes,
artist-enrichment package and tests, UI recommendation/artist-profile updates,
and documentation cleanup. Historical implementation documents were moved to
`implemented_archive/`, and current status/API coverage was consolidated in
`docs/Jul_24_summary.md`; these changes have not yet been assigned a commit.
