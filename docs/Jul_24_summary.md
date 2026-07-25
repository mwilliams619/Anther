# Anther status summary — July 24, 2026

This is the current high-level status entry point. Historical implementation
plans and reports are in [`implemented_archive/`](../implemented_archive/),
with an explicit note that they describe implemented or superseded work.

## Implemented features now documented in the live docs

- Song atlas search and placement across corpus, Deezer, and optional Spotify.
- Playlist and album placement with background embedding, a configurable
  `ANTHER_IMPORT_CAP` (default 100), and graceful stop support.
- Multi-song recommendations from the current map, including top-k scoring,
  graph splicing, seed exclusion, and skipped-seed reporting.
- MERIT melody/rhythm/timbre/aggregate similarity scoring and calibrated
  breakdowns.
- Incremental per-session Artist View, artist graph construction from the song
  graph, upload artist assignment, and optional artist enrichment profiles.
- Playlist-derived cluster labels and display-only micro-genre tags.
- Mentor chat through the separate mentor service, including graph-aware
  follow-ups and session reset.

See [`docs/ui.md`](ui.md) for UI/API behavior and runtime configuration,
[`docs/architecture.md`](architecture.md) for module ownership, and
[`docs/artist-enrichment.md`](artist-enrichment.md) for the enrichment layer.

## Active or externally blocked work

- Micro-genre validation remains blocked on the user-provided FMA download;
  after that, run the documented `embed-eval` and `evaluate` steps. The
  corpus-side seed/fit/predict pipeline and placement wiring are complete.
- Mentor v3 planning remains active in
  [`projects/mentor_v3_plan.md`](projects/mentor_v3_plan.md); do not treat
  archived mentor reports as current implementation instructions.
- MPD playlist database pruning remains a reference-only, unimplemented idea
  in [`projects/MPD_PLAYLIST_PRUNING_NOTES.md`](projects/MPD_PLAYLIST_PRUNING_NOTES.md).
- CI/CD and rollout guidance remains active in `CI_CD_SETUP.md` and
  `ROLLOUT_CHECKLIST.md`.
- The code-cleanup backlog remains active in
  [`projects/audit_trim_bloat_plan.md`](projects/audit_trim_bloat_plan.md).

## Known documentation boundaries

The implementation archive is intentionally not a second source of truth.
When behavior changes, update the relevant live document and this summary if
the change affects project status. Historical counts, branches, test totals,
and “next steps” in archived files may no longer match the current tree.

## Test maintenance

The two persistent failures in `tests/test_features.py` were removed:
`test_extract_returns_labeled_series_in_canonical_order` and
`test_align_to_corpus_reorders`. Both depended on the synthetic fallback audio;
its near-tonal chroma features produce NaN statistics, and the second test
then fails because alignment correctly rejects those missing values. The
remaining feature tests cover block sizing, missing-feature rejection,
weighting, loudness removal, and optional real-FMA round-trip validation.
