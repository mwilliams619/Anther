# Micro-Genre Tagging — Next Steps

Continuation of [MICROGENRE_TAGGING_BUILD_PLAN.md](MICROGENRE_TAGGING_BUILD_PLAN.md).
Status as of 2026-07-08.

## Where things stand

| Step | Status |
|---|---|
| T1 — vocab + Stage-A seeds | **Done.** 78.7% of 99,618 tracks carry ≥1 seed; 1,261 genres hit (plan floors: 29% / 800). `tag_seeds.npz` + `tag_support.json` in the bundle. |
| T2 — probe fit | **Done.** 482 learnable genres, 43,968 train / 10,944 val (artist-stratified). Internal val macro-F1 0.18; median 0.15; 87 genres ≥0.3, 15 ≥0.5, 65 dead (<0.05). F1 tracks seed support (≥500 seeds → median 0.28; <100 → 0.09), as predicted. |
| T3 — predict + kNN propagation | **Done** (`--knn-smooth 10`, α=0.5). 99.8% of tracks tagged (up from 78.7% seed-only ✓). `track_tags.json` (26 MB) in the bundle. |
| T4 — FMA download + embed-eval | **Blocked on FMA download** (user step, below). Code ready. |
| T5 — held-out evaluation | Blocked on T4. Code ready (`tagging evaluate`). |
| T6 — placement wiring | **Done.** `place()` returns `tags` (probe → neighbor-inherited → `[]`). |
| T7 — Jamendo pretraining | Optional; only if T5 shows weak tail genres. |

Everything code-side is implemented and tested (20 tests in
`tests/test_corpus_tagging.py`). Deviations from the original plan, all
deliberate: JSON instead of parquet (no pyarrow dep), fuzzy tier
import-guarded (rapidfuzz optional), catalog-name discount for multi-genre
playlist names, `EXACT_ONLY` guard for generic-English vocab entries
("sound", "sleep", "focus", "rain" — substring tier seeded thousands of false
positives before the guard), probe transform = its own frozen StandardScaler.
`docs/invariants.md` now states the genre carve-out explicitly.

## 1. Verify T2/T3 when the background job finishes

```bash
tail models/corpus_corpus_mpd_100k/tag_fit.log
```

Acceptance to check:
- `tag_vocab.json` fit_report: internal val macro-F1 reported, per-genre F1 +
  support present (a good aggregate hiding dead genres is a failure).
- `track_tags.json` covers all 99,618 rows; tagged fraction **rises** vs the
  78.7% seed-only coverage (that's what propagation is for).
- Spot-check ~20 tracks by hand: pick a few tracks whose seed came from one
  stray playlist and confirm neighbors corrected them (T3 acceptance), e.g.

```bash
python - <<'EOF'
import json, random
rows = json.load(open('models/corpus_corpus_mpd_100k/track_tags.json'))
meta = json.load(open('models/corpus_corpus_mpd_100k/index.json'))['metadata']
for r in random.sample([r for r in rows if r['tags']], 20):
    m = meta[r['idx']]
    print(f"{m['artist']} — {m['name']}: ",
          [(t['genre'], t['score'], t['source']) for t in r['tags']])
EOF
```

If tags look over-smoothed (everything pulled to big genres), refit is cheap:
lower `--alpha` or `--knn-smooth` in `predict` only — no re-fit needed.

## 2. Download FMA (user step, ~23 GB)

```bash
cd data
curl -LO https://os.unil.cloud.switch.ch/fma/fma_metadata.zip     # 342 MB
unzip fma_metadata.zip && rm fma_metadata.zip                     # → data/fma_metadata/
cd audio
curl -LO https://os.unil.cloud.switch.ch/fma/fma_medium.zip       # 22 GB, 25k tracks
unzip fma_medium.zip && rm fma_medium.zip                         # → data/audio/fma_medium/
```

Fallback if disk/time is tight: `fma_small.zip` (7.2 GB, 8k tracks) —
workable but coarser eval coverage.

## 3. T4 — embed FMA with the frozen recipe (GPU)

```bash
python -m anther_ml.corpus.tagging embed-eval \
    --audio-dir data/audio/fma_medium \
    --corpus models/corpus_corpus_mpd_100k
```

- Checkpoints every 200 tracks to `data/fma_eval_embeddings.npz.ckpt.npz`;
  rerun the same command to resume. Bad MP3s are skipped, not fatal.
- First run auto-writes `data/fma_to_everynoise.json` from exact matches +
  the curated table in `crosswalk.py`. **Review it before T5**: FMA genres
  mapped to `[]` are dropped from eval — check the list for salvageable
  entries and extend `CURATED` (one-time, 161 genres total).
- Acceptance: `assert_compatible` passes (it runs before any embedding);
  crosswalk covers the FMA genres present.

## 4. T5 — held-out evaluation

```bash
python -m anther_ml.corpus.tagging evaluate \
    --corpus models/corpus_corpus_mpd_100k \
    --eval-embeddings data/fma_eval_embeddings.npz
```

Writes `tag_eval_report.json` + `tag_eval_f1_by_genre.png` into the bundle.
Read it with the caveats attached (plan §10 + review):
- FMA is CC/library music, distribution-shifted from MPD — F1 is a floor.
- The crosswalk only reaches the **coarse** end of the vocabulary; tail
  microgenres stay unvalidated by T5. The report's per-genre support column
  says which numbers to trust.
- `corpus_seed_agreement` in the report is a coherence check, not truth:
  ~100% means the probe memorized the noise, ~0% means it learned nothing.
- **Never** feed these numbers into `build_scorecard` or map hyperparameters
  (docs/invariants.md — enforced by an import-guard test).

## 5. Close the microgenre validation gap (T5 can't)

Pick one after seeing T5:
- **Hand audit**: sample ~100 tracks across tail genres from
  `track_tags.json`, listen, mark right/wrong — the only direct microgenre
  precision readout available.
- **T7 Jamendo pretraining** (plan §9): reuse the crosswalk pattern for
  Jamendo's 87 tags, pretrain the probe, fine-tune on seeds. Only worth it if
  T5/tail results are weak.

## 6. Product surfacing (after quality is believed)

- `place()` already returns `tags` — exercise end-to-end with a real upload:
  `python -m anther_ml.corpus place song.mp3 --corpus models/corpus_corpus_mpd_100k --json`
- UI rendering stays a non-goal until the `ui/jobs.py` legacy→`place()`
  migration lands (plan §11).
- Commit the tagging package + docs changes on `ml-dev` once T2/T3 outputs
  are verified.
