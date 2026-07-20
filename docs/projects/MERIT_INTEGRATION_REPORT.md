# MERIT disentangled factor scores — integration report & keep/revert call

**Date:** 2026-07-17
**Scope:** Add MERIT's three pretrained factor heads (melody / rhythm / timbre)
as an *additive* layer over the frozen MERT-v1-330M backbone Anther already
uses, so the UI can explain **why** two songs connect. Harmony axis scoped out
for this round (per direction). Evaluated on the personal corpus via `eval.py`
plus a label-free perturbation-robustness probe.

**Recommendation in one line:** **KEEP** — wire the four scores (aggregate +
melody + rhythm + timbre) into the song-detail panel. The aggregate ranks your
real alternates better than the current MERT baseline, and the factors behave
exactly as their names promise, which is the explainability the UI needs. Keep
MERT-1024 as the map/clustering/tag backbone; MERIT is additive, not a
replacement.

---

## What was built (all additive; 1024-d MERT path untouched)

| Change | File | Notes |
|---|---|---|
| `merit_concat` aggregation mode → 5120-d backbone (layers {3,4,5,6,23}) | `anther_ml/embedding.py` | `mean`/`last` paths byte-identical; regression-tested |
| `ProjectionHead`, `load_heads`, `project`, `aggregate_similarity`, `merit_config` | `anther_ml/merit.py` (new) | dims read from each checkpoint's own metadata |
| 8 unit tests | `tests/test_merit.py` (new) | 16 pass with existing embedding tests |
| Dual-space encoder (MERT-1024 + MERIT factors in one MERT pass) | `scratch_eval/encode_*.py` | 484-track checkpoint |
| 5 `SongIndex` instances | `models/merit_indices/` | mel, rhy, tim (128-d), aggregate (384-d), mert_baseline (1024-d) |

The heads are the off-the-shelf `amaai-lab/merit` weights (3 × 10.75 MB, no
training). MERIT deliberately has **no native aggregate** — its whole thesis is
that one scalar mixes perceptual axes — so Anther synthesizes an aggregate as a
weighted sum of the three factor cosines: `S = α·S_mel + β·S_rhy + γ·S_tim`
(weights become UI presets).

---

## Eval corpus (an honest caveat up front)

The documented 107-track / 8-group personal corpus no longer exists on disk, and
there is no `related_pairs.json`. The on-disk personal set is **84 encodable
tracks** with only **2 auto-detected related groups / 5 query tracks** (`mic
check` ×3, `hang` ×2). To make retrieval a real discrimination test rather than
5-against-83, I added **400 FMA tracks as pure distractors** (never query
targets, never valid retrievals) → a 484-track pool.

**n = 5 self-retrieval queries is coarse.** Treat the retrieval table as
suggestive; the **perturbation probe below is the stronger, label-free signal**
and it aligns with your "does it still recognize the song" essence-capture idea.

---

## Result 1 — self-retrieval (5 queries vs 483 others)

Each query must surface its true alternate out of 483 other tracks.

| space | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| **MERT baseline (1024)** | 0.40 | 0.60 | 0.80 | 0.473 |
| melody (128) | 0.40 | 0.60 | 0.80 | 0.491 |
| rhythm (128) | 0.40 | 0.40 | 0.40 | 0.424 |
| timbre (128) | 0.40 | 0.60 | 0.60 | 0.515 |
| **aggregate (equal wts)** | 0.40 | **1.00** | **1.00** | **0.667** |
| aggregate (melody-heavy) | **0.60** | 1.00 | 1.00 | **0.717** |

**Read:** individual factors ≈ MERT. The **aggregate is the win** — it pulls
every relative into the top-5 (worst rank went 46 → 3) and lifts MRR from 0.473
to 0.667 (0.717 melody-heavy). Combining three disentangled views is more robust
than any single space, including the current MERT embedding. This is the classic
multi-view result and it holds even at n=5.

## Result 2 — perturbation robustness (the essence-capture probe)

12 seed tracks, perturbed and re-embedded; does the perturbed copy still find
its *own clean original* as the #1 neighbor out of 484? This is exactly the
"add noise / nudge pitch & tempo, is it still recognized" test you described.

| perturbation | MERT | melody | rhythm | timbre | aggregate |
|---|---|---|---|---|---|
| pitch ±1–2 st (top-1) | **1.00** | 0.06 | 0.71 | 0.38 | 0.52 |
| time-stretch ±5–10% (top-1) | 1.00 | 1.00 | 0.60 | 0.77 | **1.00** |
| additive noise 20/10/5 dB (top-1) | 0.17 | **0.33** | 0.03 | 0.08 | 0.14 |
| **overall mean rank** ↓ | 37.9 | 49.9 | 36.5 | 34.3 | **11.3** |

**This is the headline finding.** The factors respond *exactly as their names
predict*:

- **Melody head encodes pitch** → a pitch shift destroys the match (top-1 drops
  to 0.06). **Rhythm head is pitch-invariant** → it barely notices (0.71). That
  double dissociation is direct evidence the heads capture distinct, nameable
  musical properties — the interpretability the UI is built to surface.
- **The aggregate has by far the best overall mean rank (11.3 vs MERT's 37.9)** —
  ~3× better at keeping a perturbed track near the top — and is perfectly robust
  to time-stretch.
- **Noise is the universal weak spot.** MERT keeps a high cosine (0.95) but its
  *ranking* collapses (mean rank 136) — noise pulls unrelated tracks even
  closer. No space is noise-robust; flag this as future work, not a blocker.

---

## Storage & memory (100k target)

| representation | size @ 99,618 tracks | role |
|---|---|---|
| MERT 1024-d (current) | 408 MB | 2D map, Leiden clustering, tag probe — **keep** |
| MERIT 3×128 = 384-d (added) | +153 MB (+38%) | the 3 explainable factor scores |
| **additive total** | **561 MB** | |
| 3 head files | 32 MB one-time | corpus-independent |
| MERT backbone | 1.26 GB frozen | unchanged; runs on every upload regardless |

**Answer to your storage question:** MERIT vectors *cannot* be derived from the
stored 1024-d MERT vectors — they need the intermediate hidden states of layers
{3,4,5,6,23}, which are never saved, so a MERIT-only design means re-running
audio through MERT (and the 100k preview clips aren't on disk). Going
MERIT-only would cut the index to 153 MB (−62%) **but breaks the 2D map, Leiden
clustering, and the tag probe — all of which live in 1024-d space, and the map
is what the UI draws.** So MERIT is best as an *additive* layer: +153 MB buys
the four explainable scores while the map/clustering keep working. The runtime
footprint does not shrink either way — the 1.26 GB frozen backbone runs on every
new upload regardless.

---

## Keep/revert call (Workstream-E style)

| factor | keep? | why |
|---|---|---|
| **aggregate** | **KEEP** | best self-retrieval MRR (0.667 vs 0.473) and best perturbation mean-rank (11.3 vs 37.9). This is the primary similarity score to show. |
| **melody** | **KEEP** | pitch-selective by design — the clearest "why" signal; ideal for a cover/melodic-match preset. |
| **rhythm** | **KEEP** | pitch-invariant, groove-focused — complements melody; good for a "similar drum feel / mix-compatible" preset. |
| **timbre** | **KEEP** | best single-factor MRR (0.515); production/texture axis. |
| MERT-1024 | **KEEP as backbone** | do not replace — the map, clustering, and tags depend on it. |

**Net:** keep all four MERIT scores as an additive layer on top of MERT.

---

## Recommended next steps (not done this round)

1. **Wire the 4 scores into the song-detail panel.** For any pair the UI draws,
   show aggregate + a small melody/rhythm/timbre breakdown so the user sees
   *why* (e.g. "connected mostly on rhythm & timbre, not melody"). Expose
   α/β/γ preset buttons ("find covers" = melody-heavy; "mix-compatible" =
   rhythm+timbre-heavy).
2. **Re-embed the 100k corpus** to populate factor vectors (the encoder already
   extracts both spaces in one MERT pass, so this is one pass, not two;
   ~0.13 s/track observed → ~3.5 h for 100k on the RTX 4080, +153 MB storage).
3. **Strengthen the eval set.** Hand-label a `related_pairs.json` (even 30–50
   pairs) so the retrieval numbers stop being n=5. The perturbation harness is
   reusable as-is for regression testing.
4. **Noise robustness** is the one real weakness — worth a small denoising or
   augmentation-consistency pass later if user uploads are noisy.

## Reproducing

```bash
# encode (both spaces, one MERT pass) — heads + MERT already cached
python scratch_eval/encode_merit_corpus.py      # personal
python scratch_eval/encode_distractors.py       # + 400 FMA distractors
# eval + figures
python scratch_eval/run_merit_eval.py           # self-retrieval table + figure
python scratch_eval/perturbation_robustness.py  # robustness records
pytest tests/test_merit.py tests/test_embedding.py   # 16 pass
```
