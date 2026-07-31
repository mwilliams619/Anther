# Build Plan — Per-Song Micro-Genre Tagging on the 100k MPD Corpus

**Executor:** Claude Code, in `/home/matt/Dev/Anther`.
**Goal:** every track in `models/corpus_corpus_mpd_100k/` gets a set of
micro-genre tags drawn from the everynoise top-2000 vocabulary
(`data/everynoise_genres_top2000.txt`), reflecting **how that track sounds**,
not just its artist. Tags are multi-label (a song can hold several), each with a
score, primary/secondary distinction.

**Two-stage supervision** (decided in conversation):
- **Stage A — playlist weak labels (primary).** Match each track's own playlist
  names against the vocabulary → noisy but in-vocab, in-distribution per-track
  seed labels. ~29% of tracks get a tag from exact match alone; normalization +
  fuzzy pushes higher.
- **Stage B — MERT probe + propagation (the actual tagger).** Train a frozen
  multi-label probe on the confident seed subset, predict tags for all 100k,
  and use k-NN propagation in MERT space to fill tracks with no playlist match
  and to denoise. This is what makes tags track-level, not artist-level.

**Validation:** FMA (human-labeled audio, multi-label genres already parsed by
`data.py`) embedded with the corpus's frozen recipe, crosswalked to the
everynoise vocab → a real held-out test set. **User will download FMA audio.**

---

## 0. Constraints (inherit from the corpus stack — non-negotiable)

1. **Display-only.** Tags never feed the MERT embedding, `SongIndex`, the Leiden
   partition, or `anther_ml.eval` (`docs/invariants.md`). The probe judges
   *itself* on its own held-out labels; it never becomes an arbiter for
   clustering hyperparameters. Two scoreboards, one wall.
2. **Frozen-recipe discipline.** Any audio embedded for this task — FMA
   validation, Jamendo pretraining — MUST use the corpus's exact
   `embedding_config` (24 kHz, 10 s × 3 windows, all-layer mean-pool, EBU R128
   −14 LUFS). Use `place.embed_query` / `embedding.get_embedding` with the
   config read from the bundle manifest; never hand-roll a second recipe. This
   is the `assert_compatible` invariant.
3. **No corpus rebuild.** The probe trains on the **already-stored**
   `embeddings.npy` (verified `(99618, 1024)` float32, raw pre-transform). No
   GPU is needed for the corpus side. GPU is needed only to embed the external
   validation audio.
4. **Bundle schema additive, version stays 1.** Tag artifacts are new files in
   the bundle dir; `place()` output gains a `tags` field. Do not bump
   `CORPUS_FORMAT_VERSION`. Loaders tolerate missing tag files (older bundles).

---

## 1. Data availability (verified — don't re-derive)

| Signal | Per-track coverage | Role |
|---|---|---|
| `embeddings.npy` (1024-d raw MERT) | 100% (`99618×1024`) | probe input + propagation space |
| `playlists[*].name` | 100%, mean 9.39/track, 218k distinct | Stage-A weak labels |
| everynoise vocab | 2000 genres, popularity-ordered | tag label space |
| playlist-name ∩ vocab (exact) | **827 genres hit; 29.1% of tracks tagged** | Stage-A seed baseline |
| `artist` | 99.99%, but 1.4 tracks/artist | optional weak-label booster only |
| `genre` field | always null | unusable |
| FMA `('track','genres')`/`('genres_all')` | parsed by `load_fma_tracks` | validation labels |

---

## 2. Module layout (new)

```
anther_ml/corpus/tagging/
    __init__.py
    vocab.py          # load + normalize the everynoise vocabulary; aliases
    weak_labels.py    # Stage A: playlist-name -> per-track seed tag matrix
    probe.py          # Stage B: TagProbe (fit/predict/save/load), propagation
    crosswalk.py      # FMA/Jamendo taxonomy -> everynoise vocab mapping
    evaluate.py       # held-out tag metrics (walled off from anther_ml.eval)
    build_tags.py     # orchestrator: corpus dir -> tag artifacts in bundle
```

Keep it under `corpus/` (it is corpus post-processing), parallel to the
`labels.py` from the playlist-labels plan. Pure `numpy`/`scikit-learn`/stdlib on
the corpus side; `torch`+`transformers` only in the FMA-embedding path (reuse
`anther_ml.embedding`, add nothing new).

---

## 3. Stage A — playlist weak labels (`weak_labels.py`)

### 3a. Vocabulary + normalization (`vocab.py`)
```
def load_vocab(path="data/everynoise_genres_top2000.txt") -> list[str]
def norm(s: str) -> str        # lowercase, strip, collapse ws, strip punctuation/emoji
```
Build a normalized `{norm(genre): canonical_genre}` map. Add a small **alias
table** for high-value near-misses observed in the data (e.g. `"dnb"` →
`"drum and bass"`, `"hip-hop"`/`"hiphop"` → `"hip hop"`, `"r and b"` → `"r&b"`).
Keep it a small curated dict, not a scrape.

### 3b. Matching a playlist name to genres
Three tiers, most precise first, each track accumulates matches from all its
playlists:
1. **Exact** (normalized playlist name == normalized genre). Highest precision.
2. **Token-substring** (genre appears as a whole-word span inside the playlist
   name: `"chill hop beats"` → `chillhop`? no — must be word-boundary;
   `"90s death metal"` → `death metal`). Guard against short-genre false
   positives (don't let `"pop"` match `"k-pop lovers"` unless `"k-pop"` is
   itself in vocab and matched first — prefer the **longest** vocab match).
3. **Fuzzy** (optional, gated): rapidfuzz token-set ratio ≥ 92 against vocab,
   only for playlist names not matched by 1–2, and only accept if the match is
   unambiguous (single vocab entry above threshold). Fuzzy is off by default;
   `--fuzzy` enables it. Report how many tags each tier contributes.

### 3c. Seed label matrix
```
def build_seed_labels(metadata, vocab_map, *, min_playlist_support=1,
                      fuzzy=False) -> tuple[csr_matrix, list[str], np.ndarray]:
    """Return (Y_seed [N x G] multi-hot sparse, genre_names[G], confidence[N]).
    A track's tag = any genre matched by any of its playlists.
    confidence[i] = f(#playlists matched, agreement) — used to weight training
    and to define the 'confident subset' for probe fitting."""
```
Also emit **per-genre support** (how many tracks carry each tag). Genres with
support below a floor (e.g. < 50 seed tracks) cannot be learned reliably — mark
them `learnable=False`; the probe only fits heads for learnable genres, the rest
stay playlist-match-only (or dropped). Save a `tag_support.json` report.

### 3d. Optional artist booster (Stage A+, off by default)
`--artist-genres path.json`: if the user later pulls Spotify artist genres,
merge them as additional seed labels (artist-grain, so lower confidence weight).
Not required for v1; the plan does not depend on it.

---

## 4. Stage B — the probe (`probe.py`)

### 4a. Model
`TagProbe`: multi-label classifier over frozen MERT vectors.
- **Baseline:** one-vs-rest logistic regression (scikit-learn
  `LogisticRegression` per learnable genre, or a single linear layer with
  `BCEWithLogitsLoss`). MERT linear-probes are strong on tagging benchmarks, so
  start linear — it's fast, CPU-fine on 100k×1024, and interpretable.
- **Upgrade path:** 1-hidden-layer MLP (torch) if linear underperforms in eval.
- **Input:** raw `embeddings.npy`, L2-normalized (match `SongIndex`'s own
  transform for consistency — but the probe fits its *own* standardization on
  the corpus and freezes it, same discipline as every other transform).

### 4b. Class imbalance (this corpus is skewed)
Seed distribution is long-tailed (smooth jazz / death metal dominate). Use
`class_weight="balanced"` (LR) or per-class `pos_weight` (BCE). **Report
per-genre F1 and support, never a single aggregate** — a 0.9 micro-F1 hiding
20 dead genres is a failure, not a success.

### 4c. Training regime
- Fit on the **confident seed subset** (tracks with confidence above a
  threshold), **artist-stratified split** so no artist appears in both train and
  the internal val fold (mirrors Jamendo/FMA discipline — prevents the probe
  memorizing artists).
- `fit(X, Y_seed_confident) -> TagProbe`; `predict_proba(X) -> [N x G]`.
- Persist `tag_probe.pkl` + `tag_vocab.json` + fitted standardizer into the
  bundle dir. Round-trip through save/load.

### 4d. Inference + propagation
```
def predict_tags(probe, embeddings, *, threshold=..., top_k=3,
                 knn_smooth_k=0) -> list[dict]:
    """Per track: sigmoid/proba over genres, keep >= threshold, cap top_k,
    flag argmax as primary. If knn_smooth_k>0, blend each track's scores with
    the mean of its k nearest neighbors in MERT space (label propagation) —
    fills tracks the probe is unsure about and denoises stray seeds."""
```
Per-genre thresholds (tuned on val to hit a target precision) beat one global
threshold given the imbalance — store them in `tag_vocab.json`.

### 4e. Output artifact
Write `track_tags.json` (or `.parquet` for 100k) into the bundle dir:
```
[{"idx", "id", "tags": [{"genre","score","primary"}], "source": "probe|seed"}]
```
Keep it aligned to metadata row order (idx). This is the deliverable a musician's
placement reads from.

---

## 5. Validation — FMA held-out set (`crosswalk.py`, `evaluate.py`)

**User downloads FMA audio** (`fma_source` already knows the path convention;
`data.py::load_fma_tracks` parses multi-label `('track','genres')` /
`('genres_all')`).

### 5a. Crosswalk (`crosswalk.py`)
FMA has 161 genres; everynoise top-2000 is a different taxonomy. Build a
mapping `fma_genre -> [everynoise_genre, ...]`:
- Exact/normalized name match first (many will hit: "Post-Rock" → "post-rock").
- Hand-curated dict for the rest (FMA is only 161 genres — a finite, one-time
  table). Store as `data/fma_to_everynoise.json`; unmapped FMA genres are
  dropped from eval, not guessed.
- Same pattern reusable for Jamendo's 87 tags if pretraining is added later.

### 5b. Embed FMA with the frozen recipe (GPU step, user's machine)
```
python -m anther_ml.corpus.tagging embed-eval \
    --audio-dir data/audio/fma_medium --meta-dir data/fma_metadata \
    --corpus models/corpus_corpus_mpd_100k --out data/fma_eval_embeddings.npz
```
Reads the bundle's `embedding_config`, calls `embed_tracks_batched` /
`get_embedding` with **identical** settings (assert via `assert_compatible`),
saves `(M x 1024)` vectors + crosswalked multi-hot labels. This is the only
GPU-bound step; checkpoint/resume like `build_corpus` does.

### 5c. Metrics (`evaluate.py`) — walled off from `anther_ml.eval`
- Run the **frozen probe** on FMA embeddings, score against crosswalked labels.
- Report **per-genre precision/recall/F1 + support**, macro & micro averages,
  and a precision@k. Write `tag_eval_report.json` + a bar chart
  (`tag_eval_f1_by_genre.png`) — publication-style, per-genre F1 sorted, support
  annotated.
- **Explicitly NOT** added to `build_scorecard` in `anther_ml/eval.py`. Separate
  file, separate CLI flag. State this in the module docstring.
- Sanity cross-check: also report agreement between probe tags and the Stage-A
  playlist seeds on the corpus itself (not ground truth, but a coherence check).

---

## 6. Orchestrator + CLI (`build_tags.py`, `__init__.py`/`__main__`)

Add a `tagging` CLI (peer to `build`/`place`/`label`):
```
python -m anther_ml.corpus.tagging seed   --corpus models/corpus_corpus_mpd_100k [--fuzzy]
python -m anther_ml.corpus.tagging fit     --corpus models/corpus_corpus_mpd_100k
python -m anther_ml.corpus.tagging predict --corpus models/corpus_corpus_mpd_100k [--knn-smooth 10]
python -m anther_ml.corpus.tagging embed-eval --audio-dir ... --corpus ...
python -m anther_ml.corpus.tagging evaluate   --corpus ... --eval-embeddings data/fma_eval_embeddings.npz
python -m anther_ml.corpus.tagging all     --corpus models/corpus_corpus_mpd_100k   # seed->fit->predict
```
`all` runs the full corpus-side pipeline (no GPU): seed → fit → predict →
`track_tags.parquet`. Eval is separate because it needs the downloaded audio.

---

## 7. Wire into placement (`place.py`)

`place()` gains per-song tags for an uploaded query. Two options:
- **Live:** load `tag_probe.pkl`, run it on the query's MERT vector, return
  `tags` in the placement dict alongside `neighbors`/`cluster`.
- **Neighbor-inherited (fallback if no probe loaded):** aggregate tags of the
  top-k neighbors from `track_tags`.

```python
return {
    "neighbors": neighbors,
    "cluster": {...},
    "tags": tag_list,          # NEW: [{genre, score, primary}]
    "coords_2d": coords_2d,
}
```
Guard behind `if probe available` so bundles without tags still place. This is
the one product payoff line: an uploaded song now comes back with micro-genre
tags.

---

## 8. Tests — `tests/test_corpus_tagging.py`

1. `norm`/alias folding; longest-match wins (`"k-pop"` not `"pop"` for
   `"k-pop hits"`).
2. `build_seed_labels` on a synthetic metadata fixture → correct multi-hot,
   correct per-genre support, confidence monotonic in #matched playlists.
3. `TagProbe` fit/predict/save/load round-trip on a tiny random-vector fixture;
   learnable-genre floor respected (starved genres get no head).
4. Artist-stratified split: assert no artist leaks across train/val.
5. Crosswalk: exact FMA→everynoise hits map; unmapped dropped, not guessed.
6. `evaluate` returns per-genre F1+support and does **not** touch
   `anther_ml.eval.build_scorecard` (import-guard test).
7. `place()` returns `tags` when a probe is present, omits/empties gracefully
   when absent; recipe-mismatch still refused by `assert_compatible`.

---

## 9. Sequenced build path

1. **T1 — Vocab + Stage-A seeds.** `vocab.py`, `weak_labels.py`; run `seed` on
   100k. *Acceptance:* `tag_support.json` shows ≥800 genres with ≥1 seed track,
   ≥29% track coverage reproduced; per-tier contribution reported.
2. **T2 — Probe fit + predict (corpus-only, no GPU).** `probe.py`; run `all`.
   *Acceptance:* `track_tags.parquet` for all 100k; per-genre support report;
   probe round-trips save/load; artist-stratified internal val F1 reported.
3. **T3 — k-NN propagation + threshold tuning.** *Acceptance:* coverage of
   tracks with ≥1 tag rises vs seed-only; denoising demonstrated on a spot-check
   (stray-seed track corrected by neighbors).
4. **T4 — FMA crosswalk + embed-eval (GPU, user's audio).** `crosswalk.py`,
   `embed-eval`. *Acceptance:* FMA embedded with matching recipe
   (`assert_compatible` passes); crosswalk table covers the FMA genres present.
5. **T5 — Held-out evaluation.** `evaluate.py`; `tag_eval_report.json` +
   `tag_eval_f1_by_genre.png`. *Acceptance:* real per-genre F1 on human labels;
   wall-off from `anther_ml.eval` verified by test.
6. **T6 — Placement wiring.** `place()` returns `tags`. *Acceptance:* an
   uploaded song returns ≤3 tags with a primary flag.
7. **T7 (optional) — Jamendo pretraining.** Reuse crosswalk pattern; pretrain
   probe on Jamendo, fine-tune on seeds. Only if T5 F1 is weak on tail genres.

---

## 10. Honest caveats (carry into the report, do not bury)

- **Seed labels are noisy and artist-grained at origin** (everynoise genres are
  Spotify *artist* tags). The probe's track-level tags are its acoustic best
  guess trained on noisy labels — a real improvement over artist tagging, but
  measured, not assumed. T5 is what makes the quality claim defensible.
- **FMA is CC/library music**, distribution-shifted from MPD/Spotify chart
  music, and its taxonomy is coarser than everynoise. Eval F1 is a floor/proxy,
  not a perfect readout; report it as such.
- **Tail genres will be weak.** Be explicit about which genres have enough
  support to be trustworthy and which are effectively playlist-match-only.

## 11. Non-goals

- No CLAP / text-audio model.
- No corpus re-embedding; no `CORPUS_FORMAT_VERSION` bump.
- No change to embeddings / `SongIndex` / Leiden / `anther_ml.eval` math.
- No UI rendering (blocked on the `ui/jobs.py` legacy→`place()` migration).
