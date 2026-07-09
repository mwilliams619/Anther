# Build Plan — Playlist-Name Cluster Labels (Scheme 2, buildable today)

**Executor:** Claude Code, working in `/home/matt/Dev/Anther`.
**Scope:** Only the playlist-aggregation labeling path. No CLAP, no MERT
retraining, no tag probe, no UI work. This produces a human-readable `label`
for every cluster in an **already-built** corpus bundle, plus the tag tallies
(`top_tags`, `tag_entropy`) that a later micro-genre scheme will reuse.

**Target bundle:** `models/corpus_corpus_mpd_100k/` — verified to have populated
playlist data (65 `top_playlists` entries across 13 clusters). Do **not** target
`models/corpus_mpd_val_25k/` (its metadata carries zero playlists — labeling
there will correctly no-op).

---

## 0. Non-negotiable constraints (read before writing code)

1. **Display-only.** Labels are computed *from* the frozen map and never feed
   back into embeddings, `SongIndex`, the Leiden partition, or `anther_ml.eval`.
   This is the repo's governing invariant (`docs/invariants.md`). Labels are the
   same *kind* of object as the existing `top_genres` tally — a read-out.
2. **No corpus rebuild.** Labeling reads an existing bundle and rewrites
   `cluster_profiles.json` in place. Re-embedding the 100k corpus takes hours;
   this step must run in seconds.
3. **Additive schema, version stays 1.** `bundle.py` has
   `CORPUS_FORMAT_VERSION = 1` and `verify()` does strict inequality
   (`if version != CORPUS_FORMAT_VERSION: raise`). Do **NOT** bump the version —
   that would invalidate every existing bundle. New profile fields are optional;
   the loader and all readers must tolerate their absence via `.get(...)`.
4. **Human override is authoritative and persistent.** An auto-generated label
   is a draft. A human override, once set, must survive re-labeling and additive
   corpus extension. Store override separately from the auto draft; never let a
   re-run clobber it.

---

## 1. What already exists (do not rebuild these)

| Thing | Location | Status |
|---|---|---|
| Per-cluster raw playlist tally | `build.py::build_cluster_profiles` → `top_playlists` (`Counter.most_common(5)`) | present, but raw/un-normalized and only top-5 |
| Playlist data on tracks | `metadata[i]["playlists"] = [{"pid","name"}, ...]` | present in `corpus_mpd_100k` |
| Bundle-wide playlist helpers | `bundle.py::playlists()`, `playlist_member_indices()` | present |
| Profile → placement passthrough | `place.py::place()` returns `cluster.profile = corpus.cluster_profile(id)` | present — new fields surface automatically |
| Cluster labels (`label`, `label_source`) | — | **MISSING — this plan adds them** |
| Normalized tag tally + entropy (`top_tags`, `tag_entropy`) | — | **MISSING — this plan adds them** |

---

## 2. New module: `anther_ml/corpus/labels.py`

Create this module. It is pure post-processing over a built bundle — no torch,
no audio, only `numpy` + stdlib (`collections`, `re`, `math`, `json`).

### 2a. Playlist-name normalization

```
def normalize_playlist_name(name: str) -> str | None:
    """Fold near-duplicate playlist names to one key.
    - lowercase, strip, collapse internal whitespace
    - strip surrounding punctuation/emoji, drop leading/trailing non-alnum
    - return None for empty / pure-emoji / single-char names (unusable signal)
    """
```

Rationale: the real 100k corpus contains `"Texas Country"`, `"texas country"`,
and `"Texas country"` as three separate raw entries. Without folding, the tally
fragments and no name reaches the top. Keep the *display* form of the most
common raw variant for the label text, but tally on the normalized key.

### 2b. Distinctiveness weighting (TF-IDF over clusters)

A name like `"favorites"`, `"my playlist"`, `"liked songs"` appears in every
cluster and carries no discriminative signal; a name like `"death metal"`
concentrated in one cluster is exactly what we want to surface. Weight each
normalized name per cluster by:

```
score(name, cluster) = tf * idf
  tf  = count(name in cluster) / cluster_size          # within-cluster frequency
  idf = log( (K + 1) / (1 + n_clusters_containing name) ) + 1
```

where `K` = number of clusters. This down-weights ubiquitous names and promotes
concentrated ones. Also maintain a small hardcoded stopword set
(`{"favorites","favourites","my playlist","liked songs","new playlist",
"untitled","music","songs","playlist"}`, matched on the normalized key) as a
cheap floor — TF-IDF handles most of it but generic names are so common they
still leak.

### 2c. Cluster coherence — entropy

```
def tag_entropy(normalized_counts: Counter) -> float:
    """Shannon entropy (natural log) of the normalized playlist-name
    distribution within a cluster. Low = coherent (few dominant names),
    high = mixed. Report raw entropy AND normalized entropy
    (entropy / log(n_distinct)) so clusters of different vocab size compare."""
```

### 2d. Label assembly

```
def generate_cluster_labels(
    metadata: list[dict],
    labels: np.ndarray,
    *,
    top_k_tags: int = 8,
    label_n: int = 3,
    min_support: int = 3,
) -> list[dict]:
    """Return, per cluster_id, a dict:
       {
         "cluster_id": int,
         "top_tags": [[display_name, count], ...],   # normalized, top_k_tags
         "tag_entropy": float,
         "tag_entropy_norm": float,
         "label": str,          # e.g. "Death Metal / Power Metal / Metalcore"
         "label_source": "playlist_tfidf",
       }
    Label = the top `label_n` names by TF-IDF score with count >= min_support,
    joined by ' / ', title-cased for display. If a cluster has no usable
    playlist names, label = "" and label_source = "none" (caller decides how
    to display an unnamed cluster — do NOT fabricate a name)."""
```

`label` deliberately reads as a slash-joined tag list, not a hand-written
sentence — it is honest about being auto-derived. A human override (§3) is where
prose names like "Dreamy lo-fi (nostalgic)" belong.

### 2e. Merge into profiles (additive, override-preserving)

```
def apply_labels_to_profiles(
    profiles: list[dict],
    generated: list[dict],
) -> list[dict]:
    """Merge generated fields into existing profiles by cluster_id.
    - Adds/overwrites: top_tags, tag_entropy, tag_entropy_norm,
      label (the auto draft), label_source.
    - PRESERVES any existing label_override and label_override_by/at.
    - Computes label_final = label_override if present else label.
    Never mutates cluster_id/size/exemplars/top_playlists/top_genres."""
```

---

## 3. Human override support

Overrides live on the profile dict and persist in `cluster_profiles.json`:

- `label_override: str` — the human-authored name (free prose allowed).
- `label_override_at: str` — ISO timestamp (reuse `bundle.utc_now_iso()`).
- `label_final: str` — resolved value = `label_override or label`. Readers
  (place, UI, viz) display `label_final`.

Re-running labeling recomputes `label`/`top_tags`/`tag_entropy` but must copy
`label_override*` through untouched, then recompute `label_final`. This is what
makes the auto step safe to re-run after an additive corpus extension.

---

## 4. Bundle I/O — round-trip without rebuild

Add a thin function (in `labels.py` or `bundle.py`, your call — keep bundle.py
lean, prefer `labels.py`):

```
def label_bundle(dir_path, *, dry_run=False, **kw) -> list[dict]:
    """Load bundle, regenerate labels from its own metadata+labels,
    merge (preserving overrides), and rewrite cluster_profiles.json in place.
    Returns the updated profiles. dry_run prints a table and writes nothing."""
```

Implementation notes:
- Load via `ReferenceCorpus.load(dir_path)` — you get `corpus.metadata` and
  `corpus.labels` directly.
- Rewrite **only** `cluster_profiles.json` (json.dump, indent=2). Do not touch
  `embeddings.npy`, `index`, `leiden.pkl`, `manifest.json`.
- After writing, re-`load` and assert the profiles carry `label_final` for every
  non-empty-playlist cluster — cheap integrity check.

To set an override programmatically:

```
def set_cluster_override(dir_path, cluster_id: int, name: str) -> None
```

which loads, sets `label_override`/`label_override_at`, recomputes `label_final`,
and rewrites `cluster_profiles.json`.

---

## 5. Wire into the build path (so future builds get labels free)

In `build.py`, after the existing:

```python
centroids, profiles = build_cluster_profiles(
    leiden["clustering_space"], leiden["labels"], metadata
)
```

add:

```python
from .labels import generate_cluster_labels, apply_labels_to_profiles
generated = generate_cluster_labels(metadata, leiden["labels"])
profiles = apply_labels_to_profiles(profiles, generated)
```

This keeps `build_cluster_profiles` unchanged (its signature is used by tests)
and layers labeling on top. New builds ship with labels; existing bundles get
them via the `label` CLI (§6).

---

## 6. CLI

Extend `anther_ml/corpus/__main__.py` with a `label` subcommand:

```
python -m anther_ml.corpus label models/corpus_corpus_mpd_100k
python -m anther_ml.corpus label models/corpus_corpus_mpd_100k --dry-run
python -m anther_ml.corpus label models/corpus_corpus_mpd_100k --set 3 "Dreamy lo-fi (nostalgic)"
```

- default: (re)generate labels, preserve overrides, write.
- `--dry-run`: print a `cluster_id | size | tag_entropy_norm | label` table,
  write nothing.
- `--set <cid> <name>`: set an override for one cluster.
- Follow the existing argparse structure in `__main__.py` (subparsers for
  `build`/`place`); add `label` as a peer.

---

## 7. Reader-side changes (small, additive)

- **`place.py`** — no code change strictly required; `label_final` rides inside
  `cluster.profile` automatically. Optionally lift it to `cluster.label` for
  convenience:
  ```python
  "cluster": {"id": ..., "confidence": ..., "label": profile.get("label_final",""),
              "profile": profile}
  ```
- **`bundle.py::load`** — already tolerant (profiles are opaque JSON). No change.
  Confirm nothing downstream indexes profile dicts by a fixed key set.

---

## 8. Tests — `tests/test_corpus_labels.py` (new)

1. `normalize_playlist_name` folds `"Texas Country"/"texas country"/"Texas
   country"` to one key; returns `None` for `""`, `"🔥"`, `"x"`.
2. TF-IDF: a name present in all clusters scores below a name concentrated in
   one; stopword names are excluded from labels.
3. `tag_entropy` is lower for a single-name cluster than a many-name cluster.
4. `generate_cluster_labels` on a synthetic 2-cluster metadata fixture yields
   the expected distinctive label per cluster and `label_source="playlist_tfidf"`;
   an all-empty-playlist cluster yields `label="" , label_source="none"`.
5. **Override persistence:** set an override, re-run `apply_labels_to_profiles`,
   assert `label_override` survives and `label_final == override` while `label`
   (draft) still updated.
6. **Round-trip:** `label_bundle` on a tiny fixture bundle rewrites only
   `cluster_profiles.json`; `ReferenceCorpus.load` still `verify()`s (version
   unchanged); every non-empty cluster has `label_final`.

Reuse the existing corpus test fixtures in `tests/test_corpus_build.py` /
`test_corpus_bundle.py` for the round-trip test rather than building a new MERT
corpus.

---

## 9. Acceptance criteria

- [ ] `python -m anther_ml.corpus label models/corpus_corpus_mpd_100k --dry-run`
      prints 13 clusters, each with a non-empty `label` for clusters that have
      playlist data, and shows sane distinctive names (metal cluster → metal
      terms, country cluster → country terms).
- [ ] Running without `--dry-run` rewrites `cluster_profiles.json`;
      `ReferenceCorpus.load("models/corpus_corpus_mpd_100k")` succeeds (version
      still 1, `verify()` passes).
- [ ] `--set 3 "..."` persists; a subsequent plain `label` run keeps the
      override and updates only the draft/tallies.
- [ ] `place(...)["cluster"]` exposes the label (via `profile["label_final"]`
      and/or lifted `cluster.label`).
- [ ] `pytest tests/test_corpus_labels.py` green; existing corpus tests still
      green.
- [ ] Labeling `models/corpus_mpd_val_25k` (no playlists) runs cleanly and
      leaves every `label=""` / `label_source="none"` — no crash, no fabrication.

---

## 10. Explicit non-goals (deferred)

- No CLAP / text-audio model.
- No per-song micro-genre tags or tag probe (separate plan — next).
- No UI rendering (blocked on the `ui/jobs.py` legacy→`place()` migration).
- No `corpus_format_version` bump.
- No change to embeddings, `SongIndex`, Leiden, or `anther_ml.eval`.
