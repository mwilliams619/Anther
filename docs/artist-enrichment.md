# Artist enrichment (`anther_ml/artist_enrichment/`)

A durable, provenance-carrying **profile layer** over the frozen corpus artists:
name, profile image, follower count, genres, origin, and associated labels.
Kept entirely separate from the immutable clustering artifacts — it never feeds
clustering, training, or the map (docs/invariants.md); it is display-only, joined
at API-read time.

Historical design source: [implemented_archive/ARTIST_ENRICHMENT_PLAN.md](../implemented_archive/ARTIST_ENRICHMENT_PLAN.md).

## Why a separate layer

`data/artist_clustering/artist_meta.json` is row-count-locked and drives Leiden
clustering — it must not gain mutable fields. Profiles instead live in their own
SQLite DB, **keyed by the normalized artist name** (`_norm(...)` from
`anther_ml.spotify_deezer`) so they survive a corpus rebuild that would shift the
positional `corpus:{index}` ids.

## Sources (v1)

| Field | Source | Notes |
|---|---|---|
| Profile image, follower count | **Deezer** | `nb_fan` stamped with an observation date (it drifts). |
| Origin, genres, labels | **MusicBrainz** | Origin = begin-area → area → country. Labels are *release-derived*, never "current label". |

Spotify is intentionally a seam only (plan §2 — 2026 dev-mode field limits); the
adapter package is structured so it can be added later without touching the
resolver.

## Pipeline

```
artist_meta.json ─▶ per artist ─▶ [Deezer search] + [MusicBrainz search]
                                        │
                                   resolve()  ── decides identity, or REVIEW
                                        │
                     [MB lookup: genres] + [MB releases: labels]
                                        │
                            build ArtistProfile ─▶ ArtistProfileStore (SQLite)
```

**Identity resolution** (`resolve.py`) is pure and unit-tested offline. It never
enriches on a name match alone (plan §3): it requires a strong fuzzy name match
*plus* corroboration (MusicBrainz's own relevance score, cross-source
agreement), and parks near-tied strong candidates in a **review queue** instead
of guessing — the short/common-name failure mode. MusicBrainz and Deezer
identities are resolved independently, so a confident Deezer match can supply
image/fans even when the MB identity is ambiguous.

## CLI

```bash
python -m anther_ml.artist_enrichment --sample 100 --dry-run   # match report, no writes
python -m anther_ml.artist_enrichment --sample 100             # write first 100
python -m anther_ml.artist_enrichment --all                    # full corpus
python -m anther_ml.artist_enrichment --artist-id corpus:42    # one artist (corpus id or key)
python -m anther_ml.artist_enrichment --missing following      # refresh one empty field
python -m anther_ml.artist_enrichment --review                 # dump the review queue
```

`scripts/artist_enrichment/enrich_artists.py` is a thin alias for the same CLI
(the path named in the plan). Everything is idempotent and resumable: the store
**is** the checkpoint (a `complete` profile is skipped on re-run), and every
source payload is cached verbatim in the `raw_cache` table so a re-run costs no
network for finished work. MusicBrainz is rate-limited to ≤1 req/s with a
required User-Agent (`--user-agent` to override).

## Storage

`data/artist_profiles.sqlite` (override with `ANTHER_ARTIST_PROFILES`):

- `profiles`     — one canonical `ArtistProfile` JSON per normalized name, plus
                   indexed `status` / `match_confidence` columns.
- `raw_cache`    — verbatim source payloads keyed by (source, source_id, kind).
- `review_queue` — ambiguous matches + the candidate list that caused it.

## API / UI

`GET /api/artist/<id>` gains an optional **`profile`** object
(`ArtistProfile.to_api_profile()`): `image_url`, `following` (+ source/as-of),
`genres`, `origin`, `labels`, `match_confidence`, `enrichment_status`. It is
`null` when the artist isn't enriched — never fabricated defaults. `atlas.py`
loads the store read-only at warm-up into `_artist_profiles` (empty and harmless
if the DB is absent) and joins by normalized name in `artist_detail()`.

The UI renders a compact profile block in the artist detail pane (`artistProfileHtml`
in `ui/static/app.js`): image, "X Deezer fans (as of …)", genre chips, **Origin**
(not "Hometown"), **Associated labels** (not "Label"), and a collapsible
sources/freshness disclosure. A not-yet-enriched artist shows a quiet hint.

## Validation (plan §8)

The main success criterion is **identity accuracy, not raw coverage**. Start with
a 100–500 artist sample, audit matches (`--dry-run` for a match report;
`--review` for the ambiguity queue), then run `--all`.
