# Artist Enrichment Plan

## Goal

Add a durable artist-profile layer to the existing corpus artist records, with:

- `name`
- `profile_image`
- `following`
- `genres`
- `hometown`
- `labels`

Use source-specific IDs and provenance for every field so data can be refreshed and corrected safely.

## 1. Define the canonical profile schema

Create an `artist_profiles` dataset keyed by Anther's normalized artist ID, separate from the current embedding/clustering metadata.

```json
{
  "artist_id": "corpus:1234",
  "name": "Artist Name",
  "identities": {
    "musicbrainz_id": "…",
    "deezer_id": 123,
    "spotify_id": "…"
  },
  "profile_image": {
    "url": "https://…",
    "source": "deezer",
    "updated_at": "2026-07-22"
  },
  "following": {
    "count": 123456,
    "source": "deezer",
    "updated_at": "2026-07-22"
  },
  "genres": [
    { "name": "indie rock", "source": "musicbrainz", "weight": 8 },
    { "name": "alternative rock", "source": "spotify" }
  ],
  "hometown": {
    "name": "Los Angeles, California, US",
    "type": "begin_area",
    "source": "musicbrainz"
  },
  "labels": [
    {
      "name": "Example Records",
      "musicbrainz_id": "…",
      "source": "musicbrainz",
      "scope": "release-derived"
    }
  ],
  "match_confidence": 0.96,
  "updated_at": "2026-07-22"
}
```

Keep `artist_meta.json` focused on clustering (`artist`, `n_tracks`, source tracks, sample track). Join the profile dataset at API-read time or materialize it into a separate indexed bundle.

## 2. Establish source responsibilities

| Field | Primary source | Fallback | Notes |
|---|---|---|---|
| Name | Existing corpus | MusicBrainz / Deezer | Corpus name remains the display identity unless manually corrected. |
| Profile image | Deezer | Spotify | Cache image URL and optionally download/serve a resized copy. |
| Following | Deezer | — | Use Deezer's `nb_fan`; timestamp it because it changes. |
| Genres | MusicBrainz | Spotify, FMA, existing track genres | Preserve source and confidence; deduplicate through a normalized taxonomy. |
| Hometown | MusicBrainz | — | Use “begin area” first, then associated area/country; call it “Origin” rather than “Hometown” where uncertain. |
| Labels | MusicBrainz | FMA | Derive from release-label relationships and label the result as release-derived, not a current exclusive affiliation. |

Spotify should be optional only. Its 2026 development-mode limits and removal of artist follower/popularity fields make Deezer and MusicBrainz more dependable core sources.

## 3. Build an identity-resolution pipeline

For each corpus artist:

1. Start from artist name and representative track/album.
2. Search MusicBrainz for candidate artists.
3. Score candidates using normalized name, known track title, release title, ISRC where available, country, and source IDs.
4. Accept only high-confidence matches automatically; write ambiguous or low-confidence cases to a review queue.
5. Use the resolved MusicBrainz ID to retrieve origin, genres, and release-label relationships.
6. Query Deezer using the same artist plus representative track context to obtain the correct Deezer artist ID, image, and fan count.
7. Save source IDs, match score, retrieval time, and raw source payload/version for auditability.

Do not enrich solely by name match—common or similarly named artists would otherwise be misidentified.

## 4. Normalize the fields

- Genres: lowercase, map aliases (`hip hop` / `hip-hop`), retain source-specific originals, and cap displayed genres to 3–5.
- Hometown: store structured area fields when available: city, region, country, plus the MusicBrainz relationship type.
- Labels: aggregate distinct labels across releases, count releases per label, and display the most represented labels. Avoid presenting it as the artist's “current label.”
- Images: validate URL, cache a thumbnail, and retain attribution/source data.
- Following: store integer, source, and observed timestamp; never imply it is cross-platform total following.

## 5. Add a batch enrichment job

Create an idempotent script such as:

```text
scripts/artist_enrichment/enrich_artists.py
```

It should support:

- `--all`, `--artist-id`, and `--missing <field>` modes
- rate limiting and exponential retry
- response caching by source ID
- checkpointing/resume after interruption
- a dry-run report of matches, skips, ambiguity, and coverage
- periodic refreshes: Deezer following/images monthly; MusicBrainz quarterly or only when missing

Write output atomically to `data/artist_profiles.json` or, preferably at this scale, SQLite/Parquet plus a compact API-read index.

## 6. Extend the API

Update `GET /api/artist/<artist_id>` to include an optional `profile` object without changing existing clustering fields:

```json
{
  "name": "Artist Name",
  "track_count": 42,
  "cluster_label": "…",
  "profile": {
    "image_url": "…",
    "following": 123456,
    "following_source": "deezer",
    "genres": ["indie rock", "dream pop"],
    "origin": "Los Angeles, California, US",
    "labels": ["Example Records"],
    "enrichment_status": "complete"
  }
}
```

Return `null` or an empty list for unavailable fields; do not fabricate defaults.

## 7. Update the UI

Add a compact profile section in the artist detail pane:

- Image, artist name, and source link
- “X Deezer fans” with “as of” date
- Genre chips
- “Origin” rather than “Hometown”
- “Associated labels” rather than “Label”
- Small source/provenance disclosure and a missing-data state

## 8. Validate before rollout

Measure:

- match rate and manual-match accuracy on a stratified sample;
- source coverage per field;
- incorrect-identity rate, especially for short/common names;
- API latency with profile joins;
- freshness of fan counts;
- missing/invalid image URLs.

Start with 100–500 artists, manually audit the matches, then run the full corpus. The main success criterion should be identity accuracy, not raw enrichment coverage.
