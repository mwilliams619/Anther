# Friends of Gunk — 9 bands (premade map for Anther)

A premade map (`ui/custom_playlists/five_artists_map.json`, pid `five_artists_map`)
placing the **discographies of 9 indie artists** on the frozen MERT/MERIT
corpus. All track vectors are prewarmed into the default-session embed cache with
read-through fallback, so the map loads **instantly** in any browser session.

## Artists (245 tracks total)

| Artist | Source | Tracks | Releases | Years | Genre (display) |
|---|---|---|---|---|---|
| Allegra Krieger | iTunes | 51 | 6 | 2020–2024 | Alternative |
| Tasha | iTunes | 44 | 11 | 2017–2026 | Indie Rock |
| K. Porcelain | Bandcamp | 38 | 5 | — | — |
| Jana Horn | iTunes | 30 | 4 | 2020–2026 | Alternative |
| Hiding Places | iTunes | 29 | 8 | 2021–2026 | Alternative |
| Little Cliff | iTunes | 22 | 5 | 2019–2023 | Alternative |
| annihil | iTunes | 17 | 3 | 2023–2025 | Techno* |
| Lowaim | iTunes | 9 | 3 | 2024–2025 | Alternative |
| Untitled Freak | iTunes | 5 | 1 | 2025–2026 | Alternative |

\* annihil is tagged "Techno" by Apple but is the correct artist — verified by the
album *AGAINST THE GRAIN OF ASSUMPTION* matching the Bandcamp release. Genre is
display-only and never enters the model.

## How each source is ingested

- **iTunes (8 artists, 207 tracks).** `itunes.apple.com/lookup` → primary-artist
  filter → dedupe by base title (earliest release) → 30 s AAC preview →
  PyAV decode to mono 24 kHz → `embed_query_dual` (MERT-v1-330M 1024-d +
  MERIT 384-d) → `place()`. Track ids `itunes:<trackId>`.
- **Bandcamp (K. Porcelain, 38 tracks).** Not on iTunes or Deezer. Streams
  parsed from each album's `data-tralbum` blob (`t4.bcbits.com`, full tracks) →
  same decode + embed core. Track ids `bandcamp:<track_id>`.

## Files

- `discography_metadata_9artists.csv` — resolved discographies (10-col schema).
- `placements_9artists.csv` — per-track cluster, confidence, 2D coords, MERIT
  sub-norms (melody/rhythm/timbre), top-5 neighbors, embed method.
- `embeddings_5artists.npz` / `embeddings_2new.npz` / `embeddings_annihil.npz` /
  `embeddings_kporcelain.npz` — raw MERT (1024) + MERIT (384) vectors per source.
- `ui/custom_playlists/five_artists_map.json` — the premade map (245 tracks).
- `nine_artists_map.png` — static map figure.
- `per_artist_summary_9.csv` — per-artist rollup.

## Using it in the app

1. Restart the Anther server so it picks up the patched `ui/atlas.py`
   (read-through cache + prefix-guard + session-artist materialization fixes).
2. Search "Friends of Gunk" in the playlist search, click place — all 245
   songs place instantly from the prewarmed cache.
3. Build the artist graph from the song graph — all 9 artists embed (verified
   end-to-end: 9/9, 0 skipped).

## To place K. Porcelain (Bandcamp) on your LIVE server

The app's SSRF guard (`net_guard`) allowlists only iTunes/Spotify/Deezer CDNs.
K. Porcelain's audio is on `t4.bcbits.com`. Its vectors are already prewarmed
into the cache, so **placement works with no network call**. Only if you later
re-embed from source would you need to add `.bcbits.com` to
`ANTHER_ALLOWED_FETCH_HOSTS` (or the `DEFAULT_ALLOWED_HOSTS` list in
`net_guard.py`).

## Caveat — blank cluster labels

The `corpus_mpd_100k_merit_ext_500k` bundle's `cluster_profiles.json` has empty
`top_genres`/no `label_final`, so cluster **labels** render blank in the UI.
Cluster IDs, confidences, coordinates, and neighbors are all valid. Running the
genre-labeling CLI on the bundle would populate them (offered separately).
