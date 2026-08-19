# Indie Artist Map — 9 bands, top ~15 tracks each

Premade map for the Anther UI: the most-popular ~15 tracks per artist placed on
the frozen MERT/MERIT corpus map (`models/corpus_mpd_100k_merit_ext_500k`).

## Roster (119 tracks)
| Artist | Source | Tracks | Notes |
|---|---|---|---|
| Allegra Krieger | iTunes | 15 | |
| Tasha | iTunes | 15 | matches corpus low-confidence pool on artist graph |
| K. Porcelain | Bandcamp | 15 | audio bundled locally (see below) |
| Jana Horn | iTunes | 15 | |
| Hiding Places | iTunes | 15 | |
| Little Cliff | iTunes | 15 | |
| annihil | iTunes | 15 | electronic — sits apart at lower-left of map |
| Lowaim | iTunes | 9 | full discography (<15) |
| Untitled Freak | iTunes | 5 | full discography (<15) |

## Popularity ranking (how "top 15" was chosen)
- **iTunes artists:** per-track Deezer popularity `rank` (lower = more played),
  matched by exact artist name; ties and Deezer's 100000 floor broken by recency.
- **K. Porcelain:** not on Deezer, so ranked by the order in the artist's own
  *Greatest Hits Mix* album, then remaining tracks by year.

## K. Porcelain audio (Bandcamp-only)
K. Porcelain is on neither iTunes nor Spotify nor Deezer, so the app's normal
preview lookup can't find playable audio. The 15 kept tracks' audio is bundled
in `ui/custom_audio/<track_id>.mp3` (~17 MB) and served by a new route
`/api/custom-audio/<name>` in `ui/app.py`. `atlas.get_preview_url` gained a
`bandcamp:` branch that returns `/api/custom-audio/<id>.mp3` when the file is
present (else None — never a wrong-song Deezer fallback). Playback is shared
across browser sessions (unlike per-session uploads) because a premade map must
play for every visitor, and supports Range requests for seeking.

## Files
- `five_artists_map.json` — the premade map (pid `five_artists_map`, 119 tracks).
- `placements_5artists_top15.csv` — 2D coords, cluster, neighbors, deezer_rank per kept track.
- `per_artist_summary_top15.csv` — per-artist counts / cluster spread / map centroid.
- `nine_artists_map_top15.png` — static rendered map figure.
- `ui/custom_audio/*.mp3` — bundled K. Porcelain audio.

## To use
1. Restart the Anther server (JSON + code are read at startup).
2. Search the map box for "Indie Artist Map" / "9 bands" / "top 15".
3. Click to place — all 119 land instantly (prewarmed into the default-session
   cache, which every session reads through). Build the artist graph to get all 9.

## Caveat
The ext_500k corpus bundle's `cluster_profiles.json` has empty `top_genres`, so
cluster *labels* render blank in the UI — a property of that bundle, not the
placement. Cluster IDs, confidences, coords, and neighbors are all valid.
