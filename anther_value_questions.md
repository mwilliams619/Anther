# Anther — 10 High-Value Questions the Cluster/Map Can Answer

The answer surface = the frozen reference corpus (99,618 MERT-embedded MPD tracks,
84 Leiden clusters w/ profiles, 2,000-term micro-genre tag vocab + per-track tags)
PLUS the live session graph (uploads, imported playlists/albums, and their placed
neighbors/clusters/tags computed after freeze).

| # | Question | Who | Value | Backed by |
|---|----------|-----|-------|-----------|
| 1 | What established artists/tracks does my uploaded demo sound closest to? | Producer | Creative positioning; "where do I land?" | `sounds_like` on placed upload |
| 2 | Which micro-genres does my track actually fall into? | Release prep | Pitch language, playlist targets, DSP genre fields | tag probe at placement (2000-term vocab) |
| 3 | What artists sit sonically between Artist A and Artist B? | DJ / curator | Transition tracks, genre-blend playlists | `bridge` (midpoint vec + A/B lean) |
| 4 | Which of Artist X's songs best matches this reference track? | Sync / A&R | Pick the right catalog track for a mood | `artist_tracks` |
| 5 | If I cross into an adjacent sound, which cluster + tracks? | Artist dev | Expand sound without abandoning identity | `crossover` (near neighbors in other clusters) |
| 6 | Is my imported playlist coherent; which tracks are outliers? | Curator | Playlist QC, prune the odd track | live-map cluster/neighbor spread of placed nodes |
| 7 | Which songs share my anchor's micro-genre descriptors, not just proximity? | Crate-digger | Same vibe, different corner | `tagmates` |
| 8 | How close are these two tracks, and what does each pull toward? | A&R | Quantify difference, see each pole's flavor | `compare` (cosine + each side's neighbors) |
| 9 | What are the major sonic territories here and what defines each? | Any / onboarding | Map literacy; macro fit | 84 cluster profiles (labels, exemplars, top playlists/tags, entropy) |
| 10 | Given my seed songs, what should I add next? | Discovery / playlisting | Personalized crate-digging from taste blend | multi-seed `recommend` |
