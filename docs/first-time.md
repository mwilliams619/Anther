# First time in this repo

1. This is a two-phase music-similarity/clustering pipeline. Start at
   [architecture.md](architecture.md) for the big picture and script index.
2. Read [invariants.md](invariants.md) **before changing anything** — a
   handful of rules make a change silently wrong (still runs, still returns
   numbers, just meaningless) rather than loudly wrong.
3. The reference corpus ([projects/REFERENCE_CORPUS_DESIGN.md](projects/REFERENCE_CORPUS_DESIGN.md))
   is a **frozen** map — songs, playlists, and albums are *placed onto* it via
   the phase 1/2 query path, never re-clustered from it. Don't reach for a
   rebuild when placement is what's needed.
4. Pick your subsystem from CLAUDE.md's Documentation map table and go
   straight there — it routes clustering, similarity, phase 1/2, the
   reference corpus, tagging, and the UI to their own short docs.
