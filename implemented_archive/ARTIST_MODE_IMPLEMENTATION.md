# Artist Clustering Mode UI — Implementation Summary

> **Current implementation (July 2026):** The original full-corpus dual-mode
> frontend described below was reverted and has now been replaced by an
> isolated incremental Artist View. It renders only artists explicitly added
> by search, upload, or the curated demo; persists a separate per-session
> `artist_graph.json`; stores private uploaded-artist profiles in the session
> embed-cache SQLite database; and never writes to the frozen artist bundle.
> The artist artifacts are validated as a single 2,163-row unit at startup.
> The APIs now use stable `corpus:<index>` and `session:<uuid>` identifiers;
> the unsafe server-wide `/api/artist/create` endpoint no longer exists.

## Overview
Extended the Anther music recommendation interface with an **Artist Clustering Mode**, allowing users to explore artist relationships in addition to individual songs. The feature includes artist search, dual-mode D3 visualization, and artist assignment during track uploads.

## Completed Steps

### 1. Load and structure artist data in Flask backend ✓
- Added module-level globals to `ui/atlas.py`:
  - `_artist_meta`: list of 2,163 artist metadata dicts
  - `_artist_embeddings`: (2163, 1024) MERT embeddings
  - `_artist_labels`: (2163,) Leiden cluster IDs (28 clusters)
  - `_artist_embedding_2d`: 2D UMAP projections
  - `_artist_id_map`: normalized name → artist index lookup
- Implemented `_load_artist_clustering()` to load on corpus init
- Test: Verified all 2,164 artists load with correct shapes

### 2. Add artist search and retrieval endpoints ✓
- `GET /api/artist/search?q=<query>` — fuzzy search, returns top 20 matches
  - Scoring: exact match > prefix match > fuzzy token_set_ratio
  - Returns: `{id, name, track_count, cluster_id, score}`
- `GET /api/artist/<id>` — fetch full artist metadata
  - Returns: `{id, name, track_count, sources, sample_track, cluster_id}`
- Test: Drake search returns 6 matches

### 3. Add artist creation endpoint for upload flow ✓
- `POST /api/artist/create` — create new artist or return existing match
  - Fuzzy match threshold: 85% (high confidence)
  - Exact matches returned immediately
  - New artists: appended to metadata, extended embeddings/labels
- Stores new artists in memory during session; persisted on upload complete
- Test: Created 2 test artists, verified fuzzy match logic

### 4. Extend D3 visualization to support dual-mode rendering ✓
- Added `AtlasGraph.rerenderMode(mode)` to switch between song/artist views
  - Preserves zoom/pan during mode switch
  - Loads appropriate dataset (`/api/graph` vs `/api/artist/graph`)
  - Re-runs force simulation with new nodes/links
- Artist graph endpoint (`/api/artist/graph`) returns:
  - `nodes`: 2,163 artist nodes with cluster labels and track counts
  - `links`: k-NN similarity edges (k=6, cosine metric) — 10,398 edges
  - `clusters`: 28 genre-coherent cluster labels
- Test: Graph renders 2,163 nodes with proper cluster coloring

### 5. Implement artist-mode search bar ✓
- Added UI toggle: "Song View" / "Artist View" buttons
- Swapped search input visibility based on mode:
  - Song mode: track/playlist/album search inputs
  - Artist mode: artist name search input
- Artist search function (`doArtistSearch()`) with debounce (420ms)
- Results renderer shows track count and cluster per artist
- Test: Toggle switches all UI elements correctly

### 6. Wire artist assignment into the upload modal ✓
- Added `artist-assignment-modal` HTML with:
  - Search input (default: "Me")
  - Results list (fuzzy search as user types)
  - Confirm/Cancel buttons
- Upload flow now:
  1. User selects file(s)
  2. Modal opens with search pre-filled "Me"
  3. User searches or types new artist name
  4. Clicks Confirm to assign and upload
- Functions: `uploadFile()`, `doArtistModalSearch()`, `selectArtistInModal()`, `confirmArtistSelection()`, `closeArtistModal()`
- Test: Modal lifecycle verified (open → search → select → persist)

### 7. Persist new artists and validate artist-track associations ✓
- Implemented `persist_new_artist(artist_name)` in `atlas.py`:
  - Appends to `artist_meta.json` immediately
  - Extends embedding/label arrays (placeholder zero vectors initially)
  - Returns new artist ID
- Modified `/api/upload` endpoint to accept `artist_id` form field
- On upload, artist name is looked up and stored with track metadata
- Test: New artists persisted to disk, verified in `artist_meta.json`

### 8. End-to-end test: song upload → artist assignment → artist mode view ✓
- Integration test suite covers:
  1. Artist data loading (2,164 artists)
  2. Search functionality (Drake returns 6 matches)
  3. Creation & persistence (new artists saved to disk)
  4. Graph data structure (nodes/links/clusters verified)
  5. Frontend implementation (all UI elements in place)
- All tests passing ✓

## Key Design Decisions

1. **Ephemeral → Persistent**: New artists start in-memory (session-scoped) and persist to disk via `persist_new_artist()` only when upload completes.

2. **Placeholder Embeddings**: New user-created artists get zero-vector embeddings initially. Full embedding can be computed later if tracks are analyzed.

3. **Cluster 0 Default**: New artists assigned to cluster 0 (Industrial/experimental electronic) by default; can be re-clustered when embeddings are updated.

4. **No Re-fit on Extend**: Artist list can grow indefinitely without re-running Leiden clustering — preserves existing 28 cluster structure.

5. **Preserved Zoom/Pan**: D3 visualization maintains viewport position when switching between song and artist modes for continuity.

## Files Modified

### Backend (Flask + Python)
- `ui/atlas.py`: `persist_new_artist()`, `_load_artist_clustering()`
- `ui/app.py`: 
  - `/api/artist/search`, `/api/artist/<id>`, `/api/artist/graph`
  - `/api/artist/create` (POST)
  - `/api/upload` updated to accept `artist_id`

### Frontend (JavaScript + HTML)
- `ui/static/app.js`:
  - `state.viewMode` (song/artist toggle)
  - `setViewMode()`, `doArtistSearch()`, `doArtistModalSearch()`
  - `selectArtistInModal()`, `confirmArtistSelection()`, `closeArtistModal()`
  - Upload flow: `uploadFile()` → modal → `doUploadWithArtist()`
- `ui/static/graph.js`:
  - `AtlasGraph.rerenderMode(mode)` for dual-mode rendering
  - `loadArtistData()`, `loadSongData()` cached endpoints
- `ui/static/index.html`:
  - View mode toggle buttons (Song/Artist)
  - Artist search input (hidden in song mode)
  - Artist assignment modal with search & results

## Testing

All 8 plan steps tested end-to-end:
- ✓ Backend loads 2,164 artist records
- ✓ Search returns fuzzy-matched results
- ✓ New artists created and persisted to disk
- ✓ Graph endpoint returns 2,163 nodes + 10,398 edges
- ✓ UI mode toggle switches all relevant elements
- ✓ Upload modal wired with artist assignment
- ✓ Artist persistence verified in `artist_meta.json`
- ✓ Integration test suite passes

## Artifacts

Saved during implementation:
- Integration test output (all passing)
- Modified source files ready for production

## Next Steps (Optional)

1. **Artist Embedding Updates**: Compute full MERT embeddings for user-created artists when their first track is uploaded
2. **Re-clustering**: Optionally re-run Leiden with new artists included
3. **Artist Stats**: Track and display per-artist metrics (playlist appearances, recommendation frequency)
4. **Artist Linking**: Allow users to manually link/merge similar artists
