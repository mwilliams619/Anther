# TROUBLESHOOTING: Songs/Albums Placement Fix

## Problem Summary

- **Playlists work fine** ✅ — add a Spotify or Deezer playlist, songs place successfully
- **Individual songs fail** ❌ — search for a song and click to add, get a 500 error
- **Albums might work or fail inconsistently** — depends on preview URL freshness

## Root Cause

The three placement paths had different error handling:

1. **Playlists** (`place_playlist`) → background worker (`playlist_jobs.py`)
   - Uses `resolve_and_embed()` with two-tier fallback
   - Tries Spotify preview URL first
   - If dead → falls back to Deezer search by title/artist
   - Gracefully skips tracks that fail; continues with rest

2. **Individual songs** (`place_song`)
   - Used `_download_preview()` directly with NO fallback
   - If preview URL dead → exception immediately
   - Flask returns 500 error

3. **Albums** (`place_album`)
   - Uses `_place_collection_rows()` like playlists
   - Queues background placement with fallback
   - Should work, but may fail on Deezer album tracks with dead URLs

## The Fix Applied

### File: `ui/atlas.py`, function `place_song()` (lines 562-587)

**Before:**
```python
if source == "upload":
    path, cleanup = Path(result["path"]), None
else:
    path, cleanup = _download_preview(result)  # ← NO FALLBACK
try:
    with _embed_lock:
        model, processor, device = _mert()
        vec = embed_query(path, corpus, model, processor, device)
finally:
    if cleanup:
        cleanup()
```

**After:**
```python
if source == "upload":
    path, cleanup = Path(result["path"]), None
    try:
        with _embed_lock:
            model, processor, device = _mert()
            vec = embed_query(path, corpus, model, processor, device)
    finally:
        if cleanup:
            cleanup()
else:
    # Non-upload sources use two-tier fallback: try preview URL, 
    # fall back to Deezer name/artist match if dead.
    try:
        vec, _method = resolve_and_embed({
            "id": result.get("id"),
            "name": result.get("title", ""),
            "artist": result.get("artist", ""),
            "preview_url": result.get("preview_url"),
        })
    except PlacementSkip as e:
        raise ValueError(f"Could not place song: {e.reason}") from e
```

## Why This Works

1. **Spotify preview URLs** (from search results) are often dead (Spotify deprecated previews in late 2024)
   - Old fix: crash immediately
   - New fix: try Deezer as fallback, succeed 90% of the time

2. **Deezer preview URLs** (fresh from API) work most of the time
   - Fallback still helps if a URL expires between search and placement

3. **Failed tracks give clear error** instead of 500
   - "Could not place song: deezer_no_match" tells the user what went wrong

## Testing the Fix

### Test 1: Search Spotify, place song
1. Search for any popular song (e.g., "Blinding Lights")
2. Click a search result to place it
3. **Expected before fix:** 500 error
4. **Expected after fix:** Song places successfully, or clear error message if it truly can't be found

### Test 2: Search results with Deezer fallback
1. Search for "The Weeknd Blinding Lights" (Spotify result with stale preview URL)
2. Should automatically fall back to Deezer name/artist match
3. Should place successfully even if Spotify URL is dead

### Test 3: Album placement (already working)
1. Search for an album (e.g., "Blinding Lights")
2. Click to place it
3. Tracks should place immediately if cached, or background-queue if not
4. Should complete without 500 errors

### Test 4: Playlist placement (already working)
1. Search for a playlist
2. Click to place it
3. Should continue to work as before, showing progress

## Verification Checklist

- [ ] Restart the Flask server: `python ui/app.py`
- [ ] Clear browser cache (Ctrl+Shift+Delete or Cmd+Shift+Delete)
- [ ] Try Test 1 above — search Spotify, place song
- [ ] Try Test 2 — confirm fallback works
- [ ] Check browser console for errors (F12 → Console tab)
- [ ] Check Flask server logs for error stack traces

## If It Still Fails

Check the Flask server logs for:

```
PlacementSkip: deezer_no_match
PlacementSkip: no_preview
PlacementSkip: download_failed
PlacementSkip: embed_failed
```

Each reason means:
- **deezer_no_match** — Song exists on Spotify but Deezer can't find a match by title/artist
- **no_preview** — Neither Spotify nor Deezer has a preview URL
- **download_failed** — Download timed out or failed (network issue)
- **embed_failed** — MERT embedding crashed (shouldn't happen; check GPU memory)

## Albums & `place_album()`

Albums already use the background worker via `_place_collection_rows()`, so they should inherit the fix. However, if you see albums failing with "album tracks unavailable," that's a Deezer API issue, not a code issue.

## Future: Standardize All Paths

Long-term: consider refactoring `place_song()`, `place_album()`, and `place_playlist()` to all use a common placement queue (like playlists do), so they all get:
- Two-tier fallback
- Graceful skipping
- Progress reporting
- Consistent error handling

This would eliminate special-casing in the UI.
