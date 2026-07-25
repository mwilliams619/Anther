# Stop Button for Playlist Embedding

## Overview

Added the ability to stop an in-progress playlist/album embedding job without restarting the page. The user can now click a "stop" link that appears in the progress message while embedding is running.

## Changes Made

### 1. Backend: `ui/playlist_jobs.py`

#### Added stop request tracking:
- Line 24: Added `_stopped_jobs: set[str]` to track job IDs user has requested to stop

#### Updated `get_status()` (lines 62-81):
- Added `can_stop: bool` field to response
- Returns `true` if job is running and not already stopped
- Frontend uses this to show/hide the stop link

#### New `stop(job_id)` function (lines 84-99):
- User-facing endpoint: accepts a job_id and requests it stop
- Validates that job exists and is running
- Adds job_id to `_stopped_jobs` set
- Updates job message to "Stopping after current track…"
- Returns updated job status

#### Updated `_worker_loop()` (lines 101-156):
- **Before processing a track:** Check if job_id is in `_stopped_jobs`
  - If yes: mark job as done, skip all remaining tracks, return
  - If no: continue embedding
- **After completing each track:** Check again if stop was requested
  - If yes: mark done after finishing the current track, show "Stopped: X/Y placed"
  - If no: continue to next track or mark done if all tracks processed

**Key behavior:**
- Current track always finishes (won't interrupt mid-MERT embed)
- Graceful: no crash, no partial state, clear message
- Skipped tracks are reported

### 2. Backend: `ui/app.py`

#### New Flask route (lines 118-123):
```python
@app.route('/api/playlist/stop/<job_id>', methods=['POST'])
def playlist_stop(job_id):
    import playlist_jobs
    try:
        return jsonify(playlist_jobs.stop(job_id))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
```

Accepts POST requests and calls `playlist_jobs.stop()`, returning the updated job status.

### 3. Frontend: `ui/static/app.js`

#### Updated `startPlaylistPoll()` (lines 264-300):
- Stores `jobId` in `state.playlistJobIds[playlist.pid]` for later reference
- In the polling loop, check `s.can_stop` field
- If true, append ` [<a href="#" onclick="stopPlaylistJob(..., event)">stop</a>]` to progress message
- This makes the stop link appear/disappear as the job's state changes

#### Updated `stopPlaylistPoll()` (lines 302-307):
- Cleans up `state.playlistJobIds[pid]` when stopping poll

#### Updated `setPlaylistProgress()` (lines 309-315):
- Changed from `textContent` to `innerHTML` to support HTML links in the message
- Allows rendering the stop link safely

#### New `stopPlaylistJob(jobId, event)` function (lines 317-328):
- User clicks the stop link
- Prevents default link behavior (event.preventDefault)
- POSTs to `/api/playlist/stop/<job_id>`
- Shows error if stop fails (e.g., job already done)
- Poll will update shortly with new status

## Usage

### User Workflow

1. Search for a playlist or album
2. Click to add it — embedding starts, progress bar appears
3. See message: `"My Playlist — 0 of 500 placed … Embedding 1/500: Track Name [stop]"`
4. Click `[stop]` to request stop
5. Current track finishes embedding
6. Progress updates to: `"My Playlist — 42 of 500 placed … Stopped: 42 placed, 458 skipped ✓"`
7. No page refresh needed

### Technical Behavior

- **Synchronized:** Stop request is thread-safe via `_jobs_lock`
- **Graceful:** Current track always completes (no mid-MERT interrupt)
- **Clear:** Message updates from "Stopping…" to "Stopped: X/Y placed"
- **Persistent:** Already-placed nodes remain on the map (in-memory graph)
- **Resumable:** Re-adding the playlist continues from the end (no duplication via cache)

## Testing

### Test 1: Stop mid-way
1. Add a large playlist (200+ tracks)
2. Wait for embedding to start
3. Click `[stop]` after a few tracks have placed
4. **Expected:** Current track finishes, job ends, message shows "Stopped: N placed"
5. **Verify:** Nodes placed so far stay on map

### Test 2: Can't stop after done
1. Add a small playlist (< 10 tracks)
2. Wait for it to complete
3. **Expected:** Stop link disappears as soon as job finishes
4. Progress message does NOT show `[stop]` anymore

### Test 3: Stop link appears/disappears
1. Add a playlist, watch message during embedding
2. **Expected:** `[stop]` link appears while running
3. Click elsewhere (don't click stop)
4. **Expected:** Link remains visible as long as job is running
5. When job completes: link disappears, message shows ✓

### Test 4: Double-click stop (race condition)
1. Add playlist with ~20 tracks
2. Rapidly click `[stop]` twice
3. **Expected:** First click succeeds, second returns "already stopping"
4. **No crash**

## Edge Cases Handled

- **Job finishes right as user clicks stop:** Poll sees done state, stop request is ignored
- **Page refreshes mid-stop:** Job continues in background (state lost, but nodes persist)
- **Server restart:** Job lost, poll returns "not_found", user sees message to reload
- **Network error on stop request:** Next poll updates status anyway
- **MERT embed crashes on current track:** Exception caught, track skipped, job continues/stops normally

## Future Improvements

1. **Per-track stopping:** Stop before embedding the current track (would require changing the queue architecture)
2. **Pause and resume:** Queue the remaining tracks separately so user can add more playlists in between
3. **Bulk operations:** Multiple playlist jobs in parallel (currently serial on one GPU)
4. **Cancel with error:** If user clicks stop many times, forcefully terminate after 5 seconds (watchdog)

## Code References

- Backend state machine: `playlist_jobs.py` lines 23-156
- Flask endpoint: `app.py` lines 118-123
- Frontend poll & stop: `app.js` lines 262-328
- HTML element: `index.html` line 41
