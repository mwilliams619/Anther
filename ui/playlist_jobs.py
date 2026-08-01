"""
Background embed-and-place runner for playlist additions.

place_playlist() returns a playlist's in-corpus/cached tracks synchronously
and hands the rest here: one persistent worker thread drains a queue of
(job_id, track) tasks — download preview (Spotify URL, Deezer fallback) →
MERT embed → merge into the atlas graph — so the GPU has a single consumer
and multiple queued playlists simply line up. The frontend polls
/api/playlist/status/<job_id>?cursor=N and splices each newly returned
fragment into the running force sim.

Job records are in-memory only: a server restart loses them (the poll gets
"not_found"), but placed nodes persist in graph.json and embeddings in the
embed cache, so re-adding the playlist is a cheap resume.
"""

import queue
import secrets
import threading

import atlas

_jobs: dict[str, dict] = {}          # job_id → record (record carries session_id)
_active_pid: dict[tuple, str] = {}   # (session_id, pid) → job_id while running (dup-add dedup)
_stopped_jobs: set[str] = set()      # job_ids that user has requested to stop
_jobs_lock = threading.Lock()
_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
_worker: threading.Thread | None = None


def forget_all() -> None:
    """Drop the *current session's* job bookkeeping (called on that session's
    map clear). Any worker thread still draining the queue for a forgotten job
    finds it gone from _jobs and just no-ops that track; new placements start a
    fresh job_id instead of being folded into a stale 'running' job that no one
    is polling for anymore. Other sessions' jobs are left untouched, so one
    user clearing their map can't kill another user's background placement."""
    sid = atlas.current_session_id()
    with _jobs_lock:
        drop = {jid for jid, j in _jobs.items() if j.get("session_id") == sid}
        for jid in drop:
            _jobs.pop(jid, None)
            _stopped_jobs.discard(jid)
        for key in [k for k in _active_pid if k[0] == sid]:
            del _active_pid[key]
    # Queued tasks for dropped jobs are left in the queue; the worker no-ops
    # them (their job_id is no longer in _jobs), which is cheap and avoids
    # disturbing other sessions' interleaved tasks in the shared queue.


def start(pid: str, name: str, pending: list[dict]) -> str:
    """Queue background placement of ``pending`` tracks
    ({"id","name","artist","preview_url"}). Idempotent per playlist: a second
    add while a job is running returns the existing job_id."""
    global _worker
    # Bind this job to the session that requested it (start runs in the Flask
    # request context, so the contextvar is set). The worker thread — which has
    # no request context — re-binds this id before placing, so tracks land in
    # the right user's map.
    sid = atlas.current_session_id()
    with _jobs_lock:
        existing = _active_pid.get((sid, pid))
        if existing is not None and _jobs[existing]["state"] == "running":
            return existing
        # Unguessable: the old pid+millisecond form could be enumerated by
        # anyone who knew the playlist id. Ownership is checked independently
        # (_owned_job) — this just removes the enumeration primitive.
        job_id = f"pl-{secrets.token_urlsafe(16)}"
        _jobs[job_id] = {
            "job_id":     job_id,
            "session_id": sid,
            "pid":        pid,
            "name":       name,
            "state":      "running",
            "total":      len(pending),
            "placed":     0,
            "skipped":    [],
            "fragments":  [],
            "message":    f"Queued {len(pending)} tracks…",
        }
        _active_pid[(sid, pid)] = job_id
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()
    for track in pending:
        _queue.put((job_id, track))
    return job_id


def _owned_job(job_id: str):
    """The job record iff it belongs to the calling session, else None.

    Job ids used to be guessable (playlist id + millisecond timestamp) and
    neither endpoint checked ownership, so anyone could poll another user's
    job — reading the fragments of their map — or cancel it. Ownership is
    enforced here rather than in the routes so every caller inherits it.
    Callers must hold _jobs_lock.
    """
    job = _jobs.get(job_id)
    if job is None or job.get("session_id") != atlas.current_session_id():
        return None
    return job


def _status_locked(job: dict, cursor: int = 0) -> dict:
    """Status payload for an already-resolved job. Caller holds _jobs_lock."""
    return {
        "state":     job["state"],
        "pid":       job["pid"],
        "name":      job["name"],
        "total":     job["total"],
        "placed":    job["placed"],
        "failed":    len(job["skipped"]),
        "message":   job["message"],
        "fragments": job["fragments"][cursor:],
        "cursor":    len(job["fragments"]),
        "skipped":   list(job["skipped"]),
        "can_stop":  job["state"] == "running" and job["job_id"] not in _stopped_jobs,
    }


def get_status(job_id: str, cursor: int = 0) -> dict:
    """Job progress plus fragments[cursor:] — the client echoes the returned
    cursor back so each fragment is delivered exactly once. Another session's
    job is reported as "not_found", same as a missing one, so the endpoint
    doesn't confirm that a given job id exists."""
    with _jobs_lock:
        job = _owned_job(job_id)
        if job is None:
            return {"state": "not_found"}
        return _status_locked(job, cursor)


def stop(job_id: str) -> dict:
    """Request that a running job stop after the current track.
    Returns the updated job status."""
    with _jobs_lock:
        job = _owned_job(job_id)
        if job is None:
            return {"state": "not_found"}
        if job["state"] != "running":
            return {"error": f"Job {job_id} is not running", "state": job["state"]}
        if job_id in _stopped_jobs:
            return {"error": f"Job {job_id} is already stopping", "state": "stopping"}

        _stopped_jobs.add(job_id)
        job["message"] = f"Stopping after current track… ({job['placed']}/{job['total']} placed)"
        # _status_locked, not get_status: _jobs_lock is not reentrant and is
        # already held here, so calling get_status would deadlock the request
        # thread with the lock held, hanging every other job endpoint.
        return _status_locked(job)


def _worker_loop() -> None:
    while True:
        job_id, track = _queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
            should_stop = job_id in _stopped_jobs if job else False
        if job is None:
            continue

        # Check if user requested stop — finish this track but then mark done
        if should_stop:
            done_before = job["placed"] + len(job["skipped"])
            job["message"] = f"Stopping… (skipping {job['total'] - done_before} remaining tracks)"
            with _jobs_lock:
                job["state"] = "done"
                akey = (job["session_id"], job["pid"])
                if _active_pid.get(akey) == job_id:
                    del _active_pid[akey]
                _stopped_jobs.discard(job_id)
            continue

        done_before = job["placed"] + len(job["skipped"])
        label = f"{track.get('name', '')} — {track.get('artist', '')}"
        with _jobs_lock:
            job["message"] = f"Embedding {done_before + 1}/{job['total']}: {label}"

        fragment, skip = None, None
        try:
            # Bind the requesting session so the embed cache read/write and the
            # graph merge land in that user's session, not the worker thread's
            # default. resolve_and_embed → cached_vec/cache_vec and
            # place_external_track → _merge_fragment all read the contextvar.
            with atlas.use_session(job["session_id"]):
                vec, backbone, _method = atlas.resolve_and_embed(track)
                fragment = atlas.place_external_track(
                    track, vec, playlist_pid=job["pid"], backbone=backbone)
        except atlas.PlacementSkip as e:
            skip = e.reason
        except Exception as e:  # noqa: BLE001 — one bad track never kills the job
            skip = f"error:{e}"

        with _jobs_lock:
            if fragment is not None:
                job["fragments"].append(fragment)
                job["placed"] += 1
            else:
                job["skipped"].append({"id": track.get("id"),
                                       "name": track.get("name", ""),
                                       "artist": track.get("artist", ""),
                                       "reason": skip})
            # Check if we should stop after this track
            if job_id in _stopped_jobs:
                job["state"] = "done"
                job["message"] = (f"Stopped: {job['placed']}/{job['total']} placed"
                                  + (f", {len(job['skipped'])} skipped"
                                     if job["skipped"] else ""))
                akey = (job["session_id"], job["pid"])
                if _active_pid.get(akey) == job_id:
                    del _active_pid[akey]
                _stopped_jobs.discard(job_id)
            elif job["placed"] + len(job["skipped"]) >= job["total"]:
                job["state"] = "done"
                job["message"] = (f"Done: {job['placed']}/{job['total']} placed"
                                  + (f", {len(job['skipped'])} skipped"
                                     if job["skipped"] else ""))
                akey = (job["session_id"], job["pid"])
                if _active_pid.get(akey) == job_id:
                    del _active_pid[akey]
