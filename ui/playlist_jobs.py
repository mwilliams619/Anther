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
import threading
import time

import atlas

_jobs: dict[str, dict] = {}          # job_id → record
_active_pid: dict[str, str] = {}     # pid → job_id while running (dup-add dedup)
_jobs_lock = threading.Lock()
_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
_worker: threading.Thread | None = None


def start(pid: str, name: str, pending: list[dict]) -> str:
    """Queue background placement of ``pending`` tracks
    ({"id","name","artist","preview_url"}). Idempotent per playlist: a second
    add while a job is running returns the existing job_id."""
    global _worker
    with _jobs_lock:
        existing = _active_pid.get(pid)
        if existing is not None and _jobs[existing]["state"] == "running":
            return existing
        job_id = f"pl-{pid[:8]}-{int(time.time() * 1000)}"
        _jobs[job_id] = {
            "job_id":    job_id,
            "pid":       pid,
            "name":      name,
            "state":     "running",
            "total":     len(pending),
            "placed":    0,
            "skipped":   [],
            "fragments": [],
            "message":   f"Queued {len(pending)} tracks…",
        }
        _active_pid[pid] = job_id
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()
    for track in pending:
        _queue.put((job_id, track))
    return job_id


def get_status(job_id: str, cursor: int = 0) -> dict:
    """Job progress plus fragments[cursor:] — the client echoes the returned
    cursor back so each fragment is delivered exactly once."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return {"state": "not_found"}
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
        }


def _worker_loop() -> None:
    while True:
        job_id, track = _queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            continue

        done_before = job["placed"] + len(job["skipped"])
        label = f"{track.get('name', '')} — {track.get('artist', '')}"
        with _jobs_lock:
            job["message"] = f"Embedding {done_before + 1}/{job['total']}: {label}"

        fragment, skip = None, None
        try:
            vec, _method = atlas.resolve_and_embed(track)
            fragment = atlas.place_external_track(track, vec, playlist_pid=job["pid"])
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
            if job["placed"] + len(job["skipped"]) >= job["total"]:
                job["state"] = "done"
                job["message"] = (f"Done: {job['placed']}/{job['total']} placed"
                                  + (f", {len(job['skipped'])} skipped"
                                     if job["skipped"] else ""))
                if _active_pid.get(job["pid"]) == job_id:
                    del _active_pid[job["pid"]]
