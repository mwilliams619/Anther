#!/usr/bin/env python
"""
Step 4: fetch Deezer preview audio for missing Billboard chart tracks and
embed them with the frozen MERT recipe (1024-d mean + 5120-d MERIT backbone).

Parallelized: the Deezer match+fetch step (network-latency-bound) runs
across a thread pool, while the GPU embedding step stays single-process
and runs on large batches -- one GPU forward pass per window, not per
thread. This overlaps ~10k I/O-bound HTTP round-trips against the 24
available CPU cores while keeping the RTX 4080 fed with full batches.

Resumable: writes one checkpoint row per attempted track to
`--out-jsonl` (append mode) as it goes, and skips any norm_key already
present in that file on restart. Safe to Ctrl-C and rerun with the same
args -- it picks up where it left off.

Usage
-----
    python scripts/billboard_expansion/03_fetch_and_embed_missing.py \\
        --missing data/billboard/missing_tracks.csv \\
        --out-jsonl data/billboard/embedded_tracks.jsonl \\
        --limit 0 \\
        --fetch-workers 16 \\
        --window-size 200 \\
        --embed-batch-size 16

Each output line is a JSON object:
    {
      "norm_key", "artist", "title", "popularity_score", ...
      "status": "ok" | "miss",
      "reason": <only if status == "miss">,
      "deezer_id", "match_method", "match_score",   <only if status == "ok">
      "embedding": [...1024 floats...],             <only if status == "ok">
      "merit_backbone": [...5120 floats...],        <only if status == "ok">
    }

Run `04_append_to_index.py` afterwards to fold the "ok" rows into the
corpus via extend_corpus() (additive, no refit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anther_ml.spotify_deezer import match_deezer_track, fetch_preview_waveform, MERT_SR
from anther_ml.embedding import load_mert, embed_tracks_batched_dual, prepare_waveform


def _norm_key(artist: str, title: str) -> str:
    def _n(s):
        s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
    return f"{_n(artist)} -- {_n(title)}"


def load_done_keys(out_jsonl: Path) -> set[str]:
    done = set()
    if out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(row["norm_key"])
    return done


def _fetch_one(row: dict, min_ratio: float, throttle: float, clip_seconds: float) -> dict:
    """Runs in a worker thread: Deezer match + preview download + decode.
    Returns a dict tagged with status='ok' (carries 'waveform', an in-memory
    np.ndarray -- never written to disk) or status='miss' (carries 'reason').
    Pure I/O + CPU decode, no GPU/model access -- safe to run concurrently.
    """
    artist, title = row["artist"], row["title"]
    norm_key = row.get("norm_key") or _norm_key(artist, title)
    base = {**row, "norm_key": norm_key}
    m = match_deezer_track({"title": title, "artist": artist, "isrc": None},
                            min_ratio=min_ratio, throttle=throttle)
    if "error" in m:
        return {**base, "status": "miss", "reason": m["error"]}
    try:
        wav, true_dur = fetch_preview_waveform(m["preview"], target_sr=MERT_SR, clip_seconds=clip_seconds)
    except Exception as e:
        return {**base, "status": "miss", "reason": f"fetch_failed:{type(e).__name__}"}
    y = prepare_waveform({"audio": wav, "sr": MERT_SR}, normalize=True)
    return {**base, "status": "fetched", "waveform": y, "deezer_id": m["deezer_id"],
            "match_method": m["match_method"], "match_score": m["match_score"],
            "true_duration": round(true_dur, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", default="data/billboard/missing_tracks.csv")
    ap.add_argument("--out-jsonl", default="data/billboard/embedded_tracks.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (process all)")
    ap.add_argument("--fetch-workers", type=int, default=16, help="thread-pool size for Deezer match+fetch")
    ap.add_argument("--window-size", type=int, default=200,
                     help="tracks fetched concurrently before handing the batch to the GPU")
    ap.add_argument("--embed-batch-size", type=int, default=16, help="GPU forward-pass sub-batch size")
    ap.add_argument("--min-ratio", type=float, default=0.82)
    ap.add_argument("--throttle", type=float, default=0.15,
                     help="seconds each worker sleeps between its own Deezer HTTP calls")
    ap.add_argument("--clip-seconds", type=float, default=30.0)
    args = ap.parse_args()

    missing_path = REPO_ROOT / args.missing
    out_path = REPO_ROOT / args.out_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(missing_path)
    if args.limit:
        df = df.head(args.limit)

    done_keys = load_done_keys(out_path)
    todo = df[~df["norm_key"].isin(done_keys)]
    print(f"{len(df)} total missing tracks, {len(done_keys)} already attempted, "
          f"{len(todo)} remaining this run", flush=True)
    if todo.empty:
        print("Nothing left to do.")
        return

    print("Loading MERT-v1-330M...", flush=True)
    model, processor, device = load_mert()

    rows = todo.to_dict("records")
    n_ok = n_miss = 0
    t0 = time.time()

    with out_path.open("a") as out_f, ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        for win_start in range(0, len(rows), args.window_size):
            window = rows[win_start: win_start + args.window_size]
            futures = [pool.submit(_fetch_one, row, args.min_ratio, args.throttle, args.clip_seconds)
                       for row in window]

            fetched_metas, fetched_waves = [], []
            for fut in as_completed(futures):
                res = fut.result()
                if res["status"] == "miss":
                    out_f.write(json.dumps({k: v for k, v in res.items() if k != "waveform"}) + "\n")
                    n_miss += 1
                else:
                    fetched_waves.append(res.pop("waveform"))
                    fetched_metas.append(res)

            # GPU embedding: single process, real batches -- not parallelized across threads.
            for b in range(0, len(fetched_waves), args.embed_batch_size):
                sub_waves = fetched_waves[b: b + args.embed_batch_size]
                sub_metas = fetched_metas[b: b + args.embed_batch_size]
                mert_mean, merit_backbone = embed_tracks_batched_dual(model, processor, sub_waves, device)
                for meta, emb, backbone in zip(sub_metas, mert_mean, merit_backbone):
                    out_f.write(json.dumps({
                        **meta, "status": "ok",
                        "embedding": emb.tolist(),
                        "merit_backbone": backbone.tolist(),
                    }) + "\n")
                    n_ok += 1
            out_f.flush()

            done_so_far = win_start + len(window)
            elapsed = time.time() - t0
            rate = done_so_far / elapsed if elapsed > 0 else 0
            remaining = len(rows) - done_so_far
            eta_min = remaining / rate / 60 if rate > 0 else float("nan")
            print(f"  {done_so_far}/{len(rows)}  ok={n_ok} miss={n_miss}  "
                  f"rate={rate:.2f}/s  eta={eta_min:.1f}min", flush=True)

    total = n_ok + n_miss
    print(f"\nDone this run: ok={n_ok} miss={n_miss}"
          + (f" ({n_ok/total*100:.1f}% match rate)" if total else ""))


if __name__ == "__main__":
    main()
