#!/usr/bin/env python
"""
Step 5: fold the successfully-embedded Billboard chart tracks into the
reference corpus via extend_corpus() -- additive append only, no refit of
StandardScaler/PCA/Leiden/UMAP, existing cluster ids and vectors untouched.

Reads `--embedded-jsonl` (output of 03_fetch_and_embed_missing.py, only
rows with status=="ok"), builds new_items for extend_corpus(), and writes
the extended bundle to `--out-dir` (a NEW directory by default -- never
overwrites the source bundle in place, so the original stays available
for rollback/comparison).

Usage
-----
    python scripts/billboard_expansion/04_append_to_index.py \\
        --bundle models/corpus_mpd_100k_merit \\
        --embedded-jsonl data/billboard/embedded_tracks.jsonl \\
        --out-dir models/corpus_mpd_100k_merit_ext_billboard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anther_ml.corpus.checkpoint_append import append_jsonl_in_batches


def make_track_id(row: dict) -> str:
    return f"billboard:deezer:{row['deezer_id']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/corpus_mpd_100k_merit")
    ap.add_argument("--embedded-jsonl", default="data/billboard/embedded_tracks.jsonl")
    ap.add_argument("--out-dir", default="models/corpus_mpd_100k_merit_ext_billboard")
    ap.add_argument("--dedupe-threshold", type=float, default=0.98)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    embedded_path = REPO_ROOT / args.embedded_jsonl
    def row_to_item(row: dict) -> dict | None:
        if row.get("status") != "ok":
            return None
        meta = {
            "id": make_track_id(row),
            "name": row["title"],
            "artist": row["artist"],
            "source": "billboard_hot100",
            "genre": None,
            "playlists": [],
            "billboard_popularity_score": row.get("popularity_score"),
            "billboard_best_peak": row.get("best_peak"),
            "billboard_weeks_on_chart": row.get("weeks_on_chart_total"),
            "deezer_match_method": row["match_method"],
            "deezer_match_score": row["match_score"],
        }
        return {
            "meta": meta,
            "raw_vector": np.asarray(row["embedding"], dtype=np.float32),
            "merit_backbone": np.asarray(row["merit_backbone"], dtype=np.float32)
            if row.get("merit_backbone") is not None else None,
        }

    print(f"Streaming append {embedded_path} in batches of {args.batch_size} …")
    report = append_jsonl_in_batches(
        REPO_ROOT / args.bundle, embedded_path, REPO_ROOT / args.out_dir,
        row_to_item=row_to_item, batch_size=args.batch_size,
        dedupe_threshold=args.dedupe_threshold, knn_k=args.knn_k,
        state_name=".billboard_append_state.json",
    )
    import json
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
