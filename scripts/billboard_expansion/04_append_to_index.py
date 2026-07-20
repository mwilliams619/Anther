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
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anther_ml.corpus.extend import extend_corpus


def load_ok_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def make_track_id(row: dict) -> str:
    return f"billboard:deezer:{row['deezer_id']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/corpus_mpd_100k_merit")
    ap.add_argument("--embedded-jsonl", default="data/billboard/embedded_tracks.jsonl")
    ap.add_argument("--out-dir", default="models/corpus_mpd_100k_merit_ext_billboard")
    ap.add_argument("--dedupe-threshold", type=float, default=0.98)
    ap.add_argument("--knn-k", type=int, default=15)
    args = ap.parse_args()

    embedded_path = REPO_ROOT / args.embedded_jsonl
    rows = load_ok_rows(embedded_path)
    print(f"Loaded {len(rows)} successfully-embedded tracks from {embedded_path}")
    if not rows:
        print("Nothing to append.")
        return

    new_items = []
    for row in rows:
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
        new_items.append({
            "meta": meta,
            "raw_vector": np.asarray(row["embedding"], dtype=np.float32),
            "merit_backbone": np.asarray(row["merit_backbone"], dtype=np.float32)
            if row.get("merit_backbone") is not None else None,
        })

    print(f"Extending bundle {args.bundle} -> {args.out_dir} "
          f"({len(new_items)} candidate new tracks)...")
    report = extend_corpus(
        REPO_ROOT / args.bundle,
        new_items,
        out_dir=REPO_ROOT / args.out_dir,
        dedupe_threshold=args.dedupe_threshold,
        knn_k=args.knn_k,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
