#!/usr/bin/env python
"""Append an existing extend_sql.py JSONL checkpoint without re-embedding."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anther_ml.corpus.checkpoint_append import append_checkpoint_in_batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=500,
                    help="checkpoint rows per append transaction (default: 500)")
    ap.add_argument("--dedupe-threshold", type=float, default=0.98)
    ap.add_argument("--knn-k", type=int, default=15)
    args = ap.parse_args()

    print(f"Appending checkpoint in batches of {args.batch_size} without embedding…",
          flush=True)
    report = append_checkpoint_in_batches(
        args.bundle, args.checkpoint, args.out_dir,
        batch_size=args.batch_size,
        dedupe_threshold=args.dedupe_threshold, knn_k=args.knn_k,
    )
    import json
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
