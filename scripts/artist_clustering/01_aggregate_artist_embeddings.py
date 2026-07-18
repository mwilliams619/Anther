"""
Aggregate per-track MERT embeddings into per-artist mean vectors.

Only the frozen 1024-d MERT mean embedding (corpus.embeddings) is aggregated --
NOT the MERIT-aggregate sidecar, which is link-calibrated for song-to-song
similarity and not designed for centroid averaging. Artists below
--min-tracks are dropped: a 1- or 2-track mean is mostly noise and would
dominate nothing but crowd the k-NN graph.

Usage:
    python scripts/artist_clustering/01_aggregate_artist_embeddings.py \
        --bundle models/corpus_mpd_100k_merit_ext_billboard \
        --min-tracks 5 \
        --out-dir data/artist_clustering
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anther_ml.corpus.bundle import ReferenceCorpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/corpus_mpd_100k_merit_ext_billboard")
    ap.add_argument("--min-tracks", type=int, default=5)
    ap.add_argument("--out-dir", default="data/artist_clustering")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = ReferenceCorpus.load(args.bundle, verify=True)
    print(f"Loaded corpus: {corpus.n_tracks} tracks")

    by_artist_idx = defaultdict(list)
    for i, m in enumerate(corpus.index.metadata):
        artist = (m.get("artist") or "").strip()
        if not artist or artist in ("???", "Unknown Artist"):
            continue
        by_artist_idx[artist].append(i)

    print(f"Unique artists (raw): {len(by_artist_idx)}")

    kept_artists = {a: idx for a, idx in by_artist_idx.items() if len(idx) >= args.min_tracks}
    print(f"Artists with >= {args.min_tracks} tracks: {len(kept_artists)}")

    names = sorted(kept_artists.keys())
    vecs = np.zeros((len(names), corpus.embeddings.shape[1]), dtype=np.float32)
    meta = []
    for k, name in enumerate(names):
        idx = kept_artists[name]
        vecs[k] = corpus.embeddings[idx].mean(axis=0)
        # representative track sample for downstream display / QA
        sample = corpus.index.metadata[idx[0]]
        sources = sorted({corpus.index.metadata[i].get("source") for i in idx})
        meta.append({
            "artist": name,
            "n_tracks": len(idx),
            "sources": sources,
            "sample_track": sample.get("name"),
        })

    np.save(out_dir / "artist_embeddings.npy", vecs)
    with open(out_dir / "artist_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Wrote {vecs.shape[0]} artist vectors (dim={vecs.shape[1]}) to {out_dir}/artist_embeddings.npy")
    print(f"Wrote artist metadata to {out_dir}/artist_meta.json")


if __name__ == "__main__":
    main()
