"""
Step 2 — Split the popularity table into "already in corpus" vs "missing",
and build the popularity.json sidecar for the tracks that matched.

The existing corpus (models/corpus_mpd_100k_merit/index.json) carries NO
ISRC field, so the join key is normalized (artist, title) — the same
normalization used in step 1. This is a metadata-only join: no MusicBrainz
lookup needed for this step (see note below).

Usage:
    python 02_match_against_corpus.py \
        --bundle models/corpus_mpd_100k_merit \
        --popularity data/billboard/popularity_songs.csv \
        --out-dir data/billboard/

Outputs:
    data/billboard/popularity.json       — norm_key -> popularity fields,
                                            for every song in the table
                                            (used by step 3's reranker)
    data/billboard/matched_existing.csv  — popularity rows that matched an
                                            existing corpus track id
    data/billboard/missing_tracks.csv    — popularity rows with NO match;
                                            these need embedding (steps 4-5)
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_key(artist: str, title: str) -> str:
    def clean(s):
        s = s or ""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        for cut in (" feat.", " feat ", " ft.", " ft ", " (", " ["):
            i = s.find(cut)
            if i != -1:
                s = s[:i]
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return s.strip()
    return f"{clean(artist)} -- {clean(title)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/corpus_mpd_100k_merit")
    ap.add_argument("--popularity", default="data/billboard/popularity_songs.csv")
    ap.add_argument("--out-dir", default="data/billboard")
    args = ap.parse_args()

    with open(Path(args.bundle) / "index.json") as f:
        index_payload = json.load(f)
    corpus_meta = index_payload["metadata"]

    # normalized (artist, name) -> corpus track id (first-occurrence wins)
    corpus_by_key = {}
    for m in corpus_meta:
        key = normalize_key(m.get("artist") or "", m.get("name") or "")
        corpus_by_key.setdefault(key, m["id"])
    print(f"corpus: {len(corpus_meta)} tracks, {len(corpus_by_key)} unique (artist,title) keys")

    pop = pd.read_csv(args.popularity)
    pop["matched_track_id"] = pop["norm_key"].map(corpus_by_key)
    matched = pop[pop["matched_track_id"].notna()].copy()
    missing = pop[pop["matched_track_id"].isna()].copy()

    print(f"popularity table: {len(pop)} songs")
    print(f"  matched to existing corpus tracks: {len(matched)} ({len(matched)/len(pop):.1%})")
    print(f"  missing (need embedding):          {len(missing)} ({len(missing)/len(pop):.1%})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched.to_csv(out_dir / "matched_existing.csv", index=False)
    missing.drop(columns=["matched_track_id"]).to_csv(out_dir / "missing_tracks.csv", index=False)

    # popularity.json sidecar: norm_key -> popularity fields, for ALL songs
    # (place.py's reranker looks this up by corpus track id after matching;
    # extend_corpus will look it up by norm_key for newly-embedded tracks)
    popularity_sidecar = {
        row["norm_key"]: {
            "artist": row["artist"], "title": row["title"],
            "best_peak": int(row["best_peak"]),
            "weeks_on_chart_total": int(row["weeks_on_chart_total"]),
            "first_charted": row["first_charted"], "last_charted": row["last_charted"],
            "popularity_score": float(row["popularity_score"]),
        }
        for _, row in pop.iterrows()
    }
    with open(out_dir / "popularity.json", "w") as f:
        json.dump(popularity_sidecar, f)
    print(f"wrote popularity.json ({len(popularity_sidecar)} entries) -> {out_dir}")

    # Also write a direct track_id -> popularity_score map for O(1) reranking
    track_id_scores = {
        row["matched_track_id"]: row["popularity_score"] for _, row in matched.iterrows()
    }
    with open(out_dir / "popularity_by_track_id.json", "w") as f:
        json.dump(track_id_scores, f)
    print(f"wrote popularity_by_track_id.json ({len(track_id_scores)} entries) -> {out_dir}")

    print("\nNote: MusicBrainz ISRC lookup was evaluated and skipped for this "
          "join — the existing corpus carries no ISRC field, so the join key "
          "has to be normalized (artist,title) either way; Deezer's own "
          "fuzzy-match (step 4) doesn't require it either.")
