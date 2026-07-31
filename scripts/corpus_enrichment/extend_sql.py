#!/usr/bin/env python
"""Add a popularity-stratified SQL/MPD sample to a frozen corpus.

This intentionally uses ``extend_corpus`` rather than ``build_corpus``:
embeddings are appended, new songs receive frozen k-NN cluster assignments,
and the existing StandardScaler/PCA/Leiden/UMAP artifacts are never refit.

Example (100k additional tracks, 40% long tail / 35% middle / 25% popular):
    python scripts/corpus_enrichment/extend_sql.py \
        --bundle models/corpus_mpd_100k_merit \
        --out-dir models/corpus_mpd_100k_merit_ext_200k \
        --sql-db data/mpd_dump/spotifydbdumpshare.sqlite \
        --target-n 100000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import json
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anther_ml.corpus.checkpoint_append import (
    append_checkpoint_in_batches,
)
from anther_ml.corpus.sources import sql_source_oversampled
from anther_ml.embedding import embed_tracks_batched_dual, load_mert, prepare_waveform


def _source(args, lo: float | None, hi: float | None, n: int, seed: int,
            exclude_ids: set[str]):
    return sql_source_oversampled(
        db_path=args.sql_db,
        sql_dump=args.sql_dump,
        target_n=n,
        oversample_ratio=args.oversample_ratio,
        tracks_per_artist_cap=args.artist_cap,
        seed=seed,
        min_popularity=lo,
        max_popularity=hi,
        with_membership=not args.no_membership,
        n_workers=args.fetch_workers,
        exclude_ids=exclude_ids,
        allow_short=True,  # pre-flight already sized each stratum; never abort mid-run
    )


def _bare(track_ids: set[str]) -> set[str]:
    """``spotify:<id>`` → ``<id>``, the form the SQL layer filters on."""
    return {x.split(":", 1)[1] if x.startswith("spotify:") else x for x in track_ids}


def _plan_strata(
    strata: list[tuple[int, int, int]],
    db_path: str,
    *,
    artist_cap: int | None,
    exclude_ids: set[str],
    spill: bool,
) -> list[tuple[int, int, int]]:
    """
    Clamp each stratum to what the DB can actually supply, optionally spilling
    the shortfall into bands that still have headroom.

    The popularity distribution in this dump is brutally skewed — 12.7M of 13.3M
    tracks sit at popularity 0-30, the 31-70 band caps out near 200k under an
    artist cap of 5, and 71-100 holds ~1.3k tracks *in total*. So a nominally
    reasonable 25/50/25 split at 600k tracks is arithmetically impossible, and
    without this check you find out one stratum at a time, hours apart.
    """
    from anther_ml.mpd_sql import count_candidates

    bare = _bare(exclude_ids)
    # Each count is a full scan of the track table, so skip the bands that
    # cannot matter: one whose quota is already filled is only interesting as a
    # spill destination, and with --no-spill it is not even that.
    avail = [
        count_candidates(
            db_path, artist_cap=artist_cap, min_popularity=lo, max_popularity=hi,
            exclude_track_ids=bare,
        ) if (want or spill) else 0
        for lo, hi, want in strata
    ]
    planned = [min(want, have) for (_, _, want), have in zip(strata, avail)]
    deficit = sum(want for _, _, want in strata) - sum(planned)

    if deficit and spill:
        # Deterministic: largest headroom first, so the low-popularity long tail
        # absorbs the shortfall before the thinner bands do.
        headroom = sorted(
            range(len(strata)),
            key=lambda i: (avail[i] - planned[i], -i),
            reverse=True,
        )
        for i in headroom:
            if deficit <= 0:
                break
            take = min(deficit, avail[i] - planned[i])
            planned[i] += take
            deficit -= take

    print("stratum plan (popularity band: requested → planned / available):")
    for (lo, hi, want), have, got in zip(strata, avail, planned):
        flag = (
            "  ← DB-limited" if got < want
            else f"  ← +{got - want:,} spilled in" if got > want
            else ""
        )
        print(f"  {lo:>3}-{hi:<3}: {want:>8,} → {got:>8,} / {have:,}{flag}")
    if deficit > 0:
        print(f"  short by {deficit:,} tracks: the DB cannot supply the requested "
              f"total under these filters (artist_cap={artist_cap}).")
    return [(lo, hi, got) for (lo, hi, _), got in zip(strata, planned)]


_ID_RE = re.compile(r'^\{"id":\s*"([^"]+)"')
_POP_RE = re.compile(r'"popularity":\s*(null|-?[\d.eE+]+)')


def _scan_row(line: str) -> tuple[str, object] | None:
    """
    ``(id, popularity)`` for one checkpoint line, or ``None`` to skip it.

    Each row carries a 1024-d embedding *and* a MERT backbone, so lines average
    ~110 KB and ``json.loads`` on a multi-checkpoint exclusion set costs ~16
    minutes of pure parse before a single track is fetched. Everything we need
    lives in the row's first few hundred bytes (``id``, then ``meta``) and its
    last few (``status``), so match those directly and only fall back to a full
    parse when the layout is not the one this script writes.
    """
    id_match = _ID_RE.match(line)
    if id_match and line.rstrip().endswith('"status": "ok"}'):
        head = line[: id_match.end() + 4096]
        pop_match = _POP_RE.search(head)
        if pop_match:
            raw = pop_match.group(1)
            return id_match.group(1), None if raw == "null" else float(raw)
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    if row.get("status") != "ok" or not row.get("id"):
        return None
    return row["id"], (row.get("meta") or {}).get("popularity")


def _checkpoint_stats(path: Path) -> tuple[set[str], Counter]:
    ids: set[str] = set()
    strata: Counter = Counter()
    if not path.exists():
        return ids, strata
    with path.open() as f:
        for line in f:
            scanned = _scan_row(line)
            if scanned is None:
                continue
            track_id, pop = scanned
            ids.add(track_id)
            if pop is None:
                continue
            strata["low" if pop <= 30 else "middle" if pop <= 70 else "high"] += 1
    return ids, strata


def _bundle_ids(bundle: Path) -> set[str]:
    index = bundle / "index.json"
    if not index.exists():
        return set()
    with index.open() as f:
        data = json.load(f)
    return {row["id"] for row in data.get("metadata", []) if row.get("id")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="new bundle directory; use a new dir for rollback safety")
    ap.add_argument("--sql-db", default=None)
    ap.add_argument("--sql-dump", default=None)
    ap.add_argument("--target-n", type=int, default=100_000)
    ap.add_argument("--long-tail-ratio", type=float, default=0.40,
                    help="Spotify popularity 0-30")
    ap.add_argument("--middle-ratio", type=float, default=0.35,
                    help="Spotify popularity 31-70")
    ap.add_argument("--popular-ratio", type=float, default=0.25,
                    help="Spotify popularity 71-100")
    ap.add_argument("--artist-cap", type=int, default=5)
    ap.add_argument("--oversample-ratio", type=float, default=1.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fetch-workers", type=int, default=16)
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--append-batch-size", type=int, default=500,
                    help="checkpoint rows per append transaction (default: 500)")
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-membership", action="store_true")
    ap.add_argument("--dedupe-threshold", type=float, default=0.98)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--checkpoint", default=None,
                    help="JSONL embedding checkpoint (default: <out-dir>.jsonl)")
    ap.add_argument("--exclude-checkpoint", action="append", default=[],
                    help="prior JSONL checkpoint to exclude before fetching; repeatable")
    ap.add_argument("--no-spill", action="store_true",
                    help="do not move a DB-limited stratum's shortfall into bands "
                         "with headroom; embed fewer tracks instead")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the per-stratum availability count (a full scan of "
                         "the track table per band, seconds to a minute each)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the stratum plan and exit without embedding")
    args = ap.parse_args()

    if not (args.sql_db or args.sql_dump):
        ap.error("one of --sql-db or --sql-dump is required")
    ratios = [args.long_tail_ratio, args.middle_ratio, args.popular_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        ap.error("long-tail, middle, and popular ratios must sum to 1")

    checkpoint = Path(args.checkpoint or (args.out_dir + ".jsonl"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done, existing = _checkpoint_stats(checkpoint)
    exclude_ids = set(done) | _bundle_ids(Path(args.bundle))
    for prior in args.exclude_checkpoint:
        prior_ids, _ = _checkpoint_stats(Path(prior))
        exclude_ids.update(prior_ids)
    counts = [round(args.target_n * r) for r in ratios]
    counts[-1] += args.target_n - sum(counts)
    existing_counts = [existing["low"], existing["middle"], existing["high"]]
    strata = [
        (0, 30, max(0, counts[0] - existing_counts[0])),
        (31, 70, max(0, counts[1] - existing_counts[1])),
        (71, 100, max(0, counts[2] - existing_counts[2])),
    ]

    if not args.no_preflight:
        if not args.sql_db:
            ap.error("--no-preflight is required when using --sql-dump without --sql-db")
        strata = _plan_strata(
            strata, args.sql_db, artist_cap=args.artist_cap,
            exclude_ids=exclude_ids, spill=not args.no_spill,
        )
    if args.plan_only:
        return

    print(
        f"Embedding {sum(n for _, _, n in strata)} remaining tracks into "
        f"{checkpoint}; excluding {len(exclude_ids)} known IDs …", flush=True
    )
    model, processor, device = load_mert()
    with checkpoint.open("a") as out:
        for stratum, (lo, hi, want) in enumerate(strata):
            if want == 0:
                continue
            source = _source(args, lo, hi, want, args.seed + stratum, exclude_ids)
            batch_items, batch_waves = [], []
            for item in source:
                if item["id"] in done:
                    continue
                try:
                    wave = prepare_waveform(item, normalize=True)
                except Exception as exc:  # keep a bad preview from killing hours of work
                    print(f"skipping {item['id']}: {exc}", flush=True)
                    continue
                batch_items.append(item)
                batch_waves.append(wave)
                if len(batch_items) < args.embed_batch_size:
                    continue
                vecs, backbones = embed_tracks_batched_dual(
                    model, processor, batch_waves, device,
                    batch_windows=args.batch_windows,
                    use_fp16=False if args.no_fp16 else None,
                )
                for it, vec, backbone in zip(batch_items, vecs, backbones):
                    row = {"id": it["id"], "meta": {k: v for k, v in it.items()
                            if k not in ("audio", "sr")},
                           "embedding": vec.tolist(),
                           "merit_backbone": backbone.tolist(), "status": "ok"}
                    out.write(json.dumps(row) + "\n")
                    done.add(it["id"])
                    exclude_ids.add(it["id"])
                out.flush()
                batch_items.clear(); batch_waves.clear()
            if batch_items:
                vecs, backbones = embed_tracks_batched_dual(
                    model, processor, batch_waves, device,
                    batch_windows=args.batch_windows,
                    use_fp16=False if args.no_fp16 else None,
                )
                for it, vec, backbone in zip(batch_items, vecs, backbones):
                    row = {"id": it["id"], "meta": {k: v for k, v in it.items()
                            if k not in ("audio", "sr")},
                           "embedding": vec.tolist(),
                           "merit_backbone": backbone.tolist(), "status": "ok"}
                    out.write(json.dumps(row) + "\n")
                    done.add(it["id"])
                    exclude_ids.add(it["id"])
                out.flush()
            print(f"stratum {lo}-{hi}: checkpoint now has {len(done)} tracks", flush=True)

    report = append_checkpoint_in_batches(
        args.bundle, checkpoint, args.out_dir,
        batch_size=args.append_batch_size,
        dedupe_threshold=args.dedupe_threshold, knn_k=args.knn_k,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
