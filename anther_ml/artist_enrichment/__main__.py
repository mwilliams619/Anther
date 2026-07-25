"""
CLI for the artist enrichment job (ARTIST_ENRICHMENT_PLAN.md §5).

    python -m anther_ml.artist_enrichment --sample 100 --dry-run   # match report
    python -m anther_ml.artist_enrichment --sample 100             # write sample
    python -m anther_ml.artist_enrichment --all                    # full corpus
    python -m anther_ml.artist_enrichment --artist-id corpus:42
    python -m anther_ml.artist_enrichment --missing following      # refresh a field
    python -m anther_ml.artist_enrichment --review                 # dump review queue
"""

from __future__ import annotations

import argparse
import json
import sys

from .enrich import (DEFAULT_ARTIST_META, load_corpus_artists, run)
from .schema import ENRICHABLE_FIELDS
from .store import ArtistProfileStore, DEFAULT_DB_PATH


def _resolve_artist_key(artist_id: str, meta_path: str) -> str:
    """Accept either a normalized key or a ``corpus:{index}`` id from the UI."""
    if artist_id.startswith("corpus:"):
        artists = load_corpus_artists(meta_path)
        try:
            idx = int(artist_id.split(":", 1)[1])
        except ValueError:
            raise SystemExit(f"bad corpus id: {artist_id}")
        for a in artists:
            if a.index == idx:
                return a.artist_key
        raise SystemExit(f"corpus index not found (or deduped): {artist_id}")
    return artist_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anther_ml.artist_enrichment",
                                description="Enrich corpus artists with profile data.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="process every corpus artist")
    mode.add_argument("--artist-id", help="one artist: normalized key or corpus:{index}")
    mode.add_argument("--missing", choices=ENRICHABLE_FIELDS,
                      help="re-enrich only artists whose given field is empty")
    mode.add_argument("--review", action="store_true", help="print the review queue and exit")

    p.add_argument("--sample", type=int, help="cap work to the first N artists")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve + report without writing profiles")
    p.add_argument("--refresh", action="store_true",
                   help="reprocess artists already marked complete")
    p.add_argument("--no-labels", action="store_true",
                   help="skip the extra MusicBrainz release-label fetch")
    p.add_argument("--meta", default=DEFAULT_ARTIST_META, help="artist_meta.json path")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="profiles SQLite path")
    p.add_argument("--user-agent", help="override the MusicBrainz User-Agent")
    p.add_argument("--mb-interval", type=float, default=1.05,
                   help="min seconds between MusicBrainz calls (default 1.05)")
    p.add_argument("--quiet", action="store_true", help="suppress per-artist log lines")
    args = p.parse_args(argv)

    if args.review:
        store = ArtistProfileStore(args.db)
        rows = store.review_rows()
        store.close()
        print(json.dumps(rows, indent=2))
        print(f"\n{len(rows)} artist(s) awaiting review", file=sys.stderr)
        return 0

    only_key = None
    if args.artist_id:
        only_key = _resolve_artist_key(args.artist_id, args.meta)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    report = run(meta_path=args.meta, db_path=args.db, only_key=only_key,
                 missing_field=args.missing, sample=args.sample,
                 refresh=args.refresh, dry_run=args.dry_run,
                 with_labels=not args.no_labels, user_agent=args.user_agent,
                 mb_min_interval=args.mb_interval, progress=log)

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
