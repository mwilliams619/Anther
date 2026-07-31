#!/usr/bin/env python
"""
Cache an artist's whole iTunes discography into the UI embed cache.

The bulk path behind "add everything by X" for artists Deezer's catalog is
missing, without clicking through the UI one track at a time. Each track is
fetched as a 30s AAC preview, transcoded (PyAV), embedded through MERT, and
written to the session's embed_cache.sqlite — so the UI then places them
instantly, and a re-run is a cheap no-op for anything already cached.

Only the embed cache is touched: this does not add nodes to the force graph
(use the UI's discography import, or /api/itunes/artist/place, for that).

Usage
-----
    python scripts/cache_itunes_artist.py "Rob Knack"
    python scripts/cache_itunes_artist.py "Matt Brade" --include-features
    python scripts/cache_itunes_artist.py --artist-id 1847129711 --limit 20
    python scripts/cache_itunes_artist.py "Rob Knack" --dry-run

Run from the repo root with the corpus-compatible env (`.venv`), since
embedding loads the frozen bundle's transform.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))

from anther_ml import itunes  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("artist", nargs="?", help="artist name (exact match preferred)")
    p.add_argument("--artist-id", help="iTunes artistId, skips the name lookup")
    p.add_argument("--include-features", action="store_true",
                   help="also cache tracks where they are a featured credit")
    p.add_argument("--limit", type=int, help="cache at most N tracks")
    p.add_argument("--session", default="default",
                   help="UI session id whose embed cache to write (default: default)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be cached; no downloads, no embedding")
    args = p.parse_args()

    if not args.artist and not args.artist_id:
        p.error("give an artist name or --artist-id")

    artist_id = args.artist_id
    if not artist_id:
        found = itunes.resolve_artist(args.artist)
        if not found:
            print(f"no exact iTunes artist match for {args.artist!r}")
            print("hint: pass --artist-id if the name differs on Apple Music")
            return 1
        artist_id = found["artist_id"]
        print(f"resolved {found['name']!r} -> artistId {artist_id} ({found['genre']})")

    tracks = itunes.artist_tracks(artist_id, include_features=args.include_features)
    if not tracks:
        print(f"no iTunes tracks for artistId {artist_id}")
        return 1
    if args.limit:
        tracks = tracks[:args.limit]
    print(f"{len(tracks)} tracks after dedup"
          f"{' (including features)' if args.include_features else ''}")

    if args.dry_run:
        for t in tracks:
            print(f"  {t['id']:22} {t['artist']} — {t['title']}")
        return 0

    # Imported late: this pulls in torch + the frozen corpus bundle, which is
    # slow and pointless for --dry-run.
    import atlas

    atlas.set_session(args.session)
    atlas.load()

    cached = embedded = failed = 0
    t_start = time.time()
    for i, t in enumerate(tracks, 1):
        label = f"{t['artist']} — {t['title']}"
        if atlas.cached_vec(t["id"]) is not None:
            cached += 1
            print(f"[{i}/{len(tracks)}] cached already: {label}")
            continue
        try:
            _, _, method = atlas.resolve_and_embed({
                "id": t["id"], "name": t["title"], "artist": t["artist"],
                "preview_url": t["preview_url"],
            })
            embedded += 1
            print(f"[{i}/{len(tracks)}] embedded ({method}): {label}")
        except atlas.PlacementSkip as exc:
            failed += 1
            print(f"[{i}/{len(tracks)}] SKIP ({exc.reason}): {label}")
        except Exception as exc:  # noqa: BLE001 — one bad track must not end the run
            failed += 1
            print(f"[{i}/{len(tracks)}] ERROR ({exc}): {label}")

    mins = (time.time() - t_start) / 60
    print(f"\ndone in {mins:.1f} min — {embedded} embedded, {cached} already cached, "
          f"{failed} failed")
    print(f"cache: {atlas.get_session().embed_cache_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
