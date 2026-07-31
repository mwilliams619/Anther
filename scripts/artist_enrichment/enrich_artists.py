#!/usr/bin/env python
"""
Thin entrypoint for the artist enrichment job — the path named in
ARTIST_ENRICHMENT_PLAN.md §5. All logic lives in the importable, tested package
``anther_ml.artist_enrichment``; this file just forwards to its CLI so both

    python scripts/artist_enrichment/enrich_artists.py --sample 100 --dry-run
    python -m anther_ml.artist_enrichment --sample 100 --dry-run

do the same thing. See ``python -m anther_ml.artist_enrichment --help``.
"""

from anther_ml.artist_enrichment.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
