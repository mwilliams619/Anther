"""
Artist enrichment: a durable, provenance-carrying profile layer over the
frozen corpus artists (ARTIST_ENRICHMENT_PLAN.md).

Adds name / image / following / genres / origin / labels to each corpus artist,
sourced from Deezer + MusicBrainz, stored separately from the immutable
clustering artifacts and keyed by normalized artist name so it survives a corpus
rebuild.

    python -m anther_ml.artist_enrichment --sample 100 --dry-run
    python -m anther_ml.artist_enrichment --all

See ``store.ArtistProfileStore`` for persistence and ``enrich.run`` for the job.
The atlas read path joins profiles by ``anther_ml.spotify_deezer._norm(name)``.
"""

from .enrich import EnrichmentReport, enrich_one, load_corpus_artists, run
from .resolve import Resolution
from .schema import ArtistProfile
from .store import ArtistProfileStore, open_readonly

# NB: the ``resolve`` *submodule* holds the ``resolve()`` function
# (``anther_ml.artist_enrichment.resolve.resolve``). It is intentionally not
# re-exported here so the submodule name stays importable without shadowing.

__all__ = [
    "ArtistProfile",
    "ArtistProfileStore",
    "EnrichmentReport",
    "Resolution",
    "enrich_one",
    "load_corpus_artists",
    "open_readonly",
    "run",
]
