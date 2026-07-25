"""External source adapters for artist enrichment.

Each adapter is a small, independently-testable client that turns one source's
raw API into normalized *candidate* dicts. Adapters never decide identity — that
is the resolver's job (see ``anther_ml.artist_enrichment.resolve``).

v1 sources: Deezer (image, fan count) and MusicBrainz (origin, genres, labels).
Spotify is intentionally left as a seam only (plan §2 — 2026 dev-mode limits).
"""

from .deezer import DeezerArtistClient
from .musicbrainz import MusicBrainzClient

__all__ = ["DeezerArtistClient", "MusicBrainzClient"]
