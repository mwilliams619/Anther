"""
Batch enrichment orchestrator (ARTIST_ENRICHMENT_PLAN.md §5).

Ties the source adapters, resolver, normalizers and store together into an
idempotent, resumable crawl. The store is the checkpoint: a profile already
marked ``complete`` is skipped on resume unless a refresh/`--missing` mode asks
for it. Every source payload is cached verbatim so a re-run costs no network for
work already done.

Network access is confined to this module (via the injected clients); the
resolver and normalizers stay pure and unit-tested offline.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from anther_ml.spotify_deezer import _norm

from .normalize import (clean_image_url, genres_from_musicbrainz,
                        normalize_labels, structure_hometown)
from .resolve import Resolution, resolve
from .schema import (ArtistProfile, ENRICHABLE_FIELDS, STATUS_COMPLETE,
                     STATUS_PARTIAL, STATUS_REVIEW, STATUS_UNMATCHED)
from .sources import DeezerArtistClient, MusicBrainzClient
from .sources.musicbrainz import DEFAULT_USER_AGENT
from .store import ArtistProfileStore, DEFAULT_DB_PATH

DEFAULT_ARTIST_META = "data/artist_clustering/artist_meta.json"


@dataclass
class CorpusArtist:
    name: str
    sample_track: str
    artist_key: str
    index: int


def load_corpus_artists(meta_path: str | Path = DEFAULT_ARTIST_META) -> list[CorpusArtist]:
    """Read artist_meta.json into resolution inputs, keyed by normalized name.

    Duplicate normalized names collapse to the first occurrence (the profile
    layer is per distinct artist identity, and the atlas join is by normalized
    name too)."""
    data = json.loads(Path(meta_path).read_text())
    out: list[CorpusArtist] = []
    seen: set[str] = set()
    for i, entry in enumerate(data):
        name = entry.get("artist", "")
        key = _norm(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(CorpusArtist(name=name, sample_track=entry.get("sample_track", ""),
                                artist_key=key, index=i))
    return out


@dataclass
class EnrichmentReport:
    total: int = 0
    processed: int = 0
    skipped_complete: int = 0
    matched_complete: int = 0
    matched_partial: int = 0
    review: int = 0
    unmatched: int = 0
    errors: int = 0
    field_coverage: dict[str, int] = field(default_factory=lambda: {f: 0 for f in ENRICHABLE_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        if self.processed:
            d["field_coverage_pct"] = {
                f: round(100 * n / self.processed, 1) for f, n in self.field_coverage.items()}
        return d


def _today() -> str:
    return _dt.date.today().isoformat()


def build_profile(artist: CorpusArtist, res: Resolution, *,
                  mb_lookup: dict[str, Any] | None,
                  mb_labels: list[dict[str, Any]] | None,
                  now: str) -> ArtistProfile:
    """Assemble the canonical profile from a resolution + fetched sub-docs."""
    prof = ArtistProfile(artist_key=artist.artist_key, name=artist.name,
                         match_confidence=res.confidence, status=res.status,
                         updated_at=now)
    if res.mb is not None:
        prof.identities["musicbrainz_id"] = res.mb.get("mbid")
        prof.hometown = structure_hometown(res.mb)
        if mb_lookup is not None:
            prof.genres = genres_from_musicbrainz(mb_lookup)
        if mb_labels:
            prof.labels = normalize_labels(mb_labels)
    if res.deezer is not None:
        prof.identities["deezer_id"] = res.deezer.get("deezer_id")
        img = clean_image_url(res.deezer.get("image_url"))
        if img:
            prof.profile_image = {"url": img, "source": "deezer", "updated_at": now}
        fans = res.deezer.get("nb_fan")
        if fans is not None:
            prof.following = {"count": int(fans), "source": "deezer", "updated_at": now}
    return prof


def _cached_search(store: ArtistProfileStore | None, client, kind: str,
                   query_key: str, fetch: Callable[[], Any]) -> Any:
    """Return a cached search/lookup payload or fetch + cache it."""
    if store is not None:
        cached = store.get_raw(client.source, query_key, kind)
        if cached is not None:
            return cached
    payload = fetch()
    if store is not None:
        store.cache_raw(client.source, query_key, kind, payload, _today())
    return payload


def enrich_one(artist: CorpusArtist, *, mb: MusicBrainzClient,
               dz: DeezerArtistClient, store: ArtistProfileStore | None,
               fetch_fields: bool = True, with_labels: bool = True,
               use_cache: bool = True, now: str | None = None) -> tuple[ArtistProfile, Resolution]:
    """Resolve + enrich a single artist. Pure of persistence when ``store`` is
    ``None`` (dry-run) — it still reads/writes the raw cache only if a store is
    passed."""
    now = now or _today()
    cache = store if use_cache else None
    mb_cands = _cached_search(cache, mb, "search", artist.artist_key,
                              lambda: mb.search_artist(artist.name))
    dz_cands = _cached_search(cache, dz, "search", artist.artist_key,
                              lambda: dz.search_artist(artist.name))
    res = resolve(artist.name, mb_cands, dz_cands)

    mb_lookup = mb_labels = None
    if fetch_fields and res.mb is not None:
        mbid = res.mb.get("mbid")
        if mbid:
            mb_lookup = _cached_search(cache, mb, "lookup", str(mbid),
                                       lambda: mb.lookup_artist(mbid))
            if with_labels:
                mb_labels = _cached_search(cache, mb, "labels", str(mbid),
                                           lambda: mb.artist_release_labels(mbid))
    prof = build_profile(artist, res, mb_lookup=mb_lookup, mb_labels=mb_labels, now=now)
    return prof, res


def run(*, meta_path: str | Path = DEFAULT_ARTIST_META,
        db_path: str | Path = DEFAULT_DB_PATH,
        only_key: str | None = None,
        missing_field: str | None = None,
        sample: int | None = None,
        refresh: bool = False,
        dry_run: bool = False,
        with_labels: bool = True,
        user_agent: str | None = None,
        mb_min_interval: float = 1.05,
        progress: Callable[[str], None] | None = None) -> EnrichmentReport:
    """Run the batch job. Returns a coverage/outcome report (plan §5/§8).

    Modes: ``only_key`` (one artist), ``missing_field`` (re-enrich artists whose
    given field is empty), ``sample`` (first N — used for the audited sample
    crawl). ``refresh`` re-processes even ``complete`` artists.
    """
    log = progress or (lambda _m: None)
    artists = load_corpus_artists(meta_path)
    report = EnrichmentReport(total=len(artists))

    store = None if dry_run else ArtistProfileStore(db_path)
    # A read-only store for cache hits even in dry-run, if one exists on disk.
    cache_store = store
    if dry_run and Path(db_path).is_file():
        cache_store = ArtistProfileStore(db_path)

    mb = MusicBrainzClient(user_agent=user_agent or DEFAULT_USER_AGENT,
                           min_interval=mb_min_interval)
    dz = DeezerArtistClient()

    selected = _select(artists, store, only_key, missing_field, refresh)
    if sample is not None:
        selected = selected[:max(0, sample)]

    try:
        for i, (artist, needs) in enumerate(selected, 1):
            if not needs:
                report.skipped_complete += 1
                continue
            try:
                prof, res = enrich_one(artist, mb=mb, dz=dz, store=cache_store,
                                       with_labels=with_labels)
            except Exception as exc:  # keep the crawl resumable on one bad artist
                report.errors += 1
                log(f"[{i}/{len(selected)}] ERROR {artist.name!r}: {exc}")
                continue

            report.processed += 1
            _tally(report, prof, res)
            if not dry_run:
                store.upsert(prof)
                if res.status == STATUS_REVIEW:
                    store.queue_review(artist.artist_key, artist.name, res.reason,
                                       res.review_candidates, _today())
                else:
                    # A previously-ambiguous artist that now resolves cleanly must
                    # not linger in the review queue.
                    store.dequeue_review(artist.artist_key)
            log(f"[{i}/{len(selected)}] {artist.name!r} → {prof.status} "
                f"(conf={prof.match_confidence}) "
                f"img={'Y' if prof.profile_image else '-'} "
                f"fans={(prof.following or {}).get('count', '-')} "
                f"genres={len(prof.genres)} origin={(prof.hometown or {}).get('name', '-')} "
                f"labels={len(prof.labels)}")
    finally:
        if store is not None:
            store.close()
        if cache_store is not None and cache_store is not store:
            cache_store.close()
    return report


def _select(artists: list[CorpusArtist], store: ArtistProfileStore | None,
            only_key: str | None, missing_field: str | None,
            refresh: bool) -> list[tuple[CorpusArtist, bool]]:
    """Pick the work set, pairing each artist with whether it needs processing."""
    out = []
    for a in artists:
        if only_key is not None and a.artist_key != only_key:
            continue
        needs = True
        if store is not None and not refresh:
            existing = store.get(a.artist_key)
            if existing is not None:
                if missing_field is not None:
                    needs = missing_field in existing.missing_fields()
                elif existing.status == STATUS_COMPLETE:
                    needs = False
        out.append((a, needs))
    return out


def _tally(report: EnrichmentReport, prof: ArtistProfile, res: Resolution) -> None:
    if prof.status == STATUS_COMPLETE:
        report.matched_complete += 1
    elif prof.status == STATUS_PARTIAL:
        report.matched_partial += 1
    elif prof.status == STATUS_REVIEW:
        report.review += 1
    else:
        report.unmatched += 1
    for f in ENRICHABLE_FIELDS:
        val = getattr(prof, f)
        if val not in (None, [], {}, ""):
            report.field_coverage[f] += 1
