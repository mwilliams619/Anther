"""
Turn a built corpus bundle into a *publishable* one.

A bundle as built carries things that must not — or need not — be redistributed:

  * **per-track playlist membership** (``metadata[i]["playlists"]``). This is
    the substance of the Million Playlist Dataset: the playlist→track
    association, not the track facts. It is also ~82% of ``index.json`` on the
    100k bundle (79 MB → 14 MB). Track id / name / artist are Spotify catalog
    facts and are kept, which is what preserves free playback (a
    ``spotify:``-prefixed node id short-circuits ``get_spotify_track_id``).
  * **``merit_backbone.npy``** — the 5120-d build intermediate. Only
    ``merit_index.py`` (rebuilding the sidecar) and ``extend.py`` (appending
    tracks) read it; nothing at query time does. 2.0 GB on the 100k bundle.
  * **``leiden.pkl``** — pickled sklearn/UMAP objects, which pin the reader to
    the Python/numba/sklearn versions that wrote them and make a downloaded
    bundle an arbitrary-code-execution vector. Replaced with the portable
    ``leiden.npz`` (``cluster.save_leiden_portable``).

Everything else is copied verbatim, so sidecars added later (calibration,
tag artifacts, factor vectors) travel without this module needing to know
about them. Files are copied unless they match ``EXCLUDE`` — the safe default
direction, since a new artifact that nobody publishes is a worse failure than
one that ships redundantly.

What a published bundle loses, all of it already guarded at the call sites:

  * ``corpus.playlists()`` / ``playlist_member_indices()`` / the in-corpus
    playlist index (``ui/atlas.py:_build_playlist_index``) go empty. The
    full-MPD playlist search is separate and already gated on ``ANTHER_MPD_DB``.
  * ``place()`` returns ``coords_2d=None`` (no ``reducer_2d``). The web UI never
    reads it — the force graph computes its own layout — so this shows up only
    in the ``corpus place`` CLI printout and as zero-filled coords for rows
    appended by ``extend_corpus``.

Cluster labels and micro-genre tags are *unaffected*: they were seeded from
playlist names at build time, but what ships is their output
(``cluster_profiles.json``, ``tag_probe.pkl``, ``track_tags.json``).

    python -m anther_ml.corpus publish models/corpus_mpd_100k_merit \\
        --out models/publish/corpus_mpd_100k_merit --dry-run
"""

import fnmatch
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..cluster import load_leiden, save_leiden_portable
from .bundle import LEIDEN_PICKLE, LEIDEN_PORTABLE, leiden_path

PUBLISH_FORMAT_VERSION = 1

#: Metadata keys stripped by ``--strip-playlists`` (the default).
STRIPPED_METADATA_KEYS = ("playlists",)

#: Build intermediates and scratch that never belong in a published bundle.
#: ``leiden.pkl`` is handled separately (converted, not merely dropped).
EXCLUDE = (
    "merit_backbone.npy",
    "*.log",
    "*.png",
    "embed_checkpoint*",
    "checkpoint*",
    ".DS_Store",
)

#: The SongIndex JSON that owns the metadata rows.
PRIMARY_INDEX_JSON = "index.json"

#: SongIndex JSONs carrying an exact duplicate of ``index.json``'s metadata.
#: Published bundles replace the block with a ``metadata_ref`` pointer, which
#: ``load_merit_aggregate_index`` resolves (or ``ReferenceCorpus.merit_index``
#: satisfies directly from the already-loaded main index).
DUPLICATE_INDEX_JSONS = ("index_merit_agg.json",)

INDEX_JSONS = (PRIMARY_INDEX_JSON, *DUPLICATE_INDEX_JSONS)


def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE)


def strip_metadata_rows(
    rows: list[dict], keys: tuple[str, ...] = STRIPPED_METADATA_KEYS
) -> tuple[list[dict], int]:
    """Drop ``keys`` from every row. Returns (rows, n_rows_actually_changed)."""
    out, touched = [], 0
    for row in rows:
        present = [k for k in keys if k in row]
        if present:
            touched += 1
            row = {k: v for k, v in row.items() if k not in keys}
        out.append(row)
    return out, touched


def _rewrite_index_json(
    src: Path, dst: Path, keys: tuple[str, ...], *, dedupe_to: str | None = None
) -> dict:
    """Rewrite one SongIndex ``.json``.

    ``dedupe_to`` replaces the whole metadata block with a pointer to the index
    that owns it. Otherwise ``keys`` are stripped from every row.
    """
    payload = _rewritten_payload(src, keys, dedupe_to=dedupe_to)
    touched = payload.pop("_rows_stripped", 0) if isinstance(payload, dict) else 0
    with open(dst, "w") as f:
        json.dump(payload, f)
    return {
        "rows_stripped": touched,
        "bytes_before": src.stat().st_size,
        "bytes_after": dst.stat().st_size,
        "deduped_to": dedupe_to,
    }


def _rewritten_payload(src: Path, keys: tuple[str, ...], *, dedupe_to: str | None):
    """The published form of a SongIndex JSON payload (shared by the dry run,
    which sizes it without writing)."""
    with open(src) as f:
        payload = json.load(f)

    # A legacy v1 index is a bare list of rows and has no place to put a
    # pointer, so it can only ever be stripped in place.
    if isinstance(payload, list):
        rows, touched = strip_metadata_rows(payload, keys)
        return rows

    if dedupe_to is not None:
        n = len(payload.get("metadata") or [])
        payload.pop("metadata", None)
        payload["metadata_ref"] = dedupe_to
        payload["_rows_stripped"] = n
        return payload

    payload["metadata"], touched = strip_metadata_rows(payload["metadata"], keys)
    payload["_rows_stripped"] = touched
    return payload


def _payload_bytes(payload) -> int:
    """Serialized size, excluding the internal bookkeeping key."""
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k != "_rows_stripped"}
    return len(json.dumps(payload).encode())


def _convert_leiden(src_dir: Path, dst_dir: Path) -> dict:
    """Write ``leiden.npz`` from whichever Leiden file the source bundle has."""
    src = leiden_path(src_dir)
    if not src.exists():
        raise FileNotFoundError(f"no Leiden file in {src_dir} ({LEIDEN_PICKLE})")
    leiden = load_leiden(src)
    dst = dst_dir / LEIDEN_PORTABLE
    meta = save_leiden_portable(dst, leiden)
    return {
        "source": src.name,
        "bytes_before": src.stat().st_size,
        "bytes_after": dst.stat().st_size,
        "reducer_2d_dropped": meta["reducer_2d_dropped"],
    }


def publish_bundle(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    strip_playlists: bool = True,
    drop_backbone: bool = True,
    portable_leiden: bool = True,
    dedupe_metadata: bool = True,
    dry_run: bool = False,
) -> dict:
    """Copy ``src_dir`` to ``out_dir`` as a publishable bundle.

    Returns a report dict (also written into the published ``manifest.json``
    under ``"publish"``, so a downloaded bundle documents its own provenance).
    ``dry_run`` computes the report and writes nothing.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if not (src_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no corpus bundle at {src_dir} (missing manifest.json)")
    if out_dir.resolve() == src_dir.resolve():
        raise ValueError("publish would overwrite the source bundle; pick a new --out")
    if not dry_run and out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise FileExistsError(
                f"publish destination {out_dir} must not exist or must be empty"
            )

    keys = STRIPPED_METADATA_KEYS if strip_playlists else ()
    exclude = EXCLUDE if drop_backbone else tuple(
        p for p in EXCLUDE if p != "merit_backbone.npy"
    )

    def _dedupe_target(name: str) -> str | None:
        if not dedupe_metadata or name not in DUPLICATE_INDEX_JSONS:
            return None
        return Path(PRIMARY_INDEX_JSON).stem

    copied, skipped, rewritten = [], [], {}
    bytes_before = bytes_after = 0

    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        bytes_before += size

        if any(fnmatch.fnmatch(path.name, pat) for pat in exclude):
            skipped.append(path.name)
            continue
        # Converted below, not copied.
        if portable_leiden and path.name == LEIDEN_PICKLE:
            continue
        if path.name in INDEX_JSONS and (keys or _dedupe_target(path.name)):
            rewritten[path.name] = {"bytes_before": size}
            continue
        copied.append(path.name)
        bytes_after += size

    report = {
        "publish_format_version": PUBLISH_FORMAT_VERSION,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(src_dir),
        "stripped_metadata_keys": list(keys),
        "excluded_files": skipped,
        "portable_leiden": portable_leiden,
        "dedupe_metadata": dedupe_metadata,
    }

    if dry_run:
        # Size the rewrites without writing them: build the same payload the
        # real run would and measure its serialization.
        for name in list(rewritten):
            payload = _rewritten_payload(
                src_dir / name, keys, dedupe_to=_dedupe_target(name)
            )
            touched = (
                payload.get("_rows_stripped", 0) if isinstance(payload, dict) else 0
            )
            after = _payload_bytes(payload)
            rewritten[name].update({
                "rows_stripped": touched,
                "bytes_after": after,
                "deduped_to": _dedupe_target(name),
            })
            bytes_after += after
        if portable_leiden:
            # Sizing the npz means actually converting it (minutes on a 100k
            # bundle), so a dry run reports it as unsized rather than dropping
            # it from the total and under-reporting the published size.
            src = leiden_path(src_dir)
            report["leiden"] = {
                "source": src.name,
                "bytes_before": src.stat().st_size if src.exists() else 0,
                "bytes_after": None,
                "unsized_in_dry_run": True,
            }
        report["rewritten_indices"] = rewritten
        report["bytes_before"] = bytes_before
        report["bytes_after"] = bytes_after
        report["bytes_after_excludes_leiden"] = portable_leiden
        report["dry_run"] = True
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in copied:
        shutil.copy2(src_dir / name, out_dir / name)
    for name in rewritten:
        stats = _rewrite_index_json(
            src_dir / name, out_dir / name, keys, dedupe_to=_dedupe_target(name)
        )
        rewritten[name].update(stats)
        bytes_after += stats["bytes_after"]

    if portable_leiden:
        report["leiden"] = _convert_leiden(src_dir, out_dir)
        bytes_after += report["leiden"]["bytes_after"]

    report["rewritten_indices"] = rewritten
    report["bytes_before"] = bytes_before
    report["bytes_after"] = bytes_after

    # Stamp provenance into the published manifest. Deliberately keeps the
    # source's own `name`/`build_params`/`embedding_config` untouched — a
    # published bundle should document what it is, not launder it.
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["publish"] = report
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return report


def _apply(leiden: dict, X: np.ndarray) -> np.ndarray:
    """The clustering-space transform a query goes through (assign_cluster_knn)."""
    if leiden["scaler"] is not None:
        X = leiden["scaler"].transform(X)
    if leiden["pca"] is not None:
        X = leiden["pca"].transform(X)
    return np.asarray(X)


def verify_published(
    out_dir: str | Path, src_dir: str | Path | None = None, n_probe: int = 256
) -> dict:
    """Load a published bundle and assert it still works end to end.

    The load-bearing check is ``transform_ok``: the frozen scaler/PCA must move
    a query to the *same place* the source bundle's pickled estimators would,
    because cluster assignment is a kNN vote in that space and a drifted
    transform would silently misplace every query. When ``src_dir`` is given
    this is checked directly against the source's own estimators; otherwise it
    falls back to agreement with the bundle's stored ``clustering_space``.

    That fallback is a *cosine* check, not an absolute one. ``clustering_space``
    is sklearn's float32 ``fit_transform`` output, and re-deriving it from
    ``embeddings.npy`` accumulates ~5e-3 relative error over a 1024-dim float32
    matmul — a property of the original bundle, not of publishing. Direction is
    what the cosine-metric kNN actually consumes.
    """
    from .bundle import ReferenceCorpus

    out_dir = Path(out_dir)
    corpus = ReferenceCorpus.load(out_dir)  # runs verify() internally

    checks = {
        "n_tracks": corpus.n_tracks,
        "pickle_free": not (out_dir / LEIDEN_PICKLE).exists(),
        "backbone_dropped": not (out_dir / "merit_backbone.npy").exists(),
        "playlists_stripped": all("playlists" not in row for row in corpus.metadata),
        "merit_index": corpus.merit_index is not None,
    }

    probe = np.asarray(corpus.embeddings[:n_probe])
    published = _apply(corpus.leiden, probe)

    if src_dir is not None:
        source = _apply(load_leiden(leiden_path(Path(src_dir))), probe)
        checks["transform_checked_against"] = "source bundle"
        checks["transform_max_abs_err"] = float(np.abs(published - source).max())
        checks["transform_ok"] = checks["transform_max_abs_err"] < 1e-4
    else:
        stored = np.asarray(corpus.leiden["clustering_space"][:n_probe])
        cos = np.sum(published * stored, axis=1) / (
            np.linalg.norm(published, axis=1) * np.linalg.norm(stored, axis=1)
        )
        checks["transform_checked_against"] = "stored clustering_space"
        checks["transform_min_cosine"] = float(cos.min())
        checks["transform_ok"] = checks["transform_min_cosine"] > 0.999
    return checks
