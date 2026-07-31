"""
extend_corpus() — additive extension of a frozen ReferenceCorpus bundle
(REFERENCE_CORPUS_DESIGN.md "Maintenance": additive extension is safe and
keeps cluster IDs stable; refitting re-labels clusters and must be a
deliberate, versioned, occasional act — never done here).

What this does, per new track:
  1. dedupe the new batch against the existing corpus (and against itself)
     in raw-embedding cosine space, at the same threshold build.py uses
  2. put the raw MERT-1024 vector through the corpus's FROZEN SongIndex
     transform (stored mean_/scale_, never refit) and append the row
  3. assign a Leiden cluster via the existing k-NN vote against the FROZEN
     clustering_space/scaler/pca (anther_ml.cluster.assign_cluster_knn) —
     never re-run Leiden itself
  4. project through the frozen 2D reducer for a display coordinate
  5. append to embeddings.npy / index.npy+json / leiden.pkl / embedding_2d.npy
  6. optionally extend the MERIT-aggregate sidecar the same way, if the
     bundle carries one and callers supply merit backbones for new tracks
  7. recompute cluster_profiles.json (sizes/exemplars only — display
     bookkeeping, NOT a re-fit) and bump the manifest's extension log

What this explicitly does NOT do: refit StandardScaler/PCA/Leiden/UMAP,
change any existing track's cluster id, or touch dedupe_threshold for
tracks already in the corpus.

Usage:
    from anther_ml.corpus.extend import extend_corpus
    report = extend_corpus(
        "models/corpus_mpd_100k_merit",
        new_items=[{"meta": {...}, "raw_vector": np.ndarray(1024,),
                    "merit_backbone": np.ndarray(5120,) | None}, ...],
        out_dir="models/corpus_mpd_100k_merit",  # in-place, or a new dir
    )
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..cluster import assign_cluster_knn
from .bundle import ReferenceCorpus, utc_now_iso
from .build import build_cluster_profiles, dedupe_near_identical


def _standardize_new(mean_, scale_, X: np.ndarray) -> np.ndarray:
    if mean_ is None:
        return X
    return (X - mean_) / scale_


def _l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def _append_npy_streaming(
    old_path: Path, out_path: Path, new_rows: np.ndarray, chunk_rows: int = 8192
) -> None:
    """Append rows to an npy file without holding old+new copies in RAM."""
    from numpy.lib.format import open_memmap

    old = np.load(old_path, mmap_mode="r")
    new_rows = np.asarray(new_rows, dtype=old.dtype)
    if old.ndim != new_rows.ndim or old.shape[1:] != new_rows.shape[1:]:
        raise ValueError(
            f"cannot append {new_rows.shape} to {old.shape} in {old_path}"
        )
    tmp = out_path.with_name(out_path.name + ".tmp.npy")
    result = open_memmap(
        tmp, mode="w+", dtype=old.dtype,
        shape=(old.shape[0] + new_rows.shape[0], *old.shape[1:]),
    )
    # NOTE: clamp the destination slice to old.shape[0] -- `result` is longer
    # than `old`, so an unclamped final chunk would run into the not-yet-written
    # new-row region and fail to broadcast whenever old.shape[0] % chunk_rows.
    for start in range(0, old.shape[0], chunk_rows):
        stop = min(start + chunk_rows, old.shape[0])
        result[start:stop] = old[start:stop]
    result[old.shape[0] :] = new_rows
    result.flush()
    del result, old
    os.replace(tmp, out_path)


def extend_corpus(
    bundle_dir: str | Path,
    new_items: list[dict],
    out_dir: str | Path | None = None,
    dedupe_threshold: float = 0.98,
    knn_k: int = 15,
    heads_dir: str = "models/merit_heads",
) -> dict:
    """
    Additively append ``new_items`` to the frozen corpus at ``bundle_dir``.

    new_items: list of ``{"meta": dict, "raw_vector": (1024,) array,
    "merit_backbone": (5120,) array | None}``. ``meta`` follows the same
    contract as ``anther_ml.corpus.sources`` items (id/name/artist/source/
    genre/playlists).

    Writes the extended bundle to ``out_dir`` (defaults to ``bundle_dir`` —
    in-place). Returns a diagnostic dict: counts before/after, how many new
    items were dropped as near-duplicates, and per-cluster size deltas.
    """
    bundle_dir = Path(bundle_dir)
    out_dir = Path(out_dir) if out_dir is not None else bundle_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = ReferenceCorpus.load(bundle_dir, verify=True)

    n_before = corpus.n_tracks
    raw_new = np.stack([np.asarray(it["raw_vector"], dtype=np.float32) for it in new_items])
    assert raw_new.shape[1] == corpus.embeddings.shape[1], (
        f"new_items raw_vector dim {raw_new.shape[1]} != corpus dim "
        f"{corpus.embeddings.shape[1]} — were these embedded with the "
        f"corpus's frozen recipe (corpus.embedding_config)?"
    )

    # --- 1. dedupe new batch against existing corpus + itself ---------------
    combined = np.concatenate([corpus.embeddings, raw_new], axis=0)
    keep_mask = dedupe_near_identical(combined, threshold=dedupe_threshold, method="auto")
    keep_new_mask = keep_mask[n_before:]
    n_dropped = int((~keep_new_mask).sum())
    del combined, keep_mask

    kept_items = [it for it, keep in zip(new_items, keep_new_mask) if keep]
    kept_raw = raw_new[keep_new_mask]
    if not kept_items:
        return {
            "n_before": n_before, "n_after": n_before, "n_input": len(new_items),
            "n_dropped_duplicate": n_dropped, "n_added": 0,
        }

    # --- 2. append to SongIndex (frozen transform, never refit) --------------
    index = corpus.index
    new_meta = [it["meta"] for it in kept_items]
    new_transformed = _l2norm(_standardize_new(index.mean_, index.scale_, kept_raw))
    index.embeddings = np.concatenate([index.embeddings, new_transformed], axis=0).astype(np.float32)
    index.metadata = index.metadata + new_meta

    # --- 3. assign Leiden clusters via frozen k-NN vote ----------------------
    leiden = corpus.leiden
    new_labels, new_clustering_rows = [], []
    for vec in kept_raw:
        cid, _conf = assign_cluster_knn(
            vec, leiden["clustering_space"], leiden["labels"],
            scaler=leiden["scaler"], pca=leiden["pca"], k=knn_k,
        )
        new_labels.append(cid)
        X = vec.reshape(1, -1)
        if leiden["scaler"] is not None:
            X = leiden["scaler"].transform(X)
        if leiden["pca"] is not None:
            X = leiden["pca"].transform(X)
        new_clustering_rows.append(X[0])
    leiden["labels"] = np.concatenate([np.asarray(leiden["labels"]), np.asarray(new_labels)])
    leiden["clustering_space"] = np.concatenate(
        [leiden["clustering_space"], np.vstack(new_clustering_rows).astype(np.float32)], axis=0
    )

    # --- 4. 2D display coords (viz-only; never a clustering input) -----------
    # A failure here (e.g. an environment-specific UMAP/numba transform bug)
    # must not block the additive append itself -- fall back to NaN rows so
    # the mismatch is visible/filterable downstream instead of silently
    # wrong, and log a warning.
    n_new = len(kept_items)
    if leiden.get("reducer_2d") is not None:
        try:
            new_coords_2d = leiden["reducer_2d"].transform(
                np.vstack(new_clustering_rows)
            ).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - viz-only, must not abort append
            print(f"[extend_corpus] WARNING: reducer_2d.transform failed ({exc!r}); "
                  f"new tracks get NaN 2D coords (display-only, does not affect "
                  f"embeddings/cluster assignment/search).")
            new_coords_2d = np.full(
                (n_new, corpus.embedding_2d.shape[1]), np.nan, dtype=np.float32
            )
    else:
        new_coords_2d = np.zeros((n_new, corpus.embedding_2d.shape[1]), dtype=np.float32)
    corpus.embedding_2d = np.concatenate([corpus.embedding_2d, new_coords_2d], axis=0)

    # --- 5. raw embeddings (enables future refits/re-dedupe) -----------------
    corpus.embeddings = np.concatenate([corpus.embeddings, kept_raw], axis=0).astype(np.float32)

    # --- 6. optional MERIT-aggregate sidecar extension -----------------------
    merit_report = None
    have_backbones = all(it.get("merit_backbone") is not None for it in kept_items)
    if corpus.merit_index is not None and have_backbones:
        from ..merit import FACTORS, load_heads, project

        heads = load_heads(heads_dir)
        backbones = np.stack([np.asarray(it["merit_backbone"], dtype=np.float32) for it in kept_items])
        proj = project(backbones, heads)
        new_concat = np.concatenate([proj[f] for f in FACTORS], axis=1).astype(np.float32)

        merit_idx = corpus.merit_index
        merit_new_transformed = _l2norm(
            _standardize_new(merit_idx.mean_, merit_idx.scale_, new_concat)
        )
        merit_idx.embeddings = np.concatenate(
            [merit_idx.embeddings, merit_new_transformed], axis=0
        ).astype(np.float32)
        merit_idx.metadata = merit_idx.metadata + new_meta

        factors_dir = bundle_dir
        for f in FACTORS:
            path = factors_dir / f"factor_{f}.npy"
            if path.exists():
                _append_npy_streaming(
                    path, out_dir / f"factor_{f}.npy", proj[f]
                )

        backbone_path = bundle_dir / "merit_backbone.npy"
        if backbone_path.exists():
            _append_npy_streaming(
                backbone_path, out_dir / "merit_backbone.npy", backbones
            )
        merit_report = {"n_extended": len(kept_items)}
    elif corpus.merit_index is not None and not have_backbones:
        merit_report = {
            "skipped": "bundle carries a MERIT-aggregate index but not all "
            "new_items supplied merit_backbone — sidecar left unextended "
            "for these tracks"
        }

    # --- 7. recompute cluster profiles (display bookkeeping only) -----------
    corpus.centroids, corpus.profiles = build_cluster_profiles(
        leiden["clustering_space"], leiden["labels"], index.metadata
    )

    # --- persist --------------------------------------------------------------
    corpus.manifest.setdefault("extensions", []).append({
        "timestamp": utc_now_iso(),
        "n_added": len(kept_items),
        "n_dropped_duplicate": n_dropped,
        "sources": sorted({it["meta"].get("source", "unknown") for it in kept_items}),
    })
    corpus.manifest["n_tracks"] = int(corpus.n_tracks)

    # corpus.leiden IS `leiden` (mutated in place above), and
    # ReferenceCorpus.save() persists self.leiden via save_leiden() itself --
    # no separate save_leiden call needed here.
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save(out_dir)
    if merit_report and merit_report.get("n_extended"):
        corpus.merit_index.save(out_dir / "index_merit_agg")

    cluster_sizes_after = {p["cluster_id"]: p["size"] for p in corpus.profiles}

    return {
        "n_before": n_before,
        "n_after": corpus.n_tracks,
        "n_input": len(new_items),
        "n_dropped_duplicate": n_dropped,
        "n_added": len(kept_items),
        "cluster_sizes_after": cluster_sizes_after,
        "merit_sidecar": merit_report,
        "out_dir": str(out_dir),
    }
