"""
Build a frozen reference-corpus bundle (REFERENCE_CORPUS_DESIGN.md §4).

Pipeline: iterate a source of tracks → embed each with the final MERT recipe
(checkpointed, resumable — the 8k build is a multi-hour job) → drop
near-identical duplicates → fit the SongIndex and Leiden transforms on the
corpus → compute per-cluster profiles → save the bundle and clear the
checkpoint.

Everything downstream of embedding composes existing anther_ml code
(SongIndex, fit_clusters_leiden); nothing here re-implements transforms.
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ..cluster import fit_clusters_leiden
from ..similarity import SongIndex
from .bundle import CORPUS_FORMAT_VERSION, ReferenceCorpus, utc_now_iso

MIN_TRACKS = 10  # fit_clusters_leiden's floor


class BuildCheckpoint:
    """
    Resumable embed-job state, keyed by each track's stable string ``id``.

    ``rows.jsonl`` holds one metadata line per embedded track (embed order);
    ``embeddings_partial.npy`` is rewritten atomically every
    ``flush_every`` adds (~32 MB at 8k tracks — negligible next to MERT
    inference). A crash therefore loses at most ``flush_every`` tracks.
    """

    def __init__(self, dir_path: str | Path, flush_every: int = 50):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._rows_path = self.dir / "rows.jsonl"
        self._emb_path = self.dir / "embeddings_partial.npy"
        self._rows: list[dict] = []
        self._vecs: list[np.ndarray] = []
        self._unflushed = 0
        if self._rows_path.exists():
            with open(self._rows_path) as f:
                self._rows = [json.loads(line) for line in f if line.strip()]
        if self._emb_path.exists():
            arr = np.load(self._emb_path)
            self._vecs = [arr[i] for i in range(len(arr))]
        # A crash between the two writes can leave one row/vector extra;
        # truncate to the shorter so the pair stays aligned.
        n = min(len(self._rows), len(self._vecs))
        self._rows, self._vecs = self._rows[:n], self._vecs[:n]

    def done_ids(self) -> set[str]:
        return {row["id"] for row in self._rows}

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, track_id: str, vec: np.ndarray, meta_row: dict) -> None:
        assert meta_row.get("id") == track_id
        self._rows.append(meta_row)
        self._vecs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        self._unflushed += 1
        if self._unflushed >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        # Name must end in .npy or np.save appends the extension itself.
        tmp = self.dir / "embeddings_partial.tmp.npy"
        np.save(tmp, np.vstack(self._vecs).astype(np.float32))
        os.replace(tmp, self._emb_path)
        with open(self._rows_path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")
        self._unflushed = 0

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if not self._rows:
            return np.empty((0, 0), dtype=np.float32), []
        return np.vstack(self._vecs).astype(np.float32), list(self._rows)

    def clear(self) -> None:
        for p in (
            self._rows_path,
            self._emb_path,
            self.dir / "embeddings_partial.tmp.npy",
        ):
            p.unlink(missing_ok=True)
        self._rows, self._vecs, self._unflushed = [], [], 0
        try:
            self.dir.rmdir()  # only if empty
        except OSError:
            pass


def dedupe_near_identical(
    embeddings: np.ndarray, threshold: float = 0.98
) -> np.ndarray:
    """
    Boolean keep-mask dropping rows whose cosine similarity to an earlier row
    exceeds ``threshold`` (first occurrence wins). Remixes/duplicate uploads
    create fake dense clusters (design §3A), so they're removed before fitting.

    Similarity is measured in *z-scored* space (stats ad hoc, fit on this
    batch): raw MERT space is anisotropic enough that unrelated tracks can
    exceed 0.9 raw cosine, which would over-drop.

    O(N²) similarity matrix — fine to ~50k rows (design's target range).
    """
    X = np.asarray(embeddings, dtype=np.float32)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    Z = (X - X.mean(axis=0)) / scale
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.where(norms == 0, 1.0, norms)
    sim = Z @ Z.T
    keep = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        if not keep[i]:
            continue
        keep[i + 1:] &= sim[i, i + 1:] <= threshold
    return keep


def build_cluster_profiles(
    clustering_space: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict],
    n_exemplars: int = 5,
) -> tuple[np.ndarray, list[dict]]:
    """
    Per-cluster centroids (in clustering_space) and human-readable profiles:
    size, exemplar tracks nearest the centroid, and display-only playlist/genre
    tallies (never clustering inputs — see CLAUDE.md invariant).

    Returns (centroids (K, d) in profile order, profiles list).
    """
    labels = np.asarray(labels)
    centroids, profiles = [], []
    for cid in sorted(int(c) for c in set(labels.tolist()) if c != -1):
        member_idx = np.flatnonzero(labels == cid)
        members = clustering_space[member_idx]
        centroid = members.mean(axis=0)
        centroids.append(centroid)

        c_norm = centroid / (np.linalg.norm(centroid) or 1.0)
        m_norms = np.linalg.norm(members, axis=1)
        sims = (members @ c_norm) / np.where(m_norms == 0, 1.0, m_norms)
        order = member_idx[np.argsort(sims)[::-1][:n_exemplars]]
        exemplars = [
            {
                "idx": int(i),
                "name": metadata[i].get("name"),
                "artist": metadata[i].get("artist"),
            }
            for i in order
        ]

        playlist_tally = Counter(
            pl.get("name")
            for i in member_idx
            for pl in metadata[i].get("playlists") or []
        )
        genre_tally = Counter(
            metadata[i].get("genre")
            for i in member_idx
            if metadata[i].get("genre")
        )
        profiles.append(
            {
                "cluster_id": cid,
                "size": int(len(member_idx)),
                "exemplars": exemplars,
                "top_playlists": playlist_tally.most_common(5),
                "top_genres": genre_tally.most_common(5),  # display only
            }
        )
    return (
        np.vstack(centroids).astype(np.float32)
        if centroids
        else np.empty((0, clustering_space.shape[1]), dtype=np.float32)
    ), profiles


def _default_embed_fn(model, processor, device, layer_aggregation, normalize):
    """
    Embed one source item with the final MERT recipe. Lazily loads MERT on
    first call. Items with a "path" go straight to get_embedding; items with
    an in-memory "audio" waveform are round-tripped through a temp WAV so both
    routes get byte-identical treatment.
    """
    state = {"model": model, "processor": processor, "device": device}

    def embed(item: dict) -> np.ndarray:
        from ..embedding import get_embedding, load_mert

        if state["model"] is None:
            state["model"], state["processor"], state["device"] = load_mert(
                state["device"]
            )
        if item.get("path"):
            return get_embedding(
                state["model"], state["processor"], item["path"], state["device"],
                layer_aggregation=layer_aggregation, normalize=normalize,
            )
        if item.get("audio") is not None:
            from ..embedding import SR
            from ..spotify_deezer import waveform_to_tempfile

            tmp = waveform_to_tempfile(item["audio"], item.get("sr", SR))
            try:
                return get_embedding(
                    state["model"], state["processor"], tmp, state["device"],
                    layer_aggregation=layer_aggregation, normalize=normalize,
                )
            finally:
                tmp.unlink(missing_ok=True)
        raise ValueError(f"source item {item.get('id')!r} has neither path nor audio")

    return embed


def build_corpus(
    source,
    name: str,
    out_dir: str | Path = "models",
    resolution: float = 1.0,
    limit: int | None = None,
    dedupe_threshold: float | None = 0.98,
    index_standardize: bool = False,
    whiten: bool = False,
    layer_aggregation: str = "mean",
    loudness_normalize: bool = True,
    n_pca_components: int | None = 100,
    seed: int = 42,
    resume: bool = True,
    checkpoint_flush_every: int = 50,
    embed_fn=None,
    model=None,
    processor=None,
    device=None,
) -> ReferenceCorpus:
    """
    Build and freeze a reference corpus from a source iterable (see
    corpus/sources.py for the item contract). Returns the saved bundle.

    ``embed_fn`` (item → (D,) vector) overrides the MERT embedder — the test
    seam. ``limit`` counts source items (including already-checkpointed ones),
    so a resumed limited build sees the same tracks.
    """
    from ..embedding import embedding_config

    bundle_dir = Path(out_dir) / f"corpus_{name}"
    checkpoint = BuildCheckpoint(
        bundle_dir / "checkpoint", flush_every=checkpoint_flush_every
    )
    if not resume:
        checkpoint.clear()
        checkpoint = BuildCheckpoint(
            bundle_dir / "checkpoint", flush_every=checkpoint_flush_every
        )
    done = checkpoint.done_ids()
    if done:
        print(f"resuming: {len(done)} tracks already embedded")

    if embed_fn is None:
        embed_fn = _default_embed_fn(
            model, processor, device, layer_aggregation, loudness_normalize
        )

    n_seen = n_failed = 0
    for item in source:
        if limit is not None and n_seen >= limit:
            break
        n_seen += 1
        if item["id"] in done:
            continue
        try:
            vec = embed_fn(item)
        except Exception as e:  # noqa: BLE001 — skip bad files, keep the job alive
            print(f"skipping {item['id']}: {e}")
            n_failed += 1
            continue
        meta_row = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in item.items()
            if k not in ("audio", "sr")
        }
        checkpoint.add(item["id"], vec, meta_row)
    checkpoint.flush()

    raw_embeddings, metadata = checkpoint.load()
    if len(raw_embeddings) < MIN_TRACKS:
        raise ValueError(
            f"corpus needs at least {MIN_TRACKS} embedded tracks, got "
            f"{len(raw_embeddings)} (embedded {n_seen - n_failed - len(done)} new, "
            f"{n_failed} failed)"
        )

    n_deduped = 0
    if dedupe_threshold is not None:
        keep = dedupe_near_identical(raw_embeddings, threshold=dedupe_threshold)
        n_deduped = int((~keep).sum())
        if n_deduped:
            dropped = [metadata[i]["id"] for i in np.flatnonzero(~keep)]
            print(f"dedupe: dropped {n_deduped} near-identical tracks: {dropped[:10]}")
            raw_embeddings = raw_embeddings[keep]
            metadata = [m for m, k in zip(metadata, keep) if k]

    config = embedding_config(
        layer_aggregation=layer_aggregation, normalize=loudness_normalize
    )
    index = SongIndex(
        raw_embeddings, metadata, standardize=index_standardize, config=config
    )
    leiden = fit_clusters_leiden(
        raw_embeddings,
        resolution=resolution,
        n_pca_components=n_pca_components,
        standardize=True,
        pca_whiten=whiten,
        random_state=seed,
    )
    centroids, profiles = build_cluster_profiles(
        leiden["clustering_space"], leiden["labels"], metadata
    )

    manifest = {
        "name": name,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "created_at": utc_now_iso(),
        "n_tracks": len(metadata),
        "embedding_config": config,
        "build_params": {
            "resolution": resolution,
            "n_pca_components": n_pca_components,
            "pca_whiten": whiten,
            "index_standardize": index_standardize,
            "dedupe_threshold": dedupe_threshold,
            "n_deduped": n_deduped,
            "limit": limit,
            "seed": seed,
        },
        "diagnostics": leiden["diagnostics"],
    }
    corpus = ReferenceCorpus(
        embeddings=raw_embeddings,
        index=index,
        leiden=leiden,
        embedding_2d=leiden["embedding_2d"],
        centroids=centroids,
        profiles=profiles,
        manifest=manifest,
    )
    corpus.save(bundle_dir)
    checkpoint.clear()
    print(
        f"corpus_{name}: {len(metadata)} tracks, "
        f"{leiden['diagnostics']['n_clusters']} clusters → {bundle_dir}"
    )
    return corpus
