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

    **Append-only / sharded (Tier 1C).** Each flush writes one *new* shard —
    ``emb_<k>.npy`` (its vectors) + ``rows_<k>.jsonl`` (their metadata) — and then
    commits by atomically rewriting ``shards.json``, the single source of truth
    for which shards count. Peak RAM is therefore flat in the number of
    *unflushed* vectors, not the whole corpus, and no flush ever re-serializes the
    growing embedding matrix (the old ``np.vstack`` rewrite that grew linearly and
    would bite at 100k). A crash mid-flush leaves uncommitted shard files that
    ``shards.json`` simply doesn't reference, so resume is exact and loses at most
    ``flush_every`` tracks.
    """

    def __init__(self, dir_path: str | Path, flush_every: int = 50):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._manifest_path = self.dir / "shards.json"
        self._shards: list[dict] = []          # committed shards, in order
        self._rows: list[dict] = []            # committed metadata rows, in order
        self._pending_rows: list[dict] = []
        self._pending_vecs: list[np.ndarray] = []
        self._load_committed()

    def _load_committed(self) -> None:
        if not self._manifest_path.exists():
            return
        try:
            manifest = json.loads(self._manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for shard in manifest.get("shards", []):
            rows_path = self.dir / shard["rows"]
            if not (self.dir / shard["emb"]).exists() or not rows_path.exists():
                break  # torn commit — stop at the last intact shard
            with open(rows_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self._shards.append(shard)
            self._rows.extend(rows)

    def done_ids(self) -> set[str]:
        return {r["id"] for r in self._rows} | {r["id"] for r in self._pending_rows}

    def __len__(self) -> int:
        return len(self._rows) + len(self._pending_rows)

    def add(self, track_id: str, vec: np.ndarray, meta_row: dict) -> None:
        assert meta_row.get("id") == track_id
        self._pending_rows.append(meta_row)
        self._pending_vecs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        if len(self._pending_vecs) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending_vecs:
            return
        k = len(self._shards)
        emb_name, rows_name = f"emb_{k:05d}.npy", f"rows_{k:05d}.jsonl"
        # tmp must still end in .npy or np.save appends the extension itself.
        emb_tmp = self.dir / f"emb_{k:05d}.tmp.npy"
        np.save(emb_tmp, np.vstack(self._pending_vecs).astype(np.float32))
        os.replace(emb_tmp, self.dir / emb_name)
        rows_tmp = self.dir / (rows_name + ".tmp")
        with open(rows_tmp, "w") as f:
            for row in self._pending_rows:
                f.write(json.dumps(row) + "\n")
        os.replace(rows_tmp, self.dir / rows_name)

        # Commit point: rewrite the manifest atomically. Until this lands, the
        # shard files above are invisible to a resume.
        self._shards.append({"emb": emb_name, "rows": rows_name,
                             "n": len(self._pending_vecs)})
        man_tmp = self.dir / "shards.json.tmp"
        man_tmp.write_text(json.dumps({"shards": self._shards}))
        os.replace(man_tmp, self._manifest_path)

        self._rows.extend(self._pending_rows)
        self._pending_rows, self._pending_vecs = [], []

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if not self._shards:
            return np.empty((0, 0), dtype=np.float32), []
        parts = [np.load(self.dir / s["emb"]) for s in self._shards]
        return np.vstack(parts).astype(np.float32), list(self._rows)

    def clear(self) -> None:
        for p in self.dir.glob("emb_*.npy*"):
            p.unlink(missing_ok=True)
        for p in self.dir.glob("rows_*.jsonl*"):
            p.unlink(missing_ok=True)
        for p in (self._manifest_path, self.dir / "shards.json.tmp"):
            p.unlink(missing_ok=True)
        self._shards, self._rows = [], []
        self._pending_rows, self._pending_vecs = [], []
        try:
            self.dir.rmdir()  # only if empty
        except OSError:
            pass


def _meta_row(item: dict) -> dict:
    """Project a source item to its stored metadata (drop the heavy audio)."""
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in item.items()
        if k not in ("audio", "sr")
    }


def _prefetch(iterable, size: int):
    """
    Run ``iterable`` in a background thread, buffering up to ``size`` items, so a
    slow consumer (the GPU) does not stall the producers (network fetch/decode) —
    the Tier 1B producer→consumer overlap. Exceptions in the producer propagate to
    the consumer; the thread is a daemon so an early ``break`` never hangs exit.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=max(1, size))
    done = object()

    def worker():
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:  # noqa: BLE001 — surface to consumer, then stop
            q.put(("__error__", e))
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            raise item[1]
        yield item


# Above this row count the exact O(N²) similarity matrix gets too big to
# materialize (100k² float32 ≈ 40 GB), so dedupe switches to the ANN path.
_DEDUPE_EXACT_MAX = 30000


def _zscore(embeddings: np.ndarray) -> np.ndarray:
    """Column z-score (stats fit on this batch). Raw MERT space is anisotropic
    enough that unrelated tracks can exceed 0.9 raw cosine, which would
    over-drop — z-scoring first is what makes the threshold meaningful."""
    X = np.asarray(embeddings, dtype=np.float32)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (X - X.mean(axis=0)) / scale


def _dedupe_exact(Z: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy first-occurrence-wins dedupe via the full N×N cosine matrix."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(norms == 0, 1.0, norms)
    sim = Zn @ Zn.T
    keep = np.ones(len(Z), dtype=bool)
    for i in range(len(Z)):
        if not keep[i]:
            continue
        keep[i + 1:] &= sim[i, i + 1:] <= threshold
    return keep


def _dedupe_ann(
    Z: np.ndarray, threshold: float, n_neighbors: int = 30, seed: int = 42
) -> np.ndarray:
    """
    Same greedy first-occurrence-wins result as :func:`_dedupe_exact`, but never
    materializes the dense matrix — an approximate-kNN graph (pynndescent, HNSW-
    style, scanpy's default) supplies only the candidate near-neighbors. This is
    the dropClust LSH pattern: for a dedupe threshold of ~0.98 the duplicates are
    each row's very nearest neighbors, so top-k retrieval catches them with
    recall ≈ 1. Turns O(N²) into ~O(N·k), so 100k+ fits in RAM.

    The distances pynndescent returns for the retrieved pairs are the *true*
    cosine distances, so an edge is never a false positive — only a missed edge
    (vanishingly rare at this threshold) could differ from exact.
    """
    from pynndescent import NNDescent

    n = len(Z)
    k = min(n_neighbors, n - 1)
    index = NNDescent(Z, metric="cosine", n_neighbors=k, random_state=seed)
    neighbors, distances = index.neighbor_graph
    max_dist = 1.0 - threshold  # cosine distance = 1 − cosine similarity

    # For each row, the earlier-indexed rows that are near-identical to it.
    earlier: list[list[int]] = [[] for _ in range(n)]
    for a in range(n):
        for b, d in zip(neighbors[a], distances[a]):
            b = int(b)
            if b == a or d > max_dist:
                continue
            hi, lo = (a, b) if a > b else (b, a)
            earlier[hi].append(lo)

    keep = np.ones(n, dtype=bool)
    for j in range(n):
        for i in earlier[j]:
            if keep[i]:
                keep[j] = False
                break
    return keep


def dedupe_near_identical(
    embeddings: np.ndarray,
    threshold: float = 0.98,
    method: str = "auto",
    n_neighbors: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Boolean keep-mask dropping rows whose cosine similarity to an *earlier, kept*
    row exceeds ``threshold`` (first occurrence wins). Remixes/duplicate uploads
    create fake dense clusters (design §3A), so they're removed before fitting.
    Similarity is measured in z-scored space.

    ``method``: ``"exact"`` (O(N²) matrix), ``"ann"`` (approximate-kNN, scales to
    100k+), or ``"auto"`` — exact at/under ``_DEDUPE_EXACT_MAX`` rows, ANN above.
    The two give the same keep-mask at the ~0.98 threshold (near-dups are top
    neighbours); ``"exact"`` remains the reference for validation.
    """
    Z = _zscore(embeddings)
    n = len(Z)
    if method == "auto":
        method = "exact" if n <= _DEDUPE_EXACT_MAX else "ann"
    if method == "exact":
        return _dedupe_exact(Z, threshold)
    if method == "ann":
        return _dedupe_ann(Z, threshold, n_neighbors=n_neighbors, seed=seed)
    raise ValueError(f"unknown dedupe method: {method!r}")


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
    batch_tracks: int = 16,
    batch_windows: int = 32,
    use_fp16: bool | None = None,
    prefetch_size: int = 64,
    embed_fn=None,
    model=None,
    processor=None,
    device=None,
) -> ReferenceCorpus:
    """
    Build and freeze a reference corpus from a source iterable (see
    corpus/sources.py for the item contract). Returns the saved bundle.

    ``embed_fn`` (item → (D,) vector) overrides the MERT embedder — the test
    seam; when it's given the build stays serial. Otherwise MERT runs the
    efficient path: a prefetch thread (``prefetch_size``) overlaps fetch/decode
    with the GPU (Tier 1B), and tracks are embedded ``batch_tracks`` at a time in
    shared fp16 forward passes (Tier 1A; ``batch_windows`` caps windows per pass,
    ``use_fp16=None`` → fp16 on CUDA). ``limit`` counts source items (including
    already-checkpointed ones), so a resumed limited build sees the same tracks.
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

    n_seen = n_failed = 0

    if embed_fn is not None:
        # Custom / test embedder: one item at a time, serial.
        for item in source:
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            if item["id"] in done:
                continue
            try:
                vec = embed_fn(item)
            except Exception as e:  # noqa: BLE001 — skip bad files, keep job alive
                print(f"skipping {item['id']}: {e}")
                n_failed += 1
                continue
            checkpoint.add(item["id"], vec, _meta_row(item))
    else:
        # Real MERT path: prefetch producers feed the GPU (Tier 1B); tracks are
        # embedded in shared fp16 batches (Tier 1A).
        from ..embedding import embed_tracks_batched, load_mert, prepare_waveform

        if model is None:
            model, processor, device = load_mert(device)

        batch: list[dict] = []

        def embed_current_batch() -> None:
            nonlocal n_failed
            waveforms, good = [], []
            for it in batch:
                try:
                    waveforms.append(
                        prepare_waveform(it, normalize=loudness_normalize)
                    )
                    good.append(it)
                except Exception as e:  # noqa: BLE001 — skip bad audio, keep going
                    print(f"skipping {it['id']}: {e}")
                    n_failed += 1
            batch.clear()
            if not good:
                return
            vecs = embed_tracks_batched(
                model, processor, waveforms, device,
                layer_aggregation=layer_aggregation,
                batch_windows=batch_windows, use_fp16=use_fp16,
            )
            for it, vec in zip(good, vecs):
                checkpoint.add(it["id"], vec, _meta_row(it))

        for item in _prefetch(source, prefetch_size):
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            if item["id"] in done:
                continue
            batch.append(item)
            if len(batch) >= batch_tracks:
                embed_current_batch()
        embed_current_batch()

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
        keep = dedupe_near_identical(
            raw_embeddings, threshold=dedupe_threshold, seed=seed
        )
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
    from .labels import apply_labels_to_profiles, generate_cluster_labels

    profiles = apply_labels_to_profiles(
        profiles, generate_cluster_labels(metadata, leiden["labels"])
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
