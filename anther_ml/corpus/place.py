"""
The placement regime (REFERENCE_CORPUS_DESIGN.md §2, §5).

A query song is embedded with the corpus's own recipe, then its raw vector
fans out to the bundle's two frozen transforms:

  * SongIndex space  → nearest released tracks, playlist-fit cosines
  * Leiden space     → cluster assignment (k-NN vote), 2D coords (display only)

The corpus is never re-fit. Playlist fit is calibrated against the corpus as
a null distribution: "fits this playlist better than N% of released music" —
a raw cosine means nothing on its own; the percentile is the headline metric.
"""

import numpy as np

from ..cluster import assign_cluster_knn
from .bundle import ReferenceCorpus
from .popularity import rerank_with_popularity


def embed_query(
    path, corpus: ReferenceCorpus, model=None, processor=None, device=None
) -> np.ndarray:
    """
    Embed one audio file with the corpus's frozen recipe (read from its
    embedding_config stamp — design §3C: a query can never be embedded with a
    different recipe than the map it lands on).
    """
    from ..embedding import embedding_config, get_embedding, load_mert

    cfg = corpus.embedding_config
    layer_aggregation = cfg.get("layer_aggregation", "mean")
    normalize = bool(cfg.get("loudness_normalize", True))
    corpus.assert_compatible(
        embedding_config(layer_aggregation=layer_aggregation, normalize=normalize)
    )
    if model is None:
        model, processor, device = load_mert(device)
    return get_embedding(
        model, processor, path, device,
        layer_aggregation=layer_aggregation, normalize=normalize,
    )


def embed_query_dual(
    path, corpus: ReferenceCorpus, model=None, processor=None, device=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed one audio file with the corpus's frozen recipe, returning both the
    1024-d MERT vector and the 5120-d MERIT backbone from one forward pass
    (query-time counterpart to ``build.py``'s ``embed_tracks_batched_dual``).
    Used when the corpus carries a MERIT-aggregate sidecar index (see
    ``corpus.merit_index``) so an out-of-corpus query can be placed and
    scored in both spaces.
    """
    from ..embedding import embedding_config, get_embedding_dual, load_mert

    cfg = corpus.embedding_config
    normalize = bool(cfg.get("loudness_normalize", True))
    corpus.assert_compatible(
        embedding_config(layer_aggregation=cfg.get("layer_aggregation", "mean"), normalize=normalize)
    )
    if model is None:
        model, processor, device = load_mert(device)
    return get_embedding_dual(model, processor, path, device, normalize=normalize)


def place(
    corpus: ReferenceCorpus, vec: np.ndarray, top_k: int = 10, knn_k: int = 15,
    merit_vec: np.ndarray | None = None,
    popularity_pct: dict[str, float] | None = None,
    popularity_beta: float = 0.15,
) -> dict:
    """
    Place a raw query vector onto the frozen map. Returns::

        {"neighbors":  index.query() rows,
         "cluster":    {"id", "confidence", "profile"},
         "tags":       [{"genre","score","primary","source"}],  # display only
         "coords_2d":  [x, y] | None}   # UMAP transform — display only

    ``tags`` come from the bundle's tag probe when present, else are
    inherited from nearest neighbors' track_tags, else ``[]`` — bundles
    without tag artifacts still place.

    Cluster assignment and 2D coords always use the MERT-1024 space ``vec``
    (Leiden was fit there and never refit on MERIT — see
    ``anther_ml.cluster.assign_cluster_knn``). ``neighbors`` — the ranked
    similarity list driving corpus search and the song-detail "similar"
    list — comes from the bundle's MERIT-aggregate index when
    ``merit_vec`` (the raw 384-d factor concat, see
    ``anther_ml.merit.merit_query_vector``) is given and the bundle carries
    one (``corpus.merit_index``); otherwise it falls back to the MERT-1024
    index, unchanged from the pre-MERIT behavior.

    ``popularity_pct`` (optional): a ``{track_id: percentile}`` map (see
    ``anther_ml.corpus.popularity.popularity_percentiles``). When given,
    ``neighbors`` is re-ranked by ``(1-beta)*cosine + beta*percentile``
    (metadata-only blend -- never touches embeddings/index/Leiden; omit it
    and behavior is byte-identical to before this parameter existed).
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    leiden = corpus.leiden
    # Popularity reranking widens the candidate pool before re-sorting so a
    # popular track ranked outside top_k by cosine alone can still surface.
    fetch_k = top_k * 4 if popularity_pct else top_k
    if merit_vec is not None and corpus.merit_index is not None:
        neighbors = corpus.merit_index.query(
            np.asarray(merit_vec, dtype=np.float32).reshape(-1), top_k=fetch_k
        )
    else:
        neighbors = corpus.index.query(vec, top_k=fetch_k)
    if popularity_pct:
        neighbors = rerank_with_popularity(
            neighbors, popularity_pct, alpha=1.0 - popularity_beta, beta=popularity_beta
        )[:top_k]
    cluster_id, confidence = assign_cluster_knn(
        vec,
        leiden["clustering_space"],
        leiden["labels"],
        scaler=leiden["scaler"],
        pca=leiden["pca"],
        k=knn_k,
    )

    coords_2d = None
    if leiden.get("reducer_2d") is not None:
        X = vec.reshape(1, -1)
        if leiden["scaler"] is not None:
            X = leiden["scaler"].transform(X)
        if leiden["pca"] is not None:
            X = leiden["pca"].transform(X)
        coords_2d = leiden["reducer_2d"].transform(X)[0].tolist()

    profile = corpus.cluster_profile(cluster_id)
    return {
        "neighbors": neighbors,
        "cluster": {
            "id": cluster_id,
            "confidence": confidence,
            "label": profile.get("label_final", ""),
            "profile": profile,
        },
        "tags": _query_tags(corpus, vec),
        "coords_2d": coords_2d,
    }


def _query_tags(
    corpus: ReferenceCorpus, vec: np.ndarray, top_k: int = 3, knn: int = 10
) -> list[dict]:
    """Micro-genre tags for a query (display only, docs/invariants.md).

    Live probe when the bundle carries one; otherwise inherit from the top-k
    neighbors' precomputed track_tags; otherwise []."""
    probe = corpus.tag_probe
    if probe is not None:
        from .tagging.probe import predict_tags

        return predict_tags(probe, vec[None], top_k=top_k)[0]["tags"]

    track_tags = corpus.track_tags
    if track_tags is None:
        return []
    q = corpus.index.transform_query(vec)
    sims = corpus.index.embeddings @ q
    nbr = np.argsort(sims)[::-1][:knn]
    score: dict[str, float] = {}
    for i in nbr:
        for t in track_tags[int(i)]["tags"]:
            score[t["genre"]] = score.get(t["genre"], 0.0) + t["score"] / knn
    top = sorted(score.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        {"genre": g, "score": round(s, 4), "primary": rank == 0,
         "source": "neighbors"}
        for rank, (g, s) in enumerate(top)
    ]


def recommend_from_seeds(
    corpus: ReferenceCorpus,
    seed_vecs,
    top_k: int = 20,
    exclude_ids=None,
    method: str = "centroid",
    per_seed_k: int = 3,
    popularity_pct: dict[str, float] | None = None,
    popularity_beta: float = 0.15,
) -> list[dict]:
    """
    Recommend corpus tracks similar to a *set* of seed songs — the multi-song
    "find music like this feeling" query (use-case 2).

    ``seed_vecs`` is an iterable of **raw** query vectors (pre-transform, one per
    searched song), exactly as passed to ``place``/``index.query``. Each is put
    through the index's frozen transform (standardize against corpus stats, then
    L2), so seeds and corpus rows are compared in the same space.

    Scoring (``method``):
      * ``"centroid"`` (default) — average the transformed unit seed vectors,
        re-normalize, and rank corpus tracks by cosine to that centroid. Rewards
        songs near the shared center of all seeds.
      * ``"topk"`` — score each candidate by the mean of its top-``per_seed_k``
        cosines across the seeds (the reverse of ``playlist_fit``). Rewards a
        song strongly similar to a *subset* of seeds, so a two-mood seed set
        doesn't collapse to an empty midpoint. Interface-compatible drop-in.

    A seed that is itself a corpus track would score ~1.0 against itself; pass
    the seed ids (plus anything the user already has) via ``exclude_ids`` to
    drop them. Returns ``top_k`` rows: ``{"rank", "score", **metadata}`` — the
    same shape as ``SongIndex.query``.

    ``popularity_pct`` (optional): a ``{track_id: percentile}`` map (see
    ``anther_ml.corpus.popularity.popularity_percentiles``). When given, the
    candidate pool is widened, re-ranked by
    ``(1-popularity_beta)*cosine + popularity_beta*percentile``, then cut to
    ``top_k`` — a metadata-only blend that never touches embeddings/index.
    Omit it and behavior is byte-identical to before this parameter existed.
    """
    index = corpus.index
    seeds = list(seed_vecs)
    if not seeds:
        raise ValueError("recommend_from_seeds needs at least one seed vector")
    Q = np.stack([index.transform_query(v) for v in seeds])  # (m, D), unit rows

    if method == "centroid":
        centroid = Q.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:  # seeds cancel exactly (antipodal) — no shared center
            raise ValueError("seed centroid is degenerate (zero vector)")
        scores = index.embeddings @ (centroid / norm)
    elif method == "topk":
        sims = index.embeddings @ Q.T  # (N, m): candidate × seed cosines
        scores = _topk_mean(sims, per_seed_k)
    else:
        raise ValueError(f"unknown method {method!r} (use 'centroid' or 'topk')")

    drop = set(exclude_ids or ())
    order = np.argsort(scores)[::-1]
    fetch_k = top_k * 4 if popularity_pct else top_k
    results = []
    for idx in order:
        meta = index.metadata[int(idx)]
        if meta.get("id") in drop:
            continue
        entry = {"rank": len(results) + 1, "score": float(scores[idx])}
        entry.update(meta)
        results.append(entry)
        if len(results) >= fetch_k:
            break
    if popularity_pct:
        results = rerank_with_popularity(
            results, popularity_pct, alpha=1.0 - popularity_beta, beta=popularity_beta
        )[:top_k]
    return results


def _topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest values along the last axis (k clipped to width)."""
    sims = np.atleast_2d(sims)
    k = min(k, sims.shape[1])
    top = np.sort(sims, axis=1)[:, ::-1][:, :k]
    return top.mean(axis=1)


def _member_indices(corpus: ReferenceCorpus, playlist) -> np.ndarray:
    member_idx = corpus.playlist_member_indices(playlist)
    if len(member_idx) == 0:
        raise ValueError(f"playlist {playlist!r} has no tracks in this corpus")
    return member_idx


def playlist_fit(
    corpus: ReferenceCorpus, vec: np.ndarray, playlist, k: int = 5
) -> float:
    """
    Raw fit of a query to a playlist: mean of its top-k cosines to playlist
    members, in the index's frozen space. Meaningless alone — calibrate it.
    """
    member_idx = _member_indices(corpus, playlist)
    q = corpus.index.transform_query(vec)
    sims = corpus.index.embeddings[member_idx] @ q
    return float(_topk_mean(sims, k)[0])


def _calibrate(
    corpus: ReferenceCorpus,
    q: np.ndarray,
    member_idx: np.ndarray,
    k: int,
    null_pool: np.ndarray,
) -> dict:
    """Percentile of the query's fit against fits of null (non-member) tracks."""
    members = corpus.index.embeddings[member_idx]
    raw_fit = float(_topk_mean(members @ q, k)[0])
    null_idx = np.setdiff1d(null_pool, member_idx, assume_unique=False)
    null_scores = _topk_mean(corpus.index.embeddings[null_idx] @ members.T, k)
    percentile = float(100.0 * np.mean(null_scores < raw_fit))
    return {
        "raw_fit": raw_fit,
        "percentile": percentile,
        "n_members": int(len(member_idx)),
        "n_null": int(len(null_idx)),
        "k": int(min(k, len(member_idx))),
    }


def _null_pool(corpus: ReferenceCorpus, n_null: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = corpus.n_tracks
    return rng.choice(n, size=min(n_null, n), replace=False)


def calibrate_fit(
    corpus: ReferenceCorpus,
    vec: np.ndarray,
    playlist,
    k: int = 5,
    n_null: int = 2000,
    seed: int = 0,
) -> dict:
    """
    The design §5 headline metric: the query's playlist fit as a percentile of
    the same score computed for random corpus tracks (members excluded) —
    "fits this playlist better than {percentile}% of released music."
    """
    member_idx = _member_indices(corpus, playlist)
    q = corpus.index.transform_query(vec)
    return _calibrate(corpus, q, member_idx, k, _null_pool(corpus, n_null, seed))


def rank_playlists(
    corpus: ReferenceCorpus,
    vec: np.ndarray,
    k: int = 5,
    n_null: int = 2000,
    min_members: int = 3,
    seed: int = 0,
) -> list[dict]:
    """
    Calibrated fit against every corpus playlist with >= min_members tracks,
    best first — "which playlist does this song fit best" (design §5
    generalization). One shared null sample across playlists for speed.
    """
    q = corpus.index.transform_query(vec)
    pool = _null_pool(corpus, n_null, seed)
    ranked = []
    for pl in corpus.playlists():
        if pl["n_tracks"] < min_members:
            continue
        member_idx = corpus.playlist_member_indices(
            pl["pid"] if pl["pid"] is not None else pl["name"]
        )
        ranked.append({**pl, **_calibrate(corpus, q, member_idx, k, pool)})
    return sorted(ranked, key=lambda r: (-r["percentile"], -r["raw_fit"]))
