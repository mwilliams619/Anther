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


def place(
    corpus: ReferenceCorpus, vec: np.ndarray, top_k: int = 10, knn_k: int = 15
) -> dict:
    """
    Place a raw query vector onto the frozen map. Returns::

        {"neighbors":  index.query() rows,
         "cluster":    {"id", "confidence", "profile"},
         "coords_2d":  [x, y] | None}   # UMAP transform — display only
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    leiden = corpus.leiden
    neighbors = corpus.index.query(vec, top_k=top_k)
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

    return {
        "neighbors": neighbors,
        "cluster": {
            "id": cluster_id,
            "confidence": confidence,
            "profile": corpus.cluster_profile(cluster_id),
        },
        "coords_2d": coords_2d,
    }


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
