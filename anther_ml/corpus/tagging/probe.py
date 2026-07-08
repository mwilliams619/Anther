"""
Stage B — the tag probe (MICROGENRE_TAGGING_BUILD_PLAN.md §4).

A multi-label linear probe over the corpus's frozen raw MERT vectors:
one-vs-rest logistic regression per *learnable* genre (seed support >= floor),
trained on the confident seed subset with an artist-stratified split so the
probe can't memorize artists. The fitted heads collapse into a single
``(G, D)`` weight matrix, so inference is one matmul + sigmoid.

Transform discipline: the probe fits its **own** StandardScaler on the corpus
embeddings and freezes it (the bundle's SongIndex was built with
``index_standardize: false``, so "match the index" would mean no scaling —
the probe standardizes because logistic regression wants it). The same frozen
scaler is applied to every input — corpus, FMA eval vectors, placed queries.

Display-only: probe outputs never feed the embedding, SongIndex, Leiden, or
anther_ml.eval, and probe metrics never tune them (docs/invariants.md).
"""

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DEFAULT_CONF_THRESHOLD = 1.0  # one exact match, or ~two substring matches
DEFAULT_MIN_SUPPORT = 50  # confident train positives below this → not learnable
SEED_MIN_WEIGHT = 0.5  # seed-fallback tags need at least this Stage-A weight


def group_split(
    groups: list[str], val_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, val_mask) with no group on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    idx_train, idx_val = next(splitter.split(np.zeros(len(groups)), groups=groups))
    train = np.zeros(len(groups), dtype=bool)
    val = np.zeros(len(groups), dtype=bool)
    train[idx_train] = True
    val[idx_val] = True
    return train, val


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Max-F1 threshold on the val fold; 0.5 when degenerate."""
    if y_true.sum() < 5 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # last point has no threshold
    return float(thresholds[best])


class TagProbe:
    """Frozen linear multi-label tagger. Build with ``TagProbe.fit``."""

    def __init__(
        self,
        genre_names: list[str],
        scaler: StandardScaler,
        coef: np.ndarray,
        intercept: np.ndarray,
        thresholds: np.ndarray,
        fit_report: dict,
    ):
        self.genre_names = list(genre_names)  # learnable genres only
        self.scaler = scaler
        self.coef = np.asarray(coef, dtype=np.float32)  # (G, D)
        self.intercept = np.asarray(intercept, dtype=np.float32)  # (G,)
        self.thresholds = np.asarray(thresholds, dtype=np.float32)  # (G,)
        self.fit_report = fit_report

    # -- training ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        seed_weights: sparse.csr_matrix,
        genre_names: list[str],
        artists: list[str],
        *,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_support: int = DEFAULT_MIN_SUPPORT,
        val_frac: float = 0.2,
        seed: int = 0,
        max_iter: int = 1000,
        verbose: bool = False,
    ) -> "TagProbe":
        """Fit one-vs-rest heads on the confident seed subset.

        X: (N, D) raw corpus embeddings; seed_weights: (N, G) Stage-A weight
        matrix; artists: length-N group labels for the leak-free split.
        """
        X = np.asarray(X, dtype=np.float32)
        n, _ = X.shape
        if seed_weights.shape != (n, len(genre_names)):
            raise ValueError(
                f"seed_weights {seed_weights.shape} disagrees with "
                f"X rows {n} / vocab {len(genre_names)}"
            )

        scaler = StandardScaler().fit(X)  # frozen on the full corpus

        confidence = np.asarray(seed_weights.sum(axis=1)).ravel()
        confident = confidence >= conf_threshold
        idx_conf = np.flatnonzero(confident)
        if len(idx_conf) < 10 * min_support:
            raise ValueError(
                f"only {len(idx_conf)} confident seed tracks at "
                f"conf_threshold={conf_threshold} — not enough to train"
            )

        groups = [artists[i] or f"__row{i}" for i in idx_conf]
        train_m, val_m = group_split(groups, val_frac=val_frac, seed=seed)
        idx_train, idx_val = idx_conf[train_m], idx_conf[val_m]

        Y = (seed_weights > 0).tocsc()
        pos_train = np.asarray(Y[idx_train].sum(axis=0)).ravel()
        pos_val = np.asarray(Y[idx_val].sum(axis=0)).ravel()
        learnable = np.flatnonzero(pos_train >= min_support)
        if len(learnable) == 0:
            raise ValueError(f"no genre reaches min_support={min_support}")

        Xs_train = scaler.transform(X[idx_train]).astype(np.float32)
        Xs_val = scaler.transform(X[idx_val]).astype(np.float32)

        from joblib import Parallel, delayed

        def _fit_one(j: int):
            y = np.asarray(Y[idx_train, j].todense()).ravel().astype(np.int8)
            lr = LogisticRegression(
                class_weight="balanced", max_iter=max_iter, solver="lbfgs"
            )
            lr.fit(Xs_train, y)
            return lr.coef_[0].astype(np.float32), float(lr.intercept_[0])

        results = Parallel(n_jobs=-1, prefer="threads", verbose=5 if verbose else 0)(
            delayed(_fit_one)(j) for j in learnable
        )
        coef = np.stack([c for c, _ in results])
        intercept = np.asarray([b for _, b in results], dtype=np.float32)

        # Per-genre max-F1 thresholds + val metrics on the held-out fold.
        probs_val = _sigmoid(Xs_val @ coef.T + intercept)
        thresholds = np.empty(len(learnable), dtype=np.float32)
        per_genre = {}
        for k, j in enumerate(learnable):
            y_true = np.asarray(Y[idx_val, j].todense()).ravel()
            thresholds[k] = _tune_threshold(y_true, probs_val[:, k])
            pred = probs_val[:, k] >= thresholds[k]
            tp = float(np.sum(pred & (y_true > 0)))
            p = tp / max(pred.sum(), 1)
            r = tp / max(y_true.sum(), 1)
            per_genre[genre_names[j]] = {
                "support_train": int(pos_train[j]),
                "support_val": int(pos_val[j]),
                "threshold": float(thresholds[k]),
                "val_precision": round(p, 4),
                "val_recall": round(r, 4),
                "val_f1": round(2 * p * r / max(p + r, 1e-12), 4),
            }

        f1s = [m["val_f1"] for m in per_genre.values()]
        fit_report = {
            "n_confident": int(len(idx_conf)),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_learnable": int(len(learnable)),
            "conf_threshold": conf_threshold,
            "min_support": min_support,
            "val_macro_f1": round(float(np.mean(f1s)), 4),
            "per_genre": per_genre,
        }
        return cls(
            [genre_names[j] for j in learnable],
            scaler, coef, intercept, thresholds, fit_report,
        )

    # -- inference ----------------------------------------------------------

    def predict_proba(self, X: np.ndarray, block: int = 8192) -> np.ndarray:
        """(N, G_learnable) sigmoid scores, blockwise for memory."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        out = np.empty((len(X), len(self.genre_names)), dtype=np.float32)
        for s in range(0, len(X), block):
            Xs = self.scaler.transform(X[s : s + block]).astype(np.float32)
            out[s : s + block] = _sigmoid(Xs @ self.coef.T + self.intercept)
        return out

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str | Path) -> "TagProbe":
        with open(path, "rb") as f:
            state = pickle.load(f)
        probe = cls.__new__(cls)
        probe.__dict__.update(state)
        return probe


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def knn_smooth(
    probs: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
    alpha: float = 0.5,
    block: int = 1024,
) -> np.ndarray:
    """Label propagation: blend each track's scores with the mean of its k
    nearest neighbors' scores, cosine in raw MERT space. Fills tracks the
    probe is unsure about and denoises stray seeds (plan §4d)."""
    Xn = np.asarray(embeddings, dtype=np.float32)
    Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
    out = np.empty_like(probs)
    for s in range(0, len(Xn), block):
        sims = Xn[s : s + block] @ Xn.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -np.inf  # never your own neighbor
        nbr = np.argpartition(sims, -k, axis=1)[:, -k:]
        out[s : s + block] = probs[nbr].mean(axis=1)
    return (1.0 - alpha) * probs + alpha * out


def predict_tags(
    probe: TagProbe,
    X: np.ndarray,
    *,
    seed_weights: sparse.csr_matrix | None = None,
    full_genre_names: list[str] | None = None,
    top_k: int = 3,
    knn_smooth_k: int = 0,
    alpha: float = 0.5,
) -> list[dict]:
    """Per-track tag lists: probe scores over its threshold, capped at top_k,
    argmax flagged primary. Genres the probe couldn't learn fall back to their
    Stage-A seeds (source "seed") when ``seed_weights``/``full_genre_names``
    are given. Returns [{"tags": [{"genre","score","primary","source"}]}]."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    probs = probe.predict_proba(X)
    if knn_smooth_k > 0 and len(X) > knn_smooth_k:
        probs = knn_smooth(probs, X, k=knn_smooth_k, alpha=alpha)

    learnable = set(probe.genre_names)
    seed_rows = None
    if seed_weights is not None and full_genre_names is not None:
        seed_rows = seed_weights.tocsr()

    out = []
    for i in range(len(X)):
        cands = [
            (probe.genre_names[j], float(probs[i, j]), "probe")
            for j in np.flatnonzero(probs[i] >= probe.thresholds)
        ]
        if seed_rows is not None:
            row = seed_rows[i]
            for j, w in zip(row.indices, row.data):
                g = full_genre_names[j]
                if g not in learnable and w >= SEED_MIN_WEIGHT:
                    cands.append((g, float(min(1.0, w / 2.0)), "seed"))
        cands.sort(key=lambda t: -t[1])
        out.append(
            {
                "tags": [
                    {
                        "genre": g,
                        "score": round(s, 4),
                        "primary": rank == 0,
                        "source": src,
                    }
                    for rank, (g, s, src) in enumerate(cands[:top_k])
                ]
            }
        )
    return out
