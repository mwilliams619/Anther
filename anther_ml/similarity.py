"""
Nearest-neighbor similarity search over pre-computed embeddings.

Works with both Phase 1 (518-dim FMA features) and Phase 2 (1024-dim MERT).
The index is just a numpy matrix — exact cosine similarity over <10k tracks.
Add FAISS later if scaling beyond that; correctness, not scale, is the concern.

**Standardization (Workstream A).** Raw librosa feature families span a ~8,000×
magnitude range (spectral rolloff/centroid in Hz ≈ 10³ vs. chroma/tonnetz
≈ 10⁻¹). L2-normalizing rows alone lets the few Hz-scale dimensions dominate
cosine similarity, so "most similar" degrades to "nearest spectral rolloff"
rather than musically similar. ``SongIndex`` therefore column-standardizes
(z-score) *before* L2-normalizing, using stats fit on the corpus only and
persisted *with* the index. A query is standardized against the corpus's stats,
never its own. Enable via ``standardize=True`` (default for Phase-1 librosa;
evaluate per-phase for Phase-2 MERT, where L2+cosine alone is often best).
"""

import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


class SongIndex:
    """
    In-memory cosine similarity index over a fixed set of songs.

    Usage:
        index = SongIndex(embeddings, metadata, standardize=True)
        index.save("models/index_phase1")     # writes .npy + .json (no ext!)

        index = SongIndex.load("models/index_phase1")
        results = index.query(query_embedding, top_k=10)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
        standardize: bool = False,
        config: dict | None = None,
    ):
        """
        embeddings:  (N, D) float32 array, one row per song
        metadata:    list of N dicts, each with at least {'name': str}
        standardize: z-score columns (fit on this corpus) before L2-normalizing.
                     Persisted with the index so queries use the same stats.
        config:      free-form dict describing how embeddings were produced
                     (phase, clip length, layers, loudness norm, ...). Stored in
                     the index JSON so incompatible indices can be detected.
        """
        assert len(embeddings) == len(metadata)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.standardize = bool(standardize)
        self.config = dict(config or {})
        self.format_version = FORMAT_VERSION

        if self.standardize:
            self.mean_ = embeddings.mean(axis=0)
            scale = embeddings.std(axis=0)
            # Guard zero-variance columns so they don't blow up to inf/nan.
            self.scale_ = np.where(scale == 0, 1.0, scale).astype(np.float32)
            self.mean_ = self.mean_.astype(np.float32)
        else:
            self.mean_ = None
            self.scale_ = None

        self.embeddings = _l2_normalize(self._standardize(embeddings)).astype(
            np.float32
        )
        self.metadata = metadata

    def _standardize(self, mat: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return mat
        return (mat - self.mean_) / self.scale_

    def transform_query(self, vec: np.ndarray) -> np.ndarray:
        """
        Put a raw query vector through the index's frozen transform
        (standardize against corpus stats, then L2-normalize) so that
        ``index.embeddings @ transform_query(vec)`` gives cosine scores
        identical to ``query()``'s.
        """
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"query dim {vec.shape[0]} != index dim {self.embeddings.shape[1]}"
            )
        return _l2_normalize(self._standardize(vec.reshape(1, -1)))[0]

    def query(self, vec: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Return top_k most similar songs.
        vec: 1D embedding of the query song. It is standardized against the
        *corpus* stats and L2-normalized, exactly like the corpus rows.

        Returns list of dicts: {'rank': int, 'score': float, **metadata_fields}
        """
        q = self.transform_query(vec)
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            entry = {"rank": rank, "score": float(scores[idx])}
            entry.update(self.metadata[idx])
            results.append(entry)
        return results

    def save(self, path: str | Path):
        """Persist as a (.npy, .json) pair. Always omit the extension."""
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.embeddings)
        meta = {
            "format_version": self.format_version,
            "standardize": self.standardize,
            "config": self.config,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "metadata": self.metadata,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str | Path) -> "SongIndex":
        path = Path(path)
        embeddings = np.load(path.with_suffix(".npy"))
        with open(path.with_suffix(".json")) as f:
            payload = json.load(f)

        obj = cls.__new__(cls)
        obj.embeddings = embeddings.astype(np.float32)

        if isinstance(payload, list):
            # Legacy v1 index: bare list of metadata, row-normalized only.
            obj.format_version = 1
            obj.standardize = False
            obj.config = {}
            obj.mean_ = None
            obj.scale_ = None
            obj.metadata = payload
        else:
            obj.format_version = payload.get("format_version", 1)
            obj.standardize = payload.get("standardize", False)
            obj.config = payload.get("config", {})
            mean = payload.get("mean")
            scale = payload.get("scale")
            obj.mean_ = np.asarray(mean, dtype=np.float32) if mean else None
            obj.scale_ = np.asarray(scale, dtype=np.float32) if scale else None
            obj.metadata = payload["metadata"]
        return obj

    def assert_compatible(self, other_config: dict) -> None:
        """
        Raise if this index was built with a config incompatible with a query
        source (Workstream G — don't silently mix indices). Compares only the
        keys present in ``other_config``.
        """
        for key, val in other_config.items():
            if key in self.config and self.config[key] != val:
                raise ValueError(
                    f"index/query config mismatch on {key!r}: index has "
                    f"{self.config[key]!r}, query has {val!r}"
                )


# Module-level convenience wrappers used in __init__.py

def build_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    standardize: bool = False,
    config: dict | None = None,
) -> SongIndex:
    return SongIndex(embeddings, metadata, standardize=standardize, config=config)


def find_nearest(
    index: SongIndex, query_vec: np.ndarray, top_k: int = 10
) -> list[dict]:
    return index.query(query_vec, top_k=top_k)
