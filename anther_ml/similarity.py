"""
Nearest-neighbor similarity search over pre-computed embeddings.

Works with both Phase 1 (568-dim FMA features) and Phase 2 (1024-dim MERT).
The index is just a numpy matrix — simple cosine similarity over <10k tracks.
Add FAISS later if scaling beyond that.
"""

from pathlib import Path

import numpy as np


class SongIndex:
    """
    In-memory cosine similarity index over a fixed set of songs.

    Usage:
        index = SongIndex(embeddings, metadata)
        index.save("models/index_phase1.npz")

        index = SongIndex.load("models/index_phase1.npz")
        results = index.query(query_embedding, top_k=10)
    """

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        embeddings: (N, D) float32 array, one row per song
        metadata:   list of N dicts, each with at least {'name': str}
        """
        assert len(embeddings) == len(metadata)
        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.metadata = metadata

    def query(self, vec: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Return top_k most similar songs.
        vec: 1D embedding of the query song (will be L2-normalised).

        Returns list of dicts:
          {'rank': int, 'score': float, **metadata_fields}
        """
        norm = np.linalg.norm(vec)
        q = vec / (norm if norm > 0 else 1.0)
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            entry = {"rank": rank, "score": float(scores[idx])}
            entry.update(self.metadata[idx])
            results.append(entry)
        return results

    def save(self, path: str | Path):
        import json
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.embeddings)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "SongIndex":
        import json
        path = Path(path)
        embeddings = np.load(path.with_suffix(".npy"))
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)
        obj = cls.__new__(cls)
        obj.embeddings = embeddings
        obj.metadata = metadata
        return obj


# Module-level convenience wrappers used in __init__.py

def build_index(embeddings: np.ndarray, metadata: list[dict]) -> SongIndex:
    return SongIndex(embeddings, metadata)


def find_nearest(
    index: SongIndex, query_vec: np.ndarray, top_k: int = 10
) -> list[dict]:
    return index.query(query_vec, top_k=top_k)
