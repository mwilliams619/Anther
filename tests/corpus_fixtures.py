"""
Shared synthetic fixtures for the corpus tests: three well-separated Gaussian
blobs standing in for MERT embeddings, a fake source, and a deterministic fake
embed_fn — no MERT download, no network, no FMA data (the _FakeModel
philosophy from test_embedding.py).
"""

import numpy as np

DIM = 32
N_BLOBS = 3
NOISE_SIGMA = 1.5  # large enough that same-blob tracks are NOT near-duplicates


def blob_centers(dim: int = DIM, n_blobs: int = N_BLOBS) -> np.ndarray:
    """Orthogonal-ish centers, well separated relative to NOISE_SIGMA."""
    rng = np.random.default_rng(1234)
    centers = rng.normal(size=(n_blobs, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return (centers * 12.0).astype(np.float32)


CENTERS = blob_centers()

# Blob 0 is playlist "pl_a", blob 1 is "pl_b", blob 2 is on no playlist.
BLOB_PLAYLISTS = {
    0: [{"pid": 1, "name": "pl_a"}],
    1: [{"pid": 2, "name": "pl_b"}],
    2: [],
}


def item_blob(i: int) -> int:
    return i % N_BLOBS


def blob_vector(i: int, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Deterministic embedding for item i: its blob center plus seeded noise."""
    rng = np.random.default_rng(10_000 + i)
    return (CENTERS[item_blob(i)] + rng.normal(scale=sigma, size=DIM)).astype(
        np.float32
    )


def make_items(n: int = 30) -> list[dict]:
    """Fake source items. Deliberately no 'genre' key — genre must never be
    required by the build (display-only invariant)."""
    return [
        {
            "id": f"fake:{i:03d}",
            "name": f"song {i:03d}",
            "artist": f"artist {i % 5}",
            "source": "fake",
            "playlists": BLOB_PLAYLISTS[item_blob(i)],
            "path": f"/nonexistent/{i:03d}.mp3",
        }
        for i in range(n)
    ]


def fake_embed_fn(item: dict) -> np.ndarray:
    return blob_vector(int(item["id"].split(":")[1]))


def build_test_corpus(tmp_path, n: int = 30, name: str = "test", **kwargs):
    """Build a real (fitted, frozen, saved) bundle from the fake blobs."""
    from anther_ml.corpus import build_corpus

    corpus = build_corpus(
        make_items(n),
        name=name,
        out_dir=tmp_path,
        embed_fn=fake_embed_fn,
        seed=42,
        **kwargs,
    )
    return corpus, tmp_path / f"corpus_{name}"
