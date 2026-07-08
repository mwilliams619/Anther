"""
The frozen reference-corpus bundle (REFERENCE_CORPUS_DESIGN.md §1).

A ``ReferenceCorpus`` is not just embeddings — it is the fitted map: raw MERT
vectors, the SongIndex (search transform), the Leiden fit dict (cluster
transform), the 2D projection, per-cluster profiles, and a manifest stamping
how it was all built. New songs are placed *onto* it, never clustered from
scratch.

Two frozen transforms live side by side, each fed the raw 1024-d vector:

  * SongIndex (optional z-score → L2)          — neighbor search, playlist fit
  * Leiden scaler → PCA (``clustering_space``) — cluster assignment, 2D coords

They are never chained; ``place()`` fans the raw vector out to both.

On-disk layout (``models/corpus_<name>/``):
  manifest.json         build params + embedding_config stamp + diagnostics
  embeddings.npy        RAW (N, D) pre-transform vectors (enables future refits)
  index.npy / .json     SongIndex.save() — the metadata's single home
  leiden.pkl            save_leiden() output
  labels.npy            bare labels so `python -m anther_ml.eval --labels` works
  embedding_2d.npy      (N, 2) viz coords (save_leiden drops the array itself)
  centroids.npy         (K, d_cluster) per-cluster centroids in clustering_space
  cluster_profiles.json per-cluster size / exemplars / display-only tallies
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..cluster import load_leiden, save_leiden
from ..similarity import SongIndex

CORPUS_FORMAT_VERSION = 1

_UNSET = object()  # lazy-load sentinel for optional tag artifacts


class ReferenceCorpus:
    """In-memory handle on a frozen corpus bundle. See module docstring."""

    def __init__(
        self,
        embeddings: np.ndarray,
        index: SongIndex,
        leiden: dict,
        embedding_2d: np.ndarray,
        centroids: np.ndarray,
        profiles: list[dict],
        manifest: dict,
    ):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = index
        self.leiden = leiden
        self.embedding_2d = np.asarray(embedding_2d, dtype=np.float32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.profiles = profiles
        self.manifest = manifest
        self.dir: Path | None = None  # set by load(); tag artifacts live there
        self._tag_probe = _UNSET
        self._track_tags = _UNSET

    # -- convenience views (single source of truth stays in the parts) --------

    @property
    def metadata(self) -> list[dict]:
        return self.index.metadata

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.leiden["labels"])

    @property
    def n_tracks(self) -> int:
        return len(self.index.metadata)

    @property
    def embedding_config(self) -> dict:
        return self.manifest["embedding_config"]

    # -- persistence -----------------------------------------------------------

    def save(self, dir_path: str | Path) -> Path:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        np.save(d / "embedding_2d.npy", self.embedding_2d)
        np.save(d / "centroids.npy", self.centroids)
        np.save(d / "labels.npy", self.labels)
        self.index.save(d / "index")
        save_leiden(d / "leiden.pkl", self.leiden)
        with open(d / "cluster_profiles.json", "w") as f:
            json.dump(self.profiles, f, indent=2)
        with open(d / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)
        return d

    @classmethod
    def load(cls, dir_path: str | Path, verify: bool = True) -> "ReferenceCorpus":
        d = Path(dir_path)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no corpus bundle at {d} (missing manifest.json)")
        with open(d / "manifest.json") as f:
            manifest = json.load(f)
        with open(d / "cluster_profiles.json") as f:
            profiles = json.load(f)
        corpus = cls(
            embeddings=np.load(d / "embeddings.npy"),
            index=SongIndex.load(d / "index"),
            leiden=load_leiden(d / "leiden.pkl"),
            embedding_2d=np.load(d / "embedding_2d.npy"),
            centroids=np.load(d / "centroids.npy"),
            profiles=profiles,
            manifest=manifest,
        )
        corpus.dir = d
        if verify:
            corpus.verify()
        return corpus

    def verify(self) -> None:
        """Raise ValueError if the bundle is internally inconsistent."""
        version = self.manifest.get("corpus_format_version")
        if version != CORPUS_FORMAT_VERSION:
            raise ValueError(
                f"corpus_format_version {version!r} != supported "
                f"{CORPUS_FORMAT_VERSION} — rebuild the corpus or upgrade the code"
            )
        # The index's own config stamp must agree with the manifest's.
        self.index.assert_compatible(self.embedding_config)
        counts = {
            "embeddings": len(self.embeddings),
            "metadata": len(self.index.metadata),
            "index_rows": len(self.index.embeddings),
            "labels": len(self.labels),
            "embedding_2d": len(self.embedding_2d),
            "clustering_space": len(self.leiden["clustering_space"]),
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"bundle track counts disagree: {counts}")

    # -- optional tag artifacts (display-only; older bundles lack them) --------

    @property
    def tag_probe(self):
        """Frozen TagProbe from the bundle dir, or None (tags are optional —
        the schema stays additive, format version unchanged)."""
        if self._tag_probe is _UNSET:
            self._tag_probe = None
            if self.dir is not None and (self.dir / "tag_probe.pkl").exists():
                from .tagging.probe import TagProbe

                self._tag_probe = TagProbe.load(self.dir / "tag_probe.pkl")
        return self._tag_probe

    @property
    def track_tags(self) -> list[dict] | None:
        """track_tags.json rows (metadata order), or None if never built."""
        if self._track_tags is _UNSET:
            self._track_tags = None
            if self.dir is not None and (self.dir / "track_tags.json").exists():
                with open(self.dir / "track_tags.json") as f:
                    self._track_tags = json.load(f)
        return self._track_tags

    # -- lookups ---------------------------------------------------------------

    def assert_compatible(self, query_config: dict) -> None:
        """Refuse queries embedded with a different recipe (design §3C)."""
        self.index.assert_compatible(query_config)

    def playlists(self) -> list[dict]:
        """All playlists in the metadata: [{pid, name, n_tracks}], largest first."""
        tally: dict = {}
        for row in self.metadata:
            for pl in row.get("playlists") or []:
                key = pl.get("pid", pl.get("name"))
                entry = tally.setdefault(
                    key, {"pid": pl.get("pid"), "name": pl.get("name"), "n_tracks": 0}
                )
                entry["n_tracks"] += 1
        return sorted(tally.values(), key=lambda e: -e["n_tracks"])

    def playlist_member_indices(self, pid_or_name) -> np.ndarray:
        """Row indices of every track on the given playlist (by pid or name)."""
        idx = [
            i
            for i, row in enumerate(self.metadata)
            if any(
                pl.get("pid") == pid_or_name or pl.get("name") == pid_or_name
                for pl in row.get("playlists") or []
            )
        ]
        return np.asarray(idx, dtype=int)

    def cluster_profile(self, cluster_id: int) -> dict:
        for profile in self.profiles:
            if profile["cluster_id"] == int(cluster_id):
                return profile
        raise KeyError(f"no cluster profile for id {cluster_id}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
