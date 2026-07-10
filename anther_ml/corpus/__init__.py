"""
Reference corpus in MERT space (REFERENCE_CORPUS_DESIGN.md / Workstream W5).

A frozen map of released music — embeddings + fitted transforms + config
stamp — that new songs are placed onto, never re-clustered from scratch.

    python -m anther_ml.corpus build --source fma --name fma_small_v1
    python -m anther_ml.corpus place song.mp3 --corpus models/corpus_fma_small_v1
"""

from .build import (
    BuildCheckpoint,
    build_cluster_profiles,
    build_corpus,
    dedupe_near_identical,
)
from .bundle import CORPUS_FORMAT_VERSION, ReferenceCorpus
from .labels import (
    apply_labels_to_profiles,
    generate_cluster_labels,
    label_bundle,
    set_cluster_override,
)
from .place import (
    calibrate_fit,
    embed_query,
    place,
    playlist_fit,
    rank_playlists,
    recommend_from_seeds,
)
from .sources import fma_source, local_source, mpd_source

__all__ = [
    "BuildCheckpoint",
    "CORPUS_FORMAT_VERSION",
    "ReferenceCorpus",
    "apply_labels_to_profiles",
    "build_cluster_profiles",
    "build_corpus",
    "calibrate_fit",
    "dedupe_near_identical",
    "embed_query",
    "fma_source",
    "generate_cluster_labels",
    "label_bundle",
    "local_source",
    "mpd_source",
    "place",
    "playlist_fit",
    "rank_playlists",
    "recommend_from_seeds",
    "set_cluster_override",
]
