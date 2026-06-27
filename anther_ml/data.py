"""
FMA data loading utilities.

Expects fma_metadata/ directory containing:
  tracks.csv, genres.csv, features.csv, echonest.csv

Download: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_fma_tracks(metadata_dir: str | Path) -> pd.DataFrame:
    """
    Load tracks.csv with proper multi-index parsing.
    Returns DataFrame indexed by track_id with columns like
    ('track', 'genre_top'), ('set', 'subset'), etc.
    """
    path = Path(metadata_dir) / "tracks.csv"
    tracks = pd.read_csv(path, index_col=0, header=[0, 1])

    # Parse list-like columns
    for col in [("track", "tags"), ("album", "tags"), ("track", "genres"),
                ("track", "genres_all")]:
        if col in tracks.columns:
            tracks[col] = tracks[col].map(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )

    return tracks


def load_fma_features(metadata_dir: str | Path,
                      subset: str = "small") -> pd.DataFrame:
    """
    Load features.csv (pre-computed audio features, 568 columns).
    Filtered to the given FMA subset ('small', 'medium', 'large').
    Returns DataFrame indexed by track_id.
    """
    meta_path = Path(metadata_dir)
    features = pd.read_csv(
        meta_path / "features.csv", index_col=0, header=[0, 1, 2]
    )

    if subset is not None:
        tracks = load_fma_tracks(metadata_dir)
        keep = tracks[tracks[("set", "subset")] <= subset].index
        features = features.loc[features.index.isin(keep)]

    return features


def load_fma_genres(metadata_dir: str | Path) -> pd.DataFrame:
    """Load genres.csv with parent hierarchy."""
    path = Path(metadata_dir) / "genres.csv"
    return pd.read_csv(path, index_col=0)


def get_top_genre(tracks: pd.DataFrame, track_id: int) -> str:
    """Return the top-level genre string for a track_id, or 'Unknown'."""
    try:
        return tracks.loc[track_id, ("track", "genre_top")]
    except KeyError:
        return "Unknown"


def get_audio_path(audio_dir: str | Path, track_id: int) -> Path:
    """
    Convert a numeric track_id to its MP3 path.
    FMA uses a two-level directory: audio_dir/AAA/AAAAAA.mp3
    """
    tid_str = f"{track_id:06d}"
    return Path(audio_dir) / tid_str[:3] / f"{tid_str}.mp3"
