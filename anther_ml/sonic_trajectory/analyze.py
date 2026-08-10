"""Turn a placements table into label-independent structural metrics per era.

All trajectory claims are grounded in geometry, never in Leiden cluster titles:
- spread_bits      : Shannon entropy (bits) of the cluster distribution
- n_distinct_clusters
- dispersion       : mean pairwise Euclidean distance in the 2-D atlas
- centroid         : mean (x, y) of the era's tracks
- region_shares    : fraction of tracks per coarse region
- top_region / top_region_share
Centroid *shifts* (era-to-era Euclidean distance) are computed separately.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .regions import region_for

# Substrings marking grouping buckets (not studio releases) — excluded from the arc.
# Matched case-insensitively against the era label so cosmetic variants
# ("Singles / Soundtrack / Other", "Soundtracks / Singles / Other", ...) all catch.
_NON_STUDIO_MARKERS = ("single", "soundtrack", "/ other", "early", "misc")


def is_non_studio(era: str) -> bool:
    e = (era or "").lower()
    return any(m in e for m in _NON_STUDIO_MARKERS)


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the coarse ``region`` column from the canonical map."""
    df = df.copy()
    df["region"] = df["cluster_id"].map(region_for)
    return df


def _entropy_bits(counts: "list[int] | np.ndarray") -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return max(h, 0.0)  # clip -0.0


def _dispersion(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    d, n = 0.0, 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d += float(np.linalg.norm(coords[i] - coords[j]))
            n += 1
    return d / n if n else 0.0


def era_metrics(df: pd.DataFrame, *, studio_only: bool = True,
                min_year: bool = True) -> pd.DataFrame:
    """Per-era structural metrics, ordered chronologically.

    ``df`` must have columns: era, year, cluster_id, x2d, y2d (region is added
    if absent). Returns one row per era.
    """
    if "region" not in df.columns:
        df = add_region(df)
    rows: list[dict[str, Any]] = []
    for era, g in df.groupby("era"):
        if studio_only and is_non_studio(era):
            continue
        counts = g["cluster_id"].value_counts().to_list()
        coords = g[["x2d", "y2d"]].dropna().to_numpy(dtype=float)
        reg = g["region"].value_counts(normalize=True)
        rows.append({
            "era": era,
            "year": int(g["year"].min()),
            "n": int(len(g)),
            "n_distinct_clusters": int(g["cluster_id"].nunique()),
            "spread_bits": round(_entropy_bits(counts), 3),
            "dispersion": round(_dispersion(coords), 3),
            "centroid_x": round(float(coords[:, 0].mean()), 4) if len(coords) else None,
            "centroid_y": round(float(coords[:, 1].mean()), 4) if len(coords) else None,
            "top_region": reg.index[0] if len(reg) else None,
            "top_region_share": round(float(reg.iloc[0]), 3) if len(reg) else None,
            "region_shares": {k: round(float(v), 3) for k, v in reg.items()},
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def centroid_shifts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Era-to-era Euclidean distance between successive centroids."""
    rows = []
    prev = None
    for _, r in metrics.iterrows():
        if r["centroid_x"] is None:
            continue
        c = np.array([r["centroid_x"], r["centroid_y"]])
        if prev is not None:
            rows.append({
                "from_era": prev[0], "to_era": r["era"],
                "shift": round(float(np.linalg.norm(c - prev[1])), 3),
            })
        prev = (r["era"], c)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, *, studio_only: bool = True) -> dict[str, Any]:
    """One-call bundle: metrics table + shifts + a couple of headline stats."""
    m = era_metrics(df, studio_only=studio_only)
    shifts = centroid_shifts(m)
    biggest = shifts.loc[shifts["shift"].idxmax()].to_dict() if len(shifts) else None
    return {
        "n_tracks": int(len(df)),
        "n_distinct_clusters": int(df["cluster_id"].nunique()),
        "mean_cluster_conf": round(float(df["cluster_conf"].mean()), 3),
        "metrics": m.to_dict(orient="records"),
        "shifts": shifts.to_dict(orient="records"),
        "biggest_shift": biggest,
    }
