"""
MERIT-aggregate SongIndex: the driving similarity signal for edges, corpus
search, and ranking (replacing the plain MERT-1024 SongIndex — see
docs/similarity.md and the historical
implemented_archive/MERIT_INTEGRATION_REPORT.md).

Built as an ADDITIVE sidecar inside an existing corpus bundle
(``models/corpus_<name>/``) that was built with ``--capture-merit-backbone``
(i.e. carries a ``merit_backbone.npy`` row-aligned to ``index.json``'s
metadata). Never touches the existing MERT-based ``index.npy`` / ``.json`` /
``embeddings.npy`` / ``leiden.pkl`` — Leiden cluster assignment still needs
the raw MERT-1024 vector (it was fit on that space; see
``anther_ml.cluster.assign_cluster_knn``), so both indices coexist:

    index.npy / index.json              MERT-1024 (kept — clustering + legacy)
    index_merit_agg.npy / .json         MERIT 384-d aggregate (mel+rhy+tim
                                         concat of 3 unit vectors — cosine of
                                         the concat == equal-weight mean of
                                         the 3 factor cosines)
    factor_mel.npy / factor_rhy.npy /
    factor_tim.npy                      (N, 128) unit vectors, metadata-row-
                                         aligned — the per-factor breakdown
                                         for the song-detail panel
    link_calibration_merit.json         LinkThresholds calibrated on
                                         index_merit_agg (separate file from
                                         link_calibration.json)
    link_calibration_merit_factors.json Per-factor LinkThresholds (melody/
                                         rhythm/timbre each calibrated off
                                         their OWN raw-cosine distribution,
                                         not the aggregate's — see
                                         calibrate_factor_link_thresholds)

Usage:
    python -m anther_ml.corpus.merit_index --bundle models/corpus_mpd_100k_merit
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..calibration import (
    CALIBRATION_FILENAME_MERIT,
    CALIBRATION_FILENAME_MERIT_FACTORS,
    calibrate_factor_link_thresholds,
    calibrate_link_thresholds,
    save_calibration,
    save_factor_calibration,
)
from ..merit import FACTORS, load_heads, merit_config, project
from ..similarity import SongIndex

INDEX_MERIT_AGG_STEM = "index_merit_agg"


def build_merit_aggregate_index(
    bundle_dir: str | Path,
    heads_dir: str = "models/merit_heads",
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict:
    """
    Project every track's ``merit_backbone.npy`` row through the 3 frozen
    MERIT heads, build the 384-d aggregate ``SongIndex``, save the per-factor
    sidecars, and calibrate+persist link thresholds for the new index. Safe
    to re-run (idempotent — overwrites only the MERIT-prefixed files).
    """
    bundle_dir = Path(bundle_dir)
    backbone_path = bundle_dir / "merit_backbone.npy"
    if not backbone_path.exists():
        raise FileNotFoundError(
            f"{backbone_path} not found — was this bundle built with "
            "--capture-merit-backbone?"
        )

    with open(bundle_dir / "index.json") as f:
        index_payload = json.load(f)
    metadata = index_payload["metadata"]

    backbone = np.load(backbone_path)
    if len(backbone) != len(metadata):
        raise ValueError(
            f"merit_backbone.npy has {len(backbone)} rows but index.json "
            f"metadata has {len(metadata)} — bundle is out of sync"
        )

    print(f"loading heads from {heads_dir} ...")
    heads = load_heads(heads_dir, device=device)

    n = len(backbone)
    out_dims = {f: heads[f].net[-1].out_features for f in FACTORS}
    factors = {f: np.zeros((n, out_dims[f]), dtype=np.float32) for f in FACTORS}
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = backbone[start : start + batch_size]
        proj = project(chunk, heads, device=device)
        for f in FACTORS:
            factors[f][start : start + len(chunk)] = proj[f]
        if start % (batch_size * 10) == 0:
            print(f"  projected {start}/{n} ({time.time()-t0:.1f}s)")
    print(f"projection done: {n} tracks in {time.time()-t0:.1f}s")

    for f in FACTORS:
        np.save(bundle_dir / f"factor_{f}.npy", factors[f])

    # 384-d concat of 3 unit vectors: cosine of the concat == mean of the 3
    # per-factor cosines (equal-weight aggregate), since each sub-vector has
    # unit norm. See anther_ml/merit.py's aggregate_similarity docstring.
    concat = np.concatenate([factors[f] for f in FACTORS], axis=1).astype(np.float32)
    cfg = merit_config(heads_dir)
    cfg["aggregate"] = "equal_weight_concat"
    cfg["factor_dim"] = out_dims[FACTORS[0]]
    agg_index = SongIndex(concat, metadata, standardize=False, config=cfg)
    agg_path = bundle_dir / INDEX_MERIT_AGG_STEM
    agg_index.save(agg_path)
    print(f"index_merit_agg: {concat.shape} -> {agg_path}.npy")

    thresholds = calibrate_link_thresholds(agg_index, n_pairs=200_000, seed=0)
    calib_path = save_calibration(bundle_dir, thresholds, filename=CALIBRATION_FILENAME_MERIT)
    print(f"{os.path.basename(calib_path)}: {thresholds.to_dict()}")

    # Per-factor thresholds — each factor's own raw-cosine distribution can
    # run much hotter/colder than the aggregate's (see calibration.py's
    # CALIBRATION_FILENAME_MERIT_FACTORS docstring), so reusing `thresholds`
    # above for melody/rhythm/timbre display scores clips one factor to 100
    # constantly while another rarely reaches it. Calibrate independently.
    factor_thresholds = calibrate_factor_link_thresholds(factors, n_pairs=200_000, seed=0)
    factor_calib_path = save_factor_calibration(
        bundle_dir, factor_thresholds, filename=CALIBRATION_FILENAME_MERIT_FACTORS)
    print(f"{os.path.basename(factor_calib_path)}: "
          f"{ {f: t.to_dict() for f, t in factor_thresholds.items()} }")

    return {
        "n_tracks": n,
        "thresholds": thresholds.to_dict(),
        "factor_thresholds": {f: t.to_dict() for f, t in factor_thresholds.items()},
        "factor_shapes": {f: factors[f].shape for f in FACTORS},
        "agg_shape": concat.shape,
    }


def load_merit_aggregate_index(bundle_dir: str | Path) -> SongIndex | None:
    """The bundle's MERIT-aggregate ``SongIndex``, or ``None`` if it hasn't
    been built yet (older bundles / not yet run through this module)."""
    bundle_dir = Path(bundle_dir)
    if not (bundle_dir / f"{INDEX_MERIT_AGG_STEM}.npy").exists():
        return None
    return SongIndex.load(bundle_dir / INDEX_MERIT_AGG_STEM)


def load_factor_vectors(bundle_dir: str | Path) -> dict[str, np.ndarray] | None:
    """``{"mel": (N,128), "rhy": (N,128), "tim": (N,128)}`` unit vectors,
    metadata-row-aligned, or ``None`` if the sidecars haven't been built."""
    bundle_dir = Path(bundle_dir)
    paths = {f: bundle_dir / f"factor_{f}.npy" for f in FACTORS}
    if not all(p.exists() for p in paths.values()):
        return None
    return {f: np.load(p) for f, p in paths.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--heads-dir", default="models/merit_heads")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    result = build_merit_aggregate_index(
        args.bundle, args.heads_dir, args.device, args.batch_size
    )
    print(json.dumps(result, indent=2, default=str))
