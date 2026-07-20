"""
Build the MERIT-aggregate SongIndex + per-factor sidecars for a corpus bundle
that was built with --capture-merit-backbone (i.e. has a merit_backbone.npy
row-aligned to the bundle's metadata/embeddings.npy).

Writes into the bundle directory (never touches the existing MERT-based
index.npy/.json/embeddings.npy/leiden.pkl):

  index_merit_agg.npy / .json   — 384-d concat-of-3-unit-factors SongIndex
                                   (equal-weight cosine == mean of the 3
                                   factor cosines; standardize=False)
  factor_mel.npy                — (N, 128) unit vectors, metadata-row-aligned
  factor_rhy.npy
  factor_tim.npy
  link_calibration_merit.json   — LinkThresholds calibrated on index_merit_agg
                                   (separate file; does NOT overwrite the
                                   existing MERT-based link_calibration.json
                                   here — the caller decides which one atlas.py
                                   reads by filename, per the plan's "wire into
                                   atlas.py" step)

Usage:
    python scratch_eval/build_merit_aggregate_index.py --bundle models/corpus_mpd_100k_merit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anther_ml.calibration import calibrate_link_thresholds, save_calibration
from anther_ml.merit import FACTORS, load_heads, merit_config, project
from anther_ml.similarity import SongIndex


def build_merit_aggregate_index(bundle_dir: str | Path, heads_dir: str = "models/merit_heads",
                                 device: str = "cpu", batch_size: int = 4096) -> dict:
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
    factors = {f: np.zeros((n, 128), dtype=np.float32) for f in FACTORS}
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
    agg_index = SongIndex(concat, metadata, standardize=False, config=cfg)
    agg_index.save(bundle_dir / "index_merit_agg")
    print(f"index_merit_agg: {concat.shape} → {bundle_dir / 'index_merit_agg.npy'}")

    thresholds = calibrate_link_thresholds(agg_index, n_pairs=200_000, seed=0)
    calib_path = bundle_dir / "link_calibration_merit.json"
    tmp = str(calib_path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(thresholds.to_dict(), fh)
    import os
    os.replace(tmp, calib_path)
    print(f"link_calibration_merit.json: {thresholds.to_dict()}")

    return {
        "n_tracks": n,
        "thresholds": thresholds.to_dict(),
        "factor_shapes": {f: factors[f].shape for f in FACTORS},
        "agg_shape": concat.shape,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--heads-dir", default="models/merit_heads")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    result = build_merit_aggregate_index(args.bundle, args.heads_dir, args.device)
    print(json.dumps(result, indent=2, default=str))
