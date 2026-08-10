"""End-to-end sound-over-time pipeline for one artist, with optional profile persistence.

    from anther_ml.sonic_trajectory import run_artist, EraRule
    res = run_artist("Beyoncé", era_rules=[EraRule("lemonade", "Lemonade (2016)"), ...],
                     out_dir="results/beyonce", llm_fn=my_llm)

Steps: resolve discography (iTunes) -> embed+place on GPU (subprocess) ->
structural metrics -> two figures -> optional blurbs -> optional profile upsert.
The heavy GPU step runs only if placements aren't already cached in out_dir.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Callable

import pandas as pd

from anther_ml.spotify_deezer import _norm

from . import analyze, blurb as _blurb, plots
from .discography import EraRule, resolve_discography
from .embed import embed_and_place

SOURCE = "anther_ml.sonic_trajectory"


def _today() -> str:
    return _dt.date.today().isoformat()


def run_artist(
    artist_name: str,
    *,
    era_rules: list[EraRule],
    out_dir: str,
    artist_id: int | None = None,
    prefix: str | None = None,
    llm_fn: Callable[[str, str], str] | None = None,
    caveats: list[str] | None = None,
    label_tracks: list[str] | None = None,
    anchor_tracks: list[str] | None = None,
    store=None,
    make_figures: bool = True,
    reuse_cached: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline; return a result dict and optionally persist it.

    ``store`` is an ArtistProfileStore (or None). ``llm_fn(prompt, system)->str``
    enables blurbs. Returns {artist, artist_key, summary, figures, data, blurbs}.
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix or _norm(artist_name).replace(" ", "_") or "artist"

    # 1. discography
    rows = resolve_discography(artist_name, era_rules, artist_id=artist_id)
    disco_csv = os.path.join(out_dir, f"{prefix}_discography.csv")
    pd.DataFrame(rows).to_csv(disco_csv, index=False)

    # 2. embed + place (skip if cached)
    plc = os.path.join(out_dir, f"{prefix}_placements.csv")
    npz = os.path.join(out_dir, f"{prefix}_embeddings.npz")
    if reuse_cached and os.path.exists(plc) and os.path.exists(npz):
        emb = {"placements_csv": plc, "embeddings_npz": npz, "cached": True}
    else:
        emb = embed_and_place(disco_csv, out_dir, prefix)
        emb["cached"] = False

    df = analyze.add_region(pd.read_csv(emb["placements_csv"]))
    df.to_csv(emb["placements_csv"], index=False)  # persist region column

    # 3. structural metrics
    summary = analyze.summarize(df, studio_only=True)

    # 4. figures
    figures: dict[str, str] = {}
    if make_figures:
        metrics_df = pd.DataFrame(summary["metrics"])
        figures["trajectory"] = plots.plot_trajectory(
            df, os.path.join(out_dir, f"{prefix}_sound_trajectory.png"),
            artist=artist_name, metrics=metrics_df)
        figures["atlas"] = plots.plot_atlas(
            df, os.path.join(out_dir, f"{prefix}_atlas_scatter.png"),
            artist=artist_name, label_tracks=label_tracks)

    # 5. blurbs (optional)
    blurbs: dict[str, str] = {}
    if llm_fn is not None:
        anchors = _blurb.neighbor_anchors(df, anchor_tracks) if anchor_tracks else ""
        blurbs["technical"] = _blurb.technical_blurb(
            summary, llm_fn, artist=artist_name, caveats=caveats, anchors=anchors)
        blurbs["fan"] = _blurb.fan_blurb(
            summary, llm_fn, artist=artist_name, caveats=caveats)
        for kind, text in blurbs.items():
            with open(os.path.join(out_dir, f"{prefix}_blurb_{kind}.md"), "w") as fh:
                fh.write(text)

    result = {
        "artist": artist_name,
        "artist_key": _norm(artist_name),
        "summary": summary,
        "figures": figures,
        "data": {"discography_csv": disco_csv,
                 "placements_csv": emb["placements_csv"],
                 "embeddings_npz": emb["embeddings_npz"]},
        "blurbs": blurbs,
    }

    # 6. persist onto the artist profile node (additive, provenance-stamped)
    if store is not None:
        _persist(store, result)

    return result


def _persist(store, result: dict[str, Any]) -> None:
    """Upsert the sonic_trajectory block onto the artist's profile."""
    from anther_ml.artist_enrichment.schema import (ArtistProfile, STATUS_PARTIAL)

    key = result["artist_key"]
    prof = store.get(key) or ArtistProfile(artist_key=key, name=result["artist"])
    s = result["summary"]
    prof.sonic_trajectory = {
        "n_tracks": s["n_tracks"],
        "n_distinct_clusters": s["n_distinct_clusters"],
        "mean_cluster_conf": s["mean_cluster_conf"],
        "metrics": s["metrics"],
        "shifts": s["shifts"],
        "biggest_shift": s["biggest_shift"],
        "figures": result["figures"],
        "data": result["data"],
        "blurbs": result["blurbs"],
        "source": SOURCE,
        "updated_at": _today(),
    }
    prof.updated_at = _today()
    if prof.status == "unmatched":
        prof.status = STATUS_PARTIAL
    store.upsert(prof)
