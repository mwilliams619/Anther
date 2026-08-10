"""Sound-over-time analysis: place an artist's discography onto the frozen
100k-song sonic atlas and read the trajectory of their sound across releases.

Genre labels are never a model input — every trajectory claim is grounded in
where the audio lands (MERT + MERIT embeddings) and in label-independent
structural metrics (spread, dispersion, centroid shifts).

Typical use::

    from anther_ml.sonic_trajectory import run_artist, EraRule

    rules = [EraRule("lemonade", "Lemonade (2016)"),
             EraRule("renaissance", "Renaissance (2022)")]
    res = run_artist("Beyoncé", era_rules=rules, out_dir="results/beyonce",
                     llm_fn=my_llm_callable)     # llm_fn optional
"""

from .discography import EraRule, resolve_discography, era_mapper
from .analyze import add_region, era_metrics, centroid_shifts, summarize
from .regions import REGION_MAP, region_for
from .embed import embed_and_place
from .plots import plot_trajectory, plot_atlas
from .blurb import build_brief, technical_blurb, fan_blurb, neighbor_anchors
from .pipeline import run_artist

__all__ = [
    "EraRule", "resolve_discography", "era_mapper",
    "add_region", "era_metrics", "centroid_shifts", "summarize",
    "REGION_MAP", "region_for",
    "embed_and_place", "plot_trajectory", "plot_atlas",
    "build_brief", "technical_blurb", "fan_blurb", "neighbor_anchors",
    "run_artist",
]
