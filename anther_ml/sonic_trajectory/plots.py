"""Render the two deliverable figures from a placements table + era metrics.

Self-contained matplotlib styling (no external figure skill). Two figures:
- trajectory : Panel A stacked region-composition bars per studio era,
               Panel B entropy-spread line over time.
- atlas      : 2-D scatter of every track, colored by era.

Both use the Agg backend and save to PNG; nothing is shown interactively.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import add_region, era_metrics, is_non_studio

_GREY = "#5b5b5b"
_OTHER = "#cccccc"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": _GREY, "axes.labelcolor": _GREY,
        "xtick.color": _GREY, "ytick.color": _GREY, "text.color": "#222222",
        "legend.frameon": False,
    })


def _region_palette(regions: Sequence[str]) -> dict[str, Any]:
    """Dominant region gets a calm blue; others get saturated distinct hues."""
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    pal, i = {}, 0
    for r in regions:
        pal[r] = base[i % 20]
        i += 1
    return pal


def plot_trajectory(df: pd.DataFrame, out_path: str, *, artist: str = "",
                    metrics: pd.DataFrame | None = None) -> str:
    """Two-panel studio-era trajectory figure. Returns ``out_path``."""
    _style()
    if "region" not in df.columns:
        df = add_region(df)
    if metrics is None:
        metrics = era_metrics(df, studio_only=True)
    eras = list(metrics["era"])
    studio = df[df["era"].isin(eras)]

    # region ordering: overall-most-common first (stable stack order)
    reg_order = list(studio["region"].value_counts().index)
    pal = _region_palette(reg_order)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.05, right=0.82, top=0.84, bottom=0.16, wspace=0.62)

    # Panel A — stacked composition bars
    x = np.arange(len(eras))
    bottoms = np.zeros(len(eras))
    for reg in reg_order:
        vals = []
        for era in eras:
            g = studio[studio["era"] == era]
            vals.append((g["region"] == reg).sum() / max(len(g), 1))
        vals = np.array(vals)
        axA.bar(x, vals, bottom=bottoms, width=0.72, color=pal[reg], label=reg,
                edgecolor="white", linewidth=0.5)
        bottoms += vals
    for xi, era in zip(x, eras):
        n = int(metrics.loc[metrics["era"] == era, "n"].iloc[0])
        axA.text(xi, 0.015, f"{n}", ha="center", va="bottom", fontsize=7,
                 color="white", fontweight="bold")
    axA.set_xticks(x)
    axA.set_xticklabels([_short(e) for e in eras], rotation=35, ha="right", fontsize=7.5)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("share of album's tracks")
    axA.set_title("Where each album's tracks land in the atlas  (n = tracks, in bar)",
                  loc="left", pad=10)
    axA.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
               handletextpad=0.4, labelspacing=0.4, title="Sonic region",
               title_fontsize=7.5)

    # Panel B — entropy spread over time
    yr = metrics["year"].to_numpy()
    sb = metrics["spread_bits"].to_numpy()
    axB.plot(yr, sb, "-", color=_GREY, lw=1.4, zorder=1)
    axB.scatter(yr, sb, s=70, c=range(len(yr)), cmap="viridis", zorder=2,
                edgecolor="white", linewidth=0.8)
    for xi, yi, era in zip(yr, sb, eras):
        axB.annotate(_short(era), (xi, yi), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=6.5, color=_GREY)
    axB.set_xlabel("year"); axB.set_ylabel("sonic spread (bits)")
    axB.set_ylim(bottom=-0.1)
    axB.set_title("How varied each album is, over time", loc="left")

    title = f"{artist}: sound over time, read from audio alone" if artist else \
        "Sound over time, read from audio alone"
    fig.suptitle(f"{title} · MERT + MERIT on Anther's frozen 100k corpus",
                 fontsize=11, y=0.97, x=0.05, ha="left")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_atlas(df: pd.DataFrame, out_path: str, *, artist: str = "",
               label_tracks: Sequence[str] | None = None) -> str:
    """Scatter of every track on the 2-D atlas, colored by era."""
    _style()
    if "region" not in df.columns:
        df = add_region(df)
    d = df.dropna(subset=["x2d", "y2d"]).copy()
    all_eras = list(d["era"].unique())
    bucket_eras = [e for e in all_eras if is_non_studio(e)]
    studio_eras = [e for e in all_eras if not is_non_studio(e)]
    order = sorted(studio_eras, key=lambda e: d[d["era"] == e]["year"].min())
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(order)))
    ecolor = {e: cmap[i] for i, e in enumerate(order)}
    for e in bucket_eras:
        ecolor[e] = _OTHER

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    fig.subplots_adjust(left=0.04, right=0.74, top=0.93, bottom=0.07)
    for e in bucket_eras + order:
        m = d["era"] == e
        if not m.any():
            continue
        bg = is_non_studio(e)
        ax.scatter(d[m]["x2d"], d[m]["y2d"], s=30 if bg else 50, color=[ecolor[e]],
                   edgecolor="white", linewidth=0.6, alpha=0.5 if bg else 0.9,
                   label=_short(e), zorder=2 if bg else 3)
    if label_tracks:
        for t in label_tracks:
            hit = d[d["title"].str.lower().str.contains(t.lower(), regex=False, na=False)]
            if len(hit):
                r = hit.iloc[0]
                ax.annotate(str(r["title"])[:22], (r["x2d"], r["y2d"]), xytext=(6, 5),
                            textcoords="offset points", fontsize=6, color="#222",
                            arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("atlas dim 1  →"); ax.set_ylabel("atlas dim 2  →")
    who = f"{artist}'s discography" if artist else "Discography"
    ax.set_title(f"{who} on Anther's 2-D sonic atlas, colored by era", loc="left", pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=7,
              handletextpad=0.3, labelspacing=0.5, title="Era (chronological)",
              title_fontsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _short(era: str, n: int = 22) -> str:
    return era if len(era) <= n else era[: n - 1] + "…"
