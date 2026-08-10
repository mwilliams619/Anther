"""Build grounded fact briefs and (optionally) generate sound-over-time blurbs.

The module never hard-codes an LLM. Callers pass ``llm_fn(prompt, system) -> str``
(local Qwen in production, or any hosted model). ``build_brief`` produces the
structural fact sheet; ``technical_blurb`` / ``fan_blurb`` wrap it in prompts.

All facts are label-independent structural metrics (spread, dispersion,
centroid shifts, region shares) plus concrete nearest-neighbor anchors — never
raw Leiden cluster titles, which are unrefined.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

LLMFn = Callable[[str, str], str]

_TECH_SYSTEM = (
    "You are a music-data analyst writing for an audience that appreciates rigor. "
    "You are given structural measurements of an artist's discography, computed from "
    "audio embeddings placed onto a frozen 100,000-song reference atlas. Genre labels "
    "were NEVER an input. Write about the SHAPE of the catalog using the quantitative "
    "structure: how concentrated vs spread each album is, how far albums move from each "
    "other, where the turning points are. Do not invent cluster names or genre claims "
    "beyond the concrete landmarks given. Honor any noted data caveats. Be precise."
)

_FAN_SYSTEM = (
    "You are chatting casually with a music fan who has zero interest in technical or "
    "analytical jargon. No math words, no 'entropy', 'clusters', 'metrics', 'atlas', "
    "'data'. Just warm, vivid, everyday language about how the music changed. Ground it "
    "in two simple ideas: (1) how VARIED the songs on an album are, and (2) how FAR each "
    "album sits from the one before it. Keep it flowing and human."
)


def build_brief(summary: dict[str, Any], *, caveats: "list[str] | None" = None) -> str:
    """Render the structural summary (from analyze.summarize) as a text brief."""
    lines = []
    for m in summary["metrics"]:
        lines.append(
            f"{m['era']}: n={m['n']} tracks, {m['n_distinct_clusters']} atlas clusters, "
            f"spread {m['spread_bits']} bits, dispersion {m['dispersion']}, "
            f"largest region holds {int((m['top_region_share'] or 0)*100)}%"
        )
    brief = "Per-album structure (label-independent):\n" + "\n".join(lines)
    if summary.get("shifts"):
        sh = "\n".join(f"{s['from_era']} -> {s['to_era']}: {s['shift']}"
                       for s in summary["shifts"])
        brief += "\n\nAlbum-to-album movement (centroid distance; larger = bigger jump):\n" + sh
    if summary.get("biggest_shift"):
        b = summary["biggest_shift"]
        brief += (f"\n\nLargest single jump: {b['from_era']} -> {b['to_era']} "
                  f"({b['shift']}).")
    if caveats:
        brief += "\n\nCAVEATS (honor these; do not over-read):\n" + \
                 "\n".join(f"- {c}" for c in caveats)
    return brief


def neighbor_anchors(df: pd.DataFrame, tracks: "list[str]") -> str:
    """Pull nearest-neighbor names for a few landmark tracks (concrete anchors)."""
    out = []
    for t in tracks:
        hit = df[df["title"].str.lower().str.contains(t.lower(), regex=False, na=False)]
        if not len(hit):
            continue
        try:
            neigh = json.loads(hit.iloc[0]["neighbors"])
        except Exception:
            continue
        names = ", ".join(f"{n['artist']} - {n['title']}" for n in neigh[:3] if n.get("artist"))
        if names:
            out.append(f"'{hit.iloc[0]['title']}' nearest audio neighbors: {names}")
    return "\n".join(out)


def technical_blurb(summary: dict[str, Any], llm_fn: LLMFn, *, artist: str,
                    caveats=None, anchors: str = "", words: int = 220) -> str:
    brief = build_brief(summary, caveats=caveats)
    prompt = (
        f"Artist: {artist}.\n\n{brief}\n\n"
        + (f"Concrete audio landmarks (genre never a model input):\n{anchors}\n\n" if anchors else "")
        + f"Write a ~{words}-word analytical blurb with a single evocative title line, "
        "describing the artist's sound over time as read from audio structure alone. "
        "Anchor every claim in the numbers."
    )
    return llm_fn(prompt, _TECH_SYSTEM)


def fan_blurb(summary: dict[str, Any], llm_fn: LLMFn, *, artist: str,
              caveats=None, words: int = 360) -> str:
    brief = build_brief(summary, caveats=caveats)
    prompt = (
        f"Here's what the sound itself (not the labels) says about {artist}'s albums:\n\n"
        f"{brief}\n\n"
        f"Write ~{words} words of casual, flowing prose with a fun title line. Tell the "
        "story of how the artist changed, grounding it in how varied each album is and "
        "how big a leap it was from the last one. No jargon."
    )
    return llm_fn(prompt, _FAN_SYSTEM)
