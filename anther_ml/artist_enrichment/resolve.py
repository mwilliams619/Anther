"""
Identity resolution (ARTIST_ENRICHMENT_PLAN.md §3).

Given a corpus artist plus candidate lists from each source, decide *which*
external entity it is — or refuse and send it to the review queue. Two identities
are resolved independently:

- **MusicBrainz** identity → origin, genres, release-derived labels.
- **Deezer** identity → profile image, fan count.

The guiding rule (plan §3, final line) is "do not enrich solely by name match":
we require a strong fuzzy name match *and* corroboration — MusicBrainz's own
relevance score for the MB pick, and cross-source agreement for the overall
confidence — and we treat near-tied top candidates as ambiguous rather than
guessing. This module is pure: candidates in, decision out, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import (STATUS_COMPLETE, STATUS_PARTIAL, STATUS_REVIEW,
                     STATUS_UNMATCHED)

# --- tunables (see docs; validated against the sample crawl) ---------------
MB_MIN_NAME_RATIO = 0.86      # below this, MusicBrainz pick is rejected
MB_AMBIGUOUS_MARGIN = 0.03    # top-2 name_ratio closer than this ⇒ maybe ambiguous…
MB_AMBIGUOUS_FLOOR = 0.86     # …only when the runner-up is itself a strong match…
MB_AMBIGUOUS_SCORE_RATIO = 0.90  # …and MB's own relevance can't separate them.
DEEZER_MIN_NAME_RATIO = 0.90  # Deezer image/fans need a near-exact name match


@dataclass
class Resolution:
    mb: dict[str, Any] | None = None       # accepted MusicBrainz candidate
    deezer: dict[str, Any] | None = None   # accepted Deezer candidate
    confidence: float = 0.0
    status: str = STATUS_UNMATCHED
    reason: str = ""
    review_candidates: list[dict[str, Any]] = field(default_factory=list)


def _pick_musicbrainz(cands: list[dict[str, Any]]) -> tuple[dict | None, float, bool, list]:
    """Return (choice, confidence, ambiguous, top_candidates).

    Confidence blends the fuzzy name match (dominant) with MusicBrainz's own
    relevance score. ``ambiguous`` is True when two strong candidates are nearly
    tied on name — the classic short/common-name failure mode (plan §8)."""
    if not cands:
        return None, 0.0, False, []
    ranked = sorted(cands, key=lambda c: (c.get("name_ratio", 0.0),
                                          c.get("mb_score", 0.0)), reverse=True)
    best = ranked[0]
    if best.get("name_ratio", 0.0) < MB_MIN_NAME_RATIO:
        return None, 0.0, False, ranked[:3]

    # Ambiguous only when the runner-up matches the name *just as well* AND
    # MusicBrainz's own relevance score can't separate them. MB commonly returns
    # the same famous artist alongside obscure same-named acts — there the top
    # hit scores ~100 and the namesakes ≤80, so the score ratio filter keeps
    # those out of the review queue while still parking genuine collisions
    # (two real, comparably-relevant artists sharing a name).
    ambiguous = False
    if len(ranked) >= 2:
        second = ranked[1]
        close_name = best["name_ratio"] - second.get("name_ratio", 0.0) < MB_AMBIGUOUS_MARGIN
        strong_second = second.get("name_ratio", 0.0) >= MB_AMBIGUOUS_FLOOR
        best_score = best.get("mb_score", 0.0)
        close_score = best_score <= 0 or \
            second.get("mb_score", 0.0) >= MB_AMBIGUOUS_SCORE_RATIO * best_score
        if close_name and strong_second and close_score:
            ambiguous = True

    conf = 0.7 * best["name_ratio"] + 0.3 * min(best.get("mb_score", 0.0), 100) / 100.0
    return best, round(conf, 4), ambiguous, ranked[:3]


def _pick_deezer(cands: list[dict[str, Any]]) -> tuple[dict | None, float]:
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda c: c.get("name_ratio", 0.0))
    if best.get("name_ratio", 0.0) < DEEZER_MIN_NAME_RATIO:
        return None, 0.0
    return best, round(float(best["name_ratio"]), 4)


def resolve(name: str,
            mb_candidates: list[dict[str, Any]],
            deezer_candidates: list[dict[str, Any]]) -> Resolution:
    """Resolve one corpus artist against its source candidates."""
    mb_choice, mb_conf, ambiguous, mb_top = _pick_musicbrainz(mb_candidates)
    dz_choice, dz_conf = _pick_deezer(deezer_candidates)

    # Ambiguous MusicBrainz identity: withhold the MB-derived fields and queue
    # the artist for a human decision. A confidently-matched Deezer identity may
    # still supply image/fans, since that's an independent match.
    if ambiguous:
        overall = dz_conf if dz_choice else round(mb_conf, 4)
        return Resolution(mb=None, deezer=dz_choice, confidence=overall,
                          status=STATUS_REVIEW,
                          reason="ambiguous MusicBrainz match (near-tied names)",
                          review_candidates=mb_top)

    if mb_choice is None and dz_choice is None:
        return Resolution(status=STATUS_UNMATCHED, reason="no confident match")

    # Overall confidence: weighted blend of whichever identities were accepted.
    parts, weights = [], []
    if mb_choice is not None:
        parts.append(mb_conf); weights.append(0.6)
    if dz_choice is not None:
        parts.append(dz_conf); weights.append(0.4)
    overall = round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 4)

    status = STATUS_COMPLETE if (mb_choice is not None and dz_choice is not None) \
        else STATUS_PARTIAL
    reason = "matched: " + ", ".join(
        s for s, c in (("musicbrainz", mb_choice), ("deezer", dz_choice)) if c)
    return Resolution(mb=mb_choice, deezer=dz_choice, confidence=overall,
                      status=status, reason=reason)
