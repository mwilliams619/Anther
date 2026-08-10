"""Canonical cluster_id -> coarse sonic region map for the frozen 100k corpus.

The Leiden cluster ids are a *stable property of the reference corpus* — they do
not change per artist — so this map ships with the code rather than being
re-derived on every run. Values are the interpretable region a cluster belongs
to; raw Leiden titles (which the corpus authors have not finished refining) are
deliberately NOT surfaced to end users. Any cluster id not listed here falls
back to ``UNKNOWN_REGION`` and should be reviewed for inclusion.

Provenance: consolidated from the Beyonce / Tyler / Lil Yachty placement runs
(2026-08), which between them exercised 31 distinct clusters. A couple of
clusters are genuinely context-straddling (e.g. 6 reads as rap for a rapper and
as Afro-Latin/dembow for a pop artist); the broader musical reading is used.
"""

from __future__ import annotations

UNKNOWN_REGION = "Other / Unmapped"

REGION_MAP: dict[int, str] = {
    0: "Hip-Hop / Rap",
    1: "Hip-Hop / Rap",
    4: "Jazz / Blues / Vintage",
    6: "Dembow / Afro-Latin",
    7: "Mainstream Pop / Vocal",
    8: "Country / Americana",
    9: "Dembow / Afro-Latin",
    10: "Country / Americana",
    13: "Drum & Bass / Bass",
    18: "Lo-Fi / Chillhop",
    19: "Choral / Orchestral",
    21: "House / Electro",
    24: "Techno / Club",
    25: "Country / Americana",
    27: "Country / Americana",
    28: "Ambient / Textural",
    31: "Reggae / Roots",
    34: "Choral / Orchestral",
    35: "Ballad / Sparse Vocal",
    37: "World / Regional",
    40: "Choral / Orchestral",
    49: "Ballad / Sparse Vocal",
    53: "Hip-Hop / Rap",
    58: "Hip-Hop / Rap",
    60: "Drum & Bass / Bass",
    61: "Hardcore / Rave",
    64: "House / Dance-Pop",
    69: "Spoken Word / ASMR",
    70: "Lo-Fi / Chillhop",
    74: "Comedy / Cabaret",
    79: "World / Regional",
}


def region_for(cluster_id: int) -> str:
    """Coarse region label for a corpus cluster id (never raises)."""
    return REGION_MAP.get(int(cluster_id), UNKNOWN_REGION)
