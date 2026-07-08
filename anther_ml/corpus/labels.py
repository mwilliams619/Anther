"""
Playlist-name cluster labels (PLAYLIST_LABELS_BUILD_PLAN.md, Scheme 2).

Pure post-processing over an already-built bundle: tally the playlist names of
each cluster's members, fold near-duplicate names, weight by TF-IDF across
clusters so ubiquitous names ("favorites") sink and concentrated ones
("death metal") surface, and join the top few into a display label.

Labels are the same kind of object as `top_genres` — a display-only read-out
of the frozen map. They never feed back into embeddings, the SongIndex, the
Leiden partition, or eval (docs/invariants.md).

Profile fields added (all optional — readers must .get() them):
  top_tags          [[display_name, count], ...] normalized playlist-name tally
  tag_entropy       Shannon entropy (nats) of the normalized name distribution
  tag_entropy_norm  entropy / log(n_distinct) — comparable across vocab sizes
  label             auto draft, e.g. "Texas Country / Red Dirt / Dansband"
  label_source      "playlist_tfidf" | "none"
  label_override    human-authored name — authoritative, survives re-labeling
  label_override_at ISO timestamp of the override
  label_final       label_override if set else label — what readers display
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Generic names so common that TF-IDF alone still lets them leak into labels.
# Matched on the normalized key.
STOPWORD_NAMES = frozenset(
    {
        "favorites",
        "favourites",
        "my playlist",
        "liked songs",
        "new playlist",
        "untitled",
        "music",
        "songs",
        "playlist",
    }
)

_WS_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[0-9a-z]")


def normalize_playlist_name(name: str) -> str | None:
    """Fold near-duplicate playlist names to one key.

    Lowercase, strip, collapse internal whitespace, and trim leading/trailing
    non-alphanumeric runs (punctuation, emoji). Returns None for names with no
    usable signal: empty, pure emoji/punctuation, or a single character.
    """
    if not name:
        return None
    key = _WS_RE.sub(" ", str(name).strip().lower())
    # Trim non-alphanumeric decoration from both ends, keep interior intact.
    start, end = 0, len(key)
    while start < end and not _ALNUM_RE.match(key[start]):
        start += 1
    while end > start and not _ALNUM_RE.match(key[end - 1]):
        end -= 1
    key = key[start:end].strip()
    if len(key) < 2:
        return None
    return key


def tag_entropy(normalized_counts: Counter) -> tuple[float, float]:
    """Shannon entropy (natural log) of a cluster's normalized playlist-name
    distribution. Low = coherent (few dominant names), high = mixed.

    Returns (entropy, normalized_entropy) where normalized_entropy is
    entropy / log(n_distinct) so clusters of different vocab size compare;
    it is 0.0 when there are fewer than two distinct names.
    """
    total = sum(normalized_counts.values())
    if total <= 0:
        return 0.0, 0.0
    entropy = 0.0
    for count in normalized_counts.values():
        p = count / total
        entropy -= p * math.log(p)
    n_distinct = len(normalized_counts)
    norm = entropy / math.log(n_distinct) if n_distinct > 1 else 0.0
    return entropy, norm


def _cluster_name_tallies(
    metadata: list[dict], labels: np.ndarray
) -> dict[int, tuple[Counter, dict[str, Counter]]]:
    """Per cluster: (Counter of normalized names, per-key Counter of raw
    display variants) — the display form of a key is its most common raw
    spelling."""
    labels = np.asarray(labels)
    tallies: dict[int, tuple[Counter, dict[str, Counter]]] = {}
    for cid in sorted(int(c) for c in set(labels.tolist()) if c != -1):
        counts: Counter = Counter()
        displays: dict[str, Counter] = {}
        for i in np.flatnonzero(labels == cid):
            for pl in metadata[i].get("playlists") or []:
                key = normalize_playlist_name(pl.get("name") or "")
                if key is None:
                    continue
                counts[key] += 1
                displays.setdefault(key, Counter())[pl.get("name")] += 1
        tallies[int(cid)] = (counts, displays)
    return tallies


def generate_cluster_labels(
    metadata: list[dict],
    labels: np.ndarray,
    *,
    top_k_tags: int = 8,
    label_n: int = 3,
    min_support: int = 3,
) -> list[dict]:
    """Compute per-cluster playlist-name tallies and a draft display label.

    Returns one dict per cluster_id:
        {"cluster_id", "top_tags", "tag_entropy", "tag_entropy_norm",
         "label", "label_source"}

    The label is the top `label_n` normalized names by TF-IDF score (count >=
    min_support, stopwords excluded), joined by ' / ' and title-cased. A
    cluster with no usable playlist names gets label="" and
    label_source="none" — never a fabricated name.
    """
    labels = np.asarray(labels)
    tallies = _cluster_name_tallies(metadata, labels)
    n_clusters = len(tallies)
    cluster_sizes = {
        cid: int(np.sum(labels == cid)) for cid in tallies
    }
    df: Counter = Counter()  # clusters containing each normalized name
    for counts, _ in tallies.values():
        df.update(counts.keys())

    out = []
    for cid, (counts, displays) in tallies.items():
        entropy, entropy_norm = tag_entropy(counts)

        def display(key: str) -> str:
            return displays[key].most_common(1)[0][0]

        top_tags = [
            [display(key), int(count)]
            for key, count in counts.most_common(top_k_tags)
        ]

        size = cluster_sizes[cid] or 1
        scored = []
        for key, count in counts.items():
            if key in STOPWORD_NAMES or count < min_support:
                continue
            tf = count / size
            idf = math.log((n_clusters + 1) / (1 + df[key])) + 1
            scored.append((tf * idf, count, key))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

        picks = [key for _, _, key in scored[:label_n]]
        if picks:
            label = " / ".join(display(key).title() for key in picks)
            source = "playlist_tfidf"
        else:
            label = ""
            source = "none"
        out.append(
            {
                "cluster_id": cid,
                "top_tags": top_tags,
                "tag_entropy": float(entropy),
                "tag_entropy_norm": float(entropy_norm),
                "label": label,
                "label_source": source,
            }
        )
    return out


def apply_labels_to_profiles(
    profiles: list[dict], generated: list[dict]
) -> list[dict]:
    """Merge generated label fields into existing profiles by cluster_id.

    Adds/overwrites the auto-derived fields (top_tags, tag_entropy,
    tag_entropy_norm, label, label_source) and recomputes label_final, but a
    human label_override (and label_override_at) is authoritative and passes
    through untouched. Never mutates cluster_id/size/exemplars/
    top_playlists/top_genres. Returns the profiles list (mutated in place).
    """
    by_cid = {g["cluster_id"]: g for g in generated}
    for profile in profiles:
        gen = by_cid.get(profile["cluster_id"])
        if gen is None:
            continue
        for field in (
            "top_tags",
            "tag_entropy",
            "tag_entropy_norm",
            "label",
            "label_source",
        ):
            profile[field] = gen[field]
        profile["label_final"] = profile.get("label_override") or profile["label"]
    return profiles


def label_bundle(dir_path: str | Path, *, dry_run: bool = False, **kw) -> list[dict]:
    """Regenerate labels for a built bundle from its own metadata + labels,
    merge (preserving overrides), and rewrite cluster_profiles.json in place.

    Touches nothing else in the bundle — no re-embedding, no version bump.
    Returns the updated profiles. With dry_run=True, prints a table and
    writes nothing.
    """
    from .bundle import ReferenceCorpus

    d = Path(dir_path)
    corpus = ReferenceCorpus.load(d)
    generated = generate_cluster_labels(corpus.metadata, corpus.labels, **kw)
    profiles = apply_labels_to_profiles(corpus.profiles, generated)

    if dry_run:
        print(f"{'cluster':>7} | {'size':>6} | {'H_norm':>6} | label")
        print("-" * 72)
        for p in profiles:
            print(
                f"{p['cluster_id']:>7} | {p['size']:>6} | "
                f"{p.get('tag_entropy_norm', 0.0):>6.3f} | "
                f"{p.get('label_final') or '(unlabeled)'}"
            )
        return profiles

    with open(d / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    # Cheap integrity check: reload, verify(), and confirm every cluster with
    # playlist data carries a label_final.
    reloaded = ReferenceCorpus.load(d)
    for p in reloaded.profiles:
        if p.get("top_tags") and not p.get("label_final"):
            raise AssertionError(
                f"cluster {p['cluster_id']} has playlist tags but no label_final"
            )
    return profiles


def set_cluster_override(dir_path: str | Path, cluster_id: int, name: str) -> None:
    """Set a human label override for one cluster and persist it.

    The override is authoritative: label_final resolves to it until it is
    cleared (pass an empty name to clear).
    """
    from .bundle import ReferenceCorpus, utc_now_iso

    d = Path(dir_path)
    corpus = ReferenceCorpus.load(d)
    profile = corpus.cluster_profile(int(cluster_id))
    if name:
        profile["label_override"] = name
        profile["label_override_at"] = utc_now_iso()
    else:
        profile.pop("label_override", None)
        profile.pop("label_override_at", None)
    profile["label_final"] = profile.get("label_override") or profile.get("label", "")
    with open(d / "cluster_profiles.json", "w") as f:
        json.dump(corpus.profiles, f, indent=2)
