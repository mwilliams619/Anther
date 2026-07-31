"""
Genre-free evaluation harness (Workstream F).

There is deliberately **no genre metric** here. Genre is an editorial label that
cuts across acoustic similarity, so it is never a target, signal, or score.
Instead we evaluate with ground truth that needs no labels:

1. Known-related-pairs self-retrieval (primary) — a song's alternate mixes /
   versions / stems should rank near the top of its neighbors. Reports
   recall@1/@5/@10 and mean reciprocal rank.
2. Cluster stability under resampling — refit clustering on bootstrap
   subsamples and measure label agreement (Adjusted Rand Index).
3. Internal cluster quality — silhouette (on the clustering-space embedding,
   not the 2D viz) and the size distribution.
4. Neighbor-audit export — dump top-k neighbors for seed tracks to CSV/HTML so
   a human can spot-listen ("do these sound alike?").

Run as:  python -m anther_ml.eval --index models/index_phase1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .similarity import SongIndex


# --------------------------------------------------------------------------- #
# Related groups (ground truth for self-retrieval)
# --------------------------------------------------------------------------- #

_VERSION_TOKENS = re.compile(
    r"\b(v\d+|version|vocals?|stems?|instrumental|inst|rough|mix|master|"
    r"remaster|edit|demo|take\d*|final|clean|dirty|acoustic|live|"
    r"copy|\d+)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def title_stem(name: str) -> str:
    """
    Normalize a track title to a stem shared by its alternates, so
    'Mic Check', 'Mic Check vocals', 'Mic Check vocals_1' collapse together.
    Strips extensions, version/mix tokens, trailing numbers, and punctuation.
    """
    name = Path(str(name)).stem.lower()
    # Normalize punctuation first so tokens like 'vocals_1' split into words
    # ('_' is a regex word char, so it would otherwise defeat \b boundaries).
    name = _NON_ALNUM.sub(" ", name)
    name = _VERSION_TOKENS.sub(" ", name)
    return " ".join(name.split()).strip()


def auto_related_groups(names: list[str]) -> dict[str, list[int]]:
    """
    Seed related groups by title-stem matching. Returns {stem: [indices]} for
    stems shared by ≥2 tracks. A starting point for a hand-labeled
    related_pairs.json, not a replacement for it.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(names):
        stem = title_stem(name)
        if stem:
            groups[stem].append(i)
    return {stem: idx for stem, idx in groups.items() if len(idx) > 1}


def load_related_groups(
    path: str | Path, names: list[str]
) -> dict[str, list[int]]:
    """
    Load hand-labeled related groups from JSON and map names → corpus indices.
    JSON format: {"group_name": ["Track A", "Track B", ...], ...}. Names are
    matched by exact title or by title stem. Unmatched names are skipped.
    """
    with open(path) as f:
        raw = json.load(f)
    by_name = {str(n): i for i, n in enumerate(names)}
    by_stem: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(names):
        by_stem[title_stem(n)].append(i)

    groups: dict[str, list[int]] = {}
    for group_name, members in raw.items():
        idxs: list[int] = []
        for m in members:
            if m in by_name:
                idxs.append(by_name[m])
            else:
                idxs.extend(by_stem.get(title_stem(m), []))
        idxs = sorted(set(idxs))
        if len(idxs) > 1:
            groups[group_name] = idxs
    return groups


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _cosine_rankings(embeddings: np.ndarray) -> np.ndarray:
    """
    Full N×N neighbor ranking (excluding self). embeddings are assumed already
    standardized + L2-normalized (as SongIndex stores them), so dot == cosine.
    Returns an (N, N-1) int array: row i is the indices of the other tracks
    sorted by descending similarity to i.
    """
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # never retrieve self
    order = np.argsort(-sims, axis=1)
    return order[:, : embeddings.shape[0] - 1]


def self_retrieval(
    embeddings: np.ndarray,
    groups: dict[str, list[int]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """
    For each track that has ≥1 known relative, is a relative in its top-k?
    Reports recall@k (fraction of query tracks with a relative in top-k) and
    mean reciprocal rank of the first relative.
    """
    rankings = _cosine_rankings(embeddings)
    relatives: dict[int, set[int]] = defaultdict(set)
    for members in groups.values():
        for m in members:
            relatives[m].update(x for x in members if x != m)

    query_tracks = [t for t, rel in relatives.items() if rel]
    if not query_tracks:
        return {"n_queries": 0, "mrr": float("nan"),
                **{f"recall@{k}": float("nan") for k in ks}}

    hits = {k: 0 for k in ks}
    rr_total = 0.0
    for t in query_tracks:
        rel = relatives[t]
        ranked = rankings[t]
        first_rank = None
        for rank, neighbor in enumerate(ranked, 1):
            if neighbor in rel:
                first_rank = rank
                break
        if first_rank is not None:
            rr_total += 1.0 / first_rank
            for k in ks:
                if first_rank <= k:
                    hits[k] += 1

    n = len(query_tracks)
    return {
        "n_queries": n,
        "mrr": rr_total / n,
        **{f"recall@{k}": hits[k] / n for k in ks},
    }


def size_distribution(labels: np.ndarray) -> dict:
    """n_clusters, noise fraction, largest-cluster share, and a size histogram."""
    labels = np.asarray(labels)
    n = len(labels)
    noise = int(np.sum(labels == -1))
    cluster_labels = labels[labels != -1]
    sizes: dict[int, int] = {}
    for c in sorted(set(cluster_labels.tolist())):
        sizes[int(c)] = int(np.sum(cluster_labels == c))
    largest = max(sizes.values()) if sizes else 0
    return {
        "n_clusters": len(sizes),
        "noise_frac": noise / n if n else 0.0,
        "largest_cluster_frac": largest / n if n else 0.0,
        "sizes": sizes,
    }


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette on the clustering-space embedding (noise excluded). NaN if
    fewer than 2 clusters remain."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return float("nan")
    return float(
        silhouette_score(embeddings[mask], labels[mask], metric="cosine")
    )


def cluster_stability(
    embeddings: np.ndarray,
    cluster_fn,
    frac: float = 0.8,
    n_seeds: int = 5,
    base_seed: int = 0,
) -> dict:
    """
    Refit clustering on bootstrap subsamples and measure label agreement (ARI)
    between each subsample and a full-data reference, on the shared tracks.
    Stable clusters reproduce; degenerate/noise-driven ones don't.
    ``cluster_fn(X) -> labels`` must accept an (M, D) array.
    """
    from sklearn.metrics import adjusted_rand_score

    n = embeddings.shape[0]
    reference = np.asarray(cluster_fn(embeddings))
    aris = []
    rng = np.random.default_rng(base_seed)
    for _ in range(n_seeds):
        idx = np.sort(rng.choice(n, size=int(frac * n), replace=False))
        sub_labels = np.asarray(cluster_fn(embeddings[idx]))
        aris.append(adjusted_rand_score(reference[idx], sub_labels))
    aris = np.asarray(aris)
    return {"mean_ari": float(aris.mean()), "std_ari": float(aris.std()),
            "n_seeds": n_seeds}


def neighbor_audit(
    embeddings: np.ndarray,
    names: list[str],
    seeds: list[int],
    k: int = 10,
) -> list[dict]:
    """Top-k neighbors for each seed track, as rows for CSV/HTML export."""
    rankings = _cosine_rankings(embeddings)
    sims = embeddings @ embeddings.T
    rows = []
    for s in seeds:
        for rank, neighbor in enumerate(rankings[s][:k], 1):
            rows.append({
                "seed": names[s],
                "rank": rank,
                "neighbor": names[neighbor],
                "score": round(float(sims[s, neighbor]), 4),
            })
    return rows


def write_audit(rows: list[dict], out_path: str | Path) -> None:
    """Write the neighbor audit to CSV and a sibling HTML table."""
    import csv

    out_path = Path(out_path)
    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "rank", "neighbor", "score"])
        writer.writeheader()
        writer.writerows(rows)

    html = ["<table border=1 cellpadding=4><tr><th>seed</th><th>rank</th>"
            "<th>neighbor</th><th>score</th></tr>"]
    for r in rows:
        html.append(
            f"<tr><td>{r['seed']}</td><td>{r['rank']}</td>"
            f"<td>{r['neighbor']}</td><td>{r['score']}</td></tr>"
        )
    html.append("</table>")
    with open(out_path.with_suffix(".html"), "w") as f:
        f.write("\n".join(html))


# --------------------------------------------------------------------------- #
# Scorecard / CLI
# --------------------------------------------------------------------------- #

def build_scorecard(
    index: SongIndex,
    related_pairs: str | Path | None = None,
    labels: np.ndarray | None = None,
) -> dict:
    names = [m.get("name", str(i)) for i, m in enumerate(index.metadata)]
    if related_pairs and Path(related_pairs).exists():
        groups = load_related_groups(related_pairs, names)
        group_source = str(related_pairs)
    else:
        groups = auto_related_groups(names)
        group_source = "auto (title-stem)"

    card: dict = {
        "n_tracks": len(names),
        "standardized": index.standardize,
        "config": index.config,
        "related_group_source": group_source,
        "n_related_groups": len(groups),
        "self_retrieval": self_retrieval(index.embeddings, groups),
    }
    if labels is not None:
        card["size_distribution"] = size_distribution(labels)
        card["silhouette"] = silhouette(index.embeddings, labels)
    return card


def _print_scorecard(card: dict) -> None:
    print("=" * 60)
    print("anther-ml eval scorecard  (genre-free)")
    print("=" * 60)
    print(f"tracks            : {card['n_tracks']}")
    print(f"standardized      : {card['standardized']}")
    print(f"related groups    : {card['n_related_groups']} "
          f"({card['related_group_source']})")
    sr = card["self_retrieval"]
    print("-- self-retrieval (known related pairs) --")
    print(f"  n_queries       : {sr['n_queries']}")
    for k in (1, 5, 10):
        print(f"  recall@{k:<2}      : {sr.get(f'recall@{k}', float('nan')):.3f}")
    print(f"  MRR             : {sr['mrr']:.3f}")
    if "size_distribution" in card:
        sd = card["size_distribution"]
        print("-- clustering --")
        print(f"  n_clusters      : {sd['n_clusters']}")
        print(f"  noise_frac      : {sd['noise_frac']:.3f}")
        print(f"  largest_frac    : {sd['largest_cluster_frac']:.3f}")
        print(f"  silhouette      : {card['silhouette']:.3f}")
        if sd["largest_cluster_frac"] > 0.60:
            print("  WARNING: largest cluster > 60% — likely degenerate")
    print("=" * 60)


def main(argv=None):
    p = argparse.ArgumentParser(description="Genre-free similarity/clustering eval")
    p.add_argument("--index", required=True, help="index path (no extension)")
    p.add_argument("--related-pairs", default=None,
                   help="related_pairs.json (else auto title-stem grouping)")
    p.add_argument("--labels", default=None, help="cluster labels .npy (optional)")
    p.add_argument("--audit-out", default=None,
                   help="write neighbor audit CSV/HTML to this path (no ext)")
    p.add_argument("--audit-k", type=int, default=10)
    p.add_argument("--json", action="store_true", help="print scorecard as JSON")
    args = p.parse_args(argv)

    index = SongIndex.load(args.index)
    labels = np.load(args.labels) if args.labels else None
    card = build_scorecard(index, args.related_pairs, labels)

    if args.json:
        print(json.dumps(card, indent=2))
    else:
        _print_scorecard(card)

    if args.audit_out:
        names = [m.get("name", str(i)) for i, m in enumerate(index.metadata)]
        seeds = list(range(min(5, len(names))))
        write_audit(neighbor_audit(index.embeddings, names, seeds, args.audit_k),
                    args.audit_out)
        print(f"neighbor audit → {args.audit_out}.csv / .html")
    return card


if __name__ == "__main__":
    main()
