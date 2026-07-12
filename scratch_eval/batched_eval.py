"""
Memory-safe genre-free eval: batched self-retrieval + size_distribution + silhouette.
Avoids materializing the full N x (N-1) similarity ranking matrix (infeasible at
500k scale: 500000^2*4 bytes = 1 TB). Computes cosine similarity in row batches
against the full corpus, using argpartition for top-k retrieval instead of a
full argsort.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/matt/Dev/Anther")
from anther_ml.eval import auto_related_groups, load_related_groups, size_distribution, silhouette
from anther_ml.similarity import SongIndex


def batched_self_retrieval(embeddings, groups, ks=(1, 5, 10), batch_size=2000, silhouette_sample=None):
    n = embeddings.shape[0]
    relatives = defaultdict(set)
    for members in groups.values():
        for m in members:
            relatives[m].update(x for x in members if x != m)

    query_tracks = np.array([t for t, rel in relatives.items() if rel], dtype=np.int64)
    if len(query_tracks) == 0:
        return {"n_queries": 0, "mrr": float("nan"),
                **{f"recall@{k}": float("nan") for k in ks}}

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    emb_f32 = embeddings.astype(np.float32)

    for start in range(0, len(query_tracks), batch_size):
        batch_idx = query_tracks[start:start + batch_size]
        sims = emb_f32[batch_idx] @ emb_f32.T  # (B, N)
        # mask self-similarity
        for row, q in enumerate(batch_idx):
            sims[row, q] = -np.inf
        # top (max_k + a bit of slack) via argpartition, then figure out true rank
        # of the FIRST relative within full order -- need full rank position, so
        # we still need argsort per-row for the relative set only. We do this by
        # finding rank of each relative track directly via order statistics on
        # sims row rather than a full sort: count how many entries are >= the
        # relative's own similarity (that's its rank).
        for row, q in enumerate(batch_idx):
            rel = relatives[int(q)]
            row_sims = sims[row]
            rel_arr = np.fromiter(rel, dtype=np.int64)
            rel_sims = row_sims[rel_arr]
            best_rel_sim = rel_sims.max()
            # rank of the best relative = 1 + count of entries strictly greater
            rank = int(np.sum(row_sims > best_rel_sim)) + 1
            rr_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    nq = len(query_tracks)
    result = {
        "n_queries": int(nq),
        "mrr": rr_total / nq,
        **{f"recall@{k}": hits[k] / nq for k in ks},
    }
    return result


def build_scorecard_scalable(index_path, labels_path=None, related_pairs=None,
                              batch_size=2000, silhouette_sample=None):
    index = SongIndex.load(index_path)
    names = [m.get("name", str(i)) for i, m in enumerate(index.metadata)]
    if related_pairs and Path(related_pairs).exists():
        groups = load_related_groups(related_pairs, names)
        group_source = str(related_pairs)
    else:
        groups = auto_related_groups(names)
        group_source = "auto (title-stem)"

    t0 = time.time()
    sr = batched_self_retrieval(index.embeddings, groups, batch_size=batch_size)
    sr_time = time.time() - t0

    card = {
        "n_tracks": len(names),
        "standardized": index.standardize,
        "related_group_source": group_source,
        "n_related_groups": len(groups),
        "self_retrieval": sr,
        "self_retrieval_time_s": sr_time,
    }
    if labels_path is not None and Path(labels_path).exists():
        labels = np.load(labels_path)
        card["size_distribution"] = size_distribution(labels)
        # silhouette on cosine still needs pairwise distances internally (sklearn
        # computes NxN under the hood for metric='cosine' with array input) --
        # subsample for large N to keep it tractable.
        if silhouette_sample and len(labels) > silhouette_sample:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(labels), size=silhouette_sample, replace=False)
            card["silhouette"] = silhouette(index.embeddings[idx], labels[idx])
            card["silhouette_sampled_n"] = silhouette_sample
        else:
            card["silhouette"] = silhouette(index.embeddings, labels)
    return card


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--related-pairs", default=None)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--silhouette-sample", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    card = build_scorecard_scalable(args.index, args.labels, args.related_pairs,
                                     args.batch_size, args.silhouette_sample)
    print(json.dumps(card, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(card, f, indent=2)
