"""
Held-out tag metrics (MICROGENRE_TAGGING_BUILD_PLAN.md §5c).

**Walled off from anther_ml.eval** (docs/invariants.md): this module scores
the display-only tag probe against human genre labels. It must never import
anther_ml.eval, never feed build_scorecard, and its numbers must never be
used to tune embedding / SongIndex / Leiden / placement hyperparameters.
Two scoreboards, one wall — this is the tag probe's own scoreboard.

Reported per plan §4b: per-genre precision/recall/F1 **with support** plus
macro & micro averages and precision@k. A single aggregate hiding 20 dead
genres is a failure, not a success.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse


def evaluate_probe(
    probe,
    X_eval: np.ndarray,
    Y_true: sparse.csr_matrix,
    genre_names: list[str],
    *,
    k: int = 3,
) -> dict:
    """Score a frozen TagProbe on held-out embeddings + multi-hot labels.

    Only genres both learnable and present in Y_true's vocab are scored;
    genres with zero eval support are listed but excluded from the averages.
    """
    col = {g: j for j, g in enumerate(genre_names)}
    scored = [(i, col[g]) for i, g in enumerate(probe.genre_names) if g in col]
    if not scored:
        raise ValueError("no overlap between probe genres and eval label vocab")

    probs = probe.predict_proba(X_eval)
    Y = Y_true.tocsc()

    per_genre = {}
    tp_all = fp_all = fn_all = 0
    for pi, yj in scored:
        y_true = np.asarray(Y[:, yj].todense()).ravel() > 0
        support = int(y_true.sum())
        pred = probs[:, pi] >= probe.thresholds[pi]
        tp = int(np.sum(pred & y_true))
        fp = int(np.sum(pred & ~y_true))
        fn = int(np.sum(~pred & y_true))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        per_genre[probe.genre_names[pi]] = {
            "support": support,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-12), 4),
        }
        if support > 0:
            tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    with_support = {g: m for g, m in per_genre.items() if m["support"] > 0}
    micro_p = tp_all / max(tp_all + fp_all, 1)
    micro_r = tp_all / max(tp_all + fn_all, 1)

    # precision@k over tracks that have at least one true label among scored
    # genres: of each track's top-k probe genres, the fraction that are true.
    scored_pi = np.asarray([pi for pi, _ in scored])
    scored_yj = np.asarray([yj for _, yj in scored])
    Yd = np.asarray(Y[:, scored_yj].todense()) > 0
    has_label = Yd.any(axis=1)
    kk = min(k, len(scored_pi))
    topk = np.argsort(probs[:, scored_pi], axis=1)[:, ::-1][:, :kk]
    hits = np.take_along_axis(Yd, topk, axis=1)
    p_at_k = float(hits[has_label].mean()) if has_label.any() else 0.0

    return {
        "n_eval_tracks": int(len(X_eval)),
        "n_genres_scored": len(with_support),
        "n_genres_zero_support": len(per_genre) - len(with_support),
        "macro_precision": round(
            float(np.mean([m["precision"] for m in with_support.values()])), 4
        ),
        "macro_recall": round(
            float(np.mean([m["recall"] for m in with_support.values()])), 4
        ),
        "macro_f1": round(
            float(np.mean([m["f1"] for m in with_support.values()])), 4
        ),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12), 4),
        f"precision_at_{k}": round(p_at_k, 4),
        "per_genre": per_genre,
    }


def seed_agreement(
    probe, probs: np.ndarray, seed_weights: sparse.csr_matrix,
    full_genre_names: list[str],
) -> dict:
    """Coherence check (plan §5c): agreement between probe predictions and the
    Stage-A playlist seeds on the corpus itself. Not ground truth — a probe
    that agrees 100% learned the noise; one that agrees 0% learned nothing."""
    col = {g: j for j, g in enumerate(full_genre_names)}
    recalls = []
    for pi, g in enumerate(probe.genre_names):
        y_seed = np.asarray(
            (seed_weights[:, col[g]] > 0).todense()
        ).ravel()
        if y_seed.sum() == 0:
            continue
        pred = probs[:, pi] >= probe.thresholds[pi]
        recalls.append(float(np.sum(pred & y_seed) / y_seed.sum()))
    return {
        "n_genres": len(recalls),
        "mean_seed_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
    }


def write_report(report: dict, out_json: str | Path, out_png: str | Path | None) -> None:
    """tag_eval_report.json + per-genre F1 bar chart, support annotated."""
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    if out_png is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        ((g, m) for g, m in report["per_genre"].items() if m["support"] > 0),
        key=lambda gm: -gm[1]["f1"],
    )
    if not rows:
        return
    names = [g for g, _ in rows]
    f1s = [m["f1"] for _, m in rows]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.28 * len(rows))))
    ax.barh(range(len(rows)), f1s, color="#4C72B0")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{g}  (n={m['support']})" for g, m in rows], fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("held-out F1")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Tag probe F1 by genre — macro {report['macro_f1']:.2f}, "
        f"micro {report['micro_f1']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
