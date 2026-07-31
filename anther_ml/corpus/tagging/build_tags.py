"""
Orchestrator: corpus bundle dir → tag artifacts
(MICROGENRE_TAGGING_BUILD_PLAN.md §6).

Corpus-side pipeline (no GPU): seed → fit → predict, all reading the bundle's
already-stored raw ``embeddings.npy`` — the corpus is never re-embedded.
The GPU enters only for ``embed-eval`` (embedding downloaded FMA audio with
the bundle's frozen recipe).

Artifacts written into the bundle dir (schema additive, format version
unchanged — loaders must tolerate their absence):

  tag_seeds.npz     Stage-A weight matrix (CSR parts) + confidence + vocab
  tag_support.json  per-genre seed support, tier contributions, coverage
  tag_probe.pkl     frozen TagProbe (scaler + heads + thresholds)
  tag_vocab.json    vocab, learnable flags, thresholds, internal-val report
  track_tags.json   per-track tags, aligned to metadata row order (idx)
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .crosswalk import (
    DEFAULT_CROSSWALK_PATH,
    build_crosswalk,
    crosswalk_labels,
    load_crosswalk,
    save_crosswalk,
)
from .evaluate import evaluate_probe, seed_agreement, write_report
from .probe import TagProbe, knn_smooth, predict_tags
from .vocab import DEFAULT_VOCAB_PATH, load_vocab
from .weak_labels import build_seed_labels, genre_support

SEEDS_FILE = "tag_seeds.npz"
SUPPORT_FILE = "tag_support.json"
PROBE_FILE = "tag_probe.pkl"
TAG_VOCAB_FILE = "tag_vocab.json"
TRACK_TAGS_FILE = "track_tags.json"


# -- bundle I/O (light: no leiden/umap import needed for tagging) -------------


def load_corpus_light(corpus_dir: str | Path) -> tuple[np.ndarray, list[dict], dict]:
    """(embeddings, metadata, manifest) straight off disk."""
    d = Path(corpus_dir)
    embeddings = np.load(d / "embeddings.npy")
    with open(d / "index.json") as f:
        metadata = json.load(f)["metadata"]
    with open(d / "manifest.json") as f:
        manifest = json.load(f)
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"{d}: embeddings rows {len(embeddings)} != metadata {len(metadata)}"
        )
    return embeddings, metadata, manifest


def load_seeds(
    corpus_dir: str | Path,
) -> tuple[sparse.csr_matrix, list[str], np.ndarray]:
    """(seed weight CSR, genre_names, confidence) from tag_seeds.npz."""
    z = np.load(Path(corpus_dir) / SEEDS_FILE)
    W = sparse.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    )
    return W, [str(g) for g in z["genres"]], z["confidence"]


# -- pipeline steps ------------------------------------------------------------


def run_seed(
    corpus_dir: str | Path,
    *,
    vocab_path: str | Path = DEFAULT_VOCAB_PATH,
    fuzzy: bool = False,
) -> dict:
    """Stage A on a bundle → tag_seeds.npz + tag_support.json."""
    d = Path(corpus_dir)
    _, metadata, _ = load_corpus_light(d)
    vocab = load_vocab(vocab_path)

    t0 = time.time()
    W, genre_names, confidence, tier_counts = build_seed_labels(
        metadata, vocab, fuzzy=fuzzy
    )
    support = genre_support(W, genre_names)

    np.savez_compressed(
        d / SEEDS_FILE,
        data=W.data, indices=W.indices, indptr=W.indptr,
        shape=np.asarray(W.shape), confidence=confidence,
        genres=np.asarray(genre_names),
    )
    n = W.shape[0]
    n_tagged = int((confidence > 0).sum())
    report = {
        "n_tracks": n,
        "n_tracks_with_seed": n_tagged,
        "coverage": round(n_tagged / n, 4),
        "n_genres_with_seed": sum(1 for c in support.values() if c > 0),
        "tier_track_genre_assignments": dict(tier_counts),
        "fuzzy": fuzzy,
        "seconds": round(time.time() - t0, 1),
        "per_genre_support": support,
    }
    with open(d / SUPPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"seed: {n_tagged}/{n} tracks ({report['coverage']:.1%}) carry >=1 seed; "
        f"{report['n_genres_with_seed']} genres hit; tiers={dict(tier_counts)}"
    )
    return report


def run_fit(
    corpus_dir: str | Path,
    *,
    min_support: int = 50,
    conf_threshold: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> TagProbe:
    """Stage B fit on the confident seeds → tag_probe.pkl + tag_vocab.json."""
    d = Path(corpus_dir)
    X, metadata, _ = load_corpus_light(d)
    W, genre_names, _ = load_seeds(d)
    artists = [row.get("artist") or "" for row in metadata]

    t0 = time.time()
    probe = TagProbe.fit(
        X, W, genre_names, artists,
        conf_threshold=conf_threshold, min_support=min_support,
        seed=seed, verbose=verbose,
    )
    probe.save(d / PROBE_FILE)
    learnable = set(probe.genre_names)
    with open(d / TAG_VOCAB_FILE, "w") as f:
        json.dump(
            {
                "genres": genre_names,
                "learnable": [g in learnable for g in genre_names],
                "thresholds": {
                    g: float(t)
                    for g, t in zip(probe.genre_names, probe.thresholds)
                },
                "fit_report": probe.fit_report,
            },
            f, indent=2,
        )
    r = probe.fit_report
    print(
        f"fit: {r['n_learnable']} learnable genres, "
        f"{r['n_train']} train / {r['n_val']} val confident tracks "
        f"(artist-stratified), internal val macro-F1 {r['val_macro_f1']} "
        f"[{time.time() - t0:.0f}s]"
    )
    return probe


def run_predict(
    corpus_dir: str | Path,
    *,
    knn_smooth_k: int = 0,
    alpha: float = 0.5,
    top_k: int = 3,
) -> list[dict]:
    """Tag all corpus tracks → track_tags.json (metadata row order)."""
    d = Path(corpus_dir)
    X, metadata, _ = load_corpus_light(d)
    W, genre_names, _ = load_seeds(d)
    probe = TagProbe.load(d / PROBE_FILE)

    t0 = time.time()
    probs = probe.predict_proba(X)
    if knn_smooth_k > 0:
        print(f"predict: kNN propagation k={knn_smooth_k} alpha={alpha}...")
        probs = knn_smooth(probs, X, k=knn_smooth_k, alpha=alpha)

    # Reuse predict_tags' assembly by passing precomputed probs through a
    # thin shim probe would complicate it; assemble here with the same rules.
    learnable = set(probe.genre_names)
    from .probe import SEED_MIN_WEIGHT

    out = []
    for i, row in enumerate(metadata):
        cands = [
            (probe.genre_names[j], float(probs[i, j]), "probe")
            for j in np.flatnonzero(probs[i] >= probe.thresholds)
        ]
        srow = W[i]
        for j, w in zip(srow.indices, srow.data):
            g = genre_names[j]
            if g not in learnable and w >= SEED_MIN_WEIGHT:
                cands.append((g, float(min(1.0, w / 2.0)), "seed"))
        cands.sort(key=lambda t: -t[1])
        out.append(
            {
                "idx": i,
                "id": row.get("id"),
                "tags": [
                    {"genre": g, "score": round(s, 4),
                     "primary": rank == 0, "source": src}
                    for rank, (g, s, src) in enumerate(cands[:top_k])
                ],
            }
        )

    with open(d / TRACK_TAGS_FILE, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n_tagged = sum(1 for r in out if r["tags"])
    print(
        f"predict: {n_tagged}/{len(out)} tracks tagged "
        f"({n_tagged / len(out):.1%}) → {d / TRACK_TAGS_FILE} "
        f"[{time.time() - t0:.0f}s]"
    )
    return out


def run_all(corpus_dir: str | Path, *, fuzzy: bool = False,
            knn_smooth_k: int = 10, **fit_kw) -> None:
    run_seed(corpus_dir, fuzzy=fuzzy)
    run_fit(corpus_dir, **fit_kw)
    run_predict(corpus_dir, knn_smooth_k=knn_smooth_k)


# -- FMA held-out validation (plan §5; GPU only for embed_eval) ----------------


def run_embed_eval(
    audio_dir: str | Path,
    meta_dir: str | Path,
    corpus_dir: str | Path,
    out: str | Path,
    *,
    subset: str = "medium",
    limit: int | None = None,
    checkpoint_every: int = 200,
) -> None:
    """Embed FMA audio with the corpus's frozen recipe + crosswalked labels.

    The only GPU-bound step. Checkpoints to <out>.ckpt.npz and resumes, like
    build_corpus. Refuses to run with a mismatched recipe (assert_compatible).
    """
    from ...data import (
        SUBSET_ORDER,
        get_audio_path,
        load_fma_genres,
        load_fma_tracks,
    )
    from ...embedding import embedding_config, get_embedding, load_mert
    from ..bundle import ReferenceCorpus

    out = Path(out)
    corpus = ReferenceCorpus.load(corpus_dir)
    cfg = corpus.embedding_config
    layer_aggregation = cfg.get("layer_aggregation", "mean")
    normalize = bool(cfg.get("loudness_normalize", True))
    corpus.assert_compatible(
        embedding_config(layer_aggregation=layer_aggregation, normalize=normalize)
    )
    vocab = load_vocab()

    tracks = load_fma_tracks(meta_dir)
    genres_df = load_fma_genres(meta_dir)
    id_to_name = genres_df["title"].to_dict()

    import pandas as pd

    subset_col = pd.Categorical(
        tracks[("set", "subset")], categories=SUBSET_ORDER, ordered=True
    )
    tracks = tracks[subset_col <= subset]

    xw_path = Path(DEFAULT_CROSSWALK_PATH)
    if xw_path.exists():
        mapping = load_crosswalk(xw_path)
    else:
        mapping = build_crosswalk(sorted(genres_df["title"]), vocab)
        save_crosswalk(mapping, xw_path)
        print(f"embed-eval: wrote crosswalk → {xw_path}")

    todo = []
    for tid, row in tracks.iterrows():
        fgs = [id_to_name.get(g) for g in row[("track", "genres_all")] or []]
        fgs = [g for g in fgs if g]
        if not any(mapping.get(g) for g in fgs):
            continue  # no crosswalked label → useless for eval
        path = get_audio_path(audio_dir, tid)
        if path.exists():
            todo.append((int(tid), path, fgs))
    if limit:
        todo = todo[:limit]
    if not todo:
        raise FileNotFoundError(
            f"no FMA audio with crosswalked labels under {audio_dir} "
            f"(subset={subset!r}) — is the audio downloaded?"
        )

    ckpt = out.with_suffix(out.suffix + ".ckpt.npz")
    done: dict[int, np.ndarray] = {}
    if ckpt.exists():
        z = np.load(ckpt)
        done = dict(zip(z["ids"].tolist(), z["X"]))
        print(f"embed-eval: resuming, {len(done)} already embedded")

    model, processor, device = load_mert()
    t0 = time.time()
    for n, (tid, path, _) in enumerate(todo, 1):
        if tid in done:
            continue
        try:
            done[tid] = get_embedding(
                model, processor, path, device,
                layer_aggregation=layer_aggregation, normalize=normalize,
            )
        except Exception as e:  # bad mp3s exist in FMA; skip, don't die
            print(f"  skip {path.name}: {e}")
            continue
        if len(done) % checkpoint_every == 0:
            _save_ckpt(ckpt, done)
            rate = n / max(time.time() - t0, 1)
            print(f"  {len(done)}/{len(todo)} embedded ({rate:.1f} tracks/s)")
    _save_ckpt(ckpt, done)

    kept = [(tid, fgs) for tid, _, fgs in todo if tid in done]
    X = np.stack([done[tid] for tid, _ in kept]).astype(np.float32)
    Y = crosswalk_labels([fgs for _, fgs in kept], mapping, vocab)
    np.savez_compressed(
        out, X=X, track_ids=np.asarray([tid for tid, _ in kept]),
        y_data=Y.data, y_indices=Y.indices, y_indptr=Y.indptr,
        y_shape=np.asarray(Y.shape), genres=np.asarray(vocab),
    )
    ckpt.unlink(missing_ok=True)
    print(f"embed-eval: {len(X)} tracks → {out}")


def _save_ckpt(path: Path, done: dict) -> None:
    ids = np.asarray(sorted(done))
    np.savez_compressed(path, ids=ids, X=np.stack([done[i] for i in ids]))


def run_evaluate(
    corpus_dir: str | Path, eval_npz: str | Path, *, k: int = 3
) -> dict:
    """Score the frozen probe on held-out FMA embeddings → report + chart."""
    d = Path(corpus_dir)
    probe = TagProbe.load(d / PROBE_FILE)
    z = np.load(eval_npz)
    Y = sparse.csr_matrix(
        (z["y_data"], z["y_indices"], z["y_indptr"]), shape=tuple(z["y_shape"])
    )
    report = evaluate_probe(
        probe, z["X"], Y, [str(g) for g in z["genres"]], k=k
    )

    seeds_path = d / SEEDS_FILE
    if seeds_path.exists():
        X, _, _ = load_corpus_light(d)
        W, genre_names, _ = load_seeds(d)
        report["corpus_seed_agreement"] = seed_agreement(
            probe, probe.predict_proba(X), W, genre_names
        )

    write_report(report, d / "tag_eval_report.json", d / "tag_eval_f1_by_genre.png")
    print(
        f"evaluate: macro-F1 {report['macro_f1']} micro-F1 {report['micro_f1']} "
        f"P@{k} {report[f'precision_at_{k}']} over "
        f"{report['n_genres_scored']} genres / {report['n_eval_tracks']} tracks "
        f"→ {d / 'tag_eval_report.json'}"
    )
    return report
