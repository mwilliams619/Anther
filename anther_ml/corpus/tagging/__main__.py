"""
CLI for per-track micro-genre tagging (peer to corpus build/place/label).

    python -m anther_ml.corpus.tagging seed    --corpus models/corpus_corpus_mpd_100k
    python -m anther_ml.corpus.tagging fit     --corpus models/corpus_corpus_mpd_100k
    python -m anther_ml.corpus.tagging predict --corpus ... --knn-smooth 10
    python -m anther_ml.corpus.tagging all     --corpus ...
    python -m anther_ml.corpus.tagging embed-eval --audio-dir data/audio/fma_medium \
        --meta-dir data/fma_metadata --corpus ... --out data/fma_eval_embeddings.npz
    python -m anther_ml.corpus.tagging evaluate --corpus ... \
        --eval-embeddings data/fma_eval_embeddings.npz
"""

import argparse

from .build_tags import (
    run_all,
    run_embed_eval,
    run_evaluate,
    run_fit,
    run_predict,
    run_seed,
)
from .vocab import DEFAULT_VOCAB_PATH


def _corpus_arg(p):
    p.add_argument("--corpus", required=True,
                   help="bundle dir, e.g. models/corpus_corpus_mpd_100k")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m anther_ml.corpus.tagging")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed", help="Stage A: playlist-name weak labels")
    _corpus_arg(p)
    p.add_argument("--vocab", default=DEFAULT_VOCAB_PATH)
    p.add_argument("--fuzzy", action="store_true",
                   help="enable the gated fuzzy tier (needs rapidfuzz)")

    p = sub.add_parser("fit", help="Stage B: fit the tag probe on confident seeds")
    _corpus_arg(p)
    p.add_argument("--min-support", type=int, default=50,
                   help="confident train positives below this → genre not learnable")
    p.add_argument("--conf-threshold", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("predict", help="tag all corpus tracks → track_tags.json")
    _corpus_arg(p)
    p.add_argument("--knn-smooth", type=int, default=0, dest="knn_smooth_k",
                   help="k for label propagation in MERT space (0 = off)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="propagation blend weight")
    p.add_argument("--top-k", type=int, default=3)

    p = sub.add_parser("all", help="seed → fit → predict (corpus-side, no GPU)")
    _corpus_arg(p)
    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("--knn-smooth", type=int, default=10, dest="knn_smooth_k")
    p.add_argument("--min-support", type=int, default=50)

    p = sub.add_parser("embed-eval",
                       help="embed FMA audio with the frozen recipe (GPU)")
    _corpus_arg(p)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--meta-dir", default="data/fma_metadata")
    p.add_argument("--out", default="data/fma_eval_embeddings.npz")
    p.add_argument("--subset", choices=("small", "medium", "large"),
                   default="medium")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("evaluate", help="score the frozen probe on held-out FMA")
    _corpus_arg(p)
    p.add_argument("--eval-embeddings", required=True)
    p.add_argument("--k", type=int, default=3)

    args = ap.parse_args(argv)
    if args.cmd == "seed":
        run_seed(args.corpus, vocab_path=args.vocab, fuzzy=args.fuzzy)
    elif args.cmd == "fit":
        run_fit(args.corpus, min_support=args.min_support,
                conf_threshold=args.conf_threshold, seed=args.seed)
    elif args.cmd == "predict":
        run_predict(args.corpus, knn_smooth_k=args.knn_smooth_k,
                    alpha=args.alpha, top_k=args.top_k)
    elif args.cmd == "all":
        run_all(args.corpus, fuzzy=args.fuzzy,
                knn_smooth_k=args.knn_smooth_k, min_support=args.min_support)
    elif args.cmd == "embed-eval":
        run_embed_eval(args.audio_dir, args.meta_dir, args.corpus, args.out,
                       subset=args.subset, limit=args.limit)
    elif args.cmd == "evaluate":
        run_evaluate(args.corpus, args.eval_embeddings, k=args.k)


if __name__ == "__main__":
    main()
