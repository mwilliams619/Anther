"""
CLI for building and querying reference corpora.

    python -m anther_ml.corpus build --source local --audio-dir DIR \
        --name smoke --limit 20
    python -m anther_ml.corpus place song.mp3 --corpus models/corpus_smoke
"""

import argparse
import json
import sys

from .build import MIN_TRACKS, build_corpus
from .bundle import ReferenceCorpus


def _add_build_parser(sub):
    p = sub.add_parser("build", help="embed a source and freeze a corpus bundle")
    p.add_argument("--source", choices=("fma", "local", "mpd"), default="fma")
    p.add_argument("--name", required=True, help="bundle name → models/corpus_<name>/")
    p.add_argument("--out", default="models", help="output parent directory")
    p.add_argument("--audio-dir", default=None,
                   help="audio folder (fma default: data/audio/fma_small; "
                        "required for --source local)")
    p.add_argument("--metadata-dir", default="data/fma_metadata")
    p.add_argument("--mpd-dir", default=None, help="MPD slice directory")
    p.add_argument("--sample-n", type=int, default=8000, help="MPD sample size")
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help=f"embed only the first N source tracks (smoke test; "
                        f"minimum {MIN_TRACKS})")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Leiden resolution (low → few broad clusters)")
    p.add_argument("--artist-cap", type=int, default=None,
                   help="max tracks per artist (design §3A hygiene)")
    p.add_argument("--dedupe-threshold", type=float, default=0.98)
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--index-standardize", action="store_true",
                   help="z-score the SongIndex (default off for MERT)")
    p.add_argument("--whiten", action="store_true",
                   help="PCA whitening in the clustering space (design §3D)")
    p.add_argument("--no-loudnorm", action="store_true",
                   help="skip EBU R128 loudness normalization (not recommended)")
    p.add_argument("--layer-agg", choices=("mean", "last"), default="mean")
    p.add_argument("--pca", type=int, default=100, dest="n_pca_components")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true",
                   help="discard any embed checkpoint instead of resuming")


def _add_place_parser(sub):
    p = sub.add_parser("place", help="place one song onto a frozen corpus")
    p.add_argument("audio", help="audio file to place")
    p.add_argument("--corpus", required=True, help="bundle dir, e.g. models/corpus_smoke")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--playlists", action="store_true",
                   help="rank fit against every corpus playlist")
    p.add_argument("--playlist", default=None,
                   help="calibrated fit against one playlist (name or pid)")
    p.add_argument("--fit-k", type=int, default=5)
    p.add_argument("--n-null", type=int, default=2000)
    p.add_argument("--json", action="store_true", dest="as_json")


def _make_source(args):
    from . import sources

    if args.source == "fma":
        return sources.fma_source(
            audio_dir=args.audio_dir or "data/audio/fma_small",
            metadata_dir=args.metadata_dir,
            artist_cap=args.artist_cap,
            seed=args.seed,
        )
    if args.source == "local":
        if not args.audio_dir:
            sys.exit("--source local requires --audio-dir")
        return sources.local_source(args.audio_dir)
    if not args.mpd_dir:
        sys.exit("--source mpd requires --mpd-dir")
    return sources.mpd_source(
        args.mpd_dir,
        sample_n=args.sample_n,
        tracks_per_artist_cap=args.artist_cap or 5,
        max_slices=args.max_slices,
        seed=args.seed,
    )


def _cmd_build(args) -> None:
    if args.limit is not None and args.limit < MIN_TRACKS:
        sys.exit(f"--limit must be >= {MIN_TRACKS} (clustering floor)")
    build_corpus(
        _make_source(args),
        name=args.name,
        out_dir=args.out,
        resolution=args.resolution,
        limit=args.limit,
        dedupe_threshold=None if args.no_dedupe else args.dedupe_threshold,
        index_standardize=args.index_standardize,
        whiten=args.whiten,
        layer_aggregation=args.layer_agg,
        loudness_normalize=not args.no_loudnorm,
        n_pca_components=args.n_pca_components,
        seed=args.seed,
        resume=not args.fresh,
    )


def _cmd_place(args) -> None:
    from .place import calibrate_fit, embed_query, place, rank_playlists

    corpus = ReferenceCorpus.load(args.corpus)
    vec = embed_query(args.audio, corpus)
    result = place(corpus, vec, top_k=args.top_k, knn_k=args.knn_k)
    if args.playlist:
        result["playlist_fit"] = calibrate_fit(
            corpus, vec, args.playlist, k=args.fit_k, n_null=args.n_null
        )
    if args.playlists:
        result["playlist_ranking"] = rank_playlists(
            corpus, vec, k=args.fit_k, n_null=args.n_null
        )

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    cluster = result["cluster"]
    print(f"\n=== {args.audio} → {args.corpus} ===")
    print(f"cluster {cluster['id']}  (confidence {cluster['confidence']:.0%}, "
          f"size {cluster['profile']['size']})")
    exemplars = ", ".join(
        e["name"] or "?" for e in cluster["profile"]["exemplars"][:3]
    )
    print(f"  cluster exemplars: {exemplars}")
    if result["coords_2d"]:
        x, y = result["coords_2d"]
        print(f"  2D coords (display only): ({x:.2f}, {y:.2f})")
    print(f"\ntop {args.top_k} neighbors:")
    for n in result["neighbors"]:
        artist = f" — {n['artist']}" if n.get("artist") else ""
        print(f"  {n['rank']:2d}. {n['score']:.3f}  {n.get('name')}{artist}")
    if "playlist_fit" in result:
        fit = result["playlist_fit"]
        print(f"\nfit to {args.playlist!r}: {fit['raw_fit']:.3f} "
              f"→ better than {fit['percentile']:.0f}% of the corpus "
              f"({fit['n_members']} members, {fit['n_null']} null tracks)")
    if "playlist_ranking" in result:
        print("\nplaylist ranking:")
        for r in result["playlist_ranking"][:10]:
            print(f"  {r['percentile']:5.1f}%  {r['raw_fit']:.3f}  "
                  f"{r['name']} ({r['n_tracks']} tracks)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m anther_ml.corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_build_parser(sub)
    _add_place_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "build":
        _cmd_build(args)
    else:
        _cmd_place(args)


if __name__ == "__main__":
    main()
