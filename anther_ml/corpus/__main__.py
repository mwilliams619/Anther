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
from .hub import CORPUS_REPO_ENV


def _add_build_parser(sub):
    p = sub.add_parser("build", help="embed a source and freeze a corpus bundle")
    p.add_argument("--source", choices=("fma", "local", "mpd", "sql", "sql_oversampled"),
                   default="fma")
    p.add_argument("--name", required=True, help="bundle name → models/corpus_<name>/")
    p.add_argument("--out", default="models", help="output parent directory")
    p.add_argument("--audio-dir", default=None,
                   help="audio folder (fma default: data/audio/fma_small; "
                        "required for --source local)")
    p.add_argument("--metadata-dir", default="data/fma_metadata")
    p.add_argument("--mpd-dir", default=None, help="MPD JSON slice directory (--source mpd)")
    p.add_argument("--sql-dump", default=None,
                   help="MySQL MPD dump, e.g. data/mpd_dump/spotifydbdumpshare.sql "
                        "(--source sql; built into a .sqlite on first use)")
    p.add_argument("--sql-db", default=None,
                   help="prebuilt SQLite DB for --source sql (default: dump + .sqlite)")
    p.add_argument("--min-popularity", type=float, default=None,
                   help="drop tracks below this Spotify popularity before sampling "
                        "(--source sql; fewer candidates → faster sampling)")
    p.add_argument("--max-popularity", type=float, default=None,
                   help="drop tracks above this Spotify popularity before sampling "
                        "(--source sql; useful for long-tail expansion)")
    p.add_argument("--no-membership", action="store_true",
                   help="skip playlist membership when building the SQLite DB "
                        "(--source sql; much faster, disables playlist-fit)")
    p.add_argument("--sample-n", type=int, default=8000, help="MPD/SQL sample size")
    p.add_argument("--oversample-ratio", type=float, default=1.15,
                   help="--source sql_oversampled: candidate pool size as a "
                        "multiple of --sample-n, to absorb dead preview URLs "
                        "while still landing exactly --sample-n tracks")
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
    p.add_argument("--batch-tracks", type=int, default=16,
                   help="tracks embedded per GPU batch (Tier 1A; raise until VRAM ~80%%)")
    p.add_argument("--batch-windows", type=int, default=32,
                   help="max equal-length windows per GPU forward pass")
    p.add_argument("--no-fp16", action="store_true",
                   help="force fp32 inference (default: fp16 autocast on CUDA)")
    p.add_argument("--n-workers", type=int, default=12,
                   help="parallel preview-fetch workers (--source sql/mpd)")
    p.add_argument("--prefetch", type=int, default=64, dest="prefetch_size",
                   help="items buffered ahead of the GPU by the fetch producers")
    p.add_argument("--capture-merit-backbone", action="store_true",
                   help="additionally save merit_backbone.npy (5120-d, layers "
                        "3/4/5/6/23) from the same forward pass as the MERT-1024 "
                        "embedding, for downstream MERIT factor-head projection")


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


def _add_label_parser(sub):
    p = sub.add_parser(
        "label",
        help="(re)generate playlist-derived cluster labels for a built bundle",
    )
    p.add_argument("corpus", help="bundle dir, e.g. models/corpus_mpd_100k")
    p.add_argument("--dry-run", action="store_true",
                   help="print the label table, write nothing")
    p.add_argument("--set", nargs=2, metavar=("CLUSTER_ID", "NAME"),
                   dest="override",
                   help="set a human label override for one cluster")
    p.add_argument("--top-k-tags", type=int, default=8)
    p.add_argument("--label-n", type=int, default=3,
                   help="names joined into the auto label")
    p.add_argument("--min-support", type=int, default=3,
                   help="min track count for a name to enter the label")


def _add_publish_parser(sub):
    p = sub.add_parser(
        "publish",
        help="copy a built bundle into a redistributable one (strip playlist "
             "membership, drop the MERIT backbone, de-pickle the Leiden fit)",
    )
    p.add_argument("corpus", help="source bundle dir, e.g. models/corpus_mpd_100k_merit")
    p.add_argument("--out", required=True, help="destination bundle dir")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change and the size delta; write nothing")
    p.add_argument("--keep-playlists", action="store_true",
                   help="keep per-track playlist membership (MPD-derived — only "
                        "for bundles you are not redistributing)")
    p.add_argument("--keep-backbone", action="store_true",
                   help="keep merit_backbone.npy (needed only to rebuild the "
                        "MERIT sidecar or extend the corpus)")
    p.add_argument("--keep-pickle", action="store_true",
                   help="copy leiden.pkl as-is instead of converting to the "
                        "portable leiden.npz (keeps reducer_2d; pins Python 3.11)")
    p.add_argument("--keep-duplicate-metadata", action="store_true",
                   help="keep index_merit_agg.json's copy of the metadata "
                        "instead of pointing it at index.json")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-publish load-and-check pass")


def _add_push_parser(sub):
    p = sub.add_parser("push", help="upload a published bundle to a HF dataset repo")
    p.add_argument("corpus", help="published bundle dir (see `publish`)")
    p.add_argument("--repo-id", required=True, help="e.g. you/anther-corpus-mpd-100k")
    p.add_argument("--revision", default=None,
                   help="branch/tag to upload to (pin this in fetch)")
    p.add_argument("--public", action="store_true",
                   help="create the repo public (default: private)")
    p.add_argument("--message", default=None, help="commit message")


def _add_fetch_parser(sub):
    p = sub.add_parser("fetch", help="download a bundle from a HF dataset repo")
    p.add_argument("--repo-id", default=None,
                   help=f"e.g. you/anther-corpus-mpd-100k (default: ${CORPUS_REPO_ENV})")
    p.add_argument("--out", required=True, help="destination bundle dir")
    p.add_argument("--revision", default=None, help="branch/tag to pin")
    p.add_argument("--include-backbone", action="store_true",
                   help="also download merit_backbone.npy (only needed to "
                        "rebuild the MERIT sidecar or extend the corpus)")


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
    if args.source == "sql":
        if not (args.sql_db or args.sql_dump):
            sys.exit("--source sql requires --sql-dump (or a prebuilt --sql-db)")
        return sources.sql_source(
            db_path=args.sql_db,
            sql_dump=args.sql_dump,
            sample_n=args.sample_n,
            tracks_per_artist_cap=args.artist_cap or 5,
            seed=args.seed,
            min_popularity=args.min_popularity,
            max_popularity=args.max_popularity,
            with_membership=not args.no_membership,
            n_workers=args.n_workers,
        )
    if args.source == "sql_oversampled":
        if not (args.sql_db or args.sql_dump):
            sys.exit("--source sql_oversampled requires --sql-dump (or a prebuilt --sql-db)")
        return sources.sql_source_oversampled(
            db_path=args.sql_db,
            sql_dump=args.sql_dump,
            target_n=args.sample_n,
            oversample_ratio=args.oversample_ratio,
            tracks_per_artist_cap=args.artist_cap or 5,
            seed=args.seed,
            min_popularity=args.min_popularity,
            max_popularity=args.max_popularity,
            with_membership=not args.no_membership,
            n_workers=args.n_workers,
        )
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
        batch_tracks=args.batch_tracks,
        batch_windows=args.batch_windows,
        use_fp16=False if args.no_fp16 else None,
        prefetch_size=args.prefetch_size,
        capture_merit_backbone=args.capture_merit_backbone,
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


def _cmd_label(args) -> None:
    from .labels import label_bundle, set_cluster_override

    if args.override:
        cid, name = args.override
        set_cluster_override(args.corpus, int(cid), name)
        print(f"cluster {cid} override set to {name!r}")
        return
    profiles = label_bundle(
        args.corpus,
        dry_run=args.dry_run,
        top_k_tags=args.top_k_tags,
        label_n=args.label_n,
        min_support=args.min_support,
    )
    if not args.dry_run:
        n_labeled = sum(1 for p in profiles if p.get("label"))
        print(f"labeled {n_labeled}/{len(profiles)} clusters → "
              f"{args.corpus}/cluster_profiles.json")


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def _cmd_publish(args) -> None:
    from .publish import publish_bundle, verify_published

    report = publish_bundle(
        args.corpus,
        args.out,
        strip_playlists=not args.keep_playlists,
        drop_backbone=not args.keep_backbone,
        portable_leiden=not args.keep_pickle,
        dedupe_metadata=not args.keep_duplicate_metadata,
        dry_run=args.dry_run,
    )

    tag = "would publish" if args.dry_run else "published"
    print(f"\n=== {tag} {args.corpus} → {args.out} ===")
    before, after = report["bytes_before"], report["bytes_after"]
    saved = 1.0 - (after / before) if before else 0.0
    caveat = " + leiden.npz" if report.get("bytes_after_excludes_leiden") else ""
    print(f"  {_mb(before)} → {_mb(after)}{caveat}  ({saved:.0%} smaller)")

    for name, stats in report.get("rewritten_indices", {}).items():
        how = (f"deduped → {stats['deduped_to']}.json"
               if stats.get("deduped_to")
               else f"{stats.get('rows_stripped', 0):,} rows stripped")
        print(f"  {name}: {_mb(stats['bytes_before'])} → "
              f"{_mb(stats['bytes_after'])} ({how})")
    if "leiden" in report:
        ld = report["leiden"]
        if ld.get("unsized_in_dry_run"):
            print(f"  {ld['source']} ({_mb(ld['bytes_before'])}) → leiden.npz: "
                  "size not computed in --dry-run")
        else:
            note = " (reducer_2d dropped)" if ld["reducer_2d_dropped"] else ""
            print(f"  {ld['source']} → leiden.npz: {_mb(ld['bytes_before'])} → "
                  f"{_mb(ld['bytes_after'])}{note}")
    for name in report["excluded_files"]:
        print(f"  excluded: {name}")

    if args.dry_run or args.no_verify:
        return

    checks = verify_published(args.out, src_dir=args.corpus)
    print("\nverify:")
    for key, val in checks.items():
        print(f"  {key}: {val:.2e}" if isinstance(val, float) else f"  {key}: {val}")
    if not checks["transform_ok"]:
        sys.exit(
            "FAILED: the published bundle's frozen scaler/PCA disagree with the "
            "source's — a query would be assigned to a different cluster"
        )


def _cmd_push(args) -> None:
    from .hub import bundle_publish_state, push_bundle

    state = bundle_publish_state(args.corpus)
    if not state["published"]:
        sys.exit(
            "refusing to push an unpublished bundle:\n  - "
            + "\n  - ".join(state["reasons"])
            + "\nRun `python -m anther_ml.corpus publish "
            f"{args.corpus} --out <dst>` first."
        )

    url = push_bundle(
        args.corpus,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
        commit_message=args.message,
    )
    visibility = "public" if args.public else "private"
    rev = f" (revision {args.revision})" if args.revision else ""
    print(f"pushed {args.corpus} → {url} [{visibility}]{rev}")


def _cmd_fetch(args) -> None:
    from .bundle import ReferenceCorpus
    from .hub import fetch_bundle, resolve_repo_id

    repo_id = resolve_repo_id(args.repo_id)
    if repo_id is None:
        sys.exit(f"--repo-id is required (or set ${CORPUS_REPO_ENV})")

    out = fetch_bundle(
        repo_id, args.out,
        revision=args.revision,
        include_backbone=args.include_backbone,
    )
    corpus = ReferenceCorpus.load(out)
    print(f"fetched {repo_id} → {out} ({corpus.n_tracks:,} tracks, "
          f"{len(corpus.profiles)} clusters)")


def main(argv=None) -> None:
    import logging

    parser = argparse.ArgumentParser(prog="python -m anther_ml.corpus")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level for the build (DEBUG/INFO/WARNING)")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_build_parser(sub)
    _add_place_parser(sub)
    _add_label_parser(sub)
    _add_publish_parser(sub)
    _add_push_parser(sub)
    _add_fetch_parser(sub)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "build":
        _cmd_build(args)
    elif args.command == "label":
        _cmd_label(args)
    elif args.command == "publish":
        _cmd_publish(args)
    elif args.command == "push":
        _cmd_push(args)
    elif args.command == "fetch":
        _cmd_fetch(args)
    else:
        _cmd_place(args)


if __name__ == "__main__":
    main()
