"""
One-shot Billboard chart-era expansion pipeline.

Chains all four steps of the corpus-expansion workflow behind a single
command, parametrized by an arbitrary date range (a decade, a specific
span of years, or the rolling "last N years" window). Each run is scoped
under its own `--label` subdirectory so multiple eras can be pulled
independently (e.g. run once for the 70s, once for the 80s, once for the
90s) and then merged.

Steps run in order:
  1. 01_build_popularity_table.py  -- Hot 100 weekly data -> popularity table
                                       for the requested date range
  2. 02_match_against_corpus.py    -- split into already-in-corpus vs missing,
                                       write this era's popularity.json /
                                       popularity_by_track_id.json
  3. 03_fetch_and_embed_missing.py -- Deezer preview fetch + MERT embed for
                                       every missing track (GPU, parallel
                                       fetch workers)
  4. 04_append_to_index.py         -- extend_corpus(): additive append into
                                       the reference-corpus bundle, no refit,
                                       cluster IDs of existing tracks untouched

After step 4, this era's popularity_by_track_id.json entries are merged
(union) into the master data/billboard/popularity_by_track_id.json used by
the UI, so popularity reranking picks up newly-added eras without any
separate step.

Usage — pull an entire decade in one shot:
    python scripts/billboard_expansion/run_pipeline.py \\
        --label 1970s --start-date 1970-01-01 --end-date 1979-12-31

Usage — rolling window (same as the original --years mode):
    python scripts/billboard_expansion/run_pipeline.py --label last10 --years 10

Each run chains onto whatever bundle currently sits at --bundle (default:
the most recently extended bundle) and extends it *in place* (default
--out-dir == --bundle), so running this multiple times for different eras
accumulates into one growing corpus. Pass --out-dir to instead write a
fresh copy and leave --bundle untouched.

Resumable: step 3's output JSONL is checkpointed per-line, so a killed run
can be restarted with the same --label and it will skip already-embedded
tracks (see 03_fetch_and_embed_missing.py's own resume logic).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"step failed (exit {result.returncode}): {' '.join(cmd)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True,
                     help="short tag for this run, e.g. '1970s', '1980s', 'last10' -- "
                          "scopes intermediate files under data/billboard/<label>/")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD, e.g. 1970-01-01")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD, e.g. 1979-12-31 (default: today)")
    ap.add_argument("--years", type=int, default=None,
                     help="alternative to --start-date/--end-date: rolling window, last N years")
    ap.add_argument("--bundle", default="models/corpus_mpd_100k_merit_ext_billboard",
                     help="corpus bundle to match against and extend (chains onto prior runs by default)")
    ap.add_argument("--out-dir", default=None,
                     help="where to write the extended bundle (default: extend --bundle in place)")
    ap.add_argument("--hot100-cache", default="data/billboard/all_hot100.json")
    ap.add_argument("--master-popularity", default="data/billboard/popularity_by_track_id.json",
                     help="master sidecar the UI reads; this run's matches are merged into it")
    ap.add_argument("--fetch-workers", type=int, default=16)
    ap.add_argument("--window-size", type=int, default=300)
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--dedupe-threshold", type=float, default=0.98)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--skip-fetch-embed", action="store_true",
                     help="stop after step 2 (useful to preview match-rate before spending GPU time)")
    args = ap.parse_args()

    era_dir = REPO_ROOT / "data" / "billboard" / args.label
    era_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir or args.bundle

    # Step 1: popularity table for this era
    step1_cmd = [
        sys.executable, str(SCRIPT_DIR / "01_build_popularity_table.py"),
        "--cache", args.hot100_cache,
        "--out", str(era_dir / "popularity_songs.csv"),
    ]
    if args.start_date:
        step1_cmd += ["--start-date", args.start_date]
        if args.end_date:
            step1_cmd += ["--end-date", args.end_date]
    else:
        step1_cmd += ["--years", str(args.years if args.years is not None else 20)]
    run(step1_cmd)

    # Step 2: match against the (current) bundle, split matched/missing
    run([
        sys.executable, str(SCRIPT_DIR / "02_match_against_corpus.py"),
        "--bundle", args.bundle,
        "--popularity", str(era_dir / "popularity_songs.csv"),
        "--out-dir", str(era_dir),
    ])

    if args.skip_fetch_embed:
        print(f"\n--skip-fetch-embed set; stopping after match step. "
              f"Inspect {era_dir}/missing_tracks.csv before continuing.")
        sys.exit(0)

    missing_csv = era_dir / "missing_tracks.csv"
    n_missing = sum(1 for _ in open(missing_csv)) - 1
    if n_missing <= 0:
        print(f"\nno missing tracks for {args.label} -- every chart song already in the corpus.")
    else:
        # Step 3: fetch + embed missing tracks
        run([
            sys.executable, str(SCRIPT_DIR / "03_fetch_and_embed_missing.py"),
            "--missing", str(missing_csv),
            "--out-jsonl", str(era_dir / "embedded_tracks.jsonl"),
            "--fetch-workers", str(args.fetch_workers),
            "--window-size", str(args.window_size),
            "--embed-batch-size", str(args.embed_batch_size),
        ])

        # Step 4: additive corpus extension (extend_corpus, no refit)
        run([
            sys.executable, str(SCRIPT_DIR / "04_append_to_index.py"),
            "--bundle", args.bundle,
            "--embedded-jsonl", str(era_dir / "embedded_tracks.jsonl"),
            "--out-dir", out_dir,
            "--dedupe-threshold", str(args.dedupe_threshold),
            "--knn-k", str(args.knn_k),
        ])

    # Merge this era's popularity_by_track_id.json into the master sidecar
    era_pop_path = era_dir / "popularity_by_track_id.json"
    master_path = REPO_ROOT / args.master_popularity
    era_pop = json.load(open(era_pop_path)) if era_pop_path.exists() else {}
    master_pop = json.load(open(master_path)) if master_path.exists() else {}
    before = len(master_pop)
    master_pop.update(era_pop)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w") as f:
        json.dump(master_pop, f)
    print(f"\nmerged {len(era_pop)} '{args.label}' popularity entries into "
          f"{master_path} ({before} -> {len(master_pop)} total tracks)")

    print(f"\ndone. Extended bundle: {out_dir}")
    print("Restart the UI (or call ui.atlas.load() again) to pick up the new tracks.")
