"""Apply human-authored cluster renames from a filled-in worksheet CSV.

Companion to `labels.py` (PLAYLIST_LABELS_BUILD_PLAN.md, Scheme 2). Reads a
CSV produced by the cluster-atlas writeup (columns: cluster_id, size,
current_label, tag_entropy_norm, top_tags, example_tracks, new_label) and
writes any filled-in `new_label` values into the bundle's
cluster_profiles.json as `label_override` / `label_override_at`, which
`apply_labels_to_profiles` (labels.py) treats as authoritative over the
auto-generated playlist-TF-IDF `label`.

This script only ever sets label_override / label_override_at / label_final
for rows whose `new_label` cell is non-empty. It never touches cluster_id,
size, exemplars, top_playlists, top_genres, label, or label_source — running
it twice with the same worksheet is a no-op the second time, and re-running
`anther_ml.corpus.labels.label_bundle` afterwards will not clobber the
overrides (see apply_labels_to_profiles).

Usage:
    python -m anther_ml.corpus.apply_cluster_renames \\
        --worksheet cluster_rename_worksheet.csv \\
        --bundle models/corpus_corpus_mpd_100k \\
        [--dry-run]

The worksheet's `cluster_id` column is matched against
cluster_profiles.json's `cluster_id` field. A worksheet row with an empty or
whitespace-only `new_label` is skipped (leaves that cluster's existing label
untouched, whether auto or a prior override). A backup of the untouched
cluster_profiles.json is written alongside it as cluster_profiles.json.bak
before any write, unless --dry-run is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_worksheet(path: str | Path) -> dict[int, str]:
    """Read the rename worksheet, return {cluster_id: new_label} for rows
    with a non-empty new_label cell."""
    renames: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "cluster_id" not in reader.fieldnames or "new_label" not in reader.fieldnames:
            raise ValueError(
                f"worksheet must have 'cluster_id' and 'new_label' columns, "
                f"got: {reader.fieldnames}"
            )
        for row in reader:
            new_label = (row.get("new_label") or "").strip()
            if not new_label:
                continue
            cid = int(row["cluster_id"])
            renames[cid] = new_label
    return renames


def apply_renames(
    profiles: list[dict], renames: dict[int, str], *, now: str | None = None
) -> tuple[list[dict], int]:
    """Set label_override/label_override_at/label_final for every profile
    whose cluster_id is in `renames`. Returns (profiles, n_updated)."""
    now = now or datetime.now(timezone.utc).isoformat()
    n_updated = 0
    for profile in profiles:
        cid = profile["cluster_id"]
        if cid not in renames:
            continue
        profile["label_override"] = renames[cid]
        profile["label_override_at"] = now
        profile["label_final"] = renames[cid]
        n_updated += 1
    return profiles, n_updated


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worksheet", required=True, help="Filled-in cluster_rename_worksheet.csv"
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="Path to the bundle directory containing cluster_profiles.json "
        "(e.g. models/corpus_corpus_mpd_100k)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; do not write anything.",
    )
    args = ap.parse_args(argv)

    renames = load_worksheet(args.worksheet)
    if not renames:
        print("No non-empty new_label rows found in worksheet — nothing to do.")
        return 0

    profiles_path = Path(args.bundle) / "cluster_profiles.json"
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)

    known_ids = {p["cluster_id"] for p in profiles}
    unknown = sorted(set(renames) - known_ids)
    if unknown:
        print(
            f"WARNING: worksheet has cluster_id(s) not present in bundle, skipping: {unknown}",
            file=sys.stderr,
        )
        for cid in unknown:
            renames.pop(cid)

    print(f"{len(renames)} cluster(s) to rename in {profiles_path}:")
    for cid in sorted(renames):
        old = next(p["label_final"] for p in profiles if p["cluster_id"] == cid)
        print(f"  #{cid}: {old!r} -> {renames[cid]!r}")

    if args.dry_run:
        print("(dry run — no files written)")
        return 0

    backup_path = profiles_path.with_suffix(".json.bak")
    shutil.copy2(profiles_path, backup_path)
    print(f"Backed up existing profiles to {backup_path}")

    profiles, n_updated = apply_renames(profiles, renames)

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Wrote {n_updated} override(s) to {profiles_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
