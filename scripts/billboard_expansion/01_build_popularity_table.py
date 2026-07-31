"""
Step 1 — Build a deduped popularity table from Billboard Hot 100 history.

Source: mhollingshead/billboard-hot-100 (all.json), weekly Hot 100 snapshots
1958-08-04 through present (updates live). Covers the full "last 20 years"
window on its own; chartscraper's 2018-cutoff CSVs are NOT needed for Hot 100
(kept optional below only if you want other charts for genre spread).

Usage:
    python 01_build_popularity_table.py \
        --years 20 \
        --out data/billboard/popularity_songs.csv

Output columns: artist, title, norm_key, peak_position, best_peak,
weeks_on_chart_total, first_charted, last_charted, popularity_score
"""
import argparse
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HOT100_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"


def normalize_key(artist: str, title: str) -> str:
    def clean(s):
        s = s or ""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        for cut in (" feat.", " feat ", " ft.", " ft ", " (", " ["):
            i = s.find(cut)
            if i != -1:
                s = s[:i]
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return s.strip()
    return f"{clean(artist)} -- {clean(title)}"


def fetch_hot100(cache_path: Path) -> list[dict]:
    if cache_path.exists():
        print(f"using cached {cache_path}")
    else:
        print(f"downloading {HOT100_URL} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOT100_URL, cache_path)
    with open(cache_path) as f:
        return json.load(f)


def build_table(weeks: list[dict], cutoff_date: str) -> pd.DataFrame:
    agg = {}  # norm_key -> stats dict
    for wk in weeks:
        date = wk["date"]
        if date < cutoff_date:
            continue
        for row in wk["data"]:
            artist, title = row.get("artist") or "", row.get("song") or ""
            key = normalize_key(artist, title)
            peak = row.get("peak_position")
            weeks_on = row.get("weeks_on_chart") or 0
            rec = agg.setdefault(key, {
                "artist": artist, "title": title,
                "best_peak": 101, "weeks_on_chart_total": 0,
                "first_charted": date, "last_charted": date,
            })
            if peak is not None:
                rec["best_peak"] = min(rec["best_peak"], peak)
            rec["weeks_on_chart_total"] = max(rec["weeks_on_chart_total"], weeks_on)
            rec["first_charted"] = min(rec["first_charted"], date)
            rec["last_charted"] = max(rec["last_charted"], date)

    rows = []
    for key, rec in agg.items():
        # simple recency-aware popularity score: higher peak (lower number) and
        # more weeks charted both push it up; recency mildly boosts newer hits.
        years_ago = (datetime.now(timezone.utc).year - int(rec["last_charted"][:4]))
        recency_weight = max(0.5, 1.0 - 0.02 * years_ago)
        score = ((101 - rec["best_peak"]) * 2 + rec["weeks_on_chart_total"]) * recency_weight
        rows.append({
            "norm_key": key,
            "artist": rec["artist"],
            "title": rec["title"],
            "best_peak": rec["best_peak"],
            "weeks_on_chart_total": rec["weeks_on_chart_total"],
            "first_charted": rec["first_charted"],
            "last_charted": rec["last_charted"],
            "popularity_score": round(score, 2),
        })
    df = pd.DataFrame(rows).sort_values("popularity_score", ascending=False)
    return df.reset_index(drop=True)


def build_table_range(weeks: list[dict], start_date: str, end_date: str) -> pd.DataFrame:
    """Same aggregation as build_table but bounded on both ends (inclusive),
    and using the *end_date* year (not "now") for the recency weight — so a
    decade request (e.g. 1970-01-01..1979-12-31) scores its own hits by
    recency-within-that-window rather than decades-stale-from-today.
    """
    agg = {}
    end_year = int(end_date[:4])
    for wk in weeks:
        date = wk["date"]
        if date < start_date or date > end_date:
            continue
        for row in wk["data"]:
            artist, title = row.get("artist") or "", row.get("song") or ""
            key = normalize_key(artist, title)
            peak = row.get("peak_position")
            weeks_on = row.get("weeks_on_chart") or 0
            rec = agg.setdefault(key, {
                "artist": artist, "title": title,
                "best_peak": 101, "weeks_on_chart_total": 0,
                "first_charted": date, "last_charted": date,
            })
            if peak is not None:
                rec["best_peak"] = min(rec["best_peak"], peak)
            rec["weeks_on_chart_total"] = max(rec["weeks_on_chart_total"], weeks_on)
            rec["first_charted"] = min(rec["first_charted"], date)
            rec["last_charted"] = max(rec["last_charted"], date)

    rows = []
    for key, rec in agg.items():
        years_ago = end_year - int(rec["last_charted"][:4])
        recency_weight = max(0.5, 1.0 - 0.02 * years_ago)
        score = ((101 - rec["best_peak"]) * 2 + rec["weeks_on_chart_total"]) * recency_weight
        rows.append({
            "norm_key": key,
            "artist": rec["artist"],
            "title": rec["title"],
            "best_peak": rec["best_peak"],
            "weeks_on_chart_total": rec["weeks_on_chart_total"],
            "first_charted": rec["first_charted"],
            "last_charted": rec["last_charted"],
            "popularity_score": round(score, 2),
        })
    df = pd.DataFrame(rows).sort_values("popularity_score", ascending=False)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=None,
                    help="rolling window: last N years from today (default mode)")
    ap.add_argument("--start-date", default=None,
                    help="absolute window start YYYY-MM-DD (e.g. 1970-01-01 for a decade pull)")
    ap.add_argument("--end-date", default=None,
                    help="absolute window end YYYY-MM-DD (e.g. 1979-12-31); defaults to today if --start-date given alone")
    ap.add_argument("--cache", default="data/billboard/all_hot100.json")
    ap.add_argument("--out", default="data/billboard/popularity_songs.csv")
    args = ap.parse_args()

    weeks = fetch_hot100(Path(args.cache))

    if args.start_date:
        start = args.start_date
        end = args.end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(f"filtering to weeks in [{start}, {end}]")
        df = build_table_range(weeks, start, end)
    else:
        years = args.years if args.years is not None else 20
        cutoff = f"{datetime.now(timezone.utc).year - years}-01-01"
        print(f"filtering to weeks >= {cutoff} (last {years} years)")
        df = build_table(weeks, cutoff)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} unique songs -> {args.out}")
    print(df.head(10).to_string())
