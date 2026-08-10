"""CLI for the sound-over-time pipeline.

    python -m anther_ml.sonic_trajectory --artist "Beyoncé" \
        --eras eras_beyonce.json --out results/beyonce [--persist] [--no-blurb]

--eras is a JSON list of [substring, era_label] pairs (first match wins). When
--persist is given, the sonic_trajectory block is written onto the artist's
profile in the enrichment SQLite store. Blurbs require an LLM; by default the
CLI skips them (pass --qwen to use the local mentor Qwen model if available).
"""

from __future__ import annotations

import argparse
import json
import sys

from .discography import EraRule
from .pipeline import run_artist


def _load_eras(path: str) -> list[EraRule]:
    pairs = json.load(open(path))
    return [EraRule(sub, era) for sub, era in pairs]


def _qwen_llm():
    """Best-effort local Qwen callable from the mentor stack; None if unavailable."""
    try:
        from mentor.mentor import load_llm  # type: ignore
    except Exception:
        return None
    gen = load_llm()

    def llm_fn(prompt: str, system: str) -> str:
        return gen(prompt, system=system)
    return llm_fn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anther_ml.sonic_trajectory")
    p.add_argument("--artist", required=True)
    p.add_argument("--eras", required=True, help="JSON [[substring, era], ...]")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--artist-id", type=int, default=None)
    p.add_argument("--persist", action="store_true", help="write onto the profile store")
    p.add_argument("--db", default=None, help="profiles SQLite path (default: package default)")
    p.add_argument("--no-blurb", action="store_true")
    p.add_argument("--qwen", action="store_true", help="use local mentor Qwen for blurbs")
    p.add_argument("--caveat", action="append", default=None, help="caveat text (repeatable)")
    p.add_argument("--label-track", action="append", default=None,
                   help="track substring to label on the atlas (repeatable)")
    args = p.parse_args(argv)

    era_rules = _load_eras(args.eras)
    llm_fn = None if args.no_blurb else (_qwen_llm() if args.qwen else None)

    store = None
    if args.persist:
        from anther_ml.artist_enrichment.store import ArtistProfileStore, DEFAULT_DB_PATH
        store = ArtistProfileStore(args.db or DEFAULT_DB_PATH)

    res = run_artist(args.artist, era_rules=era_rules, out_dir=args.out,
                     artist_id=args.artist_id, llm_fn=llm_fn, caveats=args.caveat,
                     label_tracks=args.label_track, store=store)
    if store is not None:
        store.close()

    print(json.dumps({"artist": res["artist"], "artist_key": res["artist_key"],
                      "summary": res["summary"], "figures": res["figures"],
                      "data": res["data"],
                      "blurbs": {k: len(v) for k, v in res["blurbs"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
