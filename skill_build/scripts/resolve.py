"""Resolve an artist's discography from iTunes with release dates + era mapping.
Usage: python resolve.py <artist_name> <era_rules_json> <out_csv> [repo] [artist_id]
era_rules_json: JSON list of [substring, era_label] pairs (first match wins, case-insensitive).
Runs as a subprocess so the network proxy is live. Writes out_csv, prints era counts.
"""
import os, re, sys, json
repo = sys.argv[4] if len(sys.argv) > 4 else "/home/matt/Dev/Anther"
sys.path.insert(0, repo); os.chdir(repo)
import requests, pandas as pd
from anther_ml import itunes

artist_name = sys.argv[1]
era_rules = json.loads(sys.argv[2])
out_csv = sys.argv[3]
forced_id = sys.argv[5] if len(sys.argv) > 5 else None

if forced_id:
    aid = int(forced_id); ainfo = {"artist_id": aid, "name": artist_name, "genre": None}
else:
    ainfo = itunes.resolve_artist(artist_name)
    if not ainfo:
        print(json.dumps({"error": f"artist not found: {artist_name}"})); sys.exit(1)
    aid = ainfo["artist_id"]

r = requests.get("https://itunes.apple.com/lookup",
                 params={"id": aid, "entity": "song", "limit": 200}, timeout=30)
data = r.json()["results"]
songs = [x for x in data if x.get("wrapperType") == "track" and x.get("kind") == "song"
         and x.get("previewUrl") and x.get("trackId") and str(x.get("artistId")) == str(aid)]

def base_title(t):
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"\s*[\(\[].*?[\)\]]", "", (t or "").lower())).strip()

seen = {}
for x in songs:
    k = base_title(x.get("trackName", "")); rd = x.get("releaseDate", "9999")
    if k not in seen or rd < seen[k].get("releaseDate", "9999"):
        seen[k] = x
rows = list(seen.values())

def era_of(coll):
    c = (coll or "").lower()
    for sub, label in era_rules:
        if sub.lower() in c:
            return label
    return "Singles / Soundtrack / Other"

recs = []
for x in rows:
    coll = x.get("collectionName", ""); rd = x.get("releaseDate", "")
    year = int(rd[:4]) if rd[:4].isdigit() else 0
    recs.append({"track_id": f"itunes:{x['trackId']}", "itunes_id": x["trackId"],
                 "title": x.get("trackName", ""), "album": coll, "release_date": rd, "year": year,
                 "duration_ms": x.get("trackTimeMillis"), "genre_itunes": x.get("primaryGenreName"),
                 "era": era_of(coll), "preview_url": x["previewUrl"]})
df = pd.DataFrame(recs).sort_values(["year", "album", "title"]).reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(json.dumps({"artist": ainfo["name"], "artist_id": aid, "genre": ainfo.get("genre"),
                  "primary_songs": len(songs), "deduped": len(df),
                  "era_counts": df["era"].value_counts().to_dict(), "out_csv": out_csv}, indent=1))
