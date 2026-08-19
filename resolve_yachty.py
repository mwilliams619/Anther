"""Resolve Lil Yachty discography from iTunes with release dates + era mapping."""
import os, re
os.chdir("/home/matt/Dev/Anther")
import requests, pandas as pd

ARTIST_ID = 1080733657
r = requests.get("https://itunes.apple.com/lookup",
                 params={"id": ARTIST_ID, "entity": "song", "limit": 200}, timeout=30)
data = r.json()["results"]
songs = [x for x in data if x.get("wrapperType")=="track" and x.get("kind")=="song"
         and x.get("previewUrl") and x.get("trackId") and str(x.get("artistId"))==str(ARTIST_ID)]
print("primary-artist songs w/ preview:", len(songs))

def base_title(t): return re.sub(r"[^a-z0-9]+"," ",re.sub(r"\s*[\(\[].*?[\)\]]","",(t or "").lower())).strip()
seen = {}
for x in songs:
    key = base_title(x.get("trackName",""))
    rd = x.get("releaseDate","9999")
    if key not in seen or rd < seen[key].get("releaseDate","9999"):
        seen[key] = x
rows = list(seen.values())
print("deduped:", len(rows))

def era_of(coll):
    c = (coll or "").lower()
    if "lil boat 2" in c: return "Lil Boat 2 (2018)"
    if "lil boat 3" in c: return "Lil Boat 3 / 3.5 (2020)"  # 3 and 3.5
    if c.startswith("lil boat"): return "Lil Boat (2015-16)"
    if "teenage emotions" in c: return "Teenage Emotions (2017)"
    if "nuthin' 2 prove" in c or "nuthin 2 prove" in c: return "Nuthin' 2 Prove (2018)"
    if "let" in c and "start here" in c: return "Let's Start Here. (2023)"
    return "Singles / Soundtrack / Other"

recs = []
for x in rows:
    coll = x.get("collectionName","")
    rd = x.get("releaseDate","")
    year = int(rd[:4]) if rd[:4].isdigit() else 0
    recs.append({
        "track_id": f"itunes:{x['trackId']}", "itunes_id": x["trackId"],
        "title": x.get("trackName",""), "album": coll,
        "release_date": rd, "year": year,
        "duration_ms": x.get("trackTimeMillis"),
        "genre_itunes": x.get("primaryGenreName"),
        "era": era_of(coll),
        "preview_url": x["previewUrl"],
    })
df = pd.DataFrame(recs).sort_values(["year","album","title"]).reset_index(drop=True)
df.to_csv("discography_metadata_yachty.csv", index=False)
print("\nera counts:")
print(df["era"].value_counts())
print("\ntotal:", len(df))
