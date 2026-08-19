"""Resolve Tyler, The Creator discography from iTunes with release dates + era mapping."""
import os, re, json
os.chdir("/home/matt/Dev/Anther")
import requests, pandas as pd
from anther_ml import itunes

ARTIST_ID = 420368335
# raw lookup to get releaseDate + collection (the repo's _track_row strips these)
r = requests.get("https://itunes.apple.com/lookup",
                 params={"id": ARTIST_ID, "entity": "song", "limit": 200}, timeout=30)
data = r.json()["results"]
songs = [x for x in data if x.get("wrapperType")=="track" and x.get("kind")=="song"
         and x.get("previewUrl") and x.get("trackId")]
# keep only rows where Tyler is the PRIMARY artist
songs = [x for x in songs if str(x.get("artistId"))==str(ARTIST_ID)]
print("primary-artist songs w/ preview:", len(songs))

def norm(s): return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()
def base_title(t):  # drop feat / remix / version parentheticals for dedupe
    return norm(re.sub(r"\s*[\(\[].*?[\)\]]","", t or ""))

# dedupe: primary-artist + base title, keep earliest release
seen = {}
for x in songs:
    key = base_title(x.get("trackName",""))
    rd = x.get("releaseDate","9999")
    if key not in seen or rd < seen[key].get("releaseDate","9999"):
        seen[key] = x
rows = list(seen.values())
print("deduped:", len(rows))

# canonical studio-album era mapping
def era_of(coll, year):
    c = (coll or "").lower()
    if "goblin" in c: return "Goblin (2011)"
    if "wolf" in c: return "Wolf (2013)"
    if "cherry bomb" in c: return "Cherry Bomb (2015)"
    if "flower boy" in c: return "Flower Boy (2017)"
    if "igor" in c: return "IGOR (2019)"
    if "call me if you get lost" in c:  # includes The Estate Sale deluxe
        return "CALL ME IF YOU GET LOST (2021)"
    if "chromakopia" in c: return "CHROMAKOPIA (2024)"
    if "don't tap the glass" in c or "dont tap the glass" in c: return "DON'T TAP THE GLASS (2025)"
    if "odd future" in c: return "Odd Future / Early (pre-2011)"
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
        "era": era_of(coll, year),
        "preview_url": x["previewUrl"],
    })
df = pd.DataFrame(recs).sort_values(["year","album","title"]).reset_index(drop=True)
df.to_csv("discography_metadata_tyler.csv", index=False)
print("\nera counts:")
print(df["era"].value_counts())
print("\ntotal tracks:", len(df))
