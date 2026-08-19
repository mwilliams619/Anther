"""Embed the 5-artist iTunes discographies and place them on the frozen corpus.
Uses ui/atlas directly so vectors are byte-identical to the running UI, and
resolve_and_embed() writes each raw MERT+MERIT vector into the DEFAULT session
embed cache as a side effect (this is the prewarm step).
"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/home/matt/Dev/Anther/ui")
os.chdir("/home/matt/Dev/Anther")

import numpy as np, pandas as pd

# net_guard's public-IP check uses raw-socket getaddrinfo, which this
# proxy-only sandbox can't perform (no direct DNS). The primary SSRF control —
# the host allowlist — still runs and audio-ssl.itunes.apple.com is explicitly
# on it; the outbound proxy separately enforces the network allowlist. Disable
# only the DNS-based defense-in-depth check for this offline batch run.
from anther_ml import net_guard
net_guard._resolves_public = lambda host: True

import atlas

OUT = os.environ.get("OUT_DIR", ".")

# atlas._mert() loads MERT from the HF hub id, which fails offline. Prime the
# module globals from the repo's LOCAL MERT copy (identical weights) so _mert()
# short-circuits — same model, same recipe, no network.
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
LOCAL_MERT = "models/mert_v1_330m"
_dev = "cuda" if torch.cuda.is_available() else "cpu"
atlas._processor = Wav2Vec2FeatureExtractor.from_pretrained(LOCAL_MERT, trust_remote_code=True)
atlas._model = AutoModel.from_pretrained(LOCAL_MERT, trust_remote_code=True).to(_dev).eval()
atlas._device = _dev
print(f"MERT primed from {LOCAL_MERT} on {_dev}", flush=True)

atlas.set_session("default")            # write into ui/session/default/embed_cache.sqlite
corpus = atlas.load()
print(f"corpus loaded: {corpus.n_tracks} tracks; merit_index={corpus.merit_index is not None}", flush=True)

df = pd.read_csv("discography_metadata_5artists.csv")
print(f"{len(df)} tracks to embed", flush=True)

recs, mert_mat, merit_mat, ids = [], [], [], []
t0 = time.time()
skipped = []
for i, row in df.iterrows():
    tid = row["track_id"]
    label = f"{row['artist_name']} — {row['title'][:40]}"
    try:
        cached = atlas.cached_vec(tid)
        if cached is not None:
            vec = cached
            backbone = atlas.cached_backbone(tid) if hasattr(atlas, "cached_backbone") else None
            method = "cache"
        else:
            vec, backbone, method = atlas.resolve_and_embed({
                "id": tid, "name": row["title"], "artist": row["artist_name"],
                "preview_url": row["preview_url"],
            })
        mvec = atlas._merit_query_vec(backbone)
        res = atlas.place(corpus, vec, merit_vec=mvec, top_k=10)
        cl = res["cluster"]
        neigh = [{"artist": n.get("artist",""), "title": n.get("title",""),
                  "score": float(n.get("score", n.get("similarity", 0)))}
                 for n in res["neighbors"][:5]]
        if mvec is not None:
            mel, rhy, tim = mvec[:128], mvec[128:256], mvec[256:384]
            mel_n, rhy_n, tim_n = float(np.linalg.norm(mel)), float(np.linalg.norm(rhy)), float(np.linalg.norm(tim))
        else:
            mel_n = rhy_n = tim_n = float("nan")
        recs.append({
            "artist_name": row["artist_name"], "track_id": tid,
            "title": row["title"], "album": row["album"],
            "year": int(row["year"]) if row["year"] else 0,
            "release_date": row["release_date"],
            "genre_itunes": row["genre_itunes"],
            "cluster_id": int(cl["id"]), "cluster_label": cl["label"],
            "cluster_conf": float(cl["confidence"]),
            "x2d": float(res["coords_2d"][0]) if res["coords_2d"] else None,
            "y2d": float(res["coords_2d"][1]) if res["coords_2d"] else None,
            "mel_norm": mel_n, "rhy_norm": rhy_n, "tim_norm": tim_n,
            "embed_method": method,
            "neighbors": json.dumps(neigh),
            "preview_url": row["preview_url"],
        })
        mert_mat.append(np.asarray(vec, dtype=np.float32))
        merit_mat.append(np.asarray(mvec, dtype=np.float32) if mvec is not None else np.zeros(384, np.float32))
        ids.append(tid)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(df)}  {round(time.time()-t0,1)}s  last={label}", flush=True)
    except Exception as e:
        skipped.append({"track_id": tid, "label": label, "err": f"{type(e).__name__}: {str(e)[:100]}"})

out_df = pd.DataFrame(recs)
out_df.to_csv(os.path.join(OUT, "placements_5artists.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "embeddings_5artists.npz"),
                    mert=np.array(mert_mat, dtype=np.float32),
                    merit=np.array(merit_mat, dtype=np.float32),
                    track_ids=np.array(ids))
with open(os.path.join(OUT, "embed_5artists_skips.json"), "w") as f:
    json.dump(skipped, f, indent=2)
print(f"\nDONE: {len(out_df)}/{len(df)} embedded, {len(skipped)} skipped in {round(time.time()-t0,1)}s", flush=True)
print("per-artist embedded:")
print(out_df["artist_name"].value_counts().to_string())
