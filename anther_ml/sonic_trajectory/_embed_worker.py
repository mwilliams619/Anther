"""Subprocess worker: embed discography previews through MERT+MERIT and place
onto the frozen corpus. Run standalone so the network proxy is live and the
local MERT weights load offline. Not imported — invoked via ``embed.py``.

    python -m anther_ml.sonic_trajectory._embed_worker <csv> <out_dir> <prefix> <repo> <mert_dir> <corpus_dir>
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

csv_path, out_dir, prefix, repo, mert_dir, corpus_dir = sys.argv[1:7]
sys.path.insert(0, repo)
os.chdir(repo)

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from anther_ml import itunes
from anther_ml.corpus.bundle import ReferenceCorpus
from anther_ml.corpus.place import place
from anther_ml.embedding import get_embedding_dual
from anther_ml.merit import load_heads, merit_query_vector

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(mert_dir, trust_remote_code=True).to(device).eval()
heads = load_heads(device="cpu")  # merit_query_vector projects on cpu; keep heads there
corpus = ReferenceCorpus.load(corpus_dir)
print(f"loaded: MERT+heads on {device}, corpus {corpus.n_tracks} tracks", flush=True)

df = pd.read_csv(csv_path)


def fetch_decode(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if len(r.content) < 1024:
        raise ValueError(f"tiny preview ({len(r.content)}B)")
    raw = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    raw.write(r.content)
    raw.close()
    try:
        return itunes.decode_preview(Path(raw.name))
    finally:
        Path(raw.name).unlink(missing_ok=True)


recs, mert_mat, merit_mat = [], [], []
t0 = time.time()
for i, row in df.iterrows():
    try:
        wavpath, cleanup = fetch_decode(row["preview_url"])
        mert_vec, backbone = get_embedding_dual(model, processor, str(wavpath), device, normalize=True)
        mvec = merit_query_vector(backbone, heads)
        res = place(corpus, mert_vec, merit_vec=mvec, top_k=10)
        cleanup()
        cl = res["cluster"]
        neigh = [
            {"artist": n.get("artist", ""), "title": n.get("title", ""),
             "score": float(n.get("score", n.get("similarity", 0)))}
            for n in res["neighbors"][:5]
        ]
        recs.append({
            "track_id": row["track_id"], "title": row["title"], "album": row["album"],
            "era": row["era"], "year": int(row["year"]), "release_date": row["release_date"],
            "cluster_id": int(cl["id"]), "cluster_label": cl["label"],
            "cluster_conf": float(cl["confidence"]),
            "x2d": float(res["coords_2d"][0]) if res["coords_2d"] else None,
            "y2d": float(res["coords_2d"][1]) if res["coords_2d"] else None,
            "neighbors": json.dumps(neigh),
        })
        mert_mat.append(mert_vec)
        merit_mat.append(mvec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df)}  {round(time.time()-t0,1)}s", flush=True)
    except Exception as e:
        print(f"  SKIP [{i}] {str(row['title'])[:40]}: {type(e).__name__}: {str(e)[:80]}", flush=True)

out_df = pd.DataFrame(recs)
plc = os.path.join(out_dir, f"{prefix}_placements.csv")
npz = os.path.join(out_dir, f"{prefix}_embeddings.npz")
out_df.to_csv(plc, index=False)
np.savez_compressed(
    npz,
    mert=np.array(mert_mat, dtype=np.float32),
    merit=np.array(merit_mat, dtype=np.float32),
    track_ids=out_df["track_id"].to_numpy(),
)
print(f"DONE: {len(out_df)}/{len(df)} embedded in {round(time.time()-t0,1)}s", flush=True)
print("RESULT " + json.dumps({"n_embedded": len(out_df), "n_requested": len(df),
                              "placements_csv": plc, "embeddings_npz": npz}), flush=True)
