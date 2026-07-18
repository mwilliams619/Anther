"""Encode FMA distractors and merge into personal_merit_vectors.npz."""
import glob, os, sys, time
import numpy as np, torch, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anther_ml.embedding import load_mert, plan_windows, aggregate_layers, SR
from anther_ml.audio import loudness_normalize
from anther_ml import merit as M

N = 400
RNG = np.random.default_rng(0)

def wins(path):
    y,_ = librosa.load(str(path), sr=SR, mono=True)
    y = loudness_normalize(y, SR)
    return [y[s:e] for s,e in plan_windows(len(y))]

d = dict(np.load("models/personal_merit_vectors.npz", allow_pickle=True))
have = set(d["names"].tolist())

fma = sorted(glob.glob("data/audio/fma_medium/**/*.mp3", recursive=True))
pick = RNG.choice(len(fma), size=min(N*2, len(fma)), replace=False)  # oversample for skips
model, processor, device = load_mert()
heads = M.load_heads(device=device)

names, mert, back = [], [], []
t0=time.time()
for i in pick:
    if len(names) >= N: break
    p = fma[i]; nm = "FMA_"+os.path.basename(p)
    try:
        w = wins(p)
        inp = processor(w, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k:v.to(device) for k,v in inp.items()}
        with torch.no_grad():
            hs = model(**inp, output_hidden_states=True).hidden_states
        v1024 = aggregate_layers(hs,"mean").mean(dim=0).cpu().numpy().astype(np.float32)
        v5120 = aggregate_layers(hs,"merit_concat").mean(dim=0).cpu().numpy().astype(np.float32)
    except Exception as e:
        continue
    names.append(nm); mert.append(v1024); back.append(v5120)
    if len(names)%100==0: print(f"  {len(names)}/{N} {(time.time()-t0)/len(names):.2f}s/tr", flush=True)

mert=np.vstack(mert); back=np.vstack(back)
fac = M.project(back, heads, device)
# merge: personal first, then distractors
out_names = np.concatenate([d["names"], np.array(names)])
out_isp   = np.concatenate([d["is_personal"], np.zeros(len(names), bool)])
np.savez("models/personal_merit_vectors.npz",
    names=out_names, is_personal=out_isp,
    mert_1024=np.vstack([d["mert_1024"], mert]),
    factor_mel=np.vstack([d["factor_mel"], fac["mel"]]),
    factor_rhy=np.vstack([d["factor_rhy"], fac["rhy"]]),
    factor_tim=np.vstack([d["factor_tim"], fac["tim"]]),
    backbone_5120=np.vstack([d["backbone_5120"], back]),
)
print(f"MERGED total={len(out_names)} personal={int(out_isp.sum())} distractors={len(names)} in {time.time()-t0:.0f}s", flush=True)
