"""Encode the personal corpus + FMA distractors in BOTH spaces in one MERT pass.

For each track we run MERT once and extract:
  * mert_1024  — mean over all 25 layers (the current Phase-2 embedding)
  * factor_{mel,rhy,tim} — 128-d from the MERIT heads on the 5120-d backbone
Saves models/personal_merit_vectors.npz (checkpoint so re-eval never re-runs GPU).
"""
import glob, os, sys, time
import numpy as np, torch, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anther_ml.embedding import (
    load_mert, plan_windows, aggregate_layers, SR, MERIT_LAYERS,
)
from anther_ml.audio import loudness_normalize
from anther_ml import merit as M

N_DISTRACTORS = 400
RNG = np.random.default_rng(0)

def track_windows(path, normalize=True):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if normalize:
        y = loudness_normalize(y, SR)
    return [y[s:e] for s, e in plan_windows(len(y))]

def main():
    personal = sorted(p for p in glob.glob("data/audio/*.mp3")
                      if not os.path.basename(p).startswith("._"))
    fma = sorted(p for p in glob.glob("data/audio/fma_medium/*.mp3")
                 if not os.path.basename(p).startswith("._"))
    fma = list(RNG.choice(fma, size=min(N_DISTRACTORS, len(fma)), replace=False)) if fma else []
    paths = personal + fma
    is_personal = np.array([True]*len(personal) + [False]*len(fma))
    print(f"personal={len(personal)} distractors={len(fma)} total={len(paths)}", flush=True)

    model, processor, device = load_mert()
    heads = M.load_heads(device=device)

    names, mert_1024, backbone_5120 = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            wins = track_windows(p, normalize=True)
            inp = processor(wins, sampling_rate=SR, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True).hidden_states
            v1024 = aggregate_layers(hs, "mean").mean(dim=0).cpu().numpy().astype(np.float32)
            v5120 = aggregate_layers(hs, "merit_concat").mean(dim=0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"skip {os.path.basename(p)}: {e}", flush=True); continue
        names.append(os.path.basename(p)); mert_1024.append(v1024); backbone_5120.append(v5120)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}  {(time.time()-t0)/(i+1):.2f}s/track", flush=True)

    mert_1024 = np.vstack(mert_1024); backbone_5120 = np.vstack(backbone_5120)
    # project all backbones → factors in one shot
    fac = M.project(backbone_5120, heads, device)
    is_personal = is_personal[:len(names)]  # in case any skipped (personal are first, unlikely)
    np.savez(
        "models/personal_merit_vectors.npz",
        names=np.array(names), is_personal=is_personal,
        mert_1024=mert_1024,
        factor_mel=fac["mel"], factor_rhy=fac["rhy"], factor_tim=fac["tim"],
        backbone_5120=backbone_5120,
    )
    print(f"SAVED {len(names)} tracks in {time.time()-t0:.0f}s -> models/personal_merit_vectors.npz", flush=True)
    print("shapes:", mert_1024.shape, fac["mel"].shape, backbone_5120.shape, flush=True)

if __name__ == "__main__":
    main()
