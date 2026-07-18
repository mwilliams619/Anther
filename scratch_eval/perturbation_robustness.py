"""Perturbation-robustness probe — the 'essence-capture' test.

For each seed track we synthesize perturbed copies (pitch shift, time-stretch,
additive noise at several strengths), embed each through all four spaces
(MERT-1024, melody, rhythm, timbre) + the synthesized aggregate, and ask: does
the perturbed copy still retrieve its own clean original as the nearest neighbor
out of the full 484-track corpus? A space that captures a track's *essence*
should keep the original at rank 1 under mild perturbation.

Reports, per space and per perturbation strength:
  * rank_of_original (mean; 1 = perfect)
  * cos_to_original  (mean cosine of perturbed copy to its clean self)
  * top1_recovery    (fraction where original is the #1 neighbor)
"""
import json, os, sys, time
import numpy as np, torch, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anther_ml.embedding import load_mert, plan_windows, aggregate_layers, SR
from anther_ml.audio import loudness_normalize
from anther_ml import merit as M

RNG = np.random.default_rng(0)
N_SEEDS = 12  # personal seeds

PERTS = (
    [('pitch', s) for s in (-2,-1,1,2)] +      # semitones
    [('stretch', r) for r in (0.90,0.95,1.05,1.10)] +   # rate
    [('noise', db) for db in (20,10,5)]        # SNR dB (lower = more noise)
)

def load_audio(path):
    y,_ = librosa.load(str(path), sr=SR, mono=True)
    return y

def perturb(y, kind, amt):
    if kind=='pitch':   return librosa.effects.pitch_shift(y, sr=SR, n_steps=amt)
    if kind=='stretch': return librosa.effects.time_stretch(y, rate=amt)
    if kind=='noise':
        p_sig = np.mean(y**2); p_noise = p_sig/(10**(amt/10))
        return y + RNG.normal(0, np.sqrt(p_noise), size=y.shape).astype(np.float32)
    raise ValueError(kind)

def embed(y, model, processor, device, heads):
    y = loudness_normalize(y, SR)
    wins = [y[s:e] for s,e in plan_windows(len(y))]
    inp = processor(wins, sampling_rate=SR, return_tensors="pt", padding=True)
    inp = {k:v.to(device) for k,v in inp.items()}
    with torch.no_grad():
        hs = model(**inp, output_hidden_states=True).hidden_states
    v1024 = aggregate_layers(hs,"mean").mean(dim=0).cpu().numpy().astype(np.float32)
    v5120 = aggregate_layers(hs,"merit_concat").mean(dim=0).cpu().numpy().astype(np.float32)
    fac = M.project(v5120, heads, device)
    return v1024, fac

def l2(a): 
    a=np.asarray(a,dtype=np.float32); n=np.linalg.norm(a,axis=-1,keepdims=True)
    return a/np.where(n==0,1,n)

# corpus (clean reference set — the perturbed copy must find its original here)
d = np.load('models/personal_merit_vectors.npz', allow_pickle=True)
names=[str(n) for n in d['names']]; isp=d['is_personal'].astype(bool)
C = {'mert': l2(d['mert_1024']), 'mel': l2(d['factor_mel']),
     'rhy': l2(d['factor_rhy']), 'tim': l2(d['factor_tim'])}

# seeds: personal tracks (their clean vectors are already row idx in corpus)
personal_rows = [i for i in range(len(names)) if isp[i]]
seed_rows = list(RNG.choice(personal_rows, size=min(N_SEEDS,len(personal_rows)), replace=False))
seed_paths = {i: os.path.join('data/audio', names[i]) for i in seed_rows}

model, processor, device = load_mert(); heads = M.load_heads(device=device)

def agg_rank_and_cos(qfac, orig_row):
    """aggregate = equal-weight mean of 3 factor cosines vs whole corpus."""
    sims = {f: l2(qfac[f])[None,:] @ C[f].T for f in ('mel','rhy','tim')}  # each (1,N)
    agg = M.aggregate_similarity({f:sims[f][0] for f in sims})            # (N,)
    rank = int((agg > agg[orig_row]).sum()) + 1
    return rank, float(agg[orig_row])

records=[]; t0=time.time()
for si, row in enumerate(seed_rows):
    p = seed_paths[row]
    try:
        y = load_audio(p)
    except Exception as e:
        print(f'skip seed {names[row]}: {e}', flush=True); continue
    for kind, amt in PERTS:
        try:
            yp = perturb(y, kind, amt)
            v1024, fac = embed(yp, model, processor, device, heads)
        except Exception as e:
            print(f'  perturb fail {names[row]} {kind}{amt}: {e}', flush=True); continue
        rec = {'seed':names[row],'kind':kind,'amt':amt}
        # single spaces
        for space, q in [('mert',l2(v1024)),('mel',l2(fac['mel'])),
                         ('rhy',l2(fac['rhy'])),('tim',l2(fac['tim']))]:
            s = q[None,:] @ C[space].T; s=s[0]
            rec[f'{space}_rank']=int((s>s[row]).sum())+1
            rec[f'{space}_cos']=float(s[row])
        # aggregate
        ar, ac = agg_rank_and_cos(fac, row)
        rec['agg_rank']=ar; rec['agg_cos']=ac
        records.append(rec)
    print(f'seed {si+1}/{len(seed_rows)} {names[row][:30]} done ({time.time()-t0:.0f}s)', flush=True)

json.dump(records, open('scratch_eval/perturbation_records.json','w'), indent=2)
print(f'\n{len(records)} records saved; {time.time()-t0:.0f}s', flush=True)
