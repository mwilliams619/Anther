"""Before/after eval: MERT baseline vs mel/rhy/tim/aggregate factor spaces.

Related groups = genuine personal alternates ONLY (distractors are unrelated
singletons in the pool, never query targets or valid retrievals). This is a real
discrimination test: each query must surface its true relative out of 483 others.
Metrics reuse eval.py's self_retrieval definition exactly.
"""
import json, csv
import numpy as np
from collections import defaultdict
from anther_ml.similarity import _l2_normalize
from anther_ml.eval import auto_related_groups, title_stem
from anther_ml import merit as M

d = np.load('models/personal_merit_vectors.npz', allow_pickle=True)
names = [str(n) for n in d['names']]
isp = d['is_personal'].astype(bool)

# Groups from PERSONAL tracks only; distractors excluded from grouping entirely.
personal_idx = [i for i in range(len(names)) if isp[i]]
pers_names = [names[i] for i in personal_idx]
pers_groups = auto_related_groups(pers_names)  # {stem:[local idx]}
# remap local personal idx -> global corpus idx
groups = {stem: [personal_idx[j] for j in members] for stem, members in pers_groups.items()}
n_groups = len(groups)
query_tracks = sorted(set(i for m in groups.values() for i in m))
print(f'corpus={len(names)} personal={int(isp.sum())} distractors={int((~isp).sum())}')
print(f'related_groups={n_groups} query_tracks={len(query_tracks)}')
for stem, m in groups.items():
    print(f'  [{stem}] {[names[i] for i in m]}')

def self_retrieval_from_sim(sim, groups, ks=(1,5,10)):
    sim = sim.copy(); np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    relatives = defaultdict(set)
    for members in groups.values():
        for mm in members: relatives[mm].update(x for x in members if x!=mm)
    q = [t for t,rel in relatives.items() if rel]
    hits={k:0 for k in ks}; rr=0.0; ranks=[]
    for t in q:
        rel=relatives[t]; fr=None
        for rank,nb in enumerate(order[t],1):
            if nb in rel: fr=rank; break
        ranks.append(fr)
        if fr:
            rr+=1.0/fr
            for k in ks:
                if fr<=k: hits[k]+=1
    n=len(q)
    return {'n_queries':n,'mrr':rr/n,**{f'recall@{k}':hits[k]/n for k in ks},
            'ranks':ranks}

def cos(embs):
    e=_l2_normalize(embs.astype(np.float32)); return e@e.T

S = {f: cos(d[k]) for f,k in [('mel','factor_mel'),('rhy','factor_rhy'),('tim','factor_tim')]}
spaces = {
    'MERT baseline (1024)': cos(d['mert_1024']),
    'melody (128)': S['mel'],
    'rhythm (128)': S['rhy'],
    'timbre (128)': S['tim'],
    'aggregate (equal wts)': M.aggregate_similarity(S),
    'agg melody-heavy': M.aggregate_similarity(S, {'mel':3,'rhy':1,'tim':1}),
    'agg rhythm-heavy': M.aggregate_similarity(S, {'mel':1,'rhy':3,'tim':1}),
    'agg timbre-heavy': M.aggregate_similarity(S, {'mel':1,'rhy':1,'tim':3}),
}
rows=[]
for label, sim in spaces.items():
    r = self_retrieval_from_sim(sim, groups)
    rows.append({'space':label,'n_queries':r['n_queries'],
                 'recall@1':r['recall@1'],'recall@5':r['recall@5'],
                 'recall@10':r['recall@10'],'mrr':r['mrr'],'ranks':r['ranks']})

with open('scratch_eval/merit_eval_comparison.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=['space','n_queries','recall@1','recall@5','recall@10','mrr'])
    wr.writeheader()
    for r in rows: wr.writerow({k:r[k] for k in wr.fieldnames})

print(f'\n=== SELF-RETRIEVAL (personal groups only; {len(query_tracks)} queries vs {len(names)-1} others) ===')
print(f"{'space':24s}{'r@1':>7}{'r@5':>7}{'r@10':>7}{'mrr':>7}  ranks")
for r in rows:
    print(f"{r['space']:24s}{r['recall@1']:7.3f}{r['recall@5']:7.3f}{r['recall@10']:7.3f}{r['mrr']:7.3f}  {r['ranks']}")

json.dump({'n_groups':n_groups,'query_tracks':[names[i] for i in query_tracks],
           'groups':{k:[names[i] for i in v] for k,v in groups.items()},
           'rows':[{k:r[k] for k in ['space','n_queries','recall@1','recall@5','recall@10','mrr','ranks']} for r in rows]},
          open('scratch_eval/merit_eval_comparison.json','w'), indent=2)
print('saved scratch_eval/merit_eval_comparison.{csv,json}')
