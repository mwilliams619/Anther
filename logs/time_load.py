import sys, os, time, resource, traceback
sys.path.insert(0,'ui')
import atlas
t0=time.time()
# instrument the sub-steps by monkeypatching timers
import anther_ml.corpus.bundle as B
_orig = B.ReferenceCorpus.load.__func__
def timed_load(cls, d, verify=True):
    s=time.time(); r=_orig(cls,d,verify); print(f'  ReferenceCorpus.load: {time.time()-s:.1f}s', flush=True); return r
B.ReferenceCorpus.load = classmethod(timed_load)
_calib = atlas._calibrate_qq_threshold
def timed_calib(corpus, n_pairs=200000):
    s=time.time(); r=_calib(corpus, n_pairs); print(f'  _calibrate_qq_threshold: {time.time()-s:.1f}s', flush=True); return r
atlas._calibrate_qq_threshold = timed_calib
print('start load()', flush=True)
atlas.load()
print(f'TOTAL atlas.load(): {time.time()-t0:.1f}s', flush=True)
print('peak RSS GB:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
# now time get_graph on biggest session
tok = atlas.set_session('js9ElgFdRK3hNJplCypcsA')
s=time.time()
g = atlas.get_graph()
print(f'get_graph(big session): {time.time()-s:.1f}s ready={g["ready"]} nodes={len(g["nodes"])} links={len(g["links"])}', flush=True)
