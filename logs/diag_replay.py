import sys, os, traceback
sys.path.insert(0,'ui')
import atlas
print('loading corpus...', flush=True)
atlas.load()
print('corpus ready, tracks=', len(atlas._corpus.metadata),
      'merit_index=', atlas._corpus.merit_index is not None, flush=True)
sessions = sorted(os.listdir('ui/session'))
bad = []
ok = 0
for sid in sessions:
    gp = os.path.join('ui/session', sid, 'graph.json')
    if not os.path.isfile(gp): continue
    st = atlas._SessionState(sid)
    try:
        atlas._load_graph(st)
        ok += 1
    except Exception as e:
        bad.append((sid, repr(e)))
        print('FAILED', sid, flush=True)
        traceback.print_exc()
        print('----', flush=True)
print('DONE ok=%d bad=%d' % (ok, len(bad)), flush=True)
for sid,e in bad: print('  BAD', sid, e, flush=True)
