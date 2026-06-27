Run a personal song through the Phase 1 pipeline and return its cluster assignment and top-10 nearest FMA tracks.

The user will provide a path to an MP3 file. If they haven't, ask for one.

Use this pattern (run from the repo root with the venv active):

```python
import sys
sys.path.insert(0, '.')

import numpy as np
from anther_ml.features import extract_librosa_features
from anther_ml.cluster import load_pipeline, assign_cluster
from anther_ml.similarity import SongIndex

SONG_PATH = "<path provided by user>"

scaler, pca, reducer, clusterer = load_pipeline('models/pipeline_phase1.pkl')
index = SongIndex.load('models/index_phase1')

vec = extract_librosa_features(SONG_PATH)
cluster_id, strength = assign_cluster(vec, scaler, reducer, clusterer, pca=pca)

print(f'Cluster: {cluster_id}  (strength: {strength:.3f})')

results = index.query(scaler.transform(vec.reshape(1, -1))[0], top_k=10)
for r in results:
    print(f"  [{r['rank']:2}] {r['score']:.4f}  {r['artist']} — {r['name']}  ({r['genre']})")
```

Run this as a Python script in a Bash tool. If the pipeline files don't exist, tell the user to run notebooks 01 and 02 first.

If cluster_id == -1, explain that the song is an outlier (noise in HDBSCAN terms) and suggest lowering min_cluster_size in cluster.py or running /refit-clusters.
