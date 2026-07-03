"""
Background cluster job runner. Single job at a time; MERT loaded once.
"""

import sys, threading, time, tempfile, pickle
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from anther_ml.embedding import load_mert, get_embedding
from anther_ml.cluster   import fit_clusters, save_pipeline
from anther_ml.similarity import SongIndex
from anther_ml.spotify_deezer import _deezer_get

MERT_SR      = 24000
SESSION_DIR  = Path(__file__).parent / 'session'

# ── Shared state ────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state = {'job_id': None, 'state': 'idle', 'progress': 0, 'total': 0,
          'message': '', 'result': None}

_model     = None
_processor = None
_device    = None


def get_status(job_id: str) -> dict:
    with _lock:
        if _state['job_id'] != job_id:
            return {'state': 'not_found'}
        return dict(_state)


def start_cluster_job(manifest: list) -> str:
    job_id = str(int(time.time() * 1000))
    with _lock:
        _state.update(job_id=job_id, state='running', progress=0,
                      total=len(manifest), message='Starting…', result=None)
    t = threading.Thread(target=_run, args=(manifest, job_id), daemon=True)
    t.start()
    return job_id


# ── Internal ─────────────────────────────────────────────────────────────────

def _update(msg: str, progress: int | None = None):
    with _lock:
        _state['message'] = msg
        if progress is not None:
            _state['progress'] = progress


def _load_mert_once():
    global _model, _processor, _device
    if _model is None:
        _update('Loading MERT model (one-time, ~30 s)…')
        _model, _processor, _device = load_mert()
    return _model, _processor, _device


def _embed_url(model, processor, device, url: str) -> np.ndarray:
    """Download a Deezer preview URL to a temp MP3, embed, delete."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = Path(tmp.name)
    try:
        return get_embedding(model, processor, tmp_path, device)
    finally:
        tmp_path.unlink(missing_ok=True)


def _fresh_preview_url(deezer_id) -> str | None:
    """Re-fetch the track to get a non-expired preview URL."""
    data = _deezer_get(f'track/{deezer_id}')
    return data.get('preview')


def _run(manifest: list, job_id: str):
    try:
        _do_cluster(manifest, job_id)
    except Exception as exc:
        with _lock:
            _state.update(state='error', message=str(exc))


def _do_cluster(manifest: list, job_id: str):
    model, processor, device = _load_mert_once()

    embeddings: list[np.ndarray] = []
    metadata:   list[dict]       = []
    n = len(manifest)

    for i, item in enumerate(manifest):
        label = item.get('title') or item.get('filename') or f'track {i+1}'
        _update(f'Embedding {i+1}/{n}: {label}', progress=i)
        try:
            if item['type'] == 'deezer':
                url = _fresh_preview_url(item['deezer_id'])
                if not url:
                    _update(f'  Skipping "{label}": preview URL unavailable', progress=i)
                    continue
                emb = _embed_url(model, processor, device, url)
            else:
                emb = get_embedding(model, processor, item['path'], device)

            embeddings.append(emb)
            metadata.append({
                'name':       item.get('title') or item.get('filename', ''),
                'artist':     item.get('artist', ''),
                'source':     item['type'],
                'deezer_id':  item.get('deezer_id'),
                'genre':      'unknown',
            })
        except Exception as exc:
            _update(f'  Failed "{label}": {exc}', progress=i)

    if len(embeddings) < 10:
        raise ValueError(
            f'Only {len(embeddings)} tracks embedded successfully (need ≥ 10).'
            ' Check network or try different tracks.'
        )

    X = np.vstack(embeddings)
    m = len(X)
    umap_k  = min(32, max(2, m // 10))
    min_cls = max(3, m // 20)
    _update(f'Clustering {m} tracks  (UMAP {umap_k}D, HDBSCAN min={min_cls})…', progress=m)

    labels, emb2d, scaler, pca, reducer, clusterer, reducer_2d = fit_clusters(
        X, n_umap_components=umap_k, min_cluster_size=min_cls
    )

    for idx, meta in enumerate(metadata):
        meta['cluster'] = int(labels[idx])

    _update('Saving results…', progress=m)
    SESSION_DIR.mkdir(exist_ok=True)
    np.save(SESSION_DIR / 'embeddings.npy', X)
    with open(SESSION_DIR / 'metadata.pkl', 'wb') as fh:
        pickle.dump(metadata, fh)
    save_pipeline(SESSION_DIR / 'pipeline_phase2.pkl',
                  scaler, reducer, clusterer, pca=pca, reducer_2d=reducer_2d)
    index = SongIndex(X, metadata)
    index.save(str(SESSION_DIR / 'index_phase2'))
    np.save(SESSION_DIR / 'labels_phase2.npy', labels)
    np.save(SESSION_DIR / 'embedding_2d_phase2.npy', emb2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = float((labels == -1).mean() * 100)

    with _lock:
        _state.update(
            state='done', progress=m,
            message=f'Done — {n_clusters} clusters, {noise_pct:.0f}% noise.',
            result={'n_tracks': m, 'n_clusters': n_clusters, 'noise_pct': round(noise_pct, 1)},
        )
