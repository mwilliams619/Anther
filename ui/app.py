"""
Flask backend for the song staging + clustering UI.
Run:  python ui/app.py
Open: http://localhost:5000
"""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

from anther_ml.spotify_deezer import _deezer_get
import jobs
import atlas

# ── Config ───────────────────────────────────────────────────────────────────

ALLOWED_EXTS  = {'.mp3', '.wav', '.flac', '.m4a'}
MAX_UPLOAD    = 25 * 1024 * 1024   # 25 MB
SESSION_DIR   = Path(__file__).parent / 'session'
UPLOADS_DIR   = SESSION_DIR / 'uploads'
MANIFEST_PATH = SESSION_DIR / 'manifest.json'

SESSION_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ── Manifest helpers ─────────────────────────────────────────────────────────

def load_manifest() -> list:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return []

def save_manifest(m: list):
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/deezer/search')
def deezer_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    data = _deezer_get('search/track', params={'q': q, 'limit': 25})
    if 'error' in data:
        return jsonify({'error': data['error'].get('message', 'Deezer error')}), 502
    results = []
    for h in (data.get('data') or []):
        if not h.get('preview'):
            continue    # skip tracks with no 30s preview
        results.append({
            'deezer_id':   h['id'],
            'title':       h.get('title', ''),
            'artist':      (h.get('artist') or {}).get('name', ''),
            'album':       (h.get('album')  or {}).get('title', ''),
            'cover':       (h.get('album')  or {}).get('cover_small', ''),
            'preview_url': h['preview'],
            'duration':    h.get('duration', 0),
        })
    return jsonify(results)


# ── Atlas: tiered search + force-graph placement ─────────────────────────────

@app.route('/api/search')
def atlas_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': [], 'tiers': {}})
    try:
        return jsonify(atlas.search(q))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 502


@app.route('/api/playlists/search')
def playlists_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    try:
        limit = int(request.args.get('limit', 20))
        return jsonify({'results': atlas.search_playlists(q, limit=limit)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 502


@app.route('/api/playlist/place', methods=['POST'])
def playlist_place():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(atlas.place_playlist(body.get('pid')))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/place', methods=['POST'])
def atlas_place():
    result = request.get_json(force=True) or {}
    try:
        fragment = atlas.place_song(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify(fragment)


@app.route('/api/graph')
def atlas_graph():
    return jsonify(atlas.get_graph())


@app.route('/api/song/<path:song_id>')      # <path:> — ids contain ':' and filenames
def atlas_song(song_id):
    try:
        detail = atlas.song_detail(song_id, top_n=int(request.args.get('n', 10)))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    if detail is None:
        return jsonify({'error': 'unknown song id'}), 404
    return jsonify(detail)


@app.route('/api/stage', methods=['GET'])
def stage_get():
    return jsonify(load_manifest())


@app.route('/api/stage', methods=['POST'])
def stage_add():
    hit      = request.get_json(force=True) or {}
    manifest = load_manifest()
    item_id  = f"deezer_{hit.get('deezer_id')}"
    if any(m['id'] == item_id for m in manifest):
        return jsonify({'status': 'duplicate', 'count': len(manifest)})
    manifest.append({'id': item_id, 'type': 'deezer', **hit})
    save_manifest(manifest)
    return jsonify({'status': 'added', 'count': len(manifest)})


@app.route('/api/stage/<path:item_id>', methods=['DELETE'])
def stage_remove(item_id):
    manifest = [m for m in load_manifest() if m['id'] != item_id]
    save_manifest(manifest)
    return jsonify({'status': 'removed', 'count': len(manifest)})


@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': f'Unsupported type: {ext}. Use MP3, WAV, FLAC or M4A.'}), 400
    data = f.read()
    if len(data) > MAX_UPLOAD:
        return jsonify({'error': 'File too large (max 25 MB)'}), 400

    safe = secure_filename(f.filename)
    dest = UPLOADS_DIR / safe
    dest.write_bytes(data)

    # Place the uploaded file onto the frozen-corpus force graph.
    try:
        fragment = atlas.place_song({
            'source': 'upload',
            'id':     f'upload:{safe}',
            'title':  Path(safe).stem,
            'artist': 'personal',
            'path':   str(dest),
        })
    except Exception as exc:
        return jsonify({'error': f'Placement failed: {exc}'}), 500
    return jsonify({'status': 'uploaded', 'id': f'upload:{safe}',
                    'title': Path(safe).stem, 'fragment': fragment})


@app.route('/api/cluster', methods=['POST'])
def cluster_start():
    manifest = load_manifest()
    if len(manifest) < 15:
        return jsonify({'error': f'Need at least 15 tracks (have {len(manifest)})'}), 400
    job_id = jobs.start_cluster_job(manifest)
    return jsonify({'job_id': job_id})


@app.route('/api/cluster/status/<job_id>')
def cluster_status(job_id):
    return jsonify(jobs.get_status(job_id))


@app.route('/api/results')
def results():
    emb_path    = SESSION_DIR / 'embedding_2d_phase2.npy'
    labels_path = SESSION_DIR / 'labels_phase2.npy'
    meta_path   = SESSION_DIR / 'metadata.pkl'
    if not (emb_path.exists() and labels_path.exists() and meta_path.exists()):
        return jsonify({'error': 'no results yet'}), 404
    embedding_2d = np.load(emb_path).tolist()
    labels       = np.load(labels_path).tolist()
    with open(meta_path, 'rb') as fh:
        metadata = pickle.load(fh)
    return jsonify({'embedding_2d': embedding_2d, 'labels': labels, 'metadata': metadata})


if __name__ == '__main__':
    atlas.warm()          # load the frozen corpus in the background at startup
    app.run(debug=True, port=5000, use_reloader=False)
