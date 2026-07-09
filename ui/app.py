"""
Flask backend for the song atlas UI.
Run:  python ui/app.py
Open: http://localhost:5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import atlas

# ── Config ───────────────────────────────────────────────────────────────────

ALLOWED_EXTS  = {'.mp3', '.wav', '.flac', '.m4a'}
MAX_UPLOAD    = 25 * 1024 * 1024   # 25 MB
SESSION_DIR   = Path(__file__).parent / 'session'
UPLOADS_DIR   = SESSION_DIR / 'uploads'

SESSION_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')




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
        return jsonify(atlas.search_playlists(q, limit=limit))
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


@app.route('/api/playlist/status/<job_id>')
def playlist_status(job_id):
    import playlist_jobs
    cursor = int(request.args.get('cursor', 0))
    return jsonify(playlist_jobs.get_status(job_id, cursor))


@app.route('/api/albums/search')
def albums_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    try:
        limit = int(request.args.get('limit', 20))
        return jsonify(atlas.search_albums(q, limit=limit))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 502


@app.route('/api/album/place', methods=['POST'])
def album_place():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(atlas.place_album(body.get('album_id')))
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


@app.route('/api/graph/clear', methods=['POST'])
def graph_clear():
    atlas.clear_graph()
    return jsonify({'status': 'cleared'})


@app.route('/api/node/<path:node_id>', methods=['DELETE'])
def node_remove(node_id):
    result = atlas.remove_node(node_id)
    if result is None:
        return jsonify({'error': 'unknown node id'}), 404
    return jsonify(result)


@app.route('/api/song/<path:song_id>')      # <path:> — ids contain ':' and filenames
def atlas_song(song_id):
    try:
        detail = atlas.song_detail(song_id, top_n=int(request.args.get('n', 10)))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    if detail is None:
        return jsonify({'error': 'unknown song id'}), 404
    return jsonify(detail)




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




if __name__ == '__main__':
    import os
    atlas.warm()          # load the frozen corpus in the background at startup
    # 0.0.0.0 so the UI is reachable from other machines on the LAN
    # (e.g. a laptop browsing to http://<dev-box-ip>:5000) without VS Code
    # port forwarding. Dev server on a trusted network only.
    app.run(debug=True,
            host=os.environ.get('ANTHER_UI_HOST', '0.0.0.0'),
            port=int(os.environ.get('ANTHER_UI_PORT', '5000')),
            use_reloader=False)
