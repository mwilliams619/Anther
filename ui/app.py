"""
Flask backend for the song atlas UI.
Run:  python ui/app.py
Open: http://localhost:5000
"""

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file (repo root), without
    a new dependency and without clobbering real exported env vars — those
    still win over the file. Must run before `import atlas`, since its
    module-level config (CORPUS_DIR, MPD_DB, …) reads os.environ at import
    time; SPOTIFY_CLIENT_ID/SECRET are read lazily so either order works
    for those, but loading first covers both."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(Path(__file__).parent.parent / '.env')

import requests
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename

import atlas

# ── Config ───────────────────────────────────────────────────────────────────

ALLOWED_EXTS  = {'.mp3', '.wav', '.flac', '.m4a'}
MAX_UPLOAD    = 25 * 1024 * 1024   # 25 MB
SESSION_DIR   = Path(__file__).parent / 'session'
UPLOADS_DIR   = SESSION_DIR / 'uploads'

# mentor/service.py — a separate warm process (see mentor/README.md on why
# the model isn't loaded in this process); started independently.
MENTOR_HOST    = os.environ.get('ANTHER_MENTOR_HOST', '127.0.0.1')
MENTOR_PORT    = os.environ.get('ANTHER_MENTOR_PORT', '5100')
MENTOR_URL     = f'http://{MENTOR_HOST}:{MENTOR_PORT}'
MENTOR_TIMEOUT = float(os.environ.get('ANTHER_MENTOR_TIMEOUT', '30'))

SESSION_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder='static', static_url_path='/static')
# Signed session cookie identifies a browser's mentor chat session; a fresh
# key each restart just means chat history resets, which is fine since the
# mentor service's own sessions are keyed the same way and get swept by TTL.
app.secret_key = os.environ.get('ANTHER_UI_SECRET_KEY', secrets.token_hex(32))

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


@app.route('/api/playlist/stop/<job_id>', methods=['POST'])
def playlist_stop(job_id):
    import playlist_jobs
    try:
        return jsonify(playlist_jobs.stop(job_id))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


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


@app.route('/api/recommend', methods=['POST'])
def atlas_recommend():
    """Multi-song recommendation: body {seed_ids: [...], top_k?, method?}."""
    body = request.get_json(force=True) or {}
    seed_ids = body.get('seed_ids') or []
    try:
        top_k  = int(body.get('top_k', 20))
        method = body.get('method', 'centroid')
        return jsonify(atlas.recommend(seed_ids, top_k=top_k, method=method))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


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
    expand = request.args.get('expand', '').lower() in ('1', 'true', 'yes')
    try:
        detail = atlas.song_detail(song_id, top_n=int(request.args.get('n', 10)),
                                   expand=expand)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    if detail is None:
        return jsonify({'error': 'unknown song id'}), 404
    return jsonify(detail)


@app.route('/api/song/<path:song_id>/preview')
def atlas_song_preview(song_id):
    try:
        url = atlas.get_preview_url(song_id)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'preview_url': url})


@app.route('/api/song/<path:song_id>/spotify')
def atlas_song_spotify(song_id):
    """Bare Spotify track id for the no-login iframe embed (full track for
    visitors already logged into Spotify in that browser; 30s preview
    otherwise — no OAuth, no app registration, no per-user quota)."""
    try:
        track_id = atlas.get_spotify_track_id(song_id)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'track_id': track_id})




@app.route('/api/mentor/chat', methods=['POST'])
def mentor_chat():
    """Forward a chat turn to the warm mentor service (mentor/service.py)."""
    body = request.get_json(force=True) or {}
    question = (body.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'question is required'}), 400
    session_id = session.get('mentor_session_id')
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        session['mentor_session_id'] = session_id
    try:
        resp = requests.post(f'{MENTOR_URL}/chat',
                              json={'session_id': session_id, 'question': question},
                              timeout=MENTOR_TIMEOUT)
    except requests.RequestException:
        return jsonify({'error': 'Mentor is currently unavailable.'}), 503
    if resp.status_code != 200:
        try:
            err = resp.json().get('error', 'mentor chat failed')
        except ValueError:
            err = 'mentor chat failed'
        return jsonify({'error': err}), 502
    return jsonify(resp.json())


@app.route('/api/mentor/reset', methods=['POST'])
def mentor_reset():
    session_id = session.get('mentor_session_id')
    if not session_id:
        return jsonify({'ok': True})
    try:
        requests.post(f'{MENTOR_URL}/reset', json={'session_id': session_id}, timeout=MENTOR_TIMEOUT)
    except requests.RequestException:
        return jsonify({'error': 'Mentor is currently unavailable.'}), 503
    return jsonify({'ok': True})


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
    atlas.warm()          # load the frozen corpus in the background at startup
    # 0.0.0.0 so the UI is reachable from other machines on the LAN
    # (e.g. a laptop browsing to http://<dev-box-ip>:5000) without VS Code
    # port forwarding. Dev server on a trusted network only.
    app.run(debug=os.environ.get('ANTHER_DEBUG') == '1',
            host=os.environ.get('ANTHER_UI_HOST', '0.0.0.0'),
            port=int(os.environ.get('ANTHER_UI_PORT', '5000')),
            use_reloader=False)
