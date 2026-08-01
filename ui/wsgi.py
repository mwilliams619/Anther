"""
Production entrypoint for the Anther UI.

``python ui/app.py`` starts Werkzeug's *development* server — no request
timeouts, no connection limit, no hardened HTTP parsing — and both systemd
units used to point straight at it. This module serves the same Flask app
under waitress instead. ``ui/app.py`` still runs the dev server directly for
local work; nothing about that changed.

Why waitress rather than gunicorn
---------------------------------
The app keeps process-local state that cannot survive being forked across
workers:

  * ``playlist_jobs._jobs`` and its single worker thread — the job registry
    is a module global, so with >1 worker a ``/api/playlist/status/<id>``
    poll can land on a process that has never heard of that job;
  * ``atlas._sessions`` — the per-browser map registry;
  * ``atlas._corpus`` and the MERT model — multi-GB, and every worker would
    load its own copy.

So this has to be one process with many threads, which is waitress's native
shape. gunicorn would additionally need ``--workers 1``, and its worker
timeout *kills and replaces* the worker — which here would silently destroy
in-flight background placements along with the whole job registry.

Run:
    python ui/wsgi.py

Env:
    ANTHER_UI_HOST     default 127.0.0.1 (the tunnel connects locally; set
                       0.0.0.0 only if something must reach it over the LAN)
    ANTHER_UI_PORT     default 5000
    ANTHER_UI_THREADS  default 8
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import app  # noqa: E402  — also applies the dotenv load in app.py
import atlas         # noqa: E402


def main() -> None:
    from waitress import serve

    atlas.warm()      # load the frozen corpus in the background at startup

    host    = os.environ.get('ANTHER_UI_HOST', '127.0.0.1')
    port    = int(os.environ.get('ANTHER_UI_PORT', '5000'))
    threads = int(os.environ.get('ANTHER_UI_THREADS', '8'))

    print(f"[wsgi] serving Anther on {host}:{port} "
          f"({threads} threads, waitress)")
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        # Single source of truth with Flask's own cap, so an oversized body is
        # refused at the server layer too rather than only after Werkzeug has
        # parsed the request.
        max_request_body_size=app.config['MAX_CONTENT_LENGTH'],
        # Drop idle/half-open connections (slowloris). Generous because a
        # placement request legitimately spends a long time downloading and
        # running MERT before it writes a byte of response.
        channel_timeout=int(os.environ.get('ANTHER_UI_CHANNEL_TIMEOUT', '300')),
        # Bound concurrent connections so a flood queues rather than growing
        # the process without limit.
        connection_limit=int(os.environ.get('ANTHER_UI_CONNECTION_LIMIT', '200')),
        ident='anther',
    )


if __name__ == '__main__':
    main()
