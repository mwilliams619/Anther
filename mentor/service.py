"""Warm HTTP inference service for the music mentor.

Loads ``MusicMentor`` once (base model + LoRA + RAG + Anther graph tools,
~10s load, ~2.6GB VRAM resident per mentor/README.md) and keeps it warm, so
the Anther UI backend (ui/app.py) never loads it per-request. Run as its own
long-lived process, independent of the Flask UI's lifecycle:

    python -m mentor.service

The UI backend talks to it over localhost (ANTHER_MENTOR_HOST/PORT) and
threads a browser-issued session_id through /chat and /reset so each
browser session gets its own ConversationState. Idle sessions are evicted
after SESSION_TTL_SECONDS so a restarted UI (whose cookies no longer match
any live session here) doesn't leak memory over time.
"""
import os
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, jsonify, request

try:
    from .mentor import ConversationState, MusicMentor
except ImportError:  # pragma: no cover - script-style import path
    from mentor import ConversationState, MusicMentor

SESSION_TTL_SECONDS = 60 * 60  # idle sessions older than this are dropped

app = Flask(__name__)

_mentor = None
_sessions: dict[str, dict] = {}   # session_id -> {"state": ConversationState, "last_used": float}
_sessions_lock = threading.Lock()


def _sweep_locked():
    cutoff = time.time() - SESSION_TTL_SECONDS
    stale = [sid for sid, rec in _sessions.items() if rec["last_used"] < cutoff]
    for sid in stale:
        del _sessions[sid]


def _get_state(session_id):
    with _sessions_lock:
        _sweep_locked()
        rec = _sessions.get(session_id)
        if rec is None:
            rec = {"state": ConversationState(), "last_used": time.time()}
            _sessions[session_id] = rec
        rec["last_used"] = time.time()
        return rec["state"]


def _reset_state(session_id):
    with _sessions_lock:
        _sweep_locked()
        _sessions[session_id] = {"state": ConversationState(), "last_used": time.time()}


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": _mentor is not None})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    question = data.get("question")
    # The node the user has selected on the map (or None) — the frontend sends
    # it with every turn so "this song" / "what do I sound like" resolve to it.
    selected_node_id = data.get("selected_node_id") or None
    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400
    state = _get_state(session_id)
    try:
        answer = _mentor.chat(question, conversation_context=state,
                              selected_node_id=selected_node_id)
    except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the service
        return jsonify({"error": f"mentor chat failed: {exc}"}), 500
    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    _reset_state(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Loading the mentor (base + LoRA + RAG + Anther)... ~10s")
    _mentor = MusicMentor()
    print("Mentor service ready.")
    app.run(host=os.environ.get("ANTHER_MENTOR_HOST", "127.0.0.1"),
            port=int(os.environ.get("ANTHER_MENTOR_PORT", "5100")),
            debug=False, use_reloader=False)
