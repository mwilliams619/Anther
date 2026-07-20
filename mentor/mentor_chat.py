#!/usr/bin/env python
"""Interactive music-mentor chat.

Run from anywhere:
    /home/matt/.claude-science/conda/envs/mentor/bin/python <this file>

All paths are resolved relative to this file's own location, so cwd does not
matter — you can launch it from the Anther directory or any other.

Commands:
    (just type a question)      ask for advice (RAG-grounded, in mentor voice)
    /sound <path-to-audio>      "what do I sound like" on an audio file
    /quit  or  Ctrl-D           exit
"""
import os, sys

# All model paths resolve via paths.py (relative to the repo root), so cwd
# does not matter -- launch from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)       # so `import mentor` / `import paths` resolve
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mentor import ConversationState, MusicMentor

print("Loading the mentor (base + LoRA + RAG + Anther)... ~10s")
m = MusicMentor()
ctx = ConversationState()
print("Ready. Ask a question, '/sound <audio file>' for a similarity read, or '/quit'.\n")

while True:
    try:
        q = input("you > ").strip()
    except EOFError:
        print()
        break
    if not q:
        continue
    if q in ("/quit", "/exit"):
        break
    if q == "/reset":
        print("\nmentor > " + m.chat("/reset", conversation_context=ctx) + "\n")
        continue
    if q.startswith("/sound"):
        parts = q.split(maxsplit=1)
        if len(parts) < 2:
            print("usage: /sound <path-to-audio-file>\n")
            continue
        path = os.path.expanduser(parts[1].strip())
        if not os.path.exists(path):
            print(f"no such file: {path}\n")
            continue
        print("\nmentor > " + m.chat("What do I sound like?", audio_path=path, conversation_context=ctx) + "\n")
        continue
    print("\nmentor > " + m.chat(q, conversation_context=ctx) + "\n")

print("later.")
