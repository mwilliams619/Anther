# Music Mentor

A lightweight, locally-served advice chatbot for aspiring artists — the second
feature of the Anther app. It **talks like** the source community (a QLoRA
fine-tune for tone), **defers to** the source advice (RAG retrieval), and can
tell an artist **what their track sounds like** by reusing Anther's similarity
engine as a tool.

The v3 runtime uses an **intent-first router** (`ADVICE | GRAPH | OFFTOPIC`):
advice routes to grounded RAG, graph requests route to a strict-JSON ReAct
tool loop, and off-topic requests redirect in mentor voice.

Runs entirely on one ~16 GB consumer GPU. Verified footprint: **training peaks
4.51 GB**, **full serving peaks 2.56 GB**.

## Layout

```
mentor/                       this package (source, tracked in git)
  paths.py                    central path resolution (repo-relative)
  mentor.py                   MusicMentor — intent router + assembled agent
  mentor_rag.py               bge-small RAG retriever over advice passages
  mentor_anther.py            anchor resolution + cluster recovery + placement
  mentor_graph.py             deterministic graph tools
  mentor_react.py             strict-JSON ReAct harness + fallback router
  mentor_train.py             QLoRA fine-tune of Qwen2.5-7B-Instruct
  mentor_demo.py              end-to-end demo + VRAM probe
  mentor_chat.py              interactive terminal REPL
  fetch_models.py            download base weights for a fresh clone
  SKILL.md                    the build/retrain playbook (gotchas + hparams)

models/mentor/                weights + data (NOT in git — see .gitignore)
  Qwen2.5-7B-Instruct/        base model (fetch_models.py)
  bge-small-en-v1.5/          retriever embedder (fetch_models.py)
  mentor_lora_7b/             the trained LoRA adapter (~78 MB)
  mentor_lora_v2_backup/      rollback snapshot before v3 retrain
  rag_index.npy, rag_index_meta.json, rag_passages.jsonl
  train.jsonl, val.jsonl
  deflection_audit.json
  regression_v3.json
  graph_demo.md

models/corpus_corpus_mpd_100k/   shared Anther similarity corpus
```

## Setup

```bash
# from the repo root, in a python 3.11 env with numpy<2
pip install -e .[mentor]          # adds peft/trl/bitsandbytes/etc. on top of Anther
python mentor/fetch_models.py     # download base weights (~6 GB) — one time
```

Critical invariants:
- Keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set before model loads.
- Keep `numpy<2` (repo uses 1.26.x); do not load `leiden.pkl` in mentor code.
- Resolve corpus through `SongIndex` + `labels.npy` + `cluster_profiles.json`.

The base weights are large and not committed. `fetch_models.py` streams them
from HuggingFace (using a plain-GET path that avoids the Xet stall seen in
restricted networks). The trained adapter and RAG index live in
`models/mentor/` and are small enough to share directly.

CUDA note: install a torch build that matches your driver. For a CUDA 12.x
driver, `torch==2.5.1+cu124` from the pytorch cu124 wheel index is verified.

## Use

```bash
# interactive chat — launch from anywhere (paths resolve to the repo root)
python mentor/mentor_chat.py
```

```python
from mentor.mentor import MusicMentor
m = MusicMentor()                                   # ~2.3 GB VRAM, ~10 s load
m.chat("How do I get my first 100 real fans?")      # advice, RAG-grounded, in-voice
m.chat("What should I focus on?", audio_path="demo.wav")   # audio fast-path sounds-like
m.sounds_like(audio_path="demo.wav")                # pure 'what do I sound like'
m.chat("What artists sit between me and Noisia?")   # graph ReAct branch
```

## Retrain / tweak

See **SKILL.md** for the full playbook — QLoRA hyperparameters, the VRAM budget,
and the offline-download / numpy-pickle / transformers-5 gotchas that otherwise
cost hours. In short:

```bash
python mentor/mentor_train.py     # QLoRA fine-tune -> models/mentor/mentor_lora/
python mentor/mentor_rag.py       # (re)build the RAG index
python mentor/mentor_demo.py      # verify + print peak serving VRAM
```

For v3 acceptance artifacts, also generate:
- `models/mentor/deflection_audit.json`
- `models/mentor/regression_v3.json`
- `models/mentor/graph_demo.md`
- `models/mentor/training_loss_v3.png`

## Web UI integration

`mentor/service.py` is that thin inference service: it loads `MusicMentor`
once (~2.6 GB VRAM, ~10s) and stays warm, independent of the Flask UI's
(`ui/app.py`) process lifecycle. Run it alongside the UI:

```bash
python -m mentor.service   # port 5100 by default (ANTHER_MENTOR_HOST/PORT)
```

It exposes:

- `POST /chat` — `{session_id, question}` → `{answer}`
- `POST /reset` — `{session_id}` → `{ok}`
- `GET /health` — `{ok, model_loaded}`

Sessions are in-memory (`ConversationState` per `session_id`), with idle
entries swept after an hour — no disk persistence, so a service restart
resets everyone's chat history. `ui/app.py` forwards `/api/mentor/chat` and
`/api/mentor/reset` to this service and assigns each browser a `session_id`
via a signed cookie; see [docs/ui.md](../docs/ui.md#mentor-chat) for the
full contract.

The web chat's ReAct tool surface is intentionally narrower than what
`mentor_graph.py`'s `TOOLS` registry supports: `mentor_react.py`'s
`REACT_SYSTEM`/`_execute()` only declare and dispatch the read-only graph
lookups (`resolve, sounds_like, compare, bridge, crossover, artist_tracks,
tagmates`) — nothing that places, removes, or otherwise mutates the map. The
atlas-primitive wrappers on `MentorGraphTools` (`search*`, `place_*`,
`recommend`, `clear_graph`, `remove_node`, …) still exist for potential
future use but aren't reachable from chat today.
