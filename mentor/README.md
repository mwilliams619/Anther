# Music Mentor

A music mentor sitting inside an interactive spatial sound map — the second
feature of the Anther app. It gives artists advice grounded in real experience,
reads the sounds on their map with the Anther similarity engine, and explains
why songs are placed where they are.

v4+ architecture: The LLM does exactly two jobs at the pipeline's ends:
  1. **INTERPRET** — classify intent (ADVICE / GRAPH / OFFTOPIC) in one call
  2. **NARRATE** — rephrase deterministic observations in mentor voice

Everything in the middle is deterministic tool logic — no ReAct loops, no regex
question classifiers, no hallucination risk. The graph UI is the user's memory.

Runs entirely on one ~16 GB consumer GPU. Verified footprint: **training peaks
4.51 GB**, **full serving peaks 2.56 GB** (Qwen2.5-7B-Instruct + LoRA + RAG +
Anther graph tools).

## Layout

```
mentor/                       this package (source, tracked in git)
  paths.py                    central path resolution (repo-relative)
  mentor.py                   MusicMentor — three-branch router (ADVICE/GRAPH/OFFTOPIC)
  agent.py                    MentorAgent — intent classifier + narrator
  graph_tools.py              seven read-only tools over the on-screen map + corpus
  graph_context.py            GraphContext (live on-screen state), MentorContext (session)
  anther_service.py           AntherSimilarityService — similarity + cluster + tags
  mentor_rag.py               bge-small RAG retriever over advice passages
  mentor_train.py             QLoRA fine-tune of Qwen2.5-7B-Instruct
  mentor_demo.py              end-to-end demo + VRAM probe
  mentor_chat.py              interactive terminal REPL
  fetch_models.py             download base weights for a fresh clone
  SKILL.md                    build/retrain playbook (gotchas + hparams)

models/mentor/                weights + data (NOT in git — see .gitignore)
  Qwen2.5-7B-Instruct/        base model (fetch_models.py)
  bge-small-en-v1.5/          retriever embedder (fetch_models.py)
  mentor_lora_7b/             the trained LoRA adapter (~78 MB)
  rag_index.npy, rag_index_meta.json, rag_passages.jsonl
  train.jsonl, val.jsonl
  demo_transcript.md, demo_results.json

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
m.chat("What do I sound like?", audio_path="demo.wav")   # audio → similarity readout
m.chat("Why is this node here?", selected_node_id="abc123")   # graph question
```

## The seven graph tools

All seven are read-only and route through the agent:

| Intent | Purpose | Returns |
|--------|---------|---------|
| `inspect` | What's on the map right now? | visible nodes, territories, groups, selection |
| `resolve` | Who/what does a reference mean? | matched artist/song, type (graph/library) |
| `neighbors` | What does X sound like? | closest songs with *why* (shared traits, territory) |
| `compare` | How do two sounds relate? | summary, shared/different traits, bridge candidates |
| `bridge` | What sits between X and Y? | midpoint tracks with *why* (lean, shared traits) |
| `explore` | Where to explore next from X? | adjacent territory, nearby artists, direction |
| `explain` | Why is X placed here? | neighbours, territory, traits, position summary |

The tool layer synthesizes `"why"` strings from deterministic evidence:
shared micro-genre tags, cluster (territory) labels, and lean/balance scores
(for bridges). No raw ML detail (embeddings, cosines, indices) reaches the LLM.

## Retrain / tweak

See **SKILL.md** for the full playbook — QLoRA hyperparameters, the VRAM budget,
and the offline-download / numpy-pickle / transformers-5 gotchas that otherwise
cost hours. In short:

```bash
python mentor/mentor_train.py     # QLoRA fine-tune -> models/mentor/mentor_lora_7b/
python mentor/mentor_rag.py       # (re)build the RAG index
python mentor/mentor_demo.py      # verify + print peak serving VRAM
```

For acceptance artifacts, also generate:
- `models/mentor/demo_transcript.md` (Q&A readout)
- `models/mentor/demo_results.json` (raw results + VRAM)

## Web UI integration

`mentor/service.py` is that thin inference service: it loads `MusicMentor`
once (~2.6 GB VRAM, ~10s) and stays warm, independent of the Flask UI's
(`ui/app.py`) process lifecycle. Run it alongside the UI:

```bash
python -m mentor.service   # port 5100 by default (ANTHER_MENTOR_HOST/PORT)
```

It exposes:

- `POST /chat` — `{session_id, question, selected_node_id}` → `{answer}`
- `POST /reset` — `{session_id}` → `{ok}`
- `GET /health` — `{ok, model_loaded}`

Sessions are in-memory (`MentorContext` per `session_id`), with idle
entries swept after an hour — no disk persistence, so a service restart
resets everyone's chat history. `ui/app.py` forwards `/api/mentor/chat` and
`/api/mentor/reset` to this service and assigns each browser a `session_id`
via a signed cookie.

The frontend sends `selected_node_id` (the pinned node on the map, or null)
with every chat turn so the mentor can resolve "this song", "why is this here",
or "what do I sound like" against the user's selection. The graph UI is the
user's memory.
