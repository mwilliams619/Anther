---
name: music-mentor-finetune
description: Build or retrain a lightweight, locally-served music-mentor chatbot on one consumer GPU (~16 GB). Combines a QLoRA fine-tune of Qwen2.5-7B-Instruct for TONE, a bge-small RAG retriever over advice passages for FACTS, and the Anther audio-similarity engine plus graph tools for "what do I sound like" workflows. Use when fine-tuning a chat model for voice/persona on a few hundred Q&A pairs, grounding the model in a source corpus via retrieval, or re-running this mentor pipeline with intent routing and graph-tool calling. Carries the QLoRA hyperparameters, VRAM budget, and the offline-download / numpy-pickle / transformers-5 gotchas that otherwise cost hours.
---

# Music Mentor fine-tune pipeline

A three-part small-model assistant that runs entirely on one ~16 GB GPU
(built and verified on an RTX 4080 SUPER). The design keeps **tone, facts, and
audio-similarity as three separate mechanisms** — fine-tuning a 7B model on a
few hundred pairs reliably moves *style* but does not install *facts*, so tone
is fine-tuned and content is grounded by retrieval.

| Job | Mechanism | Component |
|---|---|---|
| Talks like the source | QLoRA fine-tune (tone only) | `mentor_train.py` → LoRA adapter |
| Defers to source info | RAG over advice passages | `mentor_rag.py` (bge-small) |
| Says what a track sounds like | Anther similarity engine as a tool | `mentor_anther.py` |
| Graph operations | strict-JSON ReAct + deterministic fallback | `mentor_graph.py`, `mentor_react.py` |
| Assembled agent | base+LoRA + RAG + Anther | `mentor.py` |
| Demo + VRAM probe | end-to-end battery | `mentor_demo.py` |

Verified footprint in v3/7B: training and serving remain inside a 16 GB card.
Track VRAM explicitly in `regression_v3.json` and `mentor_demo.py` outputs.

## Environment (the `mentor` conda env)

python=3.11 with **numpy<2** (pinned — the torch/bitsandbytes stack wants it).
Install order that works:

1. Base: `torch transformers>=4.44 peft trl bitsandbytes accelerate datasets sentence-transformers` (pip).
2. **CUDA build matters.** Default pip may give a `+cu130` wheel whose runtime
   is newer than the driver → `torch.cuda.is_available()` is False. Force the
   matching CUDA build. For a 12.x driver, install the cu124 wheel explicitly:
   `torch==2.5.1+cu124` from `https://download.pytorch.org/whl/cu124/` (needs
   `request_network_access("download.pytorch.org")`). If you install it with
   `--no-deps`, you must ALSO install the `nvidia-*-cu12` runtime packages
   (cublas 12.4.5.8, cuda-runtime 12.4.127, cudnn 9.1.0.70, cufft, cusolver,
   cusparse, nccl 2.21.5, nvjitlink) or you get `libcudart.so.12 not found`.
3. For the Anther tool: `librosa soundfile umap-learn hdbscan` (Anther's
   `__init__` imports `features.py` → librosa; `place()` pulls in umap/hdbscan).

Verify: `torch.cuda.is_available()` True, a `Linear4bit` forward runs.

## Data prep

Source pairs are `{instruction, input, output}` dicts. Two real-world traps in
the original data — always inspect before trusting counts:
- `music_qa.json` was **77% empty padding** (1191 of 1548 rows blank); only 356
  had content. Filter out rows with blank instruction+input+output.
- Two distinct voices: real answers median **65 words** (punchy Reddit-mentor),
  synthetic median **240 words** (long coach-essays). **Train tone on the real,
  punchy answers only**; use ALL deduped pairs (406) for the RAG corpus.

Outputs: `train.jsonl`/`val.jsonl` (chat `messages` format, 90/10 split,
system prompt below) and `rag_passages.jsonl` (keys: `id, source, question,
passage`). System prompt used for both fine-tune and serving:

> You are a seasoned music mentor for aspiring artists. You give direct,
> encouraging, no-nonsense advice grounded in real experience. Be concrete and
> honest; skip empty platitudes. When source advice is provided, defer to it
> and weave it into your answer in your own voice.

## QLoRA hyperparameters (in `mentor_train.py`)

Base `Qwen/Qwen2.5-7B-Instruct` (ungated, Apache-2.0). 4-bit NF4 double-quant,
bf16 compute. LoRA r=16, alpha=32, dropout=0.05, targets
q/k/v/o/gate/up/down_proj. 3 epochs, per_device_batch=2 x grad_accum=8 (eff 16),
lr 2e-4 cosine, warmup_ratio 0.05, max_length=1024, `assistant_only_loss=True`
(Qwen chat template has `{% generation %}` markers so loss lands on assistant
tokens only), gradient_checkpointing (use_reentrant=False), optim
paged_adamw_8bit. ~134 s for 60 steps; train loss 3.31 → ~2.06. Adapter is
~60 MB.

## Gotchas that cost hours (READ before re-running)

1. **Offline flags are mandatory.** `from_pretrained` HANGS with the GPU idle
   (phones home through the sandbox SOCKS proxy) unless you set
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and pass `local_files_only=True`.
   With them, the 3B loads 4-bit in ~2 s at 2.06 GB. All scripts set these at
   the top.
2. **Download weights with curl, not the HF client.** HuggingFace's Xet
   chunked-transfer protocol STALLS in the sandbox (only JSON metadata lands).
   Classic streaming works (~3.4 MB/s). Download the `.safetensors` shards with
   `curl` in a background bash cell to `models_cache/<model>/`. Needs
   `request_network_access` for `huggingface.co`, `cdn-lfs.huggingface.co`, and
   `us.aws.cdn.hf.co` (the 302 redirect target). Same trick for `bge-small-en-v1.5`.
3. **`nohup ... &` does not survive** a bash cell — children are reaped when the
   launcher cell completes. Run training as the cell's OWN foreground process in
   a `background:true` bash cell (`stdbuf -oL python mentor_train.py | tee log`).
4. **Anther `leiden.pkl` won't unpickle under numpy 1.26** — it was pickled
   under numpy 2.x (`state is not a legacy MT19937 state`). Don't upgrade numpy
   (breaks the torch stack). In mentor code, do not load `leiden.pkl`; recover
   clusters from `labels.npy` via neighbor voting, and hydrate labels from
   `cluster_profiles.json`.
5. **transformers 5.x chat-template change.** `apply_chat_template(...,
   return_tensors="pt")` returns a `BatchEncoding` dict, not a tensor. Use
   `return_dict=True`, read `enc["input_ids"].shape[1]` for the prompt length,
   and call `model.generate(**enc, ...)`.
6. **Give the "sounds like" readout its own method.** Routing the Anther
   analysis through the generic advice+RAG `chat()` made the small model
   hallucinate an identity statement. `mentor.py:sounds_like()` skips RAG, uses
   tighter framing ("Do not invent other artists"), temperature 0.5.

## The Anther tool

Frozen bundle `models/corpus_corpus_mpd_100k/` = 99,618 MPD tracks, MERT-v1-330M
embeddings (1024-dim, sr 24000). `AntherSoundsLike` now supports:
- `resolve_anchor(spec)` for audio or fuzzy name resolution.
- `cluster_of(vec)` using `labels.npy` (no `leiden.pkl`).
- deterministic graph operations (`sounds_like`, `bridge`, `crossover`, `tagmates`, `resolve`).

Validate without audio by feeding real corpus vectors (`embeddings.npy` rows) as
stand-in queries — a track should self-match at sim ~1.0.

## Retraining / tweaking checklist

- New advice data → redo data prep (watch for padding + voice split), then
  `mentor_train.py` (adjust epochs/lr if the set is much bigger).
- Bigger base (e.g. 8B) → change the base id, re-curl weights, keep everything
  else; serving still fits 16 GB.
- Better coherence → lower serving temperature; the RAG cosine scores in
  `demo_results.json` show when grounding was weak (< ~0.5 = off-corpus).
- Always re-run `mentor_demo.py` after a change — it prints peak serving VRAM
  and writes a fresh transcript.

## Files (saved as project artifacts)
`mentor_train.py`, `mentor_rag.py`, `mentor_anther.py`, `mentor.py`,
`mentor_demo.py`, `mentor_lora.tar.gz` (adapter), `rag_index.npy` +
`rag_index_meta.json`, `mentor_report.md`, `training_loss.png`, `data_stats.png`.
