# Music Mentor v3 — Implementation Plan (Claude Code handoff)

**Scope:** two work items, in order.
**Part 1 — Scope gate + data cleanup + LoRA retrain** (correctness: kill the AI-assistant/deflection voice and the off-topic leaks at the source).
**Part 2 — Graph tool harness** (capability: a native tool-calling / ReAct loop over the Anther similarity graph).

This document is written to be executed by Claude Code against the live repo. Every path, signature, and number below was verified against the working tree on 2026-07-10. Do Part 1 fully (including validation) before starting Part 2 — Part 2's routing depends on the intent classifier built in Part 1.

The plan follows a define-done discipline: each phase ends with a **Done when** block naming a concrete, observable check. Do not mark a phase complete until its named check passes. Do not narrate phase numbers in anything the end user reads.

---

## 0. Environment & ground truth (read before touching anything)

**Repo root:** `/home/matt/Dev/Anther`
**Conda env:** `mentor` — activate with the repo's interpreter at `/home/matt/.claude-science/conda/envs/mentor/bin/python`. Key versions (do not change): torch 2.5.1+cu124, transformers 5.13.0, peft 0.19.1, trl 1.7.1, bitsandbytes 0.49.2, sentence-transformers 5.6.0, **numpy 1.26.4** (pinned — see invariant below).
**GPU:** RTX 4080 SUPER, 16 GiB. Training peaks ~4.5 GB, serving ~2.6 GB. You have ample headroom.

### Package layout (source, git-tracked)
`/home/matt/Dev/Anther/mentor/`
- `paths.py` — central path resolver. **Import all paths from here; never hardcode.** Exposes `REPO_ROOT, ASSETS, BASE_DIR, EMB_DIR, LORA_DIR, INDEX_NPY, INDEX_META, PASSAGES, TRAIN_JSONL, VAL_JSONL, CORPUS_DIR`.
- `mentor.py` — `MusicMentor` agent (load + `chat()` + `sounds_like()`).
- `mentor_rag.py` — `MentorRAG` retriever; `retrieve(query, top_k)` returns `(hits, max_score)`.
- `mentor_anther.py` — `AntherSoundsLike` graph/similarity tool.
- `mentor_train.py` — QLoRA training script.
- `mentor_demo.py`, `mentor_chat.py`, `fetch_models.py`, `__init__.py`, `README.md`, `SKILL.md`.

### Assets (weights + data, gitignored under `models/`)
`/home/matt/Dev/Anther/models/mentor/`
- `Qwen2.5-3B-Instruct/` (base weights), `bge-small-en-v1.5/` (RAG embedder), `mentor_lora/` (current adapter).
- `train.jsonl` (319 rows), `val.jsonl` (36 rows) — chat-format fine-tune data.
- `rag_passages.jsonl` (v1, 406), `rag_passages_v2.jsonl` (v2, 371 — deflections already removed on the RAG side).
- `rag_index_v2.npy` / `rag_index_meta_v2.json` (371×384, the live index).

### Shared Anther corpus (frozen, gitignored)
`/home/matt/Dev/Anther/models/corpus_corpus_mpd_100k/`
- `index.npy` (99618×1024 MERT embeddings) + `index.json` (per-track metadata: `id, name, artist, source, genre, playlists`). Loaded via `SongIndex.load(<dir>/index)`.
- `labels.npy` — `(99618,) int64`, cluster id 0–12 for **every** track. **This is the key to cluster features without leiden.pkl.**
- `centroids.npy` — `(13, 100)` float32. NOTE: this lives in the PCA-100 *clustering* space, not the 1024-d embedding space, so you cannot project a raw query vector onto it. Use `labels.npy` + neighbor voting instead (see Phase 2.3).
- `embedding_2d.npy` — `(99618, 2)` float32 UMAP coords (for optional 2-D "map" readouts / viz).
- `cluster_profiles.json` — **list of 13** dicts, each `{cluster_id, size, exemplars: [{idx, name, artist}, ...], ...}`. Use for human-readable cluster names.
- `track_tags.json` — per-track micro-genre tags (26 MB).
- `leiden.pkl` — **DO NOT LOAD.** Pickled under numpy 2.x; unpicklable under this env's numpy 1.26.4. All cluster work below is designed to avoid it.

### Invariants (violating these breaks silently)
1. **`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` must be set before importing transformers**, or `from_pretrained` hangs through the sandbox SOCKS proxy. Every entry-point module already sets these via `os.environ.setdefault`; keep that at the top of any new module that loads a model.
2. **numpy stays at 1.26.4.** Do not upgrade to satisfy some other package; it re-breaks the corpus load.
3. `SongIndex.load(path)` takes the path **without** extension (it appends `.npy`/`.json`).
4. `SongIndex.query(vec, top_k)` returns `[{rank:int, score:float, **metadata}]` where score is cosine similarity (embeddings are standardized+normalized). `index.embeddings` is the standardized matrix; `index.transform_query(vec)` maps a raw 1024-d vector into that space.

### Known bug to fix in passing (Phase 2.1)
`mentor_anther.py::sounds_like_from_audio` references `self.corpus`, which no longer exists after the leiden bypass (the class now holds `self.index`). It has never been exercised with real audio. Fix to `self.index` and confirm the MERT embed path.

---

## Architecture decisions (locked by the user — do not re-litigate)

- **D1. Graph invocation = native tool-calling / ReAct loop.** The 3B emits strict-JSON tool calls; a constrained decode + schema validation + a deterministic keyword fallback make it reliable at 3B and let it scale to 7B/8B later. (User overrode the NL-router recommendation.)
- **D2. Part 1 = full retrain AND a runtime scope/intent gate.** Both. Clean the baked-in deflection voice out of the weights *and* gate at inference.
- **D3. Graph query anchor = audio upload OR a corpus artist/track resolved by name.** Name resolution needs fuzzy matching against the 99,618-row metadata.

**Unifying insight (applies the fable-method finding "weak models follow rules at decision points, not rules in lists"):** Part 1's scope gate and Part 2's tool routing are the *same mechanism* — a forced first-token **INTENT** classification. Build it once in Part 1 as a 3-way router `ADVICE | GRAPH | OFFTOPIC`; Part 2 hangs the ReAct loop off the `GRAPH` branch. This is the single highest-leverage transfer from fable-method: make the model *emit* a decision artifact before acting, rather than describing the rule in a prompt.

---

# PART 1 — Scope gate + data cleanup + retrain

### Phase 1.1 — Audit and clean the training data
**Files:** `models/mentor/train.jsonl`, `models/mentor/val.jsonl`.
**Do:**
1. Load both JSONL files. Each row is a chat-format example (`messages` with system/user/assistant).
2. Identify the deflection/AI-assistant examples: assistant turns that decline household-chore or off-topic questions by claiming to be an AI assistant, or that answer clearly non-music questions. From the v2 RAG cleanup, the count is **27 in train.jsonl and 1 in val.jsonl**. Re-detect programmatically rather than trusting the count: flag rows whose assistant text contains AI-assistant self-reference ("I'm your go-to AI", "as an AI", "I am an AI assistant", "I don't have the ability", etc.) OR whose user turn is off-topic (non-music) with a compliant answer.
3. Write the flagged indices to `models/mentor/deflection_audit.json` (`{"train": [...], "val": [...]}`) for review before deleting.
4. Remove the flagged rows, writing `train.jsonl`/`val.jsonl` in place (keep a `.bak` copy of the originals first).

**Done when:** `deflection_audit.json` exists and lists the flagged rows; a re-scan of the cleaned files finds 0 AI-assistant self-references in assistant turns; row counts printed (expect ~292 train / ~35 val).

### Phase 1.2 — Add in-voice decline examples + intent labels
**Do:**
1. Author **~15 new decline examples** in the punchy mentor voice (see `mentor.py::UNGROUNDED_FEWSHOT` for the target register). Cover the leak classes seen in the v1 transcript and v2 regression: coding requests ("write me a python function…"), shopping/deals, recipes, weather, generic factual Q&A, "who are you"/"are you an AI". Each declines briefly *in character* and steers back to music — never claims to be an AI, never answers the off-topic ask. Append to `train.jsonl` (put 1–2 in `val.jsonl`).
2. Build the **intent-classification training file** `models/mentor/intent_train.jsonl` — a small dataset (~60–90 rows) mapping a user message to a single label token `ADVICE`, `GRAPH`, or `OFFTOPIC`. Draw ADVICE examples from the real music_qa questions, GRAPH examples from "what do I sound like / who am I close to / what's between me and X / what scene should I cross into" phrasings, OFFTOPIC from the decline set. This trains the router used in Phase 1.4. (Alternative if you prefer not to train the router: skip this file and implement the router as a constrained few-shot classifier on the same base model — see Phase 1.4 note. Recommended: few-shot classifier, no separate training, to keep the advice adapter's output format clean.)

**Done when:** cleaned `train.jsonl` contains the new decline rows (count printed); the decline examples read in-voice on manual inspection (no AI self-reference); router data exists (or the decision to use a few-shot router is recorded in a comment).

### Phase 1.3 — Retrain the LoRA
**File:** `mentor/mentor_train.py` (no hyperparameter changes).
**Hyperparameters (keep exactly):** 4-bit NF4 QLoRA; LoRA r=16, alpha=32, dropout=0.05; target modules q/k/v/o/gate/up/down_proj; 3 epochs; per_device_train_batch_size=2, grad_accum=8 (eff 16); lr 2e-4 cosine; max_length=1024; `assistant_only_loss=True`; gradient checkpointing; optimizer `paged_adamw_8bit`.
**Do:**
1. Run `mentor_train.py` on the cleaned data. It writes the adapter to `models/mentor/mentor_lora/`. **Back up the current adapter first** to `models/mentor/mentor_lora_v2_backup/` so you can roll back.
2. Training runs on the foreground of the process (a bare `nohup … &` does NOT survive cell completion — run it as a blocking foreground call). Expect ~60 steps, ~2–3 min, peak VRAM ~4.5 GB, final train loss ~2.0–2.1, eval loss ~2.4–2.5.
3. Save the loss curve to `models/mentor/training_loss_v3.png`.

**Done when:** a new `adapter_model.safetensors` (~60 MB) exists in `mentor_lora/`; training completed 3 epochs without OOM; loss curve saved; final metrics printed and within the expected band.

### Phase 1.4 — Forced INTENT router in `chat()`
**File:** `mentor/mentor.py`.
**Do:**
1. Add a method `classify_intent(question) -> str` returning one of `ADVICE | GRAPH | OFFTOPIC`. Implement as a **constrained-decode classification**: a dedicated system prompt with 2–3 few-shot exemplars per class, and generation limited so the model's first output IS the label. Enforce by: prompting for the bare token, `max_new_tokens` small, then mapping the decoded text to the nearest valid label; if it doesn't match, fall back to a deterministic keyword heuristic + the RAG cosine (below).
2. Wire the router as the top of `chat()`, replacing the current cosine-only `in_scope` branch:
   - `GRAPH` → Part 2's graph handler (Phase 2.5). Until Part 2 lands, stub it to the existing `sounds_like`/audio path or a "point me at a track" message.
   - `ADVICE` → existing GROUNDED branch (RAG retrieve → inject → generate). Keep the RAG cosine as a **secondary guard**: if intent says ADVICE but `max_score < SCOPE_THRESHOLD` (0.63), fall through to OFFTOPIC (belt-and-suspenders against a mis-route).
   - `OFFTOPIC` → existing UNGROUNDED redirect branch (few-shot redirect, no RAG context).
3. Keep `SCOPE_THRESHOLD = 0.63` and the `UNGROUNDED_FEWSHOT` block; they now back up the router instead of being the sole gate.

**Done when:** `classify_intent` returns a valid label for a batch of test prompts (music advice, "what do I sound like", salsa/python/weather); `chat()` routes each to the correct branch in `verbose=True` output.

### Phase 1.5 — Validate Part 1 (regression)
**Do:** Run the existing regression style (see `regression_v2.json` for the prior format) over three buckets and write `models/mentor/regression_v3.json`:
- **On-topic advice (≥8 prompts):** all route ADVICE, all answers in-voice, no AI self-reference.
- **Off-topic (the v2 leak set incl. "write me a python function to sort a list", salsa, prime-day, weather, car repair):** all route OFFTOPIC and produce an in-voice redirect. **The Python-function leak is the key acceptance case — it must now redirect, not emit code.**
- **Identity probes ("are you an AI?", "who made you?"):** must decline in-voice, must NOT say "I am an AI assistant created by Anthropic" or similar.
Record VRAM (load + peak) — expect ~2.3 / ~2.6 GB.

**Done when:** `regression_v3.json` shows 0 AI-self-reference across all buckets, the Python-function prompt redirects, and on-topic answers remain grounded. This is the bar for "Part 1 complete."

---

# PART 2 — Graph tool harness (native ReAct loop)

Goal: turn the one-shot "sounds like" into an agentic set of graph operations the mentor chooses among. All tools operate on the frozen corpus via `SongIndex` + `labels.npy` + `track_tags.json` + `cluster_profiles.json`. No `leiden.pkl`.

### Phase 2.1 — Fix & extend anchor resolution
**File:** `mentor/mentor_anther.py`.
**Do:**
1. **Fix the `self.corpus` bug** in `sounds_like_from_audio` → `self.index`. Verify the MERT embed→`transform_query`→`query` path returns neighbors for a real audio file (use any short wav; if no audio is handy, at minimum confirm the code path executes to the embed call).
2. Add `resolve_anchor(spec) -> np.ndarray` that produces a 1024-d query vector from either:
   - an **audio path** (embed with MERT — reuse the existing audio path), or
   - a **name string** — fuzzy-match against corpus metadata (`index.json` → per-track `artist` and `name`). Use `rapidfuzz` (add to env if missing) or `difflib.get_close_matches` as a no-dep fallback. Match against `"{artist} — {name}"` and against `artist` alone. If the match is an **artist with multiple tracks**, return the **mean of that artist's track vectors** (an "artist centroid"); if a single track, return that track's vector. Return the resolved label(s) too, so the readout can say what it matched.
3. Add `resolve_two(spec_a, spec_b)` for bridge queries.

**Done when:** `resolve_anchor("Neroptik")` returns a vector and reports the matched artist; a deliberately fuzzy query ("neroptic") still resolves; the audio path runs without the `self.corpus` AttributeError.

### Phase 2.2 — Cluster recovery (no leiden)
**File:** `mentor/mentor_anther.py`.
**Do:** Add `cluster_of(vec, knn=25) -> {cluster_id, label, confidence, exemplars}`:
- Query the index for `knn` neighbors, look up each neighbor's cluster via `labels.npy`, take the majority; `confidence` = fraction of neighbors in the winning cluster.
- `label`/`exemplars` come from `cluster_profiles.json[cluster_id]`.
This restores the cluster placement the current tool had to drop after the leiden bypass.

**Done when:** `cluster_of` on a known corpus track returns that track's own cluster (cross-check against `labels.npy[idx]`) with high confidence.

### Phase 2.3 — Graph tools module
**New file:** `mentor/mentor_graph.py` — a set of pure, deterministic functions over `AntherSoundsLike`. Each returns a compact JSON-serializable dict (the ReAct "observation"). Proposed tool set (all implementable from the verified assets):

1. `sounds_like(anchor, top_k=8)` — nearest neighbors + inherited micro-genre tags + `cluster_of`. (The enhanced version of today's tool.)
2. `bridge(anchor_a, anchor_b, k=6)` — resolve both, take the **normalized midpoint** of the two vectors, query the index → the artists sitting "between" the two sounds. Report each neighbor's cluster so the user sees which side they lean.
3. `crossover(anchor, k=6)` — find the anchor's cluster (Phase 2.2), then return nearest neighbors that fall in a **different** cluster (the closest adjacent scene). Name the target cluster via its profile exemplars.
4. `tagmates(anchor, k=8)` — get the anchor's top tags via the existing `_tags_for`, then return neighbors that **share** at least one of those tags (tag-filtered retrieval), so the user gets sound-alikes constrained to a micro-genre.
5. `resolve(name)` — expose the fuzzy resolver as its own tool so the model can confirm "did you mean X?" before a heavier call.

Keep each function's output small (names/artists/scores/cluster labels — no raw vectors). Add a `TOOLS` registry dict `{name: (callable, json_schema)}` for the harness.

**Done when:** each tool runs standalone on a named anchor and returns a well-formed dict; `bridge("Neroptik", <some other corpus artist>)` returns plausible in-between artists; `crossover` returns neighbors from a different cluster than the anchor's.

### Phase 2.4 — The ReAct harness (strict JSON + constrained + deterministic fallback)
**New file:** `mentor/mentor_react.py`.
**Protocol:** the model, given the tool schemas and the user question, emits exactly one JSON object per step:
`{"tool": "<name>", "args": {...}}` or `{"tool": "final", "answer": "<mentor-voice reply>"}`.
**Loop:**
1. System prompt = mentor persona + a compact tool catalog (name, one-line purpose, args) + 2–3 **few-shot ReAct traces** (user → tool call → observation → final). Few-shot is mandatory at 3B; a bare schema is not followed reliably.
2. Generate one step. **Parse the first JSON object** out of the output (regex for the first balanced `{...}`; tolerate leading prose).
3. **Validate** against the tool registry schema. On parse/validation failure, invoke the **deterministic fallback router**: keyword → tool + naive arg extraction (quoted strings / "between X and Y" / capitalized spans as names). This guarantees forward progress even when the model emits malformed JSON.
4. Execute the tool, append the observation as a `tool`/user message, loop. **Cap at 3 tool calls**, then force a `final` step (append "Now answer the user in your mentor voice using what you found.").
5. The final answer must name real artists returned by the tools and must **not** invent artists (reuse the "Do not invent other artists" constraint from `sounds_like`).

**Done when:** on a set of graph questions, the harness completes within the call cap and returns an in-voice answer citing tool-returned artists; a deliberately malformed model output still resolves via the deterministic fallback (test by forcing a bad first token).

### Phase 2.5 — Integrate into `chat()`
**File:** `mentor/mentor.py`.
**Do:** In the `GRAPH` branch of the router (Phase 1.4), call the ReAct harness. Anchor resolution: if `audio_path` is provided use it; else let the harness's `resolve` tool pull a name out of the question. If no anchor can be resolved, return an in-voice prompt asking the user to name a track/artist or upload audio (do not fabricate). Keep the direct `audio_path` fast-path (a bare "what do I sound like" + audio) routing straight to `sounds_like` without a full loop, for latency.

**Done when:** `m.chat("what artists sit between me and <corpus artist>?")` runs the loop and returns a bridge answer; `m.chat("what do I sound like?", audio_path=...)` still works via the fast path; an anchor-less graph question asks for a track instead of hallucinating.

### Phase 2.6 — Validate Part 2 + update docs
**Do:**
1. Write `models/mentor/graph_demo.md` — a transcript exercising each tool (sounds_like, bridge, crossover, tagmates) with a named anchor, plus one malformed-JSON fallback case and one anchor-less case.
2. Record VRAM (loading MERT + Qwen + bge together): expect ~8–9 GB peak, well within 16 GB.
3. Update `mentor/README.md` and `mentor/SKILL.md` (the `music-mentor-finetune` skill) with: the new intent router, the graph tool set, the anchor-resolution behavior, and the numpy-1.26/leiden-bypass invariant restated for the cluster-recovery code.

**Done when:** `graph_demo.md` shows each tool returning real corpus artists, the fallback and anchor-less paths behave, VRAM is within budget, and the README/SKILL reflect v3.

---

## Risks & gotchas (carry-over from the build)
1. **Offline flags** — set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before importing transformers or model loads hang.
2. **numpy 1.26.4 pin** — do not upgrade; it re-breaks the corpus load.
3. **`leiden.pkl` is off-limits** — all cluster logic uses `labels.npy` + neighbor voting.
4. **Background cells** — run training/generation as a blocking foreground process; `nohup … &` does not survive cell completion.
5. **3B + structured output** — never rely on a bare instruction or bare JSON schema; always back it with few-shot exemplars AND a deterministic fallback. This is the whole reason the intent router and ReAct loop are built the way they are.
6. **transformers 5 chat template** — use `apply_chat_template(..., return_dict=True)`, read `enc["input_ids"].shape[1]` for the prompt length, and call `generate(**enc)`.
7. **Roll-back safety** — back up `mentor_lora/` before retraining and keep the v2 RAG index; if v3 regresses, the prior adapter + index restore the working v2 system.

## Suggested commit checkpoints
- After Part 1.5: commit cleaned data + retrained adapter reference + router + `regression_v3.json`.
- After Part 2.6: commit the graph module, ReAct harness, chat integration, demo, and doc updates.
(Nothing in the repo is committed yet — the user handles git.)
