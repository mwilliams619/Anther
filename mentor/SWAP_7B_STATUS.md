# 7B swap — in-progress status (delete this file once the swap is verified/documented)

Full plan: `/home/matt/.claude/plans/i-want-to-try-cached-pinwheel.md`
(side-by-side toggle: keep 3B assets on disk, add 7B alongside via one variable
in `paths.py`).

## Done

1. **`mentor/paths.py`** — edited. Added `MODEL_NAME = "Qwen2.5-7B-Instruct"`,
   `BASE_DIR` derives from it, `LORA_DIR` changed to
   `mentor_lora_7b` (3B adapter untouched at `mentor_lora/`). Verified by
   reading the file back after edit.

2. **`mentor/fetch_models.py`** — edited. `FILES` dict now keys on
   `"Qwen/Qwen2.5-7B-Instruct"` with the **verified real shard manifest** (4
   shards, confirmed via curl against the actual
   `model.safetensors.index.json` on HF — do NOT re-guess this, it's already
   correct):
   ```
   model-00001-of-00004.safetensors
   model-00002-of-00004.safetensors
   model-00003-of-00004.safetensors
   model-00004-of-00004.safetensors
   ```
   plus the same config/tokenizer files as before.

3. **Downloaded the 7B base weights** — `python mentor/fetch_models.py` ran
   successfully (exit confirmed via "All base weights present" in the log).
   Verified on disk at `models/mentor/Qwen2.5-7B-Instruct/`: all 4 safetensors
   shards (~15GB total) + config.json, generation_config.json, merges.txt,
   model.safetensors.index.json, tokenizer.json, tokenizer_config.json,
   vocab.json all present with real sizes. **No need to re-download.**

4. GPU confirmed idle: RTX 4080 SUPER, 16376 MiB total, ~0 used. Disk: 399GB
   free at time of download.

## In progress / next step

**Training attempt #1** failed on `ModuleNotFoundError: No module named
'torch'` — the repo's `venv` wasn't activated in that exact backgrounded Bash
invocation (activation in one Bash call doesn't persist to a separate call).

**Training attempt #2** (venv activated correctly) failed on
`ModuleNotFoundError: No module named 'datasets'` — turned out the mentor
extras (`peft, trl, bitsandbytes, datasets, sentence-transformers`) were
never installed in this repo's `venv` (only `accelerate` was present; no
conda env on this machine has them either — the "mentor conda env" described
in `SKILL.md` was evidently a different, one-off sandbox from original
development, not something present here). **Fixed**: ran
`pip install -e '.[mentor]'` from repo root inside `venv` — installed cleanly
(peft 0.19.1, trl 1.8.0, bitsandbytes 0.49.2, datasets 5.0.0,
sentence-transformers 5.6.0) despite the repo venv being Python 3.12.7 /
numpy 2.4.6, NOT the "python 3.11 + numpy<2" the README's Setup section and
SKILL.md's environment section describe — that guidance is stale/was for a
different sandbox; the actual repo venv works fine with newer versions.
Verified via `python -c "import torch, peft, trl, bitsandbytes, datasets,
accelerate"` — all clean, `torch.cuda.is_available() == True`.
**Worth a docs fix later**: README.md's "in a python 3.11 env with numpy<2"
setup note and SKILL.md's environment section are inaccurate for this
machine — flag but don't block on it.

Training attempt #3 (all deps now installed) is running in the background —
see "Exact next command" below is just for reference/resuming if it's not
still running when you read this; check
`/tmp/claude-1000/-home-matt-Dev-Anther/e6b17e32-bf9b-48d8-a6a3-17371f3d77b6/scratchpad/train_7b.log`
first for a result before re-running.

### Exact next command to run (only if attempt #3's log shows failure/is missing)

```bash
source /home/matt/Dev/Anther/venv/bin/activate && python /home/matt/Dev/Anther/mentor/mentor_train.py
```

Per the plan:
- **Leave hyperparameters unchanged for this first real attempt**
  (`per_device_train_batch_size=2, gradient_accumulation_steps=8` in
  `mentor/mentor_train.py`) — just observe the printed `PEAK_TRAIN_VRAM_GB`.
- **If it OOMs**, edit `mentor/mentor_train.py` to
  `per_device_train_batch_size=1, gradient_accumulation_steps=16` (same
  effective batch of 16) and retry — that's the only sanctioned first
  fallback, don't touch `max_length` or anything else unless that also OOMs.
- Training writes the new adapter into `mentor_lora_7b/` (per the updated
  `LORA_DIR`) — this is separate from the existing 3B adapter at
  `mentor_lora/`, so nothing old is destroyed regardless of outcome.
- Expect this to take a few minutes (60 steps took ~134s for the 3B; 7B will
  be slower per step).

## Remaining steps after training succeeds (from the approved plan)

1. **No RAG rebuild needed** (confirmed independent of `BASE_DIR`, already
   verified by reading `mentor_rag.py` during planning — skip straight to
   verification).
2. **Verify end-to-end**, in order:
   - `source venv/bin/activate && python mentor/mentor.py` — canned question,
     confirm base+adapter load with no shape-mismatch errors, plausible
     mentor-voice answer.
   - `source venv/bin/activate && python mentor/mentor_demo.py` — observe
     printed serving VRAM peak (expect higher than the old 2.56 GB baseline,
     should still be comfortably under 16 GB), the 5 Q&A answers + RAG cosine
     scores, and that the "sounds like" section runs clean.
   - `source venv/bin/activate && python mentor/mentor_chat.py` — spot-check
     a couple of questions including one clearly out-of-scope one (confirm
     the ungrounded-question redirect still triggers — `mentor.py:164`'s
     comment flags this few-shot behavior was tuned for 3B's weakness, worth
     explicitly re-checking on 7B, not assuming it carries over).
   - Treat as verified when: training had no OOM with peak comfortably under
     16 GB, demo runs clean with serving peak also comfortably under 16 GB,
     and spot-checked answers are qualitatively at least as good as the 3B
     baseline.
3. **Docs pass (last, using the real numbers just measured)** — update:
   - `mentor/README.md` (VRAM footprint sentence ~line 9-10, layout tree
     ~line 20/27)
   - `mentor/mentor_train.py`, `mentor/mentor.py`, `mentor/__init__.py`
     docstrings (currently say "Qwen2.5-3B-Instruct")
   - `mentor/fetch_models.py:22`'s comment (already says "Qwen2.5-7B-Instruct
     base" post-edit above — double check it reads correctly)
   - `mentor/SKILL.md`: frontmatter description, "Verified footprint"
     paragraph (4.51 GB / 2.56 GB → new measured numbers), "QLoRA
     hyperparameters" section (base id + new loss curve + new step timing +
     batch/grad-accum if it was retuned), and the "Retraining/tweaking
     checklist" line that predicted "Bigger base (e.g. 8B)..." — update to
     reflect this was done for 7B specifically with real numbers.
   - `mentor/mentor_chat.py`'s "~10s" load-time estimate comment, only if the
     observed load time in verification differs meaningfully.
4. **Delete this status file** (`mentor/SWAP_7B_STATUS.md`) once steps 2-3
   above are complete — it's a working handoff note, not permanent repo docs.

## Todo list state at time of writing this note

- [x] Edit mentor/paths.py to add MODEL_NAME toggle and repoint BASE_DIR/LORA_DIR
- [x] Verify Qwen2.5-7B-Instruct safetensors shard manifest and update mentor/fetch_models.py FILES dict
- [x] Run fetch_models.py to download 7B base weights
- [ ] Run mentor_train.py, watch VRAM peak, retune batch size if OOM  ← **resume here**
- [ ] Verify end-to-end: mentor.py sanity check, mentor_demo.py, mentor_chat.py spot-check
- [ ] Update docs (README, SKILL.md, docstrings) with real post-retrain numbers
