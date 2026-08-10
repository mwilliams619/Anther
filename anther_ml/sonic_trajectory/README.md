# anther_ml.sonic_trajectory

Read an artist's **sound over time** by placing their discography onto the frozen
100k-song sonic atlas (MERT + MERIT embeddings) and measuring how the sound moves
across releases. **Genre labels are never a model input** — every claim is grounded
in where the audio lands and in label-independent structural metrics.

## Quick start

```python
from anther_ml.sonic_trajectory import run_artist, EraRule

rules = [
    EraRule("dangerously in love", "Dangerously in Love (2003)"),
    EraRule("lemonade",            "Lemonade (2016)"),
    EraRule("renaissance",         "Renaissance (2022)"),
    EraRule("cowboy carter",       "Cowboy Carter (2024)"),
]
res = run_artist("Beyoncé", era_rules=rules, out_dir="results/beyonce",
                 llm_fn=my_llm)          # llm_fn optional: (prompt, system) -> str
```

`run_artist` → resolve discography (iTunes) → embed + place on GPU → per-era
structural metrics → two figures → optional blurbs → optional profile upsert.
The GPU step is skipped when `<prefix>_placements.csv` + `_embeddings.npz` are
already cached in `out_dir` (`reuse_cached=True`).

## CLI

```bash
python -m anther_ml.sonic_trajectory --artist "Beyoncé" \
    --eras eras_beyonce.json --out results/beyonce --persist --qwen
```
`--eras` is a JSON list of `[substring, era_label]` pairs (first match wins).
`--persist` writes the `sonic_trajectory` block onto the artist's profile in the
enrichment SQLite store. `--qwen` uses the local mentor Qwen model for blurbs;
default skips blurbs (no LLM required for figures + metrics).

## Modules

| file | role |
|---|---|
| `regions.py` | canonical `cluster_id → region` map (a property of the frozen corpus) |
| `discography.py` | iTunes resolve + era grouping (`EraRule`, `resolve_discography`) |
| `_embed_worker.py` | GPU subprocess: MERT+MERIT embed + corpus placement |
| `embed.py` | launches the worker, streams progress, parses the result |
| `analyze.py` | per-era metrics: spread (bits), dispersion, centroid shifts, region shares |
| `plots.py` | self-contained matplotlib: trajectory + atlas figures |
| `blurb.py` | grounded fact brief + injectable `llm_fn` blurb generation |
| `pipeline.py` | `run_artist` orchestrator + profile persistence |

## Design notes

- **Injectable LLM.** Blurb functions take `llm_fn(prompt, system) -> str`; the
  module ships fact-brief + prompt templates but never binds a model. Use the
  local Qwen in production, or `host.llm` in a sandbox.
- **Additive persistence.** A `sonic_trajectory` field was added to
  `ArtistProfile` (schema + api profile). It is provenance-stamped
  (`source`, `updated_at`) and never touches the row-locked `artist_meta.json`
  clustering artifacts.
- **Embedding runs in a subprocess** so the preview-fetch proxy is live and the
  offline MERT weights load cleanly. Set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.
- **Non-studio buckets** (singles / soundtrack / early) are matched by substring
  (`analyze.is_non_studio`) and excluded from the studio-era arc.
