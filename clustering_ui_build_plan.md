# Build Plan — Song Staging + Clustering Web UI

A local Flask app that lets you **search Deezer**, **upload local files**, stage a
corpus, and **cluster it** using the existing `anther_ml` pipeline. Browser-only is
impossible (MERT can't run client-side) and direct browser→Deezer calls hit CORS, so
the backend does both: it proxies Deezer server-side and runs MERT + HDBSCAN.

Reuse, don't reinvent. The clustering, embedding, indexing, and Deezer fetch logic
already exist — wire them, don't rewrite them.

---

## Reuse these existing pieces (verify signatures in-repo before calling)

- `anther_ml.embedding` → `load_mert()`, `get_embedding(model, processor, path, device)`, `embed_batch(...)`
- `anther_ml.cluster` → `fit_clusters(X, n_umap_components=, min_cluster_size=)` returns
  `(labels, embedding_2d, scaler, pca, reducer, clusterer, reducer_2d)`;
  also `save_pipeline`, `load_pipeline`, `assign_cluster`, `cluster_summary`
- `anther_ml.similarity.SongIndex(X, metadata)` → `.query(vec, top_k)`, `.save(path)`, `.load(path)`, `.metadata`
- `spotify_deezer.py` → `_deezer_get(path, params=...)`, `fetch_preview_waveform(url, target_sr, clip_seconds, offset_seconds)`,
  `clip_waveform(wav, sr, clip_seconds, offset_seconds)`, `waveform_to_tempfile(wav, sr)`

Note: this UI searches **Deezer directly**, so the Spotify→Deezer ISRC *matcher* isn't
needed on the search path — tracks are keyed by `deezer_id`. The `fetch_preview_waveform`
and `clip_waveform` helpers ARE reused for the embed step.

---

## File tree to create

```
ui/
  app.py                  # Flask backend
  jobs.py                 # tiny in-process job runner (threading)
  static/
    index.html            # single-page UI
    app.js                # vanilla JS, fetch + Plotly
    style.css
  session/                # runtime state (gitignore this)
    manifest.json         # staged corpus
    uploads/              # user-uploaded audio (kept; it's your own)
    embeddings.npy        # mirrors notebook 04 outputs so notebooks can open it
    metadata.pkl
    pipeline_phase2.pkl
    index_phase2.*
    labels_phase2.npy
    embedding_2d_phase2.npy
```

Keep `session/` artifact names identical to notebook 04 so a UI-built corpus opens in
notebook 05 and vice versa.

---

## Backend endpoints

| Method | Route | Does |
|---|---|---|
| GET  | `/` | serve `index.html` |
| GET  | `/api/deezer/search?q=` | server-side Deezer search via `_deezer_get("search/track", params={"q": q, "limit": 25})`; return normalized hits |
| POST | `/api/stage` | add a Deezer hit OR uploaded file ref to `manifest.json`; dedup by `deezer_id` / filename |
| GET  | `/api/stage` | list staged corpus |
| DELETE | `/api/stage/<id>` | remove one |
| POST | `/api/upload` | accept multipart audio (mp3/wav/flac/m4a), save to `session/uploads/`, auto-stage |
| POST | `/api/cluster` | launch background cluster job; return `job_id` |
| GET  | `/api/cluster/status/<job_id>` | `{state, progress, message}`; on done include result summary |
| GET  | `/api/results` | latest `labels`, `embedding_2d`, `metadata` for the scatter |

Normalized Deezer hit shape: `{deezer_id, title, artist, album, cover, preview_url, duration}`.
Flag and skip hits where `preview_url` is null (some tracks have none).

---

## The cluster job (jobs.py + /api/cluster)

Background thread, single job at a time. Load the MERT model **once** at first use (or at
boot) — it's ~1.3 GB. Steps:

1. Read staged corpus from `manifest.json`.
2. For each item, produce a **clipped waveform at one consistent length** (default
   `CLIP_SECONDS = 30`, the preview length):
   - Deezer item → `fetch_preview_waveform(preview_url, MERT_SR, clip_seconds=30)`
   - Uploaded item → `librosa.load(path, sr=MERT_SR, mono=True)` then `clip_waveform(..., 30)`
   This shared clip is the whole point — it keeps previews and full uploads comparable.
3. Embed. Simplest path that touches no existing code: `waveform_to_tempfile(wav, sr)` →
   `get_embedding(model, processor, tmp_path, device)` → unlink temp. (If `get_embedding`
   already accepts arrays, pass the array and skip the temp file.)
4. Stack to `X`, scale small-N hyperparameters exactly like notebook 04:
   `umap_components = min(32, max(2, n//10))`, `min_cluster = max(3, n//20)`.
5. `fit_clusters(X, n_umap_components=umap_components, min_cluster_size=min_cluster)`.
6. Save `embeddings.npy`, `metadata.pkl`, `save_pipeline(...)`, `SongIndex(X, meta).save(...)`,
   `labels`, `embedding_2d` into `session/` (notebook-compatible names).
7. Report `n_clusters`, `noise_pct`, and per-track assignments back through status.

Progress: update job state per-track during embed (that's the slow part — surface it,
e.g. "embedding 12/40").

Guardrail: gate the Cluster button until the corpus has ≥ ~15 tracks; below that HDBSCAN
mostly returns noise. Show the resulting `noise_pct` so a mostly-noise run is obvious.

---

## Front-end (static/index.html + app.js)

Vanilla JS + `fetch`, no build step. Plotly via CDN for the scatter. Four panels:

1. **Search** — text input → `/api/deezer/search`; result rows show cover, title, artist,
   a ▶ button (`<audio src=preview_url>` plays fine cross-origin), and **Add**.
2. **Upload** — drag-drop zone → `/api/upload` (multipart). Show added filenames.
3. **Staging corpus** — the "cart": list of staged tracks with source badge
   (deezer / upload) and a remove ✕. Live count. **Cluster N songs** button (disabled < 15).
4. **Results** — after a job completes, fetch `/api/results` and draw a Plotly scatter of
   `embedding_2d` colored by cluster label (−1 = noise, greyed), plus an assignments table.

Keep it clean and legible over flashy — light theme, generous spacing, monospace for IDs.
Poll `/api/cluster/status/<job_id>` every ~1.5 s while a job runs; show the progress text.

---

## Guardrails / gotchas to bake in

- Deezer previews: never persist the audio (ToS) — embed transiently, store only
  embeddings. Uploaded files are the user's own and are kept in `session/uploads/`.
- Preview URLs are signed and expire in hours: only the embed step fetches them; the
  manifest stores `deezer_id` + a fresh fetch at cluster time (re-resolve via
  `_deezer_get("track/<id>")` if the cached URL is stale).
- Dedup on add by `deezer_id` and by upload filename.
- Cap upload size (e.g. 25 MB) and whitelist extensions.
- MERT: load once, note MPS vs CPU device, warn that large corpora are slow
  (~seconds/track on MPS).
- Re-cluster is full-refit (cheap at this scale); don't try incremental HDBSCAN.

---

## Build order for Claude Code

1. Scaffold `ui/` tree + `app.py` serving a static shell; confirm `localhost:5000` loads.
2. `/api/deezer/search` + Search panel (verifies the server-side Deezer proxy end-to-end).
3. `/api/upload` + `/api/stage` + staging cart with persistence to `manifest.json`.
4. `jobs.py` + `/api/cluster` wiring the embed→`fit_clusters`→save flow; load MERT once.
5. Status polling + Results scatter via `/api/results`.
6. Polish: guardrails, dedup, empty/edge states, noise% display.

## Run

```
pip install flask librosa soundfile requests   # plus existing anther_ml deps
python ui/app.py
# open http://localhost:5000
```
