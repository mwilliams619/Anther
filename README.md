# anther-ml

A music similarity engine and interactive map of songs. Anther listens to audio
directly — no genre labels, no tags, no play counts — and places every track on
a shared map where "close together" means "actually sounds alike."

- **Phase 1** — pre-computed librosa features (fast, no GPU), matching FMA's 518-dim schema.
- **Phase 2** — MERT neural embeddings + MERIT factor heads (the signal that drives the map today).

Both phases share the same clustering and similarity API in `anther_ml`; only the
input embeddings differ. See `CLAUDE.md` for the full architecture and notebook
pipeline order.

## Install

Requires **Python 3.11** (the frozen corpus artifacts were pickled under 3.11 and
won't deserialize on 3.12).

```bash
git clone <this-repo> && cd Anther
python3.11 -m venv .venv && source .venv/bin/activate

pip install -e .            # editable install — no sys.path hacks needed
pip install -e '.[dev]'     # + pytest for the test suite
pip install -e '.[notebooks]'  # + jupyter, if you want to run the pipeline notebooks

pytest                      # confirm the install
```

That gets you the code. To actually see a map you also need a **corpus bundle**
and the **MERIT heads**:

```bash
# MERIT projection heads (~11 MB each, pretrained — no training required)
hf download amaai-lab/merit head_mel/best_head.pt --local-dir models/merit_heads
hf download amaai-lab/merit head_rhy/best_head.pt --local-dir models/merit_heads
hf download amaai-lab/merit head_tim/best_head.pt --local-dir models/merit_heads
```

`data/` and `models/` are gitignored and large (tens of GB once populated — raw
audio, FMA metadata, reference-corpus bundles). Cloning this repo gets you code
only, not a working corpus. The primary bundle is the frozen 100k-track MPD
corpus at `models/corpus_corpus_mpd_100k/`; see
[docs/projects/REFERENCE_CORPUS_DESIGN.md](docs/projects/REFERENCE_CORPUS_DESIGN.md)
for how one is built. There is currently no shortcut path to a pre-built bundle
for a new collaborator — ask the repo owner.

Then run the web UI:

```bash
python ui/app.py   # http://localhost:5000
```

Configure with `ANTHER_CORPUS` (default `models/corpus_corpus_mpd_100k`) and,
for full-MPD playlist/album search, `ANTHER_MPD_DB`. See
[docs/ui.md](docs/ui.md) for routes, env vars, and session-state details.

## What MERT and MERIT are

**MERT** (`m-a-p/MERT-v1-330M`) is a large neural network that was trained on a
huge pile of music to *understand* audio. Feed it 30 seconds of a song and it
returns a list of 1024 numbers — a "fingerprint" that captures what the music is
like. Songs that sound alike get similar fingerprints. Nobody told it what genres
are; it learned the structure of music on its own.

**MERIT** ([arXiv:2605.27346](https://arxiv.org/abs/2605.27346)) sits on top of
the same MERT model and splits that one fingerprint into **three** separate ones:

| Factor | Roughly captures |
|---|---|
| **melody** | the tune — the notes and harmony |
| **rhythm** | the groove — tempo and feel |
| **timbre** | the texture — the *sound*, instrumentation, production |

This matters because "similar" isn't one thing. Two tracks can share a groove
while sounding nothing alike, or use the same instruments over completely
different melodies. A single similarity score blurs those together; three scores
let you say *how* two songs are related. Anther's map uses the average of the
three, and the song-detail panel shows the breakdown.

Both models are used **frozen** — Anther doesn't train them, it just runs audio
through them.

## How similarity works (the ELI5 version)

1. **Every song becomes a point.** Run the audio through MERT/MERIT and you get
   a list of numbers. Think of that list as coordinates — the song's address in a
   very high-dimensional space.

2. **Similar songs land near each other.** Because the fingerprints of similar-
   sounding music are similar, distance in that space *is* musical similarity.
   Finding "songs like this one" is just looking at which points are nearby.

3. **Everything is measured the same way.** Before comparing, every fingerprint
   is put on the same scale, so no single loud feature (a track's brightness, its
   mastering volume) can hijack the comparison. Audio is loudness-normalized
   first for the same reason — a louder master shouldn't count as a different
   song.

4. **Neighbors become a map.** Draw a line between two songs whenever they're
   close enough, let the lines pull connected songs together, and clumps form on
   their own. Those clumps are the clusters — regions of music that hang
   together. They're discovered from sound alone, and only *labeled* with genre
   names afterwards for readability.

5. **New songs get dropped in, not re-mixed.** The reference map of 100k tracks
   is frozen. When you add a song, it's fingerprinted the same way and placed
   onto the existing map next to its neighbors. The map doesn't shift underneath
   you, so two people looking at the same corpus see the same geography.

The one rule that makes all of this honest: **genre is never an input.** Not to
the clustering, not to the training, not to the evaluation metrics. It's display
text only. If genres show up as coherent regions on the map, that's a result, not
an assumption.

## Example use cases

### 1. Analyzing music similarity

Ask concrete questions about how music relates and get measurable answers:

- **"What does this song actually sound like?"** Query a track and get its
  nearest neighbors across a 100k-track corpus, plus the melody/rhythm/timbre
  breakdown showing *which* dimension drives each match.
- **"Is this artist's catalog coherent, or all over the place?"** Place a whole
  discography and look at the spread — a tight clump versus scattered points is a
  real, quantified answer.
- **"Do genre labels hold up?"** Since genre never touches the model, you can use
  it as an independent yardstick: cluster on audio, then check whether the
  clusters agree with human labels, and study the cases where they don't.
- **"Does this change actually help?"** Every design decision goes through the
  genre-free eval harness rather than vibes:

```bash
python -m anther_ml.eval --index models/index_phase1
```

See [docs/evaluation.md](docs/evaluation.md) and
[docs/similarity.md](docs/similarity.md).

### 2. Interactive playlist visualization

The Flask + d3 web UI (`python ui/app.py`) turns a playlist into a picture you
can explore:

- **Drop a playlist, album, or artist discography onto the map** — search
  Deezer/Spotify/iTunes or upload your own files, and every track gets embedded
  and placed among its neighbors. Tracks are embedded in the background so the
  map fills in live.
- **See the shape of your taste.** A well-sequenced playlist forms a path; a
  grab-bag scatters. Gaps between clumps are the transitions that don't work
  yet — and the corpus tracks sitting in those gaps are the songs that would
  bridge them.
- **Play the map.** "Play map" walks the graph as a playlist, hopping between
  connected songs so you *hear* the region you're looking at
  ([docs/projects/GRAPH_AUTOPLAY_PLAN.md](docs/projects/GRAPH_AUTOPLAY_PLAN.md)).
- **Recommend from the map.** Select several songs as seeds and pull in corpus
  tracks near all of them at once — recommendation as a spatial query rather than
  a black box.
- **Switch to Artist View** to see how the artists themselves connect, with
  enriched profiles (image, genres, origin, labels).

There's also a static export if you just want a shareable snapshot:

```bash
python export_viz.py    # bakes index + 2D embedding into a self-contained song_view.html
```

Re-run it after rebuilding an index — the HTML does not read `models/` live.

## Regenerating artifacts

Model pickles and indices in `models/` are derived (gitignored). Regenerate them
by running the notebooks in order (`01` → `05`); see the table in `CLAUDE.md`.
The clustering/similarity notebooks (`01`, `02`, `04`, `05`) use the Leiden +
standardized-index stack; `03` is an educational spectrogram demo.
