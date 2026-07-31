"""Internal similarity service over the frozen Anther reference corpus.

Wraps the frozen corpus (99,618 MPD tracks, MERT-v1-330M embeddings) as a
small internal API the mentor's graph tools call. Nothing here is exposed to
the LLM directly — the tool layer (graph_tools.py) turns these raw reads into
music-concept observations, and the agent narrates those.

Public surface (everything else is implementation detail):
    resolve(spec, graph_ctx=None) -> (vec, info)   name/audio -> query vector
    similar(vec, top_k)           -> corpus nearest neighbours
    cluster(vec)                  -> cluster id + label via neighbour voting
    tags(vec, top_k, knn)         -> micro-genre tags inherited from neighbours
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import numpy as np
from difflib import get_close_matches

# ---- numpy 2.x -> 1.26 pickle shim ------------------------------------------
# The Anther corpus (leiden.pkl) was pickled under numpy 2.x, which stores a
# BitGenerator *class*; numpy 1.26's __bit_generator_ctor expects a string name.
# Patch it to accept either, so the frozen bundle loads unchanged.
import numpy.random._pickle as _nrp
_orig_ctor = _nrp.__bit_generator_ctor
def _compat_ctor(bit_generator="MT19937"):
    if isinstance(bit_generator, type):
        return bit_generator()
    return _orig_ctor(bit_generator)
_nrp.__bit_generator_ctor = _compat_ctor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO_ROOT, CORPUS_DIR
# the Anther package (anther_ml) lives at the repo root
ANTHER_ROOT = REPO_ROOT
if ANTHER_ROOT not in sys.path:
    sys.path.insert(0, ANTHER_ROOT)


class AntherSimilarityService:
    """Loads the frozen corpus's SongIndex + track_tags directly.

    We deliberately load the (index.npy, index.json) pair and track_tags.json
    rather than the full ReferenceCorpus bundle: the bundle's leiden.pkl was
    pickled under numpy 2.x and won't unpickle under this env's numpy 1.26.
    Cluster membership is recovered from labels.npy by neighbour voting and
    labels are hydrated from cluster_profiles.json.
    """
    def __init__(self, corpus_dir=CORPUS_DIR):
        import json
        from anther_ml.similarity import SongIndex
        from anther_ml import calibration as link_calibration
        self.corpus_dir = corpus_dir
        self.index = SongIndex.load(os.path.join(corpus_dir, "index"))
        with open(os.path.join(corpus_dir, "track_tags.json")) as f:
            self.track_tags = json.load(f)

        # ── link-similarity calibration (shared with ui/atlas.py) ──────────
        # ui/atlas.py runs in a separate process (docs/mentor-graph-aware.md)
        # and stamps a link_calibration.json sidecar into corpus_dir every
        # time it calibrates. Read it if present; otherwise reproduce the
        # exact same deterministic 200k-pair draw here (seed=0 — see
        # anther_ml.calibration) so this process's bands agree with atlas's
        # even on a fresh checkout that hasn't run the UI yet.
        self._link_calibration = link_calibration
        thresholds = link_calibration.load_calibration(corpus_dir)
        if thresholds is None:
            thresholds = link_calibration.calibrate_link_thresholds(self.index)
        self.link_thresholds = thresholds
        # optional cluster labels for readout (profiles JSON is plain, no pickle)
        try:
            with open(os.path.join(corpus_dir, "cluster_profiles.json")) as f:
                raw_profiles = json.load(f)
        except Exception:
            raw_profiles = []
        self.profiles = {}
        for p in raw_profiles:
            cid = p.get("cluster_id")
            if cid is not None:
                self.profiles[int(cid)] = p

        labels_path = os.path.join(corpus_dir, "labels.npy")
        self.labels = np.load(labels_path) if os.path.exists(labels_path) else None

        emb_path = os.path.join(corpus_dir, "embeddings.npy")
        self.raw_embeddings = np.load(emb_path, mmap_mode="r") if os.path.exists(emb_path) else None

        self._name_keys = None
        self._artist_to_indices = None
        self._mert = None  # (model, processor, device) lazy-loaded

    # ---- public API ---------------------------------------------------------
    def resolve(self, spec, graph_ctx=None):
        """Resolve an anchor spec (audio path or name) to a query vector.

        Resolution order: audio file -> on-screen session (``graph_ctx``) ->
        frozen corpus. Returns (vec, info); ``info["type"]`` distinguishes
        ``onscreen_*`` matches from corpus ``artist``/``track`` matches.
        Context references ("me", "this song") are the tool layer's job — by
        the time a spec reaches this service it must be a concrete name/path.
        """
        spec_text = str(spec).strip() if spec is not None else ""
        if isinstance(spec, str) and os.path.exists(spec):
            vec = self._embed_audio(spec)
            return vec, {"input": spec, "type": "audio", "resolved": True, "path": spec}
        if isinstance(spec, dict) and spec.get("audio_path"):
            path = spec["audio_path"]
            vec = self._embed_audio(path)
            return vec, {"input": path, "type": "audio", "resolved": True, "path": path}
        if graph_ctx is not None:
            try:
                if graph_ctx.has_nodes():
                    vec, info = graph_ctx.resolve(spec_text)
                    if info.get("resolved"):
                        return vec, info
            except Exception:
                pass
        return self._resolve_name(spec_text)

    def similar(self, vec, top_k=8):
        """Corpus nearest neighbours, deduped by (artist, name).

        Returns [{artist, name, score, index}] — ``index`` kept so callers can
        look up cluster/tags for a neighbour; it never reaches the LLM.
        """
        q = self.index.transform_query(vec)
        sims = self.index.embeddings @ q
        order = np.argsort(sims)[::-1][: max(top_k * 3, top_k)]
        seen, out = set(), []
        for idx in order:
            idx = int(idx)
            m = self.index.metadata[idx]
            key = (m.get("artist", ""), m.get("name", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "artist": m.get("artist", ""),
                "name": m.get("name", ""),
                "score": round(float(sims[idx]), 4),
                "index": idx,
            })
            if len(out) >= top_k:
                break
        return out

    def cluster(self, vec, knn=25):
        """Infer cluster by neighbour voting over labels.npy (no leiden.pkl)."""
        if self.labels is None:
            return {}
        q = self.index.transform_query(vec)
        sims = self.index.embeddings @ q
        nbr = np.argsort(sims)[::-1][:knn]
        nbr_labels = self.labels[nbr].astype(int)
        counts = np.bincount(nbr_labels)
        cid = int(np.argmax(counts))
        conf = float(counts[cid] / max(1, len(nbr_labels)))
        profile = self.profiles.get(cid, {})
        exemplars = profile.get("exemplars", [])[:3]
        if profile.get("label_final"):
            label = profile.get("label_final")
        elif exemplars:
            label = ", ".join(f"{e.get('artist','')} - {e.get('name','')}" for e in exemplars)
        else:
            label = f"cluster {cid}"
        return {
            "cluster_id": cid,
            "label": label,
            "confidence": round(conf, 3),
            "exemplars": exemplars,
        }

    def tags(self, vec, top_k=4, knn=12):
        """Micro-genre tags inherited from the nearest neighbours' track_tags
        (mirrors anther_ml.corpus.place._query_tags neighbor-inheritance path)."""
        q = self.index.transform_query(vec)
        sims = self.index.embeddings @ q
        nbr = np.argsort(sims)[::-1][:knn]
        score = {}
        for i in nbr:
            for t in self.track_tags[int(i)]["tags"]:
                score[t["genre"]] = score.get(t["genre"], 0.0) + t["score"] / knn
        top = sorted(score.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"genre": g, "score": round(s, 4)} for g, s in top]

    # ---- shared similarity banding ------------------------------------------
    def band_for_cosine(self, raw_cos):
        """One of 'near-identical' / 'close' / 'related' / 'distant', using
        the same calibrated cutoffs ui/atlas.py stamps on persisted map
        edges (see self.link_thresholds / anther_ml.calibration)."""
        return self._link_calibration.band_for_cosine(raw_cos, self.link_thresholds)

    def display_score(self, raw_cos, clip_low=False):
        """The 0-100 human-readable score for a raw cosine, calibrated the
        same way as a persisted map edge's ``score`` field. ``clip_low=False``
        by default here (fresh mentor-side cosines, unlike drawn map edges,
        are not structurally bounded below the qq_threshold)."""
        return self._link_calibration.display_score(raw_cos, self.link_thresholds, clip_low=clip_low)

    # ---- convenience reads the tool layer uses ------------------------------
    def tags_for_index(self, idx, top_k=4):
        """Micro-genre names stored on a corpus track, best first."""
        tags = self.track_tags[int(idx)].get("tags", [])
        tags = sorted(tags, key=lambda t: -float(t.get("score", 0.0)))
        return [t["genre"] for t in tags[:top_k] if t.get("genre")]

    def cluster_label(self, cluster_id):
        if cluster_id is None:
            return ""
        try:
            profile = self.profiles.get(int(cluster_id), {})
        except (TypeError, ValueError):
            return ""
        if profile.get("label_final"):
            return profile["label_final"]
        ex = profile.get("exemplars", [])[:2]
        return ", ".join(f"{e.get('artist','')} - {e.get('name','')}" for e in ex)

    def cluster_of_index(self, idx):
        if self.labels is None:
            return None
        return int(self.labels[int(idx)])

    def vector_for_index(self, idx):
        if self.raw_embeddings is not None:
            return np.asarray(self.raw_embeddings[idx], dtype=np.float32).reshape(-1)
        # Fallback only if raw embeddings file is unavailable.
        return np.asarray(self.index.embeddings[idx], dtype=np.float32).reshape(-1)

    # ---- implementation ------------------------------------------------------
    @staticmethod
    def _norm_text(s):
        return " ".join(str(s or "").strip().lower().split())

    def _ensure_lookup(self):
        if self._name_keys is not None and self._artist_to_indices is not None:
            return
        self._name_keys = []
        self._artist_to_indices = {}
        for i, m in enumerate(self.index.metadata):
            artist = self._norm_text(m.get("artist", ""))
            name = self._norm_text(m.get("name", ""))
            key = f"{artist} - {name}".strip(" -")
            self._name_keys.append(key)
            if artist:
                self._artist_to_indices.setdefault(artist, []).append(i)

    def _resolve_name(self, spec):
        self._ensure_lookup()
        q = self._norm_text(spec)
        if not q:
            return None, {"input": spec, "type": "name", "resolved": False}

        # Exact artist match first -> artist centroid.
        if q in self._artist_to_indices:
            idxs = self._artist_to_indices[q]
            vecs = np.stack([self.vector_for_index(i) for i in idxs])
            out = vecs.mean(axis=0).astype(np.float32)
            return out, {
                "input": spec,
                "type": "artist",
                "resolved": True,
                "match": q,
                "count": len(idxs),
            }

        # Exact full-key contains / equality next.
        exact = [i for i, k in enumerate(self._name_keys) if q == k or q in k]
        if exact:
            i = exact[0]
            m = self.index.metadata[i]
            return self.vector_for_index(i), {
                "input": spec,
                "type": "track",
                "resolved": True,
                "match": f"{m.get('artist','')} - {m.get('name','')}",
                "index": int(i),
            }

        # Fuzzy match over full key, then artist names.
        full = get_close_matches(q, self._name_keys, n=1, cutoff=0.72)
        if full:
            i = self._name_keys.index(full[0])
            m = self.index.metadata[i]
            return self.vector_for_index(i), {
                "input": spec,
                "type": "track",
                "resolved": True,
                "match": f"{m.get('artist','')} - {m.get('name','')}",
                "index": int(i),
                "fuzzy": True,
            }

        artists = list(self._artist_to_indices.keys())
        artist_hit = get_close_matches(q, artists, n=1, cutoff=0.78)
        if artist_hit:
            a = artist_hit[0]
            idxs = self._artist_to_indices[a]
            vecs = np.stack([self.vector_for_index(i) for i in idxs])
            out = vecs.mean(axis=0).astype(np.float32)
            return out, {
                "input": spec,
                "type": "artist",
                "resolved": True,
                "match": a,
                "count": len(idxs),
                "fuzzy": True,
            }

        return None, {"input": spec, "type": "name", "resolved": False}

    def _embed_audio(self, audio_path):
        from anther_ml.embedding import get_embedding, load_mert
        if self._mert is None:
            self._mert = load_mert()
        model, processor, device = self._mert
        cfg = self.index.config or {}
        layer_aggregation = cfg.get("layer_aggregation", "mean")
        normalize = bool(cfg.get("loudness_normalize", True))
        return get_embedding(
            model, processor, audio_path, device,
            layer_aggregation=layer_aggregation,
            normalize=normalize,
        )


if __name__ == "__main__":
    svc = AntherSimilarityService()
    print("corpus loaded:", len(svc.index.metadata), "tracks")
    # validate with a real corpus vector as a stand-in query (no audio needed)
    raw = np.load(os.path.join(CORPUS_DIR, "embeddings.npy"), mmap_mode="r")
    meta = svc.index.metadata
    for probe_i in [0, 5000, 40000]:
        q = np.asarray(raw[probe_i], dtype=np.float32)
        print(f"\n=== query = corpus track #{probe_i}: "
              f"{meta[probe_i].get('artist','')} — {meta[probe_i].get('name','')} ===")
        for n in svc.similar(q, top_k=6):
            print(f"  - {n['artist']} — {n['name']} (sim {n['score']})")
        c = svc.cluster(q)
        print(f"  cluster: {c.get('label')} (conf {c.get('confidence')})")
        print("  tags:", ", ".join(t["genre"] for t in svc.tags(q)))
