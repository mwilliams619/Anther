"""The Anther 'sounds like' tool.

Wraps the frozen Anther reference corpus (99,618 MPD tracks, MERT-v1-330M
embeddings) so the mentor can tell an artist which released songs / artists
their track sounds like, plus the micro-genre tags and cluster it lands in.

Two entry points:
  * sounds_like_from_audio(path)  -> embeds audio with MERT, then place()
  * sounds_like_from_vec(vec)     -> place() a raw 1024-d vector (no audio;
                                     used for validation with corpus vectors)

Both return a compact dict the LLM can read out:
  {neighbors: [{artist, name, score}], cluster: {...}, tags: [...]}
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


class AntherSoundsLike:
    """Loads the frozen corpus's SongIndex + track_tags directly.

    We deliberately load the (index.npy, index.json) pair and track_tags.json
    rather than the full ReferenceCorpus bundle: the bundle's leiden.pkl was
    pickled under numpy 2.x and won't unpickle under this env's numpy 1.26.
    Neighbors (nearest released tracks) + inherited micro-genre tags are the
    core of the 'sounds like' answer and need neither leiden nor the audio
    feature stack.
    """
    def __init__(self, corpus_dir=CORPUS_DIR):
        import json
        from anther_ml.similarity import SongIndex
        self.corpus_dir = corpus_dir
        self.index = SongIndex.load(os.path.join(corpus_dir, "index"))
        with open(os.path.join(corpus_dir, "track_tags.json")) as f:
            self.track_tags = json.load(f)
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

    def _vector_for_index(self, idx):
        if self.raw_embeddings is not None:
            return np.asarray(self.raw_embeddings[idx], dtype=np.float32).reshape(-1)
        # Fallback only if raw embeddings file is unavailable.
        return np.asarray(self.index.embeddings[idx], dtype=np.float32).reshape(-1)

    def _resolve_name(self, spec):
        self._ensure_lookup()
        q = self._norm_text(spec)
        if not q:
            return None, {"input": spec, "type": "name", "resolved": False}

        # Exact artist match first -> artist centroid.
        if q in self._artist_to_indices:
            idxs = self._artist_to_indices[q]
            vecs = np.stack([self._vector_for_index(i) for i in idxs])
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
            return self._vector_for_index(i), {
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
            return self._vector_for_index(i), {
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
            vecs = np.stack([self._vector_for_index(i) for i in idxs])
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

    def _resolve_spec(self, spec_text, graph_ctx=None):
        """On-screen-first name resolution: live graph tiers, then corpus.

        ``graph_ctx`` is a ``mentor_graphctx.GraphContext`` (or None). When it has
        placed nodes, we try to resolve the anchor against the on-screen session
        first; only if that misses do we fall back to the frozen corpus. Passing
        ``graph_ctx=None`` reproduces the original corpus-only behaviour, so
        existing callers keep working.
        """
        if graph_ctx is not None:
            try:
                if graph_ctx.has_nodes():
                    vec, info = graph_ctx.resolve(spec_text)
                    if info.get("resolved"):
                        return vec, info
            except Exception:
                pass
        return self._resolve_name(spec_text)

    def resolve_anchor(self, spec, context=None, graph_ctx=None):
        """Resolve a query anchor from audio path or name string.

        Resolution order: context ref ("me") -> audio -> on-screen session
        (``graph_ctx``) -> frozen corpus. Returns (vec, info) where info carries
        resolution metadata (``type`` distinguishes onscreen_* from corpus).
        """
        spec_text = str(spec).strip() if spec is not None else ""
        spec_key = self._norm_text(spec_text)
        if spec_key in ("me", "my sound", "my track", "my song", "those artists"):
            ref = getattr(context, "last_anchor", None) if context is not None else None
            if ref:
                vec, info = self._resolve_spec(ref, graph_ctx=graph_ctx)
                if info.get("resolved"):
                    info["input"] = spec_text
                    info["type"] = "context_ref"
                    info["ref"] = ref
                    return vec, info
            return None, {
                "input": spec_text,
                "type": "context_ref",
                "resolved": False,
                "error": "no_prior_anchor",
            }

        if isinstance(spec, str) and os.path.exists(spec):
            vec = self._embed_audio(spec)
            return vec, {"input": spec, "type": "audio", "resolved": True, "path": spec}
        if isinstance(spec, dict) and spec.get("audio_path"):
            path = spec["audio_path"]
            vec = self._embed_audio(path)
            return vec, {"input": path, "type": "audio", "resolved": True, "path": path}
        return self._resolve_spec(spec_text, graph_ctx=graph_ctx)

    def resolve_two(self, spec_a, spec_b, context=None, graph_ctx=None):
        vec_a, info_a = self.resolve_anchor(spec_a, context=context, graph_ctx=graph_ctx)
        vec_b, info_b = self.resolve_anchor(spec_b, context=context, graph_ctx=graph_ctx)
        return (vec_a, info_a), (vec_b, info_b)

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

    def cluster_of(self, vec, knn=25):
        """Infer cluster by neighbor voting over labels.npy (no leiden.pkl)."""
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

    def _tags_for(self, vec, top_k=4, knn=10):
        """Micro-genre tags inherited from the nearest neighbors' track_tags
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

    # ---- placement ---------------------------------------------------------
    def _place(self, vec, top_k=8):
        neighbors = self.index.query(vec, top_k=top_k)
        seen, neigh = set(), []
        for n in neighbors:
            key = (n.get("artist", ""), n.get("name", ""))
            if key in seen:
                continue
            seen.add(key)
            neigh.append({"artist": n.get("artist", ""),
                          "name": n.get("name", ""),
                          "score": round(n["score"], 3)})
        return {
            "neighbors": neigh,
            "cluster": self.cluster_of(vec),
            "tags": self._tags_for(vec),
        }

    def sounds_like_from_vec(self, vec, top_k=8):
        return self._place(np.asarray(vec, dtype=np.float32).reshape(-1), top_k)

    def sounds_like_from_audio(self, audio_path, top_k=8):
        vec = self._embed_audio(audio_path)
        return self._place(vec, top_k)

    # ---- for the LLM tool interface ---------------------------------------
    def format_for_prompt(self, result):
        lines = []
        neigh = result["neighbors"][:5]
        if neigh:
            lines.append("Nearest released tracks:")
            for n in neigh:
                lines.append(f"  - {n['artist']} — {n['name']} (sim {n['score']})")
        if result["tags"]:
            tg = ", ".join(f"{t['genre']}" for t in result["tags"][:4])
            lines.append(f"Micro-genre tags: {tg}")
        c = result.get("cluster") or {}
        if c.get("label"):
            lines.append(f"Sits in the '{c['label']}' cluster (conf {c.get('confidence','')}).")
        return "\n".join(lines)


if __name__ == "__main__":
    tool = AntherSoundsLike()
    print("corpus loaded:", len(tool.index.metadata), "tracks")
    # validate with a real corpus vector as a stand-in query (no audio needed)
    raw = np.load(os.path.join(CORPUS_DIR, "embeddings.npy"), mmap_mode="r")
    meta = tool.index.metadata
    for probe_i in [0, 5000, 40000]:
        q = np.asarray(raw[probe_i], dtype=np.float32)
        res = tool.sounds_like_from_vec(q, top_k=6)
        print(f"\n=== query = corpus track #{probe_i}: "
              f"{meta[probe_i].get('artist','')} — {meta[probe_i].get('name','')} ===")
        print(tool.format_for_prompt(res))
