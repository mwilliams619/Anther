"""RAG retriever over the music-mentor advice passages.

Embeds all answer passages with bge-small-en-v1.5 and does cosine top-k
retrieval. Used to ground the fine-tuned model in the source advice.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from paths import EMB_DIR, INDEX_NPY, INDEX_META, PASSAGES, ASSETS

# v2 index (2nd-person corpus, deflection templates removed). Falls back to v1
# if the v2 files are not present.
INDEX_NPY_V2  = os.path.join(ASSETS, "rag_index_v2.npy")
INDEX_META_V2 = os.path.join(ASSETS, "rag_index_meta_v2.json")

# bge asks for this prefix on the *query* side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _load_passages(path=None):
    if path is None:
        path = PASSAGES
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def build_index(model=None):
    from sentence_transformers import SentenceTransformer
    if model is None:
        model = SentenceTransformer(EMB_DIR, device="cuda")
    rows = _load_passages()
    # embed the answer text (the retrievable advice)
    texts = [r["passage"] for r in rows]
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    np.save(INDEX_NPY, emb)
    with open(INDEX_META, "w") as f:
        json.dump(rows, f)
    return emb, rows


class MentorRAG:
    """Loads the prebuilt passage index and retrieves grounding advice."""

    def __init__(self, device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMB_DIR, device=device)
        # prefer the v2 index (2nd-person, deflections removed)
        if os.path.exists(INDEX_NPY_V2) and os.path.exists(INDEX_META_V2):
            self.emb = np.load(INDEX_NPY_V2)
            self.rows = json.load(open(INDEX_META_V2))
            self.version = "v2"
        else:
            self.emb = np.load(INDEX_NPY)             # (N, d), normalized
            self.rows = json.load(open(INDEX_META))
            self.version = "v1"

    def retrieve(self, query, top_k=3, min_score=0.30):
        """Return (hits, max_score).

        `hits` is the list of passages above `min_score` (may be empty).
        `max_score` is the top cosine over the whole index regardless of the
        floor, so the caller can decide whether the query is in scope at all.
        """
        q = self.model.encode([QUERY_PREFIX + query],
                              normalize_embeddings=True).astype("float32")[0]
        scores = self.emb @ q                          # cosine (both normed)
        order = np.argsort(-scores)[:top_k]
        max_score = float(scores[order[0]]) if len(order) else 0.0
        hits = []
        for i in order:
            if scores[i] < min_score:
                continue
            r = self.rows[int(i)]
            hits.append({"score": float(scores[i]),
                         "question": r.get("question", ""),
                         "advice": r["passage"],
                         "source": r.get("source", "")})
        return hits, max_score


if __name__ == "__main__":
    emb, rows = build_index()
    print(f"BUILT index: {emb.shape[0]} passages x {emb.shape[1]} dims")
    # smoke test
    rag = MentorRAG()
    for q in ["How do I deal with writer's block?",
              "Should I release music on all platforms?",
              "how do i get more listeners"]:
        hits, ms = rag.retrieve(q, top_k=2)
        print(f"\nQ: {q}  (max_score {ms:.2f})")
        for h in hits:
            print(f"  [{h['score']:.2f}] {h['advice'][:90]}...")
