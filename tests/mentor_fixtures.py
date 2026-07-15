"""Shared fixtures for the mentor test suite.

The philosophy (set during the v4 agent refactor): mentor tests place songs on
a synthetic on-screen graph and interact with it the way the UI does — no GPU,
no LLM weights, no 100k corpus. The similarity service is faked at its public
API (resolve/similar/cluster/tags), vectors are tiny 2-d embeddings, and the
LLM is a scripted stand-in.
"""
import json

import numpy as np

from mentor.graph_context import GraphContext, MentorContext
from mentor.graph_tools import MentorGraphTools

# Two sonic territories, 2-d embedding space: "Dark Electronic" points toward
# [1, 0], "Pop Anthems" toward [0, 1].
DARK, POP = "Dark Electronic", "Pop Anthems"

GRAPH_NODES = [
    {"id": "n1", "artist": "Embolo", "name": "First Demo", "cluster": 0,
     "cluster_label": DARK,
     "tags": [{"genre": "dark", "score": 0.9}, {"genre": "bass", "score": 0.7}],
     "vec": [0.99, 0.05]},
    {"id": "n2", "artist": "Burial", "name": "Archangel", "cluster": 0,
     "cluster_label": DARK,
     "tags": [{"genre": "dark", "score": 0.8}, {"genre": "garage", "score": 0.6}],
     "vec": [0.97, 0.12]},
    {"id": "n3", "artist": "Aphex Twin", "name": "Windowlicker", "cluster": 0,
     "cluster_label": DARK,
     "tags": [{"genre": "idm", "score": 0.8}, {"genre": "dark", "score": 0.5}],
     "vec": [0.95, 0.2]},
    {"id": "n4", "artist": "Beyonce", "name": "Halo", "cluster": 1,
     "cluster_label": POP,
     "tags": [{"genre": "pop", "score": 0.9}, {"genre": "anthemic", "score": 0.6}],
     "vec": [0.08, 0.99]},
    {"id": "n5", "artist": "Beyonce", "name": "Irreplaceable", "cluster": 1,
     "cluster_label": POP,
     "tags": [{"genre": "pop", "score": 0.8}, {"genre": "rnb", "score": 0.7}],
     "vec": [0.15, 0.97]},
]

GRAPH_GROUPS = {"playlist:77": {"name": "Late Night Drive"}}

CORPUS = [
    {"artist": "Clark", "name": "Winter Linn", "cluster": 0,
     "tags": ["dark", "idm"], "vec": [0.96, 0.1]},
    {"artist": "Boards of Canada", "name": "Roygbiv", "cluster": 0,
     "tags": ["dark", "ambient"], "vec": [0.93, 0.25]},
    {"artist": "Kode9", "name": "9 Samurai", "cluster": 0,
     "tags": ["dark", "bass"], "vec": [0.98, 0.08]},
    {"artist": "Rihanna", "name": "Umbrella", "cluster": 1,
     "tags": ["pop", "rnb"], "vec": [0.1, 0.98]},
    {"artist": "Sia", "name": "Chandelier", "cluster": 1,
     "tags": ["pop", "anthemic"], "vec": [0.2, 0.96]},
]

CLUSTER_LABELS = {0: DARK, 1: POP}


def _norm_text(s):
    return " ".join(str(s or "").strip().lower().split())


class FakeIndex:
    def __init__(self, embeddings, metadata):
        mat = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        self.metadata = metadata

    @staticmethod
    def transform_query(vec):
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        return v / max(1e-8, float(np.linalg.norm(v)))


class FakeSimilarity:
    """Mirrors AntherSimilarityService's public API over the tiny fake corpus."""

    def __init__(self, corpus=CORPUS, cluster_labels=CLUSTER_LABELS):
        self.corpus = corpus
        self.cluster_labels = cluster_labels
        self.index = FakeIndex([r["vec"] for r in corpus],
                               [{"artist": r["artist"], "name": r["name"]} for r in corpus])
        self.labels = np.array([r["cluster"] for r in corpus])

    def resolve(self, spec, graph_ctx=None):
        if graph_ctx is not None and graph_ctx.has_nodes():
            vec, info = graph_ctx.resolve(spec)
            if info.get("resolved"):
                return vec, info
        q = _norm_text(spec)
        if not q:
            return None, {"input": spec, "type": "name", "resolved": False}
        for i, r in enumerate(self.corpus):
            key = _norm_text(f"{r['artist']} - {r['name']}")
            if q in (_norm_text(r["artist"]), _norm_text(r["name"])) or q in key:
                return self.index.embeddings[i], {
                    "input": spec, "type": "track", "resolved": True,
                    "match": f"{r['artist']} - {r['name']}", "index": i,
                }
        return None, {"input": spec, "type": "name", "resolved": False}

    def similar(self, vec, top_k=8):
        q = self.index.transform_query(vec)
        sims = self.index.embeddings @ q
        order = np.argsort(sims)[::-1][:top_k]
        return [{"artist": self.corpus[i]["artist"], "name": self.corpus[i]["name"],
                 "score": round(float(sims[i]), 4), "index": int(i)}
                for i in order]

    def cluster(self, vec, knn=3):
        top = self.similar(vec, top_k=knn)
        cid = int(self.labels[top[0]["index"]])
        return {"cluster_id": cid, "label": self.cluster_labels[cid],
                "confidence": 1.0, "exemplars": []}

    def tags(self, vec, top_k=4, knn=3):
        # The real service inherits tags from ~12 neighbours out of 100k, all
        # genuinely close; on a 5-track corpus that would drag in far-away
        # tracks, so the fake only ever consults the 3 nearest.
        knn = min(knn, 3)
        top = self.similar(vec, top_k=knn)
        score = {}
        for rank, r in enumerate(top):
            for t in self.corpus[r["index"]]["tags"]:
                score[t] = score.get(t, 0.0) + 1.0 / (rank + 1)
        best = sorted(score.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"genre": g, "score": round(s, 3)} for g, s in best]

    def tags_for_index(self, idx, top_k=4):
        return list(self.corpus[int(idx)]["tags"])[:top_k]

    def cluster_label(self, cluster_id):
        try:
            return self.cluster_labels.get(int(cluster_id), "")
        except (TypeError, ValueError):
            return ""

    def cluster_of_index(self, idx):
        return int(self.labels[int(idx)])

    def vector_for_index(self, idx):
        return np.asarray(self.index.embeddings[int(idx)], dtype=np.float32)


def place_graph(tmp_path, nodes=GRAPH_NODES, groups=GRAPH_GROUPS):
    """Write a synthetic on-screen session graph and return a GraphContext
    over it — the test-side equivalent of placing songs in the UI."""
    vecs = {n["id"]: np.asarray(n["vec"], dtype=np.float32) for n in nodes}
    stored = [{k: v for k, v in n.items() if k != "vec"} for n in nodes]
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({"nodes": stored, "links": [], "groups": groups}))
    return GraphContext(graph_path=str(graph_path), vec_reader=vecs.get)


def make_tools(tmp_path, nodes=GRAPH_NODES, groups=GRAPH_GROUPS):
    gc = place_graph(tmp_path, nodes=nodes, groups=groups)
    return MentorGraphTools(FakeSimilarity(), graph_ctx=gc)


def make_context(**kw):
    return MentorContext(**kw)
