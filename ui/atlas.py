"""
Frozen-corpus atlas: tiered search + placement for the d3 force-graph UI.

A song is resolved by a three-tier search — (1) the local corpus embedding,
(2) Deezer, (3) Spotify (last resort, gated on credentials) — then placed onto
the frozen 100k reference corpus. Placement returns the song's nearest corpus
neighbors (with cosine scores) and a cluster id; the frontend grows a d3 force
graph from these fragments. The corpus is loaded once and never re-fit; the
exact UMAP coords are irrelevant here (positions come from the force sim).

This module owns all corpus/MERT state so ui/app.py stays a thin router.
"""

import os
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import requests

from anther_ml.corpus.bundle import ReferenceCorpus
from anther_ml.corpus.place import embed_query, place
from anther_ml.spotify_deezer import _deezer_get, match_deezer_track, _norm, _ratio

# ── Config ───────────────────────────────────────────────────────────────────

CORPUS_DIR      = os.environ.get("ANTHER_CORPUS", "models/corpus_corpus_mpd_100k")
TOP_K           = 8       # neighbors pulled in per placed song
CORPUS_MIN_HITS = 5       # < this many strong corpus hits → fall through to Deezer
STRONG_SCORE    = 0.6     # _ratio threshold for a "strong" corpus match

# MERT vectors sit in a narrow cone: two *random* corpus songs are ~0.96 cosine
# apart, so a raw cosine floor means nothing (a low one links everything into one
# blob). We instead draw a query↔query edge only when the pair is more similar
# than QUERY_LINK_PCTL % of random corpus pairs — the same null-distribution
# framing the corpus placement uses. The concrete cosine cutoff is calibrated
# from the corpus once at load (see _calibrate_qq_threshold).
QUERY_LINK_PCTL = float(os.environ.get("ANTHER_QQ_PCTL", "95"))
_qq_threshold   = 0.981   # replaced at load() with the corpus-calibrated value

SESSION_DIR = Path(__file__).parent / "session"
GRAPH_PATH  = SESSION_DIR / "graph.json"

# ── Lazy singletons ──────────────────────────────────────────────────────────

_corpus = None
_id_to_idx: dict = {}
_id_to_cluster: dict = {}
_corpus_lock = threading.Lock()

_model = _processor = _device = None
_mert_lock = threading.Lock()

_graph = {"nodes": {}, "links": []}          # nodes keyed by id; links is a list
_link_keys: set = set()                      # (source, target) dedupe
_query_vecs: dict = {}                        # placed-song id → index-space unit vec
_graph_lock = threading.Lock()


# ── Corpus + MERT loading ────────────────────────────────────────────────────

def load() -> ReferenceCorpus:
    """Load the frozen corpus once (idempotent, thread-safe)."""
    global _corpus, _id_to_idx, _id_to_cluster
    with _corpus_lock:
        if _corpus is not None:
            return _corpus
        corpus = ReferenceCorpus.load(CORPUS_DIR)
        labels = corpus.labels
        id_to_idx, id_to_cluster = {}, {}
        for i, m in enumerate(corpus.metadata):
            tid = m.get("id")
            if tid is None:
                continue
            id_to_idx[tid] = i
            id_to_cluster[tid] = int(labels[i])
        _corpus, _id_to_idx, _id_to_cluster = corpus, id_to_idx, id_to_cluster
        _calibrate_qq_threshold(corpus)
        _load_graph()
        return _corpus


def _calibrate_qq_threshold(corpus, n_pairs: int = 200_000) -> None:
    """Set the query↔query cosine cutoff to the QUERY_LINK_PCTL percentile of
    random corpus-pair cosines (index space) — so an edge means "more similar
    than that fraction of released music," not an arbitrary absolute cosine."""
    global _qq_threshold
    E = corpus.index.embeddings                       # standardized + L2-normalized
    n = E.shape[0]
    if n < 2:
        return
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    mask = a != b
    cos = np.einsum("ij,ij->i", E[a[mask]], E[b[mask]])
    _qq_threshold = float(np.percentile(cos, QUERY_LINK_PCTL))


def warm() -> None:
    """Warm corpus (and later MERT) in a background thread at startup."""
    threading.Thread(target=load, daemon=True).start()


def is_ready() -> bool:
    return _corpus is not None


def _mert():
    """Lazily load MERT once (~30 s first call)."""
    global _model, _processor, _device
    with _mert_lock:
        if _model is None:
            from anther_ml.embedding import load_mert
            _model, _processor, _device = load_mert()
    return _model, _processor, _device


# ── Tiered search ────────────────────────────────────────────────────────────

def spotify_configured() -> bool:
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


def search(q: str, limit: int = 25) -> dict:
    """
    Three-tier search. Corpus first; fall through to Deezer only if the corpus
    yields fewer than CORPUS_MIN_HITS strong matches; fall through to Spotify
    only if Deezer is also empty. Returns {results, tiers, spotify_configured}.
    """
    load()
    q = (q or "").strip()
    if not q:
        return {"results": [], "tiers": {}, "spotify_configured": spotify_configured()}

    corpus_hits = _search_corpus(q, limit)
    strong = [h for h in corpus_hits if h["score"] >= STRONG_SCORE]
    results = list(corpus_hits)
    tiers = {"corpus": len(corpus_hits)}

    if len(strong) < CORPUS_MIN_HITS:
        deezer_hits = _search_deezer(q, limit)
        results += deezer_hits
        tiers["deezer"] = len(deezer_hits)
        if not deezer_hits:
            sp = _search_spotify(q, limit)
            results += sp["results"]
            tiers["spotify"] = len(sp["results"])

    return {"results": results, "tiers": tiers, "spotify_configured": spotify_configured()}


def _search_corpus(q: str, limit: int) -> list:
    """Substring pre-filter (all query tokens present) then _ratio ranking."""
    tokens = _norm(q).split()
    if not tokens:
        return []
    scored = []
    for i, m in enumerate(_corpus.metadata):
        name = m.get("name") or ""
        artist = m.get("artist") or ""
        hay = _norm(f"{name} {artist}")
        if not all(t in hay for t in tokens):
            continue
        score = max(_ratio(q, name), _ratio(q, artist), _ratio(q, f"{artist} {name}"))
        scored.append((score, i, m))
    scored.sort(key=lambda t: -t[0])
    out = []
    for score, i, m in scored[:limit]:
        tid = m.get("id")
        out.append({
            "source":  "corpus",
            "id":      tid,
            "idx":     i,
            "title":   m.get("name") or "",
            "artist":  m.get("artist") or "",
            "cluster": _id_to_cluster.get(tid),
            "score":   round(float(score), 3),
        })
    return out


def _search_deezer(q: str, limit: int) -> list:
    data = _deezer_get("search/track", params={"q": q, "limit": limit})
    if "error" in data:
        return []
    out = []
    for h in (data.get("data") or []):
        if not h.get("preview"):
            continue
        out.append({
            "source":      "deezer",
            "id":          f"deezer:{h['id']}",
            "deezer_id":   h["id"],
            "title":       h.get("title", ""),
            "artist":      (h.get("artist") or {}).get("name", ""),
            "album":       (h.get("album") or {}).get("title", ""),
            "cover":       (h.get("album") or {}).get("cover_small", ""),
            "preview_url": h["preview"],
        })
    return out


def _search_spotify(q: str, limit: int) -> dict:
    """
    Last-resort tier: Spotify search → ISRC → Deezer preview (so we still have
    audio to embed). Fully gated on SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET.
    """
    if not spotify_configured():
        return {"configured": False, "results": []}
    try:
        from anther_ml.mpd_ingest import get_spotify_token
        token = get_spotify_token()
        r = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": q, "type": "track", "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        items = ((r.json().get("tracks") or {}).get("items")) or []
    except Exception as exc:                                    # auth / network
        return {"configured": True, "error": str(exc), "results": []}

    out = []
    for t in items:
        artist = (t.get("artists") or [{}])[0].get("name", "")
        title = t.get("name", "")
        m = match_deezer_track({
            "isrc":   (t.get("external_ids") or {}).get("isrc"),
            "title":  title,
            "artist": artist,
        })
        if "error" in m:
            continue                                            # no Deezer preview → skip
        out.append({
            "source":      "spotify",
            "id":          f"spotify:{t.get('id')}",
            "deezer_id":   m.get("deezer_id"),
            "title":       title,
            "artist":      artist,
            "preview_url": m["preview"],
        })
    return {"configured": True, "results": out}


# ── Placement → graph fragment ───────────────────────────────────────────────

def place_song(result: dict) -> dict:
    """
    Place one search result onto the frozen corpus and merge it into the graph.
    Returns the fragment {nodes, links} that was newly added (for the frontend
    to splice into the running force simulation).
    """
    corpus = load()
    source = result.get("source")

    if source == "corpus":
        idx = result.get("idx")
        if idx is None:
            idx = _id_to_idx.get(result.get("id"))
        if idx is None:
            raise ValueError("corpus result missing idx/id")
        self_id = corpus.metadata[idx].get("id")
        raw_vec = corpus.embeddings[idx]
        neighbors = corpus.index.query(raw_vec, top_k=TOP_K + 1)
        neighbors = [n for n in neighbors if n.get("id") != self_id][:TOP_K]
        node = {
            "id":         self_id,
            "name":       corpus.metadata[idx].get("name", ""),
            "artist":     corpus.metadata[idx].get("artist", ""),
            "cluster":    _id_to_cluster.get(self_id),
            "kind":       "query",
            "confidence": 1.0,
            "source":     "corpus",
        }
    else:
        if source == "upload":
            path, cleanup = Path(result["path"]), None
        else:
            path, cleanup = _download_preview(result)
        try:
            model, processor, device = _mert()
            vec = embed_query(path, corpus, model, processor, device)
        finally:
            if cleanup:
                cleanup()
        raw_vec = vec
        res = place(corpus, vec, top_k=TOP_K)
        neighbors = res["neighbors"]
        node = {
            "id":         result.get("id"),
            "name":       result.get("title", ""),
            "artist":     result.get("artist", ""),
            "cluster":    int(res["cluster"]["id"]),
            "kind":       "query",
            "confidence": round(float(res["cluster"]["confidence"]), 3),
            "source":     source,
        }

    qvec = corpus.index.transform_query(raw_vec)
    return _merge_fragment(node, neighbors, qvec)


def _download_preview(result: dict):
    """Download a (possibly re-freshed) Deezer preview to a temp mp3."""
    url = result.get("preview_url")
    did = result.get("deezer_id")
    if did:                                                     # signed URLs expire
        fresh = _deezer_get(f"track/{did}").get("preview")
        if fresh:
            url = fresh
    if not url:
        raise ValueError("no preview URL available")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(r.content)
    tmp.close()
    p = Path(tmp.name)
    return p, lambda: p.unlink(missing_ok=True)


def _merge_fragment(node: dict, neighbors: list, qvec=None) -> dict:
    """Upsert the query node + its neighbor nodes/links into the graph.

    Beyond the query→corpus neighbor edges, this also wires the placed song
    directly to every *other* placed song whose index-space cosine clears the
    corpus-calibrated ``_qq_threshold`` — so similar songs you add pull together
    in the force sim regardless of which Leiden cluster each landed in.
    """
    added_nodes, added_links = [], []
    with _graph_lock:
        existing = _graph["nodes"].get(node["id"])
        if existing is None:
            _graph["nodes"][node["id"]] = node
            added_nodes.append(node)
        else:                                                  # re-add → promote to query
            existing.update(kind="query", confidence=node.get("confidence"))

        # ── query↔query similarity edges ──
        if qvec is not None:
            for other_id, ovec in _query_vecs.items():
                if other_id == node["id"]:
                    continue
                score = float(np.dot(qvec, ovec))
                if score < _qq_threshold:
                    continue
                key = (node["id"], other_id)
                rkey = (other_id, node["id"])
                if key in _link_keys or rkey in _link_keys:
                    continue
                link = {"source": node["id"], "target": other_id,
                        "value": round(score, 3), "kind": "qq"}
                _graph["links"].append(link)
                _link_keys.add(key)
                added_links.append(link)
            _query_vecs[node["id"]] = qvec

        for n in neighbors:
            nid = n.get("id")
            if not nid:
                continue
            if nid not in _graph["nodes"]:
                nnode = {
                    "id":      nid,
                    "name":    n.get("name", ""),
                    "artist":  n.get("artist", ""),
                    "cluster": _id_to_cluster.get(nid, n.get("cluster")),
                    "kind":    "corpus",
                    "source":  "corpus",
                }
                _graph["nodes"][nid] = nnode
                added_nodes.append(nnode)
            key = (node["id"], nid)
            if key not in _link_keys and node["id"] != nid:
                link = {"source": node["id"], "target": nid,
                        "value": round(float(n.get("score", 0.0)), 3)}
                _graph["links"].append(link)
                _link_keys.add(key)
                added_links.append(link)
        _save_graph()
    return {"nodes": added_nodes, "links": added_links}


# ── Graph persistence ────────────────────────────────────────────────────────

def get_graph() -> dict:
    with _graph_lock:
        return {"nodes": list(_graph["nodes"].values()), "links": list(_graph["links"])}


def _save_graph() -> None:
    SESSION_DIR.mkdir(exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(
        {"nodes": list(_graph["nodes"].values()), "links": _graph["links"]}))


def _load_graph() -> None:
    global _graph, _link_keys, _query_vecs
    if not GRAPH_PATH.exists():
        return
    try:
        data = json.loads(GRAPH_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    links = data.get("links", [])
    _graph = {"nodes": nodes, "links": links}
    _link_keys = {(l["source"], l["target"]) for l in links}

    # Best-effort: rebuild query→query similarity vecs for corpus-source songs so
    # newly placed songs can still cross-link against them. Deezer/upload vecs are
    # not persisted and can't be recovered — those keep their saved edges only.
    _query_vecs = {}
    for n in nodes.values():
        if n.get("kind") != "query":
            continue
        idx = _id_to_idx.get(n["id"])
        if idx is None:
            continue
        _query_vecs[n["id"]] = _corpus.index.transform_query(_corpus.embeddings[idx])
