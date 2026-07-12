"""Deterministic graph tools over the frozen Anther similarity corpus."""

from __future__ import annotations

import os
import sys
from difflib import get_close_matches

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(HERE), "ui")
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import atlas


def _norm(s):
    return " ".join(str(s or "").strip().lower().split())


class MentorGraphTools:
    def __init__(self, anther, graph_ctx=None):
        self.anther = anther
        # Live on-screen session (mentor_graphctx.GraphContext). When present and
        # populated, anchors resolve on-screen-first and candidate pools default
        # to placed nodes. When None, behaviour is corpus-only (backward compat).
        if graph_ctx is None:
            try:
                from mentor_graphctx import GraphContext
                graph_ctx = GraphContext()
            except Exception:
                graph_ctx = None
        self.graph_ctx = graph_ctx

    def _artist_indices(self, artist):
        self.anther._ensure_lookup()
        q = self.anther._norm_text(artist)
        if not q:
            return [], {"input": artist, "resolved": False, "type": "artist"}
        if q in self.anther._artist_to_indices:
            return list(self.anther._artist_to_indices[q]), {
                "input": artist,
                "resolved": True,
                "type": "artist",
                "match": q,
                "count": len(self.anther._artist_to_indices[q]),
            }
        keys = list(self.anther._artist_to_indices.keys())
        hit = get_close_matches(q, keys, n=1, cutoff=0.72)
        if hit:
            idxs = list(self.anther._artist_to_indices[hit[0]])
            return idxs, {
                "input": artist,
                "resolved": True,
                "type": "artist",
                "match": hit[0],
                "count": len(idxs),
                "fuzzy": True,
            }
        return [], {"input": artist, "resolved": False, "type": "artist"}

    @staticmethod
    def _index_to_result(m, score, idx, rank):
        return {
            "rank": rank,
            "index": int(idx),
            "score": round(float(score), 4),
            "artist": m.get("artist", ""),
            "name": m.get("name", ""),
        }

    def _neighbors_with_indices(self, vec, top_k=8):
        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings @ q
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        out = []
        for rank, idx in enumerate(order, start=1):
            m = self.anther.index.metadata[int(idx)]
            out.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "score": round(float(sims[idx]), 4),
                    "artist": m.get("artist", ""),
                    "name": m.get("name", ""),
                }
            )
        return out

    # ---- on-screen candidate pools (PR-2) ---------------------------------
    def _onscreen_available(self):
        return self.graph_ctx is not None and self.graph_ctx.has_nodes()

    @staticmethod
    def _node_result(node, score, rank):
        return {
            "rank": rank,
            "id": node.get("id"),
            "score": round(float(score), 4),
            "artist": node.get("artist", ""),
            "name": node.get("name", ""),
            "cluster_id": node.get("cluster"),
            "cluster_label": node.get("cluster_label", ""),
        }

    def _onscreen_neighbors(self, vec, top_k=8, exclude_ids=None, pool_ids=None):
        """Nearest on-screen nodes to ``vec`` (index-space cosine).

        pool_ids restricts the candidate set (e.g. one artist's songs); otherwise
        every placed node with a cached vector is a candidate. Returns [] when the
        session has no usable vectors, so callers can fall back to corpus.
        """
        gc = self.graph_ctx
        if gc is None:
            return []
        if pool_ids is not None:
            ids, mats, nodes = [], [], []
            for nid in pool_ids:
                v = gc.vec_for_id(nid)
                if v is None:
                    continue
                ids.append(nid); mats.append(v); nodes.append(gc.node(nid))
            if not mats:
                return []
            mat = np.stack(mats).astype(np.float32)
        else:
            ids, mat, nodes = gc.candidate_vectors(exclude_ids=exclude_ids)
            if mat is None:
                return []
        q = self.anther.index.transform_query(vec)
        cand = np.stack([self.anther.index.transform_query(m) for m in mat])
        sims = cand @ q
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        return [self._node_result(nodes[int(i)], sims[int(i)], rank)
                for rank, i in enumerate(order, start=1)]

    def resolve(self, name, context=None):
        vec, info = self.anther.resolve_anchor(name, context=context, graph_ctx=self.graph_ctx)
        return {
            "tool": "resolve",
            "ok": bool(info.get("resolved")),
            "resolved": info,
            "vector_dim": int(vec.shape[0]) if vec is not None else 0,
        }

    def _exclude_ids_for(self, info):
        """Node ids to exclude from an on-screen neighbour search: the anchor's
        own node(s), so a track/artist isn't returned as its own neighbour."""
        gc = self.graph_ctx
        if gc is None:
            return set()
        if info.get("type") == "onscreen_track" and info.get("id"):
            return {info["id"]}
        if info.get("type") == "onscreen_artist" and info.get("match"):
            return set(gc.artist_ids(info["match"]))
        return set()

    def sounds_like(self, anchor, top_k=8, context=None, scope="onscreen"):
        vec, info = self.anther.resolve_anchor(anchor, context=context, graph_ctx=self.graph_ctx)
        if vec is None:
            return {"tool": "sounds_like", "ok": False, "error": "anchor_not_resolved", "resolved": info}
        neighbors, used_scope = [], scope
        if scope == "onscreen" and self._onscreen_available():
            neighbors = self._onscreen_neighbors(
                vec, top_k=top_k, exclude_ids=self._exclude_ids_for(info))
        if not neighbors:                      # empty onscreen pool -> corpus
            placed = self.anther.sounds_like_from_vec(vec, top_k=top_k)
            neighbors = placed.get("neighbors", [])[:top_k]
            used_scope = "corpus"
        return {
            "tool": "sounds_like",
            "ok": True,
            "resolved": info,
            "scope": used_scope,
            "neighbors": neighbors,
            "cluster": self.anther.cluster_of(vec),
            "tags": self.anther._tags_for(vec, top_k=4, knn=12),
        }

    def compare(self, anchor_a, anchor_b, top_k=8, context=None, scope="onscreen"):
        (vec_a, info_a), (vec_b, info_b) = self.anther.resolve_two(anchor_a, anchor_b, context=context, graph_ctx=self.graph_ctx)
        if vec_a is None or vec_b is None:
            return {
                "tool": "compare",
                "ok": False,
                "error": "anchor_not_resolved",
                "anchor_a": info_a,
                "anchor_b": info_b,
            }

        qa = self.anther.index.transform_query(vec_a)
        qb = self.anther.index.transform_query(vec_b)
        score = float(np.dot(qa, qb))
        used_scope = scope
        if scope == "onscreen" and self._onscreen_available():
            left = self._onscreen_neighbors(vec_a, top_k=top_k, exclude_ids=self._exclude_ids_for(info_a))
            right = self._onscreen_neighbors(vec_b, top_k=top_k, exclude_ids=self._exclude_ids_for(info_b))
            if not left and not right:
                left = self._neighbors_with_indices(vec_a, top_k=top_k)
                right = self._neighbors_with_indices(vec_b, top_k=top_k)
                used_scope = "corpus"
        else:
            left = self._neighbors_with_indices(vec_a, top_k=top_k)
            right = self._neighbors_with_indices(vec_b, top_k=top_k)
            used_scope = "corpus"
        return {
            "tool": "compare",
            "ok": True,
            "anchor_a": info_a,
            "anchor_b": info_b,
            "scope": used_scope,
            "similarity": round(score, 4),
            "left_neighbors": left,
            "right_neighbors": right,
            "left_cluster": self.anther.cluster_of(vec_a),
            "right_cluster": self.anther.cluster_of(vec_b),
        }

    def bridge(self, anchor_a, anchor_b, k=6, context=None, scope="onscreen", within=None):
        (vec_a, info_a), (vec_b, info_b) = self.anther.resolve_two(anchor_a, anchor_b, context=context, graph_ctx=self.graph_ctx)
        if vec_a is None or vec_b is None:
            return {
                "tool": "bridge",
                "ok": False,
                "error": "anchor_not_resolved",
                "anchor_a": info_a,
                "anchor_b": info_b,
            }

        na = self.anther.index.transform_query(vec_a)
        nb = self.anther.index.transform_query(vec_b)
        mid = (na + nb) / 2.0
        mid = mid / max(1e-8, np.linalg.norm(mid))

        # Relative lean with a scaled tie-band (the MERT cone is tight: between-
        # artist gaps ~0.03-0.04, within-artist std ~0.005, so the old absolute
        # +/-0.02 margin mislabelled at on-screen scale).
        TIE = 0.005

        def _lean(sa, sb):
            if sa > sb + TIE:
                return "a"
            if sb > sa + TIE:
                return "b"
            return "mid"

        # ---- within=<artist>: which of that artist's songs bridges A and B ----
        # This is the motivating question shape ("what song from UMO bridges the
        # Rae Sremmurd and Jimi Hendrix clusters"). Candidate pool = that artist's
        # on-screen songs; rank by min-similarity to both anchors.
        if within:
            pool_ids = self.graph_ctx.artist_ids(within) if self._onscreen_available() else []
            if not pool_ids:
                return {
                    "tool": "bridge", "ok": False, "error": "within_artist_not_onscreen",
                    "anchor_a": info_a, "anchor_b": info_b, "within": within,
                }
            enriched = []
            for nid in pool_ids:
                qv = self.graph_ctx.vec_for_id(nid)
                if qv is None:
                    continue
                tv = self.anther.index.transform_query(qv)
                sa = float(tv @ na); sb = float(tv @ nb)
                node = self.graph_ctx.node(nid)
                enriched.append({
                    "id": nid,
                    "artist": node.get("artist", ""),
                    "name": node.get("name", ""),
                    "sim_a": round(sa, 4),
                    "sim_b": round(sb, 4),
                    "bridge_score": round(min(sa, sb), 4),
                    "balance": round(1.0 - abs(sa - sb), 4),
                    "lean": _lean(sa, sb),
                    "cluster_id": node.get("cluster"),
                })
            enriched.sort(key=lambda e: -e["bridge_score"])
            return {
                "tool": "bridge", "ok": True, "within": within,
                "anchor_a": info_a, "anchor_b": info_b,
                "neighbors": enriched[:k],
                "most_balanced": max(enriched, key=lambda e: e["balance"]) if enriched else None,
                "scope": "onscreen",
            }

        # ---- ordinary bridge: nodes/tracks nearest the A-B midpoint ----
        used_scope = scope
        raw = []
        if scope == "onscreen" and self._onscreen_available():
            raw = self._onscreen_neighbors(mid, top_k=k)
        if not raw:
            raw = self._neighbors_with_indices(mid, top_k=k)
            used_scope = "corpus"

        enriched = []
        for n in raw:
            if "id" in n and n["id"] is not None and self._onscreen_available():
                qv = self.graph_ctx.vec_for_id(n["id"])
            else:
                qv = self.anther._vector_for_index(n["index"]) if "index" in n else None
            n2 = dict(n)
            if qv is not None:
                tv = self.anther.index.transform_query(qv)
                sa = float(tv @ na); sb = float(tv @ nb)
                n2["sim_a"] = round(sa, 4)
                n2["sim_b"] = round(sb, 4)
                n2["lean"] = _lean(sa, sb)
            if "index" in n and self.anther.labels is not None:
                n2["cluster_id"] = int(self.anther.labels[n["index"]])
            enriched.append(n2)

        return {
            "tool": "bridge",
            "ok": True,
            "anchor_a": info_a,
            "anchor_b": info_b,
            "scope": used_scope,
            "neighbors": enriched,
            "midpoint_cluster": self.anther.cluster_of(mid),
        }

    def crossover(self, anchor, k=6, context=None, scope="corpus"):
        vec, info = self.anther.resolve_anchor(anchor, context=context, graph_ctx=self.graph_ctx)
        if vec is None:
            return {"tool": "crossover", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        base = self.anther.cluster_of(vec)
        base_id = base.get("cluster_id")
        nbr = self._neighbors_with_indices(vec, top_k=250)
        out = []
        for n in nbr:
            idx = n["index"]
            if self.anther.labels is None:
                continue
            cid = int(self.anther.labels[idx])
            if cid == base_id:
                continue
            n2 = dict(n)
            n2["cluster_id"] = cid
            profile = self.anther.profiles.get(cid, {})
            ex = profile.get("exemplars", [])[:2]
            n2["cluster_label"] = profile.get("label_final") or ", ".join(
                f"{e.get('artist','')} - {e.get('name','')}" for e in ex
            )
            out.append(n2)
            if len(out) >= k:
                break

        return {
            "tool": "crossover",
            "ok": True,
            "resolved": info,
            "anchor_cluster": base,
            "neighbors": out,
        }

    def tagmates(self, anchor, k=8, context=None, scope="onscreen"):
        vec, info = self.anther.resolve_anchor(anchor, context=context, graph_ctx=self.graph_ctx)
        if vec is None:
            return {"tool": "tagmates", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        # Anchor tags: use the node's own probe tags when the anchor is on-screen
        # (all vec-bearing placed nodes carry probe tags), else infer from corpus.
        anchor_tags = self._onscreen_node_tags(info)
        if not anchor_tags:
            anchor_tags = [t["genre"] for t in self.anther._tags_for(vec, top_k=4, knn=12)]
        tag_set = set(anchor_tags)
        if not tag_set:
            return {
                "tool": "tagmates",
                "ok": True,
                "resolved": info,
                "anchor_tags": [],
                "neighbors": [],
            }

        # On-screen first: other placed nodes sharing a micro-genre tag.
        if scope == "onscreen" and self._onscreen_available():
            out = self._onscreen_tagmates(info, tag_set, k=k)
            if out:
                return {
                    "tool": "tagmates",
                    "ok": True,
                    "resolved": info,
                    "scope": "onscreen",
                    "anchor_tags": anchor_tags,
                    "neighbors": out,
                }

        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings @ q
        order = np.argsort(sims)[::-1][:800]
        out = []
        for idx in order:
            idx = int(idx)
            tags = self.anther.track_tags[idx].get("tags", [])
            genres = {t.get("genre") for t in tags if t.get("genre")}
            shared = sorted(genres & tag_set)
            if not shared:
                continue
            m = self.anther.index.metadata[idx]
            out.append(
                {
                    "index": idx,
                    "artist": m.get("artist", ""),
                    "name": m.get("name", ""),
                    "score": round(float(sims[idx]), 4),
                    "shared_tags": shared,
                }
            )
            if len(out) >= k:
                break

        return {
            "tool": "tagmates",
            "ok": True,
            "resolved": info,
            "scope": "corpus",
            "anchor_tags": anchor_tags,
            "neighbors": out,
        }

    # ---- on-screen tag helpers (PR-2 / micro_genres) ----------------------
    def _onscreen_node_tags(self, info):
        """Probe tags stored on the anchor's on-screen node(s), highest score
        first. For an artist anchor, aggregate across the artist's placed nodes."""
        gc = self.graph_ctx
        if gc is None:
            return []
        ids = []
        if info.get("type") == "onscreen_track" and info.get("id"):
            ids = [info["id"]]
        elif info.get("type") == "onscreen_artist" and info.get("match"):
            ids = gc.artist_ids(info["match"])
        if not ids:
            return []
        scored = {}
        for nid in ids:
            node = gc.node(nid) or {}
            for t in node.get("tags", []):
                g = t.get("genre")
                if not g:
                    continue
                scored[g] = max(scored.get(g, 0.0), float(t.get("score", 0.0)))
        return [g for g, _ in sorted(scored.items(), key=lambda kv: -kv[1])]

    def _onscreen_tagmates(self, info, tag_set, k=8):
        gc = self.graph_ctx
        exclude = self._exclude_ids_for(info)
        out = []
        for node in gc.nodes:
            nid = node.get("id")
            if nid in exclude:
                continue
            genres = {t.get("genre") for t in node.get("tags", []) if t.get("genre")}
            shared = sorted(genres & tag_set)
            if not shared:
                continue
            out.append({
                "id": nid,
                "artist": node.get("artist", ""),
                "name": node.get("name", ""),
                "cluster_id": node.get("cluster"),
                "shared_tags": shared,
            })
        out.sort(key=lambda e: -len(e["shared_tags"]))
        return out[:k]

    def artist_tracks(self, anchor, artist, top_k=8, context=None, scope="onscreen"):
        vec, info = self.anther.resolve_anchor(anchor, context=context, graph_ctx=self.graph_ctx)
        if vec is None:
            return {"tool": "artist_tracks", "ok": False, "error": "anchor_not_resolved", "resolved": info}

        # On-screen first: rank *that artist's placed songs* by similarity to the
        # anchor (Q4 "which of Jimi Hendrix's songs is closest to Embolo").
        if scope == "onscreen" and self._onscreen_available():
            pool_ids = self.graph_ctx.artist_ids(artist)
            if pool_ids:
                out = self._onscreen_neighbors(vec, top_k=top_k, pool_ids=pool_ids)
                if out:
                    return {
                        "tool": "artist_tracks",
                        "ok": True,
                        "resolved": info,
                        "scope": "onscreen",
                        "artist": {"input": artist, "match": artist, "count": len(pool_ids)},
                        "neighbors": out,
                    }

        idxs, artist_info = self._artist_indices(artist)
        if not idxs:
            return {
                "tool": "artist_tracks",
                "ok": False,
                "error": "artist_not_resolved",
                "resolved": info,
                "artist": artist_info,
            }

        q = self.anther.index.transform_query(vec)
        sims = self.anther.index.embeddings[idxs] @ q
        order = np.argsort(sims)[::-1][: max(top_k, 1)]
        out = []
        for rank, rel_idx in enumerate(order, start=1):
            idx = int(idxs[int(rel_idx)])
            m = self.anther.index.metadata[idx]
            out.append(self._index_to_result(m, sims[int(rel_idx)], idx, rank))

        return {
            "tool": "artist_tracks",
            "ok": True,
            "resolved": info,
            "scope": "corpus",
            "artist": artist_info,
            "neighbors": out,
        }

    # ---- PR-3 new tools ---------------------------------------------------
    def cluster_summary(self, cluster_id=None, top_n=None):
        """The map's sonic territories, read from cluster_profiles.json.

        With ``cluster_id`` -> one profile in detail; without -> every territory
        summary in the corpus (label, size, top tags, exemplars) unless ``top_n``
        explicitly caps it. This is what Q9 ("what are the major sonic
        territories?") reads instead of the model free-styling artist names,
        which is how it used to hallucinate.

        top_n=None means "no cap" — return all of them. (This used to default
        to a hardcoded 13, which was correct back when the corpus had exactly
        13 clusters; after reclustering to a different count it silently
        truncated the answer. Pass an explicit int to cap on purpose.)
        """
        profiles = self.anther.profiles or {}
        if not profiles:
            return {"tool": "cluster_summary", "ok": False, "error": "no_profiles"}

        def _summ(p):
            return {
                "cluster_id": p.get("cluster_id"),
                "label": p.get("label_final") or p.get("label", ""),
                "size": p.get("size"),
                "top_tags": [t[0] if isinstance(t, (list, tuple)) else t
                             for t in (p.get("top_tags") or [])[:6]],
                "exemplars": [f"{e.get('artist','')} — {e.get('name','')}"
                              for e in (p.get("exemplars") or [])[:3]],
                "tag_entropy_norm": round(float(p.get("tag_entropy_norm", 0.0)), 3),
            }

        if cluster_id is not None:
            try:
                key = int(cluster_id)
            except (TypeError, ValueError):
                key = cluster_id
            p = profiles.get(key)
            if p is None:
                return {"tool": "cluster_summary", "ok": False,
                        "error": "unknown_cluster", "cluster_id": cluster_id}
            return {"tool": "cluster_summary", "ok": True, "cluster": _summ(p)}

        keys = sorted(profiles.keys(), key=lambda k: (isinstance(k, str), k))
        clusters = [_summ(profiles[k]) for k in keys]
        if top_n is not None:
            clusters = clusters[:top_n]
        return {"tool": "cluster_summary", "ok": True,
                "n_clusters": len(clusters), "clusters": clusters}

    # get_cluster_profiles is the name the user referenced; keep it as an alias.
    def get_cluster_profiles(self, cluster_id=None, top_n=None):
        return self.cluster_summary(cluster_id=cluster_id, top_n=top_n)

    def micro_genres(self, anchor, context=None, top_k=6, scope="onscreen"):
        """Micro-genre tags for a track/artist (Q2).

        On-screen anchors read the probe tags already stored on their node(s)
        (2000-term vocabulary, tagged at placement). Off-screen anchors infer
        tags from corpus neighbours.
        """
        vec, info = self.anther.resolve_anchor(anchor, context=context, graph_ctx=self.graph_ctx)
        if vec is None:
            return {"tool": "micro_genres", "ok": False,
                    "error": "anchor_not_resolved", "resolved": info}
        source = "onscreen_probe"
        tags = self._onscreen_node_tags(info)[:top_k]
        if not tags:
            tags = [t["genre"] for t in self.anther._tags_for(vec, top_k=top_k, knn=12)]
            source = "corpus_neighbors"
        return {"tool": "micro_genres", "ok": True, "resolved": info,
                "source": source, "tags": tags}

    def coherence(self, scope="album", target=None, context=None):
        """How tight is a placed album / playlist / artist, and what are its
        outliers (Q6).

        ``scope`` in {"album","playlist","artist"} selects how to gather the
        node set; ``target`` names it (album/playlist id or name, or artist
        name). Reports centroid coherence, distinct-cluster spread, and z-score
        outliers (absolute cosines all sit ~0.98 in the tight MERT cone, so an
        absolute cutoff would flag all-or-nothing — z-score is the honest test).
        """
        gc = self.graph_ctx
        if gc is None or not gc.has_nodes():
            return {"tool": "coherence", "ok": False, "error": "no_onscreen_graph"}

        ids, label = self._coherence_pool(scope, target)
        if not ids:
            return {"tool": "coherence", "ok": False, "error": "group_not_found",
                    "scope": scope, "target": target}

        vecs, kept = [], []
        for nid in ids:
            v = gc.vec_for_id(nid)
            if v is None:
                continue
            vecs.append(self.anther.index.transform_query(v))
            kept.append(nid)
        if len(vecs) < 2:
            return {"tool": "coherence", "ok": False, "error": "too_few_vectors",
                    "scope": scope, "target": label, "n": len(vecs)}

        mat = np.stack(vecs)
        centroid = mat.mean(axis=0)
        centroid = centroid / max(1e-8, np.linalg.norm(centroid))
        sims = mat @ centroid
        mean_s, std_s = float(sims.mean()), float(sims.std())
        clusters = {}
        members = []
        for nid, s in zip(kept, sims):
            node = gc.node(nid) or {}
            z = (float(s) - mean_s) / std_s if std_s > 1e-9 else 0.0
            members.append({
                "id": nid, "artist": node.get("artist", ""), "name": node.get("name", ""),
                "cluster_id": node.get("cluster"),
                "sim_to_centroid": round(float(s), 4), "z": round(z, 2),
            })
            cid = node.get("cluster")
            clusters[cid] = clusters.get(cid, 0) + 1
        members.sort(key=lambda m: m["sim_to_centroid"])
        outliers = [m for m in members if m["z"] <= -2.0] or members[:1]
        return {
            "tool": "coherence", "ok": True, "scope": scope, "target": label,
            "n": len(kept),
            "coherence_mean": round(mean_s, 4), "coherence_std": round(std_s, 4),
            "distinct_clusters": len(clusters),
            "cluster_spread": sorted(clusters.items(), key=lambda kv: -kv[1]),
            "outliers": outliers,
            "members": members,
        }

    def _coherence_pool(self, scope, target):
        """(node_ids, human_label) for a coherence group."""
        gc = self.graph_ctx
        t = _norm(target) if target else ""
        if scope == "artist":
            if t:
                ids = gc.artist_ids(target)
                return ids, target
            return [], target
        # album / playlist -> match graph groups, then node playlist_pid
        groups = gc.groups or {}
        gid = None
        for key, meta in groups.items():
            nm = _norm(meta.get("name", ""))
            if not t or t in nm or _norm(key) == t or t in _norm(key):
                gid = key
                label = meta.get("name", key)
                break
        else:
            label = target
        if gid is None and t:
            # allow bare pid like "455130" or "album:455130"
            for n in gc.nodes:
                if _norm(n.get("playlist_pid", "")) in (t, f"album:{t}"):
                    gid = n.get("playlist_pid"); label = gid; break
        if gid is None:
            return [], target
        ids = [n["id"] for n in gc.nodes if n.get("playlist_pid") == gid]
        return ids, label

    def search(self, query, limit=25):
        return atlas.search(query, limit=limit)

    def search_playlists(self, query, limit=20):
        return atlas.search_playlists(query, limit=limit)

    def search_albums(self, query, limit=20):
        return atlas.search_albums(query, limit=limit)

    def place_song(self, result):
        return atlas.place_song(result)

    def place_playlist(self, pid):
        return atlas.place_playlist(pid)

    def place_album(self, album_id):
        return atlas.place_album(album_id)

    def recommend(self, seed_ids, top_k=20, method="centroid"):
        return atlas.recommend(seed_ids, top_k=top_k, method=method)

    def song_detail(self, song_id, top_n=10):
        return atlas.song_detail(song_id, top_n=top_n)

    def get_graph(self):
        return atlas.get_graph()

    def clear_graph(self):
        atlas.clear_graph()
        return {"tool": "clear_graph", "ok": True, "status": "cleared"}

    def remove_node(self, node_id):
        result = atlas.remove_node(node_id)
        if result is None:
            return {"tool": "remove_node", "ok": False, "error": "unknown_node"}
        result = dict(result)
        result["tool"] = "remove_node"
        result["ok"] = True
        return result


TOOLS = {
    "search": {
        "callable": MentorGraphTools.search,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "search_playlists": {
        "callable": MentorGraphTools.search_playlists,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "search_albums": {
        "callable": MentorGraphTools.search_albums,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "place_song": {
        "callable": MentorGraphTools.place_song,
        "schema": {
            "type": "object",
            "properties": {"result": {"type": "object"}},
            "required": ["result"],
        },
    },
    "place_playlist": {
        "callable": MentorGraphTools.place_playlist,
        "schema": {
            "type": "object",
            "properties": {"pid": {}},
            "required": ["pid"],
        },
    },
    "place_album": {
        "callable": MentorGraphTools.place_album,
        "schema": {
            "type": "object",
            "properties": {"album_id": {}},
            "required": ["album_id"],
        },
    },
    "recommend": {
        "callable": MentorGraphTools.recommend,
        "schema": {
            "type": "object",
            "properties": {
                "seed_ids": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "method": {"type": "string"},
            },
            "required": ["seed_ids"],
        },
    },
    "song_detail": {
        "callable": MentorGraphTools.song_detail,
        "schema": {
            "type": "object",
            "properties": {
                "song_id": {"type": "string"},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["song_id"],
        },
    },
    "graph": {
        "callable": MentorGraphTools.get_graph,
        "schema": {"type": "object", "properties": {}},
    },
    "clear_graph": {
        "callable": MentorGraphTools.clear_graph,
        "schema": {"type": "object", "properties": {}},
    },
    "remove_node": {
        "callable": MentorGraphTools.remove_node,
        "schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    "resolve": {
        "callable": MentorGraphTools.resolve,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "sounds_like": {
        "callable": MentorGraphTools.sounds_like,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
    "compare": {
        "callable": MentorGraphTools.compare,
        "schema": {
            "type": "object",
            "properties": {
                "anchor_a": {"type": "string"},
                "anchor_b": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor_a", "anchor_b"],
        },
    },
    "bridge": {
        "callable": MentorGraphTools.bridge,
        "schema": {
            "type": "object",
            "properties": {
                "anchor_a": {"type": "string"},
                "anchor_b": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor_a", "anchor_b"],
        },
    },
    "crossover": {
        "callable": MentorGraphTools.crossover,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
    "artist_tracks": {
        "callable": MentorGraphTools.artist_tracks,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "artist": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor", "artist"],
        },
    },
    "tagmates": {
        "callable": MentorGraphTools.tagmates,
        "schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["anchor"],
        },
    },
}
