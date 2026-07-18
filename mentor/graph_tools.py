"""The mentor's LLM-facing graph tools — music concepts only.

Eight read-only tools over the user's on-screen map (first) and the frozen
Anther corpus (fallback). Each tool retrieves facts deterministically and
phrases *why* two things are related from evidence it can prove: shared
micro-genre tags, cluster (territory) labels, and lean/balance scores.
Raw ML detail (embeddings, cosines, indices) stays internal — the agent and
the LLM behind it only ever see artists, songs, territories, and traits.

    inspect_graph(context)                what is on the map right now
    resolve_anchor(anchor, context)       who/what does a reference mean
    connections(anchor, k, context)       what is this ALREADY drawn to on the map
    neighbors(anchor, k, context)         what does this sound like (live kNN)
    compare(anchor_a, anchor_b, context)  how do these two relate
    bridge(anchor_a, anchor_b, k, ...)    what sits between these sounds
    explore_cluster(anchor, context)      where to explore next
    explain_node(anchor, context)         why is this node here

``connections`` and ``neighbors`` answer different questions and can
legitimately disagree: ``connections`` reads the edges the UI already drew
and persisted at placement time (frozen, ranked by the atlas's own
calibrated score); ``neighbors`` computes a fresh embedding kNN against
whatever is on-screen right now. A song's placed connections are not always
its current nearest neighbours — that's a real fact about the map, not a bug.

Anchor resolution order (the graph UI is the user's memory):
    1. the selected graph node ("me", "this song", or no anchor at all)
    2. visible graph nodes (uploads included — they are nodes too)
    3. the frozen corpus
"""

from __future__ import annotations

import numpy as np


def _norm(s):
    return " ".join(str(s or "").strip().lower().split())


# References that mean "the thing I have selected / my own sound".
SELF_REFS = {
    "", "me", "my", "mine", "my sound", "my track", "my song", "my demo",
    "this", "this song", "this track", "this node", "this one", "selected",
    "the selected node", "my upload", "my uploaded demo",
}

# Intent name -> (method name, allowed args). The agent validates against this.
INTENTS = {
    "inspect": ("inspect_graph", ()),
    "resolve": ("resolve_anchor", ("anchor",)),
    "connections": ("connections", ("anchor", "k")),
    "neighbors": ("neighbors", ("anchor", "k")),
    "compare": ("compare", ("anchor_a", "anchor_b")),
    "bridge": ("bridge", ("anchor_a", "anchor_b", "k")),
    "explore": ("explore_cluster", ("anchor",)),
    "explain": ("explain_node", ("anchor",)),
}


# 4-tier similarity bands, applied consistently wherever a tool cites a
# concrete similarity relationship (connections, neighbors, compare, bridge,
# explain_node — NOT inspect/resolve/explore, which don't score a pair).
# The bands themselves are calibrated centrally in anther_ml.calibration and
# shared with ui/atlas.py's persisted map edges (see mentor-graph-aware.md);
# this tuple is just the fallback ordering if a similarity backend doesn't
# expose calibration.
SIMILARITY_BANDS = ("near-identical", "close", "related", "distant")


class MentorGraphTools:
    def __init__(self, similarity, graph_ctx=None):
        self.similarity = similarity
        if graph_ctx is None:
            try:
                try:
                    from .graph_context import GraphContext
                except ImportError:
                    from graph_context import GraphContext
                graph_ctx = GraphContext()
            except Exception:
                graph_ctx = None
        self.graph_ctx = graph_ctx

    # ---- dispatcher ---------------------------------------------------------
    def execute(self, intent, args, context=None):
        """Run one intent with validated args. Unknown intents fail honestly."""
        spec = INTENTS.get(intent)
        if spec is None:
            return {"tool": intent, "ok": False, "error": "unknown_intent",
                    "message": "I don't have a move for that."}
        method_name, allowed = spec
        kwargs = {k: v for k, v in (args or {}).items() if k in allowed}
        if "k" in kwargs:
            try:
                kwargs["k"] = max(1, min(20, int(kwargs["k"])))
            except (TypeError, ValueError):
                kwargs.pop("k")
        return getattr(self, method_name)(context=context, **kwargs)

    # ---- 1. inspect_graph ----------------------------------------------------
    def inspect_graph(self, context=None):
        """What the user's map looks like right now."""
        gc = self.graph_ctx
        if gc is None or not gc.has_nodes():
            base = {"tool": "inspect_graph", "ok": True, "visible_nodes": 0,
                    "selected_node": None, "territories": [], "artists": [],
                    "groups": [], "recent_additions": [],
                    "message": "The map is empty — nothing has been placed yet."}
            return self._envelope(base, intent="inspect", subject=None, scope=None,
                                   provenance={"source": "graph_session"})
        nodes = gc.nodes
        territories, artists = {}, []
        seen_artists = set()
        for n in nodes:
            label = n.get("cluster_label") or ""
            if label:
                territories[label] = territories.get(label, 0) + 1
            a = n.get("artist", "")
            if a and _norm(a) not in seen_artists:
                seen_artists.add(_norm(a))
                artists.append(a)
        groups = [g.get("name", gid) for gid, g in (gc.groups or {}).items()]
        recent = [f"{n.get('artist','')} - {n.get('name','')}".strip(" -")
                  for n in nodes[-5:]]
        territory_rows = sorted(
            ({"territory": t, "songs": c} for t, c in territories.items()),
            key=lambda d: -d["songs"])
        selected = self._selected_summary(context)
        base = {
            "tool": "inspect_graph",
            "ok": True,
            "selected_node": selected,
            "visible_nodes": len(nodes),
            "territories": territory_rows,
            "artists": artists[:25],
            "groups": groups,
            "recent_additions": recent,
        }
        subject = None
        if selected:
            subject = {"label": f"{selected['artist']} - {selected['name']}".strip(" -"),
                       "kind": "graph_node", "id": getattr(context, "selected_node_id", None),
                       "on_map": True}
        return self._envelope(base, intent="inspect", subject=subject, scope="map",
                              findings=territory_rows,
                              provenance={"source": "graph_session"})

    def _selected_summary(self, context):
        node = self._selected_node(context)
        if node is None:
            return None
        return {
            "name": node.get("name", ""),
            "artist": node.get("artist", ""),
            "territory": node.get("cluster_label", ""),
            "traits": self._node_tag_names(node),
        }

    # ---- 2. resolve_anchor -----------------------------------------------------
    def resolve_anchor(self, anchor="", context=None):
        """Resolve a user reference ("me", "this song", "Halo", "Burial")."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            base = {"tool": "resolve_anchor", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
            return self._envelope(base, intent="resolve", subject=None,
                                  provenance={"source": "anchor_resolution"})
        base = {
            "tool": "resolve_anchor",
            "ok": True,
            "resolved": True,
            "type": self._public_type(info),
            "name": self._display_name(info),
            "artist": info.get("artist", ""),
            "id": info.get("id"),
        }
        return self._envelope(base, intent="resolve", subject=self._subject(info),
                              provenance={"source": "anchor_resolution"})

    # ---- 3. connections ---------------------------------------------------------
    def connections(self, anchor="", k=6, context=None):
        """"What is this connected to?" — persisted map edges ONLY.

        Strict and persisted-only by design: this reads exactly the qq edges
        ``ui/atlas.py`` already drew and stored in the session's graph.json
        at placement time (via GraphContext.links_for_id) — it never falls
        back to a live kNN search or the corpus. If the anchor isn't a node
        placed on the current map, or a placed node simply has no drawn
        edges, that is the honest answer: "not on the map" / "no connections
        yet", never a silently-substituted live-similarity guess. That
        substitution is exactly the bug this tool exists to prevent — see
        ``neighbors`` for the live-kNN equivalent, which is a DIFFERENT
        question and can legitimately return a different answer.
        """
        vec, info = self._resolve(anchor, context)
        if vec is None:
            base = {"tool": "connections", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
            return self._envelope(base, intent="connections", subject=None,
                                  scope="persisted",
                                  provenance={"source": "persisted_links"})
        if self._public_type(info) != "graph_node" or not info.get("id") or self.graph_ctx is None:
            name = self._display_name(info)
            base = {"tool": "connections", "ok": False, "error": "not_on_map",
                    "anchor": anchor, "name": name,
                    "message": (f"“{name}” isn't placed on your map, so it has no drawn "
                                "connections to read. Place it on the map, or ask what it "
                                "sounds like instead (that's a live comparison, not a map edge).")}
            return self._envelope(base, intent="connections", subject=self._subject(info),
                                  scope="persisted",
                                  provenance={"source": "persisted_links"})
        name = self._display_name(info)
        anchor_tags, anchor_territory = self._traits(vec, info)
        edges = self.graph_ctx.links_for_id(info["id"])
        edges = sorted(edges, key=lambda e: -(e.get("value") if e.get("value") is not None else -1))
        edges = edges[: max(k, 1)]
        findings = []
        for rank, e in enumerate(edges, start=1):
            other_node = self.graph_ctx.node(e["id"])
            other_tags = self._node_tag_names(other_node) if other_node is not None else []
            band, score = self._band_score(e.get("value"))
            findings.append({
                "artist": e.get("artist", ""),
                "song": e.get("name", ""),
                "territory": e.get("cluster_label", ""),
                "relationship": self._relationship(
                    anchor_tags, other_tags, anchor_territory, e.get("cluster_label", ""), rank),
                "band": band,
                "score": score,
                "rank": rank,
            })
        base = {
            "tool": "connections",
            "ok": True,
            "anchor": name,
            "territory": anchor_territory,
            "traits": anchor_tags,
            "connections": findings,
            "message": (f"{name} has no drawn connections on the map yet."
                        if not findings else None),
        }
        return self._envelope(base, intent="connections", subject=self._subject(info),
                              scope="persisted", findings=findings,
                              provenance={"source": "persisted_links"})

    # ---- 4. neighbors ---------------------------------------------------------
    def neighbors(self, anchor="", k=6, context=None):
        """"What does this sound like?" — closest songs, with reasons.

        This is LIVE embedding kNN — computed fresh against whatever is
        on-screen (or the corpus fallback) right now. Distinct from
        ``connections``, which reads edges the UI already drew and persisted
        at placement time; the two can diverge (a song's live nearest
        neighbours aren't necessarily the ones it has a drawn map edge to)."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            base = {"tool": "neighbors", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
            return self._envelope(base, intent="neighbors", subject=None,
                                  provenance={"source": "live_knn"})
        anchor_tags, anchor_territory = self._traits(vec, info)
        rows, scope = self._nearest(vec, k, exclude=self._own_ids(info))
        out = []
        for r in rows:
            band, score = self._band_score(r.get("cosine"))
            out.append({
                "artist": r["artist"],
                "song": r["song"],
                "relationship": self._relationship(
                    anchor_tags, r["tags"], anchor_territory, r["territory"], r["rank"]),
                "band": band,
                "score": score,
            })
        base = {
            "tool": "neighbors",
            "ok": True,
            "anchor": self._display_name(info),
            "scope": scope,
            "territory": anchor_territory,
            "traits": anchor_tags,
            "neighbors": out,
        }
        return self._envelope(base, intent="neighbors", subject=self._subject(info),
                              scope=scope, findings=out,
                              provenance={"source": "live_knn"})

    # ---- 5. compare -------------------------------------------------------------
    def compare(self, anchor_a="", anchor_b="", context=None):
        """How two sounds relate: shared traits, differences, what links them."""
        vec_a, info_a = self._resolve(anchor_a, context)
        vec_b, info_b = self._resolve(anchor_b, context)
        if vec_a is None or vec_b is None:
            miss = anchor_a if vec_a is None else anchor_b
            info = info_a if vec_a is None else info_b
            base = {"tool": "compare", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": miss, "message": self._miss_message(miss, info)}
            return self._envelope(base, intent="compare", subject=None,
                                  provenance={"source": "pairwise_cosine"})
        tags_a, terr_a = self._traits(vec_a, info_a)
        tags_b, terr_b = self._traits(vec_b, info_b)
        name_a, name_b = self._display_name(info_a), self._display_name(info_b)
        shared = [t for t in tags_a if t in tags_b]
        only_a = [t for t in tags_a if t not in tags_b]
        only_b = [t for t in tags_b if t not in tags_a]

        qa = self.similarity.index.transform_query(vec_a)
        qb = self.similarity.index.transform_query(vec_b)
        cosine = float(np.dot(qa, qb))

        if terr_a and terr_a == terr_b:
            summary = f"{name_a} and {name_b} sit in the same {terr_a} territory"
        elif terr_a and terr_b:
            summary = (f"{name_a} lives in the {terr_a} territory while "
                       f"{name_b} sits in {terr_b}")
        else:
            summary = f"{name_a} and {name_b} occupy different parts of the map"
        if shared:
            summary += f"; they share a {', '.join(shared[:3])} character"

        mid = (qa + qb) / 2.0
        mid = mid / max(1e-8, np.linalg.norm(mid))
        bridge_rows, _ = self._nearest(mid, 3, exclude=self._own_ids(info_a) | self._own_ids(info_b))
        band, score = self._band_score(cosine)
        base = {
            "tool": "compare",
            "ok": True,
            "anchor_a": name_a,
            "anchor_b": name_b,
            "similarity_summary": summary,
            "shared_traits": shared,
            "differences": {"only_" + name_a: only_a, "only_" + name_b: only_b},
            "bridge_candidates": [f"{r['artist']} - {r['song']}".strip(" -")
                                  for r in bridge_rows],
            "band": band,
            "score": score,
            "_similarity": round(cosine, 4),  # internal; not narrated
        }
        contrasts = ([f"only {name_a}: {', '.join(only_a[:3])}"] if only_a else []) + \
                    ([f"only {name_b}: {', '.join(only_b[:3])}"] if only_b else [])
        # One finding: the A<->B relationship itself (not two separate rows —
        # there is exactly one pairwise similarity here, unlike
        # neighbors/connections which rank several candidates against one anchor).
        finding = {"pair": [name_a, name_b], "shared_traits": shared,
                  "band": band, "score": score}
        return self._envelope(
            base, intent="compare",
            subject=[self._subject(info_a), self._subject(info_b)],
            findings=[finding],
            contrasts=contrasts,
            provenance={"source": "pairwise_cosine"})

    # ---- 6. bridge ---------------------------------------------------------------
    def bridge(self, anchor_a="", anchor_b="", k=5, context=None):
        """"What sits between these sounds?" — midpoint tracks with a why."""
        vec_a, info_a = self._resolve(anchor_a, context)
        vec_b, info_b = self._resolve(anchor_b, context)
        if vec_a is None or vec_b is None:
            miss = anchor_a if vec_a is None else anchor_b
            info = info_a if vec_a is None else info_b
            base = {"tool": "bridge", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": miss, "message": self._miss_message(miss, info)}
            return self._envelope(base, intent="bridge", subject=None,
                                  provenance={"source": "midpoint_knn"})
        name_a, name_b = self._display_name(info_a), self._display_name(info_b)
        tags_a, _ = self._traits(vec_a, info_a)
        tags_b, _ = self._traits(vec_b, info_b)

        na = self.similarity.index.transform_query(vec_a)
        nb = self.similarity.index.transform_query(vec_b)
        mid = (na + nb) / 2.0
        mid = mid / max(1e-8, np.linalg.norm(mid))
        rows, scope = self._nearest(mid, k, exclude=self._own_ids(info_a) | self._own_ids(info_b))

        # Relative lean with a scaled tie-band (the MERT cone is tight: between-
        # artist gaps ~0.03-0.04, within-artist std ~0.005, so an absolute
        # +/-0.02 margin mislabels at on-screen scale).
        TIE = 0.005
        tracks = []
        for r in rows:
            why = []
            qv = r.get("_vec")
            if qv is not None:
                tv = self.similarity.index.transform_query(qv)
                sa, sb = float(tv @ na), float(tv @ nb)
                if sa > sb + TIE:
                    why.append(f"leans toward {name_a}'s side of the map")
                elif sb > sa + TIE:
                    why.append(f"leans toward {name_b}'s side of the map")
                else:
                    why.append(f"balanced between {name_a} and {name_b}")
            with_a = [t for t in r["tags"] if t in tags_a]
            with_b = [t for t in r["tags"] if t in tags_b]
            if with_a:
                why.append(f"shares {', '.join(with_a[:2])} with {name_a}")
            if with_b:
                why.append(f"shares {', '.join(with_b[:2])} with {name_b}")
            if r["territory"]:
                why.append(f"from the {r['territory']} territory")
            band, score = self._band_score(r.get("cosine"))
            tracks.append({
                "artist": r["artist"],
                "song": r["song"],
                "why": "; ".join(why) if why else "sits at the midpoint of the two sounds",
                "band": band,
                "score": score,
            })
        base = {
            "tool": "bridge",
            "ok": True,
            "anchor_a": name_a,
            "anchor_b": name_b,
            "scope": scope,
            "bridge_tracks": tracks,
        }
        return self._envelope(
            base, intent="bridge",
            subject=[self._subject(info_a), self._subject(info_b)],
            scope=scope, findings=tracks,
            provenance={"source": "midpoint_knn"})

    # ---- 7. explore_cluster --------------------------------------------------------
    def explore_cluster(self, anchor="", context=None):
        """"What area should I explore next?" — the nearest adjacent territory."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            base = {"tool": "explore_cluster", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
            return self._envelope(base, intent="explore", subject=None,
                                  provenance={"source": "cluster_scan"})
        name = self._display_name(info)
        cl = self.similarity.cluster(vec)
        base_id = cl.get("cluster_id")
        base_label = cl.get("label", "")

        # Scan outward through corpus neighbours for the first different territory.
        rows = self.similarity.similar(vec, top_k=250)
        nearby, adj_label, adj_tags = [], "", []
        seen_artists = set()
        for r in rows:
            cid = self.similarity.cluster_of_index(r["index"])
            if cid is None or cid == base_id:
                continue
            if not adj_label:
                adj_label = self.similarity.cluster_label(cid) or f"territory {cid}"
                adj_tags = self.similarity.tags_for_index(r["index"], top_k=3)
            if self.similarity.cluster_label(cid) != adj_label and adj_label:
                continue
            a = r["artist"]
            if a and _norm(a) not in seen_artists:
                seen_artists.add(_norm(a))
                nearby.append(a)
            if len(nearby) >= 6:
                break
        if not adj_label:
            base = {"tool": "explore_cluster", "ok": True, "cluster": base_label,
                    "nearby_artists": [], "recommended_direction":
                    f"{name} sits deep inside the {base_label} territory — "
                    "no neighbouring territory is close enough to point at yet."}
            return self._envelope(base, intent="explore", subject=self._subject(info),
                                  scope="corpus",
                                  provenance={"source": "cluster_scan"})
        direction = (f"{name} sits in the {base_label} territory. The closest "
                     f"neighbouring territory is {adj_label}")
        if nearby:
            direction += f" — artists like {', '.join(nearby[:3])}"
        if adj_tags:
            direction += f". Expect a {', '.join(adj_tags[:3])} character"
        direction += ". That's the natural next area to explore."
        base = {
            "tool": "explore_cluster",
            "ok": True,
            "anchor": name,
            "cluster": base_label,
            "adjacent_territory": adj_label,
            "nearby_artists": nearby,
            "recommended_direction": direction,
        }
        finding = {"adjacent_territory": adj_label, "nearby_artists": nearby}
        return self._envelope(base, intent="explore", subject=self._subject(info),
                              scope="corpus", findings=[finding],
                              provenance={"source": "cluster_scan"})

    # ---- 8. explain_node -------------------------------------------------------------
    def explain_node(self, anchor="", context=None):
        """"Why is this here?" — a node's neighbours, territory, and traits."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            base = {"tool": "explain_node", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
            return self._envelope(base, intent="explain", subject=None,
                                  provenance={"source": "live_knn"})
        tags, territory = self._traits(vec, info)
        rows, scope = self._nearest(vec, 5, exclude=self._own_ids(info))
        neighbor_names = [f"{r['artist']} - {r['song']}".strip(" -") for r in rows]
        position = ""
        if neighbor_names:
            position = (f"It sits here because its closest company is "
                        f"{neighbor_names[0]}" +
                        (f" and {neighbor_names[1]}" if len(neighbor_names) > 1 else ""))
            shared_any = [t for t in tags if any(t in r["tags"] for r in rows)]
            if shared_any:
                position += f" — they share a {', '.join(shared_any[:3])} character"
            if territory:
                position += f" — inside the {territory} territory"
            position += "."
        findings = []
        for r in rows:
            band, score = self._band_score(r.get("cosine"))
            findings.append({"artist": r["artist"], "song": r["song"],
                             "band": band, "score": score})
        base = {
            "tool": "explain_node",
            "ok": True,
            "song": info.get("song", self._display_name(info)),
            "artist": info.get("artist", ""),
            "territory": territory,
            "traits": tags,
            "neighbors": neighbor_names,
            "position_summary": position,
        }
        return self._envelope(base, intent="explain", subject=self._subject(info),
                              scope=scope, findings=findings,
                              provenance={"source": "live_knn"})

    # ---- the Evidence envelope --------------------------------------------------
    # Every intent (7 existing + connections) returns the same outer shape on
    # top of its own established fields, so the narrator/agent layer can
    # handle any tool result generically:
    #   intent      short INTENTS key ("neighbors", "connections", ...)
    #   subject     {label, kind, id, on_map} for a one-anchor tool, or
    #               [subject_a, subject_b] for a two-anchor tool (compare/bridge);
    #               None when the anchor didn't resolve.
    #   scope       "map" | "library" | "persisted" | "corpus" | None
    #   findings    normalized list of what was found; similarity-scored
    #               entries (connections/neighbors/compare/bridge/explain_node)
    #               each carry "band" (one of SIMILARITY_BANDS) + "score" (0-100)
    #   contrasts   list of distinguishing facts, [] when none apply
    #   provenance  {"source": ...} — e.g. persisted_links vs live_knn vs corpus
    # All of a method's own existing keys (ok, message, error, neighbors, ...)
    # are preserved untouched — this is purely additive.
    def _subject(self, info):
        if not info or info.get("resolved") is False:
            return None
        return {
            "label": self._display_name(info),
            "kind": self._public_type(info),
            "id": info.get("id"),
            "on_map": self._public_type(info) == "graph_node",
        }

    def _band_score(self, cosine):
        """(band, score) for a raw cosine, via whatever calibration the
        similarity service loaded (mirrors persisted map-edge scoring — see
        anther_ml.calibration / ui/atlas.py). (None, None) if the backend
        doesn't expose calibration or cosine is None — callers must accept
        that rather than assume it's always populated."""
        if cosine is None:
            return None, None
        band_fn = getattr(self.similarity, "band_for_cosine", None)
        score_fn = getattr(self.similarity, "display_score", None)
        if band_fn is None or score_fn is None:
            return None, None
        try:
            return band_fn(cosine), round(float(score_fn(cosine)), 1)
        except Exception:
            return None, None

    @staticmethod
    def _envelope(result, *, intent, subject=None, scope=None, findings=None,
                  contrasts=None, provenance=None):
        """Layer the shared Evidence envelope on top of a tool's own result
        dict (additive — every existing key the method already set is kept
        untouched). ``scope`` defaults to whatever the method already put in
        result["scope"] (neighbors/bridge already carry one); pass it
        explicitly for tools that don't set that key themselves."""
        result = dict(result)
        result["intent"] = intent
        result["subject"] = subject
        result["scope"] = scope if scope is not None else result.get("scope")
        result["findings"] = findings if findings is not None else []
        result["contrasts"] = contrasts if contrasts is not None else []
        result["provenance"] = provenance or {}
        return result

    # ---- anchor resolution (selected -> visible -> corpus) -----------------------
    def _selected_node(self, context):
        gc = self.graph_ctx
        nid = getattr(context, "selected_node_id", None) if context is not None else None
        if gc is None or not nid:
            return None
        return gc.node(nid)

    def _resolve(self, anchor, context):
        """(vec, info). info carries display fields; on failure info["error"]."""
        spec = str(anchor or "").strip()
        if _norm(spec) in SELF_REFS:
            node = self._selected_node(context)
            if node is not None:
                vec = self.graph_ctx.vec_for_id(node["id"])
                if vec is not None:
                    return vec, self._node_info(node, ref=spec or "selected node")
            # fall back to the last anchor this conversation resolved
            last = getattr(context, "last_anchor", None) if context is not None else None
            if last and _norm(last) not in SELF_REFS:
                vec, info = self._resolve_concrete(last)
                if vec is not None:
                    info["ref"] = spec or "me"
                    return vec, info
            return None, {"input": spec, "resolved": False, "error": "no_selection"}
        return self._resolve_concrete(spec)

    def _resolve_concrete(self, spec):
        vec, info = self.similarity.resolve(spec, graph_ctx=self.graph_ctx)
        if vec is None:
            info = dict(info or {})
            info.setdefault("error", "anchor_not_resolved")
            return None, info
        # normalize display fields across onscreen/corpus/audio hits
        if info.get("type", "").startswith("onscreen") and info.get("id"):
            node = self.graph_ctx.node(info["id"]) if self.graph_ctx else None
            if node is not None:
                return vec, self._node_info(node, ref=spec)
        out = dict(info)
        match = str(info.get("match", spec))
        if info.get("type") == "artist":
            out["artist"] = match
            out["song"] = ""
        elif " - " in match:
            artist, song = match.split(" - ", 1)
            out["artist"], out["song"] = artist.strip(), song.strip()
        else:
            out["song"] = match
        return vec, out

    def _node_info(self, node, ref=""):
        return {
            "input": ref,
            "resolved": True,
            "type": "onscreen_track",
            "id": node.get("id"),
            "artist": node.get("artist", ""),
            "song": node.get("name", ""),
            "match": f"{node.get('artist','')} - {node.get('name','')}".strip(" -"),
        }

    def _display_name(self, info):
        artist, song = info.get("artist", ""), info.get("song", "")
        if artist and song:
            return f"{artist} - {song}"
        return artist or song or str(info.get("match", info.get("input", "")))

    @staticmethod
    def _public_type(info):
        t = info.get("type", "")
        if t.startswith("onscreen"):
            return "graph_node"
        if t == "audio":
            return "upload"
        if t == "artist":
            return "artist"
        return "library_track"

    def _own_ids(self, info):
        """The anchor's own node ids, so it isn't returned as its own neighbour."""
        gc = self.graph_ctx
        if gc is None:
            return set()
        if info.get("id"):
            return {info["id"]}
        if info.get("type") == "onscreen_artist" and info.get("match"):
            return set(gc.artist_ids(info["match"]))
        return set()

    def _miss_message(self, anchor, info):
        if info.get("error") == "no_selection":
            return ("Nothing is selected on your map and I don't have a track of "
                    "yours from earlier. Click a node, or name an artist or song.")
        return (f"“{anchor}” isn't on your map or in the library I know. "
                "Add it to the map (search it or upload audio) and ask again.")

    # ---- traits + neighbour pools -------------------------------------------------
    @staticmethod
    def _node_tag_names(node, top_k=4):
        tags = sorted(node.get("tags", []), key=lambda t: -float(t.get("score", 0.0)))
        return [t["genre"] for t in tags[:top_k] if t.get("genre")]

    def _traits(self, vec, info):
        """(tag_names, territory_label) for an anchor — the node's own probe
        tags/cluster when it's on screen, corpus-inferred otherwise."""
        if info.get("id") and self.graph_ctx is not None:
            node = self.graph_ctx.node(info["id"])
            if node is not None:
                tags = self._node_tag_names(node)
                territory = node.get("cluster_label", "")
                if tags or territory:
                    return tags, territory
        tags = [t["genre"] for t in self.similarity.tags(vec, top_k=4, knn=12)]
        territory = self.similarity.cluster(vec).get("label", "")
        return tags, territory

    def _nearest(self, vec, k, exclude=None):
        """Nearest songs to ``vec``: on-screen first, corpus fallback.

        Returns (rows, scope) where each row has artist/song/tags/territory/
        rank and keeps the raw vector privately for lean scoring.
        """
        gc = self.graph_ctx
        if gc is not None and gc.has_nodes():
            ids, mat, nodes = gc.candidate_vectors(exclude_ids=exclude or set())
            if mat is not None and len(ids):
                q = self.similarity.index.transform_query(vec)
                cand = np.stack([self.similarity.index.transform_query(m) for m in mat])
                sims = cand @ q
                order = np.argsort(sims)[::-1][: max(k, 1)]
                rows = []
                for rank, i in enumerate(order, start=1):
                    node = nodes[int(i)]
                    rows.append({
                        "artist": node.get("artist", ""),
                        "song": node.get("name", ""),
                        "tags": self._node_tag_names(node),
                        "territory": node.get("cluster_label", ""),
                        "rank": rank,
                        "cosine": float(sims[int(i)]),
                        "_vec": mat[int(i)],
                    })
                if rows:
                    return rows, "map"
        rows = []
        for rank, r in enumerate(self.similarity.similar(vec, top_k=k), start=1):
            cid = self.similarity.cluster_of_index(r["index"])
            rows.append({
                "artist": r["artist"],
                "song": r["name"],
                "tags": self.similarity.tags_for_index(r["index"]),
                "territory": self.similarity.cluster_label(cid),
                "rank": rank,
                "cosine": r.get("score"),
                "_vec": self.similarity.vector_for_index(r["index"]),
            })
        return rows, "library"

    # ---- deterministic relationship phrasing ----------------------------------------
    @staticmethod
    def _relationship(anchor_tags, other_tags, anchor_territory, other_territory, rank):
        """A SHORT reason a neighbour is close — just the shared traits, if any.
        We deliberately do NOT tack on the (often long, 3-part) cluster label per
        connection; the caller mentions territory once at most. Empty string is
        fine: the connection itself is the answer."""
        shared = [t for t in anchor_tags if t in other_tags]
        if shared:
            return "shares " + ", ".join(shared[:2])
        return ""
