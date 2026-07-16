"""The mentor's LLM-facing graph tools — music concepts only.

Seven read-only tools over the user's on-screen map (first) and the frozen
Anther corpus (fallback). Each tool retrieves facts deterministically and
phrases *why* two things are related from evidence it can prove: shared
micro-genre tags, cluster (territory) labels, and lean/balance scores.
Raw ML detail (embeddings, cosines, indices) stays internal — the agent and
the LLM behind it only ever see artists, songs, territories, and traits.

    inspect_graph(context)                what is on the map right now
    resolve_anchor(anchor, context)       who/what does a reference mean
    neighbors(anchor, k, context)         what does this sound like
    compare(anchor_a, anchor_b, context)  how do these two relate
    bridge(anchor_a, anchor_b, k, ...)    what sits between these sounds
    explore_cluster(anchor, context)      where to explore next
    explain_node(anchor, context)         why is this node here

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
    "neighbors": ("neighbors", ("anchor", "k")),
    "compare": ("compare", ("anchor_a", "anchor_b")),
    "bridge": ("bridge", ("anchor_a", "anchor_b", "k")),
    "explore": ("explore_cluster", ("anchor",)),
    "explain": ("explain_node", ("anchor",)),
}


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
            return {"tool": "inspect_graph", "ok": True, "visible_nodes": 0,
                    "selected_node": None, "territories": [], "artists": [],
                    "groups": [], "recent_additions": [],
                    "message": "The map is empty — nothing has been placed yet."}
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
        return {
            "tool": "inspect_graph",
            "ok": True,
            "selected_node": self._selected_summary(context),
            "visible_nodes": len(nodes),
            "territories": sorted(
                ({"territory": t, "songs": c} for t, c in territories.items()),
                key=lambda d: -d["songs"]),
            "artists": artists[:25],
            "groups": groups,
            "recent_additions": recent,
        }

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
            return {"tool": "resolve_anchor", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
        return {
            "tool": "resolve_anchor",
            "ok": True,
            "resolved": True,
            "type": self._public_type(info),
            "name": self._display_name(info),
            "artist": info.get("artist", ""),
            "id": info.get("id"),
        }

    # ---- 3. neighbors ---------------------------------------------------------
    def neighbors(self, anchor="", k=6, context=None):
        """"What does this sound like?" — closest songs, with reasons."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            return {"tool": "neighbors", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
        anchor_tags, anchor_territory = self._traits(vec, info)
        rows, scope = self._nearest(vec, k, exclude=self._own_ids(info))
        out = []
        for r in rows:
            out.append({
                "artist": r["artist"],
                "song": r["song"],
                "relationship": self._relationship(
                    anchor_tags, r["tags"], anchor_territory, r["territory"], r["rank"]),
            })
        return {
            "tool": "neighbors",
            "ok": True,
            "anchor": self._display_name(info),
            "scope": scope,
            "territory": anchor_territory,
            "traits": anchor_tags,
            "neighbors": out,
        }

    # ---- 4. compare -------------------------------------------------------------
    def compare(self, anchor_a="", anchor_b="", context=None):
        """How two sounds relate: shared traits, differences, what links them."""
        vec_a, info_a = self._resolve(anchor_a, context)
        vec_b, info_b = self._resolve(anchor_b, context)
        if vec_a is None or vec_b is None:
            miss = anchor_a if vec_a is None else anchor_b
            info = info_a if vec_a is None else info_b
            return {"tool": "compare", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": miss, "message": self._miss_message(miss, info)}
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
        return {
            "tool": "compare",
            "ok": True,
            "anchor_a": name_a,
            "anchor_b": name_b,
            "similarity_summary": summary,
            "shared_traits": shared,
            "differences": {"only_" + name_a: only_a, "only_" + name_b: only_b},
            "bridge_candidates": [f"{r['artist']} - {r['song']}".strip(" -")
                                  for r in bridge_rows],
            "_similarity": round(cosine, 4),  # internal; not narrated
        }

    # ---- 5. bridge ---------------------------------------------------------------
    def bridge(self, anchor_a="", anchor_b="", k=5, context=None):
        """"What sits between these sounds?" — midpoint tracks with a why."""
        vec_a, info_a = self._resolve(anchor_a, context)
        vec_b, info_b = self._resolve(anchor_b, context)
        if vec_a is None or vec_b is None:
            miss = anchor_a if vec_a is None else anchor_b
            info = info_a if vec_a is None else info_b
            return {"tool": "bridge", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": miss, "message": self._miss_message(miss, info)}
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
            tracks.append({
                "artist": r["artist"],
                "song": r["song"],
                "why": "; ".join(why) if why else "sits at the midpoint of the two sounds",
            })
        return {
            "tool": "bridge",
            "ok": True,
            "anchor_a": name_a,
            "anchor_b": name_b,
            "scope": scope,
            "bridge_tracks": tracks,
        }

    # ---- 6. explore_cluster --------------------------------------------------------
    def explore_cluster(self, anchor="", context=None):
        """"What area should I explore next?" — the nearest adjacent territory."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            return {"tool": "explore_cluster", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
        name = self._display_name(info)
        base = self.similarity.cluster(vec)
        base_id = base.get("cluster_id")
        base_label = base.get("label", "")

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
            return {"tool": "explore_cluster", "ok": True, "cluster": base_label,
                    "nearby_artists": [], "recommended_direction":
                    f"{name} sits deep inside the {base_label} territory — "
                    "no neighbouring territory is close enough to point at yet."}
        direction = (f"{name} sits in the {base_label} territory. The closest "
                     f"neighbouring territory is {adj_label}")
        if nearby:
            direction += f" — artists like {', '.join(nearby[:3])}"
        if adj_tags:
            direction += f". Expect a {', '.join(adj_tags[:3])} character"
        direction += ". That's the natural next area to explore."
        return {
            "tool": "explore_cluster",
            "ok": True,
            "anchor": name,
            "cluster": base_label,
            "adjacent_territory": adj_label,
            "nearby_artists": nearby,
            "recommended_direction": direction,
        }

    # ---- 7. explain_node -------------------------------------------------------------
    def explain_node(self, anchor="", context=None):
        """"Why is this here?" — a node's neighbours, territory, and traits."""
        vec, info = self._resolve(anchor, context)
        if vec is None:
            return {"tool": "explain_node", "ok": False,
                    "error": info.get("error", "anchor_not_resolved"),
                    "anchor": anchor, "message": self._miss_message(anchor, info)}
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
        return {
            "tool": "explain_node",
            "ok": True,
            "song": info.get("song", self._display_name(info)),
            "artist": info.get("artist", ""),
            "territory": territory,
            "traits": tags,
            "neighbors": neighbor_names,
            "position_summary": position,
        }

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
