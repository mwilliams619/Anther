"""MentorAgent — the graph-question pipeline.

    question -> LLM intent classifier -> MentorGraphTools -> evidence -> LLM narrator

The LLM does exactly two jobs, at the two ends of the pipeline:
  1. INTERPRET: map the user's words to one intent + args (strict JSON, one
     call, validated; a retry then an honest "help me out" if it can't).
  2. NARRATE: rephrase the deterministic evidence in mentor voice. The output
     is discarded if it names an artist/song/territory the tools didn't return.

Everything in between is deterministic tool logic (graph_tools.py). There is
no ReAct loop and no regex question-shape classifier — the old deterministic
classifier chose tools from surface patterns and was a standing source of
misroutes; intent now comes from the model, grounding from the tools.
"""

from __future__ import annotations

import json

try:
    from .graph_tools import INTENTS, MentorGraphTools, SELF_REFS
except ImportError:  # pragma: no cover - script-style import path
    from graph_tools import INTENTS, MentorGraphTools, SELF_REFS


CLASSIFY_SYSTEM = (
    "You route questions for a music mentor who sits inside an interactive "
    "sound map. The user sees placed songs (nodes); they may have one selected. "
    "Return exactly one JSON object: {\"intent\": \"<name>\", \"args\": {...}} — "
    "no prose, no markdown.\n"
    "Intents:\n"
    "- inspect {}: what is on the map / what am I looking at\n"
    "- neighbors {anchor}: what does X sound like, who is X close to\n"
    "- compare {anchor_a, anchor_b}: how do two sounds relate or differ\n"
    "- bridge {anchor_a, anchor_b}: what sits between two sounds\n"
    "- explore {anchor}: where should X go / what area to explore next\n"
    "- explain {anchor}: why is this node here / why is X placed there\n"
    "- followup {}: a short reference to the previous answer (\"which ones?\", "
    "\"tell me more\", \"list them\")\n"
    "Anchors are copied from the user's words. Use \"me\" when they mean "
    "themselves, their track, or the selected node (\"this song\", \"this\")."
)

CLASSIFY_FEWSHOT = [
    {"role": "user", "content": "What do I sound like?"},
    {"role": "assistant", "content": '{"intent":"neighbors","args":{"anchor":"me"}}'},
    {"role": "user", "content": "what sits between Burial and Aphex Twin?"},
    {"role": "assistant", "content": '{"intent":"bridge","args":{"anchor_a":"Burial","anchor_b":"Aphex Twin"}}'},
    {"role": "user", "content": "why is this node here?"},
    {"role": "assistant", "content": '{"intent":"explain","args":{"anchor":"me"}}'},
    {"role": "user", "content": "how does Halo compare to my track?"},
    {"role": "assistant", "content": '{"intent":"compare","args":{"anchor_a":"Halo","anchor_b":"me"}}'},
    {"role": "user", "content": "what's on my map right now?"},
    {"role": "assistant", "content": '{"intent":"inspect","args":{}}'},
    {"role": "user", "content": "which ones?"},
    {"role": "assistant", "content": '{"intent":"followup","args":{}}'},
    {"role": "user", "content": "what area should I explore next?"},
    {"role": "assistant", "content": '{"intent":"explore","args":{"anchor":"me"}}'},
]

NARRATE_SYSTEM = (
    "You are a music mentor talking to an artist about their sound map. "
    "Answer using ONLY the FACTS below — every artist, song, and territory you "
    "name must already appear in FACTS. Never invent lyrics, moods, meanings, "
    "themes, or genres, and never describe a song you were not given facts "
    "about. Be brief and direct: 1-3 sentences. When you list what something "
    "connects to, just name the tracks — do NOT explain the genre or cluster of "
    "each one unless the artist explicitly asked why. Do not mention tools, "
    "scores, or how any of this was computed."
)

# High-precision references to "the node I have selected right now". A question
# built from these must name the pinned node or honestly say nothing is
# selected — it must NEVER be answered by the LLM inventing a song.
SELECTION_PHRASES = (
    "selected node", "selected song", "selected track", "what is selected",
    "what's selected", "whats selected", "currently selected", "the selection",
    "am i selecting", "am i clicking", "i selected", "i clicked", "i've selected",
    "i've clicked", "node i selected", "node i clicked", "song i selected",
    "song i clicked", "this selected", "highlighted node",
)

VALID_INTENTS = set(INTENTS) | {"followup"}


def _extract_first_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = text[start: i + 1]
                try:
                    return json.loads(raw)
                except Exception:
                    return None
    return None


class MentorAgent:
    def __init__(self, tools, generate_fn):
        self.tools = tools
        self.generate_fn = generate_fn

    @classmethod
    def build(cls, similarity, generate_fn, graph_ctx=None):
        return cls(MentorGraphTools(similarity, graph_ctx=graph_ctx), generate_fn)

    # ---- 1. interpret ------------------------------------------------------
    def classify(self, question, context=None):
        """One LLM call -> {"intent", "args"} or None if unusable after retry."""
        if self.generate_fn is None:
            return None
        hint = self._context_hint(context)
        msgs = [{"role": "system", "content": CLASSIFY_SYSTEM}]
        msgs += CLASSIFY_FEWSHOT
        msgs += [{"role": "user", "content": hint + question}]
        for attempt in range(2):
            try:
                raw = self.generate_fn(msgs, max_new_tokens=80, temperature=0.0)
            except Exception:
                return None
            cand = _extract_first_json(raw or "")
            if self._valid(cand):
                cand.setdefault("args", {})
                return cand
            if attempt == 0:
                msgs = msgs + [
                    {"role": "assistant", "content": raw or ""},
                    {"role": "user", "content":
                        "That was not valid. Reply with ONLY the JSON object "
                        "{\"intent\": ..., \"args\": {...}} and nothing else."},
                ]
        return None

    @staticmethod
    def _valid(obj):
        return (isinstance(obj, dict)
                and obj.get("intent") in VALID_INTENTS
                and isinstance(obj.get("args", {}), dict))

    def _context_hint(self, context):
        """Ground the classifier in what the user is looking at."""
        lines = []
        node = self.tools._selected_node(context)
        if node is not None:
            label = f"{node.get('artist','')} - {node.get('name','')}".strip(" -")
            lines.append(f"[Selected node: {label}]")
        else:
            lines.append("[Selected node: none]")
        if context is not None and getattr(context, "last_intent", None):
            lines.append(f"[Previous turn intent: {context.last_intent}]")
        return "\n".join(lines) + "\n" if lines else ""

    @staticmethod
    def _is_selection_question(question):
        q = " ".join(str(question or "").lower().split())
        return any(p in q for p in SELECTION_PHRASES)

    def _answer_selection(self, question, context):
        """Deterministic, honesty-critical: a question about the current
        selection names the pinned node or plainly says nothing is selected —
        the LLM is never given the chance to invent a song here."""
        node = self.tools._selected_node(context)
        if node is None:
            return {
                "intent": "explain", "observation": None, "status": "no_selection",
                "answer": ("No node is selected on your map right now. Click a song "
                           "on the map and ask again, or name the track you mean."),
            }
        obs = self.tools.explain_node("me", context=context)
        self._remember(context, "explain", {"anchor": "me"}, obs)
        if not obs.get("ok"):
            return {"intent": "explain", "observation": obs,
                    "status": obs.get("error", "failed"),
                    "answer": obs.get("message", "I couldn't read that node.")}
        return {"intent": "explain", "observation": obs, "status": "ok",
                "answer": self._compose(question, obs)}

    # ---- the pipeline --------------------------------------------------------
    def run(self, question, context=None):
        # Selection questions are answered deterministically (see above) so they
        # can't be misrouted to a map census or hallucinated by the model.
        if self._is_selection_question(question):
            return self._answer_selection(question, context)

        call = self.classify(question, context)
        if call is None:
            return {
                "intent": None, "observation": None, "status": "unclassified",
                "answer": ("I couldn't pin down what you're asking about the map. "
                           "Name an artist or song, or click a node and ask again."),
            }
        intent, args = call["intent"], call.get("args", {})

        # follow-up: re-narrate the previous observation for the new question
        if intent == "followup":
            prev = getattr(context, "last_observation", None) if context is not None else None
            if not prev:
                return {"intent": intent, "observation": None, "status": "no_prior",
                        "answer": "We haven't mapped anything yet this session — "
                                  "ask about an artist, a song, or your own sound first."}
            answer = self._compose(question, prev)
            return {"intent": intent, "observation": prev, "status": "ok", "answer": answer}

        observation = self.tools.execute(intent, args, context=context)
        self._remember(context, intent, args, observation)

        if not observation.get("ok"):
            return {"intent": intent, "observation": observation,
                    "status": observation.get("error", "failed"),
                    "answer": observation.get("message",
                              "I couldn't read that from the map.")}
        answer = self._compose(question, observation)
        return {"intent": intent, "observation": observation, "status": "ok",
                "answer": answer}

    def _remember(self, context, intent, args, observation):
        if context is None:
            return
        context.last_intent = intent
        if observation.get("ok"):
            context.last_observation = observation
            # Remember a concrete anchor so "me"/follow-ups survive deselection.
            anchor = observation.get("anchor") or observation.get("song")
            if not anchor:
                for key in ("anchor", "anchor_a", "anchor_b"):
                    v = str(args.get(key, "")).strip()
                    if v and v.lower() not in SELF_REFS:
                        anchor = v
                        break
            if anchor:
                context.last_anchor = str(anchor)

    # ---- 2. narrate -----------------------------------------------------------
    def _compose(self, question, observation):
        """Narrate the evidence. Names come only from tool output; the voiced
        answer is discarded if it introduces a name that wasn't observed."""
        facts, allowed = self._evidence(observation)
        if not facts:
            return "I mapped that, but there was nothing on the map to report."
        if self.generate_fn is None:
            return facts
        prompt = [
            {"role": "system", "content": NARRATE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nFACTS:\n{facts}"},
        ]
        try:
            voiced = (self.generate_fn(prompt, max_new_tokens=160, temperature=0.3) or "").strip()
        except Exception:
            voiced = ""
        if voiced and self._names_ok(voiced, allowed):
            return voiced
        return facts

    # ---- evidence formatting (music concepts, no ML detail) --------------------
    def _evidence(self, o):
        """(facts_text, allowed_names) — the ground truth the narrator may use."""
        lines, allowed = [], set()

        def name(*vals):
            for v in vals:
                if v:
                    allowed.add(str(v).lower())

        tool = o.get("tool")
        if tool == "neighbors":
            name(o.get("anchor"))
            # Just the connections — one short shared-trait note inline, no
            # per-track genre lecture (the mentor explains clusters only on ask).
            lines.append(f"{o.get('anchor')} is closest to:")
            for n in o.get("neighbors", []):
                name(n.get("artist"), n.get("song"))
                label = " - ".join(x for x in (n.get("artist"), n.get("song")) if x)
                rel = n.get("relationship")
                if rel:
                    name(*rel.replace(";", ",").split(","))
                    lines.append(f"- {label} ({rel})")
                else:
                    lines.append(f"- {label}")
        elif tool == "compare":
            name(o.get("anchor_a"), o.get("anchor_b"))
            lines.append(o.get("similarity_summary", ""))
            if o.get("shared_traits"):
                name(*o["shared_traits"])
                lines.append("Shared traits: " + ", ".join(o["shared_traits"]))
            for side, tags in (o.get("differences") or {}).items():
                if tags:
                    name(*tags)
                    lines.append(f"Distinct ({side.replace('only_', '')}): " + ", ".join(tags))
            if o.get("bridge_candidates"):
                name(*o["bridge_candidates"])
                lines.append("Between them: " + "; ".join(o["bridge_candidates"]))
        elif tool == "bridge":
            name(o.get("anchor_a"), o.get("anchor_b"))
            lines.append(f"Tracks between {o.get('anchor_a')} and {o.get('anchor_b')}:")
            for t in o.get("bridge_tracks", []):
                name(t.get("artist"), t.get("song"))
                label = " - ".join(x for x in (t.get("artist"), t.get("song")) if x)
                lines.append(f"- {label}")
                if t.get("why"):
                    lines.append(f"  Why: {t['why']}")
                    name(*t["why"].replace(";", ",").split(","))
        elif tool == "explore_cluster":
            name(o.get("anchor"), o.get("cluster"), o.get("adjacent_territory"))
            name(*o.get("nearby_artists", []))
            lines.append(o.get("recommended_direction", ""))
            if o.get("nearby_artists"):
                lines.append("Artists there: " + ", ".join(o["nearby_artists"]))
        elif tool == "explain_node":
            name(o.get("artist"), o.get("song"), o.get("territory"))
            name(*o.get("traits", []))
            name(*o.get("neighbors", []))
            label = " - ".join(x for x in (o.get("artist"), o.get("song")) if x)
            lines.append(f"Node: {label}")
            if o.get("position_summary"):
                lines.append(o["position_summary"])
            if o.get("neighbors"):
                lines.append("Closest company: " + "; ".join(o["neighbors"]))
            if o.get("traits"):
                lines.append("Traits: " + ", ".join(o["traits"]))
            if o.get("territory"):
                lines.append(f"Territory: {o['territory']}")
        elif tool == "inspect_graph":
            if not o.get("visible_nodes"):
                lines.append("The map is empty — nothing placed yet.")
            else:
                lines.append(f"The map has {o['visible_nodes']} placed songs.")
                sel = o.get("selected_node")
                if sel:
                    name(sel.get("artist"), sel.get("name"), sel.get("territory"))
                    name(*(sel.get("traits") or []))
                    label = " - ".join(x for x in (sel.get("artist"), sel.get("name")) if x)
                    lines.append(f"Selected: {label}" +
                                 (f" ({sel['territory']})" if sel.get("territory") else ""))
                terr = o.get("territories", [])
                if terr:
                    for t in terr[:8]:
                        name(t.get("territory"))
                    lines.append("Territories: " + "; ".join(
                        f"{t['territory']} ({t['songs']})" for t in terr[:8]))
                if o.get("artists"):
                    name(*o["artists"])
                    lines.append("Artists: " + ", ".join(o["artists"][:12]))
                if o.get("groups"):
                    name(*o["groups"])
                    lines.append("Collections: " + ", ".join(str(g) for g in o["groups"]))
                if o.get("recent_additions"):
                    name(*o["recent_additions"])
                    lines.append("Recently added: " + "; ".join(o["recent_additions"]))
        elif tool == "resolve_anchor":
            name(o.get("name"), o.get("artist"))
            where = "on your map" if o.get("type") == "graph_node" else "in the library"
            lines.append(f"{o.get('name')} — found {where}.")
        return "\n".join(l for l in lines if l), allowed

    @staticmethod
    def _names_ok(text, allowed):
        """True unless the voiced text introduces a MULTI-word Proper-Case name
        absent from the allowed set. Multi-word Title Case is where hallucinated
        artist/track names appear; single capitalized words are almost always
        sentence starts, so we don't flag them (avoids false rejections)."""
        import re
        if not allowed:
            return True
        allowed = {a.strip().lower() for a in allowed if str(a).strip()}
        cands = re.findall(r"\b([A-Z][a-zA-Z0-9&']+(?:\s+[A-Z][a-zA-Z0-9&']+){1,3})\b", text)
        for c in cands:
            cl = c.lower()
            if any(cl in a or a in cl for a in allowed):
                continue
            toks = set(cl.split())
            if any(toks & set(a.split()) for a in allowed):
                continue
            return False
        return True
