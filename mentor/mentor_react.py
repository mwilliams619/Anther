"""Reasoning-mode ReAct loop for graph questions.

The loop follows the fable-method discipline: a mid-tier model that follows a
structured loop (classify -> gather evidence -> surprise-route -> verify by
observation -> report) beats a stronger model that free-styles. Concretely:

  1. The question SHAPE is classified deterministically and seeds the first tool
     call, so the loop is correct even when the LLM emits nothing usable (the
     old loop fell to a single-guess regex on any bad parse).
  2. A surprise (``ok:False``) ROUTES the loop instead of terminating it: we try
     a narrower/broader tool, and only stop when an anchor is genuinely absent.
  3. The answer is COMPOSED deterministically from the observations — names come
     only from tool results, so a hallucinated artist can't appear. The LLM is
     used only to phrase that readout in voice, and its output is discarded if it
     introduces a name that wasn't observed.
  4. Deflection is HONEST: "Rae Sremmurd isn't on your map" (absent) is distinct
     from "which Halo did you mean?" (ambiguous), not one generic line.
"""

from __future__ import annotations

import json
import re

try:
    from .mentor_graph import MentorGraphTools
except ImportError:  # pragma: no cover - script-style import path
    from mentor_graph import MentorGraphTools


REACT_SYSTEM = (
    "You are a music mentor with read-only graph tools over a sound-similarity map. "
    "For each step, output exactly one JSON object: {\"tool\":\"<name>\",\"args\":{...}} "
    "or {\"tool\":\"final\",\"answer\":\"...\"}. "
    "Tools: resolve(name); sounds_like(anchor, top_k); compare(anchor_a, anchor_b, top_k); "
    "bridge(anchor_a, anchor_b, k, within); crossover(anchor, k); artist_tracks(anchor, artist, top_k); "
    "tagmates(anchor, k); micro_genres(anchor); cluster_summary(cluster_id); coherence(scope, target). "
    "Anchors resolve against what's on the user's screen first, then the frozen library. "
    "These tools never place, remove, or modify anything. Keep calls minimal and grounded; "
    "answer only with artists and tracks the tools actually returned."
)

REACT_FEWSHOT = [
    {
        "role": "user",
        "content": "what artists sit between me and Neroptik?",
    },
    {
        "role": "assistant",
        "content": '{"tool":"bridge","args":{"anchor_a":"me","anchor_b":"Neroptik","k":6}}',
    },
    {
        "role": "user",
        "content": 'OBSERVATION: {"tool":"bridge","ok":true,"neighbors":[{"artist":"A","name":"B"}]}'
    },
    {
        "role": "assistant",
        "content": '{"tool":"final","answer":"You sit between artists like A - B. Lean into that hybrid lane."}',
    },
    {
        "role": "user",
        "content": "what do I sound like near Noisia?",
    },
    {
        "role": "assistant",
        "content": '{"tool":"sounds_like","args":{"anchor":"Noisia","top_k":8}}',
    },
    {
        "role": "user",
        "content": "what xxxtentacion songs are the closest to Halo by Beyonce?",
    },
    {
        "role": "assistant",
        "content": '{"tool":"artist_tracks","args":{"artist":"xxxtentacion","anchor":"Halo by Beyonce","top_k":6}}',
    },
]


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
                raw = text[start : i + 1]
                try:
                    return json.loads(raw)
                except Exception:
                    return None
    return None


def _pick_anchor(question):
    q = question.strip()
    quoted = re.findall(r'"([^"]+)"', q)
    if quoted:
        return quoted[0]
    proper = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b", q)
    if proper:
        return proper[-1]
    tail = q.split("?")[0].strip()
    return tail if tail else ""


def _fallback_call(question):
    q = question.lower()
    m = re.search(r"what\s+(.+?)\s+songs?\s+are\s+the\s+closest\s+to\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if m:
        return {
            "tool": "artist_tracks",
            "args": {
                "artist": m.group(1).strip(),
                "anchor": m.group(2).strip(),
                "top_k": 6,
            },
        }
    m = re.search(r"compare\s+(.+?)\s+to\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if m:
        return {"tool": "compare", "args": {"anchor_a": m.group(1).strip(), "anchor_b": m.group(2).strip(), "top_k": 6}}
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if m:
        return {"tool": "bridge", "args": {"anchor_a": m.group(1).strip(), "anchor_b": m.group(2).strip(), "k": 6}}
    if "what do i sound like" in q or "sounds like" in q or "closest" in q and "songs" in q:
        return {"tool": "sounds_like", "args": {"anchor": _pick_anchor(question), "top_k": 8}}
    if "crossover" in q or "cross into" in q or "adjacent" in q:
        return {"tool": "crossover", "args": {"anchor": _pick_anchor(question), "k": 6}}
    if "tag" in q:
        return {"tool": "tagmates", "args": {"anchor": _pick_anchor(question), "k": 8}}
    if "sound like" in q or "sounds like" in q or "close to" in q or "similar" in q:
        return {"tool": "sounds_like", "args": {"anchor": _pick_anchor(question), "top_k": 8}}
    return {"tool": "resolve", "args": {"name": _pick_anchor(question)}}


def _validate_call(obj):
    if not isinstance(obj, dict):
        return False
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool:
        return False
    if tool == "final":
        return isinstance(obj.get("answer"), str)
    args = obj.get("args")
    return isinstance(args, dict)


# ── Deterministic question-shape classification ──────────────────────────────
# Each entry: (compiled regex, builder(match, question) -> {shape, tool, args}).
# Order matters — more specific shapes first. The classifier is the loop's seed;
# it must return a runnable call for every graph question so the loop never
# depends on the LLM emitting good JSON.

_ME = {"me", "my", "my track", "my sound", "my song", "my demo", "mine", "this"}


def _clean_anchor(s):
    s = (s or "").strip().strip("\"'").strip()
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(cluster|clusters|region|territory)$", "", s, flags=re.IGNORECASE)
    return s.strip()


def classify_question(question):
    """Map a graph question to a seed tool call. Returns {shape, tool, args}."""
    q = (question or "").strip()
    ql = q.lower()

    # three-anchor "within": what song from X bridges/links A and B
    m = re.search(
        r"what\s+(?:song|track|songs|tracks)?\s*(?:from|by|of)\s+(.+?)\s+"
        r"(?:bridges?|links?|connects?|sits?\s+between|between)\s+"
        r"(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?:\s+clusters?)?[\?\.]?$",
        q, re.IGNORECASE)
    if m:
        return {"shape": "within_bridge", "tool": "bridge",
                "args": {"anchor_a": _clean_anchor(m.group(2)),
                         "anchor_b": _clean_anchor(m.group(3)),
                         "within": _clean_anchor(m.group(1)), "k": 6}}

    # which of X's songs is closest to Y  /  what X songs are closest to Y
    m = re.search(r"which\s+(?:of\s+)?(.+?)(?:['\u2019]s|s['\u2019])?\s+(?:song|track)s?\s+"
                  r"(?:is|are)\s+(?:the\s+)?closest\s+(?:match\s+)?to\s+(.+?)[\?\.]?$",
                  q, re.IGNORECASE)
    if not m:
        m = re.search(r"what\s+(.+?)\s+songs?\s+are\s+(?:the\s+)?closest\s+to\s+(.+?)[\?\.]?$",
                      q, re.IGNORECASE)
    if m:
        return {"shape": "artist_tracks", "tool": "artist_tracks",
                "args": {"artist": _clean_anchor(m.group(1)),
                         "anchor": _clean_anchor(m.group(2)), "top_k": 6}}

    # compare A to/and/with B
    m = re.search(r"compare\s+(.+?)\s+(?:to|and|with|against)\s+(.+?)[\?\.]?$", q, re.IGNORECASE)
    if not m:
        m = re.search(r"how\s+(?:similar|close|different)\s+(?:is|are)\s+(.+?)\s+"
                      r"(?:to|and|from)\s+(.+?)[\?\.]?$", q, re.IGNORECASE)
    if m:
        return {"shape": "compare", "tool": "compare",
                "args": {"anchor_a": _clean_anchor(m.group(1)),
                         "anchor_b": _clean_anchor(m.group(2)), "top_k": 6}}

    # artists/songs (that sit) between A and B
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\?\.]?$", q, re.IGNORECASE)
    if m:
        return {"shape": "bridge", "tool": "bridge",
                "args": {"anchor_a": _clean_anchor(m.group(1)),
                         "anchor_b": _clean_anchor(m.group(2)), "k": 6}}

    # cluster overview — no anchor
    if re.search(r"sonic\s+territor|major\s+.*(territor|region|cluster)|"
                 r"what\s+(?:are\s+)?(?:the\s+)?(?:main|major)?\s*clusters|"
                 r"overview\s+of\s+the\s+map|map\s+regions|how\s+many\s+clusters", ql):
        return {"shape": "cluster_summary", "tool": "cluster_summary", "args": {}}

    # coherence / outliers over a placed album or playlist
    if re.search(r"coheren|cohesive|how\s+tight|consistent|outlier|hold\s+together|"
                 r"fit\s+together|stick\s+out|which.*doesn'?t\s+fit", ql):
        scope = "playlist"
        if "album" in ql:
            scope = "album"
        elif "artist" in ql:
            scope = "artist"
        target = _coherence_target(q)
        return {"shape": "coherence", "tool": "coherence",
                "args": {"scope": scope, "target": target}}

    # micro-genres
    if re.search(r"micro[-\s]?genre|which\s+genres|what\s+genres|genre\s+fields?|"
                 r"what.*(sub|micro)genre", ql):
        return {"shape": "micro_genres", "tool": "micro_genres",
                "args": {"anchor": _anchor_from(q)}}

    # tagmates
    if re.search(r"tagmate|share.*tags?|same\s+tags?|tagged\s+like|other.*same\s+descriptors?", ql):
        return {"shape": "tagmates", "tool": "tagmates",
                "args": {"anchor": _anchor_from(q), "k": 8}}

    # crossover / adjacent cluster to explore
    if re.search(r"adjacent\s+cluster|cross\s*over|cross\s+into|neighbou?r(?:ing)?\s+cluster|"
                 r"which\s+cluster.*(study|explore|move|expand|try)|next\s+cluster", ql):
        return {"shape": "crossover", "tool": "crossover",
                "args": {"anchor": _anchor_from(q), "k": 6}}

    # sounds-like / closest artists (single anchor), incl. "my uploaded demo"
    if re.search(r"sound(?:s)?\s+(?:closest|like|similar)|what\s+do(?:es)?\s+.*sound|"
                 r"closest\s+to|similar\s+to|what.*sound\s+like|land\??$", ql) or \
       re.search(r"upload|demo|my\s+(?:track|song|sound)", ql):
        return {"shape": "sounds_like", "tool": "sounds_like",
                "args": {"anchor": _anchor_from(q), "top_k": 8}}

    # multi-seed recommend (Q10)
    if re.search(r"recommend|suggest|based\s+on|from\s+these|seeds?", ql):
        seeds = _seeds_from(q)
        if len(seeds) >= 2:
            return {"shape": "bridge", "tool": "bridge",
                    "args": {"anchor_a": seeds[0], "anchor_b": seeds[1], "k": 8}}
        return {"shape": "sounds_like", "tool": "sounds_like",
                "args": {"anchor": seeds[0] if seeds else _anchor_from(q), "top_k": 8}}

    # default: resolve the most likely anchor and read it
    return {"shape": "sounds_like", "tool": "sounds_like",
            "args": {"anchor": _anchor_from(q), "top_k": 8}}


_INTERROGATIVE = {"which", "what", "who", "how", "where", "when", "does", "do",
                  "is", "are", "can", "should", "would", "will", "the", "a", "an"}


def _anchor_from(question):
    """Best single anchor. A concrete named anchor (quoted, parenthetical, or the
    subject of "does X ...") wins over the generic 'me'; only fall back to 'me'
    when the question is about the user's own sound with no concrete name."""
    q = question.strip()
    quoted = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', q)
    if quoted:
        return quoted[0].strip()
    paren = re.findall(r"\(([^)]+)\)", q)
    if paren:
        return paren[-1].strip()
    # "... does <anchor> (actually) fall/sound/have/match ..." — the anchor is the
    # subject after "does/do", and it is frequently typed lowercase.
    m = re.search(r"\bdo(?:es)?\s+(.+?)\s+(?:actually\s+)?"
                  r"(?:fall|sound|sit|have|match|land|belong|go)\b", q, re.IGNORECASE)
    if not m:
        # "... should <anchor> cross/explore/study/move ..."
        m = re.search(r"\bshould\s+(.+?)\s+"
                      r"(?:cross|explore|study|move|expand|try|go|sit)\b", q, re.IGNORECASE)
    if m:
        cand = _clean_anchor(m.group(1))
        if cand and cand.lower() not in _INTERROGATIVE:
            return cand
    if re.search(r"\b(my|me|uploaded|upload|demo|mine)\b", q, re.IGNORECASE):
        return "me"
    # Proper-Case spans, but skip a leading interrogative ("Which", "What").
    proper = re.findall(r"\b([A-Z][a-zA-Z0-9&']+(?:\s+[A-Z][a-zA-Z0-9&']+){0,3})\b", q)
    proper = [p for p in proper if p.lower() not in _INTERROGATIVE]
    if proper:
        return proper[-1].strip()
    tail = q.split("?")[0].strip()
    return tail if tail else "me"


def _seeds_from(question):
    seeds = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', question)
    if seeds:
        return [s.strip() for s in seeds]
    # isolate the seed list: text after "based on / from / like / seeds:"
    m = re.search(r"(?:based\s+on|from|like|seeds?:?|using)\s+(.+)$",
                  question.split("?")[0], re.IGNORECASE)
    tail = m.group(1) if m else question.split("?")[0]
    parts = re.split(r"\s*(?:,|\band\b|\+|/|&)\s*", tail)
    out = []
    for p in parts:
        p = _clean_anchor(p)
        if not p:
            continue
        # prefer a Proper-Case span, else take the whole cleaned fragment (seeds
        # may be typed lowercase)
        mm = re.search(r"\b([A-Z][a-zA-Z0-9&']+(?:\s+[A-Z][a-zA-Z0-9&']+){0,3})\b", p)
        out.append(mm.group(1).strip() if mm else p)
    return [s for s in out if s]


def _coherence_target(question):
    quoted = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', question)
    if quoted:
        return quoted[0].strip()
    paren = re.findall(r"\(([^)]+)\)", question)
    if paren:
        return paren[-1].strip()
    # name follows "album/playlist/record [called] X", stopping at a clause break
    m = re.search(r"(?:album|playlist|record|artist)\s+(?:called\s+|named\s+)?"
                  r"([^,\?\.]+?)(?:\s+(?:and|,|\bwhat\b|\bhow\b|\bwhich\b|\bare\b|\bis\b)|[,\?\.]|$)",
                  question, re.IGNORECASE)
    if m:
        return _clean_anchor(m.group(1))
    proper = re.findall(r"\b([A-Z][a-zA-Z0-9&']+(?:\s+[A-Z][a-zA-Z0-9&']+){0,3})\b", question)
    return proper[-1].strip() if proper else None


class MentorReAct:
    def __init__(self, anther, generate_fn, graph_ctx=None):
        self.graph = MentorGraphTools(anther, graph_ctx=graph_ctx)
        self.generate_fn = generate_fn

    def _execute(self, call, context=None):
        tool = call["tool"]
        args = call.get("args", {})

        def _int(key, default):
            try:
                return int(args.get(key, default))
            except (TypeError, ValueError):
                return default

        if tool == "resolve":
            return self.graph.resolve(args.get("name", "") or args.get("anchor", ""), context=context)
        if tool == "sounds_like":
            return self.graph.sounds_like(args.get("anchor", ""), top_k=_int("top_k", 8), context=context)
        if tool == "compare":
            return self.graph.compare(args.get("anchor_a", ""), args.get("anchor_b", ""),
                                      top_k=_int("top_k", 8), context=context)
        if tool == "artist_tracks":
            return self.graph.artist_tracks(args.get("anchor", ""), args.get("artist", ""),
                                            top_k=_int("top_k", 8), context=context)
        if tool == "bridge":
            return self.graph.bridge(args.get("anchor_a", ""), args.get("anchor_b", ""),
                                     k=_int("k", 6), context=context, within=args.get("within"))
        if tool == "crossover":
            return self.graph.crossover(args.get("anchor", ""), k=_int("k", 6), context=context)
        if tool == "tagmates":
            return self.graph.tagmates(args.get("anchor", ""), k=_int("k", 8), context=context)
        if tool == "micro_genres":
            return self.graph.micro_genres(args.get("anchor", ""), context=context, top_k=_int("top_k", 6))
        if tool in ("cluster_summary", "get_cluster_profiles"):
            # top_n default is None (no cap, return every cluster) — see the
            # top_n docstring in mentor_graph.py for why this must not be a
            # hardcoded count.
            top_n = args.get("top_n")
            return self.graph.cluster_summary(cluster_id=args.get("cluster_id"),
                                              top_n=int(top_n) if top_n is not None else None)
        if tool == "coherence":
            return self.graph.coherence(scope=args.get("scope", "album"),
                                        target=args.get("target"), context=context)
        return {"tool": tool, "ok": False, "error": "unknown_tool"}

    @staticmethod
    def _is_followup(question):
        q = (question or "").lower()
        markers = [
            "what specific songs",
            "specific songs",
            "which ones",
            "those artists",
            "tell me more",
            "of those",
            "list them",
        ]
        return any(m in q for m in markers)

    @staticmethod
    def _extract_anchor(calls):
        for call in calls:
            if not isinstance(call, dict):
                continue
            tool = call.get("tool")
            args = call.get("args") or {}
            if tool in ("resolve",):
                if args.get("name"):
                    return str(args.get("name"))
            if tool in ("sounds_like", "crossover", "tagmates", "artist_tracks"):
                if args.get("anchor"):
                    return str(args.get("anchor"))
            if tool in ("bridge", "compare"):
                a = str(args.get("anchor_a", "")).strip()
                b = str(args.get("anchor_b", "")).strip()
                if a and a.lower() not in {"me", "my", "my track", "my sound"}:
                    return a
                if b:
                    return b
                if a:
                    return a
        return None

    # ---- surprise routing --------------------------------------------------
    def _reroute(self, call, obs, question):
        """A surprise (ok:False) routes the loop. Return a new call to try, or
        None to stop. We only give up when an anchor is genuinely absent."""
        tool = call.get("tool")
        args = call.get("args", {})
        err = obs.get("error")

        # within-bridge but the 'within' artist isn't on screen -> drop to a
        # plain bridge between the two anchors (still answers "what sits between").
        if tool == "bridge" and err == "within_artist_not_onscreen":
            a2 = dict(args); a2.pop("within", None)
            return {"tool": "bridge", "args": a2}

        # artist_tracks anchor resolved but artist pool empty -> compare instead
        if tool == "artist_tracks" and err == "artist_not_resolved":
            return {"tool": "sounds_like", "args": {"anchor": args.get("artist", ""), "top_k": 8}}

        # anchor didn't resolve at all -> nothing to reroute to; caller deflects.
        return None

    @staticmethod
    def _absent_anchor(observations):
        """Name the first anchor that failed to resolve, for honest deflection."""
        for o in observations:
            if o.get("ok"):
                continue
            if o.get("error") not in ("anchor_not_resolved", "within_artist_not_onscreen",
                                      "artist_not_resolved", "group_not_found"):
                continue
            res = o.get("resolved") or {}
            name = res.get("input") or o.get("within") or o.get("target")
            if name:
                return str(name)
        return None

    def run(self, question, max_steps=4, verbose=False, context=None):
        # Follow-up ("which ones?", "list them") over the prior graph turn.
        if context is not None and self._is_followup(question):
            prior = getattr(context, "last_graph_observations", None) or []
            if prior:
                answer = self._compose(question, prior)
                return {"answer": answer, "calls": [], "observations": prior,
                        "used_fallback": False, "graph_state": {"anchor": getattr(context, "last_anchor", None)}}

        # Step 0 — classify the shape deterministically; this seeds the loop and
        # backs up any unusable LLM emission.
        seed = classify_question(question)
        seed_call = {"tool": seed["tool"], "args": seed["args"]}

        # Optionally let the model refine the seed (voice/params), but the
        # deterministic seed is authoritative when the model produces junk.
        call = seed_call
        if self.generate_fn is not None:
            try:
                msgs = [{"role": "system", "content": REACT_SYSTEM},
                        {"role": "user", "content": question}]
                raw = self.generate_fn(msgs, max_new_tokens=120, temperature=0.0)
                cand = _extract_first_json(raw) if raw else None
                if cand and _validate_call(cand) and cand.get("tool") != "final" \
                        and cand.get("tool") == seed_call["tool"]:
                    # accept only if it agrees on the tool; merge args over the seed
                    merged = dict(seed_call["args"]); merged.update(cand.get("args", {}))
                    call = {"tool": cand["tool"], "args": merged}
            except Exception:
                call = seed_call

        calls, observations = [], []
        for _ in range(max_steps):
            calls.append(call)
            obs = self._execute(call, context=context)
            observations.append(obs)
            if context is not None:
                try:
                    context.last_graph_observations = observations
                except Exception:
                    pass
            if obs.get("ok"):
                break
            nxt = self._reroute(call, obs, question)
            if nxt is None:
                break
            call = nxt

        anchor = self._extract_anchor(calls)
        ok_obs = [o for o in observations if o.get("ok")]

        # Honest deflection: distinguish absent anchor from generic failure.
        if not ok_obs:
            missing = self._absent_anchor(observations)
            if missing:
                answer = (
                    f"\u201c{missing}\u201d isn't on your map yet, so I can't place it against the "
                    "others. Add it (paste a track or upload audio) and ask again, or point me at "
                    "an artist that's already on screen."
                )
            else:
                answer = (
                    "I couldn't pin that to anything on your map. Name an artist or track that's "
                    "on screen, or upload audio, and I'll map it."
                )
            return {"answer": answer, "calls": calls, "observations": observations,
                    "used_fallback": True, "graph_state": {"anchor": anchor},
                    "status": "absent_anchor"}

        # Verify by observation: compose the answer from tool results only.
        answer = self._compose(question, observations)
        if context is not None:
            try:
                context.last_anchor = anchor
            except Exception:
                pass
        return {"answer": answer, "calls": calls, "observations": observations,
                "used_fallback": False, "graph_state": {"anchor": anchor}}

    # ---- deterministic answer composer ------------------------------------
    def _compose(self, question, observations):
        """Build the answer from observations. Names come only from tool output;
        the LLM (if present) only rephrases in voice, and is discarded if it
        introduces a name that wasn't observed."""
        facts, allowed = self._facts(observations)
        base = self._readout(observations)
        if self.generate_fn is None or not base:
            return base or "I mapped that, but the tools returned nothing to summarize."
        prompt = [
            {"role": "system", "content": (
                "You are a seasoned music mentor. Rewrite the FACTS into a warm, concise answer "
                "(2-4 sentences) in your own voice. Use ONLY the artists, tracks, and clusters in "
                "FACTS. Do not invent or add any name that isn't in FACTS.")},
            {"role": "user", "content": "Question: " + question + "\n\nFACTS:\n" + facts},
        ]
        try:
            voiced = self.generate_fn(prompt, max_new_tokens=220, temperature=0.4)
        except Exception:
            voiced = ""
        voiced = (voiced or "").strip()
        # Guard: reject a voiced answer that names something not in the allowed set.
        if voiced and self._names_ok(voiced, allowed):
            return voiced
        return base

    def _facts(self, observations):
        """(facts_text, allowed_names_set) — the ground truth the composer may use."""
        lines, allowed = [], set()

        def add_name(*vals):
            for v in vals:
                if v:
                    allowed.add(str(v).lower())

        for o in observations:
            if not o.get("ok"):
                continue
            tool = o.get("tool")
            if tool == "bridge" and o.get("within"):
                lines.append(f"Songs from {o.get('within')} bridging the two anchors, best first:")
                for n in o.get("neighbors", [])[:6]:
                    add_name(n.get("name"), n.get("artist"), o.get("within"))
                    lines.append(f"  - {n.get('name')} (bridge score {n.get('bridge_score')}, "
                                 f"lean {n.get('lean')})")
                mb = o.get("most_balanced")
                if mb:
                    add_name(mb.get("name"))
                    lines.append(f"  most balanced: {mb.get('name')}")
            elif tool in ("bridge", "sounds_like", "tagmates", "artist_tracks", "crossover"):
                for n in o.get("neighbors", [])[:8]:
                    add_name(n.get("name"), n.get("artist"))
                    label = " - ".join(x for x in (n.get("artist"), n.get("name")) if x)
                    extra = []
                    if "score" in n:
                        extra.append(f"sim {n['score']}")
                    if n.get("shared_tags"):
                        extra.append("tags: " + ", ".join(n["shared_tags"]))
                    if n.get("cluster_label"):
                        extra.append(str(n["cluster_label"]))
                    lines.append(f"  - {label}" + (f" ({'; '.join(extra)})" if extra else ""))
            elif tool == "compare":
                a = o.get("anchor_a", {}).get("match", "A"); b = o.get("anchor_b", {}).get("match", "B")
                add_name(a, b)
                lines.append(f"{a} vs {b}: similarity {o.get('similarity')}.")
                for side, key in (("left", "left_neighbors"), ("right", "right_neighbors")):
                    for n in o.get(key, [])[:4]:
                        add_name(n.get("name"), n.get("artist"))
                        lines.append(f"  {side}: {n.get('artist','')} - {n.get('name','')}")
            elif tool == "micro_genres":
                lines.append("Micro-genres: " + ", ".join(o.get("tags", [])) +
                             f" (from {o.get('source')}).")
                for t in o.get("tags", []):
                    add_name(t)
            elif tool == "cluster_summary":
                if o.get("cluster"):
                    c = o["cluster"]; add_name(c.get("label"))
                    lines.append(f"Cluster {c.get('cluster_id')}: {c.get('label')} "
                                 f"(size {c.get('size')}); tags {', '.join(c.get('top_tags', []))}.")
                else:
                    all_clusters = o.get("clusters", [])
                    n_total = o.get("n_clusters", len(all_clusters))
                    # Cap what's actually listed in a chat answer at the biggest
                    # territories (a corpus can have far more clusters than fit
                    # in a readable message — e.g. 84); n_clusters above always
                    # reports the true total, this only bounds the display.
                    DISPLAY_CAP = 15
                    shown = sorted(all_clusters, key=lambda c: -(c.get("size") or 0))[:DISPLAY_CAP]
                    if n_total > len(shown):
                        lines.append(f"The map has {n_total} sonic territories "
                                     f"(showing the {len(shown)} largest):")
                    else:
                        lines.append(f"The map has {n_total} sonic territories:")
                    for c in shown:
                        add_name(c.get("label"))
                        lines.append(f"  [{c.get('cluster_id')}] {c.get('label')} "
                                     f"(size {c.get('size')}; {', '.join(c.get('top_tags', [])[:3])})")
            elif tool == "coherence":
                lines.append(f"{o.get('target')}: coherence {o.get('coherence_mean')} "
                             f"(std {o.get('coherence_std')}), spread across "
                             f"{o.get('distinct_clusters')} clusters.")
                for m in o.get("outliers", []):
                    add_name(m.get("name"), m.get("artist"))
                    lines.append(f"  outlier: {m.get('artist','')} - {m.get('name','')} (z {m.get('z')})")
        return "\n".join(lines), allowed

    def _readout(self, observations):
        """Plain-language answer straight from facts (used when no LLM, or as the
        hallucination-safe fallback)."""
        facts, _ = self._facts(observations)
        return facts

    @staticmethod
    def _names_ok(text, allowed):
        """True unless the voiced text introduces a MULTI-word Proper-Case name
        absent from the allowed set. Multi-word Title Case is where hallucinated
        artist/track names appear; single capitalized words are almost always
        sentence starts, so we don't flag them (avoids false rejections)."""
        if not allowed:
            return True
        cands = re.findall(r"\b([A-Z][a-zA-Z0-9&']+(?:\s+[A-Z][a-zA-Z0-9&']+){1,3})\b", text)
        for c in cands:
            cl = c.lower()
            # allow if the candidate overlaps any observed name in either direction
            if any(cl in a or a in cl for a in allowed):
                continue
            # allow overlap on any single token (e.g. "Are You Experienced?" vs
            # an allowed "Are You Experienced")
            toks = set(cl.split())
            if any(toks & set(a.split()) for a in allowed):
                continue
            return False
        return True
