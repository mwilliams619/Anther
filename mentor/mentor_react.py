"""Strict-JSON ReAct loop for graph questions."""

from __future__ import annotations

import json
import re

try:
    from .mentor_graph import MentorGraphTools
except ImportError:  # pragma: no cover - script-style import path
    from mentor_graph import MentorGraphTools


REACT_SYSTEM = (
    "You are a music mentor with graph tools. For each step, output exactly one JSON object. "
    "Use either {\"tool\":\"<name>\",\"args\":{...}} or {\"tool\":\"final\",\"answer\":\"...\"}. "
    "Available tools: resolve(name), sounds_like(anchor, top_k), compare(anchor_a, anchor_b, top_k), "
    "bridge(anchor_a, anchor_b, k), crossover(anchor, k), artist_tracks(anchor, artist, top_k), tagmates(anchor, k). "
    "These are read-only lookups over the sound-similarity graph — they never place, remove, or modify "
    "anything. Keep calls minimal and grounded."
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


class MentorReAct:
    def __init__(self, anther, generate_fn):
        self.graph = MentorGraphTools(anther)
        self.generate_fn = generate_fn

    def _execute(self, call, context=None):
        tool = call["tool"]
        args = call.get("args", {})
        if tool == "resolve":
            return self.graph.resolve(args.get("name", ""), context=context)
        if tool == "sounds_like":
            return self.graph.sounds_like(args.get("anchor", ""), top_k=int(args.get("top_k", 8)), context=context)
        if tool == "compare":
            return self.graph.compare(args.get("anchor_a", ""), args.get("anchor_b", ""), top_k=int(args.get("top_k", 8)), context=context)
        if tool == "artist_tracks":
            return self.graph.artist_tracks(args.get("anchor", ""), args.get("artist", ""), top_k=int(args.get("top_k", 8)), context=context)
        if tool == "bridge":
            return self.graph.bridge(args.get("anchor_a", ""), args.get("anchor_b", ""), k=int(args.get("k", 6)), context=context)
        if tool == "crossover":
            return self.graph.crossover(args.get("anchor", ""), k=int(args.get("k", 6)), context=context)
        if tool == "tagmates":
            return self.graph.tagmates(args.get("anchor", ""), k=int(args.get("k", 8)), context=context)
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

    def run(self, question, max_steps=3, verbose=False, context=None):
        msgs = [{"role": "system", "content": REACT_SYSTEM}] + REACT_FEWSHOT
        if context is not None and self._is_followup(question):
            prior = getattr(context, "last_graph_observations", None) or []
            if prior:
                msgs.append(
                    {
                        "role": "user",
                        "content": "PRIOR_OBSERVATIONS: " + json.dumps(prior[:2], ensure_ascii=True),
                    }
                )
        msgs.append({"role": "user", "content": question})
        calls = []
        observations = []

        for _ in range(max_steps):
            raw = self.generate_fn(msgs, max_new_tokens=180, temperature=0.0)
            call = _extract_first_json(raw)
            if call is None or not _validate_call(call):
                call = _fallback_call(question)
            calls.append(call)

            if call.get("tool") == "final":
                anchor = self._extract_anchor(calls)
                return {
                    "answer": call.get("answer", ""),
                    "calls": calls,
                    "observations": observations,
                    "used_fallback": call == _fallback_call(question),
                    "graph_state": {"anchor": anchor},
                }

            obs = self._execute(call, context=context)
            observations.append(obs)
            msgs.append({"role": "assistant", "content": json.dumps(call, ensure_ascii=True)})
            msgs.append({"role": "user", "content": "OBSERVATION: " + json.dumps(obs, ensure_ascii=True)})

            if context is not None:
                try:
                    context.last_graph_observations = observations
                except Exception:
                    pass

        # Forced final step after tool cap.
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a seasoned music mentor. Answer in voice using ONLY the tool observations below. "
                    "Name artists from the observations. Do not invent artists."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question: "
                    + question
                    + "\nObservations:\n"
                    + "\n".join(json.dumps(o, ensure_ascii=True) for o in observations)
                ),
            },
        ]
        answer = self.generate_fn(summary_prompt, max_new_tokens=220, temperature=0.5)
        anchor = self._extract_anchor(calls)
        return {
            "answer": answer,
            "calls": calls,
            "observations": observations,
            "used_fallback": True,
            "graph_state": {"anchor": anchor},
        }
