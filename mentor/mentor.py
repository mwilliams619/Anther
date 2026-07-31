"""The music mentor agent.

A music mentor sitting inside an interactive spatial sound map. Three
mechanisms, kept separate:
  1. TONE   — Qwen2.5-7B-Instruct + QLoRA adapter (punchy Reddit-mentor voice)
  2. INFO   — RAG over the source advice passages (defer to the training data)
  3. EARS   — the graph agent (agent.py): LLM interprets intent, deterministic
              tools read the user's map + the Anther corpus, LLM narrates.

Usage:
    m = MusicMentor()
    print(m.chat("How do I stop scrapping every song halfway through?"))
    print(m.chat("What do I sound like?", audio_path="my_demo.wav"))
    print(m.chat("Why is this node here?", selected_node_id="abc123"))
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
import torch
from paths import BASE_DIR, LORA_DIR
from graph_context import MentorContext, _norm_text as _norm

# Pipeline trace (QUESTION / CLASSIFICATION / TOOL / OBSERVATION / ANSWER) is
# printed when chat(verbose=True) or ANTHER_MENTOR_TRACE=1 (the service sets it).
TRACE = os.environ.get("ANTHER_MENTOR_TRACE", "") not in ("", "0", "false")

# re-exported for service.py / tests (the session-state object lives with the
# other context machinery in graph_context.py)
ConversationState = MentorContext

# Persona shared by both branches (kept for the sounds_like readout too).
SYSTEM = (
    "You are a seasoned music mentor for aspiring artists. You give direct, "
    "encouraging, no-nonsense advice grounded in real experience. Be concrete "
    "and honest; skip empty platitudes. When source advice is provided, defer "
    "to it and weave it into your answer in your own voice."
)

# Used when the query is IN SCOPE (RAG found relevant source advice).
GROUNDED_SYSTEM = (
    "You are a seasoned music mentor for aspiring artists. You give direct, "
    "encouraging, no-nonsense advice grounded in real experience. Be concrete "
    "and honest; skip empty platitudes. Source advice from experienced "
    "musicians is provided below — defer to it and rephrase it in your own "
    "voice. Speak directly to the artist in the second person. Never describe "
    "yourself, your background, or your credentials; just give the advice."
)

# Used when the query is OUT OF SCOPE (no relevant source advice found).
# In-persona redirect: acknowledge briefly, steer back to music, invent nothing.
UNGROUNDED_SYSTEM = (
    "You are a seasoned music mentor for aspiring artists. The user has asked "
    "something outside your lane — it is not about making music, an artist's "
    "craft, or a music career. In one or two sentences, in your blunt, warm "
    "mentor voice, tell them that is not your wheelhouse and steer them back to "
    "music: songwriting, production, releasing, promotion, performing, or the "
    "business. Do NOT answer the off-topic question. Do NOT invent facts, "
    "products, or deals. Do NOT describe yourself or claim credentials. Keep it "
    "short."
)

# Few-shot exemplars for the ungrounded redirect. Small models follow a
# demonstrated pattern far more reliably than a bare instruction.
UNGROUNDED_FEWSHOT = [
    {"role": "user", "content": "how do I make salsa I have tortilla chips already"},
    {"role": "assistant", "content": "Ha — that's not my lane, I'm here for the music, not the snacks. But if you want to talk about writing a track, mixing it, or getting it heard, I'm all in. What are you working on?"},
    {"role": "user", "content": "what's the best prime day deal this year"},
    {"role": "assistant", "content": "Can't help you shop, that's outside what I do. What I can help with is your music — recording, releasing, building an audience. What do you need?"},
    {"role": "user", "content": "what's the weather tomorrow"},
    {"role": "assistant", "content": "Not my department. I'm your mentor for making and putting out music. Got a song, a release, or a promo problem you're stuck on?"},
    {"role": "user", "content": "write me a python function to sort a list"},
    {"role": "assistant", "content": "Wrong lane. I coach artists, not code. If you're trying to finish a track, tighten your mix, or plan a release, let's do that."},
    {"role": "user", "content": "are you an AI assistant"},
    {"role": "assistant", "content": "Labels don't matter here. I'm focused on your music and your next move. What are you building right now?"},
]

# Scope threshold, calibrated from real cosines: on-topic advice queries score
# >=0.66, clearly off-topic queries (salsa, prime-day, weather, car repair)
# score <=0.62. 0.63 sits in that gap. Tune against the regression set.
SCOPE_THRESHOLD = 0.63

INTENT_SYSTEM = (
    "Classify the user's request for a music mentor assistant that sits inside "
    "an interactive sound map. Return exactly one token from this set: "
    "ADVICE, GRAPH, OFFTOPIC.\n"
    "ADVICE: asks for music guidance, strategy, songwriting, production, "
    "release, promotion, audience growth, career decisions.\n"
    "GRAPH: asks about the map or the sounds on it — what they sound like, who "
    "they are close to, what is between two sounds, why a node is placed where "
    "it is, what is on the map, where to explore next, or a short follow-up to "
    "a previous map answer (\"which ones?\", \"list them\").\n"
    "OFFTOPIC: asks for non-music help (coding, shopping, recipes, weather, "
    "repair, trivia, travel, generic non-music support)."
)

INTENT_FEWSHOT = [
    {"role": "user", "content": "How do I stop overproducing and finally finish tracks?"},
    {"role": "assistant", "content": "ADVICE"},
    {"role": "user", "content": "What artists sit between me and Neroptik?"},
    {"role": "assistant", "content": "GRAPH"},
    {"role": "user", "content": "Why is this song placed next to Burial?"},
    {"role": "assistant", "content": "GRAPH"},
    {"role": "user", "content": "[Previous turn: GRAPH]\nwhich ones?"},
    {"role": "assistant", "content": "GRAPH"},
    {"role": "user", "content": "[Previous turn: GRAPH]\nwhat are the scores?"},
    {"role": "assistant", "content": "GRAPH"},
    {"role": "user", "content": "Write me a python function to sort a list."},
    {"role": "assistant", "content": "OFFTOPIC"},
    {"role": "user", "content": "Are you an AI?"},
    {"role": "assistant", "content": "OFFTOPIC"},
]

VALID_INTENTS = ("ADVICE", "GRAPH", "OFFTOPIC")

# ---- retrieval-mode router (replaces the old binary SCOPE_THRESHOLD gate) --
#
# The old gate: `in_scope = (intent == "ADVICE") and (max_score >= SCOPE_THRESHOLD)`.
# If in_scope and hits: quote-and-defer. Otherwise: fall through to the SAME
# UNGROUNDED/redirect branch used for genuinely off-topic questions -- even
# though `intent` was already ADVICE. That is the whack-a-mole bug: a
# brainstorm/plan-drafting question ("help me plan my release rollout") is
# on-topic but rarely has one passage that matches it well, so it either got
# a verbatim tip quoted at it (if some passage scored just high enough) or
# got told "that's not my lane" (if nothing scored high enough) -- there was
# no answer shape for "synthesize something new for me from what you know."
#
# This router gives ADVICE questions three destinations instead of one
# pass/fail check, following the adaptive-retrieval literature:
#   - Self-RAG (Asai et al. 2023, arXiv:2310.11511) retrieves on demand
#     rather than every turn -- the model decides whether retrieval helps.
#   - Adaptive-RAG (Jeong et al. 2024, arXiv:2403.14403) routes queries by
#     complexity to no-retrieval / single-step / multi-step strategies
#     using a lightweight trained classifier, instead of one fixed policy
#     for every query.
#   - CRAG (Yan et al. 2024, arXiv:2401.15884) treats retrieval quality as
#     a graded signal (correct / incorrect / ambiguous) with a distinct
#     corrective action per grade, rather than a single relevant/irrelevant
#     cutoff.
#
# GROUNDED     -- a direct, settleable question with a strong passage match:
#                 quote-and-defer (the original, still-correct behavior).
# SYNTHESIS    -- brainstorming, drafting, planning: retrieve passages as
#                 optional supporting texture, but the model reasons beyond
#                 them and produces an original answer for this artist.
# NO_RETRIEVAL -- on-topic, but no passage is a good enough match to be worth
#                 quoting (Self-RAG's "retrieval would not help" case): answer
#                 from the mentor persona directly. Never a redirect -- the
#                 question was already classified ADVICE, so "not my lane" is
#                 wrong here by construction.
VALID_RETRIEVAL_MODES = ("GROUNDED", "SYNTHESIS", "NO_RETRIEVAL")

# Deterministic pre-filter for the clearest SYNTHESIS case (mirrors the
# _GRAPH_PHRASES / _short_graph_followup pattern above): if the question
# explicitly asks to brainstorm/draft/plan/outline, don't even ask the LLM.
# Two layers: (1) fixed phrases for common exact wording, (2) a verb+noun
# combination check so wording variants ("draft a 4-week plan for my EP",
# "sketch out some options for the rollout") still hit the deterministic
# path instead of falling through to the LLM every time.
_SYNTHESIS_SIGNAL_PHRASES = (
    "brainstorm", "help me plan", "plan out", "plan for my",
    "come up with ideas", "come up with some ideas", "give me ideas",
    "ideas for my", "roadmap for", "map out",
    "strategy for my", "help me think through", "think through",
    "workshop this", "generate ideas", "brainstorm ideas",
    "what are some ideas", "give me options", "walk me through planning",
    "help me draft",
)
_SYNTHESIS_VERBS = ("draft", "outline", "sketch", "workshop", "brainstorm", "map out")
_SYNTHESIS_NOUNS = ("plan", "ideas", "idea", "options", "strategy", "roadmap")

# Used for SYNTHESIS: passages (if any) are texture, not a script to recite.
SYNTHESIS_SYSTEM = (
    "You are a seasoned music mentor for aspiring artists. The user wants "
    "help brainstorming, drafting, or planning something for their own "
    "situation -- not a quoted tip. If source passages are provided below, "
    "use them as inspiration and texture, but do not recite them verbatim "
    "and do not just pick one and defer to it. Reason beyond the passages: "
    "synthesize an original, concrete plan or set of ideas shaped to what "
    "the artist described. Be direct and specific -- concrete next steps, "
    "not platitudes. Speak to the artist in the second person. Never "
    "describe yourself, your background, or your credentials."
)

# Used for NO_RETRIEVAL: on-topic, but nothing in the corpus is worth
# quoting. This is NOT the off-topic redirect -- it answers the question.
REFORMAT_SYSTEM = (
    "You are a music mentor restyling your own previous answer at the "
    "artist's request. Keep every fact, artist, song, and piece of advice "
    "already in the previous answer -- do not invent anything new and do "
    "not drop the substance. Only change presentation: structure (e.g. "
    "bullet points), length, or tone, exactly as asked. If asked to "
    "shorten or condense, actually make it shorter -- never just repeat "
    "the previous answer back unchanged."
)

NO_RETRIEVAL_SYSTEM = (
    "You are a seasoned music mentor for aspiring artists. Answer the "
    "user's music question directly from your own judgment and experience "
    "-- no source material is being provided because nothing in the corpus "
    "is a close enough match to be worth quoting. Do not say you lack "
    "information or apologize for it; just give direct, concrete, honest "
    "advice. Speak to the artist in the second person. Never describe "
    "yourself, your background, or your credentials."
)

RETRIEVAL_MODE_SYSTEM = (
    "The user's question has already been classified as music ADVICE for a "
    "music mentor assistant. Decide how it should be answered. Return "
    "exactly one token from this set: GROUNDED, SYNTHESIS, NO_RETRIEVAL.\n"
    "GROUNDED: a direct, answerable question where one specific piece of "
    "advice would settle it (how do I fix X, what's the right way to Y, is "
    "Z a good idea, should I do A or B).\n"
    "SYNTHESIS: asks to brainstorm, draft, plan, outline, or generate "
    "original ideas or options for their specific situation -- they want "
    "something built for them, not a quoted tip.\n"
    "NO_RETRIEVAL: general encouragement, opinion, or a personal/subjective "
    "question with no single settled answer, where quoting outside advice "
    "would not add anything -- answer from your own judgment as a mentor."
)

RETRIEVAL_MODE_FEWSHOT = [
    {"role": "user", "content": "How do I stop clipping when I master my tracks?"},
    {"role": "assistant", "content": "GROUNDED"},
    {"role": "user", "content": "Help me brainstorm ideas for my album rollout."},
    {"role": "assistant", "content": "SYNTHESIS"},
    {"role": "user", "content": "Draft a 4-week release plan for my next single."},
    {"role": "assistant", "content": "SYNTHESIS"},
    {"role": "user", "content": "Do you think I should keep making music even though I'm not blowing up yet?"},
    {"role": "assistant", "content": "NO_RETRIEVAL"},
    {"role": "user", "content": "What's the best way to pitch my song to playlist curators?"},
    {"role": "assistant", "content": "GROUNDED"},
    {"role": "user", "content": "Can you help me plan out my next few months as an artist?"},
    {"role": "assistant", "content": "SYNTHESIS"},
]


class MusicMentor:
    def __init__(self, use_rag=True, use_anther=True, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(LORA_DIR, local_files_only=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, quantization_config=bnb, torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="eager", local_files_only=True)
        self.model = PeftModel.from_pretrained(base, LORA_DIR)
        self.model.eval()

        self.rag = None
        if use_rag:
            from mentor_rag import MentorRAG
            self.rag = MentorRAG(device=device)

        self.similarity = None
        self.agent = None
        self.graph_ctx = None
        self._session_state = MentorContext()
        if use_anther:
            from anther_service import AntherSimilarityService
            from agent import MentorAgent
            self.similarity = AntherSimilarityService()
            # Live on-screen session (graph.json + embed_cache), read fresh per
            # graph-turn. The mentor runs in a separate process from the UI, so
            # this reads state from disk rather than the UI's in-memory atlas.
            # It is re-pointed at the browser's session each turn (see chat()).
            from graph_context import GraphContext
            self.graph_ctx = GraphContext()
            self.agent = MentorAgent.build(self.similarity, self._generate,
                                           graph_ctx=self.graph_ctx)

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def _generate(self, messages, max_new_tokens=320, temperature=0.7):
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(self.device)
        in_len = enc["input_ids"].shape[1]
        do_sample = temperature > 0
        gen_kwargs = {
            **enc,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        out = self.model.generate(**gen_kwargs)
        return self.tok.decode(out[0, in_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _normalize_intent(raw):
        if not raw:
            return None
        m = re.search(r"\b(ADVICE|GRAPH|OFFTOPIC)\b", raw.upper())
        return m.group(1) if m else None

    # Unambiguous "this is about the map in front of me" phrases. When any of
    # these appears the question is a GRAPH question no matter what a 7B router
    # guesses — this is the gate that stops "what is X connected to on the map"
    # from being misread as off-topic. It never chooses a tool (the agent's
    # intent classifier does that); it only guarantees the graph branch runs.
    _GRAPH_PHRASES = (
        "on the map", "on my map", "on the graph", "on screen", "on-screen",
        "on the board", "on the canvas", "in the map", "in my map",
        "what's here", "whats here", "what is here", "what am i looking at",
        "looking at right now", "connected to", "linked to", "next to",
        "closest to", "nearest to", "sit between", "sits between",
        "sit sonically", "what do i sound like", "what does my", "sound like",
        "what's on the map", "whats on the map", "what is on the map",
        "why is this", "why is that", "this node", "this song", "selected node",
    )

    def _graph_signal(self, question):
        """High-precision GRAPH detector: an explicit map phrase, or a mention
        of something actually placed on the map. False is inconclusive (defer
        to the LLM), never a claim that it ISN'T a graph question."""
        q = _norm(question)
        if any(p in q for p in self._GRAPH_PHRASES):
            return True
        gc = self.graph_ctx
        try:
            if gc is not None and gc.has_nodes() and gc.mentions_entity(question):
                return True
        except Exception:
            pass
        return False

    # A short reply right after a GRAPH turn ("what are the scores?", "which
    # ones?", "and the second one?") is almost never a topic change — it is
    # someone drilling into the answer they were just given. The bug this
    # guards against: "what are the scores?" contains no _GRAPH_PHRASES
    # entry and no on-map entity name, so it fell through to the 4-token LLM
    # classifier, which read the bare word "scores" as sports trivia and
    # misrouted it OFFTOPIC — losing the conversation thread entirely. The
    # old "[Previous turn: GRAPH]" hint was only ever advisory text inside
    # that same LLM call; it did not stop the misread. This gate is a hard
    # rule the LLM never gets a vote on, but it only fires for genuinely
    # short questions, and backs off the moment the question clearly asks
    # for something else (a new advice topic, or an unrelated OFFTOPIC ask)
    # so a real subject change right after a map answer isn't swallowed.
    SHORT_FOLLOWUP_MAX_WORDS = 6

    _ADVICE_SIGNAL_WORDS = (
        "release", "promot", "market", "mixing", "mastering", "royalt",
        "distribut", "audience", "gig", "label deal", "playlist pitch",
        "press kit", "album rollout", "merch", "tour",
    )
    _OFFTOPIC_SIGNAL_WORDS = (
        "weather", "recipe", "cook", "python", "sort a list", "function",
        "shopping", "best deal", "flight", "restaurant", "football",
        "basketball", "who won", "game score", "election", "stock price",
    )

    def _short_graph_followup(self, question, state):
        """True if this looks like a short drill-in on the previous GRAPH
        answer rather than a new topic. See SHORT_FOLLOWUP_MAX_WORDS above."""
        if state is None or state.last_intent != "GRAPH":
            return False
        q = _norm(question)
        if not q:
            return False
        if len(q.split()) > self.SHORT_FOLLOWUP_MAX_WORDS:
            return False
        if any(w in q for w in self._ADVICE_SIGNAL_WORDS):
            return False
        if any(w in q for w in self._OFFTOPIC_SIGNAL_WORDS):
            return False
        return True

    # A reformat/continuation request right after an ADVICE turn ("can you
    # put that in bullet points?", "shorten that", "tl;dr") is the same
    # class of bug as _short_graph_followup but in the other direction: the
    # shipped failure (live session, 2026-07-16) was "help me plan my
    # release" (correctly ADVICE) followed by "can you pu tthst in bullet
    # points for me?/" -- 9 words, no _GRAPH_PHRASES match, no on-map
    # entity -- which the 4-token LLM classifier misread as GRAPH, and the
    # graph agent then failed anchor resolution ("Nothing is selected on
    # your map..."), losing the advice thread entirely.
    #
    # Deliberately NOT a bare word-count rule like _short_graph_followup:
    # an ambiguous short question after ADVICE (e.g. "what are the
    # scores?") should still legitimately defer to the LLM, since it could
    # be a genuine pivot to the map.
    #
    # NOTE: reformat/continuation requests ("put that in bullet points",
    # "reformat your answer please") are intercepted in chat() BEFORE
    # classify_intent ever runs -- see _is_reformat_request / REFORMAT_SYSTEM
    # below. A fixed phrase list used to live here and force ADVICE
    # continuity, but that is exactly the whack-a-mole pattern this whole
    # fix is meant to replace: "reformat your answer please" wasn't
    # literally on the list, fell through to the LLM classifier, and got
    # misread as a GRAPH question. The replacement detector is
    # combinatorial (style-verb + reference-token) instead of enumerated,
    # and it owns rewriting the previous answer directly rather than
    # re-deriving fresh content through ADVICE/RAG.

    def classify_intent(self, question, state=None):
        """Three-way routing (ADVICE / GRAPH / OFFTOPIC), graph-aware.

        1. Unambiguous map language or an on-map entity → GRAPH outright.
        2. A short follow-up right after a GRAPH turn, with no clear sign of
           a topic change → GRAPH outright (see _short_graph_followup).
        3. Otherwise the LLM decides, but it is TOLD what is on the map so it
           can recognise a bare artist/song name as a graph entity instead of
           guessing (the old blind router sent "what is Big On Big connected
           to" to OFFTOPIC because it read it as geography).
        The previous turn's route is also passed to the LLM as a soft hint so
        longer follow-ups still lean the right way even when no hard rule
        above fires.

        Reformat/continuation requests on the previous ADVICE answer never
        reach this classifier -- chat() intercepts them first (see
        _is_reformat_request).
        """
        if self._graph_signal(question):
            return "GRAPH"

        if self._short_graph_followup(question, state):
            return "GRAPH"

        hint = ""
        if state is not None and state.last_intent in ("GRAPH", "ADVICE"):
            hint = f"[Previous turn: {state.last_intent}]\n"
        census = ""
        try:
            if self.graph_ctx is not None:
                census = self.graph_ctx.summary()
        except Exception:
            census = ""
        if census:
            hint += (f"[On the map right now: {census}]\n"
                     "[If the question is about any of those, or about the map, it is GRAPH.]\n")
        messages = [{"role": "system", "content": INTENT_SYSTEM}]
        messages += INTENT_FEWSHOT
        messages += [{"role": "user", "content": hint + question}]
        for _ in range(2):
            raw = self._generate(messages, max_new_tokens=4, temperature=0.0)
            label = self._normalize_intent(raw)
            if label in VALID_INTENTS:
                return label
        return "ADVICE"

    # Reformat/continuation detector: combinatorial (style-verb x
    # reference-token), not an enumerated phrase list -- the same pattern
    # already used successfully by _synthesis_signal. A phrase list missed
    # "reformat your answer please" because that exact wording wasn't in
    # it; requiring a verb from one small set AND a reference to "that
    # answer" from another generalizes to wording neither list anticipated
    # (see tests/test_reformat_request.py for the literal repro cases).
    _REFORMAT_VERBS = (
        "reformat", "format", "rephrase", "reword", "restate", "shorten",
        "condense", "simplify", "summarize", "summarise", "bullet",
        "bulletize", "list out", "clean up", "polish", "tighten", "redo",
        "make",  # only counts combined with a _REFORMAT_STYLE_WORD below
    )
    # Style adjectives/adverbs that pair with a bare "make" ("make your
    # response clearer", "make that shorter") -- checked independently of
    # word order so "make X clearer/shorter" matches regardless of what
    # sits between "make" and the adjective.
    _REFORMAT_STYLE_WORDS = (
        "shorter", "longer", "punchier", "clearer", "simpler", "cleaner",
        "tighter", "snappier",
    )
    _REFORMAT_REFERENTS = (
        "that", "this", "it", "your answer", "your response", "the answer",
        "the response", "that answer", "that response", "previous answer",
        "last answer", "what you said", "what you just said",
    )
    # Strong standalone phrases: sufficient on their own, no referent needed.
    _REFORMAT_STRONG_PHRASES = (
        "bullet points", "bullet point", "in bullets", "as a list",
        "in a list", "tl;dr", "tldr", "in plain english", "in fewer words",
    )

    def _is_reformat_request(self, question, state):
        """True if this looks like a request to restyle the PREVIOUS ADVICE
        answer (bullet points, shorter, reworded, ...), as opposed to a new
        question. Requires state.last_intent == "ADVICE" and a stored prior
        answer to restyle; backs off if the question actually names the map
        or an on-map entity, or reads as clearly off-topic, so a genuine
        pivot still wins.
        """
        if state is None or state.last_intent != "ADVICE":
            return False
        if not state.history:
            return False
        q = _norm(question)
        if not q:
            return False
        if not any(h.get("role") == "assistant" for h in state.history):
            return False
        has_referent = any(r in q for r in self._REFORMAT_REFERENTS)
        verb_signal = (
            any(v in q for v in self._REFORMAT_VERBS if v != "make")
            or ("make" in q.split() and any(s in q for s in self._REFORMAT_STYLE_WORDS))
        )
        signal = any(p in q for p in self._REFORMAT_STRONG_PHRASES) or (
            verb_signal and has_referent
        )
        if not signal:
            return False
        # Back off if the question actually names the map or an on-map
        # entity -- a real pivot to GRAPH must still win.
        if self._graph_signal(question):
            return False
        if any(w in q for w in self._OFFTOPIC_SIGNAL_WORDS):
            return False
        return True

    def _last_assistant_answer(self, state):
        for turn in reversed(state.history):
            if turn.get("role") == "assistant" and turn.get("content", "").strip():
                return turn["content"].strip()
        return None

    def _synthesis_signal(self, question):
        """Deterministic pre-filter: explicit brainstorm/draft/plan language
        is SYNTHESIS regardless of retrieval score. See _SYNTHESIS_SIGNAL_PHRASES
        / _SYNTHESIS_VERBS / _SYNTHESIS_NOUNS -- a fixed-phrase match OR a
        verb+noun combination (so "draft a 4-week plan for my EP" hits this
        without needing every exact wording enumerated)."""
        q = _norm(question)
        if any(p in q for p in _SYNTHESIS_SIGNAL_PHRASES):
            return True
        has_verb = any(v in q for v in _SYNTHESIS_VERBS)
        has_noun = any(n in q for n in _SYNTHESIS_NOUNS)
        return has_verb and has_noun

    def _normalize_mode(self, raw):
        if not raw:
            return None
        m = re.search(r"\b(GROUNDED|SYNTHESIS|NO_RETRIEVAL)\b", raw.upper())
        return m.group(1) if m else None

    def classify_retrieval_mode(self, question, max_score, hits):
        """Route an ADVICE question to GROUNDED / SYNTHESIS / NO_RETRIEVAL.

        Replaces the old binary `in_scope = intent=="ADVICE" and max_score
        >= SCOPE_THRESHOLD` gate, whose failure branch was the SAME
        off-topic redirect used for genuinely off-topic questions -- so a
        brainstorm/plan request that didn't happen to match one passage
        well got told "that's not my lane" instead of getting help. See the
        VALID_RETRIEVAL_MODES comment block above for the literature this
        mirrors (Self-RAG / Adaptive-RAG / CRAG).

        The explicit brainstorm/plan/draft phrasing is caught deterministically
        (mirrors _graph_signal / _short_graph_followup above); everything else
        goes to the same lightweight LLM classifier pattern classify_intent
        uses, with a safety-net fallback if the LLM misfires: quote a strong
        match if one exists, otherwise answer without one -- never redirect.
        """
        if self._synthesis_signal(question):
            return "SYNTHESIS"
        messages = [{"role": "system", "content": RETRIEVAL_MODE_SYSTEM}]
        messages += RETRIEVAL_MODE_FEWSHOT
        messages += [{"role": "user", "content": question}]
        for _ in range(2):
            raw = self._generate(messages, max_new_tokens=4, temperature=0.0)
            label = self._normalize_mode(raw)
            if label in VALID_RETRIEVAL_MODES:
                return label
        return "GROUNDED" if (hits and max_score >= SCOPE_THRESHOLD) else "NO_RETRIEVAL"

    # ---- dedicated 'sounds like' readout -----------------------------------
    def sounds_like(self, audio_path=None, analysis=None, top_k=8):
        """Natural-language readout of what a track sounds like.

        Pass an audio_path to embed+place, or a precomputed ``analysis`` dict
        {neighbors, cluster, tags}. RAG is skipped here — the grounding is the
        neighbour list itself, not the advice corpus.
        """
        if analysis is None:
            if audio_path is None:
                raise ValueError("give analysis or audio_path")
            analysis = self._analyze_audio(audio_path, top_k=top_k)
        lines = []
        neigh = analysis.get("neighbors", [])[:5]
        if neigh:
            lines.append("Nearest released tracks:")
            for n in neigh:
                lines.append(f"  - {n['artist']} — {n['name']}")
        tags = analysis.get("tags", [])
        if tags:
            lines.append("Micro-genre tags: " + ", ".join(t["genre"] for t in tags[:4]))
        c = analysis.get("cluster") or {}
        if c.get("label"):
            lines.append(f"Sits in the '{c['label']}' territory.")
        snd = "\n".join(lines)
        user = (
            "A developing artist ran their track through my similarity engine. "
            "Here is the analysis — the nearest released tracks and the "
            "micro-genre tags it matched:\n\n" + snd +
            "\n\nIn your mentor voice, tell them what they sound like based on "
            "these matches, name a couple of the comparison artists, and give "
            "one concrete thing to lean into. Do not invent other artists.")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=260, temperature=0.5)

    def _analyze_audio(self, audio_path, top_k=8):
        vec, _ = self.similarity.resolve(audio_path)
        return {
            "neighbors": self.similarity.similar(vec, top_k=top_k),
            "cluster": self.similarity.cluster(vec),
            "tags": self.similarity.tags(vec),
        }

    # ---- the agent loop ----------------------------------------------------
    def chat(self, question, audio_path=None, top_k=3, verbose=False,
             conversation_context=None, selected_node_id=None, graph_session_id=None):
        state = conversation_context if conversation_context is not None else self._session_state
        state.selected_node_id = selected_node_id
        if graph_session_id is not None:
            state.graph_session_id = graph_session_id
        # Re-point the shared GraphContext at THIS browser's map for the turn.
        # The mentor process serves many browsers; each /chat carries its own
        # graph_session_id, and the graph lives at ui/session/<sid>/graph.json.
        if self.graph_ctx is not None and state.graph_session_id is not None:
            try:
                self.graph_ctx.for_session(state.graph_session_id)
            except Exception:
                pass
        trace = verbose or TRACE
        if trace:
            gc = self.graph_ctx
            census = ""
            try:
                census = gc.summary() if gc is not None else ""
            except Exception:
                census = "<summary failed>"
            print(f"\n[TRACE] QUESTION: {question}")
            print(f"[TRACE] SESSION: graph={state.graph_session_id} "
                  f"selected={selected_node_id}")
            print(f"[TRACE] MAP: {census or '<empty>'}")

        ql = (question or "").strip().lower()
        if ql in ("/reset", "/clear", "/new"):
            state.reset()
            return "Session reset. Ask about your music and I'll keep context from there."

        # EARS: audio present -> this is a 'what do I sound like' turn; the
        # neighbour list is the grounding, so route to the dedicated readout
        # rather than the RAG advice branch.
        if audio_path and self.similarity is not None:
            analysis = self._analyze_audio(audio_path, top_k=8)
            out = self.sounds_like(analysis=analysis)
            state.last_intent = "GRAPH"
            state.record("user", question or f"/sound {audio_path}")
            state.record("assistant", out)
            return out

        # Reformat/continuation request on the previous ADVICE answer
        # ("put that in bullet points", "reformat your answer please"):
        # intercepted BEFORE classify_intent so it can never be misrouted
        # to GRAPH/OFFTOPIC, and answered by restyling the actual stored
        # previous answer -- never by re-deriving fresh content through
        # RAG or the graph agent (which was the second failure mode this
        # guards against: a verbatim repeat of the same passage).
        if self._is_reformat_request(question, state):
            prev = self._last_assistant_answer(state)
            if prev is not None:
                messages = [
                    {"role": "system", "content": REFORMAT_SYSTEM},
                    {"role": "user", "content": (
                        f"[Your previous answer]\n{prev}\n\n"
                        f"[Restyling request]\n{question}")},
                ]
                out = self._generate(messages, max_new_tokens=420)
                state.last_intent = "ADVICE"
                state.record("user", question)
                state.record("assistant", out)
                if trace:
                    print(f"[TRACE] CLASSIFICATION: ADVICE (reformat intercept)")
                return out

        intent = self.classify_intent(question, state=state)
        if trace:
            print(f"[TRACE] CLASSIFICATION: {intent}")

        if intent == "GRAPH":
            if self.agent is None:
                out = (
                    "I can do sound-map reads, but the graph agent isn't loaded in "
                    "this session. Ask for advice, or restart with Anther enabled."
                )
                state.record("user", question)
                state.record("assistant", out)
                return out
            result = self.agent.run(question, context=state)
            if trace:
                obs = result.get("observation")
                print(f"[TRACE] TOOL CALL: {result.get('intent')}")
                print(f"[TRACE] OBSERVATION: ok={None if obs is None else obs.get('ok')} "
                      f"status={result.get('status')}")
                print(f"[TRACE] FINAL ANSWER: {result.get('answer','')[:200]}")
            out = result.get("answer", "Tell me an anchor artist or upload audio, and I'll map your sound.")
            state.record("user", question)
            state.record("assistant", out)
            return out

        if intent == "ADVICE":
            # Retrieve once; the retrieval-MODE decision (not a single
            # pass/fail score) decides what happens with the hits. See the
            # VALID_RETRIEVAL_MODES comment block for why this replaced the
            # old binary SCOPE_THRESHOLD gate.
            max_score = 0.0
            hits = []
            if self.rag is not None:
                hits, max_score = self.rag.retrieve(question, top_k=top_k)

            mode = self.classify_retrieval_mode(question, max_score, hits)
            strong_hits = [h for h in hits if h["score"] >= SCOPE_THRESHOLD]
            if verbose:
                print(f"intent=ADVICE  mode={mode}  max_score={max_score:.3f}  "
                      f"hits={[round(h['score'], 2) for h in hits]}")

            if mode == "GROUNDED" and strong_hits:
                # Direct, settleable question, strong passage match: inject
                # the source advice, defer to it (original behavior).
                block = "\n\n".join(f"- {h['advice']}" for h in strong_hits)
                ctx_block = state.recent_block()
                user_content = (
                    question +
                    ctx_block +
                    "\n\n[Source advice from experienced musicians — defer to this, "
                    "rephrase in your own voice]\n" + block)
                messages = [{"role": "system", "content": GROUNDED_SYSTEM},
                            {"role": "user", "content": user_content}]
                out = self._generate(messages)
                state.last_intent = "ADVICE"
                state.record("user", question)
                state.record("assistant", out)
                return out

            if mode == "SYNTHESIS":
                # Brainstorm/plan/draft: passages (if any) are optional
                # texture, not a script. Never quote-and-defer, never redirect.
                ctx_block = state.recent_block()
                user_content = question + ctx_block
                if hits:
                    block = "\n\n".join(f"- {h['advice']}" for h in hits[:top_k])
                    user_content += (
                        "\n\n[Related passages — inspiration only, do not quote "
                        "verbatim or just pick one]\n" + block)
                messages = [{"role": "system", "content": SYNTHESIS_SYSTEM},
                            {"role": "user", "content": user_content}]
                out = self._generate(messages, max_new_tokens=420)
                state.last_intent = "ADVICE"
                state.record("user", question)
                state.record("assistant", out)
                return out

            # mode == "NO_RETRIEVAL" (or GROUNDED with no strong hits after
            # all): on-topic ADVICE, but nothing in the corpus is worth
            # quoting. Answer from the mentor persona directly -- this is
            # NOT the off-topic redirect below; the question was already
            # classified ADVICE, so "not my lane" would be wrong here.
            ctx_block = state.recent_block()
            messages = [{"role": "system", "content": NO_RETRIEVAL_SYSTEM},
                        {"role": "user", "content": question + ctx_block}]
            out = self._generate(messages)
            state.last_intent = "ADVICE"
            state.record("user", question)
            state.record("assistant", out)
            return out

        # OFFTOPIC: genuinely out of scope. In-persona redirect, no RAG context.
        # Few-shot exemplars pin the behavior — a bare instruction is not
        # reliably followed at this size, but 2-3 examples make it stick.
        if any(t in ql for t in ["are you an ai", "ai assistant", "who made you", "who created you"]):
            out = (
                "Not the useful question. If you want better music results, tell me "
                "what you're making and where it's stuck, and I'll give you a concrete next move."
            )
            state.last_intent = "OFFTOPIC"
            state.record("user", question)
            state.record("assistant", out)
            return out
        messages = [{"role": "system", "content": UNGROUNDED_SYSTEM}]
        messages += UNGROUNDED_FEWSHOT
        messages += [{"role": "user", "content": question}]
        out = self._generate(messages, max_new_tokens=90, temperature=0.5)
        # Safety catch: never leak AI-assistant framing in off-topic redirects.
        if re.search(r"\b(ai assistant|as an ai|i am an ai|i'm an ai|created by)\b", out, re.I):
            out = (
                "That's outside my lane. I'm here to help with music: writing, "
                "production, release strategy, and audience growth. Tell me what "
                "you're working on and where you're stuck."
            )
        state.last_intent = "OFFTOPIC"
        state.record("user", question)
        state.record("assistant", out)
        return out


if __name__ == "__main__":
    m = MusicMentor()
    q = "I keep starting songs and never finishing them. How do I actually ship?"
    print("Q:", q, "\n")
    print(m.chat(q, verbose=True))
