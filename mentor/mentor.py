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
from graph_context import MentorContext

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
    {"role": "user", "content": "Write me a python function to sort a list."},
    {"role": "assistant", "content": "OFFTOPIC"},
    {"role": "user", "content": "Are you an AI?"},
    {"role": "assistant", "content": "OFFTOPIC"},
]

VALID_INTENTS = ("ADVICE", "GRAPH", "OFFTOPIC")


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
            try:
                from graph_context import GraphContext
                self.graph_ctx = GraphContext()
            except Exception:
                self.graph_ctx = None
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

    def classify_intent(self, question, state=None):
        """LLM three-way routing. The previous turn's route is passed as a hint
        so short follow-ups ("which ones?") stay on the graph branch. Retries
        once at temperature 0; if the model still emits nothing usable we
        default to ADVICE (the RAG scope gate below handles the rest)."""
        hint = ""
        if state is not None and state.last_intent in ("GRAPH", "ADVICE"):
            hint = f"[Previous turn: {state.last_intent}]\n"
        messages = [{"role": "system", "content": INTENT_SYSTEM}]
        messages += INTENT_FEWSHOT
        messages += [{"role": "user", "content": hint + question}]
        for _ in range(2):
            raw = self._generate(messages, max_new_tokens=4, temperature=0.0)
            label = self._normalize_intent(raw)
            if label in VALID_INTENTS:
                return label
        return "ADVICE"

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
             conversation_context=None, selected_node_id=None):
        state = conversation_context if conversation_context is not None else self._session_state
        state.selected_node_id = selected_node_id
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

        intent = self.classify_intent(question, state=state)

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
            if verbose:
                print(f"graph intent={result.get('intent')} status={result.get('status')}")
            out = result.get("answer", "Tell me an anchor artist or upload audio, and I'll map your sound.")
            state.record("user", question)
            state.record("assistant", out)
            return out

        # Retrieve, and get the top cosine over the whole index to decide scope.
        max_score = 0.0
        hits = []
        if self.rag is not None:
            hits, max_score = self.rag.retrieve(question, top_k=top_k)

        in_scope = (intent == "ADVICE") and (max_score >= SCOPE_THRESHOLD)
        if verbose:
            print(f"intent={intent}  max_score={max_score:.3f}  in_scope={in_scope}  "
                  f"hits={[round(h['score'], 2) for h in hits]}")

        if in_scope and hits:
            # GROUNDED branch: inject the source advice, defer to it.
            block = "\n\n".join(f"- {h['advice']}" for h in hits)
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

        # UNGROUNDED branch: out of scope. In-persona redirect, no RAG context.
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
