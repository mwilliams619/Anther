# Graph-Continuity Misroute + RAG Scope-Gate Diagnosis and Fix

**Session scope:** two bugs surfaced in one screenshot + one follow-up description.
1. A GRAPH follow-up question ("what are the scores?") got misrouted to an OFFTOPIC
   sports redirect.
2. The RAG "in scope" gate in `mentor.py` had exactly one failure mode shared between
   "no good passage" and "genuinely off-topic," so brainstorm/plan-drafting questions on
   an on-topic subject either got a verbatim knowledgebase quote or a "not my lane"
   redirect — never a synthesized answer.

Both are now fixed in `mentor/mentor.py`, covered by 28 new tests (9 + 19), with zero
regressions against the pre-existing 202-test baseline (2 failed / 24 errored tests are
pre-existing and unrelated — confirmed via `git stash` bisection in an earlier session).

---

## 1. Graph-continuity misroute — root cause

**Screenshot transcript (verbatim turn sequence):**
1. "what's the closest song to tv off on the map?" → correct 2-song GRAPH answer
2. "what are the scores?" → **misrouted**: "I'm here for musicians' questions, not sports..."
3. "no the similarity score?" → correct answer (rescued by re-mentioning "score" alongside "similarity")

**Mechanism.** `mentor.py`'s `classify_intent()` has two tiers:

- **Tier 1 — `_graph_signal()`:** a fixed list of ~25 map-language phrases (`"on the map"`,
  `"connected to"`, `"why is this"`, …) or a mention of an on-map entity name. High
  precision, but "scores" / "similarity score" alone matches none of these phrases and
  names no entity.
- **Tier 2 — LLM classifier:** a 3-way ADVICE/GRAPH/OFFTOPIC call with a 4-token budget,
  given only a soft hint — `f"[Previous turn: {state.last_intent}]\n"` — prepended to the
  question. The hint is advisory text inside the same prompt the classifier is about to
  misjudge; it has no mechanism to force a decision.

"what are the scores?" fails Tier 1 (no phrase match, no entity mention) and falls to
Tier 2, where the small model reads the bare word "scores" as a sports reference and
returns OFFTOPIC. The soft hint doesn't prevent this because it's just more text for the
same classifier to weigh, not a rule that overrides it.

**Reproduced directly** (this session, before/after, scripted LLM forced to say OFFTOPIC
on both follow-ups to isolate the hint's effect):

```
BEFORE FIX (soft hint only):
  "what's the closest song to tv off on the map?" -> GRAPH
  "what are the scores?"                          -> OFFTOPIC   <- bug
  "no the similarity score?"                      -> OFFTOPIC   <- bug

AFTER FIX (hard short-followup gate):
  "what's the closest song to tv off on the map?" -> GRAPH
  "what are the scores?"                          -> GRAPH      <- fixed
  "no the similarity score?"                      -> GRAPH      <- fixed
```

### Fix

Added `_short_graph_followup(question, state)` to `mentor.py`, inserted as a hard rule
directly after `_graph_signal()` in `classify_intent()`, before the LLM is ever consulted:

```
if self._graph_signal(question):
    return "GRAPH"
if self._short_graph_followup(question, state):
    return "GRAPH"
# ... LLM classifier only reached if neither hard rule fires
```

Fires only when **all** of the following hold — a short question, right after a GRAPH
turn, with no signal of an actual topic change:

- `state.last_intent == "GRAPH"` (the previous turn was a graph answer)
- the question is ≤ `SHORT_FOLLOWUP_MAX_WORDS = 6` words
- it contains no `_ADVICE_SIGNAL_WORDS` (`"release"`, `"promot"`, `"mixing"`, `"royalt"`, …)
- it contains no `_OFFTOPIC_SIGNAL_WORDS` (`"weather"`, `"football"`, `"game score"`, …)

The word-budget and signal-word back-off keep this from swallowing genuine topic changes:
a 6-word cap plus explicit advice/off-topic vocabulary lets "how's my release going?"
(short, but names a real subject) and "what's today's game score?" (short, but a real
off-topic subject) both defer to the LLM as before. A defense-in-depth few-shot example
(`"[Previous turn: GRAPH]\nwhat are the scores?"` → `GRAPH`) was also added to
`INTENT_FEWSHOT` for the cases where the hard rule doesn't fire (long follow-ups) but the
hint should still lean the classifier the right way.

**Tests** (`tests/test_mentor_routing.py`, 9 new): the literal screenshot repro, "which
ones?", back-off after ADVICE, back-off with no previous intent, back-off on long
follow-ups, back-off on genuine advice-topic and off-topic-topic changes, and the helper
tested directly. All pass with the LLM stub scripted to answer OFFTOPIC — i.e. the fix
holds even when the underlying classifier would still misfire.

---

## 2. RAG scope-gate — root cause and literature-grounded diagnosis

### The gate as shipped

```python
in_scope = (intent == "ADVICE") and (max_score >= SCOPE_THRESHOLD)
if in_scope and hits:
    # GROUNDED: inject retrieved passages, "defer to this, rephrase in your own voice"
    ...
# else: falls through to the SAME branch used for genuinely OFFTOPIC questions
# "You are ... The user has asked something outside your lane ... Do NOT answer
#  the off-topic question."
```

`SCOPE_THRESHOLD = 0.63` is a single global cosine cutoff over a `max_score` that
represents how well the top-matching corpus passage matches the raw query text. This is
a **binary retrieve-and-quote gate with one failure branch**, and that failure branch is
identical to the off-topic redirect — even though `intent` was already `ADVICE`.

### Why this produces whack-a-mole symptoms

A direct, narrowly-scoped question ("how do I stop clipping when mastering?") tends to
have one clearly-matching corpus passage, so `max_score` clears 0.63 and the quote-and-
defer branch fires correctly. A **brainstorm or plan-drafting** question on the exact same
topic ("help me plan my release rollout," "brainstorm ideas for my album") is semantically
diffuse relative to any single stored passage — cosine similarity against a fixed corpus
of discrete tips is a poor proxy for "does this deserve a synthesized answer," so
`max_score` frequently lands under 0.63. The user's report — asking for advice on release
gets grounded advice, but brainstorming or drafting a plan gets either direct quotes or a
redirect — is exactly what a single retrieve-then-threshold gate produces on two different
question *shapes* asked about the same *topic*: no branch exists for "on-topic, wants
synthesis," so it collapses into whichever of the two available branches the threshold
happens to land on.

### Literature grounding

**Self-RAG** (Asai et al., ICLR 2024, arXiv:2310.11511) frames the core problem directly:
indiscriminately retrieving and incorporating a fixed number of passages regardless of
whether retrieval is necessary or the passages are relevant can diminish an LM's
versatility. Its fix is retrieval **on demand** — the official project summary states that
standard RAG retrieves a fixed number of times, "while Self-RAG enables Adaptive retrieval
... or completely skip retrieval" (selfrag.github.io) — direct precedent for a
`NO_RETRIEVAL` branch that is not the off-topic redirect. The paper also splits objectives
by task type in a way that maps onto exactly the distinction the user described: factual
tasks call for retrieving more to stay evidence-aligned, while open-ended tasks shift
toward "prioritizing the overall creativity or utility score" (arXiv:2310.11511) — grounded
advice vs. brainstorm/plan is the same split one level down.

**Adaptive-RAG** (Jeong, Baek, Cho, Hwang & Park, NAACL 2024, arXiv:2403.14403) supplies the
concrete routing mechanism: a small classifier predicts query complexity and dispatches to
one of several retrieval strategies rather than a single fixed policy, offering "a balanced
strategy, seamlessly adapting between... iterative and single-step... as well as the
no-retrieval methods" (arXiv:2403.14403). A later baseline study of the same design space
summarizes it as training a classifier "to route among three retrieval strategies" and
showing a three-class router "can match always-expensive baselines with substantially
lower cost" (arXiv:2604.03455). The 3-way mode split implemented here (GROUNDED /
SYNTHESIS / NO_RETRIEVAL) is a lighter-weight instance of this pattern: a deterministic
phrase/verb-noun pre-filter catches the unambiguous brainstorm/plan cases, and a small LLM
classifier (mirroring `classify_intent`'s existing pattern) handles the rest.

**CRAG / Corrective RAG** (Yan, Gu, et al., arXiv:2401.15884) motivates treating retrieval
quality as a *graded* signal rather than a pass/fail cutoff: a lightweight evaluator
assigns retrieved documents a confidence degree, triggering "different knowledge retrieval
actions" of Correct, Incorrect, or Ambiguous (arXiv:2401.15884). This is the precedent for
keeping `max_score` as a graded input to the mode decision (it still gates whether a
passage is "strong" enough to quote) rather than collapsing it back into a single binary
switch — the router here folds CRAG's three-way confidence grading and Adaptive-RAG's
three-way complexity routing into one 3-mode decision, since in this system the two
questions ("is retrieval good?" and "does the query want synthesis?") turned out to mostly
co-vary in the failure the user reported.

### Fix: 3-mode retrieval router

`classify_retrieval_mode(question, max_score, hits)` replaces the binary gate. ADVICE
questions now route to exactly one of:

| Mode | When | Behavior |
|---|---|---|
| **GROUNDED** | direct, settleable question + a passage clears `SCOPE_THRESHOLD` | quote-and-defer (original, unchanged behavior) |
| **SYNTHESIS** | brainstorm / draft / plan / outline language, regardless of passage score | passages (if any) offered as inspiration only; model reasons beyond them to produce an original answer |
| **NO_RETRIEVAL** | on-topic, but nothing scores well enough to quote | answers directly from the mentor persona — **never** the off-topic redirect |

Routing order:
1. **Deterministic pre-filter** (`_synthesis_signal`): fixed brainstorm/plan phrases, plus
   a verb+noun combination check (`draft`/`outline`/`sketch`/`workshop`/`brainstorm`/`map out`
   × `plan`/`ideas`/`options`/`strategy`/`roadmap`) so wording variants like "draft a
   4-week plan for my EP" hit this without enumerating every phrasing. This takes priority
   over passage score — an explicit brainstorm ask still gets SYNTHESIS even if one
   passage happens to score above threshold (tested explicitly).
2. **LLM classifier** (`RETRIEVAL_MODE_SYSTEM` + `RETRIEVAL_MODE_FEWSHOT`) for the
   remaining GROUNDED-vs-NO_RETRIEVAL distinction, mirroring `classify_intent`'s existing
   pattern (small model, 4-token budget, 2 retries).
3. **Safety-net fallback** if the LLM returns garbage on both retries: quote a strong
   match if `hits` cleared `SCOPE_THRESHOLD`, otherwise answer without one — never a
   redirect, since the question was already classified `ADVICE`.

`chat()`'s ADVICE branch now dispatches on `mode` into three distinct system prompts
(`GROUNDED_SYSTEM` unchanged, new `SYNTHESIS_SYSTEM`, new `NO_RETRIEVAL_SYSTEM`); the old
"quote-or-redirect" fallthrough that produced the reported symptom no longer exists —
`UNGROUNDED_SYSTEM` (the off-topic redirect) is now reachable only when `intent ==
"OFFTOPIC"`, never as an ADVICE fallback.

### Validation against reported and new queries

Direct reproduction of the reported two-question pattern (a real cosine-scored passage,
scripted LLM):

```
"What's the best strategy for releasing my next single?"        max_score=0.88  -> GROUNDED
"Can you help me brainstorm a full release plan for it?"        max_score=0.88  -> SYNTHESIS   (same passage; no longer quoted verbatim)
"Draft a 4-week rollout plan for my EP."                         max_score=0.05  -> SYNTHESIS   (was: redirect)
"Help me think through how to grow my audience this year."      max_score=0.40  -> SYNTHESIS   (was: redirect, borderline score)
"Do you think my sound has matured this year?"                  max_score=0.08  -> NO_RETRIEVAL (was: redirect; now answered on-topic)
```

**Tests** (`tests/test_retrieval_router.py`, 19 new, all passing): deterministic-filter
priority over passage score, LLM-classifier GROUNDED/NO_RETRIEVAL cases, safety-net
fallback on garbage LLM output, full `chat()` dispatch proving three distinct system
prompts/answer shapes, weak-hit GROUNDED→NO_RETRIEVAL fallback (never quotes an
irrelevant-scoring passage), OFFTOPIC unaffected, `last_intent` continuity across all
three ADVICE modes, and a direct two-turn reproduction of the user's reported pattern
(`test_release_advice_then_brainstorm_followup_both_land_on_topic`).

---

## 3. Regression status

```
Baseline (start of this task): 202 passed, 2 failed, 1 skipped, 24 errors
After graph-continuity fix:    211 passed, 2 failed, 1 skipped, 24 errors  (+9)
After retrieval router:        230 passed, 2 failed, 1 skipped, 24 errors  (+19)
```

The 2 failures (`test_features.py::test_extract_returns_labeled_series_in_canonical_order`,
`::test_align_to_corpus_reorders`) and 24 errors (`AttributeError:
<module 'atlas'> has no attribute '_graph'` in `test_atlas_graph.py` /
`test_atlas_playlist.py` / `test_atlas_search.py`) were confirmed pre-existing via
`git stash` bisection in an earlier session and are unrelated to this work.

---

## 4. Files changed

- `mentor/mentor.py` — `_short_graph_followup` + hard-rule insertion in `classify_intent`;
  `VALID_RETRIEVAL_MODES`, `SYNTHESIS_SYSTEM`, `NO_RETRIEVAL_SYSTEM`,
  `RETRIEVAL_MODE_SYSTEM`/`_FEWSHOT`, `_synthesis_signal`, `classify_retrieval_mode`;
  `chat()`'s ADVICE branch rewritten to dispatch on retrieval mode.
- `tests/test_mentor_routing.py` — 9 new short-follow-up tests.
- `tests/test_retrieval_router.py` — new file, 19 tests for the 3-mode router.

Full diff: `graph_continuity_and_rag_router.diff` (737 lines).

## References

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to
  Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. arXiv:2310.11511.
- Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). *Adaptive-RAG: Learning
  to Adapt Retrieval-Augmented Large Language Models through Question Complexity.*
  NAACL 2024. arXiv:2403.14403.
- Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). *Corrective Retrieval Augmented
  Generation.* arXiv:2401.15884.


---

## Addendum (2026-07-17): companion bug — ADVICE-continuity misroute

A second live-session report surfaced the mirror-image of the graph-continuity bug fixed
above. Turn sequence:

1. "help me plan my release" → correctly classified ADVICE, answered with release-plan
   advice.
2. "can you pu tthst in bullet points for me?/" → **misrouted to GRAPH**, and since
   nothing was selected on the map and no track was on hand, the graph agent's anchor
   resolution failed honestly: "Nothing is selected on your map and I don't have a track
   of yours from earlier. Click a node, or name an artist or song." — losing the advice
   thread entirely.

**Root cause.** This is the same structural gap as the original bug, just missing a
gate in the other direction: `classify_intent()` had a hard rule for short follow-ups
after a **GRAPH** turn (`_short_graph_followup`), but no equivalent rule for a
reformat/continuation request after an **ADVICE** turn. The message is 9 words — over
even the GRAPH-followup's 6-word cap — and contains no `_GRAPH_PHRASES` entry and no
on-map entity mention, so it fell straight to the 4-token LLM classifier, which read
"bullet points" as map/layout language and answered GRAPH.

**Fix.** Added `_short_advice_followup(question, state)` plus a
`_CONTINUATION_SIGNAL_PHRASES` list (`"bullet points"`, `"shorten that"`, `"tl;dr"`,
`"summarize that"`, `"put that in"`, `"format that"`, …) and inserted it as a third hard
rule in `classify_intent()`, checked after the GRAPH-continuity rules and before the LLM
call:

```
if self._graph_signal(question):
    return "GRAPH"
if self._short_graph_followup(question, state):
    return "GRAPH"
if self._short_advice_followup(question, state):
    return "ADVICE"
# ... LLM classifier only reached if none of the three hard rules fire
```

Deliberately **not** a bare word-count rule like `_short_graph_followup` — a short,
ambiguous question after ADVICE (e.g. "what are the scores?") could still be a genuine
pivot to the map, so it should keep deferring to the LLM. Only an explicit
reformat/continuation phrase forces ADVICE continuity, and the rule backs off the moment
the question actually names the map or an on-map entity (`_graph_signal` check inside
`_short_advice_followup`), so a real pivot to GRAPH still wins.

**Reproduced directly** (scripted LLM forced to answer GRAPH to isolate the gate's effect):

```
BEFORE FIX:
  "help me plan my release"                     -> ADVICE
  "can you pu tthst in bullet points for me?/"   -> GRAPH   <- bug

AFTER FIX:
  "help me plan my release"                      -> ADVICE
  "can you pu tthst in bullet points for me?/"   -> ADVICE  <- fixed
```

**Tests** (`tests/test_mentor_routing.py`, 8 new): the literal live repro, "shorten
that"/"tldr?" variants, back-off when the previous turn was GRAPH (a short reformat
right after GRAPH correctly stays GRAPH via the existing rule — continuing *that*
thread), back-off with no previous intent, back-off when the question explicitly names
the map, back-off on a short-but-non-continuation question (still defers to the LLM),
and the helper tested directly. Full suite: 246 passed (up from 230), same pre-existing
2 failed / 24 errors baseline — no regressions.

Updated diff: `advice_continuity_fix.diff` (cumulative session diff against
`mentor/mentor.py` + `tests/test_mentor_routing.py`, 494 insertions across both fixes).


---

## Addendum 2 (2026-07-17): reformat/continuation requests — root cause, fix, and tests

### What broke, again

Two related failures surfaced in a follow-on live transcript, both after an ADVICE
("help me plan my release") turn:

1. **"format that in bullet points please"** — the bot repeated the exact same
   paragraph back verbatim; no reformatting happened at all.
2. **"reformat your answer please. But good answer!"** — came back with a
   completely unrelated GRAPH-mode answer about a different track never mentioned
   in the conversation.

The first companion-bug fix (documented in Addendum 1) added a fixed
`_CONTINUATION_SIGNAL_PHRASES` list inside `classify_intent` to force ADVICE
continuity for known reformat phrasings. Failure 2 is exactly what that approach
was always going to produce eventually: "reformat your answer please" was not
literally on the list, so it fell through to the LLM classifier and got misread
as GRAPH — the same whack-a-mole pattern the user flagged from the start, just
recurring one layer deeper.

### Root cause

Both symptoms trace to one systemic gap: there was no dedicated pathway for
"restyle what you just told me." ADVICE (GROUNDED/SYNTHESIS/NO_RETRIEVAL),
GRAPH, and OFFTOPIC all *re-derive* content — from a retrieved passage, from
graph tools, or not at all — none of them operates on the actual stored
previous answer. `GROUNDED_SYSTEM`'s prompt always regenerates from the
retrieved passage with no "restyle" concept, which explains the verbatim-repeat
failure directly. And because the continuity gate was an enumerated phrase
list rather than a generalizable rule, any wording outside that list reverted
to full LLM classification with all its usual failure modes, which explains
the GRAPH misfire.

### Fix

- Removed `_CONTINUATION_SIGNAL_PHRASES` / `_continuation_signal` /
  `_short_advice_followup` from `classify_intent` entirely. `classify_intent`
  no longer special-cases reformat phrasing — that responsibility moved
  upstream, out of the router.
- Added `REFORMAT_SYSTEM`, a system prompt instructing the model to restyle
  the previous answer — preserve all facts/advice, change only
  structure/length/tone, never just repeat it unchanged.
- Added `_is_reformat_request(question, state)`: a **combinatorial** detector
  (style-verb × reference-token, or a small set of strong standalone phrases
  like "bullet points"/"tl;dr"), the same pattern already used successfully
  for `_synthesis_signal`. True only if the previous turn was ADVICE, history
  contains an assistant turn, and the phrasing signals a restyle request; it
  backs off if the question names the map/an on-map entity or reads as
  off-topic, so a genuine pivot still wins. Word-order-robust: "make your
  response clearer" and "make that shorter" both fire via a shared
  `make` + `_REFORMAT_STYLE_WORDS` check rather than requiring the exact
  bigram.
- Added `_last_assistant_answer(state)` to retrieve the actual stored prior
  answer.
- Wired into `chat()` *before* `classify_intent` is ever called: on a match,
  builds a two-part prompt (`[Your previous answer]` + `[Restyling request]`)
  against `REFORMAT_SYSTEM`, generates, sets `last_intent = "ADVICE"`, and
  returns directly — bypassing RAG and the graph agent entirely, so a
  reformat request can never be misrouted to GRAPH/OFFTOPIC and never
  silently re-derives (and thus repeats) the same content.

### Tests

`tests/test_reformat_request.py` (15 tests, new file): unit tests on
`_is_reformat_request` (including the literal "reformat your answer please.
But good answer!" repro, five previously-unlisted phrasings such as "make
your response clearer" and "polish that answer for me", and backoff cases —
no prior ADVICE turn, no history, explicit map pivot), plus `chat()`-level
tests proving the interception fires before `classify_intent`, the previous
answer is actually included in the restyling prompt, `last_intent` stays
ADVICE for further continuity, and a genuine map-pivot question after ADVICE
still reaches the graph agent unimpeded.

`tests/test_mentor_routing.py` was updated (4 tests replaced/renamed) to
reflect that `classify_intent` alone no longer special-cases reformat
phrasing — that assertion now lives in `test_reformat_request.py` instead.

Full suite: **257 passed**, same baseline 2 failed / 24 errored (both
pre-existing and unrelated to mentor.py, confirmed via `git stash` earlier in
this investigation) / 1 skipped.

### Live-transcript validation

Reproduced both reported turns end-to-end against a scripted mentor:

| Turn | Question | Before | After |
|---|---|---|---|
| 2 | "format that in bullet points please" | Verbatim repeat of turn 1 | Genuinely restyled (bulleted) answer via `REFORMAT_SYSTEM` |
| 3 | "reformat your answer please. But good answer!" | Unrelated GRAPH-mode answer | Restyled answer via `REFORMAT_SYSTEM`, `last_intent` stays ADVICE |

### Files changed

- `mentor/mentor.py` — removed phrase-list gate; added `REFORMAT_SYSTEM`,
  `_is_reformat_request`, `_last_assistant_answer`, and the `chat()`
  interception.
- `tests/test_mentor_routing.py` — migrated 4 tests off the removed
  phrase-list architecture.
- `tests/test_reformat_request.py` — new, 15 tests.

Diff: [reformat_request_fix.diff](reformat_request_fix.diff)
