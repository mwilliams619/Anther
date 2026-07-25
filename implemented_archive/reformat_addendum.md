

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
