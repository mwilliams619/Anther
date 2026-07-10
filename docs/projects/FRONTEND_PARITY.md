## Plan: Frontend Capability Parity Wiring

Close the UI/backend gap by wiring currently latent atlas recommendation features and mentor/LLM graph-chat capabilities into the web app, while preserving existing atlas behavior. The recommended path is phased: expose stable mentor APIs first, then add UI surfaces, then harden ReAct tool execution and session continuity.

**Steps**
1. Baseline feature-gap inventory and acceptance matrix (*done; blocks 2-6*)
- Confirm current UI-wired features vs latent backend/mentor features.
- Freeze a parity checklist with explicit acceptance criteria for each missing frontend capability.

2. Add backend API surface for mentor chat in Flask (*depends on 1*)
- Add a mentor chat route in /home/matt/Dev/Anther/ui/app.py that accepts user query + session identifier and returns answer + route metadata (intent, optional graph trace summary).
- Instantiate/load a long-lived mentor runtime once at process start (do not load per request).
- Add robust error handling and timeouts suitable for model-backed calls.

3. Add conversation-state storage for multi-turn web chat (*depends on 2*)
- Create a lightweight session store under /home/matt/Dev/Anther/ui/session/ (in-memory + persisted fallback) to map session_id -> ConversationState.
- Thread ConversationState into mentor chat calls so follow-up requests can reuse prior graph observations/anchors.
- Define expiry/reset semantics and explicit reset endpoint/flag behavior.

4. Expose latent recommendation behavior in frontend controls (*parallel with 2-3, integration depends on 2 if shared UI panel*)
- Add recommendation trigger(s) in /home/matt/Dev/Anther/ui/static/index.html and client handlers in /home/matt/Dev/Anther/ui/static/app.js.
- Wire client calls to existing /api/recommend endpoint with seed_ids, top_k, method.
- Integrate recommendation responses into graph/map list UX (selection, add/zoom, error states).

5. Add mentor chat UI surface and client orchestration (*depends on 2-3*)
- Add chat panel/components in /home/matt/Dev/Anther/ui/static/index.html.
- Add mentor chat client flow in /home/matt/Dev/Anther/ui/static/app.js (send message, render assistant output, maintain session_id, reset state).
- Define behavior for graph-specific responses (optional expandable trace: called tools, resolved anchors, neighbor summaries) without exposing raw internals by default.

6. Align mentor ReAct execution with declared toolset (*depends on 2; can begin parallel with 4-5*)
- Update /home/matt/Dev/Anther/mentor/mentor_react.py _execute() dispatch to either:
  - Match only the intentionally supported tool subset and trim prompt/tool declarations, or
  - Fully dispatch all tools declared in REACT_SYSTEM and mentor_graph TOOLS registry.
- Ensure context is passed into tool calls requiring follow-up resolution continuity.
- Add validation/normalization for tool args before execution to reduce unknown_tool churn.

7. Optional direct-tool API layer for debug/admin workflows (*parallel with 6; excluded from core MVP if desired*)
- Add guarded endpoints in /home/matt/Dev/Anther/ui/app.py for direct mentor graph tool invocation when needed for testing and demos.
- Keep this behind debug/development flags to avoid expanding public surface area unintentionally.

8. Docs and contract updates (*depends on 2-6*)
- Update /home/matt/Dev/Anther/docs/ui.md with new routes, UI surfaces, and session semantics.
- Update /home/matt/Dev/Anther/mentor/README.md to document web integration path and runtime expectations.
- Record API request/response contracts for frontend consumers.

9. Verification and rollout gates (*depends on 4-8*)
- Add/extend tests for Flask routes, mentor session continuity, and recommendation UI wiring.
- Run manual end-to-end checks: recommendation from selected map seeds, mentor multi-turn follow-ups, reset behavior, and error handling when mentor model unavailable.
- Validate startup/runtime impact (mentor model warm load, memory/VRAM headroom, responsiveness under concurrent requests).

**Relevant files**
- /home/matt/Dev/Anther/ui/app.py — add mentor chat API route(s), initialize shared mentor runtime, maintain request contracts; /api/recommend already exists and is latent from frontend.
- /home/matt/Dev/Anther/ui/static/index.html — add recommendation controls and mentor chat panel markup.
- /home/matt/Dev/Anther/ui/static/app.js — connect frontend actions to /api/recommend and mentor chat routes; session_id and UX state handling.
- /home/matt/Dev/Anther/ui/static/graph.js — optional hooks for recommendation highlighting/focus behavior.
- /home/matt/Dev/Anther/ui/atlas.py — reuse recommend(seed_ids, top_k, method, splice) contract; avoid duplicating ranking logic in frontend.
- /home/matt/Dev/Anther/mentor/mentor.py — reuse MusicMentor.chat and ConversationState threading for web requests.
- /home/matt/Dev/Anther/mentor/mentor_react.py — align REACT_SYSTEM tool declarations with _execute dispatch; pass context through tool execution.
- /home/matt/Dev/Anther/mentor/mentor_graph.py — source of deterministic tool methods and TOOLS schema metadata to align API/tool contracts.
- /home/matt/Dev/Anther/mentor/mentor_chat.py — reference interaction semantics for reset/follow-up behavior (CLI baseline).
- /home/matt/Dev/Anther/docs/ui.md — document new frontend-available capabilities and route map.
- /home/matt/Dev/Anther/mentor/README.md — document web chat integration/runtime constraints.

**Verification**
1. API contract checks
- Confirm /api/recommend is invoked by frontend with valid seed_ids from current graph selections.
- Confirm mentor chat endpoint returns stable schema for success/failure and includes session continuity markers.

2. Functional UI checks
- Seed 1+ tracks on map, trigger recommendations, verify returned items are visible/actionable on graph.
- Run a 3-turn mentor conversation with a graph follow-up query and confirm contextual continuity.
- Run reset flow and confirm subsequent turn is stateless/fresh.

3. ReAct/tool integrity checks
- Validate every tool exposed in prompt is either executable or removed from prompt declarations.
- Validate context propagation in graph tools that rely on prior anchors/follow-ups.

4. Reliability/performance checks
- Verify mentor model loads once and remains warm; measure first-token and subsequent-token latency.
- Verify behavior when mentor runtime is unavailable (clear UI error and degraded atlas-only operation).

**Decisions**
- Included scope
- Frontend parity for latent recommendation behavior.
- Frontend availability of mentor LLM chat with multi-turn continuity.
- Backend/tool alignment needed to make declared graph capabilities dependable in web use.

- Excluded scope (unless explicitly requested)
- Major visual redesign of atlas UI beyond minimum controls/panels needed for new capabilities.
- Training/fine-tuning/model-quality work in mentor weights.
- Public internet hardening/auth for mentor endpoints (assume trusted/local deployment unless specified).

**Further Considerations**
1. Tool exposure strategy
- Option A: keep ReAct prompt to the currently supported subset (faster, safer).
- Option B: fully implement all declared tools (richer capability, more test burden).
- Recommendation: Option A for MVP, then phase to B.

2. Session store depth
- Option A: in-memory only (simple, no restart persistence).
- Option B: persisted store in /home/matt/Dev/Anther/ui/session/ (better continuity across restarts).
- Recommendation: Option B with TTL.

3. Recommendation UX trigger
- Option A: explicit “Recommend from selected seeds” control.
- Option B: auto-recommend after each placement.
- Recommendation: Option A to keep map changes intentional and debuggable.
