## Plan: Audit and trim bloat in `anther_ml` and `mentor`

Audit only the Python scripts under `/home/matt/Dev/Anther/anther_ml` and `/home/matt/Dev/Anther/mentor`; ignore markdown/docs. The goal is to identify the highest-bloat modules, explain why they are bloated, and outline small behavior-preserving refactors that reduce duplication and control-flow density without changing observable outputs.

**Steps**
1. Rank the code by practical bloat score, with separate lists for `anther_ml` and `mentor`.
2. For each top candidate, capture: current responsibility, specific bloat sources, and the smallest safe simplification target.
3. Prefer refactors that extract repeated parsing/normalization/dispatch logic into shared helpers before touching larger orchestration functions.
4. Keep any proposed refactors file-local or module-local first; avoid broad cross-package rewrites unless duplication is clearly repeated in multiple places.
5. Use characterization tests or existing regression coverage for any future implementation step, especially for parser, routing, and embedding code.

**Relevant files**
- `/home/matt/Dev/Anther/anther_ml/corpus/build.py` — largest and densest corpus-builder; checkpoint management, dedupe, and profile generation are the main bloat sources.
- `/home/matt/Dev/Anther/anther_ml/cluster.py` — duplicated clustering paths and repeated diagnostics.
- `/home/matt/Dev/Anther/anther_ml/mpd_sql.py` — extended-INSERT parsing is split across multiple near-overlapping helpers.
- `/home/matt/Dev/Anther/anther_ml/spotify_deezer.py` — matching, retry, and waveform handling are interleaved.
- `/home/matt/Dev/Anther/anther_ml/embedding.py` — window planning and waveform preparation are split across several call paths.
- `/home/matt/Dev/Anther/anther_ml/corpus/sources.py` — repeated sampling/capping logic across sources.
- `/home/matt/Dev/Anther/anther_ml/eval.py` — title-stem normalization, ranking, and HTML export are somewhat duplicated.
- `/home/matt/Dev/Anther/anther_ml/corpus/labels.py` — normalization and profile mutation are a moderate cleanup target.
- `/home/matt/Dev/Anther/mentor/mentor.py` — primary orchestration hotspot; intent routing and chat branch handling are concentrated here.
- `/home/matt/Dev/Anther/mentor/mentor_graph.py` — repeated tool schemas and passthrough wrappers dominate the file.
- `/home/matt/Dev/Anther/mentor/mentor_react.py` — fallback call dispatch and prompt wiring are the main complexity source.
- `/home/matt/Dev/Anther/mentor/mentor_anther.py` — anchor resolution is complex but mostly necessary; thin wrappers and lookup setup are the small cleanup targets.

**Verification**
1. Re-check the exact control-flow and helper boundaries in the ranked files before any future edits.
2. Use narrow, file-scoped tests for parser/routing/embedding behavior if implementation starts.
3. Prefer targeted runtime checks over broad suite runs when validating individual simplifications.

**Decisions**
- Scope is limited to Python scripts in `anther_ml` and `mentor`.
- Markdown/docs are out of scope.
- No behavior changes are assumed; any refactor must preserve current outputs and public interfaces.
- Highest-priority cleanup candidates are `anther_ml/corpus/build.py`, `mentor/mentor.py`, `mentor/mentor_graph.py`, and `anther_ml/cluster.py`.

**Further Considerations**
1. If you want, the next pass can turn this into a file-by-file refactor backlog with estimated risk and effort.
2. If you want implementation instead of audit only, the safest first cuts are `mentor/mentor_graph.py` tool-schema extraction and `anther_ml/corpus/build.py` checkpoint/dedupe simplification.
