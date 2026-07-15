**refactor rather than delete** the existing files so you preserve working behavior.

```markdown
# Refactor Anther LLM Graph Agent Architecture

## Objective

Refactor the current LLM graph interaction system so the LLM interacts with a clean music-mentor API instead of directly reasoning over backend ML/corpus functions.

The current implementation overexposes implementation details:
- MERT embeddings
- corpus placement
- tag inheritance
- cluster inference
- similarity search

The LLM should not know these exist.

The product experience is:
> "A music mentor sitting inside an interactive spatial sound map."

The agent should reason about:
- what is currently visible
- selected nodes
- relationships between songs/artists
- exploration paths
- why songs are close
- where to explore next

---

# Current Architecture Problems

Current flow:
```

User question
|
v
MentorReAct
|
v
MentorGraphTools
|
v
AntherSoundsLike
|
v
Corpus / embeddings

```
Problems:

1. The LLM is choosing backend operations instead of music concepts.

2. The graph state is not a first-class input.

3. `sounds_like` is overloaded:
   - resolve anchor
   - search corpus
   - infer cluster
   - infer tags
   - format explanation

4. ReAct loop adds unnecessary failure modes because deterministic classification already exists.

5. Tool outputs contain too much ML detail and not enough musical interpretation.

---

# Desired Architecture

Implement:
```

User question
|
v
MentorAgent
|
v
Intent classifier
|
v
MentorGraphTools
|
v
GraphContext + AntherSoundsLike
|
v
Evidence
|
v
LLM narrator

```
The LLM should only:
- interpret the user's intent
- phrase answers naturally

The tools should:
- retrieve facts
- never hallucinate
- expose music concepts

---

# Part 1: Refactor MentorGraphTools

Create a clean public API.

The LLM-facing tools should be:

## 1. inspect_graph()

Purpose:
Allow the agent to understand the current map.

Returns:

```json
{
  "selected_node": {
    "name": "",
    "artist": ""
  },
  "visible_nodes": 120,
  "clusters": [],
  "recent_uploads": []
}
```

------

## 2. resolve_anchor(anchor)

Purpose:
Resolve a user reference.

Examples:

"me"
"this song"
"Halo"
"Burial"

Resolution order:

1. selected graph node
2. visible graph nodes
3. uploaded songs
4. frozen corpus

Return:

```json
{
 "resolved": true,
 "type": "graph_node",
 "name": "",
 "artist": "",
 "id": ""
}
```

Do not expose embedding details.

------

## 3. neighbors(anchor, k)

Purpose:

"What does this sound like?"

Return:

```json
{
 "anchor": "",
 "neighbors": [
   {
    "artist":"",
    "song":"",
    "relationship":"similar melodic/electronic texture"
   }
 ]
}
```

Internally this may use:

- SongIndex
- embeddings
- tags

but hide this.

------

## 4. compare(anchor_a, anchor_b)

Return:

```json
{
 "similarity_summary":"",
 "shared_traits":[],
 "differences":[],
 "bridge_candidates":[]
}
```

Do not only return cosine similarity.

------

## 5. bridge(anchor_a, anchor_b)

Purpose:

"What sits between these sounds?"

Return:

```json
{
 "bridge_tracks":[
   {
    "artist":"",
    "song":"",
    "why":""
   }
 ]
}
```

------

## 6. explore_cluster(anchor)

Purpose:

"What area should I explore next?"

Return:

```json
{
 "cluster":"",
 "nearby_artists":[],
 "recommended_direction":""
}
```

------

## 7. explain_node(anchor)

Purpose:

"Why is this here?"

Return:

```json
{
 "song":"",
 "artist":"",
 "neighbors":[],
 "cluster":"",
 "traits":[]
}
```

------

# Part 2: Simplify MentorReAct

Do not delete this file.

Rename conceptually:

MentorReAct -> MentorAgent

Remove:

- LLM tool selection
- JSON tool generation
- `_extract_first_json`
- `_fallback_call`
- `_validate_call`

The model should NOT output:

```json
{
 "tool":"bridge",
 "args":{}
}
```

The classifier should determine intent.

New flow:

```python
def run(question, context):

    intent = classify_question(question)

    observation = graph.execute(
        intent.tool,
        intent.args,
        context
    )

    return compose_answer(
        question,
        observation
    )
```

------

# Part 3: Keep deterministic classification

Keep:

- classify_question()
- _anchor_from()
- _coherence_target()
- _seeds_from()

However:

Change output format from:

```python
{
 "shape":"bridge",
 "tool":"bridge",
 "args":{}
}
```

to:

```python
{
 "intent":"bridge",
 "args":{}
}
```

The classifier chooses intent.

The LLM does not.

------

# Part 4: Make graph context first-class

Every tool should receive:

```python
context
```

with:

```python
context.selected_node
context.visible_nodes
context.last_anchor
context.last_observations
```

The graph UI is the user's memory.

Prioritize:

1. selected node
2. visible graph
3. corpus

------

# Part 5: Simplify AntherSoundsLike

Keep this class.

It is valuable.

But change its role.

It becomes an internal service:

```python
AntherSimilarityService
```

Expose only:

```python
resolve()
similar()
cluster()
tags()
```

Do not expose:

```python
sounds_like_from_audio()
sounds_like_from_vec()
_place()
_format_for_prompt()
```

Those are implementation functions.

------

# Part 6: Improve evidence formatting

Current:

```
Nearest released tracks:
- Artist - Song (sim 0.87)
```

Replace with:

```
Closest neighbors:
- Artist - Song
  Reason: shared atmospheric electronic texture

Cluster:
- Experimental bass / ambient electronic

Tags:
- dark
- rhythmic
- cinematic
```

Similarity scores can remain available internally.

------

# Part 7: Preserve hallucination safety

Keep:

- `_facts()`
- `_names_ok()`
- observation-grounded answers

The final LLM prompt should be:

```
You are a music mentor.

Rewrite ONLY the provided facts.

Do not introduce:
- artists
- songs
- genres
- clusters

that are not in FACTS.
```

------

# Part 8: Add tests

Create tests for:

## Context resolution

Input:

"what do I sound like?"

Expected:

Uses selected node.

------

## Bridge

Input:

"what sits between Burial and Aphex Twin?"

Expected:

Calls bridge.

------

## Follow-up

Input:

"which ones?"

Expected:

Uses previous observations.

------

## Unknown artist

Input:

"what about Taylor Swift?"

Expected:

Honest unresolved response.

------

## UI state

Input:

"why is this node here?"

Expected:

Uses selected graph node.

------

# Success Criteria

The final system should:

✅ understand the graph state
✅ answer questions about visible songs
✅ explore relationships
✅ explain why nodes are connected
✅ recommend directions
✅ avoid hallucinating songs/artists
✅ keep ML implementation hidden from the LLM
✅ use the graph as the user's context

Do not optimize for a generic chatbot. Optimize for an AI music mentor embedded inside a spatial exploration tool.

```
I would also have Claude **show you the diff before deleting anything**, because the current `react.py` has some genuinely good safety logic that should survive the refactor.
```

--- PHASE 0: DISCOVERY & CLARIFICATION ---

STOP. Do not begin refactoring, writing code, or deleting files yet.

Because autonomous agents often hallucinate implementations when faced with missing context, your very first task is to ask me a set of specific follow-up questions.

Review the plan above, search the codebase if necessary, and then output a numbered list of questions for me to answer regarding the following implementation details:

Explanation Generation: In the new MentorGraphTools API, fields like "relationship": "similar melodic/electronic texture" or "why": "" are required. Since the LLM is now at the end of the pipeline, ask me exactly how these strings should be generated or synthesized within the tool logic.

File Mapping: Ask me to confirm the exact file paths for MentorGraphTools, AntherSoundsLike, MentorAgent, and the classifier.

The Origin of Context: Ask me where GraphContext originates in the application layer. (e.g., Is it passed from the frontend in an API request? Is there a state manager? How does the entry point receive it?)

Classifier Prompt Updates: The output format for classify_question() is changing to {intent, args}. Ask me where the underlying LLM prompt template for this classifier is located so you can update its schema instructions.

Testing Framework: Ask me which testing framework you should use for Part 8, and specifically what parts of the system should be mocked versus executed.