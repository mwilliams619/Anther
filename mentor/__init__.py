"""Music mentor — a lightweight, locally-served advice chatbot for aspiring
artists: a music mentor sitting inside an interactive spatial sound map.

Three mechanisms: a QLoRA fine-tune of Qwen2.5-7B for TONE, a bge-small RAG
retriever over advice passages for FACTS, and a graph agent for EARS —
the LLM interprets intent and narrates; deterministic tools (graph_tools.py)
read the user's on-screen map and the Anther similarity corpus in between.

    from mentor.mentor import MusicMentor
    m = MusicMentor()
    m.chat("How do I find my sound?")
    m.chat("Why is this node here?", selected_node_id="abc123")
"""
