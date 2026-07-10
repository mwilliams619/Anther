"""Music mentor — a lightweight, locally-served advice chatbot for aspiring
artists, built as a second feature of the Anther app.

Three mechanisms: a QLoRA fine-tune of Qwen2.5-7B for TONE, a bge-small RAG
retriever over advice passages for FACTS, and the Anther similarity engine as
a "what do I sound like" tool.

    from mentor.mentor import MusicMentor
    m = MusicMentor()
    m.chat("How do I find my sound?")
"""
