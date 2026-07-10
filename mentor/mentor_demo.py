"""End-to-end demo + VRAM check for the music mentor.

Runs a battery of advice questions (tone + RAG grounding) and one
'what do I sound like' query that exercises the Anther tool via a stand-in
corpus vector (so no audio file is required for the demo). Writes a markdown
transcript and a JSON of raw results, and reports peak serving VRAM.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch
from mentor import MusicMentor
from mentor_anther import CORPUS_DIR
from paths import ASSETS

t0 = time.time()
m = MusicMentor(use_rag=True, use_anther=True)
load_vram = torch.cuda.memory_allocated() / 1e9
print(f"[load] mentor ready in {time.time()-t0:.1f}s | VRAM {load_vram:.2f} GB")

ADVICE_Q = [
    "I keep starting songs and never finishing them. How do I actually ship one?",
    "How do I deal with writer's block?",
    "Should I put my music on all the streaming platforms or just one?",
    "I have zero listeners. How do I actually get people to hear my stuff?",
    "Is it worth paying for mixing and mastering when I'm just starting out?",
]

transcript = ["# Music Mentor — demo transcript\n"]
raw = {"advice": [], "sounds_like": None,
       "vram": {"load_gb": round(load_vram, 2)}}

for q in ADVICE_Q:
    t = time.time()
    ans = m.chat(q, top_k=3)
    dt = time.time() - t
    hits, _ = m.rag.retrieve(q, top_k=3)
    print(f"\n### Q: {q}\n{ans}\n({dt:.1f}s, RAG {[round(h['score'],2) for h in hits]})")
    transcript.append(f"## Q: {q}\n\n{ans}\n\n"
                      f"_RAG grounding cosines: {[round(h['score'],2) for h in hits]}_\n")
    raw["advice"].append({"q": q, "answer": ans,
                          "rag_scores": [round(h['score'], 3) for h in hits]})

# --- 'what do I sound like' via a stand-in corpus vector -----------------------
raw_emb = np.load(os.path.join(CORPUS_DIR, "embeddings.npy"), mmap_mode="r")
meta = m.anther.index.metadata
probe_i = 40000  # an electronic track — clear genre signature
qvec = np.asarray(raw_emb[probe_i], dtype=np.float32)
snd = m.anther.sounds_like_from_vec(qvec, top_k=8)
snd_txt = m.anther.format_for_prompt(snd)
print(f"\n### 'What do I sound like?' (stand-in = corpus #{probe_i}: "
      f"{meta[probe_i].get('artist','')} — {meta[probe_i].get('name','')})")
print(snd_txt)

# feed the Anther analysis into the mentor for a natural-language readout
messages_ans = m.sounds_like(anther_result=snd)
print("\nMentor's take:\n", messages_ans)

transcript.append("## 'What do I sound like?' (Anther tool)\n\n"
                  f"Stand-in query = corpus track #{probe_i} "
                  f"({meta[probe_i].get('artist','')} — {meta[probe_i].get('name','')})\n\n"
                  f"```\n{snd_txt}\n```\n\n**Mentor's take:**\n\n{messages_ans}\n")
raw["sounds_like"] = {"probe_track": f"{meta[probe_i].get('artist','')} — {meta[probe_i].get('name','')}",
                      "result": snd, "mentor_take": messages_ans}

peak = torch.cuda.max_memory_allocated() / 1e9
raw["vram"]["peak_serving_gb"] = round(peak, 2)
print(f"\n[VRAM] peak serving = {peak:.2f} GB (budget 16 GB)")
transcript.append(f"\n---\n**Peak serving VRAM: {peak:.2f} GB** "
                  f"(load {load_vram:.2f} GB; budget 16 GB on the RTX 4080 SUPER)\n")

with open(os.path.join(ASSETS, "demo_transcript.md"), "w") as f:
    f.write("\n".join(transcript))
with open(os.path.join(ASSETS, "demo_results.json"), "w") as f:
    json.dump(raw, f, indent=2)
print("WROTE demo_transcript.md + demo_results.json in", ASSETS)
