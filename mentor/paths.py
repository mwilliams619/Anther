"""Central path resolution for the mentor package.

Everything is resolved relative to the Anther repo root (the parent of this
`mentor/` package), so scripts work no matter the current working directory.

Layout:
    <repo>/mentor/            <- this package (source, tracked in git)
    <repo>/models/mentor/     <- weights + data (gitignored via models/)
    <repo>/models/corpus_corpus_mpd_100k/  <- shared Anther corpus
"""
import os

MENTOR_PKG = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(MENTOR_PKG)

# model + data assets (not committed; created by fetch_models.py / training)
ASSETS     = os.path.join(REPO_ROOT, "models", "mentor")
MODEL_NAME = "Qwen2.5-7B-Instruct"          # was "Qwen2.5-3B-Instruct"
BASE_DIR   = os.path.join(ASSETS, MODEL_NAME)
EMB_DIR    = os.path.join(ASSETS, "bge-small-en-v1.5")
LORA_DIR   = os.path.join(ASSETS, "mentor_lora_7b")   # 3B adapter stays at mentor_lora/

# RAG index + training data
INDEX_NPY   = os.path.join(ASSETS, "rag_index.npy")
INDEX_META  = os.path.join(ASSETS, "rag_index_meta.json")
PASSAGES    = os.path.join(ASSETS, "rag_passages.jsonl")
TRAIN_JSONL = os.path.join(ASSETS, "train.jsonl")
VAL_JSONL   = os.path.join(ASSETS, "val.jsonl")

# shared Anther similarity corpus (produced by the Anther pipeline)
CORPUS_DIR = os.path.join(REPO_ROOT, "models", "corpus_corpus_mpd_100k")
