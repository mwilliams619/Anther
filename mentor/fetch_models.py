"""Download the base weights the mentor needs, for a fresh clone.

The trained LoRA adapter and the prebuilt RAG index are small and can be
committed or shared directly; the two *base* models are large and are NOT in
git. This script fetches them into <repo>/models/mentor/.

    python mentor/fetch_models.py

Note on the download method: HuggingFace's Xet chunked-transfer protocol can
stall in restricted-network environments (only the small JSON metadata lands).
Classic streaming via a plain GET works. This script streams each file with
`requests` directly from the resolve/ endpoint, which avoids the Xet path.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE_DIR, EMB_DIR
import requests

HF = "https://huggingface.co"

FILES = {
    # Qwen2.5-7B-Instruct base
    "Qwen/Qwen2.5-7B-Instruct": (BASE_DIR, [
        "config.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]),
    # bge-small-en-v1.5 retriever embedder
    "BAAI/bge-small-en-v1.5": (EMB_DIR, [
        "config.json", "config_sentence_transformers.json",
        "sentence_bert_config.json", "modules.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
        "model.safetensors", "1_Pooling/config.json",
    ]),
}


def stream(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  skip (exists): {os.path.basename(dst)}")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dst + ".part", "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {os.path.basename(dst)}: "
                                     f"{got/1e6:.0f}/{total/1e6:.0f} MB")
                    sys.stdout.flush()
        os.replace(dst + ".part", dst)
    print()


def main():
    for repo, (dst_dir, files) in FILES.items():
        print(f"\n== {repo} -> {dst_dir}")
        for rel in files:
            url = f"{HF}/{repo}/resolve/main/{rel}"
            stream(url, os.path.join(dst_dir, rel))
    print("\nAll base weights present. The trained adapter + RAG index are in "
          "models/mentor/ already (committed or copied separately).")


if __name__ == "__main__":
    main()
