"""Drive the GPU embedding worker as a subprocess and collect its outputs.

The heavy step (MERT inference + corpus placement) runs in a fresh Python
process via ``_embed_worker`` so that (a) the network proxy used to fetch
previews is live, and (b) the offline MERT weights load cleanly. This module
just launches it, streams progress, and parses the RESULT line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_MERT_DIR = "models/mert_v1_330m"
DEFAULT_CORPUS_DIR = "models/corpus_mpd_100k_merit"


def embed_and_place(
    csv_path: str,
    out_dir: str,
    prefix: str,
    *,
    repo: str = REPO_ROOT,
    mert_dir: str = DEFAULT_MERT_DIR,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    stream: bool = True,
) -> dict[str, Any]:
    """Embed every preview in ``csv_path`` and place onto the corpus.

    Writes ``<out_dir>/<prefix>_placements.csv`` and
    ``<out_dir>/<prefix>_embeddings.npz``. Returns the parsed RESULT dict
    ({n_embedded, n_requested, placements_csv, embeddings_npz}). Raises
    ``RuntimeError`` if the worker exits non-zero or emits no RESULT line.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "anther_ml.sonic_trajectory._embed_worker",
        csv_path, out_dir, prefix, repo, mert_dir, corpus_dir,
    ]
    env = {**os.environ, "PYTHONPATH": repo}
    proc = subprocess.Popen(
        cmd, cwd=repo, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    result: dict[str, Any] | None = None
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if stream:
            print(line, flush=True)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    proc.wait()
    if proc.returncode != 0 or result is None:
        raise RuntimeError(
            f"embed worker failed (rc={proc.returncode}); last lines:\n" + "\n".join(tail)
        )
    return result
