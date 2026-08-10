"""
Hugging Face Hub transport for published corpus bundles.

A bundle is data, not a model, so it lives in a **dataset** repo: git-LFS
storage, resumable and parallel downloads, Xet chunk dedupe (re-uploading a
changed bundle pushes only the changed chunks), and revisions/tags that line up
with the bundle's own frozen-map semantics — pin a revision and every user has
byte-identical geography.

    python -m anther_ml.corpus push  models/publish/corpus_mpd_100k_merit \\
        --repo-id you/anther-corpus-mpd-100k
    python -m anther_ml.corpus fetch --repo-id you/anther-corpus-mpd-100k \\
        --out models/corpus_mpd_100k_merit --revision v1

Push refuses any bundle that has not passed ``corpus publish``: shipping
``leiden.pkl`` or raw playlist membership would defeat the point of the
publication step.

Downloads are never implicit. ``ReferenceCorpus.load`` fetches only when handed
a ``repo_id`` (or ``ANTHER_CORPUS_REPO`` is set) *and* the bundle is missing
locally; otherwise a missing bundle raises with the exact ``corpus fetch``
command to run.
"""

import json
import os
from pathlib import Path

#: Env var naming the dataset repo a missing bundle may be fetched from.
CORPUS_REPO_ENV = "ANTHER_CORPUS_REPO"

REPO_TYPE = "dataset"

#: Never uploaded, whatever is sitting in the directory.
PUSH_IGNORE = ("*.log", ".DS_Store", "__pycache__/*", "*.pyc")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "huggingface_hub is required for corpus push/fetch — "
            "pip install -e . (or pip install huggingface_hub)"
        ) from exc
    return HfApi()


def bundle_publish_state(bundle_dir: str | Path) -> dict:
    """What ``corpus publish`` did to this directory, if anything.

    Returns ``{"published": bool, "reasons": [...]}`` — ``reasons`` lists the
    artifacts that make it unsafe to redistribute.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no corpus bundle at {bundle_dir} (missing manifest.json)"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    reasons = []
    if "publish" not in manifest:
        reasons.append("manifest has no 'publish' block — never went through `corpus publish`")
    if (bundle_dir / "leiden.pkl").exists():
        reasons.append("leiden.pkl present (pickled estimators; pins Python 3.11)")
    if (bundle_dir / "merit_backbone.npy").exists():
        reasons.append("merit_backbone.npy present (2 GB build intermediate)")

    index_json = bundle_dir / "index.json"
    if index_json.exists():
        with open(index_json) as f:
            payload = json.load(f)
        rows = payload if isinstance(payload, list) else payload.get("metadata") or []
        if any("playlists" in row for row in rows):
            reasons.append("index.json still carries per-track playlist membership")

    return {"published": not reasons, "reasons": reasons}


def push_bundle(
    bundle_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a published bundle to a HF dataset repo. Returns the repo URL.

    Creates the repo **private** by default — flipping to public is a
    deliberate act, not a side effect of the first upload.
    """
    bundle_dir = Path(bundle_dir)
    state = bundle_publish_state(bundle_dir)
    if not state["published"]:
        raise ValueError(
            "refusing to push an unpublished bundle:\n  - "
            + "\n  - ".join(state["reasons"])
            + "\nRun `python -m anther_ml.corpus publish` first."
        )

    api = _api()
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    if revision:
        api.create_branch(
            repo_id, branch=revision, repo_type=REPO_TYPE, exist_ok=True
        )
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        ignore_patterns=list(PUSH_IGNORE),
        commit_message=commit_message or f"publish {bundle_dir.name}",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def fetch_bundle(
    repo_id: str,
    local_dir: str | Path,
    *,
    revision: str | None = None,
    include_backbone: bool = False,
) -> Path:
    """Download a bundle from a HF dataset repo into ``local_dir``.

    ``merit_backbone.npy`` is skipped unless ``include_backbone`` — it is only
    needed to rebuild the MERIT sidecar or extend the corpus, and it is by far
    the largest file when a repo does carry it.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    ignore = None if include_backbone else ["merit_backbone.npy"]
    snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=str(local_dir),
        ignore_patterns=ignore,
    )
    return local_dir


def resolve_repo_id(repo_id: str | None) -> str | None:
    """Explicit argument wins, else ``ANTHER_CORPUS_REPO``, else None."""
    return repo_id or os.environ.get(CORPUS_REPO_ENV) or None


def ensure_bundle(
    local_dir: str | Path,
    repo_id: str | None = None,
    *,
    revision: str | None = None,
) -> Path:
    """Return ``local_dir``, downloading it first if it isn't there yet.

    A bundle already on disk is never re-fetched — a frozen map that silently
    updated underneath a running session would break the one guarantee the
    corpus makes.
    """
    local_dir = Path(local_dir)
    if (local_dir / "manifest.json").exists():
        return local_dir

    repo_id = resolve_repo_id(repo_id)
    if repo_id is None:
        raise FileNotFoundError(
            f"no corpus bundle at {local_dir} and no repo to fetch from. "
            f"Either point ANTHER_CORPUS at an existing bundle, or run:\n"
            f"  python -m anther_ml.corpus fetch --repo-id <user/repo> "
            f"--out {local_dir}"
        )
    return fetch_bundle(repo_id, local_dir, revision=revision)
