#!/usr/bin/env bash
# Shared helpers for deploy_stable.sh / deploy_staging.sh / setup_staging_worktree.sh.
#
# Design note: production and staging each run out of a DEDICATED git worktree
# (Anther-stable, Anther-staging), never out of the developer's interactive
# checkout (~/Dev/Anther). That checkout can be on any branch at any time —
# mid-rebase, mid-feature, uncommitted changes and all — and a deploy must
# never `git reset --hard` it out from under whoever is working there. Each
# worktree gets its own checkout of code, but data/ and models/ (gitignored,
# tens of GB) are symlinked back to the main checkout rather than duplicated.
set -euo pipefail

MAIN_REPO="${ANTHER_MAIN_REPO:-/home/matt/Dev/Anther}"

# ensure_worktree <worktree_dir>
# Creates the worktree (detached, arbitrary placeholder commit) if it doesn't
# exist yet. Idempotent — safe to call on every deploy.
ensure_worktree() {
    local wt_dir="$1"
    if git -C "$MAIN_REPO" worktree list --porcelain | grep -qxF "worktree $wt_dir"; then
        return 0
    fi
    echo "[deploy] creating worktree at $wt_dir"
    git -C "$MAIN_REPO" worktree add --detach "$wt_dir" HEAD
}

# symlink_shared_dirs <worktree_dir>
#
# MUST be called AFTER checkout_ref, not before: data/ is only *partially*
# gitignored (data/artist_clustering/, everynoise_genres*.txt,
# fma_to_everynoise.json are tracked — 13MB total, small enough to let git
# check out its own copy per worktree) while data/audio/, data/fma_metadata/,
# data/mpd_dump/, data/billboard/, and the eval-embedding checkpoint are
# gitignored and large. `git checkout` never touches gitignored paths, so
# doing the symlinking after checkout means the tracked files land normally
# and only the large untracked ones get shared. models/ is entirely
# untracked (verified: `git ls-files models/` is empty) so the whole
# directory is symlinked. Uses -sfn so re-running is a no-op / safe to
# repeat.
symlink_shared_dirs() {
    local wt_dir="$1"

    # models/ — 100% gitignored, safe to symlink wholesale.
    ln -sfn "$MAIN_REPO/models" "$wt_dir/models"

    # data/ subpaths — only the large gitignored ones. Tracked files
    # (data/artist_clustering/, everynoise_genres*.txt, fma_to_everynoise.json)
    # are left alone; git's own checkout populates those per worktree.
    mkdir -p "$wt_dir/data"
    for shared in audio fma_metadata mpd_dump billboard; do
        ln -sfn "$MAIN_REPO/data/$shared" "$wt_dir/data/$shared"
    done
    for f in "$MAIN_REPO"/data/*.zip "$MAIN_REPO"/data/fma_eval_embeddings.npz*; do
        [ -e "$f" ] || continue
        ln -sfn "$f" "$wt_dir/data/$(basename "$f")"
    done
}

# checkout_ref <worktree_dir> <branch-or-ref>
# Fetches and detached-checks-out a ref inside a worktree. Detached (not a
# named branch checkout) so the same branch can also be checked out in the
# developer's main checkout at the same time without git's
# "already checked out" conflict.
checkout_ref() {
    local wt_dir="$1" ref="$2"
    git -C "$wt_dir" fetch origin "$ref"
    git -C "$wt_dir" checkout --detach FETCH_HEAD
    echo "[deploy] $wt_dir now at $(git -C "$wt_dir" rev-parse --short HEAD) ($ref)"
}

# health_check <port> <timeout_seconds>
# Polls the Flask health/search endpoint until it responds or times out.
# Fails loudly (non-zero exit) rather than silently leaving a stale/crashed
# service running under a "deployed" label.
health_check() {
    local port="$1" timeout="${2:-30}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if curl -sf -o /dev/null "http://127.0.0.1:${port}/api/search?q=a"; then
            echo "[deploy] health check OK on port $port after ${waited}s"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "[deploy] HEALTH CHECK FAILED on port $port after ${timeout}s" >&2
    return 1
}
