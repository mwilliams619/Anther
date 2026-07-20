#!/usr/bin/env bash
# One-time (and idempotent) setup of the staging git worktree.
#
# Note: the sibling stable worktree (~/Dev/Anther-stable) is created the
# same way, inline in deploy_stable.sh (ensure_worktree + symlink_shared_dirs
# from _common.sh) rather than via a separate setup script — it never needs
# to check out an arbitrary branch by name the way staging does (it's always
# origin/ml-dev), so there's no extra parameterization worth a dedicated
# script for it. Run scripts/deploy/deploy_stable.sh once manually to create
# and populate it the same way this script does for staging.
#
# Creates ~/Dev/Anther-staging as a separate worktree of the same repo,
# symlinks data/ and models/ back to the main checkout (so the ~90GB corpus
# isn't duplicated), and reuses the main checkout's .venv directly rather
# than building a second one — Anther's dependency set (torch, transformers,
# librosa, umap, leidenalg...) is heavy enough that a second venv would burn
# several more GB and minutes on every fresh setup for no benefit, since
# staging runs the same interpreter/deps as stable, just different code.
#
# Usage:
#   scripts/deploy/setup_staging_worktree.sh
#
# Env overrides:
#   ANTHER_MAIN_REPO      default /home/matt/Dev/Anther
#   ANTHER_STAGING_DIR     default /home/matt/Dev/Anther-staging
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

STAGING_DIR="${ANTHER_STAGING_DIR:-/home/matt/Dev/Anther-staging}"

ensure_worktree "$STAGING_DIR"
# checkout_ref isn't called here (this script just creates the worktree at
# whatever ref `git worktree add` defaulted to — deploy_staging.sh does the
# real branch checkout) but symlink_shared_dirs is still safe to run before
# any checkout: it only ever touches gitignored subpaths (models/ wholesale,
# specific large data/ subdirs), never the small tracked ones a later
# checkout would populate.
symlink_shared_dirs "$STAGING_DIR"

# Reuse the main .venv rather than a second one. app.py/atlas.py only need the
# venv's site-packages on PYTHONPATH plus its python binary — no venv-local
# files are written at runtime by this app, so a symlink is safe (if that
# ever changes, switch this to `python -m venv --system-site-packages`
# pointed at the main .venv instead).
if [ ! -e "$STAGING_DIR/.venv" ]; then
    ln -s "$MAIN_REPO/.venv" "$STAGING_DIR/.venv"
fi

echo "[setup] staging worktree ready at $STAGING_DIR"
echo "[setup] current ref: $(git -C "$STAGING_DIR" rev-parse --short HEAD 2>/dev/null || echo '(none yet — run deploy_staging.sh to check out a branch)')"
echo "[setup] data/ -> $(readlink -f "$STAGING_DIR/data" 2>/dev/null)"
echo "[setup] models/ -> $(readlink -f "$STAGING_DIR/models" 2>/dev/null)"
