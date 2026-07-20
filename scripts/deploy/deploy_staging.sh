#!/usr/bin/env bash
# Deploy a FEATURE branch to staging (second port, second worktree).
#
# Called by .github/workflows/deploy-staging.yml on every push to any branch
# other than ml-dev/master, or run manually. Always deploys whatever branch
# name is passed as $1 (or ANTHER_STAGING_BRANCH) — CI passes the branch that
# was just pushed, so staging always reflects the most recently pushed
# feature branch, not necessarily your current local HEAD.
#
# Same sequence as deploy_stable.sh: fetch+checkout branch into the staging
# worktree -> symlink data/models back to the main checkout -> smoke-test
# subset -> restart anther-flask-staging -> poll healthy.
#
# Usage:
#   scripts/deploy/deploy_staging.sh <branch-name>
#
# Env overrides:
#   ANTHER_MAIN_REPO      default /home/matt/Dev/Anther
#   ANTHER_STAGING_DIR     default /home/matt/Dev/Anther-staging
#   ANTHER_STAGING_BRANCH  fallback if no $1 given — no default, must resolve
#   ANTHER_STAGING_PORT    default 5001
#   ANTHER_SKIP_TESTS      set to 1 to skip the smoke-test gate (not recommended)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

STAGING_DIR="${ANTHER_STAGING_DIR:-/home/matt/Dev/Anther-staging}"
BRANCH="${1:-${ANTHER_STAGING_BRANCH:-}}"
PORT="${ANTHER_STAGING_PORT:-5001}"

if [ -z "$BRANCH" ]; then
    echo "[deploy-staging] no branch given — pass as \$1 or set ANTHER_STAGING_BRANCH" >&2
    exit 1
fi

ensure_worktree "$STAGING_DIR"
checkout_ref "$STAGING_DIR" "$BRANCH"
symlink_shared_dirs "$STAGING_DIR"

if [ ! -e "$STAGING_DIR/.venv" ]; then
    ln -s "$MAIN_REPO/.venv" "$STAGING_DIR/.venv"
fi

if [ "${ANTHER_SKIP_TESTS:-0}" != "1" ]; then
    # Staging deploys arbitrary feature branches, including ones that predate
    # files this list names (e.g. test_merit.py didn't exist before the
    # MERIT work landed) — unlike deploy_stable.sh, where ml-dev is always
    # current, a hardcoded file list here would hard-fail on any older or
    # unrelated branch before the app ever got a chance to run. So: run
    # exactly the smoke-test files that exist in *this* checkout, and only
    # skip the gate entirely (with a loud warning, not a silent no-op) if
    # none of them do.
    smoke_tests=()
    for t in test_merit test_corpus_merit_index test_corpus_build \
             test_corpus_bundle test_corpus_place test_corpus_recommend \
             test_corpus_sources test_embedding test_similarity; do
        f="$STAGING_DIR/tests/$t.py"
        [ -f "$f" ] && smoke_tests+=("$f")
    done
    if [ "${#smoke_tests[@]}" -eq 0 ]; then
        echo "[deploy-staging] WARNING: none of the expected smoke-test files exist on this branch — deploying WITHOUT a test gate"
    else
        echo "[deploy-staging] running smoke-test subset (${#smoke_tests[@]}/9 files present on this branch)..."
        "$STAGING_DIR/.venv/bin/python" -m pytest "${smoke_tests[@]}" -q
    fi
else
    echo "[deploy-staging] ANTHER_SKIP_TESTS=1 — skipping smoke-test gate"
fi

echo "[deploy-staging] restarting anther-flask-staging..."
sudo /usr/bin/systemctl restart anther-flask-staging

health_check "$PORT" 45
echo "[deploy-staging] deploy complete: $(git -C "$STAGING_DIR" rev-parse --short HEAD) ($BRANCH) on port $PORT"
