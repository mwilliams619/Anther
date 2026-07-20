#!/usr/bin/env bash
# Deploy the STABLE branch (ml-dev) to production.
#
# Called by .github/workflows/deploy-stable.yml on every push to ml-dev, or
# run manually for a first-time setup / manual redeploy. Never touches your
# interactive ~/Dev/Anther checkout — deploys into a dedicated
# ~/Dev/Anther-stable worktree that anther-flask.service points at.
#
# Sequence: fetch+checkout ml-dev into the stable worktree -> symlink
# data/models back to the main checkout -> run a fast smoke-test subset ->
# restart the systemd service -> poll it healthy. Any stage failing aborts
# with a non-zero exit (visible as a failed GitHub Actions run), and — since
# systemctl restart isn't reached until after tests pass — a broken smoke
# test never takes down the currently-running production service.
#
# Usage:
#   scripts/deploy/deploy_stable.sh
#
# Env overrides:
#   ANTHER_MAIN_REPO    default /home/matt/Dev/Anther
#   ANTHER_STABLE_DIR    default /home/matt/Dev/Anther-stable
#   ANTHER_STABLE_BRANCH default ml-dev
#   ANTHER_STABLE_PORT   default 5000
#   ANTHER_SKIP_TESTS    set to 1 to skip the smoke-test gate (not recommended)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

STABLE_DIR="${ANTHER_STABLE_DIR:-/home/matt/Dev/Anther-stable}"
BRANCH="${ANTHER_STABLE_BRANCH:-ml-dev}"
PORT="${ANTHER_STABLE_PORT:-5000}"

ensure_worktree "$STABLE_DIR"
checkout_ref "$STABLE_DIR" "$BRANCH"
symlink_shared_dirs "$STABLE_DIR"

if [ "${ANTHER_SKIP_TESTS:-0}" != "1" ]; then
    # Only run smoke-test files that actually exist in this checkout (see
    # the longer explanation in deploy_staging.sh) — ml-dev should always be
    # current, but this keeps a manual rollback to an older ml-dev commit
    # from hard-failing the deploy on a missing test file instead of just
    # running fewer checks.
    smoke_tests=()
    for t in test_merit test_corpus_merit_index test_corpus_build \
             test_corpus_bundle test_corpus_place test_corpus_recommend \
             test_corpus_sources test_embedding test_similarity; do
        f="$STABLE_DIR/tests/$t.py"
        [ -f "$f" ] && smoke_tests+=("$f")
    done
    if [ "${#smoke_tests[@]}" -eq 0 ]; then
        echo "[deploy-stable] WARNING: none of the expected smoke-test files exist on this branch — deploying WITHOUT a test gate"
    else
        echo "[deploy-stable] running smoke-test subset (${#smoke_tests[@]}/9 files present on this branch)..."
        "$STABLE_DIR/.venv/bin/python" -m pytest "${smoke_tests[@]}" -q
    fi
else
    echo "[deploy-stable] ANTHER_SKIP_TESTS=1 — skipping smoke-test gate"
fi

echo "[deploy-stable] restarting anther-flask..."
sudo /usr/bin/systemctl restart anther-flask

health_check "$PORT" 45
echo "[deploy-stable] deploy complete: $(git -C "$STABLE_DIR" rev-parse --short HEAD) on port $PORT"
