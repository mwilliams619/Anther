# Rollout checklist — Anther CI/CD

Ordered, one-shot list to go from "files committed" to "push to `ml-dev`
auto-deploys, push to any feature branch auto-deploys to staging." Full
detail and copy-paste commands for each item are in
[CI_CD_SETUP.md](CI_CD_SETUP.md) — this is the tick-list version.

## Before you start

- [ ] Review and commit the new files: `.github/workflows/deploy-stable.yml`,
      `.github/workflows/deploy-staging.yml`, `scripts/deploy/_common.sh`,
      `scripts/deploy/deploy_stable.sh`, `scripts/deploy/deploy_staging.sh`,
      `scripts/deploy/setup_staging_worktree.sh`, `systemd/anther-flask.service`,
      `systemd/anther-flask-staging.service`, `CI_CD_SETUP.md`,
      `ROLLOUT_CHECKLIST.md`, and the `.gitignore` addition
      (`scripts/deploy/dryrun_tmp/`).
- [ ] Clean up dry-run test worktrees if still present (see
      "Cleanup after the dry-run" in CI_CD_SETUP.md).

## Root-level setup (needs sudo / GitHub access — see CI_CD_SETUP.md §1-5)

- [ ] Create `~/Dev/Anther-stable` worktree + symlink `.venv` (§1)
- [ ] Create `~/Dev/Anther-staging` worktree via `setup_staging_worktree.sh` (§1)
- [ ] Install both systemd units from `systemd/` to `/etc/systemd/system/`,
      `daemon-reload`, `enable --now` both (§2)
- [ ] Confirm both services answer on ports 5000 and 5001 (§2)
- [ ] Add a scoped `/etc/sudoers.d/anther-deploy` — restart-only, two lines,
      validated with `visudo -c` (§3)
- [ ] Register a GitHub Actions self-hosted runner with label `anther`,
      install as a systemd service (§4)
- [ ] Add the staging hostname to the Cloudflare Tunnel `config.yml`, run
      `cloudflared tunnel route dns`, restart `cloudflared` (§5)

## First deploys (manual, to validate before trusting push-triggers)

- [ ] `scripts/deploy/deploy_stable.sh` — confirm it prints
      `health check OK on port 5000`
- [ ] `scripts/deploy/deploy_staging.sh <a-feature-branch>` — confirm
      `health check OK on port 5001`
- [ ] Load the production domain and the staging hostname in a browser —
      confirm both serve the app and are actually running the expected
      commit (`git -C ~/Dev/Anther-stable rev-parse HEAD` /
      `~/Dev/Anther-staging`)

## Confirm the automated path

- [ ] Push a trivial commit to `ml-dev` — confirm a
      `Deploy stable (ml-dev)` run appears and succeeds under
      **GitHub → Actions**
- [ ] Push a trivial commit to a feature branch — confirm
      `Deploy staging (feature branches)` runs and succeeds
- [ ] Deliberately break a smoke test on a feature branch, push, confirm the
      staging deploy fails loudly in Actions **and** the previously-running
      staging service is left untouched (the restart step is never reached)

## Known gaps / things to revisit later

- No automatic rollback trigger — reverting a bad `ml-dev` push means
  pushing a revert commit (or manually pinning a worktree to an older SHA
  per CI_CD_SETUP.md's Rollback section).
- Concurrency: two rapid staging pushes cancel the first deploy in favor of
  the second (`cancel-in-progress: true`); stable pushes queue instead
  (`cancel-in-progress: false`) so a stable deploy always finishes.
- Secrets (`SPOTIFY_CLIENT_ID`/`SECRET`, `ANTHER_UI_SECRET_KEY`) are not
  currently anywhere in the new worktrees — each needs its own `.env` file,
  since worktrees don't share dotfiles the way they share `data/`/`models/`.
