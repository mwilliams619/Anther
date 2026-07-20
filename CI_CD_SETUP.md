# CI/CD Setup — one-time manual steps

This document lists every step that needs root or a GitHub token, in the
order to run them. Everything else (workflows, deploy scripts, worktree
setup, systemd unit *templates*) is already in the repo and needs no manual
editing — this doc is the bridge from "files exist" to "pipeline is live."

Assumes: repo at `~/Dev/Anther`, user `matt`, Cloudflare Tunnel already
routing your domain to `localhost:5000` via `anther-flask.service`.

---

## 0. What changes vs. what you already have

Your existing `anther-flask.service` runs out of `~/Dev/Anther` directly.
This setup moves production to a **dedicated worktree** (`~/Dev/Anther-stable`)
so an automated deploy on push never touches your interactive checkout —
you keep developing in `~/Dev/Anther` on whatever branch, uninterrupted.
A parallel `~/Dev/Anther-staging` worktree serves feature branches on a
second port. Both worktrees share your existing `data/` and `models/` via
symlinks — nothing is re-downloaded or duplicated.

---

## 1. Create the stable and staging worktrees

```bash
cd ~/Dev/Anther

# Stable worktree — same mechanism deploy_stable.sh uses, run once by hand
# so the directory exists before you point systemd at it.
git worktree add --detach ~/Dev/Anther-stable ml-dev
ln -sfn ~/Dev/Anther/data   ~/Dev/Anther-stable/data
ln -sfn ~/Dev/Anther/models ~/Dev/Anther-stable/models
ln -s   ~/Dev/Anther/.venv  ~/Dev/Anther-stable/.venv

# Staging worktree — or just run the provided script:
bash scripts/deploy/setup_staging_worktree.sh
```

Verify both:

```bash
git worktree list
# should show ~/Dev/Anther, ~/Dev/Anther-stable, ~/Dev/Anther-staging
```

## 2. Replace the systemd units

Your current `/etc/systemd/system/anther-flask.service` points at
`~/Dev/Anther`. Replace it with the version in this repo (same name, new
`WorkingDirectory`/`ExecStart` pointing at `Anther-stable`), and add the new
staging unit:

```bash
sudo cp ~/Dev/Anther/systemd/anther-flask.service         /etc/systemd/system/anther-flask.service
sudo cp ~/Dev/Anther/systemd/anther-flask-staging.service /etc/systemd/system/anther-flask-staging.service
sudo systemctl daemon-reload
sudo systemctl enable --now anther-flask
sudo systemctl enable --now anther-flask-staging
```

Confirm both are up:

```bash
systemctl status anther-flask --no-pager
systemctl status anther-flask-staging --no-pager
curl -s http://127.0.0.1:5000/api/search?q=a | head -c 200
curl -s http://127.0.0.1:5001/api/search?q=a | head -c 200
```

## 3. Scope sudo for the deploy scripts (no blanket sudo)

The deploy scripts call `sudo systemctl restart anther-flask` /
`anther-flask-staging` — nothing else. Grant exactly that, passwordless,
via a dedicated sudoers file (never edit `/etc/sudoers` directly):

```bash
sudo visudo -f /etc/sudoers.d/anther-deploy
```

Contents (replace `matt` if the runner service runs as a different user):

```
matt ALL=(root) NOPASSWD: /usr/bin/systemctl restart anther-flask
matt ALL=(root) NOPASSWD: /usr/bin/systemctl restart anther-flask-staging
```

Validate the syntax before leaving the editor closes it:

```bash
sudo visudo -c
```

## 4. Register a GitHub Actions self-hosted runner

Do this from your GitHub repo page: **Settings → Actions → Runners → New
self-hosted runner** (Linux, x64). GitHub will show a registration token —
copy the exact commands it gives you (the token is single-use and expires
in ~1 hour, so run these promptly rather than pre-copying from here):

```bash
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# from GitHub's own "New runner" page, with the real token substituted:
./config.sh --url https://github.com/mwilliams619/Anther --token <TOKEN_FROM_GITHUB> \
  --labels anther --name anther-box --work _work
```

The `--labels anther` matters — both workflow files target
`runs-on: [self-hosted, anther]`.

Install it as a systemd service so it survives reboots and runs unattended:

```bash
sudo ./svc.sh install matt
sudo ./svc.sh start
sudo ./svc.sh status
```

The runner now polls GitHub for jobs targeting this repo and executes them
as user `matt` — which is why step 3's sudoers scoping is what stands
between "push to ml-dev" and "systemctl restart runs as root."

## 5. Add the staging subdomain to your Cloudflare Tunnel

Add a second ingress rule to your tunnel's `config.yml` (commonly
`/etc/cloudflared/config.yml` or `~/.cloudflared/config.yml` — wherever
yours already lives) pointing a staging hostname at port 5001:

```yaml
ingress:
  - hostname: your-domain.com
    service: http://localhost:5000
  - hostname: dev.your-domain.com     # <-- add this block above the catch-all
    service: http://localhost:5001
  - service: http_status:404          # existing catch-all stays last
```

Then bind the DNS record and restart the tunnel:

```bash
cloudflared tunnel route dns <your-tunnel-name> dev.your-domain.com
sudo systemctl restart cloudflared
```

Verify:

```bash
curl -s https://dev.your-domain.com/api/search?q=a | head -c 200
```

## 6. First deploys (manual, to confirm before relying on push-triggers)

```bash
cd ~/Dev/Anther
scripts/deploy/deploy_stable.sh
scripts/deploy/deploy_staging.sh add-artist-clustering   # or whatever branch you're building
```

Each should print a `[deploy] health check OK on port ...` line at the end.
If a health check fails, the script exits non-zero without leaving the
prior service in an ambiguous state — `systemctl status` on the relevant
unit plus `journalctl -u anther-flask -n 50` (or `-staging`) is the first
place to look.

## 7. Confirm the automated path

Push a trivial commit to `ml-dev` and to your feature branch, then check
**GitHub → Actions** on the repo — both `Deploy stable (ml-dev)` and
`Deploy staging (feature branches)` should show a run, and each run's log
is the same output you saw manually in step 6.

---

## Rollback

Every deploy checks out a specific commit into a worktree — rolling back is
just re-running the deploy script pointed at an older ref:

```bash
cd ~/Dev/Anther-stable && git fetch origin
scripts/deploy/deploy_stable.sh   # re-reads whatever ml-dev points to now,
                                   # so roll back by first pushing a revert
                                   # to ml-dev, or:
git -C ~/Dev/Anther-stable checkout --detach <previous-good-sha>
sudo systemctl restart anther-flask   # manual restart, bypassing the script's
                                       # git-fetch step, to pin at that sha
                                       # until you're ready to push a real fix
```

## Cleanup after the dry-run

The deploy scripts were dry-run tested against throwaway worktrees under
`scripts/deploy/dryrun_tmp/` (now gitignored). If any of those directories
are still on disk, remove them from your own shell:

```bash
cd ~/Dev/Anther
git worktree remove --force scripts/deploy/dryrun_tmp/anther-stable-dryrun
git worktree remove --force scripts/deploy/dryrun_tmp/anther-staging-dryrun
git worktree remove --force scripts/deploy/dryrun_tmp/anther-staging-dryrun3
git worktree prune
rm -rf scripts/deploy/dryrun_tmp
```

## What's NOT covered here

- TLS/cert management for the domain — assumed already handled by your
  existing Cloudflare Tunnel setup.
- Log rotation for `journalctl` output — systemd's own journald retention
  policy applies; adjust `/etc/systemd/journald.conf` if you want a longer
  or shorter window than the system default.
- Secrets (e.g. `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET`, `ANTHER_UI_SECRET_KEY`)
  are read from the environment by `ui/app.py`'s `.env`-loading helper — set
  them in a `.env` file in each worktree root (`~/Dev/Anther-stable/.env`,
  `~/Dev/Anther-staging/.env`) since worktrees don't share dotfiles the way
  they share `data/`/`models/`. Neither the deploy scripts nor the workflows
  read or transmit these values.
