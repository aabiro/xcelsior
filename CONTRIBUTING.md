# Contributing

## Local setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ruff black pytest pre-commit
pre-commit install   # one-time: installs the git hook
```

## Quality gates

CI runs the same checks locally:

```bash
ruff check .
black --check *.py routes/*.py tests/
./run-tests.sh          # full suite (excludes live e2e)
./run-tests.sh quick    # ~15s unit tests
```

Pre-commit runs ruff + black + whitespace hygiene on every commit; it mirrors
what CI enforces, so a clean local commit won't be surprised by a red PR.

## Commit style

Conventional commits (`feat:`, `fix:`, `chore:`, `style:`, `test:`, `docs:`)
with a scope matching the plan milestone when applicable, e.g. `feat(p1.3):`.

## Running a single test file

```bash
./run-tests.sh tests/test_agent_upgrade.py
```

## Deploy

```bash
bash scripts/deploy.sh --quick   # VPS only (api/frontend/nginx)
# Default sync uses rsync deltas + SSH mux (pixelspark-style). Legacy: XCELSIOR_DEPLOY_SYNC=tarball
# Wire compression (default zstd; installs zstd/pigz): XCELSIOR_DEPLOY_COMPRESS=zstd
# Parallel code mirrors: XCELSIOR_DEPLOY_SYNC_HOSTS=100.64.0.5,100.64.0.6
# Per-step timings: XCELSIOR_DEPLOY_TIMING=1 bash scripts/deploy.sh
```

Worker agents on GPU hosts update themselves via the admin rollout endpoint
(`POST /api/admin/agent/rollout`) — no manual scp needed.
