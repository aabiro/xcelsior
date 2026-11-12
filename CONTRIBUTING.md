# Contributing

## Local setup

```bash
uv sync
pre-commit install   # one-time: installs the git hook
```

## Quality gates

CI runs the same checks locally:

```bash
uv run ruff check .
uv run black --check *.py routes/*.py tests/
./run-tests.sh          # full suite (excludes live e2e)
./run-tests.sh quick    # ~15s unit tests
```

Pre-commit runs ruff + black + whitespace hygiene on every commit; it mirrors
what CI enforces, so a clean local commit won't be surprised by a red PR.

## Commit style

Conventional commits (`feat:`, `fix:`, `chore:`, `style:`, `test:`, `docs:`)
with a scope matching the plan milestone when applicable, e.g. `feat(p1.3):`.

## Internal documents stay out of this repository

Internal review, evidence and handoff documents, and generated billing exports
(the Stripe `stripe_catalog.json` and `stripe_dashboard.json` files), must never
be committed to this repository. It is public, and on 2026-11-11 its history was
rewritten to remove documents of exactly this kind.

- Write internal documents to `~/xcelsior-internal-docs`, outside the checkout.
  This applies to agents too: a review, evidence or handoff record does not go in
  `api/docs/review/`, `docs/review/`, `reports/` or `research_notes/`.
- The block at the end of `.gitignore` ("Internal documents and generated
  exports - never commit") ignores those paths. Keep it last in the file, so no
  earlier `!` rule can re-include one.
- `tests/test_internal_documents_are_not_tracked.py` fails if a file at any of
  those paths is tracked, including one added with `git add -f`, or if a
  `.gitignore` rule stops covering one. It runs in the full suite, the sandboxed
  CI gates and the pre-push hook (`api/scripts/install_git_hooks.sh`).
- If it fails, untrack the file with `git rm --cached <path>` (the working copy
  stays on disk) and move it to `~/xcelsior-internal-docs`.

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
