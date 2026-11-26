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

## Releasing

Each shipped component has its own version and its own tag prefix:

| Component | Version lives in | Tag |
|---|---|---|
| Platform (API, website, Python package) | `pyproject.toml` | `v2.3.0` |
| Worker agent | `VERSION` in `workers/worker_agent.py` | `agent-v2.4.1` |
| MCP server | `api/mcp/package.json` | `mcp-server-v4.0.0` |
| `@xcelsior-gpu/mcp` (npm) | `api/mcp-cli/package.json` | `mcp-v0.2.0` |
| `@xcelsior-gpu/wizard` (npm) | `api/wizard/package.json` | `wizard-v0.1.11` |
| Desktop app | `api/desktop/package.json` | `desktop-v0.1.0` |

`api/scripts/release.py` does the rest, locally:

```bash
python3 api/scripts/release.py status          # versions, tags, unreleased commits
python3 api/scripts/release.py notes wizard    # notes since the last wizard tag
python3 api/scripts/release.py tag wizard      # tag HEAD (clean, pushed main only)
python3 api/scripts/release.py pack wizard     # tarball, SBOM, SHA256SUMS, signature
python3 api/scripts/release.py verify dist/release/wizard-v0.1.11
```

Tag a release when it ships: when the package is published, the agent is
rolled out or the platform is deployed. Notes come from conventional commit
subjects (`feat`, `fix`, `perf`, and anything marked breaking), so write the
subject as the sentence a user should read.

`pack` signs `SHA256SUMS` with the release key from `release.py keygen` (kept
outside the repository; its public half is `api/scripts/release-signing.pub`).
Anyone can check a download with:

```bash
openssl pkeyutl -verify -pubin -inkey release-signing.pub -rawin -in SHA256SUMS -sigfile SHA256SUMS.sig
sha256sum -c SHA256SUMS
```

The worker agent's files are signed separately by `sign-static-agent.py`, with
the key the fleet trusts for self-updates. The release key never signs agent
code.
