# Working in this repository

Instructions for AI coding agents, and a short version for people. `CLAUDE.md`
points here, so every agent reads the same file. The frontend has its own
`api/frontend/AGENTS.md` about its Next.js version; read it before touching
`api/frontend/`.

## What this is

Xcelsior is a global GPU compute marketplace. Independent hosts run a worker
agent on their own machines; customers rent per-second GPU instances
(on-demand or spot), call per-token serverless inference endpoints, attach
volumes, and pay from a prepaid wallet. AI agents can drive all of it through
the MCP server.

## Layout

- Python 3.12, FastAPI, Postgres, Redis. Top-level packages: `api/` (app and
  routes), `core/`, `auth/`, `billing/`, `hosts/`, `workers/` (including the
  worker agent), `control_plane/`, `privacy/`. Modules import each other by
  bare name; the search path is `pythonpath` in `pyproject.toml`.
- `api/migrations/`: Alembic, numbered files. Schema changes go here; runtime
  DDL is refused by a test.
- `api/frontend/`: Next.js app (website and dashboard), English and French.
- `api/mcp/`: the MCP server. `api/mcp-cli/`: the `@xcelsior-gpu/mcp` npm
  package. `api/wizard/`: the `@xcelsior-gpu/wizard` setup wizard.
  `api/desktop/`: the Tauri desktop app. `api/fern/`: public OpenAPI and docs.
- `tests/`: the Python suite. Test files and test names are sentences that
  state the rule they hold.

## Before you say it works

Hosted GitHub Actions does not run on this repository, so a green push proves
nothing. Run the gates yourself:

```bash
.venv/bin/ruff check .
.venv/bin/black --check <files you changed>
.venv/bin/python api/scripts/pyright_gate.py
./run-tests.sh tests/test_<area>.py     # one file
./run-tests.sh quick                    # fast unit tests
./run-tests.sh                          # full suite
```

- Use `.venv` (Python 3.12, synced from `uv.lock`). A different interpreter or
  black version gives different answers than the gate.
- Format only the files you change; the tree is black-clean, so a file that
  fails is drift you introduced.
- Frontend, from `api/frontend/`: `npm run lint`, `npx tsc --noEmit`,
  `npm test`, `npm run i18n:check`. MCP, from `api/mcp/`: `npm run typecheck`,
  `npm test`.
- Say what you ran and what you did not. "Not verified" is an acceptable
  answer; an unverified "done" is not.

## Rules that are easy to break

- **One source for any repeated value.** Counts, versions, prices and limits
  that appear in more than one place are read from one file
  (`core/release_facts.py` names them), with a test that fails on a copy.
  Add the test when you add the second place.
- **Money is integer micros** in storage and arithmetic. Currency appears only
  at the API boundary and on payment, invoice, legal and tax surfaces; UI copy
  says "/hr", not "CAD/hr".
- **The UI never hardcodes GPU lists or prices.** Pickers read live inventory
  and the published rate card. No emoji or text glyphs standing in for icons
  (use lucide icons). Regions are shown by name, country first, never as a
  code like `ca-on`.
- **Xcelsior is global.** Never write data-residency or sovereignty claims.
- **Many routes were scaffolded and never wired.** Before building on an
  endpoint, follow it end to end and find the test that exercises it.
- **Anything an agent can do that acts previews first.** MCP tools that change
  state take `confirm:false` for a preview and execute only an approved plan;
  `api/mcp/tests/unit/every-act-is-previewed.test.ts` enforces it.
- **Worker distribution files are signed.** After editing one (the list is
  `SIGNABLE` in `api/scripts/sign-static-agent.py`), re-sign and run
  `sign-static-agent.py --verify`, or hosts refuse the update.
- **Workflows run only pinned code.** Every action is pinned to a commit SHA
  (`api/scripts/pin_actions.py`), every token defaults to read-only, and every
  checkout drops its credentials; `tests/test_workflow_actions_are_pinned.py`
  holds all three.

## The repository is public

- No secrets, tokens, private hostnames, internal IP addresses or customer data
  in code, comments, tests or commit messages.
- Internal documents (reviews, evidence, handoffs, research notes) and
  generated billing exports never go in the tree. Write them to
  `~/xcelsior-internal-docs`. `tests/test_internal_documents_are_not_tracked.py`
  fails if one is tracked.

## Commits and branches

- The maintainer's agents work on `main`, commit small and often, and push to
  `origin/main`. Outside contributors open a pull request.
- Conventional commits: `type(scope): subject`. The subject is a sentence a
  user can read ("a refresh racing its own rotation no longer signs everyone
  out"), because release notes are generated from it. The body says why.
- Stage files by path. Another session may be editing the same checkout;
  never commit, reset or reformat work that is not yours.

## Production is not yours to change

Deploys, migrations against production, worker agent rollouts, feature-flag
flips, payouts and anything sent to customers are run by the maintainer. You
may prepare them: write the change, test it, and hand over the exact command.
The deploy ships the working tree, so it runs only from a clean `main` equal to
`origin/main`.

## Releasing

See "Releasing" in `CONTRIBUTING.md`: `api/scripts/release.py` tags each
component, writes its notes, and checksums and signs its files.

## Reviewing a change

When asked to review a diff or pull request:

- Rank each finding **P0** (breaks production, loses money or data, or opens a
  security hole), **P1** (a real bug or broken rule above), or **P2** (worth
  fixing, not urgent).
- **Proven or silent.** Report a finding only with the evidence for it: the
  input that fails, the line that is wrong, or the test that would catch it.
  If you cannot show it, leave it out.
- At most ten findings, most severe first. Style preferences are not findings.
- Say what you did not review.
