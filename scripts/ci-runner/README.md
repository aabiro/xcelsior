# Sandboxed self-hosted Actions runner

CI has not executed since ~2026-07-21 (Actions billing), so every gate in this
repository is local-only and runs when someone remembers. This is the free way to
get remote verification back without handing a public repository's workflow code
the keys to production.

**Read the threat model before enabling it.** It is safe only as a whole; each
piece assumes the others.

---

## The threat

`aabiro/xcelsior` is **public**, and the machine that would host a runner also
holds:

- `~/.ssh/xcelsior` — the production deploy key
- `.env` — 325 secret lines, including live Stripe credentials
- the worker agent, on the platform tailnet

A self-hosted runner in the naive configuration lets anyone who can trigger a
workflow run code with access to all of that. GitHub documents this warning for
public repositories, and it is not theoretical: workflow code is arbitrary code.

So the exposure is closed in two independent places.

## 1. Fork code cannot reach the runner

`.github/workflows/gates-sandboxed.yml` triggers on **`push` only**. Pushing a
branch to this repository requires write access; a fork cannot do it. There is no
`pull_request` trigger and no `pull_request_target`.

`run-runner.sh` **refuses to start** if a `pull_request` trigger appears in that
workflow. The constraint is enforced, not remembered — which is the difference
between this and the twelve compromises in
`docs/review/workaround-elimination-plan.md`.

## 2. The runner holds nothing worth stealing

| Property | Why |
|---|---|
| `--ephemeral` | one job per container, then the runner unregisters — nothing survives into the next job |
| no docker socket | a job cannot start a privileged container or reach the host daemon. Also why workflow `services:` do not work here |
| no mounts from the host | no `~/.ssh`, no `.env`, no project directory. The job clones the repo itself, like a hosted runner |
| `--read-only` + tmpfs | the writable surface is `/tmp` and the work directory, both wiped with the container |
| `--cap-drop ALL`, `no-new-privileges` | no capability escalation |
| unprivileged uid 10001 | not root, not in the docker group |
| `--memory 6g --cpus 6 --pids-limit 512` | a runaway job cannot take the host down. This box has 16 cores and 15 GB |

`entrypoint.sh` refuses to start if it finds a docker socket or a credential file
inside the sandbox — the two mistakes that would quietly turn the container back
into the host.

## What it does *not* cover, and why that is stated everywhere

The runner has no database, so it runs only the pure-parse gates: `ruff`, the
migration ledger, the runtime-DDL guard, the env-wiring gate, the source-tree
gate, schema discipline, the SQL-injection guard, and the migration lock
discipline.

**Not run:** the 4600-test DB suite, the compose check, pyright, the frontend and
MCP suites, and every live gate. The workflow writes that list into its own job
summary on every run, because a partial gate that reads as a full one is the defect
this whole review sequence has been about.

`./run-tests.sh` locally remains the only thing that has ever run the whole suite.

## Using it

```bash
# one job, then the container is gone
./scripts/ci-runner/run-runner.sh

# keep serving jobs
./scripts/ci-runner/run-runner.sh --loop
```

The registration token is minted fresh from the API on each start, expires in an
hour, and is never written to disk. It is not a repository secret.

## When to throw this away

The moment hosted Actions billing is restored. This exists because remote
verification is worth having and the alternative was none; a GitHub-hosted runner
needs no threat model at all. Plan item A1 is unchanged by this being here.
