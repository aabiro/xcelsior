# Sandboxed self-hosted Actions runner

Hosted Actions remains gated by the account's billing restriction. This pool
provides remote static checks and CodeQL on existing hardware, without exposing
production credentials to workflow code. It does not replace the database-backed
suite or prove application release readiness.

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
between this and the twelve compromises in the internal
`~/xcelsior-internal-docs/api/docs/review/workaround-elimination-plan.md`.

## 2. The runner holds nothing worth stealing

| Property | Why |
|---|---|
| `--ephemeral` | one job per container, then the runner unregisters — nothing survives into the next job |
| no docker socket | a job cannot start a privileged container or reach the host daemon. Also why workflow `services:` do not work here |
| no mounts from the host | no `~/.ssh`, no `.env`, no project directory. The job clones the repo itself, like a hosted runner |
| `--read-only` + tmpfs | the writable surface is `/tmp` and the work directory, both wiped with the container |
| `--cap-drop ALL`, `no-new-privileges` | no capability escalation |
| unprivileged uid 10001 | not root, not in the docker group |
| `--memory 4g --cpus 4 --pids-limit 512` per executor | bound each job; the service runs two isolated executors, with at most 8 GiB of container memory in total |

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
./api/scripts/ci-runner/run-runner.sh

# keep serving jobs
./api/scripts/ci-runner/run-runner.sh --loop

# two independent jobs, with separate workspaces and runner registrations
./api/scripts/ci-runner/run-runner.sh --loop --workers 2
```

The registration token is minted fresh from the API on each start, expires in an
hour, and is never written to disk. It is not a repository secret.

### Serving another repository

`XCELSIOR_CI_REPO` registers the runner for a different repository, with the same
sandbox. `aabiro/aabiro.github.io` deploys its GitHub Pages site this way:

```bash
XCELSIOR_CI_REPO=aabiro/aabiro.github.io ./api/scripts/ci-runner/run-runner.sh
```

The fork-trigger check follows the variable. For this repository it reads the
workflows on this checkout's HEAD; for any other it reads them from that
repository's default branch through the API, and refuses to start if it cannot.
The rule is the same either way: no workflow that targets
`[self-hosted, sandboxed]` may have a `pull_request` or `pull_request_target`
trigger.

### As a service, so it outlives the terminal that started it

`--loop` is a foreground loop, so a hand-started runner dies with its shell — and
a push then sits `queued` with nothing to serve it, which looks identical to a
broken workflow. `xcelsior-ci-runner.service` fixes that:

```bash
mkdir -p ~/.config/systemd/user
ln -sf "$PWD/api/scripts/ci-runner/xcelsior-ci-runner.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now xcelsior-ci-runner
sudo loginctl enable-linger "$USER"   # survive logout and reboot

systemctl --user status xcelsior-ci-runner
journalctl --user -u xcelsior-ci-runner -f
```

A **user** unit, not a system one, and the reason is in the file: the token is
minted with `gh`, which reads this user's credentials. Running as root would mean
copying a GitHub token somewhere harder to revoke, to satisfy systemd rather than
to satisfy anything real.

`enable-linger` is not optional. Without it the unit stops when the user's last
session ends, which is the same failure the unit was written to prevent, arriving
later and less obviously.

The service starts two executors. Lint and static contracts have no dependency
between them, so they can run concurrently. Each job retains an anonymous
workspace/cache volume that is destroyed on exit; `actions/cache` restores only
uv package downloads keyed by OS, architecture, Python version and `uv.lock`.
Registration files, credentials and virtual environments are never cached.
Runner names are generated on the host so simultaneous PID-1 containers cannot
replace one another. Database, Docker and GPU tests remain separate from this
unprivileged runner pool.

## When to throw this away

The self-hosted pool can remain in use independently of hosted Actions. Hosted
executors are an optional later addition for generic matrices; no vendor switch
is required to restore these checks.

## Verified recovery on 2026-11-15

The user service was not installed. Its existing definition is now linked,
enabled and running with lingering already enabled. Two executors registered
with distinct bounded names and accepted real queued jobs. The old runner
version, 2.336.0, was rejected by GitHub; the image now pins 2.338.2, verified
against the official runner release. Restart the service to apply later image
updates, after checking that its executors are idle.

CodeQL's JavaScript extractor requires Node on PATH; its workflow now provisions
Node 22 explicitly. Its matrix is limited to one concurrent job so static checks
retain an executor. Dependency downloads are saved after a successful uv sync,
so a later test failure does not force another cold install.

Before each new registration the supervisor checks the official runner release
using a one-hour API cache, compares the image's version label, and rebuilds when
needed. An image-update lock prevents duplicate work by the two executors. The
version build argument follows the stable dependency layers, so an update reuses
apt/uv layers. A listener rejected for age therefore refreshes on its next
registration instead of remaining offline until someone edits a pin.

The first real static run exposed a separate application-data dependency:
`host_metadata` imports the ignored `billing/stripe_catalog.json` at collection
time, and the authorization tests fail with `KeyError: service_tier_catalog` in
a clean checkout. This remains a failing gate. Restoring a runner is not proof
that those tests, or the full database suite, pass remotely.

Base-image digests are pinned in the API, frontend and runner Dockerfiles.
Refresh those pins deliberately when taking upstream updates. The API installs
locked dependencies before copying application sources, preserving that layer
on source-only edits. Cold/warm end-to-end deployment speed is not yet measured.
