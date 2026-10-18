# The CI Cockpit

**Status:** plan, not implementation. Written 2026-08-07.
**Origin:** the sandboxed self-hosted runner in `scripts/ci-runner/`, built on
2026-08-06 because hosted Actions is blocked at the account level.
**Portability target:** this is written to be lifted into `pixelenhance-labs`
with two environment variables changed. See §7.

> **Nothing here is built.** This document exists so the decisions are made once,
> in writing, before anyone opens an editor. Where a choice has a real cost, the
> cost is stated rather than the option quietly dropped.

---

## 1. What problem this actually solves

Three things are true at once right now, and each one is invisible:

1. **The only working CI is a container on a laptop.** Hosted `ubuntu-latest`
   jobs fail in three seconds with `runner_name: ""`, zero steps and no logs —
   an account-level block, not a spending limit (the repository is public, where
   standard runners are free). The sandboxed runner in `scripts/ci-runner/` is
   what actually verifies pushes.
2. **Nobody can see whether it is up.** It was hand-started twice on 2026-08-06
   and died with the shell both times. A push then sits `queued` forever, which
   on the Actions page is indistinguishable from a broken workflow. It is now a
   systemd unit, which fixes the *dying* but not the *seeing*.
3. **The two sources of truth disagree, and only one is reachable.** GitHub knows
   whether a runner is registered and what a job concluded. The host knows
   whether systemd is up, whether the container is alive, and what it is chewing
   on. Neither knows the other's half.

A cockpit is the answer to (2) and (3): one screen that reconciles the external
record with the host's own account of itself, and says plainly when they
disagree. **The disagreement is the interesting signal** — a runner GitHub calls
`online` while its host has no container running is the exact failure that turns
a push into a silent queue.

---

## 2. Two data planes, and why both

The instinct is to pick one. Don't — they answer different questions and fail
in different directions.

| | **Plane A — GitHub API** | **Plane B — runner heartbeat** |
|---|---|---|
| Authority on | registration, `online`/`busy`, job conclusions, queue depth | systemd state, container liveness, host CPU/RAM/disk, image age |
| Fails when | the token expires or rate-limits | the host is down — *which is the thing you wanted to know* |
| Latency | polling, seconds | push, sub-second |
| Cost | a credential on the production VPS | an agent process and an authenticated endpoint |
| Blind to | anything host-side | anything GitHub-side |

**Plane B cannot report its own death.** That is not a flaw to engineer around,
it is the reason Plane A exists: a missing heartbeat is only meaningful if
something independent can say the runner was *supposed* to be there. Plane A
supplies that expectation.

**Plane A cannot see a wedged container.** A runner process that registered and
then hung holds `online` while serving nothing.

So the cockpit's core view is not either plane. It is the **join**, with a
reconciliation state:

```
GitHub says          Host says            State
─────────────────────────────────────────────────────────────────────
online, idle         systemd active       HEALTHY
online, busy         container running    WORKING
online, idle         no heartbeat >90s    ORPHANED     ← push will queue forever
offline              systemd active       REGISTERING  (or token failure)
offline              no heartbeat         DOWN         ← expected after a reboot
online, busy         container exited     WEDGED       ← the silent killer
```

`ORPHANED` and `WEDGED` are the two states neither plane can name alone, and
they are the two that actually cost you a day.

---

## 3. Modes

The user asked for "different modes", and the honest reading is that this must
degrade rather than require both planes to be configured. Three, in increasing
capability:

### Mode 1 — `github-only`
No agent, no host access. Backend polls the GitHub API. Shows registration,
online/busy, and the last N runs with conclusions.

*Enough for:* "did my push get verified?"
*Cannot answer:* "why is nothing picking up my job?"
*Cost:* one fine-grained PAT (§5).

### Mode 2 — `heartbeat-only`
No GitHub credential. The runner host POSTs its own state. Shows systemd,
container, resource pressure.

*Enough for:* a fleet you own end to end, where you'd rather not put a GitHub
token on a production box.
*Cannot answer:* "does GitHub think this runner exists?" — so `ORPHANED` is
undetectable.
*Cost:* an agent and an endpoint.

### Mode 3 — `reconciled` (the actual product)
Both. The state table in §2 becomes computable. This is the only mode where the
cockpit tells you something you could not have learned from the GitHub UI.

**Mode is a deployment configuration, not a build flag.** The UI renders what it
has and says what it lacks — a panel with no heartbeat plane shows
`host: not reporting (mode: github-only)`, never a plausible-looking blank.

---

## 4. Telemetry — yes, and be specific about it

"See the stats of everything" is telemetry, and telemetry is where dashboards go
to die: a wall of numbers nobody reads because none of them imply an action.
The discipline that keeps this useful — **every panel answers a question someone
actually asks out loud**:

| Question actually asked | Panel | Source |
|---|---|---|
| "Is CI up?" | reconciliation state (§2) | both |
| "Did my push pass?" | run timeline, newest first, conclusion + duration | A |
| "Why is my job queued?" | queue depth vs runner availability | A |
| "Is it stuck?" | current job elapsed vs that workflow's p50 | A + B |
| "Is CI getting slower?" | duration trend per workflow, last 30 runs | A |
| "Which gate keeps failing?" | failure count by job name | A |
| "Can this box take another job?" | host CPU/RAM/disk, container count | B |
| "Is the image stale?" | runner version vs GitHub's minimum | B |
| "How much did the eval cost?" | token spend, cache hit ratio | eval artifact |

That last row matters more than it looks. `eval-baseline.json` already records
`usage` and `estimated_cost_usd` (added 2026-08-07, $0.35 with prompt caching
versus ~$1.50 without). Cost-per-verification belongs on a CI dashboard —
it is the number that decides whether a gate runs per-push or nightly.

**The anti-goal:** no gauge whose needle nobody would act on. If a metric has no
row in that table, it does not get a panel.

### Retention

Runs are already retained by GitHub; do not re-store them. Heartbeats are
high-frequency and low-value after an hour — keep raw for 24h, then roll up to
5-minute buckets for 30 days. A `ci_heartbeats` table with a partial index on
recent rows, dropped by the same retention job pattern the platform already
uses.

---

## 5. The credential decision, stated honestly

Mode 1 and 3 put a GitHub token on the production VPS — the same box holding
`~/.ssh/xcelsior` (the production deploy key) and a 325-line `.env`. That is not
nothing, and `scripts/ci-runner/README.md` was written specifically about what
lives there.

**The shape that makes it acceptable:** a *fine-grained* PAT, scoped to one
repository, with `Actions: read-only` and nothing else. It cannot push, cannot
read code beyond what a public repo already exposes, cannot alter workflows. It
is revocable in one click and observable in the account's token list.

**What would not be acceptable:** a classic PAT with `repo` scope, which on a
public repository is a push credential. If the implementation reaches for one
because it is easier to mint, that is the moment to stop.

**Enforce it rather than document it.** The runner already refuses to start when
the docker socket is mounted or a credential is present in the sandbox
(`entrypoint.sh`). The same discipline applies here: the backend should verify
at startup that its token cannot write, by calling an endpoint that would
succeed with write scope and asserting the failure. A comment asking for a
read-only token is not a mechanism — that lesson cost this repository two
separate scope drifts in one week.

---

## 6. Heartbeats — reuse, do not invent

The platform already has this exact shape: `worker_agent.py` reports host state
on a timer, authenticated as a machine principal, and `routes/agent.py` receives
it. A CI heartbeat is the same pattern with a different payload, and it should
reuse:

* the machine-credential auth path, not a new bearer scheme;
* a dedicated scope — `ci:report` — because the existing worker scopes are about
  GPU workloads and a CI reporter must not inherit them;
* the same `_require_scope` enforcement, so this does not become the fourth
  place where a capability is promised to a credential that cannot reach it
  (that mistake happened twice on 2026-08-06 and both times every layer was
  correct in isolation).

**Payload** — deliberately small, because a heartbeat that carries a lot becomes
a thing people query instead of a liveness signal:

```json
{
  "runner_name": "xcelsior-ephemeral-1",
  "systemd_active": true,
  "container_running": true,
  "current_job_started_at": "2026-08-07T03:44:30Z",
  "runner_version": "2.336.0",
  "host": {"cpu_pct": 41.2, "mem_pct": 63.0, "disk_free_gb": 812},
  "last_exit_code": 0
}
```

**Cadence:** 30s while idle, 10s while a job runs. `ORPHANED` fires at 90s of
silence — three missed idle beats, so a single dropped request is not an alert.

---

## 7. Extract, or leave it in the repository?

The honest answer is **both, in sequence** — and the reason is that the two
halves have genuinely different portability.

### The runner is nearly generic already. Measured:

| File | Project-specific references |
|---|---|
| `Dockerfile` | **0** |
| `run-runner.sh` | 5 — repo default, image name, workflow path (×3) |
| `entrypoint.sh` | 2 — one is the forbidden-paths list, which *should* be per-project |
| `xcelsior-ci-runner.service` | names and paths, as expected |

Seven substitutions. `run-runner.sh` already reads `XCELSIOR_CI_REPO` from the
environment; generalising means `CI_REPO`, `CI_WORKFLOW`, `CI_IMAGE` and
`CI_FORBIDDEN_PATHS`, and templating the unit. That is an afternoon.

### The cockpit splits, and only one half is hard to move.
The **embedded card** (§8a) is a page inside an authenticated admin panel with
this platform's auth, theme, chart library and API client. Extracting *that*
means extracting the admin shell — a different and much larger project.

A **standalone cockpit** (§8b) has the opposite property: it is a client of one
JSON endpoint, so it is portable by construction and its stack is a free choice
(§8c). Pointing it at another repository's deployment is a base URL and a token.

**Which is why the endpoint is the thing to design first.** `GET /api/admin/ci/status`
is the portable artifact; every UI — embedded card, standalone app, terminal —
is a consumer of it. Get the contract right and the extract-or-not question stops
being architectural and becomes a matter of taste, which is where it belongs.

### So:

**Extract the runner. Leave the cockpit.**

*Reasons to extract the runner:*
- It is genuinely reusable and the value is the threat model, not the code — the
  two enforced refusals (fork-triggerable workflow; credentials inside the
  sandbox) are the parts worth having, and they are what someone copying a
  Dockerfile from a blog post will not get.
- `pixelenhance-labs` has the same problem shape: a public repository, a machine
  with capacity, and no reason to pay for hosted minutes.
- A package with its own README forces the threat model to be re-read on adoption
  rather than inherited by copy-paste.

*Reasons to leave it where it is:*
- Every extraction adds a version skew. The runner is currently pinned to this
  repository's workflow by an *enforced check* — `run-runner.sh` refuses to start
  if `gates-sandboxed.yml` gained a `pull_request` trigger. Generalising that
  check to "whatever workflow you name" makes it weaker unless the consuming
  repository is disciplined about naming.
- Two consumers is not yet a library. One repository using it and one that might
  is the classic point at which premature extraction costs more than it saves.

*The tie-breaker:* extract only when `pixelenhance-labs` actually needs it. Until
then, the seven substitutions are a known, small, written-down cost — and this
section is the record so nobody re-derives it.

---

## 8. The cockpit UI — two artifacts, not one

An earlier draft of this section argued for restraint across the board. That was
wrong, and the mistake is worth recording rather than quietly editing out: it
took the constraints of *one* artifact — a card wedged into an existing admin
page, glanced at while something is on fire — and applied them to a *different*
artifact that does not have those constraints. They are separate products with
separate budgets.

### 8a. The embedded card (`admin/infrastructure`)

Genuinely constrained, and not by taste. It inherits the admin theme, sits beside
fleet health, and competes for a few hundred pixels. It is a **readout**: state,
last runs, live job, reconciliation line. Density and monospace numerics here are
a consequence of the space, not an aesthetic position.

### 8b. The standalone cockpit — as expressive as you want

If it becomes its own application (the §7 branch, still optional), essentially
none of 8a's constraints carry. It has the whole viewport, its own stack, and one
job it can do properly instead of squeezing. That is the case for building it
separately at all — and "make it look like a cockpit" is a legitimate reason,
not a frivolous one, because a thing people enjoy opening is a thing that gets
looked at.

Full-bleed layouts, animated state transitions, a run strip that actually reads
like telemetry, sound on state change, an ambient idle mode for a spare monitor,
CRT/scanline treatment if that is the register you want. None of it is in tension
with the tool's purpose once it is not fighting for room inside another page.

**Three things survive in both, and they are correctness rather than taste:**

1. **The state is legible without colour.** `ORPHANED` reads as the word, not as
   "the amber one" — it survives a colourblind reader and a monochrome
   screenshot pasted into an issue. Colour on top of the word, freely.
2. **Numbers that get compared are monospace**, so digits line up between rows.
   Everything else can be whatever typeface suits.
3. **State changes are announced**, not only shown — a live region for screen
   readers, and a real focus ring on anything interactive.

Those three cost nothing and hold at any level of visual ambition. Beyond them,
this document has no opinion.

### 8c. Stack, if it is standalone

Deliberately not decided here — it depends on who maintains it and whether it
ever leaves this machine. The honest trade-offs:

| Stack | Fits when | Costs |
|---|---|---|
| **Next.js** | you want the richest UI, and the existing frontend's components and theme are worth reusing | a second Node app to deploy and keep patched |
| **FastAPI + HTMX** | the backend already speaks Python and the UI is mostly server-rendered state | less interactive polish without more work |
| **Textual (TUI)** | it lives on the machine that runs the runner, and a terminal is where you already are | terminal-only; no remote viewing without ssh |
| **NiceGUI / Reflex** | one Python codebase for logic and UI, no separate frontend build | smaller ecosystems; you own more of the edges |
| **Streamlit** | you want it working this afternoon and will not extend it | fights back the moment layout matters |

A terminal cockpit and a browser cockpit are not competing options — the data
planes in §2 are the same either way, and both can exist against one
`GET /api/admin/ci/status`. **Design the endpoint first and the choice stops
being load-bearing**, which is the real recommendation of this section.

## 9. Sequencing

Each phase is independently useful and independently abandonable. That property
is deliberate — this is a side project competing with plan work, and a phase
that only pays off if the next one lands is a phase that will strand.

**C0 — Read-only, Mode 1.** Backend endpoint `GET /api/admin/ci/status`, a
fine-grained PAT, one card on `admin/infrastructure`: state, last 10 runs, live
job. *Ships the answer to "is CI up".*

**C1 — Heartbeat, Mode 2/3.** `ci:report` scope, agent in the runner host's
systemd unit, `ci_heartbeats` table, reconciliation state computed. *Ships
`ORPHANED` and `WEDGED`.*

**C2 — Trends.** Duration and failure-rate history, the cost row from
`eval-baseline.json`. *Ships "is CI getting slower".*

**C3 — Control.** Restart the runner, cancel a wedged job, rebuild the image.
**Deliberately last, and the phase most likely to be wrong.** A button that
restarts CI is a button that stops a verification mid-flight; it needs the same
confirm-and-preview discipline the MCP tools use (`confirm:false` returns a
preview and changes nothing), and it needs an audit row per action. If C0–C2 are
enough, C3 should not be built — it is the phase with real blast radius and the
weakest case.

**C4 — Extract the runner** as §7, *only* if `pixelenhance-labs` adopts it.

---

## 10. What would make this a bad idea

Recorded so the decision is reversible with evidence rather than by mood:

* **If hosted Actions comes back.** GitHub's own UI answers most of §4 already.
  This cockpit's value comes almost entirely from the runner being self-hosted
  and invisible — restore hosted CI and C0's case mostly evaporates.
  `scripts/ci-runner/README.md` already says the runner itself should be thrown
  away at that point; the same applies here.
* **If it becomes a second monitoring system.** The platform has Prometheus
  metrics and Jaeger. If CI telemetry wants alerting, thresholds and history, it
  belongs there — the cockpit is a *view*, not a monitoring stack, and the moment
  someone adds an alert rule to it, that boundary has been crossed.
* **If the credential grows.** The read-only, single-repository PAT in §5 is what
  makes Mode 1 acceptable on a production box. A future feature that needs write
  scope is not a small change to this document, it is a different document.
