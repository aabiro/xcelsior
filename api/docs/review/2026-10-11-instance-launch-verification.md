# Live instance launch and terminal verification — 2026-10-11

The continuation started with local `main`, `origin/main`, and production at
`15a18009`. All core production containers were healthy. `/readyz` reported
PostgreSQL, Redis, the NFS export root, and schema revision 119 ready. The
maintenance heartbeat and telemetry retention task both had successful runs.
This does not establish that every individual volume's backing data exists.

## Reproduced startup race

A deliberate interactive launch (`85991cce`) on `aaryn-tuf-rtx2060` reproduced
the failure against the real API and worker:

- The API's direct-host path assigned the legacy job and immediately started
  pulling/running its image over SSH.
- The enrolled worker polled that assignment, claimed its lease, and created
  `xcl-85991cce` first.
- The API's competing `docker run` failed with a container-name conflict and
  marked the job failed while the worker was still installing SSH support.
- The web terminal consequently rejected the failed instance.

The temporary container was removed. This was a duplicate execution path,
not an instance-placement failure or missing GPU capacity.

## Repair and validation

Direct-host submission now returns the assignment to an enrolled worker
instead of starting the same container over SSH. The legacy assigned-job
runner and its low-level SSH entry point apply the same rule. A stale
heartbeat never switches an agent host to SSH startup. Hosts without an
enrolled agent retain their existing SSH path; transactional scheduler
ownership still takes precedence.

Terminal connections made after startup has already failed now show the
recorded failure reason, just as connections waiting through startup do.
Deployment readiness checks also retain their command output in the parent
shell; background assignments previously produced false missing-probe and
disabled-PayPal warnings.

Targeted placement, terminal, and worker regressions: **163 passed, 4 skipped**.
The new cases cover v1/v2 agents, stale heartbeats, SSH-only compatibility,
checkpoint dispatch, and failure details on first connection. Scheduler/API
compatibility checks: **105 passed**. Ruff, Black on changed Python files,
the Pyright gate (zero tracked findings), shell syntax, and whitespace checks
passed. The post-repair live launch is pending.
