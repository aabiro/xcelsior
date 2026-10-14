# Control-plane execution — 2026-10-14

The first implementation increment closes silent loss of job-reconciliation
requests and unsafe legacy inference-response reuse. The complete programme
remains open. New job findings default to report-only; fenced stop repair is
available behind a per-type flag. No new remedy has been enabled in production.

## Repository and execution evidence

- Started from clean `main` at `2ede8974`; this increment is local working-tree
  work on `main`, not a deployment.
- Created a separate local PostgreSQL database,
  `xcelsior_goal_20261014_1419e9cb`, owned by the existing test role. Migrated
  from empty through head `121`, then upgraded to `122`; production databases
  were not changed.
- Four queue regressions failed against the previous implementation: silent
  job success, silent unsupported-controller success, SQL-error poisoning of
  the batch, and ignoring a competing resource advisory lock.
- Five cache regressions failed before the fix: four authenticated HTTP cases
  returned a seeded legacy response despite identical/different role/case/
  temperature context; the repository also returned a textually similar prompt.
- Combined acceptance: **227 passed, 1 skipped** across 20 test files. The skip
  is `test_migration_ledger.py`'s contract-cleanup check: B16.2 has not created
  that migration. This is an outstanding programme item, not a passed gate.
- Ruff, targeted Pyright, and `git diff --check` passed. The complete backend
  and frontend suites and current production configuration were not verified.

The second increment added durable enqueue/recovery and fenced stop repair:
**210 passed** across reconciliation, lifecycle, schema, startup, deployment-env
and background-worker suites. A queue test initially counted unrelated durable
requests; it now processes only its prioritized fixture entries without deleting
other work. Migration gates: **34 passed, 1 skipped**, including empty bootstrap
and populated up/down/up. The same contract-cleanup skip remains outstanding.
Eight additional regressions then demonstrated lease expiry undoing stop or
terminal state and leaving stale payload intent; after the fix the job,
lease and resume/restart suites passed **61 tests**. These counts overlap and
are separate runs, not an aggregate total.

The protected test runner and logs are under
`/tmp/xcelsior-control-plane-goal/`. `acceptance.log` holds the combined result;
`regression-before.log` and `cache-safety-before.log` hold the confirmed failures.
The runner supplies the isolated DSN without printing its credentials.

## Implemented behavior

`control_plane/job_reconcile.py` compares job/attempt/host/fence/spec identity,
lease validity, API-receipt observation freshness, stop-command delivery and
active-meter presence. It records tenant-attributed, deduplicated findings and
resolves only its own finding types. A fresh matching running observation
projects the desired generation; an exited, paused, unknown, mismatched or
unversioned workload cannot establish successful running/terminal state.

`process_due` dispatches jobs, retains periodic checks for active jobs, takes
resource advisory locks and contains each entry in a savepoint. Failed entries
keep their error/backoff; partial findings or actions roll back. Unsupported
controllers retain their request instead of being counted as reconciled.

Migration `122` enqueues relevant job mutations in their own transaction.
Observations enqueue current attempts belonging to their reporting host, without
trusting workload-supplied job IDs. The 60-second recovery task fills a bounded
batch of missing active-job requests. Job locks are skipped during queue
processing so a writer waiting on the queue cannot deadlock that controller.

`job_stop_command_missing` can enforce through the lifecycle domain using the
same caller transaction and original intent/attempt idempotency key. It checks
the full current lease authority, preserves stop/removal intent, and refuses
existing exhausted commands. Commands for a different intent cannot satisfy the
current request. Lease expiry preserves terminal/stop intent; an explicit
restart is requeued once and consumed. The attempt remains `lost`, preserving
the distinction between authority expiry and proof of physical container stop.
Operator instructions are in `docs/reconciliation.md`.

The OpenAI chat proxy always executes and meters fresh inference. It does not
read/write the prompt-only cache, including when the former enablement flag is
set. Historical cache reads remain exact for migration; string approximation
and fabricated cache savings are removed from the serving path. Requests that
previously hit this cache can incur additional model latency and token charges.
Full-context cache v2 remains necessary before response reuse can return.

## Reconciled planning records

The Track B checklist now records head `122`, B2's existing launch machinery,
the job-controller increments, the existing live-event cursor,
telemetry retention/alerting, service readiness and durable privacy deletion.
Incomplete requirements retain `[~]` with their exact residual. In particular,
bounded raw telemetry is not the companion's complete downsampling/partitioning
decision, simulated readiness failures are not a deployed failure drill, and
deletion workflow tests with controlled sink handlers do not prove every
external sink or durable export parity.

The gate truth table covers P0–P7 and keeps its recorded clause verdicts. It
now separates those narrow acceptance clauses from pipeline execution,
migration persistence and Track B implementation. Historical NFS/host/webhook
observations require fresh probes before they can explain a current failure.

## Remaining execution order

1. Complete the remaining B2.9 lifecycle/active-meter remedies and immediate
   command-failure/lease-only request coverage through the existing authorities.
   Keep staged enforcement and prove idempotency, crash rollback and default
   report-only behavior for each new action. Measure
   production findings before enabling remediation. Consolidate remaining
   submission paths behind the launch service without weakening approval or
   wallet authority.
2. Expose/enforce endpoint and MCP client budgets; finish durable lifecycle
   emitters/checkpoint delivery and artifact verification/deletion consistency.
   Keep warm-versus-active pricing as an explicit product decision.
3. Wire real pipeline stage execution and durable migration history/callers/UI.
   Preserve approval, budget, fencing and checkpoint-integrity gates.
4. Refresh staging and execute complete access, promotion, two-host migration,
   provider payout and multi-node sweep journeys. Retain fresh proof, including
   deliberate changed-member detection and byte-identical artifact reads.
5. Complete UI acceptance and deployed identity/replica/restart/draining/
   rollback/recovery/chaos/runbook gates. Inspect the deployed commit and
   service configuration before deciding which historical operator residuals
   still apply.
6. Validate retrieval models before fixing dimensions or adding vector schema;
   build retrieval/cache v2, governed analytics and IaC. Retire compatibility
   paths only after their measured usage is zero for the required interval.

The authoritative residuals remain in `docs/track-b-implementation-checklist.md`
and the expanded agent-native plans. No operator residual or live journey is
marked complete by this local increment.
