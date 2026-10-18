# Control-plane runtime reconciliation requests — 2026-10-16

This increment closes B2.9's immediate command-outcome and lease-change request
gap. The remaining lifecycle/active-meter remedies and production rollout gates
remain open. CI cockpit is outside this work's scope.

## Behavior and rollout

Migration `123` requests job reconciliation in the same transaction as changes
to `agent_commands` or `placement_leases`. Requests require a matching current
job/attempt/host/fence. Command failure, ACK/NACK, delivery retry/dead-letter,
lease claim/release/fencing, shortened deadlines and removal of authority wake
the job. Healthy renewal extensions and duplicate ACK/release calls add no
work. Existing queue claims survive, and coalescing never postpones due work.
Wall-clock expiry without a database mutation still uses periodic controllers.

Reconciler repairs already hold queue rows; runtime writers hold command/lease
rows before their triggers request those queue rows. Stop-command repair and
lease-expiry repair now acquire runtime rows with `NOWAIT`. Contention rolls
back the entry's savepoint and retains a retry. A waiting writer then commits
its event and advances the request's due time. Fencing, expired-lease rejection,
command retry budgets and default report-only behavior remain enforced.

The migration has no backfill. Its downgrade removes only the added triggers
and functions, preserving requests, commands and billing data. The application
schema floor is `123`; deploy a compatible preceding application before
downgrading. No production database or remediation setting was changed.

## Executed evidence

Work started from `8b2a18bd` in the isolated worktree
`xcelsior-worktrees/reconcile-events-20261016`, preserving unrelated changes in
the primary checkout. Tests used a separate local PostgreSQL database,
`xcelsior_reconcile_events_20261016_962e12e7`, with credentials supplied by a
protected runner rather than printed in commands or logs.

- Before the fix: **13 failed, 2 passed** for the initial runtime-event and
  lock-contention regressions. The two negative controls already passed.
- Final runtime and migration-ledger run: **166 passed, 1 skipped** across
  runtime events, job queue/observations/requests, reconciliation, command
  outbox, leases, fenced lifecycle, reservations and the migration ledger.
  The skip is the existing B16 contract-cleanup migration gate; that migration
  is not implemented and the skip does not satisfy it.
- Separate migration acceptance: all **10** empty-bootstrap, populated-snapshot
  and schema-floor tests passed. The populated fixture is reproducible test
  data, not a production backup. The new `123` upgrade/downgrade/upgrade test
  preserves queued work, command state, legacy row counts and billing totals.
  That run also exposed a stale README head assertion; it was corrected and
  the full ledger passed in the final run above. These are overlapping runs,
  not an aggregate test count.
- The expanded event suite includes a real concurrent lease writer versus
  queue-owning reconciler, proving a retry and subsequent wakeup without a
  deadlock. An initially incorrect expired-renewal expectation was corrected
  to require `LeaseRenewRejected`; lease authority was not weakened.
- Ruff on changed Python files, targeted Pyright on the four changed
  control-plane modules, and `git diff --check` passed.

Reproducible gates:

```sh
python -m pytest tests/test_reconcile_runtime_events.py \
  tests/test_job_reconcile_queue.py tests/test_job_reconcile_observations.py \
  tests/test_job_reconcile_requests.py tests/test_control_plane_reconcile.py \
  tests/test_control_plane_commands_outbox.py tests/test_control_plane_leases.py \
  tests/test_fenced_lifecycle_controller.py tests/test_control_plane_reservation.py \
  tests/test_migration_ledger.py -q
python -m pytest tests/test_from_empty_bootstrap.py \
  tests/test_migration_from_production_snapshot.py \
  tests/test_schema_floor_covers_what_the_code_uses.py -q
```

Supply an isolated test PostgreSQL DSN using the repository's test environment
conventions. Local evidence is under
`/tmp/xcelsior-reconcile-events-20261016/`: `before.log`, `final-runtime.log`,
`migration-acceptance.log` and `pyright.log`. These temporary logs are not a
durable CI artifact or production proof.

## Remaining work

B2.9 still needs remaining lifecycle/active-meter remedies through the existing
domain authorities, per-type staged enforcement, production false-positive
measurement and deployed recovery/failure drills. This increment did not run
the full backend/frontend suites, provision or cut over service roles, deploy
the migration, or exercise live workloads.

The wider execution order remains in
[`2026-10-14-control-plane-execution.md`](2026-10-14-control-plane-execution.md);
its immediate command/lease event item is completed by this increment. The
authoritative unfinished requirements remain in the Track B checklist and
expanded plans. No other programme item is marked complete by these checks.
