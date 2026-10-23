# Retire lifecycle projection triggers

Migration **131** removes the job and host projection triggers and their
functions. Native job fields now remain authoritative through metadata,
compatibility-status and priority-column writes. In particular, late worker
reports no longer erase controller stop intent, and raw priority changes no
longer discard a persisted effective-priority decision. The qualified native
writers perform their explicit lifecycle transitions in the existing fenced
transactions.

New hosts default to `administrative_state='pending'`. The validated
`ck_hosts_admission_authority` constraint refuses admitted or draining
administration unless normalized `admission_state` is admitted. Caller JSON
cannot confer that authority. The contract permits conservative pending or
disabled administration for an admitted host; availability remains an observed
state. Migration does not infer admission from payload, promote pending hosts,
or rewrite inconsistent historical authority. Inconsistent rows refuse the
host unit for review.

Host and job changes commit in separate bounded-lock transactions. Both units
are idempotent, so a timeout after host contraction can leave the ledger at 130
with the host unit committed. Retrying completes the job unit. Downgrade
restores the exact 130 job and 082 host function/trigger definitions and prior
host default, without replaying their backfills or changing stored rows. Its
interrupted second unit is also resumable. An unreviewed function dependency
refuses removal; there is no cascading drop.

Current-schema fixtures now supply explicit phase, desired state, priority,
queue time and host scheduling state. Their simulated transitions write the
native fields they intend to change. Tests for historical migrations retain
the original legacy-shaped inserts and trigger behavior. No test-only trigger,
SQL interceptor or global normalization supplies missing production fields.
Invalid lifecycle values now raise constraint violations instead of being
silently normalized. The tests that previously used trigger presence as a
schema prerequisite check native schema columns instead, preserving execution
after the bridge is gone.

The production-derived rehearsal uses the earlier minimized fixture: 364 jobs,
the four hashed jobs' exact execution evidence, minimized host/other job
payloads and no wallet-hold references. It recovers four exact specs and keeps
360 unavailable assessments. Upgrade through 131 and downgrade/replay preserve
the other job fields, host records and execution histories. This is a scoped
data/DDL rehearsal, not a financial or complete system restore. The aggregate
audit now reports actual trigger retirement separately from complete spec
qualification; the complete-spec CLI correctly returns exit 2 for the gaps.

Final validation, source hashes, dataset/runner provenance and disposable
service cleanup are recorded in the
[machine-readable receipt](2026-10-22-projection-trigger-contract.json).
The expanded run selected 3,096 tests: 3,091 passed and five failed on one
billing fixture's obsolete `terminal` phase. Its completed-job fixture now
supplies `phase='succeeded'` and the prior effective desired state explicitly.
All five pass in the final **49-test** migration/native/token/metrics/billing
gate, with no failures, errors or skips. The production-derived replay also
passes. The original expanded failure receipt is retained alongside the
correction and rerun; the full expanded run was not repeated after that
fixture-only correction. The same three baseline exclusions remain.

Black checks all 71 changed Python files; repository Ruff, Pyright (247 files,
zero errors/warnings), and all 12 worker signatures pass. Frozen rollback SQL
spaces are encoded as Python escapes: the whole parsed module and SQL bytes
are identical, and migration/rollback tests were rerun on the final spelling.
Owned PostgreSQL/Redis services and mock artifact files were removed. The
read-only production probe preserves all 12 protected job hashes and eight
report-only policies; nine unrelated dirty root files remain unchanged.

The focused gate is runnable from the repository root with the project Python
environment active. Its runner owns a disposable PostgreSQL cluster and never
uses an application database:

```sh
python api/scripts/run_security_gates.py --output /tmp/xcelsior-trigger-check -- \
  tests/test_projection_trigger_retirement.py \
  tests/test_native_projection_writers.py \
  tests/test_host_token_survives_the_auth_middleware.py \
  tests/test_terminal.py::TestPrometheusMetrics \
  tests/test_billing_controller.py -vv
```

Production rollout still requires migration 126, the native API/agent/service
handoff, 127, then the later data/column migrations. Before 131, **every** API,
scheduler and maintenance replica must run the native writers qualified at
`c0f4e4f4` or a qualified descendant. Running `alembic upgrade head` ahead of
that handoff is not a supported rollout. Runtime floor remains 126 so those
writers can be deployed before contraction. This change has not deployed the
candidate or applied migrations 126–131 to production; repairs remain
`report_only`.

B16 stays open. This scoped migration does not make the 360 missing specs
qualified, remove the historical payload bridge, eliminate every compatibility
status/JSONB reader, restrict every legacy status-column write, finish schema
contraction, or implement durable fairness/preemption. Admission replay and
observation-time spec construction, worker restart resource/log cleanup and
terminal-abort NACK review findings also remain separate fixes. The source
trigger prerequisite for durable fairness is now available; live enablement
still requires the staged handoff and migration acceptance.
