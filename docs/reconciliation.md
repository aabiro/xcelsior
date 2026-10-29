# Job reconciliation operation

Migration `122` enqueues a job in the same transaction as insertion or a change
to status, desired state, phase, host, active attempt, generation, or lifecycle
intent. Deletion removes that job's queue entry. Queue writes use the caller's
database role; all roles with control-table write access already have these
rights. The trigger is not a security-definer function.

Worker observations enqueue the host and jobs whose current attempts belong
to that host. An empty snapshot still enqueues those jobs. Workload-supplied
job identifiers cannot enqueue unrelated jobs. Duplicate snapshots add no work.
The maintenance task `job_reconciliation_recovery` runs every 60 seconds and
adds at most 1,000 missing active-job requests per pass. Active jobs retain a
periodic queue entry after reconciliation; existing requests are never postponed
by recovery. Terminal jobs with an active attempt remain eligible.

The scheduler processes the queue under resource advisory locks and per-entry
savepoints. A locked job is retried instead of waiting while holding its queue
row. Failure preserves the request, records its error, and backs off. A finding
is evidence of a discrepancy; it does not by itself authorize billing or prove
container termination.

## Stop-command repair

`XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING=report_only` is the default.
The setting reaches both the scheduler and the API/background-worker container
environments. Startup validation and deployment preflight reject misspellings;
the runtime action resolver also defaults unknown values to report-only.

After reviewing production findings, an operator can set this one action to
`enforce`. A missing stop command is then created through the lifecycle service
only when desired state is stopped and job, attempt, host, lease and fence agree.
The command and finding's action evidence commit together. Stop preserves the
container; cancel, terminate and restart retain their recorded removal intent.
The controller does not change a terminal projection back to running/stopping.
Enabling the flag also repairs a finding first recorded in report-only mode.

The original `intent:attempt_id` idempotency key is reused. A pre-existing
expired or exhausted command is recorded as `fenced_stop_refused`, with reason
`command_requires_operator_retry`; this controller cannot grant it a new retry
budget. Inspect worker delivery, command result and current authority before
using the existing operator command retry workflow. An acknowledged command
does not prove the workload stopped: authority stays live until the fenced
attempt/lease services settle it.

When a lease expires, the attempt is recorded as failed/lost and its authority
is cleared. Stop, cancel, termination and already terminal job states survive;
only work still desired running or explicitly restarting returns to the queue.
Restart intent is consumed and job payload status is updated with the projection.
Lease expiry remains authority loss, not proof that a remote container stopped.

To disable repair, restore the setting to `report_only` and restart the
scheduler. Already committed commands remain delivery obligations. Findings,
periodic checks and recovery continue. Downgrading `122` removes the trigger
and function but retains requests and findings; deploy the prior application
first because the new application requires schema `122`.

Billing actions remain independently controlled by
`XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER` and
`XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER`, also report-only by default.
No job controller action creates an active meter, assumes an exit code, or
restarts a job without established execution authority.

## Acceptance and remaining work

The PostgreSQL gates are `tests/test_job_reconcile_queue.py`,
`tests/test_job_reconcile_requests.py`, and
`tests/test_job_reconcile_observations.py`. Migration gates exercise empty
bootstrap and `122` up/down/up over representative lifecycle and billing rows.
The populated input is a reproducible production-shaped fixture, not a copy
of production data.

This increment has not been deployed or enabled on production. Measure the
finding rate and investigate false positives before enabling a remedy. B2.9
still needs the remaining lifecycle/billing remedies, immediate command-failure
and lease-event request coverage, and deployed recovery/failure drills. General
job submission consolidation and retirement of legacy writers remain separate
programme gates.
