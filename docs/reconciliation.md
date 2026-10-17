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

Migration `123` adds requests for changes to current attempt-bound commands and
leases in the mutation's own transaction. ACK/NACK, retry/dead-letter, release,
fencing, shortened deadlines and removal of authority wake the owning job.
The join verifies job, current attempt, host and fence. Historical attempts and
unbound admin commands do not route requests. Healthy lease extensions and
duplicate ACK/release calls do not add work. Wall-clock expiry without a write
remains covered by periodic controllers; this is not a new timer per lease.

Coalescing preserves any existing claim and never postpones a request. Runtime
writers lock commands/leases before the request trigger touches the queue;
the reconciler already owns its queue row, so its stop/lease-expiry repairs use
`NOWAIT` runtime locks. Contention rolls the entry's savepoint back and records a
retry. After the queue transaction commits, a waiting writer commits its change
and makes the request immediately due again. No repair bypasses fencing or
retries an exhausted command automatically.

## Expired-lease repair

`XCELSIOR_RECONCILE_ACTION_JOB_LEASE_AUTHORITY_MISSING=report_only` is the default.
After reviewing findings, `enforce` delegates to `expire_stale_leases` for only
that job's current matching attempt/host/fence. An unclaimed offer must be past
its deadline; an active lease must be past its expiry plus the existing
30-second grace. The controller refuses missing, released, rebound or otherwise
unproven authority. It never manufactures a replacement lease or shortens grace.

Settlement, allocation release, pending-command cancellation, the expiry outbox
event and finding evidence share one transaction. Runtime locks use `NOWAIT`;
contention retries through the queue's savepoint. Stop/cancel/terminal intent
survives, and restart intent is consumed once. The controller then re-reads the
job before resolving findings. A running attempt becomes `lost`, which does not
prove that the physical container stopped. Duplicate repairs add no event or
new attempt. Set the flag back to `report_only` and restart to disable this
remedy; committed expiry is not undone. Startup/deploy checks reject invalid
values, and Compose passes the flag to API, maintenance and scheduler services.

Gates: `tests/test_job_reconcile_lease_repair.py` plus the existing lease,
queue, lifecycle and runtime-event suites. No production enforcement is enabled
by this change; active-meter repair uses the separate billing authority below.

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
first because the application requires its declared schema floor. Downgrading
`123` removes only the runtime triggers/functions, retaining requests and the
job-change trigger from `122`; deploy the preceding application first.

## Active-meter repair and settlement

The `billing_meter_reconcile` maintenance task runs every 300 seconds under its
existing API database identity. Scheduler/reconciler roles gain no meter-write
privileges. All billing actions remain independently report-only by default:

| Setting | Opt-in action |
| --- | --- |
| `XCELSIOR_RECONCILE_ACTION_BILLING_ACTIVE_METER_MISSING` | Open a missing active attempt meter |
| `XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER` | Reconstruct missing terminal usage |
| `XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER` | Settle an open meter after terminal authority exists |

Compose passes these settings to API and maintenance; startup/deploy preflight
rejects values other than `report_only` or `enforce`. The job controller continues
to report `job_active_meter_missing`; billing owns `billing_active_meter_missing`
findings and their repair evidence. On its next pass the job controller resolves
its finding when the active meter exists.

Active repair locks and revalidates current job/attempt/host/fence, running
intent/status, server-recorded start, and an active unexpired matching lease.
It requires exactly one matching running workload in the latest observation,
received within 180 seconds by the API. Locks also serialize host observation
receipt and price changes. `NOWAIT` contention records `authority_locked` and
leaves repair for a later sweep. Rebound/stopped attempts, stale or conflicting
observations, missing owner, and missing/nonfinite/negative prices are refused.
Spot repair requires the locked rate and an explicit positive GPU count; it
never fetches a replacement quote. On-demand follows the existing explicit
host-rate model. This is not historical quote reconstruction.

A repaired meter captures its owner, start, and rate with `completed_at=NULL`
and zero duration/cost. It causes no wallet debit, invoice line or provider
payout. One meter per attempt remains enforced by the existing unique index.
Repair and action evidence commit or roll back together. A pre-existing closed
meter on a running attempt is reported and never reopened.

`BillingEngine.meter_job` and orphan repair settle the same row from the bound
terminal attempt's database end time and captured rate, including a zero rate.
Missing, reversed or future end times and conflicting host/job ownership refuse
settlement. No current-time or current-price fallback fills missing evidence.
Concurrent/repeated completion returns the same settled meter. Missing-terminal
repair now uses the caller's transaction and preserves locked spot pricing
fields instead of dropping them. Provider settlement refuses any open meter.
This repair does not debit wallets or prove physical container termination.

Before enabling active repair, review findings and prove the completion or
orphan-settlement path in the target environment. Orphan enforcement is separate;
active repair alone cannot settle a terminal event that no completion writer
processed. To disable a remedy, restore its flag to `report_only` and restart
API/maintenance. Existing open meters remain obligations and must still settle;
do not delete them or roll back to an application that treats every existing
meter as completed. Rollout, false-positive measurement and real worker-to-billing
acceptance remain outstanding.

Gates: `tests/test_active_meter_repair.py`, `tests/test_billing_controller.py`,
`tests/test_attempt_scoped_usage_meters.py`, `tests/test_provider_settlement.py`,
`tests/test_invoice_reads_match_billed_usage.py`, and startup/role gates.
No job controller action creates a meter, assumes an exit code, or restarts a
job without established execution authority.

## Acceptance and remaining work

The PostgreSQL gates are `tests/test_job_reconcile_queue.py`,
`tests/test_job_reconcile_requests.py`, and
`tests/test_job_reconcile_observations.py`. Migration gates exercise empty
bootstrap and `122`/`123` up/down/up over representative lifecycle and billing rows.
`tests/test_reconcile_runtime_events.py` proves atomic wakeups, rollback,
coalescing, renewal/replay suppression and concurrent writer/repair contention.
The populated input is a reproducible production-shaped fixture, not a copy
of production data.

This increment has not been deployed or enabled on production. Measure the
finding rate and investigate false positives before enabling a remedy. B2.9
still needs the remaining lifecycle/billing remedies and deployed
recovery/failure drills. General
job submission consolidation and retirement of legacy writers remain separate
programme gates.

## Isolated worker-to-invoice acceptance

Run the explicit hardware gate on a workstation with local Docker, the NVIDIA
container runtime, a GPU with at least 1 GiB free, and a cached PyTorch CUDA image:

```sh
python scripts/run_security_gates.py --worker-billing-acceptance \
  --gpu-device 0 \
  --cuda-image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --output /tmp/xcelsior-worker-billing-acceptance
```

This creates a disposable local PostgreSQL cluster and a loopback HTTP API,
with synthetic jobs and a newly issued host token. It refuses remote/existing
database targets and Docker contexts. The cached image is resolved to its exact
image ID and runs with no network, host mounts or privileges, bounded memory/CPU,
and a 90-second internal deadline. Only the gate's named containers are removed.
Worker tokens/journals live inside the temporary state directory, including when
the same runner executes regular worker tests.

The gate places two attempts through the real scheduler and exercises worker
command claim/ACK, lease claim, CUDA computation, running observation, active
meter repair, exit monitoring, fenced terminal reporting and invoice generation.
Exit 0 must become `succeeded`; exit 7 must become `failed`. Both must release
allocations/leases, settle their original meter exactly once, reject stale
running reports, and match invoice line micro-CAD. Display totals are checked
separately using the currency rounding rule. A queued job, an observation without
matching authority, a zero-cost terminal meter or an incomplete cleanup fails.
Logs, source hashes and `worker_billing.json` remain in the output directory.

This proves the worker protocol/observation/exit path against real CUDA and
PostgreSQL. The harness launches the container after lease claim; public launch,
`run_job` provisioning, deployed service identities, private ingress, wallet debit,
provider payment and lost-terminal-report recovery remain separate gates. It does
not enable production reconciliation policies. See
`docs/review/2026-10-17-worker-billing-acceptance.md` for executed evidence.
