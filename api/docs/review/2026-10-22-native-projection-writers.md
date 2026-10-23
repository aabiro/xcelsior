# Native lifecycle writers and trigger removal rehearsal

Job persistence, fenced attempt reports, lifecycle stop/resume/restart, lease
expiry and unstarted serverless cancellation now write native lifecycle fields
in their existing transactions. With the projection triggers removed, a worker
observation changes phase while preserving controller stop intent. Requeue
records one new submission time in the native queue column and compatibility
payload. Unrelated observations keep
effective priority and fair-share state; an actual administrative priority
change updates effective priority deliberately.

Host persistence derives administrative and availability state from status and
the current normalized admission authority. Registration cannot admit a host.
Inventory upserts ignore caller admission fields and rebuild their compatibility
mirror from the locked row, including the decision ID and version. Operator
admission decisions write the native states alongside the recorded decision.

The writer audit also found privacy deletion dependencies. Workload withdrawal
now explicitly writes phase and desired state. Provider deletion drains admitted
hosts, keeps pending hosts pending, and leaves disabled hosts disabled. Its
availability value is valid without a trigger rewriting it.

Periodic and preemption wallet billing now select identity, GPU count/model and
pricing constraints from the admitted spec, falling back to payload only when
the historical spec is null. Both paths exclude admitted serverless workers
from generic instance billing even if observations omit or replace `job_type`.
Attempt metering also reads workload inputs from the admitted spec; its captured
owner, rate and server timestamps remain authoritative.

`tests/test_native_projection_writers.py` drops **both actual projection
triggers and functions** in disposable PostgreSQL, drives the production
writers and restores the original DDL for surrounding regression tests. No
replacement trigger or default supplies missing fields. It covers admission,
reservation, claim, phase reports, terminal settlement, late reports after
stop/cancel/terminate/restart, fresh resume, offered/active expiry, serverless
cancellation, privacy withdrawal, provider deletion and host admission/revocation.
Real wallet checks exercise periodic and preemption charges after conflicting
observation upserts. An AST inventory gate checks runtime SQL writers for
explicit native lifecycle fields, including the isolated acceptance tools.

The [machine-readable receipt](2026-10-22-native-projection-writers.json)
records final test counts, source hashes, static checks and disposable-service
cleanup: **2,664 expanded tests** and **186 focused tests** pass, with no
failures/errors/skips and the same three documented expanded-suite exclusions.
The expanded suite includes 22 trigger-free transition checks. Black (16 Python
files), repository Ruff, Pyright (247 files, zero errors/warnings) and all 12
worker signatures pass. Both disposable PostgreSQL runs and the owned Redis
instance were cleaned up. Live GPU acceptance and service handoff remain open.
This receipt qualifies the listed writers; the shipped Alembic head
remains **130**, with both transitional triggers still present.
At that head the job bridge can still derive desired state from a late status
change; the trigger-free intent guarantee requires the permanent contraction.

Permanent trigger contraction remains a separate change. It must migrate the
remaining raw SQL test fixtures and trigger-specific acceptance assertions,
check safe defaults/constraints for host admission, and provide a reversible
contract migration plus a rollout gate after the native service handoff.
In particular, migration 054's administrative-state default remains
`admitted`; the permanent contract must change it and validate consistency
against normalized admission before allowing raw inserts without the trigger.
The review also leaves two admission issues (observation-time spec construction
and replay after server policy changes) and worker restart resource/log cleanup
and aborted-start NACK behavior open. They are not marked resolved by this gate.

Historical disposition remains independent: the earlier rehearsal recovered
four execution specs and assessed 360 unavailable histories. This writer change
does not manufacture those missing specs, contract global spec nullability,
remove the historical payload bridge, or implement durable fairness/preemption.
B16 remains open. Production has not received this candidate or migrations
126–130; the read-only boundary probe still reports schema 125, unchanged
protected jobs, no active jobs and all repairs in `report_only`.
