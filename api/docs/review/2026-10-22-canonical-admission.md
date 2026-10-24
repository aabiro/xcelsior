# Canonical job admission — 2026-10-22

New PostgreSQL job admissions now populate the existing `jobs.spec` and
`jobs.spec_hash` columns in the submission transaction. The execution snapshot
contains an explicit set of workload, storage, runtime, placement and billing
constraints. Job IDs, status, timestamps, container observations and callback
credentials are outside that snapshot. Nested inputs are detached from caller
objects before persistence.

The two hashes have distinct meanings:

- `spec.launch_spec_hash` retains the logical workload hash from an approved
  launch plan. Its exact canonical arguments remain in `spec.launch_spec`.
- `jobs.spec_hash` binds the execution snapshot, including server-selected
  owner, egress policy, exclusivity and serverless endpoint/worker bindings.
  Attempts and start commands carry this execution hash. The existing fenced
  worker independently recomputes it before claiming the lease.

Launch-plan execution passes its approved arguments to submission, which
refuses disagreement before writing a job. Retrying the same submission ID
leaves existing status and placement untouched; different execution inputs
under that ID are refused. Concurrent retries create one job and one outbox
submission event. Ordinary observation upserts preserve the admitted spec.

Native canary scope, scheduling filters, fleet capacity accounting and locked
reservation read the admitted snapshot when present. Reservation rechecks its
hash and derives GPU/VRAM requirements under lock, uses its host pin and serverless
binding, and creates the command from that same snapshot. A changed hash
between claim and reservation, corrupt content, absent hash or unsupported
execution version cannot acquire execution authority. The scheduler records
`execution_spec_unqualified` with backoff for a claimed invalid spec.
Trace context remains outside execution identity and still reaches attempts.
Serverless callback minting/authentication, cancellation, fenced stop and
settlement, lease expiry and mining attribution use the same stored execution
identity. Mutable stop intent and observations remain outside it.

Pre-cutover rows remain explicit technical debt: a row with no stored spec
uses the existing payload bridge and freezes that exact evidence at its first
successful reservation. A previously recorded hash must agree; it is never
replaced to make evidence appear qualified. This bridge still needs a reviewed
historical qualification/backfill and removal before the schema can require
canonical specs and drop legacy payload authority. Existing historical records
are not rewritten by this source change, and there is no new migration in
this slice; the required spec columns already exist from migration 054.

The read-only production inventory confirms the qualification boundary:
all 364 historical jobs lack a stored spec; 360 lack a hash and four retain
hashes that differ from their current mutable payload. Both transitional
projection triggers still exist. Copying current payload into all spec columns
would therefore misrepresent recorded execution for those four rows. The
remaining migration must qualify retained execution evidence or record an
explicit historical disposition instead of inventing approved specs.
All four saved hashes have matching retained start-command and attempt evidence,
providing a recorded source for that subsequent backfill.

The runnable gates are `tests/test_job_spec_admission.py`,
`tests/test_launch_plan_execute.py`, `tests/test_control_plane_reservation.py`,
the launch/claim/shadow regression suites and the expanded isolated security
runner recorded in the [machine-readable receipt](2026-10-22-canonical-admission.json).
The expanded run passes **2,513 tests**, with no failures, errors or skips and
the same three explicit baseline exclusions. The focused gate passes **199
tests**; Black checks all 18 changed Python files, repository Ruff and Pyright
pass, and all 12 worker distribution signatures verify.
The worker handoff fixture drives the real FastAPI claim/status paths and
disposable PostgreSQL while isolating Docker execution. This does not prove a
live GPU startup or production handoff.

Production remains unchanged at schema 125 with repairs `report_only`.
Historical qualification, remaining JSONB compatibility, canonical column and
trigger contraction, all-table semantic review, durable fairness/preemption
and coordinated rollout remain open in B16.
