# Native spot admission, pool reservation and quote integrity

**Accepted follow-up:** deployed as `1a0ae249`/schema 141 after 4,064 passing checks and real spot CUDA/billing acceptance. The [executed receipt](2026-10-25-native-release-production-handoff.md) supersedes the source-candidate status and diagnostics below. Repairs remain `report_only`.

The removed queue walker owned spot gating and set its rate after admission.
The native engine now filters spot-disabled hosts and exhausted GPU pools, and
revalidates that policy while holding the host/device locks. Unreleased native
allocations count even before lease claim; compatibility job status cannot free
the pool. Concurrent claims for one spot slot cannot overcommit it. On-demand
work can use another physical device when the spot pool is full. A host floor
above the admitted price refuses placement instead of increasing the charge.

New spot admissions require an explicit GPU model and active spot/on-demand
catalog prices. The quote function itself refuses its compatibility price
defaults on the admission path. Price is frozen before persisting the immutable
spec and native micro-CAD column. Replaying a committed submission never
re-quotes it. Unknown models cannot create a job or outbox event; API rejection
releases its launch hold. Read-only market estimates retain their prior behavior.

The reservation/quote checks pass against real isolated PostgreSQL: pool disabled,
policy changed after scoring, simultaneous claims for one spot slot, on-demand
capacity, exact admitted price, replay without repricing, missing/unknown model
and missing catalog price. The strict spot, native monetary writer, lifecycle
outbox and API selection passed 98 tests with zero failures/errors/skips; a
preceding 149-check scheduler/spot selection passed before the strict quote
follow-up. Scoped Black/Ruff/Pyright checks pass.

The e57fc242 full diagnostic recorded 3,956 passes and three failures: spot
placement/pricing, a test resetting an already-running attempt into an offer,
and a stale assertion requiring the removed process-local queue notifier. The
new tests start from a real offer, and queue notifications remain transactional.
That failed full diagnostic is retained; no deployment/passing claim is made.
A new complete unified source gate is required before publishing this candidate.

Runnable gates include tests/test_native_spot_capacity.py,
tests/test_spot_capacity.py, tests/test_spot_pricing_unified.py,
tests/test_reconcile_runtime_events.py, tests/test_native_projection_writers.py
and tests/test_host_job_lifecycle_outbox.py through the security harness.
Production remains b935d694/schema 141 with all eight repairs report_only and
preemption/deadline force disabled. No external provider payment is claimed.
