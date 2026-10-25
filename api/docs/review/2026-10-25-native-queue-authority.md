# Native queue authority and claim projection

**Accepted follow-up:** deployed as `1a0ae249`/schema 141 after 4,064 passing checks and real spot CUDA/billing acceptance. The [executed receipt](2026-10-25-native-release-production-handoff.md) supersedes the source-candidate status and diagnostics below. Repairs remain `report_only`.

The four former queue walkers no longer assign raw job rows. process_queue is
now a thin adapter around SchedulerService; binpack/filtered/ranked walkers are
physically removed. Paused/shadow modes create no placements, canary only claims
its scope, and native failure never falls back to allocate. Fair-share ordering,
admission, pins, quotas and fenced reservations remain owned by the native engine.

The two inline API placement endpoints are authenticated 410 tombstones.
Serverless warm/scale, inference provisioning, autoscale and failover do not
reserve jobs inline. Admission persists intent; the dedicated scheduler observes
queued work and capacity. The CLI's explicit scheduler command uses the same
native engine. Pure read-only allocators and old diagnostic helpers remain and
are not described as removed.

Lease claim now locks the job before its lease and applies lease_claimed through
the native status writer. The lease, attempt, jobs leased/scheduled projection
and durable notification commit together. The old path changed the attempt first,
so the worker's later identical report returned early while jobs stayed assigned.
An outbox SQL failure rolls back every half of the new claim. Queue claim release
similarly persists reason/backoff and its notification in one transaction; a stale
claim emits nothing and repeated same-generation reasons are idempotent.

Integration fixtures now use real native placement, lease/status HTTP and device
exclusivity. A one-GPU host cannot run two jobs merely because their summed VRAM
fits. Drain, eviction refusal and cross-tenant reads run on actual native attempts;
retained unfenced history is explicit only where that refusal is being tested.

Focused queue boundaries passed 99 checks. A wider 469-check diagnostic had three
fixture-only failures (nondeterministic host choice and two reason-code spellings)
after 466 checks passed. The corrected cases, native rollback/idempotency gates,
durable queue fan-out and generated contracts passed a fresh 76-check run. The
independent native drain selection passed 16 checks. Scoped Ruff/Black/Pyright
checks pass; the complete frozen candidate gate is still required.

Runnable gates: tests/test_native_claim_projection.py,
tests/test_scheduler_placement_partition.py, tests/test_control_plane_claim.py,
tests/test_instance_flow.py, tests/test_control_plane_v1_drain_evict.py,
tests/test_queue_diagnostics.py and tests/test_launch_instance_adapter.py through
api/scripts/run_security_gates.py with isolated PostgreSQL. This source follow-up
is not yet deployed. Production remains b935d694/schema 141 with all eight repairs
report_only. Native fleet JSONB reads/synthetic inventory, historical invoice
disposition and independent offsite recovery remain open.
