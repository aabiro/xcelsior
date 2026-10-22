# Control-plane review repairs — 2026-10-22

This candidate repairs all seven reported behaviors after integrating current
`main`, including its expanded dashboard and dedicated-role cutover changes.
These are source repairs; they do not deploy the candidate or apply migrations
to production.

| Finding | Result and runnable proof |
| --- | --- |
| Services tile sums replicated database observations | The expanded dashboard counts distinct fresh scheduler/reconciler/outbox/maintenance identities after deduplicating API observations. Extra replicas and unrelated services cannot replace a missing required identity. Six real Prometheus expression fixtures cover duplication, missing/stale/absent data and a single service with many replicas. `test_control_plane_dashboard_counts_distinct_required_service_identities` binds those fixtures to the shipped query and threshold. |
| Compatibility placement ignores a host pin | The common allocation candidate filter now enforces `preferred_host_id` before ranking. `test_non_native_queue_respects_host_pin_without_fallback` covers available/missing pins in paused and canary modes against PostgreSQL. |
| Stop loses a placement/claim race | After cancellation loses the race, deprovision reloads and locks the current job/worker binding instead of reusing either stale attempt snapshot. Drain and stop intent commit before fenced command enqueue. `test_stop_reresolves_attempt_after_cancel_loses_placement_race` checks the persisted intent and exact attempt command. Existing enqueue-failure/retry coverage remains applicable. |
| Missing submission leaves a zombie worker | Worker, scheduler job and submission outbox event now share a caller-owned transaction. Both pre-insert and post-insert failures leave no partial submission visible to another replica. Existing missing-job rows can settle only when no attempt, host, allocation, heartbeat, concurrency, lease or command evidence exists. The reaper includes old failed submissions and ages booting from creation, since allocation remains zero until placement. Orphan/refusal and reaper replenishment tests exercise both booting and error rows. |
| Serverless charges omit customer wallet notices | Low-balance notices and the transition-to-suspended notice commit with the locked wallet debit and billing cycle. The six-hour warning watermark shares that transaction. Push projection starts after commit. `test_wallet_notices_rollback_with_cycle_then_commit_once_and_push_after_commit` verifies rollback, committed visibility, rate limiting, exact exhaustion and replay. |
| Claim commits without its callback credential | A serverless job or worker binding requires successful credential minting. A mint miss raises inside the claim transaction; the lease stays offered and the attempt stays reserved. `test_callback_mint_miss_rolls_back_claim_and_can_retry_same_offer` successfully retries that same offer after the mint fault clears. Ordinary jobs do not require a serverless token. |
| Migration 128 misses recorded `owner_id` | Both job and host extraction now prefer valid recorded `owner_id`, then `owner`, while known canonical values retain authority. The resumable migration regression covers ID-only, conflicting, invalid and fallback evidence and archive attribution through an ID-only job. Migration 128 is still unapplied in production. |

The combined gate passes **2,386 tests**, with no failures, errors or skips and
the same three explicit baseline exclusions. The focused repair gate passes
**112 tests**; the merged restricted-role
workflows additionally exercise actual serverless placement and strict
maintenance startup. The scheduler retains only the required column grants,
including allocation time; credential retention explicitly uses the reconciler
identity. Runtime writes to the legacy archive remain denied.

A new read-only production export includes both recorded owner keys:
**364 jobs, 6 hosts and 302 lease records**. The separate up/down/up rehearsal
passes and preserves all source lease records and unrelated job fields. It
recovers the same 52 job identities and 36 archive attributions; this snapshot
contains no additional ID-only missing-owner rows. **20 jobs and eight archive
attributions remain unresolved** and are not assigned guessed identities.

The broader gate result, source hashes and production boundary are recorded in
[the machine-readable receipt](2026-10-22-control-plane-review-repairs.json).
Local detailed evidence is under
`artifacts/bugbot-cutover-repairs-20261022/`. Private row exports remain outside
the source commit. Three pre-existing broad-suite exclusions remain explicit
in the runner arguments; passing this candidate does not clear those gates.

The production read-only check still reports schema **125**, no active jobs and
all eight repair modes **report_only**. No runtime handoff, migration, physical
GPU acceptance or repair enforcement was performed. Canonical specs, remaining
JSONB compatibility paths, column/trigger contraction, historical disposition,
all-table semantic review and durable fairness/preemption remain open in B16.
