# Serverless authority-loss and request transactions — 2026-10-22

This source candidate closes the serverless side of native lease expiry. An
unclaimed offer fails; an expired claimed lease records a lost attempt. In the
same transaction, the matching worker becomes unavailable, its routing/billing
authority ends at the native attempt timestamp, its slot count becomes zero,
and its in-flight requests return to the queue. Bound usage meters settle in
that transaction. Projection failure rolls the entire expiry back.

The expired serverless scheduler job closes instead of reusing its worker ID
for a later attempt. Endpoint reconciliation provisions a fresh worker and
scheduler job. A single queued request can start replacement capacity from zero
workers; the ordinary queue growth threshold no longer strands it. Normal
non-serverless jobs retain their existing requeue/higher-fence behavior, and
stop/cancel/terminate intent remains authoritative.

Authority loss is not evidence that the old container physically exited. No
stop ACK is invented. The attempt remains lost and native stale-fence handling
still owns physical cleanup. The tests qualify database state and the native
replacement protocol; production/GPU acceptance remains separate.

Request claims now lock the worker, validate readiness/capacity, claim one
queued request and increment its slot count together. Dispatcher and SDK
service paths share that operation. A stale worker selection cannot claim
after expiry or drain, and concurrent claims cannot exceed capacity. Failure
to update the slot rolls the request assignment back.

Completion holds the worker and request locks through token accrual, token
ledger insertion, request result, slot release, endpoint totals and the result
event. Retries return the stored terminal request without repeating those
writes. Cancellation follows the same lock order and releases a slot only
once, including races with completion. A queue timeout cannot overwrite a
claimed or terminal request. The old dispatcher slot-release writer is removed.
Stream sequence allocation is serialized on its request row. Notifications,
webhooks and metric refresh remain after the authoritative commit.

Late callbacks using the closed worker ID cannot complete, read or append
events to requeued/reassigned requests. This is database ownership fencing,
not completion of the credential handoff: worker containers still receive the
broad host credential, and callback routes still need an attempt-scoped token
and the corresponding SDK/gateway changes. Unbound historical worker rows are
not adopted or silently assigned native authority by this change.

The reconciliation expiry path acquires its additional worker/request/meter
locks with NOWAIT when requested, so an in-progress callback defers repair
instead of waiting while holding reconciliation locks. Repairs remain
`report_only` in production.

Tests exercise offered/claimed expiry, rollback, a real submission/reservation/
lease/replacement sequence with a higher fence, rejection of the old identity,
bounded concurrent claims, slot-update failure, token/result rollback,
concurrent completion/cancellation, concurrent event sequences, usage-meter
settlement and NOWAIT callback-lock contention. The final results, source
hashes and read-only production preservation checks are in the
[JSON receipt](2026-10-22-serverless-authority-loss.json).

The final isolated PostgreSQL/Redis gate passed 2,208 tests with zero failures,
errors or skips. The three existing exclusions remain: optional Compose smoke,
future B16.2 contraction and the unavailable companion-prose source check.
Repository Ruff, Black for all 11 changed Python files, Pyright and all 12 worker
signatures pass. Changed Python hashes match the final passing manifest, and
generated artifact checks pass. Four existing OpenAPI/dependency warnings remain.
Disposable PostgreSQL and Redis instances were removed after testing.

No deployment, migration, host restart, historical deletion or repair enforcement
was performed. B16.2/B16.3/B16.4 remain open. Attempt-scoped callback credentials,
SDK/gateway and API/agent rollout, live GPU acceptance, remaining SSH/backend
compatibility paths, schema/trigger contraction, historical-data disposition,
dedicated service roles and durable fairness/preemption still need their own
completion evidence.
