# Native serverless lifecycle candidate — 2026-10-22

The candidate removes serverless service/reaper SSH kills and the premature
release/charge writers. Deprovision persists stop intent and requests an exact
native `stop_attempt`. Running capacity, release timestamps and in-flight
requests remain unchanged until the native attempt settles. Repeated requests
reuse the command. A persisted intent survives failure before command enqueue;
ordinary autoscaler draining alone is not interpreted as a forced stop.

An unplaced queued worker can cancel without inventing runtime. An unclaimed
lease offer can be revoked while holding its lock, serializing against lease
claim; a late claim is rejected. Already claimed work follows the worker stop
protocol. Missing or mismatched runtime authority does not release historical
capacity, charge a wallet or claim successful cleanup.

Readiness and terminal serverless projections now run inside the fenced attempt
transaction. Worker/job/host/endpoint bindings must agree. Native running
promotes booting workers after health confirmation, and renews free warm leases.
A terminal transition closes routing, captures the native server timestamp,
zeros concurrency and requeues in-flight requests atomically with attempt,
placement lease and allocation settlement. Failure rolls all these writes back.
Late heartbeats and concurrency callbacks cannot revive released capacity.

Worker provisioning creates the worker first and includes its complete execution
configuration in the initial scheduler submission. There is no runnable interval
before worker IDs, environment, port and health settings exist. Reservation checks
the stored worker/job/endpoint binding and refuses closed or draining workers.
Native serverless execution uses the same labeled-container terminal monitor as
batch work. It retains evidence on report loss and recovers from its journal;
readiness reporting retries after a transport failure or restart. Health requires
a 2xx response. Startup reports and local authority cleanup remain bound to the
runner's original attempt even when another attempt replaces it concurrently.

Old ready/exited callbacks, including their `/agent/v2` aliases, are authenticated
permanent 410 tombstones. The worker callback sender, separate serverless monitor,
SDK readiness sender and old service/dispatcher release methods are removed.
The queue SDK exposes its health server for the host's native readiness check.
The SDK-only package no longer imports API/database modules. Its health callback
is kept unbound to the HTTP handler so a zero-argument checker returns readiness
instead of an accidental 503. Real HTTP checks cover healthy, draining/unhealthy
and failed checkers.
The generated endpoint inventory documents these tombstones; 568 operations
remain classified.

Final billing retries on the normal periodic scan, including soft-deleted
endpoints, only from a matching native terminal timestamp. The stored release
bounds every retry so later cron time cannot extend the charge. A worker billing
lock serializes overlapping scans; wallet debit, transaction journal, billing
cycle, token consumption and endpoint totals commit together. A ledger failure
rolls the debit and token consumption back. Failed or cold-start metadata rows do
not advance the paid billing cursor. Generic instance billing excludes serverless
workers so it cannot consume their billing interval before blended metering.
Optional Stripe projection occurs after the authoritative commit.

The focused PostgreSQL tests exercise actual placement, lease claim, offer
revocation, terminal confirmation, stale fences, projection rollback, durable
stop intent, free warm readiness, late callbacks, billing rollback, overlapping
settlement and soft deletion. Broader fixtures now isolate Stripe calls and local
webhook transport, restore ingress settings after each test, and require explicit
release proof before a mocked reaper reports successful reprovisioning. No new
exclusions were introduced. Redis integration uses an explicitly supplied Unix
socket from a disposable network-disabled, nonpersistent local container, rather
than inheriting credentials or skipping the real store tests. The runner rejects
a missing/non-socket target before creating its PostgreSQL cluster. The final
results and source identities are in the
[machine-readable receipt](2026-10-22-serverless-native-lifecycle.json).

The final gate passed 2,185 tests with zero failures, errors or skips. The same
three baseline exclusions remain: optional Compose smoke, future B16.2 schema
contraction and the unavailable companion-prose source check. Repository Ruff,
Black for all 39 changed Python files, Pyright and all 12 worker signatures pass.
All changed Python/signature hashes match the final passing manifest. Two
existing duplicate OpenAPI operation-ID warnings remain. The disposable Redis
container and PostgreSQL cluster were removed after testing.

The updated queue SDK image was built locally and booted as an unprivileged user
with a read-only filesystem, no capabilities and no external network. Its real
health endpoint returned 200 and the queue runtime drained cleanly without the
retired readiness callback. The image ID and packaged-source hashes are recorded
in the receipt. This qualifies CPU SDK packaging, not GPU inference or a live
API/agent journey. Registry publishing/pinning remains before the API/agent handoff;
older queue SDKs still call the retired readiness URL. Production remains schema
125 on application PostgreSQL port 5434, with repairs in `report_only`. No
migration, application deployment, worker restart, historical deletion or repair
enforcement was performed.

B16.2/B16.3 remain open. Native lease-expiry/serverless replacement qualification,
attempt-bound request callbacks and atomic request-claim/concurrency handling,
remaining inference/billing/checkpoint SSH callers, backend compatibility paths,
worker decomposition, dedicated-role deployment, schema/trigger contraction and
historical data disposition remain. Fairness/preemption, repair enforcement and
coordinated rollout require their own completion evidence.
