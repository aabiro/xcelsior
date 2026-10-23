# Native billing repair authority

Active and terminal missing-meter repair now use the admitted execution spec and
the quote captured with the exact attempt. They cannot reconstruct customer,
resources or price from job observation JSON or current host price. Spec hashes,
owner, resource count, pricing mode, locked spot rate, bound host, explicit finite
price and the admitted ceiling must agree. Missing qualification keeps a visible
refusal finding. Existing meters still settle from their own captured evidence.
Terminal repair shares normal settlement and uses job/attempt locks without
waiting on a lifecycle writer. All production repair policies remain `report_only`.

Fixtures approve initial native workloads explicitly. Billing incident fixtures
remove only their own newly created test meter to reproduce a missing-meter
incident; owner, resources and price are fixed before admission/placement. They
no longer introduce a new owner or reprice resources after an attempt starts.
Positive and negative cases cover metadata/host-price drift, missing admission,
hash disagreement, absent quote, user ceiling, locked multi-GPU spot cost, actual
restricted API login, transaction rollback, terminal timestamps and idempotency.

The earlier frozen full run selected 3,154 cases: 3,017 passed, 18 failed and 119
errored, with three explicit baseline deselections. Most errors shared an
unadmitted reconciliation fixture; other failures were additional unadmitted
fixtures, a hard-coded heartbeat revision, a source-tree scanner and four real
Redis cases using a 124-byte Unix socket path. The fixture/scanner failures are
corrected without restoring a production payload fallback. The gate runner now
refuses overly long socket paths; its owned harness uses a short temporary path.

The corrected combined PostgreSQL/Redis gate passes **199** checks and the
reconciliation/billing gate passes **103**, with zero failures, errors or skips.
Selections overlap. The ceiling qualification has a separate final gate. Source
hashes, exact manifests, original full-run failures, cleanup and protected root
file hashes are in the [JSON receipt](2026-10-23-native-billing-repair.json).

This is source qualification, not production acceptance. The complete combined
candidate still needs its full regression rerun, remaining lifecycle JSONB and
redundant column contraction, and staged production acceptance. B16 remains open.

The combined native/billing/durable-scheduling run subsequently selected 3,210
cases: **3,207 passed**, three failed, with no errors/skips and the same three
baseline deselections. One test at migration 130 called today's scheduler
without its 133 tables; it now tests the frozen migration's column semantics,
while current scheduler coverage remains at its runtime floor. A serverless
meter fixture now supplies its owner and explicit rate before admission. The
terminal resize test waits for the server's close and session-slot cleanup
before client teardown, preserving its resize/oversized-frame assertions. All
three corrections pass the final **13-case** selection. The full combined run
was not repeated after these test-only corrections. Database/Redis cleanup
completed and the original failure manifest remains available.
