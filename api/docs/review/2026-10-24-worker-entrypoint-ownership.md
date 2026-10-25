# Worker entrypoint and signal ownership

Signal handlers now install when the worker main function starts. Importing a
worker helper leaves the caller's signal ownership intact. This corrects a real
qualification failure: imported worker handlers consumed both SIGINT and SIGTERM
inside pytest, preventing ordinary interruption. The deliberately stopped run
is retained as interrupted evidence; its PostgreSQL and Redis cleanup passed.

The executable entrypoint now follows all function definitions. Previously the
main loop started before the artifact promotion and environment fingerprint
helpers were defined, so those helpers existed during import-based tests but
were unavailable when the signed file ran as the actual worker script.

The real script-entrypoint gate substitutes only the first configuration
validation boundary. It executes the actual main path, verifies the late
helpers are defined and signal handlers installed, then exits before credentials,
GPU probing or network calls. An independent import subprocess proves the
caller's handlers remain unchanged. Worker shutdown, fencing, journal recovery,
resource cleanup, artifact copy and preemption gates pass: 150 checks, zero
failures/errors/skips. A preceding 207-check boundary selection also passed.

Runnable gates: `tests/test_worker_signal_ownership.py`,
`tests/test_worker_claim_handoff.py`, `tests/test_worker_terminal_recovery.py`,
`tests/test_worker_resource_recovery.py`,
`tests/test_promotion_copies_and_verifies.py` and
`tests/test_worker_preemption_protocol.py`, through the isolated security harness.

The worker distribution is re-signed. Frozen b935d694 passed 3,897 tests with
zero failures/errors/skips and 1,129 unchanged source hashes, plus the actual
non-root image checks. API, service and both idle signed-worker handoffs are
executed at schema 141. Fresh CUDA, ledger and role checks pass; see the
[current production receipt](2026-10-25-native-startup-production-handoff.md).
