# Permanent native execution transport

The v1 work poll, destructive command drain and instance status PATCH are
permanent authenticated HTTP 410 tombstones. Their execution code, process-memory
work cache, volume enrichment and raw status/runtime mutations are physically
removed. No scheduler mode or retirement setting can revive those transports.
Host protocol negotiation always reports protocol 2; a disabled native host
cannot negotiate a v1 fallback. Native ACK/NACK schemas require the claiming
worker session and delivery attempt in every mode.

Real PostgreSQL/HTTP gates preserve complete job rows and pending commands across
retired requests, reject missing native command identity without changing its
claim, and retain valid native completion/replay behavior. Authentication still
precedes the tombstones. The generated endpoint inventory, OpenAPI and client
schema are refreshed; 18 generated-contract checks passed. Ruff, Black and scoped
Pyright pass.

The initial focused selection exposed obsolete v1 success expectations. The
updated 426-check run had one test-only failure reading a deliberately private
attempt ID from a public projection; 425 checks passed. That assertion now reads
the database directly and the final 35-check authority selection passed with no
failures/errors/skips. These diagnostic runs remain recorded. The complete bcec08cb run subsequently recorded 3,900 passing checks and 23
failures in old v1-fixture/status-retirement expectations. That diagnostic is
retained; the native queue follow-up converts those fixtures and requires a
new frozen full gate before deployment.

Runnable gates include `tests/test_native_transport_retirement.py`,
`tests/test_agent_v2.py`, `tests/test_legacy_observation_gates.py`,
`tests/test_cutover_audit_regressions.py`, `tests/test_host_agent_tokens.py`,
`tests/test_generated_artifacts_are_current.py` and
`tests/test_public_openapi.py`, through `api/scripts/run_security_gates.py` with
fresh Unix-socket PostgreSQL and a stripped environment.

This is source work, separate from production fdaab814/schema 141. Remaining
legacy queue walkers and synthetic inventory, native fleet projection reads,
historical invoice disposition and independent offsite recovery are explicit
residuals. Repairs remain report_only; production preemption stays disabled.
