# Native host eviction and clean candidate checks

This records the `821d1a07` qualification. The current `23158934` image,
subsequent concurrency/quota/settlement fixes and full restore qualification
are recorded in the [follow-up receipt](2026-10-23-native-review-and-full-restore.md).
The image and restore limitations below describe this earlier qualification.

Runtime commit `821d1a07` replaces the manual eviction executor's 501 with a
human-approved native action. Preview binds the exact job generation, attempt,
lease, fence, spec hash and device set on a drained host. Execution rechecks
that snapshot under locks, then commits the complete durable stop-command set
and executing plan together. A changed snapshot needs a new approval; partial
enqueue failure leaves no stop commands or consumed plan.

Workers receive graceful TERM with force escalation disabled. ACK or elapsed
grace never proves capacity release. The plan stays pending until every named
attempt is terminal, its exact lease inactive and its allocations released.
The scheduler can finish that proof after a restart; execution replays return
progress without another command set. Compute jobs requeue after settlement;
serverless worker identities terminate for normal pool replacement. Later user
stop/cancel/terminate intent takes precedence. No optimistic detach or allocation
release was added.

Drain now locks and changes the native host row with an atomic version check.
It works when observation JSON omits a duplicated host ID, which previously
could return success without changing the database. Draining still does not
stop workloads. The old direct eviction endpoint requires an approved plan.
The generic instance launcher rejects host action plans, including completed
replays. Cross-tenant plan reads/execution return not found.

Migration 137 permits a completed host action to retain its actual host result
without inventing a launch job ID. Other action types retain the job result
requirement; approval, consumption, response and failure checks stay intact.
Up/down/up preserves prior plan rows. Downgrade refuses when completed native
host results would violate the older constraint, preserving their evidence.
The final runtime floor is 137.

The MCP tool now distinguishes pending from settled execution and sends only
the approved plan identifier. Operator guidance, endpoint classification,
generated endpoint inventory, mounted-v1 OpenAPI snapshot and TypeScript client
types match the new behavior. Public customer OpenAPI remains current.

| Gate | Result and scope |
|---|---|
| `native-complete-clean`, frozen `5b51d90e` baseline | 3,336 passed, zero failures/errors/skips; three explicit baseline exclusions. This predates the operator runtime changes. |
| `operator-eviction-final` | 178 passed: native eviction/drain, race/rollback, user precedence, serverless settlement, restricted scheduler login, migration round trips/bootstrap, fairness/preemption, worker protocol, writer inventory and generated artifacts. One existing ledger case excluded. |
| `operator-launch-final` | 69 passed after the launch action guard: eviction HTTP flow, launch/approval/adapters, mounted MCP schema and inventory/public OpenAPI currency. |
| MCP unit/type checks | 268 passed, one existing unit skip; TypeScript and generated API types compile. |
| `history-production-operator-final` | Production-derived scoped fixture preserves all 364 jobs and execution history through 137 and rollback/replay: four exact specs recovered, 360 explicitly retained as unavailable. Financial references/rows are outside this fixture. |
| `built-operator-image` | Actual shipped UID 10001, read-only root, no network/capabilities; real Gunicorn readiness at schema/floor 137, twelve verified worker signatures and metrics availability gauge equal to one. Owned container and PostgreSQL cleaned up. |
| Static/source checks | Black and Ruff pass on changed Python; targeted Pyright reports zero errors/warnings. Final Python hashes match the qualified source. All nine protected root files retain their saved hashes. |

Counts overlap and must not be summed as unique coverage. Failed/superseded
diagnostic runs remain outside Git; the adjacent JSON links the passing
manifests, source hashes, aggregate production probe and image custody evidence.
The full 3,336-test selection was not repeated after `821d1a07`; its affected
paths are qualified by the focused gates above.

The final image is `xcelsior-api:cutover-operator-821d1a07`, ID
`sha256:8fbee511098c660770ae8980de49bb7bc02ace99f4269eb67cfeb11b172d7752`.
The schema-126 transition and schema-134 intermediate images remain required.
The three-stage procedure and immutable artifact identities are in the
[handoff runbook](2026-10-23-staged-control-plane-handoff.md).

Production was only read at 2026-10-23 16:04 UTC: schema 125, 364 jobs, zero live
attempt pointers/leases/allocations, three fresh service heartbeat identities,
dedicated roles still without login, and all eight repairs `report_only`.
No production mutation, complete financial restore or deployed end-to-end
acceptance is claimed. The staged schema/role/worker handoff, scoped deployed
launch/stop/restart/billing/provider settlement acceptance, remaining observation
adapters/module decomposition and the disabled old spot-preemption execution
surface remain open. The three baseline exclusions and one MCP unit skip also
remain explicit. B16 is not marked complete.
