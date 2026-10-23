# Stored admission is required for native execution

Queue claims, reservations and new lease claims require a native immutable
admission snapshot. Reservations independently verify its hash and version;
an old schedule claim cannot promote missing or recovered historical evidence
into a new attempt. Lease claims also check the current job, generation,
attempt, host and bound spec hash. Mutable payload no longer supplies execution
inputs to placement, attempt billing, serverless identity or callback issuance.
Unadmitted observation rows cannot trigger automatic wallet charges.

The scheduler snapshot reads native pending/running-desired state and stored
queue keys. Live device allocations consume capacity regardless of a
compatibility status change. Host administration fails closed when missing.
The physical host telemetry and the remaining mutable lifecycle fields are
still observations/compatibility data; this change does not remove all JSONB
or every legacy column.

Native protocol tests now explicitly approve their initial fixture workloads.
The helper is never automatic, imported by production, or called by a global
test SQL interceptor. Real submission tests still exercise real admission.
Pin tests bind the original approved pin and verify that a later payload edit
cannot redirect it. The capacity test makes a real device reservation and
proves a status edit cannot release its GPU.

The main isolated run passed **205** checks with two failures from obsolete
schema-compatibility assertions that expected 126 and 127 to serve the current
132 runtime. The corrected explicit floor, negative boundary and generated
inventory gate passes **24** checks. The final billing/native boundary gate
passes **28** checks, including refusal to charge an unadmitted observation.
The full main selection has not been repeated after these corrections.
Original failures and exact source hashes remain in the
[JSON receipt](2026-10-23-strict-native-specs.json). Interrupted prior runs have
separate incomplete receipts and do not count as acceptance.

Repository Ruff, Black on changed Python, Pyright on 247 files with zero
errors/warnings, fresh generated endpoint inventory and all twelve worker
signatures pass. Owned disposable databases were stopped and removed. Nine
unrelated dirty root-worktree files retain their hashes.

This is a source candidate, not production acceptance. Production remains at
125; the staged native-writer handoff must precede migrations through 132 and
this runtime. Repairs remain `report_only`. Remaining work includes broader
native fixture qualification, lifecycle JSONB/physical column contraction,
durable fairness/preemption and production acceptance.
