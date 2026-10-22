# Native-only worker execution — 2026-10-22

This follow-up removes the worker's v1 work polling, destructive command drain,
claim/renew/release client functions, legacy lease renewal/reclaim thread, raw
instance-status PATCH, unfenced start branch, and separate legacy batch and
interactive monitor loops. Native claim/ACK delivery is the only work transport.
`run_job` requires an exact local attempt, fence and container name before any
container changes; there is no lease fallback for unmarked jobs.

Negotiation requires protocol 2, an explicit native boolean and the full
host-claim/ACK, session-fencing and runtime-metadata feature set. A v1 response,
missing feature, malformed feature list or failed negotiation leaves execution
unconfirmed. An old `legacy_execution_allowed` field cannot enable execution or
bypass the feature checks. Uncertain startup retries negotiation and recovers
only journaled native attempts.

Status reporting captures exact attempt authority and passes it to the native
status writer, including when the caller did not supply an expected tuple.
Replacing the local attempt between those reads cannot borrow the replacement's
fence. A lost or rejected report never falls back to a raw job mutation.

Batch and interactive execution use the existing labeled-container terminal
monitor. It retains the journal and exited-container evidence on uncertain
reports, retries after restart, refuses another attempt's labels, and preserves
clean interactive stops. Native maintenance delivery, durable ACK replay and
signature verification remain covered. Already queued `start_container`
directives receive a nonretryable NACK; only `start_attempt` can authorize a new
execution. State-preserving stops of verified historical containers remain
available without adopting them or rewriting their job history.

Startup regression fixtures now carry matching native authority, so partial
volume-mount cleanup and exclusive-GPU refusal still exercise setup failures
instead of passing because an unfenced job was rejected early. Host directive
handler tests deliver commands directly; separate native transport tests verify
claim, ACK, NACK, persistence failure and restart replay. Signature files are
explicitly marked binary in Git because their raw Ed25519 bytes can contain
whitespace and NULs. Worker configuration templates drop the unused preemption
grace setting and document the startup-verification switch with a blank
placeholder and its default explained in the comment.

The initial focused gate passed 264 tests. The complete validation and source
identities are recorded in the [machine-readable receipt](2026-10-22-native-only-worker.json).
Tests use a fresh Unix-socket-only PostgreSQL database; Docker and network IO in
worker execution tests are isolated. These gates are repository evidence, not a
live GPU execution or worker-rollout claim.

Production was queried read-only: schema 125, application database port 5434,
active scheduler mode, no active jobs and all eight repairs in `report_only`.
The 302 historical lease rows retain the same full-row digest as the previous
cleanup wave. No migration, application deployment, worker restart, historical
record deletion or repair enforcement was performed.

B16.2/B16.3 remain open. Serverless callback settlement and its separate native
monitor still require durable integration, alongside remaining SSH lifecycle,
inference, serverless reaper and migration/checkpoint callers. Backend legacy
queue/status/resume paths, retained URL aliases, SQL JSONB decomposition,
dedicated-role deployment, schema/trigger contraction, historical disposition,
worker modularization, fairness/preemption and repair qualification remain
separate work. This receipt does not claim that all compatibility code is gone.

The final combined gate passed **1,673 tests**, with zero failures, errors or
skips. The same three named exclusions remain: optional Compose smoke, future
B16.2 contraction, and the missing companion-prose source check. All worker
suites, configuration-template derivation, generated artifacts, signatures,
lease/reconciliation, placement, cancellation, billing and lifecycle checks are
included. Repository Ruff, Black on all thirteen changed Python files, and
Pyright (241 files, zero errors) passed. Two existing duplicate operation-ID
warnings and the whole-tree formatting backlog remain. All changed Python
hashes match the final passing manifest. Six protected unrelated workspace
file hashes are unchanged.
