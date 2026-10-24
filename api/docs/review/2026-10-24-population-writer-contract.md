# Population writer and canonical authority follow-up

Production remains at revision 139 / runtime `30b3daa6`. This source candidate
requires revision 140; it is not a production migration receipt. Repairs remain
`report_only`.

The read-only post-cutover scan covered 176 physical tables and 2,078 columns,
including partitions, with 92 nonempty tables and zero scan errors. A separate
integrity scan checked all 48 foreign keys with zero orphan rows, zero
unvalidated constraints and zero invalid indexes. A column-to-source inventory
is an audit aid, not proof that every nullable field should be populated.
Unknown historical ownership, traces, payment dates and outcomes remain unknown.

Confirmed writer corrections:

- Job and host reads project canonical identity; general upserts preserve known
  identity and stop duplicating it in JSONB. Concurrency admission uses native
  owner/phase and fails closed on database errors. SSH, SLA and provider billing
  stop recovering job ownership from observation payloads.
- Authenticated HTTP requests record server-generated request IDs and verified
  actor/client IDs in transactional outbox events. Active trace context flows
  through launch plans, submissions and reservation. No caller header assigns
  identity and no sealed historical event is rewritten.
- A reservation records its actual start-command timestamp. Fenced terminal
  reports persist measured exit codes and error messages; a stale attempt
  cannot replace them. A fresh attempt clears the previous outcome.
- Signed workers report a complete NVML physical inventory, capabilities and
  conditions. Unavailable probes remain distinct from a complete empty list.
  Device UUID/PCI/model/capacity and host observation session become native
  fields. Device membership changes only while allocations are released.
  Older/out-of-order retained sessions cannot overwrite the current snapshot.
- Job-scoped commands require a current native attempt/lease, matching host,
  spec hash and fence, and no stop intent. The worker verifies journal authority,
  renews the exact lease and checks the actual container label before dispatch.
  A historical container name cannot authorize execution.
- Support-document refresh uses transactional DELETE under its existing lock,
  which works with the restricted API role without granting TRUNCATE.

Revision 140 backfills command time only from the exact recorded start binding.
It verifies native Lightning facts before removing their compatibility trigger
and requiring canonical fields. It drops the unused `agent_commands.result` and
unimplemented observation journal cursor only when they contain no evidence;
otherwise it refuses. Required reconciliation/immutability triggers stay.

The schema change needs a quiesced handoff of every writer and diagnostic reader.
The old runtime names a removed diagnostic column. Keep a full checked backup,
run the real migration entrypoint using the direct migrator, then start the
qualified image and signed workers. Verify schema, dedicated roles, request
provenance, physical GPU identities, measured job outcomes, ledger invariants,
repair modes, public distributions and the full restore before calling it live.
Rollback requires a compatible forward fix; revision 140 does not resurrect
unqualified writers or destroy recovered facts.

Qualification uses `api/scripts/run_security_gates.py` with a fresh Unix-only
PostgreSQL 16 and stripped credentials. The focused provenance/native-outcome
selection passed 47 tests without skips. Twelve generated-contract checks passed;
MCP OpenAPI/types are regenerated. Scoped Black/Ruff and Pyright pass. The full
expanded regression is a separate required gate before promotion.

Runnable regression sources: `tests/test_population_writer_contracts.py`,
`tests/test_runtime_metadata_contract.py`, `tests/test_event_request_context.py`,
`tests/test_native_job_directives.py`, `tests/test_launch_plan_execute.py`,
`tests/test_generated_artifacts_are_current.py` and
`tests/test_mcp_openapi_contract.py`.

Remaining contraction includes Lightning float/epoch twins and the residual
compatibility execution handlers. Historical invoice disposition, independent
offsite/PITR/object recovery and full semantic review of optional feature fields
remain explicit; this candidate makes no blanket completion claim.
