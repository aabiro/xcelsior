# Canonical host identity and native SSH routes

Admission evidence now requires recorded native tenant/owner identity. The
marketplace provider comes from the host provider/owner columns; mutable JSONB
aliases and the host ID cannot fabricate a customer or payment beneficiary.

The SSH gateway reads the canonical job host and confirmed public SSH mapping.
A current attempt must match the job pointer, generation, host, spec hash and an
active unexpired fenced lease. Stop intent and suspended native owner/tenant
identities exclude a route. Requested ports and payload host/owner aliases
cannot authorize access. The restricted gateway role gains read-only access to
attempts and placement leases; it still has no command, money or credential
write authority. Converge those two SELECT privileges before the new handoff.

Native worker terminal retry now journals measured exit codes and keeps them
through lost reports, container loss and restart. It does not journal raw
failure messages. A stale attempt cannot overwrite the replacement's outcome.

Qualification: 222 tests passed with zero failures, errors or skips, including
real PostgreSQL admission, marketplace and restricted gateway queries; forged
payload identities, expired leases, stop intent and suspended accounts are
refused. Worker journal/retry/adoption gates prove the exit code survives a
transient status failure and later container loss. Scoped Black/Ruff pass.

Runnable gates: `tests/test_native_host_authority.py`,
`tests/test_host_admission.py`, `tests/test_ssh_gateway_revocation.py`,
`tests/test_db_service_roles.py`, `tests/test_worker_terminal_recovery.py`,
`tests/test_complete_worker_cutover.py`, `tests/test_native_job_directives.py`
and `tests/test_population_writer_contracts.py` through the isolated security
harness. The generated writer inventory now detects assignments rather than
mistaking a CASE predicate for a status write. The host-loader guard recognizes
canonical delegation rather than requiring duplicated SQL.

The full backup rehearsal independently restored all 154 original table hashes
from revision 139, then applied 140/141. All 149 unaffected logical tables
remained exact, all remaining execution/payment fields stayed unchanged, all
seven attempt command times matched recorded authority, both Lightning
projection archives matched and the wallet ledger mismatch count stayed zero.
This is isolated restore qualification, not deployment. Production is still
revision 139 / runtime `30b3daa6`; repairs remain `report_only`.
