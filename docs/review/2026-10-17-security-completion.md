# Security and control-plane completion — 2026-10-17

This work finishes the pending public-HTTP, database-role, ingress-monitoring
and dependency changes, and adds the next B2.9 expired-lease remedy. CI cockpit
is excluded. All evidence here is local; no production service, role, secret,
migration or remediation setting was changed.

## Completed behavior

Untrusted webhook and OAuth client-metadata URLs use one bounded HTTP transport.
It checks every DNS answer, connects to the checked numeric address, preserves
TLS hostname verification, rejects redirects/private destinations, and bounds
DNS work, total elapsed time and response bytes. Webhook signatures remain
compatible during dedicated-key rollout; production without any configured
signing key fails closed. Receiver, DNS-rebinding, slow-header, TLS hostname,
payload-size and sanitized-error tests exercise this boundary.

Strict database mode requires explicit service DSNs and checks actual login
identity, ownership, memberships and effective table/column/sequence authority.
Task scopes use separate pools and restore the executor identity afterward.
Unknown tasks cannot claim work under an unqualified identity. Partition DDL
moves to a separate migration-login command. The SSH gateway receives only
route-inventory table reads, plus schema-version access. Mixed-domain legacy
maintenance tasks still explicitly use the API writer; eliminating that broad
authority remains work, not a completed least-privilege cutover.

The placement fencing sequence is standalone, so serial-column discovery alone
missed it. Provisioning now explicitly classifies its control-domain authority;
the scheduler can advance it and billing cannot. Unclassified sequences stop
provisioning. Runtime audit checks detect extra sequence grants as well.

Ingress installation checks the abuse-policy include in every public server
block, including TLS. Comments or an include nested inside a location cannot
satisfy the gate. Security metrics distinguish unavailable evidence from zero
violations. The host collector checks enabled ingress configuration and the
existing worker egress verification without applying repairs.

The pending Python upgrades are resolved in `uv.lock` and exported to
`requirements.txt`. FIDO2 was moved to its compatible 2.2 line because the old
2.1 pin rejected cryptography 50. The MCP lock and SDK requirement are validated
together. The editor's separate model configuration is outside this change.

Expired-lease remediation is documented in
[`../reconciliation.md`](../reconciliation.md). Its flag defaults to report-only.
It reuses lease settlement, keeps grace and intent, and records evidence in the
same transaction. It does not repair missing active meters or prove container
termination.

## Local evidence

Validation uses `scripts/run_security_gates.py` with a fresh Unix-socket-only
PostgreSQL cluster and no inherited credentials or dotenv. The runner records
source/lock hashes, Python version, JUnit results and the gate exit status, and
stops/removes its disposable database. Any skip makes the release gate incomplete.

- Initial existing-environment check: 247 passed, no skips.
- Updated dependency environment: 492 passed, four optional absent-Prometheus
  fallback checks skipped. This run is useful regression evidence, not a
  complete release gate.
- Direct strict-role workflows: 83 passed, no skips, including real task-login
  switching, restricted SSH login, migration-role partition maintenance twice,
  preserved runtime grants, and standalone fencing-sequence authority.
- Lease repair before implementation: six failed, eight passed. After the
  initial implementation, 111 lease/reconciliation/lifecycle tests passed.
- MCP: 299 passed, two opt-in network citation checks skipped; TypeScript
  typecheck passed. This is not a hosted connector journey.

Final acceptance on Python 3.12 (the production image's Python version), using
the updated frozen dependencies: **593 passed, no skips**, across 29 suites.
This includes all 16 new lease-repair cases, service-role workflows, MFA,
OAuth/MCP contracts, ingress, monitoring, startup/deployment validation and
dependency-export consistency. The runner exited zero and removed the isolated
database. Evidence: `/tmp/xcelsior-finish-py312-final-20261017/manifest.json` and
`junit.xml`. Ruff, targeted Pyright, shell syntax, lock consistency and
`git diff --check` passed. Compose action defaults and runbook targets were
checked; a container build and live deployment were not performed.

Counts describe separate, overlapping runs. Temporary logs are under `/tmp/xcelsior-finish-*` and
`/tmp/xcelsior-lease-repair-*`; they are not durable CI or production evidence.

## Monitoring and response

For unavailable security evidence, check the API's database-query errors and
collector schedule first. An absent measurement is not a passing zero. For a
revived abuse-blocked job, preserve job/attempt/finding history and identify the
writer; do not resume it or remove its durable security block to silence an alert.

For database identity mismatch, run `scripts/audit_runtime_db_roles.py --service
SERVICE` under the actual service environment. Output contains issue descriptions,
not DSNs/passwords. Compare effective privileges with `control_plane/db_roles.py`;
do not broaden grants to make the check green. Strict mode stays disabled until
every workflow has passed with its real login, object ownership has been
transferred deliberately, and external partition maintenance is scheduled.
Provisioning and changing production credentials are separate deployment steps.

For repeated public-destination refusals, inspect sanitized OAuth/webhook error
counts. Do not log callback URLs, query strings, HMAC keys or metadata credentials.
Use a dedicated webhook HMAC key only with a coordinated receiver rollout.

For ingress or egress failures, run the host collector with `--ingress` on the
gateway or `--egress` on workers. Investigate enabled vhosts, the existing abuse
map/snippet and the egress verifier's current status. Publish collector output
atomically to a configured Prometheus textfile directory and monitor its
timestamp. Deployment of that schedule and fresh probes remains required;
source-code checks alone do not prove running Nginx or container containment.

## Remaining gates

No full backend/frontend run, built-image verification, live role cutover,
collector deployment, hosted OAuth/MCP journey, or production recovery drill is
claimed. B2.9 retains remaining lifecycle/active-meter remedies and per-type
production measurement/enforcement gates. The Track B checklist remains the
authority for the wider unfinished programme.
