# Control-plane production rollout — 2026-10-17

The explicitly approved candidate deployed successfully at **08:10:55
America/Toronto**. `scripts/deploy.sh --docker` exited 0, applied **122 → 123**,
and completed the API, scheduler, maintenance, SSH gateway, frontend and MCP
handoffs. Repair actions remained **`report_only`**. The deployed checkout was
`582859df`, whose application source is the approved `da4442cf`; the intervening
commit contains only the readiness report.

This supersedes the approval-pending status in the
[readiness record](2026-10-17-control-plane-rollout-readiness.md).

## Deployment evidence

The deploy ran from an isolated clean worktree. Production `.env` was preserved;
local worker-mode switching and mirror synchronization were disabled. Backend
and frontend builds ran serially. Migration 123 succeeded before standby API
readiness and nginx promotion. SSH standby health, MCP authenticated protocol
smoke, the canonical MCP metadata/challenge checks and final API readiness all
passed. All six application containers reported healthy at script completion.

The prepared private database/configuration backup and retained image tags are
recorded in the readiness report. The script also created
`/opt/xcelsior-backups/xcelsior_20261017_080754.tar.gz` before synchronization.
Backup preparation is not a restore drill.

The local evidence directory is
`/mnt/storage/projects/xcelsior/artifacts/release-evidence/`:
`production-deploy.log` records the handoffs, and `production-deploy.exit` is 0.
The [sanitized evidence manifest](2026-10-17-control-plane-production-evidence.json)
records source fingerprints, current images, public results, repair flags,
aggregate database observations and local evidence hashes.

## Later verification and concurrent changes

When verification resumed that evening, production had independently advanced
to **`1ac59848dc56a578b6db069566b9f449f22bffe7`**, a descendant containing the approved
release, and schema **125**. The newer security releases and migrations 124–125
were not performed by this deployment. They were preserved. Comparing current
containers only with the older candidate initially reported differences; all
**76 sampled backend fingerprints** matched the newer recorded commit exactly.

At **19:05:50**, a follow-up check observed four unhealthy backend containers.
Docker events subsequently showed another operation stopping and recreating
the API, scheduler, maintenance worker and SSH gateway. They became healthy
again by **19:06:12**, with unchanged image IDs. This continuation did not restart
them. Their recreation is established; its initiating actor and cause were not
established. One read-only aggregate probe failed during this interval and is
not counted as a pass.

The environment checksum matched the pre-deployment value at the first evening
probe, then changed during the separate operation. Comparing the release backup
on the production server showed changes only to `DATABASE_URL`,
`XCELSIOR_PG_DSN`, `XCELSIOR_POSTGRES_DSN`, `XCELSIOR_POSTGRES_EXPORTER_URI`,
`XCELSIOR_POSTGRES_PORT` and `XCELSIOR_TEST_POSTGRES_DSN`; no keys were added or
removed. Values and credentials were not exported. Those settings were preserved.
The final probe used the API's effective DSN and connected successfully.

The final read-only verification at **19:07:33** established:

- All six application containers healthy; API readiness true with storage, NFS
  and auth-cache checks passing.
- Public homepage, `/healthz`, `/readyz` and canonical MCP protected-resource
  metadata returned HTTP 200. The separate unauthenticated MCP probe returned
  HTTP 401 with the `resource_metadata` challenge.
- Schema 125, with both migration-123 command/lease reconciliation triggers
  enabled. No source mismatches among the 76 sampled files.
- Every configured repair action remained `report_only` in API, maintenance
  and scheduler containers. Strict database roles remained disabled; the API
  still connected as the shared `xcelsior` identity.
- No open reconciliation findings or queued reconciliation requests; five
  closed usage meters and no open meters. There were no running jobs, so this
  observation does not qualify active-job repair or false-positive rates.

A follow-up at **19:08:32** confirmed billing-meter reconciliation succeeded
at 19:08:04 and job-reconciliation recovery succeeded at 19:08:12, after the
container recreation and database configuration change.

## Remaining programme gates

This closes deployment of the committed control-plane increment and its
report-only configuration check. It does not close B2.9, security acceptance or
the wider programme. Remaining work includes active-workload measurement,
individual enforcement decisions, lost-terminal-report and failure/recovery
drills, full public launch/provisioning, wallet/provider settlement qualification
and dedicated runtime database identities. No repair enforcement was enabled.
The earlier **148 passed, zero skipped** rollout gates and isolated real CUDA
worker-to-invoice acceptance remain their own evidence; they are not presented
as new production workload or payment tests.
