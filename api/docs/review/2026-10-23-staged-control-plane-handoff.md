# Staged control-plane production handoff

This is a prepared release procedure, not a deployment receipt. The read-only
production baseline at 2026-10-23 20:04 UTC remains schema 125, no live attempt pointers/leases/device
allocations, and eight `report_only` repair policies. Dedicated service roles
are still `NOLOGIN`. B16 remains open until the deployment evidence below passes.

## Frozen artifacts

| Purpose | Source | Local image | Image ID |
|---|---|---|---|
| Native writers compatible with schema 126, before removing legacy triggers | `1af572b6` | `xcelsior-api:cutover-transition-1af572b6` | `sha256:86a4545e3d73fdb2c0e9a089e3bf0dd693a165fbf8b048f458e4a1c14990f3a6` |
| Intermediate native lifecycle, admitted preferences, schema contraction and retired task | `7757bdc3` | `xcelsior-api:cutover-native-7757bdc3` | `sha256:57107a070e5bacb95dd94760dc4231f2681713a02c02d69e1f03f2ac1c0fc8c6` |
| Native host eviction, serialized snapshots, net replacement quotas and schema-137 results | `23158934` | `xcelsior-api:cutover-review-23158934` | `sha256:788f6b8084caa223fd4a3a8d4b2e4f4444822041e6048ac8f2948f187ce71ec3` |

All three builds use tracked-file archives, exclude runtime environment files and
private signing keys, and retain build logs/context hashes outside Git.
Later `c0f4af05` and `5b51d90e` change only fixtures/review evidence.
`23158934` supersedes the `821d1a07` operator runtime; do not substitute the
intermediate or the older operator image for it.
Do not rebuild or substitute a moving tag when transferring these artifacts.
Retain archive/image checksums and compare imported Config/RootFS/platform with
the frozen local image. A matching tag alone is insufficient.

The local transfer archive is `review-handoff-images.tar.gz`
(966587391 bytes, SHA-256 `5ed3be54e4f272a390b75dca4084c28adad16d8249861958de7d6925fefb5a4b`).
Its saved tags, Config payload hashes and RootFS layers independently match all
three frozen local images. It has not been transferred to production.
All three images passed real Gunicorn readiness, metrics and signature checks
on an isolated PostgreSQL 16 cluster: transition at 126, intermediate at 136,
and final at 137. The final image runs as UID 10001 with a read-only root,
no network and no capabilities. See the
[review follow-up receipt](2026-10-23-native-review-and-full-restore.md).

The final image's runtime floor is 137. It cannot serve earlier schemas.
The intermediate `7757bdc3` image has floor 134 and is needed for 134–136.
The transition image is not the final post-contraction runtime. Never use the
normal automatic `deploy.sh` path to jump from 125 directly to repository head:
it migrates before replacing the writers. The staged runner supports explicit
absolute targets and keeps migration credentials in the one-shot process.

## Before the production change

1. Retain the current images, Compose configuration privately, service/worker
   source identities, effective roles, repair modes, protected job/volume
   hashes, native backup inventory and live sessions. Recheck that no workload
   has acquired a live attempt or device since the baseline. Do not force a
   handoff over newly running work.
2. Make a fresh checksummed backup of the configured Xcelsior database on host
   port **5434**. Pass an isolated restore/migration drill and retain its
   cleanup/invariant receipt. Port 5432 belongs to other services. Do not treat
   the scoped historical fixture as a full financial backup/restore proof.
   The complete 19:20 UTC snapshot has now passed a PostgreSQL 16 restore of
   all 138 source tables and ordered migrations 126–137, preserving 34 protected
   tables at every target. Retain that receipt and recheck freshness before the
   actual production change; it does not capture writes after its snapshot.
3. Stage the frozen images and source under a private release directory. Keep
   saved credentials/configuration private; never print rendered Compose
   environments or place real DSNs in release logs, argv, Git or image layers.
4. Prepare three explicit image overlays naming `api`, `api-blue`,
   `scheduler-worker` and `bg-worker`. Use distinct transition, intermediate
   native and final operator overlays. The one-shot migration job must use
   the final operator image (which contains revisions 126–137), independently of
   which
   application writer is serving. Pin `COMPOSE_PROJECT_NAME=xcelsior` and merge
   after the existing operational override so a later normal deploy preserves
   the reviewed service configuration.

## Ordered schema and writer changes

Run commands from `/opt/xcelsior` using the staged source. `COMPOSE_FILE` selects
only the intended private overlays for that step; its values contain paths,
not credentials. The examples show runner targets, not a single unattended
script. Capture each stage's schema, image, role, readiness and row invariants
before advancing.

1. Select the final operator image **for the one-shot migration job** and run:

   ```sh
   python3 api/scripts/run_db_migrations.py --target 126
   ```

   Hand off all API replicas, scheduler and maintenance to the frozen
   **transition** image. Upgrade both signed worker distributions with journal
   preservation and signature/source verification. A health response alone is
   insufficient: prove current worker-session claim/ACK and confirmed restart
   adoption. Preserve Tower/VTT serving containers and unrelated processes.
   Verify every writer is native before removing projection triggers.

2. Use the final operator migration image for the ordered targets below. Stop on a
   refusal and retain the evidence; do not raise lock waits or invent missing
   identities/specs to make a migration pass.

   ```sh
   python3 api/scripts/run_db_migrations.py --target 127
   python3 api/scripts/run_db_migrations.py --target 128
   python3 api/scripts/run_db_migrations.py --target 129
   python3 api/scripts/run_db_migrations.py --target 130
   python3 api/scripts/run_db_migrations.py --target 131
   python3 api/scripts/run_db_migrations.py --target 132
   python3 api/scripts/run_db_migrations.py --target 133
   ```

   127 refuses live legacy leases before archive/drop. 128 recovers only
   recorded ownership/tenancy. 129 recovers exact hash-bound specs; 132 records
   explicit retained historical disposition. Expected rehearsal evidence was
   four exact specs and 360 unavailable specs, not 364 recovered workloads.
   130 makes native lifecycle fields required; 131 removes both authority
   projection triggers. 133 adds durable scheduling policy/accounting and
   preemption records with preemption disabled by default.

3. Quiesce **all** API, scheduler and maintenance lifecycle writers. Preserve
   worker journals; drain or settle any newly active work before this boundary.
   Confirm no old replica, background process or in-flight transaction can
   write JSONB lifecycle intent. Then apply:

   ```sh
   python3 api/scripts/run_db_migrations.py --target 134
   ```

   Replace every lifecycle writer with the frozen **intermediate native**
   `7757bdc3` image before
   allowing traffic. Verify runtime floor 134, native intent and admitted
   preference enforcement. This is a quiesced handoff: do not mix older JSONB
   intent writers with native intent readers.

4. After all writers stop naming the retired placement columns, apply:

   ```sh
   python3 api/scripts/run_db_migrations.py --target 135
   python3 api/scripts/run_db_migrations.py --target 136
   python3 api/scripts/run_db_migrations.py --target 137
   ```

   135 checks/archives four original placement columns and old payload intent
   keys before dropping them; no job or financial history is deleted. 136
   archives only the obsolete stop-redelivery task and prevents resurrection.
   A live task claim or archive mismatch refuses 136. The new maintenance
   writer must already be installed. 137 allows completed host eviction results
   without a fabricated launch job. Converge grants for the archive tables, then
   replace all writers with the final `23158934` image and verify floor 137.
   Deploy the MCP source/types at the same commit and verify that pending
   eviction remains pending in tool output until fenced settlement.

5. Complete the dedicated-role procedure in
   [the role cutover runbook](2026-10-21-db-role-cutover.md): privately generate
   distinct credentials; provision only used runtime/migrator logins; hand off
   API, scheduler, reconciler/task identities and SSH gateway; install native
   external partition maintenance; audit actual service sessions; preview and
   apply the scoped ownership plan with retained inverse evidence. Use the
   strict Compose overlay after operational overrides. Never copy the migrator
   or shared-owner credential into runtime services. Do not disable the old
   login until no runtime session needs it and inverse ownership/configuration
   evidence is retained.

## Required production acceptance

- Schema 137, final image/source identities, both signed workers, every service
  heartbeat and `/startupz`/`readyz`; verify the configured Prometheus target is
  actually ingesting `/metrics/prometheus` within its 10-second scrape budget.
- Eight repairs remain `report_only`. Operator policy still enables no
  preemption or deadline force. Local synthetic enforcement tests do not
  authorize changing production flags or evicting other workloads.
- Counts/hashes of protected jobs, money rows, execution history and volumes
  match the planned migration changes. Historical jobs cannot execute or bill
  from observations, and all disposition/archive rows remain available.
- Per-service runtime role audits pass, no runtime uses the shared owner or
  migrator, and native partition maintenance/backup/restore metrics are ingested.
- An explicitly scoped native launch, host pin/preference, claim, GPU execution,
  logs, stop/fresh restart, terminal-report retry and wallet/invoice settlement
  pass through the deployed API and worker. No stale fence can change the new
  attempt. Isolated CUDA and selected regression results are prerequisites,
  not substitutes for this deployed journey or provider payment reconciliation.

## Failure and rollback

Keep writers quiesced on schema/handoff failure. Restore only images compatible
with the current schema; never restart the original 125 writer after dropping
its columns. Review rollback in reverse writer/schema order. The 135 downgrade
restores archived projections but does not resurrect stale JSONB intent; 134
copies the latest native value (including consumed NULL) for old readers. 137
refuses downgrade once native host results would violate the older launch-only
constraint; retain schema 137 and use a compatible runtime rollback. 136
restores its recorded task row only after its anti-resurrection constraint is
removed. Retain archive evidence and do not run the restored obsolete handler
accidentally. Restore/inverse ownership and role changes require their retained
plans. A financial backup restore is not a shortcut for undoing metadata/config
changes while newer writes exist.

The final deployment requires approval of this staged schema 126–137 and role/
worker handoff candidate. Earlier approval of migration 123 does not establish
acceptance of these later contractions.


## Source residuals outside this handoff proof

Manual operator eviction is now native and qualified: preview binds current
attempt/lease/fence/spec snapshots, human approval gates commands, and progress
requires terminal release proof. The old spot-preemption execution surface still
returns 501; it must not bypass explicit policy/approval through legacy code.
API observation/projection compatibility and the planned
scheduler/worker module decomposition also remain. Do not describe every legacy
adapter or every B16 exit requirement as removed.

Three existing broad-selection exclusions remain explicit: the Docker Compose
smoke class, the migration ledger's historical contract-cleanup-order check, and
the absent companion document's conformance-prose check. Their exclusion is not
passing evidence. Fresh real migration/bootstrap, archive and rollback behavior
are qualified by separate runnable database gates.
