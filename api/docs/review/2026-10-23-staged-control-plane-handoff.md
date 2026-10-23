# Staged control-plane production handoff

This is a prepared release procedure, not a deployment receipt. The read-only
production baseline remains schema 125, no live attempt pointers/leases/device
allocations, and eight `report_only` repair policies. Dedicated service roles
are still `NOLOGIN`. B16 remains open until the deployment evidence below passes.

## Frozen artifacts

| Purpose | Source | Local image | Image ID |
|---|---|---|---|
| Native writers compatible with schema 126, before removing legacy triggers | `1af572b6` | `xcelsior-api:cutover-transition-1af572b6` | `sha256:86a4545e3d73fdb2c0e9a089e3bf0dd693a165fbf8b048f458e4a1c14990f3a6` |
| Native lifecycle, admitted preferences, schema contraction and retired task | `7757bdc3` | `xcelsior-api:cutover-native-7757bdc3` | `sha256:57107a070e5bacb95dd94760dc4231f2681713a02c02d69e1f03f2ac1c0fc8c6` |

Both builds use tracked-file archives, exclude runtime environment files and
private signing keys, and retain build logs/context hashes outside Git. Later
`c0f4af05` changes only two test fixtures; it does not change the runtime image.
Do not rebuild or substitute a moving tag when transferring these artifacts.
Retain archive/image checksums and compare imported Config/RootFS/platform with
the frozen local image. A matching tag alone is insufficient.

The final image's runtime floor is 134. It cannot serve schema 125 or 126.
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
3. Stage the frozen images and source under a private release directory. Keep
   saved credentials/configuration private; never print rendered Compose
   environments or place real DSNs in release logs, argv, Git or image layers.
4. Prepare two explicit image overlays naming `api`, `api-blue`,
   `scheduler-worker` and `bg-worker`. One points to the frozen transition
   image; one to the frozen native image. The one-shot migration job must use
   the native image (which contains revisions 126–136), independently of which
   application writer is serving. Pin `COMPOSE_PROJECT_NAME=xcelsior` and merge
   after the existing operational override so a later normal deploy preserves
   the reviewed service configuration.

## Ordered schema and writer changes

Run commands from `/opt/xcelsior` using the staged source. `COMPOSE_FILE` selects
only the intended private overlays for that step; its values contain paths,
not credentials. The examples show runner targets, not a single unattended
script. Capture each stage's schema, image, role, readiness and row invariants
before advancing.

1. Select the native image **for the one-shot migration job** and run:

   ```sh
   python3 api/scripts/run_db_migrations.py --target 126
   ```

   Hand off all API replicas, scheduler and maintenance to the frozen
   **transition** image. Upgrade both signed worker distributions with journal
   preservation and signature/source verification. A health response alone is
   insufficient: prove current worker-session claim/ACK and confirmed restart
   adoption. Preserve Tower/VTT serving containers and unrelated processes.
   Verify every writer is native before removing projection triggers.

2. Use the native migration image for the ordered targets below. Stop on a
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

   Replace every lifecycle writer with the frozen **native** image before
   allowing traffic. Verify runtime floor 134, native intent and admitted
   preference enforcement. This is a quiesced handoff: do not mix older JSONB
   intent writers with native intent readers.

4. After all writers stop naming the retired placement columns, apply:

   ```sh
   python3 api/scripts/run_db_migrations.py --target 135
   python3 api/scripts/run_db_migrations.py --target 136
   ```

   135 checks/archives four original placement columns and old payload intent
   keys before dropping them; no job or financial history is deleted. 136
   archives only the obsolete stop-redelivery task and prevents resurrection.
   A live task claim or archive mismatch refuses 136. The new maintenance
   writer must already be installed. Converge grants for the new archive tables.

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

- Schema 136, final image/source identities, both signed workers, every service
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
copies the latest native value (including consumed NULL) for old readers. 136
restores its recorded task row only after its anti-resurrection constraint is
removed. Retain archive evidence and do not run the restored obsolete handler
accidentally. Restore/inverse ownership and role changes require their retained
plans. A financial backup restore is not a shortcut for undoing metadata/config
changes while newer writes exist.

The final deployment requires approval of this staged schema 126–136 and role/
worker handoff candidate. Earlier approval of migration 123 does not establish
acceptance of these later contractions.


## Source residuals outside this handoff proof

The scheduler's durable preemption controller exists, but manual operator
host-eviction execution and the old spot-preemption execution surface still
return 501. Their dedicated durable API integration is not implied by scheduler
preemption tests. API observation/projection compatibility and the planned
scheduler/worker module decomposition also remain. Do not describe every legacy
adapter or every B16 exit requirement as removed.

Three existing broad-selection exclusions remain explicit: the Docker Compose
smoke class, the migration ledger's historical contract-cleanup-order check, and
the absent companion document's conformance-prose check. Their exclusion is not
passing evidence. Fresh real migration/bootstrap, archive and rollback behavior
are qualified by separate runnable database gates.
