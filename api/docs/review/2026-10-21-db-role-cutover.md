# Dedicated database roles: candidate and production gates

**Executed status, 2026-10-24:** production schema **141** uses seven dedicated
service logins on PostgreSQL port **5434**. All seven runtime-context audits
pass; no old-owner or migrator session serves application traffic. Ownership
transfer applied 189 scoped statements with private inverse SQL retained. The
old login is limited to offline backups without application DML or DDL; unused
retrieval/read-only logins remain disabled. All repairs remain `report_only`.
See the [executed receipt](2026-10-24-executed-control-plane-data-cutover.md).

The candidate description and initial catalog below are historical preparation
evidence; their schema-125/NOLOGIN baseline is superseded by that receipt.

The main push is complete at `ac1852ad`. This change follows the protected
branch PR workflow. Dedicated-role deployment remains an explicit residual
until the production gates below pass. Schema contraction, physical legacy
removal, durable fairness/preemption, repair qualification and historical
recovery/disposition remain separate work. Repairs stay `report_only`.

## Implemented behavior

- Scheduler serverless placement can update only `attempt_id`, `host_id`
  and `updated_at` in `serverless_workers`. Its state and endpoint definitions
  remain outside scheduler write authority. The role auditor recognizes this
  exception while rejecting extra column or table grants.
- Strict maintenance starts as `xcelsior_reconciler`; declared task bodies
  retain their separate API, billing and projector scopes. Runtime partition
  DDL stays disabled. The Compose default now matches that executor identity.
- Strict Alembic requires `XCELSIOR_POSTGRES_DSN_MIGRATOR` and verifies a
  direct `xcelsior_migrator` session before running migrations. Legacy
  deployments keep their existing DSN fallback until strict roles are enabled.
- `run_db_migrations.py` reads the private host credential as data and passes
  its variable name to the temporary Compose job. The normal deployment
  entrypoint uses this helper. The secret is absent from command arguments
  and long-running service configuration.
- `docker-compose.db-roles.yml` requires every deployed runtime DSN, pins
  strict mode, replaces generic connection aliases with that service's
  runtime identity, and clears admin/migrator entries. Background execution
  receives only its required runtime scopes.
- `transfer_db_ownership.py` previews application-only ownership changes and
  records inverse ownership SQL. It refuses an unexpected database, port, owner,
  unclassified relation/sequence or escalating migrator. It avoids
  `REASSIGN OWNED`, preserves extension ownership, and does not change rows or
  logins. Offline backup access includes future migrator-owned objects.
- Native daily partition maintenance uses the one running API generation's
  immutable image ID and a separate migrator environment. Two running API
  generations during a handoff cause a refusal. Native restore drills accept
  that same private credential, restore under the migrator, and inject it
  only into the disposable migration job.
- Native partition maintenance publishes its latest outcome and retained
  last-success timestamp to node-exporter. A separate cutover rule group
  detects missing evidence, a failed run or more than 26 hours without success;
  it remains observable while its in-process task rows are disabled.

## Qualification

Run from the candidate checkout using the project virtual environment:

```sh
.venv/bin/python api/scripts/run_security_gates.py \
  --output artifacts/runtime-db-role-qualification -- \
  tests/test_db_role_workflows.py tests/test_db_role_cutover.py \
  tests/test_db_migration_entrypoint.py tests/test_db_role_restore_drill.py \
  tests/test_db_service_roles.py tests/test_bg_worker.py \
  tests/test_serverless_worker_attempt_binding.py \
  tests/test_restore_tools_cluster.py tests/test_restore_migration_isolation.py \
  tests/test_migration_lock_discipline.py tests/test_telemetry_partitions_are_maintained.py
```

This harness creates and removes a fresh Unix-socket-only PostgreSQL cluster;
it does not load production credentials or contact Tower. Real database tests
cover restricted placement, privilege denials, maintenance registration and
task identities, scoped ownership transfer and inverse ownership restoration,
direct migrator login, new-object backup grants, partition maintenance,
normal Alembic entry, and a full dump/restore/migration/invariant drill.
The Docker boundary is substituted in the native restore test; PostgreSQL,
`pg_dump`, `pg_restore`, disposable database creation/removal and Alembic are
real. This is local qualification, not a production restore receipt.

The production preview ran through a network-disabled container with a
read-only PostgreSQL session. It identified 179 ownership statements. The
eight repair policies remained `report_only`; all twelve protected stopped
job row hashes matched the previous baseline. See the adjacent evidence JSON
for final gate outcomes, hashes and artifact paths.

Final qualification: **110 passed** across database roles, workflows,
ownership, real dump/restore, Alembic and partition behavior; **107 static
checks passed**, with the existing B16.2 contraction and absent-companion-prose
checks explicitly deselected. The updated native wrapper's metric success and
failure paths passed a focused rerun. Ruff, scoped Black, Pyright, shell syntax
and systemd unit syntax checks passed. Whole-tree Black still has the unchanged
252-file baseline documented in the push-readiness audit.

## Ordered production procedure

1. Merge the reviewed candidate through the PR workflow. Build and verify
   the resulting API image. Do not reinstall or restart GPU workers, VTT or
   Tower serving for this database-only cutover.
2. Capture current container/image/config identities, schema, effective
   repair policies, protected job/volume hashes, native backup inventory,
   database ownership/grants and current sessions. Make a fresh checksum
   backup on **5434** and pass an isolated restore before changing logins.
3. Generate distinct credentials privately for API, scheduler, reconciler,
   billing, projector, SSH gateway and migrator. Provision via
   `provision_db_roles.py` using the administrative session; never print the
   credentials. Leave unused read-only/retrieval logins disabled. Save the
   old configuration and login state for rollback.
4. Keep migration credentials only in root-owned mode `0600`
   `/etc/xcelsior/db-maintenance.env`. Runtime configuration must not contain
   that credential or the old owner's password. Persist the resolved strict
   service environments in the private deployment override so later normal
   deploys preserve them. The checked-in strict Compose overlay is the
   template; merge it after existing operational overrides and validate the
   final rendered configuration privately.
5. Install the native partition service/timer and updated restore unit;
   reload systemd. Stage the API with strict roles, verify readiness and an
   actual restricted login, then hand over the API, maintenance, scheduler
   and SSH gateway. Maintenance must use `reconciler` and external partition
   maintenance. Wait for task claims/settlements and service heartbeats.
   Audit every used runtime login; verify no runtime session uses `xcelsior`.
6. Preview the ownership transfer against the explicitly verified application
   database and port. Review and retain its inverse SQL. Apply only that plan
   under the administrative session. A lock failure must roll back and stop
   the cutover; do not increase lock waits around live traffic. The ownership
   plan is preparation evidence; require the successful process exit and a
   fresh catalog audit as proof of execution.
7. Run the partition maintenance one-shot as a direct migrator login, verify
   new partition grants and task status, then enable its daily timer. Audit
   retention candidates first; historical legacy-record disposition is not
   authorized by this identity cutover. Install the separate
   `native-db-role-rules.yml` group only with this cutover; verify both new
   textfile metrics are ingested and the three alerts are healthy.
8. Make another fresh backup using the offline backup login. Run the native
   isolated restore with the private migrator credential. Verify schema,
   invariants, cleanup on both known PostgreSQL clusters, and Prometheus
   backup/restore freshness ingestion. Recheck public readiness, runtime
   grants, protected rows/volume, workers/serving and all repair policies.

Example ownership invocation (the admin DSN belongs in the private process
environment, not command arguments):

```sh
python api/scripts/transfer_db_ownership.py \
  --expected-database xcelsior --expected-port 5434 --from-owner xcelsior \
  --evidence /private-release/ownership-plan.json
```

Add `--apply` only to the reviewed production operation, with a new evidence
filename. The script defaults to preview.

## Rollback boundary

Before ownership transfer, restore saved runtime configuration and images,
verify readiness and resume the previous executor. After transfer, the saved
inverse ownership statements are also required if returning to runtime DDL;
retain grants needed by either running generation throughout the handoff.
Stop the external partition timer before returning DDL to the old executor.
Disable newly introduced logins only after their sessions have drained and
the previous service identities are verified. Do not restore a backup over
the live database to undo metadata-only ownership/configuration changes.
Repairs remain `report_only` during forward rollout and rollback.
