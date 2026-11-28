# Xcelsior Operator Runbook

## Failover
1. Trigger `/failover` to detect dead hosts + reassign queued work.
2. Verify `/metrics` queue depth decreases and failed job count stabilizes.
3. If a host is permanently dead, remove with `DELETE /host/{host_id}`.

## Requeue workflow
1. Use `POST /job/{job_id}/requeue` for recoverable failures.
2. Confirm status transitions back to `queued` via `GET /job/{job_id}`.
3. Run `POST /queue/process` to assign immediately.

## Billing reconciliation
1. Bill all completed jobs with `POST /billing/bill-all`.
2. Compare `GET /billing` totals with `/metrics` -> `billing_totals.total_revenue`.
3. For marketplace hosts, verify `GET /marketplace/stats` against recent completed jobs.

## Recovery steps
1. Stop API + worker.
2. Backup `XCELSIOR_DB_PATH`.
3. Restore DB backup if needed.
4. Start worker first, then API.
5. Check `/readyz`, then `/metrics`.

## Off-host backups and secrets

Use a private S3-compatible bucket outside the VPS. Set
`XCELSIOR_OBJECT_STORE_BUCKET` and `XCELSIOR_OBJECT_STORE_REQUIRED=1` in
`/etc/xcelsior/db-maintenance.env`. AWS S3 can use the workload credential
chain. For a custom HTTPS origin set `XCELSIOR_OBJECT_STORE_ENDPOINT` and
`XCELSIOR_OBJECT_STORE_REGION` if needed, and supply scoped object-store
credentials from Secrets Manager. Limit the maintenance identity to the
relevant prefixes. Set a bucket lifecycle and protect backups and volume
checkpoints with the access, encryption, and retention appropriate to the
data they hold.

For a small initial backup set, use an existing allowance or Backblaze B2
within its free allocation. Check account-wide storage and egress before
enabling uploads. Count retained and hidden versions and incomplete
multipart uploads. Configure lifecycle rules for the backup prefixes and
uncompleted multipart uploads, and verify the account's storage and
transaction caps. A backup must exit with a failure if the provider cap
stops the upload. Leave bulk volume copies disabled until their size,
retention, and restore traffic also fit the budget.
[Backblaze pricing](https://www.backblaze.com/cloud-storage/pricing)
[Backblaze data caps](https://www.backblaze.com/docs/cloud-storage-data-caps-and-alerts)

Install the project's locked Python environment with
`cd /opt/xcelsior && uv sync --locked --no-dev`.
Put the Secrets Manager secret ID in `/etc/xcelsior/runtime-secret-id`.
Use a workload identity with access to that one secret and configure its AWS
region for maintenance and deployment. The secret's JSON fields are
`XCELSIOR_POSTGRES_PASSWORD` and `XCELSIOR_SECRETS_KEY`, plus
`XCELSIOR_OBJECT_STORE_ACCESS_KEY`, `XCELSIOR_OBJECT_STORE_SECRET_KEY`, and
optionally `XCELSIOR_OBJECT_STORE_SESSION_TOKEN` for a custom object store.
Copy the existing Fernet key value exactly. Existing encrypted records
depend on it. Include the separately managed process-role DSNs where the
stack uses them. Run Compose through `api/scripts/with-runtime-secrets.sh docker compose`
when creating or recreating a service. The standard deploy uses this wrapper
for the API, scheduler, maintenance worker, and SSH gateway. Once the staged
start and readiness checks pass, remove the migrated values from the host
`.env`. The optional break-glass file
`/run/xcelsior/runtime-break-glass.env` accepts literal `KEY=value` lines and
must be root-owned with mode 0600. Explicitly select it with
`XCELSIOR_USE_RUNTIME_BREAK_GLASS=1`. The wrapper accepts it for one hour.

To drill Postgres password rotation on staging, hold scheduling, change the
role's password with psql's `\password` prompt, and publish that value to the
secret. Recreate the database clients with the wrapper. Confirm an
authenticated connection and `/readyz` from the recreated API and verify
fresh backup and restore. Apply the same sequence to dedicated role
credentials if the staging stack uses them. Substitute the active API and
gateway service names below.

```bash
api/scripts/with-runtime-secrets.sh docker compose --profile blue up -d \
  --no-deps --force-recreate "$live_api" scheduler-worker bg-worker "$live_gateway" postgres-exporter
```

Install the backup and restore units from `api/infra/systemd/` and run
`systemctl daemon-reload`.

```bash
systemctl start xcelsior-db-backup.service
systemctl start xcelsior-db-restore-drill.service
systemctl enable --now xcelsior-db-backup.timer xcelsior-db-restore-drill.timer
```

Read the journal for each job and check the latest JSON in
`/var/backups/xcelsior/restore-evidence/`. Require
`backup_source=object_storage`, a successful invariant check, and a
download-to-validation duration under 30 minutes. The drill downloads the
most recent object for `XCELSIOR_OBJECT_STORE_BACKUP_DB` (default `xcelsior`),
verifies its digest, and restores to a disposable database. It uses the
existing PostgreSQL cluster and verifies that the restored schema can migrate.
Prometheus alerts on stale results, failures, and recovery-time overruns.

Compose keeps Redis AOF and RDB files on the `redis-data` volume. To send an
RDB snapshot to the same bucket, add `XCELSIOR_REDIS_BACKUP_URL` to the
restricted maintenance configuration and enable
`xcelsior-redis-backup.timer`. The authentication and rate-limiting cache is
ephemeral. After infrastructure loss, begin with a clean cache so expired and
consumed OAuth state cannot reappear from an older snapshot. Users may need
to retry sign-in and short-lived rate-limit windows may reset. PostgreSQL
holds refresh state and the durable jobs, payments, and account records.
