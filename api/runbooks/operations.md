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
