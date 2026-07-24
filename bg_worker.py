import logging
import time
import os
import signal
import sys
import threading
import uuid

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from control_plane.scheduled_tasks import register_task, claim_and_run_tasks

log = logging.getLogger("xcelsior.bg_worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [bg-worker] %(message)s",
)

# P3/C3 — scrub PII from bg-worker logs too.
try:
    from log_pii_filter import install as _install_pii_scrub
    _install_pii_scrub()
except Exception:  # pragma: no cover
    pass

_stop = threading.Event()
WORKER_ID = f"bg-worker-{uuid.uuid4().hex[:8]}"


def is_fenced_history_job(
    *,
    active_attempt_id: object | None = None,
    has_fenced_history: bool = False,
) -> bool:
    """True when stop/command authority is the fenced lifecycle path.

    Matches billing/lifecycle classification: active attempt ownership
    *or* any ``job_attempts`` history means unfenced ``stop_container``
    redelivery must not dual-command the host.
    """
    if active_attempt_id is not None:
        return True
    return bool(has_fenced_history)


def reconcile_paused_stopped_jobs() -> int:
    """Re-enqueue unfenced stop for pure-legacy stale ``stopped`` jobs.

    Attempt-owned and fenced-history jobs are excluded at SQL and again
    in Python (defense in depth). Legacy jobs keep throttle, max-attempt,
    and pending-command dedupe.

    Returns the number of stop directives enqueued.
    """
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from routes.agent import enqueue_agent_command

    now = time.time()
    stale_cutoff = now - 120.0
    retry_sec = max(int(os.environ.get("XCELSIOR_RECONCILE_RETRY_SEC", "3600")), 60)
    max_attempts = max(int(os.environ.get("XCELSIOR_RECONCILE_MAX_ATTEMPTS", "3")), 1)
    pool = _get_pg_pool()
    enqueued = 0
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            """
            SELECT j.job_id, j.status, j.host_id, j.active_attempt_id,
                   j.payload->>'container_name' AS container_name,
                   COALESCE(
                       (j.payload->>'paused_at')::float,
                       (j.payload->>'completed_at')::float,
                       j.submitted_at
                   ) AS state_age_ts,
                   COALESCE((j.payload->>'last_reconcile_stop_at')::float, 0)
                       AS last_reconcile_stop_at,
                   COALESCE((j.payload->>'reconcile_stop_count')::int, 0)
                       AS reconcile_stop_count,
                   EXISTS (
                       SELECT 1 FROM job_attempts a WHERE a.job_id = j.job_id
                   ) AS has_fenced_history
              FROM jobs j
             WHERE j.status = 'stopped'
               AND j.host_id IS NOT NULL AND j.host_id <> ''
               AND j.active_attempt_id IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM job_attempts a WHERE a.job_id = j.job_id
               )
             ORDER BY j.submitted_at DESC
             LIMIT 200
            """
        ).fetchall()
        for row in rows:
            # Defense in depth: never unfenced-stop fenced-history jobs even
            # if a future query change reintroduces them into the result set.
            if is_fenced_history_job(
                active_attempt_id=row.get("active_attempt_id"),
                has_fenced_history=bool(row.get("has_fenced_history")),
            ):
                continue
            state_ts = row.get("state_age_ts") or 0.0
            if state_ts > stale_cutoff:
                continue
            last_reconcile_at = float(row.get("last_reconcile_stop_at") or 0.0)
            reconcile_count = int(row.get("reconcile_stop_count") or 0)
            if reconcile_count >= max_attempts:
                continue
            if last_reconcile_at and last_reconcile_at > now - retry_sec:
                continue
            host_id = row["host_id"]
            job_id = row["job_id"]
            cname = row.get("container_name") or f"xcl-{job_id}"

            pending = conn.execute(
                """
                SELECT 1 FROM agent_commands
                 WHERE host_id = %s
                   AND status = 'pending'
                   AND command IN ('stop_container','pause_container')
                   AND args->>'job_id' = %s
                 LIMIT 1
                """,
                (host_id, job_id),
            ).fetchone()
            if pending:
                continue

            try:
                enqueue_agent_command(
                    host_id,
                    "stop_container",
                    {"container_name": cname, "job_id": job_id},
                    created_by="reconcile_sweep",
                    ttl_sec=600,
                )
                conn.execute(
                    """
                    UPDATE jobs
                       SET payload = jsonb_set(
                           jsonb_set(
                               COALESCE(payload, '{}'::jsonb),
                               '{last_reconcile_stop_at}',
                               to_jsonb(%s::float),
                               true
                           ),
                           '{reconcile_stop_count}',
                           to_jsonb(%s::int),
                           true
                       )
                     WHERE job_id = %s
                    """,
                    (now, reconcile_count + 1, job_id),
                )
                enqueued += 1
            except Exception as e:
                log.warning("reconcile enqueue failed job=%s: %s", job_id, e)
            if enqueued >= 50:
                break
        conn.commit()
    if enqueued:
        log.info("Reconcile sweep: re-enqueued %d stop/pause directives", enqueued)
    return enqueued


def main():
    log.info("bg-worker starting — loading tasks…")

    # Force bg tasks env so any transitive import of api.py sees it
    os.environ["XCELSIOR_BG_TASKS"] = "true"

    # 1. Auto-billing cycle (every 5 minutes)
    def _billing_cycle():
        from billing import get_billing_engine
        be = get_billing_engine()
        be.auto_billing_cycle()
        be.check_low_balance_and_topup()
        be.stop_jobs_for_suspended_wallets()
    register_task("billing_cycle", _billing_cycle, 300)

    # 1a-ii. Billing controller (§12.4, Track B B3.3): surface meter invariants
    # — a ran attempt with no meter (billing leak) or an orphaned open meter.
    # Report-only by default; enforcing billing_missing_meter is opt-in via
    # XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER=enforce.
    def _billing_meter_reconcile():
        from control_plane.billing_controller import reconcile_billing_meters_task

        reconcile_billing_meters_task()
    register_task("billing_meter_reconcile", _billing_meter_reconcile, 300)

    # 1b. Expire past-due wallet launch holds (frees available balance)
    def _wallet_hold_expiry():
        from billing import get_billing_engine
        n = get_billing_engine().expire_stale_wallet_holds(limit=500)
        if n:
            log.info("wallet_hold_expiry: expired %d hold(s)", n)
    register_task("wallet_hold_expiry", _wallet_hold_expiry, 60)

    # 1c. Expire past-deadline per-host agent tokens (blueprint §19.2).
    # Verification already refuses an expired credential; this keeps the
    # one-active-token index and operator views honest about what is live.
    def _host_agent_token_expiry():
        from control_plane.agent_tokens import expire_stale_tokens
        from control_plane.db import run_transaction

        n = run_transaction(
            lambda c: expire_stale_tokens(c, limit=500),
            what="host_agent_token_expiry",
        )
        if n:
            log.info("host_agent_token_expiry: expired %d token(s)", n)
    register_task("host_agent_token_expiry", _host_agent_token_expiry, 300)

    # 2. Stripe webhook event processor (every 30 seconds)
    def _webhook_processor():
        from stripe_connect import get_stripe_manager
        sm = get_stripe_manager()
        sm.process_pending_events()
    register_task("webhook_processor", _webhook_processor, 30)

    # 3. Spot price updater (every 10 minutes)
    def _spot_updater():
        from marketplace import get_marketplace_engine
        from spot_pricing import update_all_spot_prices
        update_all_spot_prices()
        me = get_marketplace_engine()
        me.expire_reservations()
    register_task("spot_updater", _spot_updater, 600)

    # 4. Serverless reconcile (autoscaler + dispatch)
    def _serverless_reconcile():
        from serverless.service import get_serverless_service
        get_serverless_service().reconcile_all()
    register_task("serverless_reconcile", _serverless_reconcile, 45)

    def _serverless_webhook_retry():
        from serverless.repo import ServerlessRepo
        from serverless.webhooks import retry_pending_webhooks
        retry_pending_webhooks(ServerlessRepo())
    register_task("serverless_webhook_retry", _serverless_webhook_retry, 30)

    # 5. Cloud burst evaluator (every 2 minutes)
    def _burst_evaluator():
        from cloudburst import get_burst_engine
        cbe = get_burst_engine()
        cbe.evaluate_burst_need()
        cbe.drain_idle_instances()
        cbe.update_burst_spending()
    register_task("burst_evaluator", _burst_evaluator, 120)

    # 6. SLA credit issuance (every hour)
    def _sla_credits():
        import datetime
        from sla import get_sla_engine
        se = get_sla_engine()
        se.auto_issue_credits(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"))
    register_task("sla_credits", _sla_credits, 3600)

    # 7. Event snapshotting (every 15 minutes)
    def _event_snapshots():
        from events import get_snapshot_manager, get_event_store
        sm = get_snapshot_manager()
        sm.snapshot_all_jobs(get_event_store())
    register_task("event_snapshots", _event_snapshots, 900)

    # 8. Data retention / privacy purge (every 6 hours)
    def _privacy_purge():
        from privacy import get_lifecycle_manager, get_consent_manager
        lm = get_lifecycle_manager()
        lm.purge_expired()
        cm = get_consent_manager()
        cm.expire_implied_consents()
    register_task("privacy_purge", _privacy_purge, 21600)

    # 9. Session cleanup (every hour)
    def _session_cleanup():
        from db import UserStore
        count = UserStore.cleanup_expired_sessions()
        if count:
            log.info("Session cleanup: purged %d expired sessions", count)
    register_task("session_cleanup", _session_cleanup, 3600)

    # 10. FINTRAC compliance check (every hour)
    def _fintrac_check():
        from billing import get_billing_engine
        be = get_billing_engine()
        be.fintrac_check_transaction(customer_id="__periodic_scan__", amount_cad=0)
    register_task("fintrac_check", _fintrac_check, 3600)

    # 11. Job log cleanup (daily)
    def _job_log_cleanup():
        from db import _get_pg_pool
        cutoff = time.time() - (7 * 86400)
        pool = _get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute("DELETE FROM job_logs WHERE ts < %s", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
        if deleted:
            log.info("Job log cleanup: purged %d old log rows", deleted)
    register_task("job_log_cleanup", _job_log_cleanup, 86400)

    # 12. Notification + push retention cleanup (every 6 hours)
    def _notification_cleanup():
        from db import NotificationStore, WebPushSubscriptionStore
        notification_retention_days = max(int(os.environ.get("XCELSIOR_NOTIFICATION_RETENTION_DAYS", "30")), 1)
        revoked_retention_days = max(int(os.environ.get("XCELSIOR_WEB_PUSH_REVOKED_RETENTION_DAYS", "30")), 1)
        deleted_notifications = NotificationStore.delete_old(notification_retention_days)
        deleted_revoked = WebPushSubscriptionStore.delete_revoked_older_than(revoked_retention_days)
        if deleted_notifications or deleted_revoked:
            log.info("Notification cleanup: purged %d notifications and %d revoked push subs", deleted_notifications, deleted_revoked)
    register_task("notification_cleanup", _notification_cleanup, 21600)

    # 14. Reconcile paused/stopped jobs (legacy unfenced stop redelivery only)
    def _reconcile_paused_stopped():
        reconcile_paused_stopped_jobs()
    register_task("reconcile_paused_stopped", _reconcile_paused_stopped, 60)

    # 15. user_images pending sweeper
    def _user_images_pending_sweeper():
        from db import _get_pg_pool
        cutoff = time.time() - 3600.0
        stale_queued_cutoff = time.time() - 86400.0
        pool = _get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute(
                "UPDATE user_images SET status = 'failed' WHERE status = 'pending' AND deleted_at = 0 AND created_at < %s",
                (cutoff,)
            )
            marked = cur.rowcount
            cur2 = conn.execute(
                "UPDATE user_images SET status = 'failed', error = 'registry_down_24h' WHERE status = 'queued_registry_down' AND deleted_at = 0 AND created_at < %s",
                (stale_queued_cutoff,)
            )
            stale_queued = cur2.rowcount
            conn.commit()
        if marked: log.info("user_images sweeper: marked %d pending→failed (>1h old)", marked)
        if stale_queued: log.info("user_images sweeper: marked %d queued_registry_down→failed (>24h old)", stale_queued)
    register_task("user_images_pending_sweeper", _user_images_pending_sweeper, 300)

    # 16. user_images hard-delete GC
    def _user_images_hard_delete_gc():
        import os as _os
        from db import _get_pg_pool
        retention_days = int(_os.environ.get("XCELSIOR_USER_IMAGES_GC_DAYS", "30"))
        if retention_days <= 0: return
        cutoff = time.time() - (retention_days * 86400.0)
        pool = _get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute(
                "DELETE FROM user_images WHERE deleted_at > 0 AND deleted_at < %s",
                (cutoff,)
            )
            purged = cur.rowcount
            conn.commit()
        if purged: log.info("user_images GC: hard-deleted %d rows (deleted_at < %d days ago)", purged, retention_days)
    register_task("user_images_hard_delete_gc", _user_images_hard_delete_gc, 86400)

    # 17. registry health probe
    def _registry_health_probe():
        import registry_health
        registry_health.probe_registry()
    probe_int = __import__("registry_health").PROBE_INTERVAL_SEC
    register_task("registry_health_probe", _registry_health_probe, probe_int)

    # 18. retry snapshots queued during registry outage
    def _retry_queued_snapshots():
        import registry_health
        from control_plane.job_targets import resolve_job_command_target
        from db import _get_pg_pool
        from routes.agent import enqueue_agent_command

        if not registry_health.is_registry_healthy(): return
        pool = _get_pg_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT image_id, owner_id, source_job_id, host_id, name, tag, image_ref, description FROM user_images WHERE status = 'queued_registry_down' AND deleted_at = 0 ORDER BY created_at ASC LIMIT 20"
            ).fetchall()

        promoted = 0
        for image_id, owner_id, job_id, host_id, name, tag, image_ref, description in rows:
            if not host_id:
                with pool.connection() as conn:
                    conn.execute("UPDATE user_images SET status='failed', error='host_missing' WHERE image_id=%s", (image_id,))
                    conn.commit()
                continue
            # Resolve residual container identity: attempt-owned must use
            # xcl-{job}-{attempt_prefix}, never bare xcl-{job}.
            target = resolve_job_command_target(str(job_id)) if job_id else None
            if target is not None and not target.allows_unfenced_container_command:
                log.warning(
                    "snapshot registry-retry skip image_id=%s job=%s: "
                    "fenced history without live authority",
                    image_id,
                    job_id,
                )
                with pool.connection() as conn:
                    conn.execute(
                        "UPDATE user_images SET status='failed', error='fenced_no_live_authority' "
                        "WHERE image_id=%s AND status='queued_registry_down'",
                        (image_id,),
                    )
                    conn.commit()
                continue
            container_name = (
                target.container_name
                if target is not None
                else f"xcl-{job_id}"
            )
            use_host = (target.host_id if target and target.host_id else host_id)
            try:
                enqueue_agent_command(
                    use_host,
                    "snapshot_container",
                    {
                        "job_id": job_id,
                        "image_id": image_id,
                        "owner_id": owner_id,
                        "image_ref": image_ref,
                        "name": name,
                        "tag": tag,
                        "description": description or "",
                        "container_name": container_name,
                    },
                    created_by="bg_worker_snapshot_registry_retry",
                )
                with pool.connection() as conn:
                    conn.execute("UPDATE user_images SET status='pending' WHERE image_id=%s AND status='queued_registry_down'", (image_id,))
                    conn.commit()
                promoted += 1
            except Exception as e:
                log.warning(
                    "snapshot registry-retry enqueue failed image_id=%s err=%s",
                    image_id,
                    type(e).__name__,
                )
        if promoted:
            log.info(
                "snapshot registry-retry: promoted %d queued snapshots to pending",
                promoted,
            )
    register_task("snapshot_queue_retry", _retry_queued_snapshots, 60)

    # 19. agent upgrade rollout watchdog
    def _agent_rollout_watchdog():
        from db import _get_pg_pool
        from routes.agent import enqueue_agent_command
        ROLLBACK_GRACE_SEC = 300
        MARK_COMPLETE_MIN_AGE_SEC = 45

        pool = _get_pg_pool()
        completed = rolled_back = 0
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT r.id, r.host_id, r.target_sha, r.enqueued_at,
                       EXTRACT(EPOCH FROM NOW()) - r.enqueued_at AS age_sec,
                       h.payload->>'agent_sha256' AS current_sha
                  FROM agent_rollouts r
             LEFT JOIN hosts h ON h.host_id = r.host_id
                 WHERE r.status = 'pending'
                 ORDER BY r.enqueued_at ASC
                 LIMIT 100
                """,
            ).fetchall()

        for rid, host_id, target_sha, enqueued_at, age_sec, current_sha in rows:
            age = float(age_sec or 0)
            if current_sha == target_sha and age >= MARK_COMPLETE_MIN_AGE_SEC:
                with pool.connection() as conn:
                    conn.execute("UPDATE agent_rollouts SET status='completed', completed_at=EXTRACT(EPOCH FROM NOW()), last_check_at=EXTRACT(EPOCH FROM NOW()) WHERE id=%s AND status='pending'", (rid,))
                    conn.commit()
                completed += 1
                continue
            if age >= ROLLBACK_GRACE_SEC:
                try:
                    enqueue_agent_command(
                        host_id, "rollback_agent", {"reason": "upgrade_timeout", "target_sha": target_sha}, created_by="bg_rollout_watchdog"
                    )
                    with pool.connection() as conn:
                        conn.execute("UPDATE agent_rollouts SET status='rolled_back', completed_at=EXTRACT(EPOCH FROM NOW()), last_check_at=EXTRACT(EPOCH FROM NOW()), error='heartbeat_stale_or_old_sha' WHERE id=%s AND status='pending'", (rid,))
                        conn.commit()
                    rolled_back += 1
                except Exception as e:
                    log.warning("rollout watchdog: enqueue rollback failed host=%s: %s", host_id, e)
                continue
            with pool.connection() as conn:
                conn.execute("UPDATE agent_rollouts SET last_check_at=EXTRACT(EPOCH FROM NOW()) WHERE id=%s", (rid,))
                conn.commit()
        if completed or rolled_back: log.info("rollout watchdog: completed=%d rolled_back=%d", completed, rolled_back)
    register_task("agent_rollout_watchdog", _agent_rollout_watchdog, 30)

    # 20. Stuck-job reaper
    try:
        from reaper import reaper_tick
        register_task("reaper_tick", reaper_tick, 60)
    except Exception as e:
        log.warning("reaper_tick not registered: %s", e)

    # 21. Lightning Network deposit watcher
    try:
        import lightning as _ln
        if _ln.LN_ENABLED:
            def _ln_credit_callback(customer_id, amount_cad, deposit_id):
                from billing import get_billing_engine
                be = get_billing_engine()
                # The idempotency key is mandatory: process_ln_deposits
                # retries any deposit whose mark_credited did not land, so
                # without it a failed mark re-credits the wallet on every
                # 5-second sweep.
                return be.deposit(
                    customer_id,
                    amount_cad,
                    f"Lightning deposit {deposit_id}",
                    idempotency_key=_ln.credit_idempotency_key(deposit_id),
                )
            # Do NOT use _ln.start_ln_watcher thread here.
            # Instead run it as a regular scheduled task since we migrated to pg!
            def _ln_watcher_task():
                _ln.process_ln_deposits(credit_callback=_ln_credit_callback)
            register_task("lightning_watcher", _ln_watcher_task, 5)
            log.info("Lightning deposit watcher registered")

            # Companion §10.1 reconciliation. Report-only: it records
            # discrepancies between deposits and the wallet ledger and
            # never moves money itself (Track B B9.3b-3). Runs whether or
            # not the watcher is healthy — a stalled watcher is precisely
            # the condition it exists to surface.
            def _ln_reconcile_task():
                from control_plane.billing_reconcile import (
                    reconcile_ln_deposits_task,
                )

                reconcile_ln_deposits_task()

            register_task("lightning_reconcile", _ln_reconcile_task, 300)
            log.info("Lightning deposit reconciler registered")
    except Exception as e:
        log.warning("Lightning watcher startup failed: %s", e)

    # 22. Artifact catalog janitor / cleanup
    def _artifact_catalog_janitor():
        from artifacts import get_artifact_manager
        mgr = get_artifact_manager()
        mgr.cleanup_expired()
    register_task("artifact_catalog_janitor", _artifact_catalog_janitor, 60)

    # 22b. Artifact inventory reconciliation (Track B B9.2d, companion §6.6).
    # Report-only: flags `available` rows whose bytes have vanished (a 404
    # waiting to happen) without touching state or bytes. Off the request
    # path by construction — HEAD-per-row is too slow for a request, and a
    # 10-minute cadence catches drift long before a retention window.
    def _artifact_inventory_reconcile():
        from artifacts import get_artifact_manager
        get_artifact_manager().reconcile_inventory()
    register_task("artifact_inventory_reconcile", _artifact_inventory_reconcile, 600)

    # ── Launch executor thread ───────────────────────────────────────
    def _executor_loop():
        log.info("Durable scheduled task executor started (worker_id=%s)", WORKER_ID)
        while not _stop.is_set():
            try:
                executed = claim_and_run_tasks(WORKER_ID)
                # Sleep briefly if we didn't do any work
                if not executed:
                    _stop.wait(2.0)
            except Exception as e:
                log.error("Executor loop error: %s", e)
                _stop.wait(5.0)

    t = threading.Thread(target=_executor_loop, daemon=True, name="bg-executor")
    t.start()

    # ── Graceful shutdown on SIGTERM / SIGINT ─────────────────────────
    def _shutdown(signum, frame):
        log.info("bg-worker received signal %d — shutting down", signum)
        _stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Block until stopped
    _stop.wait()
    t.join(timeout=5)
    log.info("bg-worker stopped")

if __name__ == "__main__":
    main()
