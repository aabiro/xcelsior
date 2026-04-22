"""Background task worker — runs all periodic tasks outside the API server.

Usage:
    python bg_worker.py

This process starts every background task that was previously embedded in the
FastAPI lifespan (billing cycle, webhook processing, spot pricing, inference
scaledown, cloudburst evaluation, SLA credits, event snapshots, privacy purge,
session cleanup, FINTRAC compliance, log cleanup, notification cleanup, and the
Lightning Network deposit watcher).

The API server (gunicorn) should set XCELSIOR_BG_TASKS=false so that it only
handles HTTP requests, keeping its PG connection pool free for user traffic.
"""

import logging
import os
import signal
import sys
import threading
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))


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


def _bg_loop(name: str, func, interval_sec: int):
    log.info("BG TASK started: %s (interval=%ds)", name, interval_sec)
    while not _stop.is_set():
        try:
            func()
        except Exception as e:
            log.error("BG TASK %s error: %s", name, e)
        _stop.wait(interval_sec)
    log.info("BG TASK stopped: %s", name)


def main():
    log.info("bg-worker starting — loading tasks…")

    # Force bg tasks env so any transitive import of api.py sees it
    os.environ["XCELSIOR_BG_TASKS"] = "true"

    # ── Collect tasks ─────────────────────────────────────────────────
    tasks: list[tuple[str, object, int]] = []

    # 1. Auto-billing cycle (every 5 minutes)
    def _billing_cycle():
        from billing import get_billing_engine

        be = get_billing_engine()
        be.auto_billing_cycle()
        be.check_low_balance_and_topup()
        be.stop_jobs_for_suspended_wallets()

    tasks.append(("billing_cycle", _billing_cycle, 300))

    # 2. Stripe webhook event processor (every 30 seconds)
    def _webhook_processor():
        from stripe_connect import get_stripe_manager

        sm = get_stripe_manager()
        sm.process_pending_events()

    tasks.append(("webhook_processor", _webhook_processor, 30))

    # 3. Spot price updater (every 10 minutes)
    def _spot_updater():
        from marketplace import get_marketplace_engine

        me = get_marketplace_engine()
        me.update_spot_prices()
        me.expire_reservations()

    tasks.append(("spot_updater", _spot_updater, 600))

    # 4. Inference scaledown (every 5 minutes)
    def _inference_scaledown():
        from inference import get_inference_engine

        ie = get_inference_engine()
        ie.scaledown_idle_workers()

    tasks.append(("inference_scaledown", _inference_scaledown, 300))

    # 5. Cloud burst evaluator (every 2 minutes)
    def _burst_evaluator():
        from cloudburst import get_burst_engine

        cbe = get_burst_engine()
        cbe.evaluate_burst_need()
        cbe.drain_idle_instances()
        cbe.update_burst_spending()

    tasks.append(("burst_evaluator", _burst_evaluator, 120))

    # 6. SLA credit issuance (every hour)
    def _sla_credits():
        import datetime
        from sla import get_sla_engine

        se = get_sla_engine()
        se.auto_issue_credits(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"))

    tasks.append(("sla_credits", _sla_credits, 3600))

    # 7. Event snapshotting (every 15 minutes)
    def _event_snapshots():
        from events import get_snapshot_manager, get_event_store

        sm = get_snapshot_manager()
        sm.snapshot_all_jobs(get_event_store())

    tasks.append(("event_snapshots", _event_snapshots, 900))

    # 8. Data retention / privacy purge (every 6 hours)
    def _privacy_purge():
        from privacy import get_lifecycle_manager, get_consent_manager

        lm = get_lifecycle_manager()
        lm.purge_expired()
        cm = get_consent_manager()
        cm.expire_implied_consents()

    tasks.append(("privacy_purge", _privacy_purge, 21600))

    # 9. Session cleanup (every hour)
    def _session_cleanup():
        from db import UserStore

        count = UserStore.cleanup_expired_sessions()
        if count:
            log.info("Session cleanup: purged %d expired sessions", count)

    tasks.append(("session_cleanup", _session_cleanup, 3600))

    # 10. FINTRAC compliance check (every hour)
    def _fintrac_check():
        from billing import get_billing_engine

        be = get_billing_engine()
        be.fintrac_check_transaction(customer_id="__periodic_scan__", amount_cad=0)

    tasks.append(("fintrac_check", _fintrac_check, 3600))

    # 11. Job log cleanup — prune old entries from job_logs (daily)
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

    tasks.append(("job_log_cleanup", _job_log_cleanup, 86400))

    # 12. Notification + push retention cleanup (every 6 hours)
    def _notification_cleanup():
        from db import NotificationStore, WebPushSubscriptionStore

        notification_retention_days = max(
            int(os.environ.get("XCELSIOR_NOTIFICATION_RETENTION_DAYS", "30")),
            1,
        )
        revoked_retention_days = max(
            int(os.environ.get("XCELSIOR_WEB_PUSH_REVOKED_RETENTION_DAYS", "30")),
            1,
        )
        deleted_notifications = NotificationStore.delete_old(notification_retention_days)
        deleted_revoked = WebPushSubscriptionStore.delete_revoked_older_than(revoked_retention_days)
        if deleted_notifications or deleted_revoked:
            log.info(
                "Notification cleanup: purged %d notifications and %d revoked push subs",
                deleted_notifications,
                deleted_revoked,
            )

    tasks.append(("notification_cleanup", _notification_cleanup, 21600))

    # 14. P3/A4 — reconcile stopped/paused jobs vs agent queue.
    # If a job has been in stopped/paused state for > 120s and has no
    # pending agent_commands row for that host, re-enqueue the
    # corresponding directive. pause/stop_container are idempotent
    # (docker stop on already-stopped = noop) so re-delivery is safe.
    # Bounded at 50 jobs per cycle to keep the queue well-behaved.
    def _reconcile_paused_stopped():
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        from routes.agent import enqueue_agent_command

        now = time.time()
        stale_cutoff = now - 120.0
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            # Pull stale jobs. paused_at lives in payload; completed_at lives
            # in payload for stopped; fall back to submitted_at if absent.
            rows = conn.execute(
                """
                SELECT j.job_id, j.status, j.host_id,
                       j.payload->>'container_name' AS container_name,
                       COALESCE(
                           (j.payload->>'paused_at')::float,
                           (j.payload->>'completed_at')::float,
                           j.submitted_at
                       ) AS state_age_ts
                  FROM jobs j
                 WHERE j.status IN ('stopped', 'paused_low_balance', 'user_paused')
                   AND j.host_id IS NOT NULL AND j.host_id <> ''
                 ORDER BY j.submitted_at DESC
                 LIMIT 200
                """
            ).fetchall()
            enqueued = 0
            for row in rows:
                state_ts = row.get("state_age_ts") or 0.0
                if state_ts > stale_cutoff:
                    continue  # fresh — agent may still be draining
                host_id = row["host_id"]
                job_id = row["job_id"]
                status = row["status"]
                cname = row.get("container_name") or f"xcl-{job_id}"

                # Is there already a pending reconcile directive for this job?
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

                cmd = "stop_container" if status == "stopped" else "pause_container"
                try:
                    enqueue_agent_command(
                        host_id,
                        cmd,
                        {"container_name": cname, "job_id": job_id},
                        created_by="reconcile_sweep",
                        ttl_sec=600,
                    )
                    enqueued += 1
                except Exception as e:
                    log.warning("reconcile enqueue failed job=%s: %s", job_id, e)
                if enqueued >= 50:
                    break
        if enqueued:
            log.info("Reconcile sweep: re-enqueued %d stop/pause directives", enqueued)

    tasks.append(("reconcile_paused_stopped", _reconcile_paused_stopped, 60))

    # 15. P3/A5 — user_images pending sweeper.
    # Snapshots that stay 'pending' for > 1h almost certainly mean the
    # worker crashed mid-commit or the callback never fired. Mark them
    # failed so the UI stops spinning and the owner can retry.
    def _user_images_pending_sweeper():
        from db import _get_pg_pool

        cutoff = time.time() - 3600.0
        # Phase E/E8 — queued_registry_down rows can linger much longer
        # (by design: we're waiting for the operator to fix the registry)
        # but 24h is a sensible upper bound. Past that we assume the
        # user has moved on and the row is dead weight.
        stale_queued_cutoff = time.time() - 86400.0
        pool = _get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute(
                """
                UPDATE user_images
                   SET status = 'failed'
                 WHERE status = 'pending'
                   AND deleted_at = 0
                   AND created_at < %s
                """,
                (cutoff,),
            )
            marked = cur.rowcount
            cur2 = conn.execute(
                """
                UPDATE user_images
                   SET status = 'failed', error = 'registry_down_24h'
                 WHERE status = 'queued_registry_down'
                   AND deleted_at = 0
                   AND created_at < %s
                """,
                (stale_queued_cutoff,),
            )
            stale_queued = cur2.rowcount
            conn.commit()
        if marked:
            log.info("user_images sweeper: marked %d pending→failed (>1h old)", marked)
        if stale_queued:
            log.info(
                "user_images sweeper: marked %d queued_registry_down→failed (>24h old)",
                stale_queued,
            )

    tasks.append(("user_images_pending_sweeper", _user_images_pending_sweeper, 300))

    # 16. P3/C2 — user_images hard-delete GC.
    # Soft-deleted rows (deleted_at > 0) pile up over time. After a
    # retention window (default 30d) the audit value is exhausted and the
    # rows bloat the table + its partial unique index. Hard-delete them.
    def _user_images_hard_delete_gc():
        import os as _os
        from db import _get_pg_pool

        retention_days = int(_os.environ.get("XCELSIOR_USER_IMAGES_GC_DAYS", "30"))
        if retention_days <= 0:
            return  # disabled
        cutoff = time.time() - (retention_days * 86400.0)
        pool = _get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM user_images
                 WHERE deleted_at > 0
                   AND deleted_at < %s
                """,
                (cutoff,),
            )
            purged = cur.rowcount
            conn.commit()
        if purged:
            log.info(
                "user_images GC: hard-deleted %d rows (deleted_at < %d days ago)",
                purged, retention_days,
            )

    tasks.append(("user_images_hard_delete_gc", _user_images_hard_delete_gc, 86400))

    # 17. Phase E/E7 — registry health probe.
    # Polls ``{XCELSIOR_REGISTRY_URL}/v2/`` on a fixed interval and
    # exports three Prometheus gauges (reachable / last_probe_ts /
    # latency_ms) plus updates the in-process cache used by E8's
    # queue-on-down path in the snapshot endpoint. The task is cheap
    # (one HTTP GET with a 5s timeout) and tolerates missing env — if
    # ``XCELSIOR_REGISTRY_URL`` is unset it no-ops silently.
    def _registry_health_probe():
        import registry_health

        registry_health.probe_registry()

    tasks.append((
        "registry_health_probe",
        _registry_health_probe,
        # registry_health.PROBE_INTERVAL_SEC defaults to 300s; read it
        # lazily so env changes are respected at startup.
        __import__("registry_health").PROBE_INTERVAL_SEC,
    ))

    # 18. Phase E/E8 — retry snapshots queued during registry outage.
    # When the snapshot endpoint runs while the registry is unhealthy
    # it inserts a ``user_images`` row with ``status='queued_registry_down'``
    # *without* enqueuing an agent command. This sweeper picks those
    # rows up as soon as the registry recovers and enqueues the real
    # ``snapshot_container`` command, flipping status to ``pending``.
    # Oldest-first, capped per-tick so a huge backlog can't monopolise
    # a single bg tick.
    def _retry_queued_snapshots():
        import registry_health
        from db import _get_pg_pool
        from routes.agent import enqueue_agent_command

        if not registry_health.is_registry_healthy():
            return  # still down — try again next tick

        pool = _get_pg_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT image_id, owner_id, source_job_id, host_id,
                       name, tag, image_ref, description
                  FROM user_images
                 WHERE status = 'queued_registry_down'
                   AND deleted_at = 0
                 ORDER BY created_at ASC
                 LIMIT 20
                """,
            ).fetchall()

        promoted = 0
        for (image_id, owner_id, job_id, host_id, name, tag, image_ref, description) in rows:
            if not host_id:
                # Host disappeared while we were queued — fail the row
                # rather than leave it stuck forever.
                with pool.connection() as conn:
                    conn.execute(
                        "UPDATE user_images SET status='failed', "
                        "error='host_missing' WHERE image_id=%s",
                        (image_id,),
                    )
                    conn.commit()
                continue
            try:
                enqueue_agent_command(
                    host_id,
                    "snapshot_container",
                    {
                        "job_id": job_id,
                        "image_id": image_id,
                        "owner_id": owner_id,
                        "image_ref": image_ref,
                        "name": name,
                        "tag": tag,
                        "description": description or "",
                    },
                    created_by="bg_worker_e8_retry",
                )
                with pool.connection() as conn:
                    conn.execute(
                        "UPDATE user_images SET status='pending' "
                        "WHERE image_id=%s AND status='queued_registry_down'",
                        (image_id,),
                    )
                    conn.commit()
                promoted += 1
            except Exception as e:
                log.warning(
                    "E8 retry enqueue failed image_id=%s err=%s",
                    image_id, type(e).__name__,
                )

        if promoted:
            log.info("E8 retry: promoted %d queued snapshots to pending", promoted)

    tasks.append(("snapshot_queue_retry", _retry_queued_snapshots, 60))

    # 19. P1.2 — agent upgrade rollout watchdog.
    # After the admin rollout endpoint enqueues ``upgrade_agent`` for a
    # host it inserts a tracking row (agent_rollouts). Hosts heartbeat
    # their current ``agent_sha256`` on every PUT /host, which lives in
    # hosts.payload->>'agent_sha256'. This watchdog:
    #   * Marks rollouts completed once the heartbeat sha matches target.
    #   * Enqueues ``rollback_agent`` once enqueued_at is older than
    #     ROLLBACK_GRACE_SEC AND the host is still reporting the old sha
    #     (or has gone silent — heartbeat freshness is the caller's
    #     problem; we only compare what we have).
    # Runs every 30s. Cheap: one SELECT per tick, bounded by LIMIT 100.
    ROLLBACK_GRACE_SEC = 300  # 5 min — generous for slow networks + systemd respawn
    MARK_COMPLETE_MIN_AGE_SEC = 45  # wait one heartbeat cycle before declaring success

    def _agent_rollout_watchdog():
        from db import _get_pg_pool
        from routes.agent import enqueue_agent_command

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

        for (rid, host_id, target_sha, enqueued_at, age_sec, current_sha) in rows:
            age = float(age_sec or 0)
            # Success path — heartbeat now reports the target sha.
            if current_sha == target_sha and age >= MARK_COMPLETE_MIN_AGE_SEC:
                with pool.connection() as conn:
                    conn.execute(
                        "UPDATE agent_rollouts SET status='completed', "
                        "completed_at=EXTRACT(EPOCH FROM NOW()), "
                        "last_check_at=EXTRACT(EPOCH FROM NOW()) "
                        "WHERE id=%s AND status='pending'",
                        (rid,),
                    )
                    conn.commit()
                completed += 1
                continue
            # Rollback path — past grace, host has not picked up the
            # new bytes. Enqueue rollback_agent exactly once (status
            # flip to 'rolled_back' guarantees this even if the task
            # re-runs before the worker drains).
            if age >= ROLLBACK_GRACE_SEC:
                try:
                    enqueue_agent_command(
                        host_id,
                        "rollback_agent",
                        {"reason": "upgrade_timeout", "target_sha": target_sha},
                        created_by="bg_rollout_watchdog",
                    )
                    with pool.connection() as conn:
                        conn.execute(
                            "UPDATE agent_rollouts SET status='rolled_back', "
                            "completed_at=EXTRACT(EPOCH FROM NOW()), "
                            "last_check_at=EXTRACT(EPOCH FROM NOW()), "
                            "error='heartbeat_stale_or_old_sha' "
                            "WHERE id=%s AND status='pending'",
                            (rid,),
                        )
                        conn.commit()
                    rolled_back += 1
                    log.warning(
                        "rollout watchdog: auto-rollback host=%s age=%.0fs "
                        "target=%s current=%s",
                        host_id, age, target_sha[:8], (current_sha or "none")[:8],
                    )
                except Exception as e:
                    log.warning(
                        "rollout watchdog: enqueue rollback failed host=%s: %s",
                        host_id, e,
                    )
                continue
            # Otherwise: still within grace, just touch last_check_at.
            with pool.connection() as conn:
                conn.execute(
                    "UPDATE agent_rollouts SET last_check_at="
                    "EXTRACT(EPOCH FROM NOW()) WHERE id=%s",
                    (rid,),
                )
                conn.commit()

        if completed or rolled_back:
            log.info(
                "rollout watchdog: completed=%d rolled_back=%d",
                completed, rolled_back,
            )

    tasks.append(("agent_rollout_watchdog", _agent_rollout_watchdog, 30))

    # 19. Lightning Network deposit watcher (every 5 seconds)
    try:
        import lightning as _ln

        if _ln.LN_ENABLED:

            def _ln_credit_callback(customer_id, amount_cad, deposit_id):
                from billing import get_billing_engine

                be = get_billing_engine()
                be.deposit(customer_id, amount_cad, f"Lightning deposit {deposit_id}")

            _ln.start_ln_watcher(interval=5, credit_callback=_ln_credit_callback)
            log.info("Lightning deposit watcher started")
        else:
            log.debug("Lightning disabled — watcher not started")
    except Exception as e:
        log.warning("Lightning watcher startup failed (non-fatal): %s", e)

    # ── Launch threads ────────────────────────────────────────────────
    threads = []
    for name, func, interval in tasks:
        t = threading.Thread(target=_bg_loop, args=(name, func, interval), daemon=True)
        t.start()
        threads.append(t)

    log.info("bg-worker running — %d tasks active", len(tasks))

    # ── Graceful shutdown on SIGTERM / SIGINT ─────────────────────────
    def _shutdown(signum, frame):
        log.info("bg-worker received signal %d — shutting down", signum)
        _stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Block until stopped
    _stop.wait()

    # Give threads a moment to finish
    for t in threads:
        t.join(timeout=5)
    log.info("bg-worker stopped")


if __name__ == "__main__":
    main()
