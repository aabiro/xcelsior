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

    # 13. Lightning Network deposit watcher (every 5 seconds)
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
