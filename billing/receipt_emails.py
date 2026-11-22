"""Email the run receipt when an instance finishes.

On by default: a renter gets the receipt (billed seconds, the locked rate, what
was debited) without asking. The account's **Run receipts** setting
(``preferences.receipt_emails``) turns it off, and so does turning off email
notifications altogether.

A periodic sweep (``bg_worker`` task ``receipt_emails``) finds finished
customer instances and emails each receipt once (``XCELSIOR_RECEIPT_EMAILS=0``
stops it for the whole service):

* Only runs that ended after ``receipt_email_settings.enabled_since`` (when
  migration 168 ran), so switching the feature on mails nobody about old runs.
* It waits until the receipt is ``final`` and, for up to
  :data:`SETTLE_GRACE_SECONDS`, until it is ``reconciled`` (the last billing
  cycle has charged). After that it sends what the ledger says, outstanding
  amount included, rather than never sending.
* ``receipt_emails.job_id`` is the key: a sweep claims the row before it sends,
  so two sweeps cannot both email one run. A failed send is retried up to
  :data:`MAX_ATTEMPTS` times; opted-out accounts, serverless workers and Slurm
  runs are recorded as ``skipped`` so they are not looked at again.

Every number in the email comes from ``run_receipts.build_receipt``, the one
receipt builder the page, the API and MCP read.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from money import micros_to_cad

log = logging.getLogger("xcelsior.receipt_emails")

PREFERENCE_KEY = "receipt_emails"
SETTLE_GRACE_SECONDS = 1800
MAX_ATTEMPTS = 3
BATCH = 25


def sweep_enabled() -> bool:
    """The operator switch: ``XCELSIOR_RECEIPT_EMAILS=0`` stops every receipt email."""
    return os.environ.get("XCELSIOR_RECEIPT_EMAILS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def receipts_wanted(user: dict) -> bool:
    """Whether this account gets receipt emails. Default on."""
    if not int(user.get("notifications_enabled", 1) or 0):
        return False
    prefs = user.get("preferences") or {}
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except ValueError:
            prefs = {}
    return (prefs if isinstance(prefs, dict) else {}).get(PREFERENCE_KEY, True) is not False


def _money(micros: int | None) -> str:
    if micros is None:
        return "-"
    return f"${micros_to_cad(int(micros)):,.2f}"


def _rate(micros: int | None) -> str:
    if micros is None:
        return "varied by run"
    value = micros_to_cad(int(micros))
    return f"${value:.4f}".rstrip("0").rstrip(".") + "/hr" if value < 1 else f"${value:,.2f}/hr"


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} s"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def render_receipt_email(receipt: dict, base_url: str) -> tuple[str, str]:
    """Subject and plain-text body for one receipt."""
    name = receipt.get("name") or receipt["job_id"]
    gpu = receipt.get("gpu_model") or "GPU"
    count = int(receipt.get("gpu_count") or 1)
    gpus = f"{count}× {gpu}" if count > 1 else gpu
    total = _money(receipt.get("total_micros"))
    subject = f"Receipt: {name} on {gpus}, {total}"
    lines = [
        f"Your instance {name} has finished. Here is what it cost.",
        "",
        f"GPU:            {gpus}",
        f"Billed time:    {_duration(float(receipt.get('billed_seconds') or 0))}"
        f" ({float(receipt.get('billed_seconds') or 0):,.0f} seconds, billed by the second)",
        f"Rate:           {_rate(receipt.get('rate_micros_per_hour'))}, locked when you launched",
        f"Total:          {total}",
    ]
    rounded = receipt.get("hourly_rounded_micros")
    if rounded is not None and rounded > (receipt.get("total_micros") or 0):
        lines.append(
            f"In whole hours: {_money(rounded)} ({receipt.get('hourly_rounded_label')}; "
            "not what you were charged)"
        )
    lines.append(f"Paid from your wallet: {_money(receipt.get('wallet_debit_micros'))}")
    if receipt.get("outstanding_micros"):
        lines.append(
            f"Still to be charged: {_money(receipt['outstanding_micros'])} "
            "(the last billing cycle had not run when this was sent)"
        )
    if receipt.get("credits_micros"):
        lines.append(f"Credits and refunds: {_money(receipt['credits_micros'])}")
    lines += [
        "",
        f"Full receipt: {base_url}/dashboard/instances/{receipt['job_id']}",
        "",
        "Amounts are in Canadian dollars.",
        f"Turn these emails off in Settings → Notifications: {base_url}/dashboard/settings",
    ]
    return subject, "\n".join(lines)


def _account_for_wallet(cur, owner_id: str) -> dict | None:
    return cur.execute(
        """SELECT email, preferences, notifications_enabled
             FROM users WHERE customer_id = %s
            ORDER BY created_at LIMIT 1""",
        (owner_id,),
    ).fetchone()


def _record(cur, job_id: str, owner_id: str, status: str, detail: str = "") -> None:
    cur.execute(
        """INSERT INTO receipt_emails (job_id, owner_id, status, detail)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (job_id) DO UPDATE
              SET status = EXCLUDED.status, detail = EXCLUDED.detail, updated_at = now()""",
        (job_id, owner_id, status, detail[:500]),
    )


def _claim(cur, job_id: str, owner_id: str) -> bool:
    """Take the run for this sweep; False when another sweep has it or it is done."""
    row = cur.execute(
        f"""INSERT INTO receipt_emails (job_id, owner_id, status)
            VALUES (%s, %s, 'sending')
            ON CONFLICT (job_id) DO UPDATE
               SET status = 'sending', attempts = receipt_emails.attempts + 1, updated_at = now()
             WHERE receipt_emails.status = 'failed'
               AND receipt_emails.attempts < {MAX_ATTEMPTS}
            RETURNING job_id""",
        (job_id, owner_id),
    ).fetchone()
    return row is not None


def _candidates(cur, since_epoch: float, limit: int) -> list[str]:
    from run_receipts import TERMINAL_JOB_STATES

    rows = cur.execute(
        f"""SELECT j.job_id FROM jobs j
             WHERE j.status = ANY(%s) AND COALESCE(j.owner_id, '') <> ''
               AND EXISTS (SELECT 1 FROM usage_meters m
                            WHERE m.job_id = j.job_id AND m.completed_at > %s)
               AND NOT EXISTS (SELECT 1 FROM receipt_emails r
                                WHERE r.job_id = j.job_id
                                  AND (r.status <> 'failed' OR r.attempts >= {MAX_ATTEMPTS}))
             ORDER BY j.submitted_at
             LIMIT %s""",
        (sorted(TERMINAL_JOB_STATES), since_epoch, limit),
    ).fetchall()
    return [row["job_id"] for row in rows]


def send_pending_receipt_emails(
    *,
    limit: int = BATCH,
    now: float | None = None,
    send: Callable[[str, str, str], bool] | None = None,
    base_url: str | None = None,
) -> dict[str, int]:
    """Email the receipts of finished runs that have not had one. Returns counts."""
    from psycopg.rows import dict_row

    from db import _get_pg_pool
    from run_receipts import build_receipt
    from scheduler import get_job

    if send is None:
        from scheduler import ALERT_CONFIG, send_email

        if not ALERT_CONFIG.get("email_enabled"):
            # No mail server here: leave every run unclaimed rather than
            # spend its retries on sends that cannot happen.
            return {"sent": 0, "failed": 0, "skipped": 0, "waiting": 0}

        def send(subject: str, body: str, to: str) -> bool:
            return bool(send_email(subject, body, to_email=to))

    if base_url is None:
        from ai_assistant_config import BASE_URL

        base_url = BASE_URL.rstrip("/")
    now = time.time() if now is None else now
    counts = {"sent": 0, "failed": 0, "skipped": 0, "waiting": 0}
    with _get_pg_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        setting = cur.execute(
            "SELECT EXTRACT(EPOCH FROM enabled_since) AS since FROM receipt_email_settings"
        ).fetchone()
        if setting is None:
            return counts
        for job_id in _candidates(cur, float(setting["since"]), limit):
            job = get_job(job_id)
            if not job:
                continue
            owner = str(job.get("owner") or job.get("owner_id") or "")
            if str(job.get("job_type") or "") == "serverless_worker" or job.get("hpc_cluster_id"):
                _record(cur, job_id, owner, "skipped", "not an instance")
                counts["skipped"] += 1
                conn.commit()
                continue
            receipt = build_receipt(cur, job)
            ended = receipt.get("ended_at") or 0
            if not receipt["final"] or (
                not receipt["reconciled"] and now - float(ended) < SETTLE_GRACE_SECONDS
            ):
                counts["waiting"] += 1
                continue
            account = _account_for_wallet(cur, owner)
            if account is None or not account.get("email"):
                _record(cur, job_id, owner, "skipped", "no account for this wallet")
                counts["skipped"] += 1
                conn.commit()
                continue
            if not receipts_wanted(account):
                _record(cur, job_id, owner, "skipped", "receipt emails turned off")
                counts["skipped"] += 1
                conn.commit()
                continue
            if not _claim(cur, job_id, owner):
                conn.commit()
                continue
            conn.commit()  # the claim is visible before the email goes out
            subject, body = render_receipt_email(receipt, base_url)
            try:
                ok = send(subject, body, account["email"])
            except Exception as exc:  # a mail outage must not stop the sweep
                log.warning("receipt email for %s failed: %s", job_id, exc)
                ok = False
            _record(cur, job_id, owner, "sent" if ok else "failed", "" if ok else "send failed")
            conn.commit()
            counts["sent" if ok else "failed"] += 1
    if counts["sent"] or counts["failed"]:
        log.info("receipt emails: %s", counts)
    return counts


def receipt_email_status(cur, job_id: str) -> dict[str, Any] | None:
    """What happened to one run's receipt email, for support and tests."""
    return cur.execute(
        "SELECT status, detail, attempts FROM receipt_emails WHERE job_id = %s", (job_id,)
    ).fetchone()
