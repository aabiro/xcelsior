"""A finished instance's receipt is emailed once, by default, unless turned off.

The email is rendered from the same receipt the page shows; a run that ended
before the feature started is never mailed; two sweeps cannot both email one
run; an opted-out account is recorded as skipped and not looked at again.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from receipt_emails import (  # noqa: E402
    receipts_wanted,
    render_receipt_email,
    send_pending_receipt_emails,
)
from tests.test_run_receipts import billed_run  # noqa: E402,F401  (fixture)


def test_on_by_default_and_off_when_either_setting_is_off():
    assert receipts_wanted({"notifications_enabled": 1, "preferences": {}})
    assert receipts_wanted({"notifications_enabled": 1, "preferences": '{"onboarding": 1}'})
    assert not receipts_wanted(
        {"notifications_enabled": 1, "preferences": {"receipt_emails": False}}
    )
    assert not receipts_wanted({"notifications_enabled": 0, "preferences": {}})


def test_the_email_says_what_the_receipt_says():
    receipt = {
        "job_id": "job-1",
        "name": "fine-tune",
        "gpu_model": "RTX 3060",
        "gpu_count": 1,
        "billed_seconds": 5432.0,
        "rate_micros_per_hour": 120_000,
        "total_micros": 181_067,
        "hourly_rounded_micros": 240_000,
        "hourly_rounded_label": "if billed in whole hours at the same rate",
        "wallet_debit_micros": 181_067,
        "outstanding_micros": 0,
        "credits_micros": 0,
    }
    subject, body = render_receipt_email(receipt, "https://xcelsior.example")
    assert subject == "Receipt: fine-tune on RTX 3060, $0.18"
    assert "1 h 30 min 32 s (5,432 seconds, billed by the second)" in body
    assert "$0.12/hr, locked when you launched" in body
    assert "In whole hours: $0.24" in body and "not what you were charged" in body
    assert "https://xcelsior.example/dashboard/instances/job-1" in body
    assert "https://xcelsior.example/dashboard/settings" in body
    assert "Still to be charged" not in body


def _status(job_id):
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        return conn.execute(
            "SELECT status, attempts FROM receipt_emails WHERE job_id=%s", (job_id,)
        ).fetchone()


def _cleanup(job_id):
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM receipt_emails WHERE job_id=%s", (job_id,))


@pytest.fixture
def started_before_the_run():
    """The feature started an hour ago, so the test run counts as new."""
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        old = conn.execute("SELECT enabled_since FROM receipt_email_settings").fetchone()[0]
        conn.execute("UPDATE receipt_email_settings SET enabled_since = now() - interval '1 hour'")
    yield
    with _get_pg_pool().connection() as conn:
        conn.execute("UPDATE receipt_email_settings SET enabled_since = %s", (old,))


@pytest.mark.needs_db
def test_a_finished_run_is_emailed_once(billed_run, started_before_the_run):
    client, headers, customer_id, native = billed_run
    job_id = native["job_id"]
    outbox = []

    def send(subject, body, to):
        outbox.append((subject, body, to))
        return True

    try:
        send_pending_receipt_emails(send=send, limit=1000)
        mine = [m for m in outbox if job_id in m[1]]
        assert len(mine) == 1
        receipt = client.get(f"/api/instances/{job_id}/receipt", headers=headers).json()["receipt"]
        assert render_receipt_email(receipt, "x")[0] == mine[0][0]
        assert tuple(_status(job_id)) == ("sent", 1)
        # A second sweep sends nothing more for this run.
        send_pending_receipt_emails(send=send, limit=1000)
        assert len([m for m in outbox if job_id in m[1]]) == 1
    finally:
        _cleanup(job_id)


@pytest.mark.needs_db
def test_an_opted_out_account_is_skipped(billed_run, started_before_the_run):
    client, headers, customer_id, native = billed_run
    job_id = native["job_id"]
    r = client.put("/api/users/me/preferences", headers=headers, json={"receipt_emails": False})
    assert r.status_code == 200, r.text
    assert (
        client.get("/api/users/me/preferences", headers=headers).json()["receipt_emails"] is False
    )
    outbox = []
    try:
        send_pending_receipt_emails(send=lambda s, b, t: outbox.append(b) or True, limit=1000)
        assert not [b for b in outbox if job_id in b]
        assert _status(job_id)[0] == "skipped"
    finally:
        _cleanup(job_id)


@pytest.mark.needs_db
def test_a_failed_send_is_retried_then_given_up(billed_run, started_before_the_run):
    _, _, _, native = billed_run
    job_id = native["job_id"]
    try:
        for _ in range(5):
            send_pending_receipt_emails(send=lambda s, b, t: False, limit=1000)
        assert tuple(_status(job_id)) == ("failed", 3)
    finally:
        _cleanup(job_id)


@pytest.mark.needs_db
def test_a_run_from_before_the_feature_is_never_emailed(billed_run):
    """enabled_since is when migration 168 ran; the fixture's run ended after it,
    so move the start past the run."""
    from db import _get_pg_pool

    _, _, _, native = billed_run
    job_id = native["job_id"]
    with _get_pg_pool().connection() as conn:
        old = conn.execute("SELECT enabled_since FROM receipt_email_settings").fetchone()[0]
        conn.execute("UPDATE receipt_email_settings SET enabled_since = now() + interval '1 hour'")
    outbox = []
    try:
        send_pending_receipt_emails(send=lambda s, b, t: outbox.append(b) or True, limit=1000)
        assert not [b for b in outbox if job_id in b]
        assert _status(job_id) is None
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("UPDATE receipt_email_settings SET enabled_since = %s", (old,))
        _cleanup(job_id)


def test_the_operator_switch_stops_the_sweep(monkeypatch):
    from receipt_emails import sweep_enabled

    monkeypatch.delenv("XCELSIOR_RECEIPT_EMAILS", raising=False)
    assert sweep_enabled()
    for off in ("0", "false", "OFF"):
        monkeypatch.setenv("XCELSIOR_RECEIPT_EMAILS", off)
        assert not sweep_enabled()
