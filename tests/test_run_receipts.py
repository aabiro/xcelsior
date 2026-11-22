"""Receipts for finished runs (Phase 1, 2026-11-22).

A receipt is read from the ledger that billed the run: the closed meter gives
the seconds and the amount owed, the charged billing cycles and the wallet
debits give what was paid, and all three must agree to the micro. Only the
owner sees it; anyone else gets not-found, as for a job that does not exist.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import run_receipts  # noqa: E402
from run_receipts import (  # noqa: E402
    HOURLY_ROUNDED_LABEL,
    decode_cursor,
    encode_cursor,
    hourly_rounded_micros,
)

# ── Pure parts ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds, hours",
    [(0, 0), (0.01, 1), (90, 1), (3600, 1), (3600.01, 2), (7322, 3)],
)
def test_the_whole_hours_figure_rounds_each_started_hour_up(seconds, hours):
    assert hourly_rounded_micros(300_000, seconds) == hours * 300_000


def test_a_quoted_run_is_rated_from_its_quote_and_a_legacy_run_from_its_meter():
    assert run_receipts._rate_micros({"pricing_snapshot": {"total_rate_micros": 1_234_567}}) == (
        1_234_567,
        "exact",
    )
    assert run_receipts._rate_micros(
        {"pricing_snapshot": None, "base_rate_per_hour": 0.5, "spot_discount": 0.2}
    ) == (400_000, "legacy_4dp")
    assert run_receipts._rate_micros({"base_rate_per_hour": None}) == (None, "legacy_4dp")


def test_cursors_round_trip_and_garbage_is_refused():
    assert decode_cursor(encode_cursor(1700000000.5, "job-1")) == (1700000000.5, "job-1")
    for bad in ("", "not-base64!", "zz", encode_cursor(float("nan"), "x")):
        with pytest.raises(ValueError):
            decode_cursor(bad)


# ── A real run, billed by the real billing loop ──────────────────────


def _client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def _register(client, prefix: str) -> tuple[dict, str]:
    email = f"{prefix}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Receipt Test"},
    )
    assert reg.status_code in (200, 201), reg.text
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}, reg.json()["user"][
        "customer_id"
    ]


@pytest.fixture
def billed_run():
    """A 90-second native run at 3.60/hr, finished and billed, owned by a real user."""
    from billing import get_billing_engine
    from db import _get_pg_pool
    from tests.fixtures.native_runtime import running_instance
    from tests.test_data_layer_readiness import finish_compute

    client = _client()
    owner_headers, customer_id = _register(client, "receipt-owner")
    engine = get_billing_engine()
    engine.deposit(customer_id, 10.0, description="receipt test")
    try:
        with running_instance(owner=customer_id, cost_per_hour=3.6, age_sec=90) as native:
            finish_compute(native)
            engine.auto_billing_cycle()
            yield client, owner_headers, customer_id, native
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM billing_cycles WHERE customer_id=%s", (customer_id,))
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (customer_id,))


def test_a_finished_run_reconciles_to_the_micro(billed_run):
    from db import _get_pg_pool

    client, headers, customer_id, native = billed_run
    r = client.get(f"/api/instances/{native['job_id']}/receipt", headers=headers)
    assert r.status_code == 200, r.text
    receipt = r.json()["receipt"]

    with _get_pg_pool().connection() as conn:
        meter = conn.execute(
            "SELECT duration_sec, total_cost_micros FROM usage_meters WHERE job_id=%s",
            (native["job_id"],),
        ).fetchone()
        debits = conn.execute(
            "SELECT -COALESCE(SUM(amount_micros),0) FROM wallet_transactions "
            "WHERE customer_id=%s AND job_id=%s AND tx_type='charge'",
            (customer_id, native["job_id"]),
        ).fetchone()[0]
    assert receipt["billed_seconds"] == pytest.approx(float(meter[0]))
    assert 89 <= receipt["billed_seconds"] <= 95
    assert receipt["total_micros"] == meter[1] > 0
    assert receipt["wallet_debit_micros"] == debits == receipt["total_micros"]
    assert receipt["cycles_charged_micros"] == receipt["total_micros"]
    assert receipt["reconciled"] is True and receipt["outstanding_micros"] == 0
    # 3.60/hr: 90 s owes 0.09; one started hour at the same rate is 3.60.
    assert receipt["rate_micros_per_hour"] == 3_600_000
    assert receipt["hourly_rounded_micros"] == 3_600_000
    assert receipt["hourly_rounded_label"] == HOURLY_ROUNDED_LABEL
    assert receipt["rounding"] == "legacy_4dp"
    assert receipt["final"] is True and receipt["currency"] == "CAD"
    assert receipt["host_verification_at_start"]["source"] == "placement"
    assert receipt["host_verification_at_start"]["state"] == "unverified"
    assert receipt["image"] == "python:3.11"
    [attempt] = receipt["attempts"]
    assert attempt["finished"] is True and attempt["total_micros"] == receipt["total_micros"]


def test_the_instance_shows_the_billed_amount_not_an_estimate(billed_run):
    client, headers, _, native = billed_run
    receipt = client.get(f"/api/instances/{native['job_id']}/receipt", headers=headers).json()
    instance = client.get(f"/instance/{native['job_id']}", headers=headers).json()["instance"]
    assert instance["cost_is_estimate"] is False
    assert instance["cost_cad"] == receipt["receipt"]["total_micros"] / 1_000_000


def test_only_the_owner_sees_a_receipt(billed_run):
    client, headers, _, native = billed_run
    intruder, _ = _register(client, "receipt-intruder")
    job_id = native["job_id"]
    for path in (f"/api/instances/{job_id}/receipt", f"/api/v1/instances/{job_id}/receipt"):
        assert client.get(path, headers=headers).status_code == 200, path
        denied = client.get(path, headers=intruder)
        assert denied.status_code == 404, (path, denied.text)
        # The same answer as for a job that does not exist.
        missing = client.get(path.replace(job_id, f"abcd{uuid.uuid4().hex[:20]}"), headers=intruder)
        assert missing.status_code == 404
    for path in ("/api/billing/receipts", "/api/v1/billing/receipts"):
        theirs = client.get(path, headers=intruder).json()
        assert all(r["job_id"] != job_id for r in theirs["receipts"]), path
        mine = client.get(path, headers=headers).json()
        assert [r["job_id"] for r in mine["receipts"]] == [job_id], path


def test_the_list_pages_with_a_cursor(billed_run):
    client, headers, _, native = billed_run
    page = client.get("/api/billing/receipts", params={"limit": 1}, headers=headers).json()
    assert page["next_cursor"] is None and len(page["receipts"]) == 1
    bad = client.get("/api/billing/receipts", params={"cursor": "zz"}, headers=headers)
    assert bad.status_code == 400
    bad_v1 = client.get("/api/v1/billing/receipts", params={"cursor": "zz"}, headers=headers)
    assert bad_v1.status_code == 422 and bad_v1.json()["code"] == "invalid_cursor"


def test_a_run_in_progress_has_no_receipt_yet():
    from billing import get_billing_engine
    from tests.fixtures.native_runtime import running_instance

    client = _client()
    headers, customer_id = _register(client, "receipt-running")
    get_billing_engine().deposit(customer_id, 10.0, description="receipt test")
    with running_instance(owner=customer_id, cost_per_hour=3.6, age_sec=60) as native:
        r = client.get(f"/api/instances/{native['job_id']}/receipt", headers=headers)
        assert r.status_code == 409, r.text
        v1 = client.get(f"/api/v1/instances/{native['job_id']}/receipt", headers=headers)
        assert v1.status_code == 409 and v1.json()["code"] == "no_finished_run"
        # The live figure is labelled an estimate and uses no invented rate.
        instance = client.get(f"/instance/{native['job_id']}", headers=headers).json()["instance"]
        assert instance["cost_is_estimate"] is True


def test_no_rate_is_invented_for_a_cost():
    """The $0.20/hr fallback priced runs nobody had billed."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "api/routes/instances.py").read_text()
    assert "or 0.20" not in source


@pytest.mark.needs_db
def test_a_quoted_run_is_billed_and_reported_exactly(paid_fleet):
    """Every new customer instance carries a service-tier quote: no 4-place rounding."""
    from psycopg.rows import dict_row

    from billing import get_billing_engine
    from control_plane.attempts import report_attempt_status
    from control_plane.db import run_transaction
    from db import _get_pg_pool
    from scheduler import get_job
    from tests.test_paid_service_tiers import claim, reserve, start, submit

    job = submit(paid_fleet, "premium")
    reservation = reserve(paid_fleet, claim(paid_fleet))
    _, args = start(paid_fleet, reservation)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-make_interval(secs=>61.25) "
            "WHERE attempt_id=%s",
            (reservation.attempt_id,),
        )
        conn.execute(
            "UPDATE usage_meters SET started_at=(SELECT EXTRACT(EPOCH FROM started_at) "
            "FROM job_attempts WHERE attempt_id=%s) WHERE attempt_id=%s",
            (reservation.attempt_id, str(reservation.attempt_id)),
        )
    run_transaction(lambda conn: report_attempt_status(conn, status="succeeded", **args))
    get_billing_engine().auto_billing_cycle()

    with _get_pg_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        receipt = run_receipts.build_receipt(cur, get_job(job["job_id"]))
        meter = cur.execute(
            "SELECT total_cost_micros FROM usage_meters WHERE attempt_id=%s",
            (str(reservation.attempt_id),),
        ).fetchone()
    assert receipt["rounding"] == "exact"
    assert receipt["rate_micros_per_hour"] == 784_000
    assert receipt["total_micros"] == meter["total_cost_micros"] > 0
    assert receipt["reconciled"] is True, receipt
    assert receipt["wallet_debit_micros"] == receipt["total_micros"]
    assert receipt["hourly_rounded_micros"] == 784_000
    assert 61 <= receipt["billed_seconds"] <= 70


from tests.test_paid_service_tiers import paid_fleet  # noqa: E402,F401  (fixture)
