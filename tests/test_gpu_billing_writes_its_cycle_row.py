"""Billing a GPU period must leave a `billing_cycles` row, or the period re-bills.

`BillingEngine._bill_gpu_period` debits the wallet and then inserts the cycle
row. Migration `097` dropped `billing_cycles.amount_cad`, and the INSERT still
named it, so every call raised `UndefinedColumn` after the debit had already
committed — `charge()` runs on its own connection via `self._conn()`, so it does
not share the transaction the INSERT aborts.

The consequence is worse than a missing audit row. Both callers derive the next
period's start from the last cycle:

    SELECT period_end FROM billing_cycles WHERE job_id = %s ORDER BY period_end DESC LIMIT 1
    period_start = last["period_end"] if last else float(job["started_at"])

With no row ever written, `last` is always `None`, so `period_start` falls back
to `started_at` on every pass. `auto_billing_cycle` runs every five minutes and
charges from the job's *start* each time — a job in its fourth hour is billed
four hours again, and again five minutes later.

`auto_billing_cycle`'s per-job handler is `except Exception as e: errors += 1`,
so this surfaced as a log line and a counter, not a failure. The existing
harness in `test_billing_periodic_harness.py` calls `auto_billing_cycle()` and
passed throughout, because it asserts the wallet debit — which happens on the
other connection — and never looks for the row.

So this asserts the row, and then asserts the thing the row exists to prevent.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")

from api import app
from billing import get_billing_engine
from db import _get_pg_pool

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _funded_owner() -> str:
    email = f"gpucycle-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Cycle"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    user = reg.json().get("user") or login.json().get("user") or {}
    owner = str(user["customer_id"])
    get_billing_engine().deposit(owner, 500.0, description="gpu-cycle-test")
    return owner


@pytest.fixture
def running_job():
    """A running GPU job that started two hours ago.

    No `hosts` row: `_bill_gpu_period` reads only `hosts.payload` and treats a
    missing row as an empty payload, so the rate resolves from the job's own
    `gpu_model` ($0.20/h for an RTX 4090 on-demand). Inserting one would mean
    satisfying a dozen NOT NULL columns this test does not exercise.
    """
    owner = _funded_owner()
    job_id = f"j-cyc-{uuid.uuid4().hex[:8]}"
    host_id = f"h-cyc-{uuid.uuid4().hex[:8]}"
    started_at = time.time() - 7200.0
    payload = {
        "job_id": job_id,
        "owner": owner,
        "started_at": started_at,
        "status": "running",
        "gpu_model": "RTX 4090",
        "host_gpu_model": "RTX 4090",
        "num_gpus": 1,
        "pricing_mode": "on_demand",
        "tier": "free",
    }
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'running', 0, %s, %s, %s)""",
            (job_id, started_at, host_id, json.dumps(payload)),
        )
        conn.commit()
    yield job_id, owner, started_at
    with pool.connection() as conn:
        conn.execute("DELETE FROM billing_cycles WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.commit()


def _cycles(job_id: str) -> list[dict]:
    from psycopg.rows import dict_row

    with _get_pg_pool().connection() as conn:
        conn.row_factory = dict_row
        return conn.execute(
            """SELECT period_start, period_end, amount_micros, duration_seconds
               FROM billing_cycles WHERE job_id = %s ORDER BY period_end""",
            (job_id,),
        ).fetchall()


def test_billing_a_gpu_period_writes_a_cycle_row(running_job):
    job_id, owner, started_at = running_job
    billing = get_billing_engine()

    result = billing.bill_running_period(job_id, period_end=started_at + 3600.0)

    assert result.get("billed") is True, result
    rows = _cycles(job_id)
    assert len(rows) == 1, f"the debit happened but no cycle row was written: {rows}"
    assert rows[0]["amount_micros"] > 0
    assert rows[0]["period_end"] == pytest.approx(started_at + 3600.0, abs=1.0)


def test_the_next_period_starts_where_the_last_one_ended(running_job):
    """The second pass bills the gap, not the whole job from its start again."""
    job_id, owner, started_at = running_job
    billing = get_billing_engine()

    billing.bill_running_period(job_id, period_end=started_at + 3600.0)

    balance_before = float(billing.get_wallet(owner)["balance_cad"])
    second = billing.bill_running_period(job_id, period_end=started_at + 3900.0)
    debited = balance_before - float(billing.get_wallet(owner)["balance_cad"])

    assert second.get("billed") is True, second
    rows = _cycles(job_id)
    assert len(rows) == 2
    assert rows[1]["period_start"] == pytest.approx(started_at + 3600.0, abs=1.0)
    assert rows[1]["duration_seconds"] == pytest.approx(300.0, abs=1.0)

    # 5 minutes, not the 65 that re-billing from started_at would charge.
    hourly = rows[0]["amount_micros"] / 1_000_000.0
    assert debited == pytest.approx(hourly * (300.0 / 3600.0), rel=0.05), (
        f"second pass debited {debited} — an hour's charge is {hourly}, so this "
        "re-billed from the job's start instead of from the last period_end"
    )
