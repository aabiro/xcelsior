"""Attempt-scoped usage meters — stamp + idempotent close.

Drives shipped ``BillingEngine.meter_job`` / ``resolve_meter_attempt_id``
against real Postgres:

- attempt-owned jobs stamp ``attempt_id`` on the meter row
- re-close of the same attempt does not create a second billable row
- pure-legacy jobs still meter without attempt_id
- structural inventory: production INSERT goes through meter_job
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_attempt_col = (
            _c.execute(
                """
                SELECT 1 FROM information_schema.columns
                 WHERE table_name='usage_meters' AND column_name='attempt_id'
                """
            ).fetchone()
            is not None
        )
        _has_job_attempts = (
            _c.execute("SELECT to_regclass('public.job_attempts')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has_attempt_col:  # pragma: no cover
        pytestmark = pytest.mark.skip(
            "usage_meters.attempt_id missing — run alembic upgrade head"
        )


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": [], "meters": [], "attempts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for mid in ids["meters"]:
            conn.execute("DELETE FROM usage_meters WHERE meter_id=%s", (mid,))
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(cleanup, host_id: str) -> str:
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"host_id": host_id, "gpu_model": "A100"})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    return host_id


def _mk_legacy_job(cleanup, *, host_id: str, owner: str) -> str:
    job_id = f"j-mtr-leg-{uuid.uuid4().hex[:8]}"
    now = time.time()
    payload = {
        "job_id": job_id,
        "owner": owner,
        "started_at": now - 3600,
        "completed_at": now,
        "status": "completed",
        "vram_needed_gb": 8,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'completed', 0, %s, %s, %s)""",
            (job_id, now - 3600, host_id, json.dumps(payload)),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_attempt_owned_job(cleanup, *, host_id: str, owner: str) -> tuple[str, str]:
    job_id = f"j-mtr-att-{uuid.uuid4().hex[:8]}"
    attempt_id = str(uuid.uuid4())
    now = time.time()
    payload = {
        "job_id": job_id,
        "owner": owner,
        "started_at": now - 1800,
        "completed_at": now,
        "status": "completed",
        "vram_needed_gb": 16,
    }
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload,
                    active_attempt_id)
               VALUES (%s, 'completed', 0, %s, %s, %s, NULL)""",
            (job_id, now - 1800, host_id, json.dumps(payload)),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'succeeded', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        # Leave active_attempt_id NULL (post-settle) — resolver must use history.
        conn.commit()
    cleanup["jobs"].append(job_id)
    cleanup["attempts"].append(attempt_id)
    return job_id, attempt_id


def _meters_for_job(job_id: str) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT meter_id, job_id, attempt_id, total_cost_cad
                 FROM usage_meters WHERE job_id=%s ORDER BY created_at""",
            (job_id,),
        ).fetchall()
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append(
                {
                    "meter_id": r[0],
                    "job_id": r[1],
                    "attempt_id": r[2],
                    "total_cost_cad": r[3],
                }
            )
    return out


def test_resolve_meter_attempt_id_policy():
    from billing import resolve_meter_attempt_id

    assert resolve_meter_attempt_id({}) is None
    assert (
        resolve_meter_attempt_id({"attempt_id": " a1 "}) == "a1"
    )
    assert (
        resolve_meter_attempt_id({"active_attempt_id": "a2"}) == "a2"
    )


def test_legacy_meter_job_inserts_without_attempt(cleanup):
    from billing import BillingEngine

    host_id = f"h-mtr-l-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"own-l-{uuid.uuid4().hex[:6]}@test"
    job_id = _mk_legacy_job(cleanup, host_id=host_id, owner=owner)
    now = time.time()
    job = {
        "job_id": job_id,
        "owner": owner,
        "started_at": now - 3600,
        "completed_at": now,
        "vram_needed_gb": 8,
    }
    host = {
        "host_id": host_id,
        "gpu_model": "RTX 4090",
        "cost_per_hour": 0.5,
        "country": "CA",
    }

    eng = BillingEngine()
    meter = eng.meter_job(job, host)
    cleanup["meters"].append(meter.meter_id)

    assert meter.attempt_id is None
    rows = _meters_for_job(job_id)
    assert len(rows) == 1
    assert rows[0]["attempt_id"] is None
    assert float(rows[0]["total_cost_cad"]) > 0


def test_attempt_owned_meter_stamps_attempt_id(cleanup):
    from billing import BillingEngine

    host_id = f"h-mtr-a-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"own-a-{uuid.uuid4().hex[:6]}@test"
    job_id, attempt_id = _mk_attempt_owned_job(
        cleanup, host_id=host_id, owner=owner
    )
    now = time.time()
    job = {
        "job_id": job_id,
        "owner": owner,
        "started_at": now - 1800,
        "completed_at": now,
        "vram_needed_gb": 16,
    }
    host = {
        "host_id": host_id,
        "gpu_model": "A100",
        "cost_per_hour": 1.0,
        "country": "CA",
    }

    eng = BillingEngine()
    meter = eng.meter_job(job, host)
    cleanup["meters"].append(meter.meter_id)

    assert meter.attempt_id == attempt_id
    rows = _meters_for_job(job_id)
    assert len(rows) == 1
    assert str(rows[0]["attempt_id"]) == attempt_id


def test_attempt_owned_meter_close_is_idempotent(cleanup):
    """Second meter_job for the same attempt must not create another row."""
    from billing import BillingEngine

    host_id = f"h-mtr-i-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"own-i-{uuid.uuid4().hex[:6]}@test"
    job_id, attempt_id = _mk_attempt_owned_job(
        cleanup, host_id=host_id, owner=owner
    )
    now = time.time()
    job = {
        "job_id": job_id,
        "owner": owner,
        "started_at": now - 900,
        "completed_at": now,
        "vram_needed_gb": 8,
        "attempt_id": attempt_id,  # explicit authority
    }
    host = {
        "host_id": host_id,
        "gpu_model": "A100",
        "cost_per_hour": 2.0,
        "country": "CA",
    }

    eng = BillingEngine()
    m1 = eng.meter_job(job, host)
    m2 = eng.meter_job(job, host)
    cleanup["meters"].append(m1.meter_id)

    assert m1.meter_id == m2.meter_id
    assert m1.attempt_id == attempt_id
    assert m2.attempt_id == attempt_id
    assert m1.total_cost_cad == m2.total_cost_cad

    rows = _meters_for_job(job_id)
    assert len(rows) == 1, f"double charge: {rows}"
    assert float(rows[0]["total_cost_cad"]) == pytest.approx(m1.total_cost_cad)


def test_active_attempt_on_job_row_preferred(cleanup):
    from billing import BillingEngine, resolve_meter_attempt_id

    host_id = f"h-mtr-act-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"own-act-{uuid.uuid4().hex[:6]}@test"
    job_id = f"j-mtr-act-{uuid.uuid4().hex[:8]}"
    attempt_id = str(uuid.uuid4())
    now = time.time()
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload,
                    active_attempt_id)
               VALUES (%s, 'running', 0, %s, %s, %s, NULL)""",
            (
                job_id,
                now,
                host_id,
                json.dumps({"owner": owner, "started_at": now - 100}),
            ),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'running', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)

    assert resolve_meter_attempt_id({"job_id": job_id}) == attempt_id

    eng = BillingEngine()
    meter = eng.meter_job(
        {
            "job_id": job_id,
            "owner": owner,
            "started_at": now - 100,
            "completed_at": now,
        },
        {"host_id": host_id, "gpu_model": "A100", "cost_per_hour": 1.0},
    )
    cleanup["meters"].append(meter.meter_id)
    assert meter.attempt_id == attempt_id


def test_production_meter_insert_inventory():
    """Static: only BillingEngine.meter_job inserts usage_meters rows."""
    root = Path(__file__).resolve().parents[1]
    billing = (root / "billing.py").read_text()
    assert "def meter_job(" in billing
    assert "INSERT INTO usage_meters" in billing
    assert "resolve_meter_attempt_id" in billing
    assert "attempt_id" in billing
    # Single production INSERT site in billing.py
    assert billing.count("INSERT INTO usage_meters") == 1

    # Callers go through meter_job, not raw SQL
    sched = (root / "scheduler.py").read_text()
    assert "meter_job" in sched
    assert "INSERT INTO usage_meters" not in sched

    mig = (root / "migrations" / "versions" / "062_usage_meters_attempt_id.py").read_text()
    assert "uq_usage_meters_one_per_attempt" in mig
    assert "ADD COLUMN IF NOT EXISTS attempt_id" in mig
