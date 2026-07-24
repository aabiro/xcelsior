"""Track B B3.3 — billing controller (§12.4) meter-invariant reconciler.

Proves the controller surfaces (and, when enabled, converges) the meter
invariants:
  * a terminal attempt that ran with no meter is a billing leak → one finding
    (report-only default), and enforce converges to **exactly one** meter;
  * an open meter on a terminal attempt is surfaced (report-only);
  * re-metering an attempt (a duplicate delivery) creates **no second charge**.

Real PostgreSQL; owns only its own rows.
"""

from __future__ import annotations

import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('usage_meters')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("usage_meters missing — upgrade head")

from control_plane.billing_controller import (
    FINDING_MISSING_METER,
    FINDING_ORPHANED_METER,
    reconcile_billing_meters,
)


@pytest.fixture
def scratch():
    made = {"jobs": [], "attempts": [], "hosts": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for aid in made["attempts"]:
            conn.execute("DELETE FROM reconciliation_findings WHERE resource_type='attempt' AND resource_id=%s", (aid,))
            conn.execute("DELETE FROM usage_meters WHERE attempt_id=%s", (aid,))
            conn.execute("DELETE FROM job_attempts WHERE attempt_id=%s", (aid,))
        for jid in made["jobs"]:
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in made["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(scratch) -> str:
    import json

    host_id = f"host-bc-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) VALUES (%s,'active',%s,%s)",
            (host_id, time.time(), json.dumps({"gpu_model": "RTX 4090", "country": "CA", "compute_score": 8.0})),
        )
        conn.commit()
    scratch["hosts"].append(host_id)
    return host_id


def _mk_job(scratch, owner: str = "cust-bc") -> str:
    import json

    job_id = f"job-bc-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state)
               VALUES (%s, 'completed', 0, %s, %s, 'terminal', 'stopped')""",
            (job_id, time.time(), json.dumps({"owner": owner, "vram_needed_gb": 8, "gpu_model": "RTX 4090", "pricing_mode": "on_demand"})),
        )
        conn.commit()
    scratch["jobs"].append(job_id)
    return job_id


def _mk_attempt(scratch, job_id: str, host_id: str, *, status="succeeded", ran=True) -> str:
    with _pool.connection() as conn:
        row = conn.execute(
            """INSERT INTO job_attempts
                   (job_id, attempt_number, status, host_id, fencing_token, job_generation,
                    reserved_at, spec_hash)
               VALUES (%s, 1, %s, %s, nextval('placement_fencing_token_seq'), 1,
                       clock_timestamp(), 'sh')
               RETURNING attempt_id""",
            (job_id, status, host_id),
        ).fetchone()
        attempt_id = str(row[0])
        if ran:
            conn.execute(
                "UPDATE job_attempts SET started_at = clock_timestamp() - interval '1 hour', "
                "ended_at = clock_timestamp() WHERE attempt_id = %s",
                (attempt_id,),
            )
        conn.commit()
    scratch["attempts"].append(attempt_id)
    return attempt_id


def _count(sql: str, *params) -> int:
    with _pool.connection() as conn:
        return conn.execute(sql, params).fetchone()[0]


def _run() -> None:
    with _pool.connection() as conn:
        reconcile_billing_meters(conn)
        conn.commit()


def test_missing_meter_opens_one_finding_report_only(scratch, monkeypatch):
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", raising=False)
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")

    _run()
    n_find = _count(
        "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
        attempt, FINDING_MISSING_METER,
    )
    assert n_find == 1
    # Report-only: no meter was created.
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 0
    # Idempotent: a second sweep does not duplicate the finding.
    _run()
    assert _count(
        "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
        attempt, FINDING_MISSING_METER,
    ) == 1


def test_enforce_converges_to_exactly_one_meter(scratch, monkeypatch):
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", "enforce")
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")

    _run()
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 1
    # Re-run: still exactly one meter (idempotent enforce — no double charge).
    _run()
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 1


def test_orphaned_open_meter_is_surfaced(scratch, monkeypatch):
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", raising=False)
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    # An open meter (completed_at NULL) left behind on a terminal attempt.
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO usage_meters (meter_id, job_id, attempt_id, owner, started_at, completed_at, pricing_mode) "
            "VALUES (%s, %s, %s, 'cust-bc', %s, NULL, 'on_demand')",
            (f"mtr-{uuid.uuid4().hex[:12]}", job, attempt, time.time()),
        )
        conn.commit()

    _run()
    assert _count(
        "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
        attempt, FINDING_ORPHANED_METER,
    ) == 1
    # The orphaned meter is surfaced, never auto-mutated (still open).
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s AND completed_at IS NULL", attempt) == 1


def test_duplicate_delivery_creates_no_second_charge(scratch):
    """meter_job is idempotent per attempt — a replayed close never double-charges."""
    from billing import get_billing_engine

    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    job_dict = {
        "job_id": job, "attempt_id": attempt, "owner": "cust-bc",
        "started_at": time.time() - 3600, "completed_at": time.time(),
        "vram_needed_gb": 8, "pricing_mode": "on_demand", "gpu_model": "RTX 4090",
    }
    host_dict = {"host_id": host, "gpu_model": "RTX 4090", "country": "CA", "compute_score": 8.0}
    eng = get_billing_engine()
    eng.meter_job(job_dict, host_dict)
    eng.meter_job(job_dict, host_dict)  # duplicate delivery
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 1
