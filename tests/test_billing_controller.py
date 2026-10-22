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
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_type='attempt' AND resource_id=%s",
                (aid,),
            )
            conn.execute("DELETE FROM usage_meters WHERE attempt_id=%s", (aid,))
            conn.execute("DELETE FROM job_attempts WHERE attempt_id=%s", (aid,))
        for jid in made["jobs"]:
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in made["hosts"]:
            conn.execute(
                "DELETE FROM observed_workloads WHERE observation_id IN "
                "(SELECT observation_id FROM host_observations WHERE host_id=%s)",
                (hid,),
            )
            conn.execute("DELETE FROM host_observations WHERE host_id=%s", (hid,))
            conn.execute("DELETE FROM reconciliation_queue WHERE resource_id=%s", (hid,))
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_type='host' AND resource_id=%s",
                (hid,),
            )
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(scratch) -> str:
    import json

    host_id = f"host-bc-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state, administrative_state, availability_state) VALUES (%s,'active',%s,%s, 'admitted', 'admitted', 'ready')""",
            (
                host_id,
                time.time(),
                json.dumps(
                    {
                        "gpu_model": "RTX 4090",
                        "country": "CA",
                        "compute_score": 8.0,
                        "cost_per_hour": 0.2,
                    }
                ),
            ),
        )
        conn.commit()
    scratch["hosts"].append(host_id)
    return host_id


def _mk_job(scratch, owner: str = "cust-bc") -> str:
    import json

    job_id = f"job-bc-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state, effective_priority, queued_at)
               VALUES (%s, 'completed', 0, %s, %s, 'succeeded', 'running', 0, NULL)""",
            (
                job_id,
                time.time(),
                json.dumps(
                    {
                        "owner": owner,
                        "vram_needed_gb": 8,
                        "gpu_model": "RTX 4090",
                        "pricing_mode": "on_demand",
                    }
                ),
            ),
        )
        from tests.native_jobs import admit_fixture_job

        admit_fixture_job(conn, job_id)
        conn.commit()
    scratch["jobs"].append(job_id)
    return job_id


def _mk_attempt(scratch, job_id: str, host_id: str, *, status="succeeded", ran=True) -> str:
    with _pool.connection() as conn:
        from control_plane.attempt_billing import capture_billing_snapshot
        from psycopg.types.json import Jsonb

        spec, digest = conn.execute(
            "SELECT spec,spec_hash FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()
        host = conn.execute("SELECT payload FROM hosts WHERE host_id=%s", (host_id,)).fetchone()[0]
        quote = capture_billing_snapshot(spec, None, {**host, "host_id": host_id})
        row = conn.execute(
            """INSERT INTO job_attempts
                   (job_id, attempt_number, status, host_id, fencing_token, job_generation,
                    reserved_at, spec_hash,placement_explanation)
               VALUES (%s, 1, %s, %s, nextval('placement_fencing_token_seq'), 1,
                       clock_timestamp(), %s,%s)
               RETURNING attempt_id""",
            (job_id, status, host_id, digest, Jsonb(dict(billing_snapshot=quote))),
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
        attempt,
        FINDING_MISSING_METER,
    )
    assert n_find == 1
    # Report-only: no meter was created.
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 0
    # Idempotent: a second sweep does not duplicate the finding.
    _run()
    assert (
        _count(
            "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
            attempt,
            FINDING_MISSING_METER,
        )
        == 1
    )


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
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", raising=False)
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
    assert (
        _count(
            "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
            attempt,
            FINDING_ORPHANED_METER,
        )
        == 1
    )
    # The orphaned meter is surfaced, never auto-mutated (still open).
    assert (
        _count(
            "SELECT count(*) FROM usage_meters WHERE attempt_id=%s AND completed_at IS NULL",
            attempt,
        )
        == 1
    )


def test_enforce_auto_closes_orphaned_open_meter(scratch, monkeypatch):
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", "enforce")
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    meter_id = f"mtr-{uuid.uuid4().hex[:12]}"
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO usage_meters
                   (meter_id, job_id, host_id, attempt_id, owner, started_at, completed_at,
                    pricing_mode, base_rate_per_hour, tier_multiplier, spot_discount)
               VALUES (%s, %s, %s, %s, 'cust-bc', %s, NULL, 'on_demand', 1.5, 1.0, 0.0)""",
            (meter_id, job, host, attempt, now - 1800),
        )
        conn.commit()

    # Pre-open a finding in report-only first
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", raising=False)
    _run()
    assert (
        _count(
            "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
            attempt,
            FINDING_ORPHANED_METER,
        )
        == 1
    )

    # Now enforce
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", "enforce")
    _run()

    # Meter is closed
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT completed_at, duration_sec, gpu_seconds, total_cost_micros FROM usage_meters WHERE meter_id=%s",
            (meter_id,),
        ).fetchone()
        assert row is not None
        completed_at, duration_sec, gpu_seconds, total_cost_micros = row
        assert completed_at is not None
        assert duration_sec > 0
        assert gpu_seconds > 0
        assert total_cost_micros > 0

    # No open meters remain
    assert (
        _count(
            "SELECT count(*) FROM usage_meters WHERE attempt_id=%s AND completed_at IS NULL",
            attempt,
        )
        == 0
    )

    # Finding was resolved
    assert (
        _count(
            "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s AND resolved_at IS NULL",
            attempt,
            FINDING_ORPHANED_METER,
        )
        == 0
    )

    # Idempotent re-run
    _run()
    assert (
        _count(
            "SELECT count(*) FROM usage_meters WHERE attempt_id=%s AND completed_at IS NULL",
            attempt,
        )
        == 0
    )


def test_duplicate_delivery_creates_no_second_charge(scratch):
    """meter_job is idempotent per attempt — a replayed close never double-charges."""
    from billing import get_billing_engine

    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    job_dict = {
        "job_id": job,
        "attempt_id": attempt,
        "owner": "cust-bc",
        "started_at": time.time() - 3600,
        "completed_at": time.time(),
        "vram_needed_gb": 8,
        "pricing_mode": "on_demand",
        "gpu_model": "RTX 4090",
    }
    host_dict = {"host_id": host, "gpu_model": "RTX 4090", "country": "CA", "compute_score": 8.0}
    eng = get_billing_engine()
    eng.meter_job(job_dict, host_dict)
    eng.meter_job(job_dict, host_dict)  # duplicate delivery
    assert _count("SELECT count(*) FROM usage_meters WHERE attempt_id=%s", attempt) == 1


def _all_findings(attempt: str, finding_type: str) -> int:
    return _count(
        "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s",
        attempt,
        finding_type,
    )


def _open_findings(attempt: str, finding_type: str) -> int:
    return _count(
        "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s AND finding_type=%s "
        "AND resolved_at IS NULL",
        attempt,
        finding_type,
    )


def test_host_pass_does_not_resolve_billing_findings(scratch, monkeypatch):
    """A host reconcile pass must not clear findings it does not evaluate.

    Regression: `reconcile_host` resolved every open finding on the host's
    attempts, so each billing sweep reopened `billing_missing_meter` and the
    critical-findings alert flapped (1,619 findings for two attempts in 3 days).
    """
    from control_plane.db import run_transaction
    from control_plane.observations import ingest_observation
    from control_plane.reconcile import reconcile_host

    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", raising=False)
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    _run()
    assert _open_findings(attempt, FINDING_MISSING_METER) == 1

    run_transaction(
        lambda c: ingest_observation(
            c,
            host_id=host,
            session_id=f"sess-{uuid.uuid4().hex[:8]}",
            observation_generation=1,
            workloads=[],
            agent_version="test",
            worker_reported_at=time.time(),
        ),
        what="test_ingest",
    )
    with _pool.connection() as conn:
        reconcile_host(conn, host)
        conn.commit()

    assert _open_findings(attempt, FINDING_MISSING_METER) == 1
    _run()
    assert _all_findings(attempt, FINDING_MISSING_METER) == 1


def test_missing_meter_finding_resolves_once_metered(scratch, monkeypatch):
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", raising=False)
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    _run()
    assert _open_findings(attempt, FINDING_MISSING_METER) == 1

    # A normal completion writer meters the attempt between sweeps.
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO usage_meters (meter_id, job_id, attempt_id, owner, started_at, completed_at, pricing_mode) "
            "VALUES (%s, %s, %s, 'cust-bc', %s, %s, 'on_demand')",
            (f"mtr-{uuid.uuid4().hex[:12]}", job, attempt, time.time() - 3600, time.time()),
        )
        conn.commit()
    _run()
    assert _open_findings(attempt, FINDING_MISSING_METER) == 0
    assert _all_findings(attempt, FINDING_MISSING_METER) == 1


def test_orphaned_meter_finding_resolves_once_closed(scratch, monkeypatch):
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", raising=False)
    host = _mk_host(scratch)
    job = _mk_job(scratch)
    attempt = _mk_attempt(scratch, job, host, status="succeeded")
    meter_id = f"mtr-{uuid.uuid4().hex[:12]}"
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO usage_meters (meter_id, job_id, attempt_id, owner, started_at, completed_at, pricing_mode) "
            "VALUES (%s, %s, %s, 'cust-bc', %s, NULL, 'on_demand')",
            (meter_id, job, attempt, time.time()),
        )
        conn.commit()
    _run()
    assert _open_findings(attempt, FINDING_ORPHANED_METER) == 1

    with _pool.connection() as conn:
        conn.execute(
            "UPDATE usage_meters SET completed_at=%s WHERE meter_id=%s", (time.time(), meter_id)
        )
        conn.commit()
    _run()
    assert _open_findings(attempt, FINDING_ORPHANED_METER) == 0
