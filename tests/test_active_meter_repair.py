"""Active meter repair and terminal settlement against real placed authority."""

import time

import pytest
from psycopg.types.json import Jsonb

from billing import BillingEngine
from control_plane.attempts import report_attempt_status
from control_plane.billing_controller import reconcile_billing_meters
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.test_job_reconcile_observations import placed, _observe  # noqa: F401

pytestmark = pytest.mark.needs_db
FLAG = "XCELSIOR_RECONCILE_ACTION_BILLING_ACTIVE_METER_MISSING"


@pytest.fixture(autouse=True)
def native_admission_fields(request):
    """Customer identity/resources/price are approved before placement."""
    return {"billable": True, **getattr(request, "param", {})}


@pytest.fixture
def running(placed):
    with placed["pool"].connection() as conn:
        # Reproduce a missing-meter incident without changing admission/quote.
        conn.execute("DELETE FROM usage_meters WHERE attempt_id=%s", (placed["attempt_id"],))
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-interval '1 hour' "
            "WHERE attempt_id=%s",
            (placed["attempt_id"],),
        )
    _observe(placed)
    return placed


def _run(p):
    with p["pool"].connection() as conn:
        return reconcile_billing_meters(conn)


def _meters(p):
    with p["pool"].connection() as conn:
        return conn.execute(
            "SELECT meter_id, completed_at, total_cost_micros, base_rate_per_hour "
            "FROM usage_meters WHERE attempt_id=%s",
            (p["attempt_id"],),
        ).fetchall()


def _finish(p, *, simulate_orphan=False, simulate_missing=False):
    with p["pool"].connection() as conn:
        report_attempt_status(
            conn,
            **{k: p[k] for k in ("job_id", "attempt_id", "host_id", "fencing_token")},
            status="succeeded",
        )
        if simulate_orphan:
            # Reproduce an orphan left by the older separate settlement path.
            # Normal fenced completion now settles in this same transaction.
            conn.execute(
                "UPDATE usage_meters SET completed_at=NULL,duration_sec=0,"
                "gpu_seconds=0,total_cost_micros=0 WHERE attempt_id=%s",
                (p["attempt_id"],),
            )
        if simulate_missing:
            conn.execute("DELETE FROM usage_meters WHERE attempt_id=%s", (p["attempt_id"],))


@pytest.mark.parametrize("policy", [None, "report_only", "typo"])
def test_active_meter_defaults_to_one_report_only_finding(running, monkeypatch, policy):
    if policy is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, policy)
    _run(running)
    _run(running)
    assert _meters(running) == []
    with running["pool"].connection() as conn:
        assert conn.execute(
            "SELECT action_taken, resolved_at FROM reconciliation_findings "
            "WHERE resource_id=%s AND finding_type='billing_active_meter_missing'",
            (running["attempt_id"],),
        ).fetchall() == [("report_only", None)]


def test_active_meter_enforce_is_atomic_idempotent_and_not_invoiced(running, monkeypatch):
    _run(running)
    monkeypatch.setenv(FLAG, "enforce")
    with running["pool"].connection() as conn:
        with pytest.raises(RuntimeError, match="abort"):
            with conn.transaction():
                assert reconcile_billing_meters(conn).meters_created
                raise RuntimeError("abort")
    assert _meters(running) == []
    created = _run(running).meters_created
    assert len(created) == 1
    assert _run(running).meters_created == []
    assert _meters(running) == [(created[0], None, 0, 1.0)]
    invoice = BillingEngine().generate_invoice(
        running["job_id"],
        "test",
        0,
        time.time() + 100,
        persist=False,
    )
    assert invoice.total_cad == 0
    with running["pool"].connection() as conn:
        action, evidence, resolved = conn.execute(
            "SELECT action_taken, action_result, resolved_at IS NOT NULL "
            "FROM reconciliation_findings WHERE resource_id=%s "
            "AND finding_type='billing_active_meter_missing'",
            (running["attempt_id"],),
        ).fetchone()
    assert action == "open_active_meter" and resolved
    assert evidence["meter_id"] == created[0] and evidence["observation_id"]


@pytest.mark.parametrize(
    "native_admission_fields", [{"cost_per_hour": 0}, {"cost_per_hour": 1.25}], indirect=True
)
def test_completion_settles_same_meter_at_captured_rate_once(
    running, monkeypatch, native_admission_fields
):
    rate = native_admission_fields["cost_per_hour"]
    monkeypatch.setenv(FLAG, "enforce")
    mid = _run(running).meters_created[0]
    _finish(running)
    # Completion caller and current host data cannot reprice captured usage.
    job = {
        "job_id": running["job_id"],
        "attempt_id": running["attempt_id"],
        "owner": running["job_id"],
        "started_at": 1,
        "completed_at": time.time() + 86400,
    }
    host = {"host_id": running["host_id"], "cost_per_hour": 100}
    engine = BillingEngine()
    settled = engine.meter_job(job, host)
    assert settled.meter_id == mid and settled.base_rate_per_hour == rate
    assert settled.duration_sec == pytest.approx(3600, abs=5)
    assert settled.total_cost_cad == pytest.approx(rate, abs=0.005)
    assert engine.meter_job(job, host) == settled
    assert len(_meters(running)) == 1


def test_orphaned_settlement_uses_same_authority_and_is_transactional(running, monkeypatch):
    monkeypatch.setenv(FLAG, "enforce")
    mid = _run(running).meters_created[0]
    _finish(running, simulate_orphan=True)
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", "enforce")
    with running["pool"].connection() as conn:
        with pytest.raises(RuntimeError, match="abort"):
            with conn.transaction():
                assert reconcile_billing_meters(conn).meters_closed == [mid]
                raise RuntimeError("abort")
    assert _meters(running) == [(mid, None, 0, 1.0)]
    assert _run(running).meters_closed == [mid]
    assert _run(running).meters_closed == []
    assert _meters(running)[0][2] == pytest.approx(1_000_000, abs=5_000)


@pytest.mark.parametrize(
    "defect",
    [
        "expired_lease",
        "wrong_fence",
        "wrong_host",
        "stale_observation",
        "missing_observation",
        "wrong_spec",
        "duplicate_workload",
        "exited",
        "stopped",
        "missing_owner",
        "owner_mismatch",
        "missing_rate",
        "negative_rate",
        "nan_rate",
        "unlocked_spot",
        "future_start",
        "missing_start",
        "completed_meter",
    ],
)
def test_repair_refuses_incomplete_or_conflicting_authority(running, monkeypatch, defect):
    monkeypatch.setenv(FLAG, "enforce")
    p = running
    with p["pool"].connection() as conn:
        if defect == "expired_lease":
            conn.execute(
                "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 hour' "
                "WHERE attempt_id=%s",
                (p["attempt_id"],),
            )
        elif defect == "wrong_fence":
            conn.execute(
                "UPDATE placement_leases SET fencing_token=fencing_token+1 WHERE attempt_id=%s",
                (p["attempt_id"],),
            )
        elif defect == "wrong_host":
            conn.execute(
                "UPDATE placement_leases SET host_id='other' WHERE attempt_id=%s",
                (p["attempt_id"],),
            )
        elif defect == "stale_observation":
            conn.execute(
                "UPDATE host_observations SET received_at=clock_timestamp()-interval '1 hour' "
                "WHERE host_id=%s",
                (p["host_id"],),
            )
        elif defect == "missing_observation":
            conn.execute("DELETE FROM host_observations WHERE host_id=%s", (p["host_id"],))
        elif defect in {"wrong_spec", "exited"}:
            conn.execute(
                "UPDATE observed_workloads SET spec_hash=%s, state=%s WHERE attempt_id=%s",
                (
                    "wrong" if defect == "wrong_spec" else p["spec_hash"],
                    "exited" if defect == "exited" else "running",
                    p["attempt_id"],
                ),
            )
        elif defect == "stopped":
            conn.execute(
                """UPDATE jobs SET status='stopping', phase='running', desired_state='stopped' WHERE job_id=%s""",
                (p["job_id"],),
            )
        elif defect == "missing_owner":
            conn.execute(
                "UPDATE jobs SET owner_id=NULL, payload=payload-'owner' WHERE job_id=%s",
                (p["job_id"],),
            )
        elif defect == "owner_mismatch":
            conn.execute(
                "UPDATE jobs SET owner_id='different-owner' WHERE job_id=%s", (p["job_id"],)
            )
        elif defect in {"missing_rate", "negative_rate", "nan_rate"}:
            value = {"missing_rate": None, "negative_rate": -1, "nan_rate": "NaN"}[defect]
            conn.execute(
                "UPDATE job_attempts SET placement_explanation=jsonb_set(placement_explanation, '{billing_snapshot,host,cost_per_hour}',%s::jsonb) WHERE attempt_id=%s",
                (Jsonb(value), p["attempt_id"]),
            )
        elif defect == "unlocked_spot":
            conn.execute(
                "UPDATE job_attempts SET placement_explanation=jsonb_set(placement_explanation,'{billing_snapshot,pricing_mode}','\"spot\"') WHERE attempt_id=%s",
                (p["attempt_id"],),
            )
        elif defect in {"future_start", "missing_start"}:
            conn.execute(
                "UPDATE job_attempts SET started_at=%s WHERE attempt_id=%s",
                (None if defect == "missing_start" else "2099-01-01", p["attempt_id"]),
            )
        elif defect == "completed_meter":
            conn.execute(
                "INSERT INTO usage_meters (meter_id, attempt_id, job_id, host_id, completed_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (p["job_id"], p["attempt_id"], p["job_id"], p["host_id"], time.time()),
            )
    if defect == "duplicate_workload":
        w = {k: p[k] for k in ("job_id", "attempt_id", "fencing_token", "spec_hash")}
        _observe(p, workloads=[{**w, "state": "running"}, {**w, "state": "running"}])
    before = _meters(p)
    assert _run(p).meters_created == []
    assert _meters(p) == before
    with p["pool"].connection() as conn:
        action, evidence, resolved = conn.execute(
            "SELECT action_taken, action_result, resolved_at FROM reconciliation_findings "
            "WHERE resource_id=%s AND finding_type='billing_active_meter_missing'",
            (p["attempt_id"],),
        ).fetchone()
    assert action == "active_meter_refused" and evidence["reason"] and resolved is None


@pytest.mark.parametrize(
    "table,key",
    [
        ("jobs", "job_id"),
        ("job_attempts", "attempt_id"),
        ("hosts", "host_id"),
        ("placement_leases", "attempt_id"),
    ],
)
def test_locked_authority_is_deferred_without_partial_meter(running, monkeypatch, table, key):
    monkeypatch.setenv(FLAG, "enforce")
    with running["pool"].connection() as holder:
        holder.execute(f"SELECT 1 FROM {table} WHERE {key}=%s FOR UPDATE", (running[key],))
        assert _run(running).meters_created == []
        assert _meters(running) == []
    assert len(_run(running).meters_created) == 1


def test_concurrent_repairs_and_completion_deliveries_create_one_settled_meter(
    running, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv(FLAG, "enforce")
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(_run, [running, running]))
    assert sum(len(r.meters_created) for r in results) == 1
    mid = _meters(running)[0][0]
    _finish(running)

    def complete(_):
        return BillingEngine().meter_job(
            {
                "job_id": running["job_id"],
                "attempt_id": running["attempt_id"],
                "owner": running["job_id"],
                "completed_at": time.time(),
            },
            {"host_id": running["host_id"], "cost_per_hour": 99},
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        settled = list(workers.map(complete, [0, 1]))
    assert settled[0] == settled[1] and settled[0].meter_id == mid
    assert settled[0].total_cost_cad == pytest.approx(1, abs=0.005)
    assert len(_meters(running)) == 1


@pytest.mark.parametrize(
    "native_admission_fields",
    [
        {"pricing_mode": "spot", "spot_rate_cad": 0.25, "num_gpus": 2},
        {"preemptible": True, "spot_rate_cad": 0.25, "num_gpus": 2},
    ],
    indirect=True,
)
def test_spot_snapshot_and_missing_terminal_repair_preserve_locked_multi_gpu_price(
    running, monkeypatch, native_admission_fields
):
    _finish(running, simulate_missing=True)
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER", "enforce")
    with running["pool"].connection() as conn:
        with pytest.raises(RuntimeError, match="abort"):
            with conn.transaction():
                assert reconcile_billing_meters(conn).meters_created
                raise RuntimeError("abort")
    assert _meters(running) == []
    assert len(_run(running).meters_created) == 1
    assert _meters(running)[0][2:] == (pytest.approx(500_000, abs=5_000), 0.5)
    assert _run(running).meters_created == []


@pytest.mark.parametrize(
    "defect", ["running", "missing_end", "reversed_end", "future_end", "wrong_host"]
)
def test_settlement_requires_terminal_binding_and_time_evidence(running, monkeypatch, defect):
    monkeypatch.setenv(FLAG, "enforce")
    _run(running)
    if defect != "running":
        _finish(running, simulate_orphan=True)
    with running["pool"].connection() as conn:
        if defect == "missing_end":
            conn.execute(
                "UPDATE job_attempts SET ended_at=NULL WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
        elif defect == "reversed_end":
            conn.execute(
                "UPDATE job_attempts SET ended_at=started_at-interval '1 hour' WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
        elif defect == "future_end":
            conn.execute(
                "UPDATE job_attempts SET ended_at=clock_timestamp()+interval '1 day' WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
        elif defect == "wrong_host":
            conn.execute(
                "UPDATE usage_meters SET host_id='other' WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
    before = _meters(running)
    with pytest.raises(ValueError, match="terminal attempt evidence|ownership"):
        BillingEngine().meter_job(
            {
                "job_id": running["job_id"],
                "attempt_id": running["attempt_id"],
                "completed_at": time.time(),
            },
            {"host_id": running["host_id"]},
        )
    assert _meters(running) == before


@pytest.mark.parametrize(
    "native_admission_fields",
    [{"pricing_mode": "spot", "spot_rate_cad": 0.25, "num_gpus": 2}],
    indirect=True,
)
def test_active_spot_snapshot_and_job_finding_converge(
    running, monkeypatch, native_admission_fields
):
    from control_plane.job_reconcile import reconcile_job

    monkeypatch.setenv(FLAG, "enforce")
    with running["pool"].connection() as conn:
        assert "job_active_meter_missing" in reconcile_job(conn, running["job_id"]).findings_opened
        assert len(reconcile_billing_meters(conn).meters_created) == 1
        assert reconcile_job(conn, running["job_id"]).findings_resolved == 1
    assert _meters(running)[0][1:] == (None, 0, 0.5)


def test_report_only_active_finding_resolves_after_terminal_completion(running):
    _run(running)
    _finish(running, simulate_missing=True)
    _run(running)
    with running["pool"].connection() as conn:
        assert conn.execute(
            "SELECT resolved_at IS NOT NULL FROM reconciliation_findings "
            "WHERE resource_id=%s AND finding_type='billing_active_meter_missing'",
            (running["attempt_id"],),
        ).fetchone() == (True,)
        # Terminal billing evidence is still missing and must remain visible.
        assert conn.execute(
            "SELECT resolved_at FROM reconciliation_findings "
            "WHERE resource_id=%s AND finding_type='billing_missing_meter'",
            (running["attempt_id"],),
        ).fetchone() == (None,)


def test_runtime_api_login_can_repair_and_settle_without_broader_role_grants(
    running,
    monkeypatch,
    provisioned_roles,
):
    import os
    import psycopg
    from psycopg.conninfo import make_conninfo

    monkeypatch.setenv(FLAG, "enforce")
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER", "enforce")
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("ALTER ROLE xcelsior_api LOGIN")
    try:
        with psycopg.connect(make_conninfo(dsn, user="xcelsior_api")) as conn:
            assert conn.execute("SELECT current_user, session_user").fetchone() == (
                "xcelsior_api",
                "xcelsior_api",
            )
            created = reconcile_billing_meters(conn).meters_created
            assert len(created) == 1
        _finish(running, simulate_orphan=True)
        with psycopg.connect(make_conninfo(dsn, user="xcelsior_api")) as conn:
            assert reconcile_billing_meters(conn).meters_closed == created
            for role in ("xcelsior_scheduler", "xcelsior_reconciler"):
                assert not conn.execute(
                    "SELECT has_table_privilege(%s, 'usage_meters', 'INSERT')",
                    (role,),
                ).fetchone()[0]
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("ALTER ROLE xcelsior_api NOLOGIN")


def test_observation_metadata_and_current_host_price_cannot_rewrite_repaired_usage(
    running, monkeypatch
):
    monkeypatch.setenv(FLAG, "enforce")
    with running["pool"].connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload||%s WHERE job_id=%s",
            (
                Jsonb(dict(owner="intruder", num_gpus=99, pricing_mode="spot", spot_rate_cad=999)),
                running["job_id"],
            ),
        )
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb(dict(cost_per_hour=999)), running["host_id"]),
        )
    assert len(_run(running).meters_created) == 1
    with running["pool"].connection() as conn:
        assert conn.execute(
            "SELECT owner,base_rate_per_hour,pricing_mode FROM usage_meters WHERE attempt_id=%s",
            (running["attempt_id"],),
        ).fetchone() == (running["job_id"], 1.0, "on_demand")


@pytest.mark.parametrize("native_admission_fields", [{"max_price_per_hour": 1}], indirect=True)
@pytest.mark.parametrize(
    "defect", ["missing_spec", "unadmitted_spec", "wrong_hash", "missing_quote", "above_ceiling"]
)
def test_active_repair_refuses_missing_native_admission_or_captured_quote(
    running, monkeypatch, defect
):
    monkeypatch.setenv(FLAG, "enforce")
    with running["pool"].connection() as conn:
        if defect == "missing_spec":
            conn.execute("UPDATE jobs SET spec=NULL WHERE job_id=%s", (running["job_id"],))
        elif defect == "wrong_hash":
            conn.execute("UPDATE jobs SET spec_hash='wrong' WHERE job_id=%s", (running["job_id"],))
        elif defect == "unadmitted_spec":
            conn.execute(
                "UPDATE jobs SET spec=spec-'execution_spec_version' WHERE job_id=%s",
                (running["job_id"],),
            )
        elif defect == "above_ceiling":
            conn.execute(
                "UPDATE job_attempts SET placement_explanation=jsonb_set(placement_explanation,'{billing_snapshot,host,cost_per_hour}','2') WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
        else:
            conn.execute(
                "UPDATE job_attempts SET placement_explanation=placement_explanation-'billing_snapshot' WHERE attempt_id=%s",
                (running["attempt_id"],),
            )
    assert _run(running).meters_created == []
    assert _meters(running) == []
