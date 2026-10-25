"""Run the actual persistence/controllers with both projection functions gone.

The fixture saves and restores the shipped DDL in the disposable test database.
No substitute trigger, default or test normalizer supplies native fields.
"""

import time
import uuid

import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from db import DatabaseOps, _get_pg_pool
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease, expire_stale_leases
from control_plane.lifecycle import (
    request_fenced_stop,
    request_fenced_stop_remove,
    request_fresh_attempt_resume,
)
from control_plane.serverless_workers import cancel_unstarted_worker
from tests.test_serverless_native_lifecycle import native as native_base  # noqa: F401
from tests.test_privacy_sinks import subject as privacy_subject_base  # noqa: F401
from tests.native_jobs import admit_fixture_job


@pytest.fixture
def without_projection_triggers():
    pool = _get_pg_pool()
    with pool.connection() as conn:
        triggers = conn.execute("""SELECT pg_get_triggerdef(oid) FROM pg_trigger
            WHERE tgname IN ('trg_jobs_control_plane_projection',
                             'trg_hosts_control_plane_projection') ORDER BY tgname""").fetchall()
        assert len(triggers) in {0, 2}, "partially removed projection schema"
        # Prepared regprocedure constants can retain a dropped function's OID.
        # Resolve live names on every rehearsal instead.
        functions = conn.execute("""SELECT pg_get_functiondef(p.oid)
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname='public' AND p.pronargs=0
              AND p.proname IN ('control_plane_project_job','control_plane_project_host')
            ORDER BY p.proname""").fetchall()
        assert len(functions) == len(triggers)
        if triggers:
            conn.execute("DROP TRIGGER trg_jobs_control_plane_projection ON jobs")
            conn.execute("DROP TRIGGER trg_hosts_control_plane_projection ON hosts")
            conn.execute("DROP FUNCTION control_plane_project_job()")
            conn.execute("DROP FUNCTION control_plane_project_host()")
        assert conn.execute("""SELECT to_regprocedure('control_plane_project_job()'),
            to_regprocedure('control_plane_project_host()')""").fetchone() == (None, None)
    try:
        yield pool
    finally:
        with pool.connection() as conn:
            for (statement,) in functions + triggers:
                conn.execute(statement)


@pytest.fixture
def native(without_projection_triggers, native_base):
    with native_base.pool.connection() as conn:
        payload = conn.execute(
            "DELETE FROM jobs WHERE job_id=%s RETURNING payload", (native_base.jid,)
        ).fetchone()[0]
        DatabaseOps.upsert_job(
            conn, {**payload, "status": "queued", "owner": native_base.owner}, backend="postgres"
        )
        admit_fixture_job(conn, native_base.jid)
    return native_base


@pytest.fixture
def privacy_subject(without_projection_triggers, privacy_subject_base):
    return privacy_subject_base


def job_row(x):
    with x.pool.connection() as conn:
        return conn.execute(
            """SELECT status,phase,desired_state,effective_priority,
                fair_share_finish,EXTRACT(EPOCH FROM queued_at),submitted_at
            FROM jobs WHERE job_id=%s""",
            (x.jid,),
        ).fetchone()


def report(x, status):
    return run_transaction(
        lambda conn: report_attempt_status(
            conn,
            job_id=x.jid,
            attempt_id=x.r.attempt_id,
            host_id=x.hid,
            fencing_token=x.r.fencing_token,
            status=status,
        )
    )


def test_admission_and_observations_write_every_native_field(without_projection_triggers):
    pool = without_projection_triggers
    expected = {
        "queued": ("pending", "running"),
        "preempted": ("pending", "running"),
        "assigned": ("scheduled", "running"),
        "leased": ("scheduled", "running"),
        "starting": ("starting", "running"),
        "restarting": ("starting", "running"),
        "running": ("running", "running"),
        "stopping": ("running", "stopped"),
        "completed": ("succeeded", "running"),
        "failed": ("failed", "running"),
        "stopped": ("stopped", "stopped"),
        "paused": ("stopped", "stopped"),
        "cancelled": ("stopped", "stopped"),
        "terminated": ("stopped", "stopped"),
    }
    # Roll back all fixture jobs, including any write preceding a refusal.
    with pool.connection() as conn:
        try:
            for status, state in expected.items():
                jid = f"native-projection-{uuid.uuid4().hex}"
                payload = dict(job_id=jid, status=status, priority=7, submitted_at=1234)
                DatabaseOps.upsert_job(conn, payload, backend="postgres")
                row = conn.execute(
                    """SELECT phase,desired_state,effective_priority,
                        EXTRACT(EPOCH FROM queued_at) FROM jobs WHERE job_id=%s""",
                    (jid,),
                ).fetchone()
                assert row == (*state, 7, 1234 if state[0] == "pending" else None)
            with pytest.raises(ValueError, match="unsupported job status"):
                DatabaseOps.upsert_job(
                    conn, dict(job_id="unknown-native", status="unknown"), backend="postgres"
                )
        finally:
            conn.rollback()


def test_upsert_preserves_controller_fields_and_updates_real_requeue(native):
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET desired_state='stopped',effective_priority=900,fair_share_finish=17 WHERE job_id=%s",
            (native.jid,),
        )
        payload = DatabaseOps.get_job(conn, native.jid, backend="postgres")
        DatabaseOps.upsert_job(conn, payload, backend="postgres")
    assert job_row(native)[2:5] == ("stopped", 900, 17)
    with native.pool.connection() as conn:
        payload.update(status="running")
        DatabaseOps.upsert_job(conn, payload, backend="postgres")
    assert job_row(native)[:5] == ("running", "running", "stopped", 900, 17)
    with native.pool.connection() as conn:
        payload.update(status="queued", submitted_at=2222)
        DatabaseOps.upsert_job(conn, payload, backend="postgres")
    assert job_row(native)[:6] == ("queued", "pending", "running", 900, 17, 2222)
    with native.pool.connection() as conn:
        payload.update(priority=8)
        DatabaseOps.upsert_job(conn, payload, backend="postgres")
    assert job_row(native)[3:5] == (8, 17)


@pytest.mark.parametrize(
    "terminal,phase,desired",
    [
        ("succeeded", "succeeded", "running"),
        ("failed", "failed", "running"),
        ("stopped", "stopped", "stopped"),
    ],
)
def test_reserve_claim_report_and_settlement_are_native(native, terminal, phase, desired):
    native.reserve()
    assert job_row(native)[:3] == ("assigned", "scheduled", "running")
    run_transaction(
        lambda conn: claim_lease(
            conn,
            lease_id=native.r.lease_id,
            job_id=native.jid,
            worker_session_id="native-projection-test",
            attempt_id=native.r.attempt_id,
            host_id=native.hid,
            fencing_token=native.r.fencing_token,
        )
    )
    for status, projected in [
        ("lease_claimed", "scheduled"),
        ("starting", "starting"),
        ("running", "running"),
    ]:
        report(native, status)
        assert job_row(native)[1:3] == (projected, "running")
    report(native, terminal)
    assert job_row(native)[1:3] == (phase, desired)
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT active_attempt_id FROM jobs WHERE job_id=%s", (native.jid,)
            ).fetchone()[0]
            is None
        )
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (native.r.lease_id,)
            ).fetchone()[0]
            == "released"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND status='active'",
                (native.r.attempt_id,),
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize(
    "intent,final_status,phase,desired",
    [
        ("stop", "stopped", "stopped", "stopped"),
        ("cancel", "cancelled", "stopped", "stopped"),
        ("terminate", "terminated", "stopped", "stopped"),
        ("restart", "queued", "pending", "running"),
    ],
)
def test_stop_intent_survives_late_worker_observation(native, intent, final_status, phase, desired):
    native.reserve()
    run_transaction(
        lambda conn: claim_lease(
            conn,
            lease_id=native.r.lease_id,
            job_id=native.jid,
            worker_session_id="native-projection-test",
            attempt_id=native.r.attempt_id,
            host_id=native.hid,
            fencing_token=native.r.fencing_token,
        )
    )
    if intent == "stop":
        report(native, "starting")
        report(native, "running")
        result = request_fenced_stop(job_id=native.jid, created_by="native-test")
    else:
        result = request_fenced_stop_remove(
            job_id=native.jid, created_by="native-test", intent=intent
        )
    assert result.ok
    assert job_row(native)[:3] == ("stopping", "running", "stopped")
    if intent != "stop":
        report(native, "starting")
        assert job_row(native)[1:3] == ("starting", "stopped")
    report(native, "running")
    assert job_row(native)[1:3] == ("running", "stopped")
    before = time.time()
    report(native, "stopped")
    row = job_row(native)
    assert row[:3] == (final_status, phase, desired)
    if intent == "restart":
        assert abs(float(row[5]) - row[6]) < 0.000001
        assert before <= row[6] <= time.time()
    else:
        result = request_fresh_attempt_resume(job_id=native.jid, created_by="native-test")
        # Only stopped work supports resume; cancel/terminate cannot resurrect.
        assert result.ok is (intent == "stop")
        if intent == "stop":
            row = job_row(native)
            assert row[:3] == ("queued", "pending", "running")
            assert abs(float(row[5]) - row[6]) < 0.000001


@pytest.mark.parametrize("claimed", [False, True])
def test_serverless_expiry_is_terminal_without_projection(native, claimed):
    native.reserve()
    if claimed:
        run_transaction(
            lambda conn: claim_lease(
                conn,
                lease_id=native.r.lease_id,
                job_id=native.jid,
                worker_session_id="native-projection-test",
                attempt_id=native.r.attempt_id,
                host_id=native.hid,
                fencing_token=native.r.fencing_token,
            )
        )
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET claim_deadline=clock_timestamp()-interval '1 hour',expires_at=clock_timestamp()-interval '1 hour' WHERE lease_id=%s",
            (native.r.lease_id,),
        )
    expired = run_transaction(lambda conn: expire_stale_leases(conn))
    assert any(row["attempt_id"] == native.r.attempt_id for row in expired)
    assert job_row(native)[:3] == ("failed", "failed", "running")
    assert native.repo.get_worker(native.wid)["released_at"] is not None


def test_unstarted_serverless_cancel_is_native(native):
    assert cancel_unstarted_worker(native.wid, native.jid)
    assert job_row(native)[:3] == ("cancelled", "stopped", "stopped")


def test_observation_cannot_turn_serverless_into_generic_wallet_billing(native):
    from billing import get_billing_engine
    from tests.test_serverless_native_lifecycle import start

    start(native)
    with native.pool.connection() as conn:
        DatabaseOps.upsert_job(
            conn,
            dict(
                job_id=native.jid,
                status="running",
                host_id=native.hid,
                owner=native.owner,
                started_at=time.time() - 3600,
                job_type="batch",
                num_gpus=9,
            ),
            backend="postgres",
        )
    engine = get_billing_engine()
    engine.deposit(native.owner, 100, description="native billing boundary test")
    assert engine.bill_running_period(native.jid)["billed"] is False
    engine.auto_billing_cycle()
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (native.jid,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("periodic", [False, True])
def test_wallet_billing_uses_admitted_identity_and_gpu_constraints(native, periodic):
    from billing import get_billing_engine

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
        DatabaseOps.upsert_job(
            conn,
            dict(
                job_id=native.jid,
                owner=native.owner,
                status="queued",
                gpu_model=native.jid,
                num_gpus=2,
                pricing_mode="spot",
                spot_rate_cad=0.10,
                image="test/image:1",
            ),
            backend="postgres",
        )
        admit_fixture_job(conn, native.jid)
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"spot_enabled": True, "spot_gpu_slots": 2}), native.hid),
        )
        conn.execute(
            """INSERT INTO host_gpu_devices(host_id,gpu_uuid,model,total_vram_mb,
                allocatable_vram_mb,health) VALUES(%s,%s,'RTX 4090',24576,24576,'healthy')""",
            (native.hid, f"GPU-extra-{native.hid}"),
        )
    native.reserve()
    run_transaction(
        lambda conn: claim_lease(
            conn,
            lease_id=native.r.lease_id,
            job_id=native.jid,
            attempt_id=native.r.attempt_id,
            host_id=native.hid,
            fencing_token=native.r.fencing_token,
            worker_session_id="native-billing-test",
        )
    )
    with native.pool.connection() as conn:
        # Recorded server start drives the real billing interval, with no sleep.
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-interval '1 hour' WHERE attempt_id=%s",
            (native.r.attempt_id,),
        )
    report(native, "running")
    with native.pool.connection() as conn:
        DatabaseOps.upsert_job(
            conn,
            dict(
                job_id=native.jid,
                status="running",
                host_id=native.hid,
                owner="wrong-owner",
                started_at=time.time() - 3600,
                gpu_model="wrong-gpu",
                num_gpus=99,
                pricing_mode="on_demand",
                tier="wrong-tier",
            ),
            backend="postgres",
        )
    engine = get_billing_engine()
    engine.deposit(native.owner, 100, description="admitted billing identity test")
    if periodic:
        engine.auto_billing_cycle()
    else:
        assert engine.bill_running_period(native.jid)["billed"]
    with native.pool.connection() as conn:
        rows = conn.execute(
            "SELECT customer_id,gpu_model,rate_per_hour,amount_micros FROM billing_cycles WHERE job_id=%s",
            (native.jid,),
        ).fetchall()
        assert len(rows) == 1
        owner, model, rate, amount = rows[0]
        assert (owner, model) == (native.owner, native.jid)
        assert float(rate) == 0.2 and 199000 <= amount <= 201000
    assert 99.79 <= engine.get_wallet(native.owner)["balance_cad"] <= 99.81


def test_privacy_stop_sets_native_phase_and_excludes_placement(native):
    from privacy_sinks import _stop_subject_workloads
    from control_plane.scheduler.claim import claim_next_job

    with native.pool.connection() as conn:
        conn.execute("UPDATE jobs SET owner_id=%s WHERE job_id=%s", (native.owner, native.jid))
        assert _stop_subject_workloads(conn, [native.owner]) == (1, 1)
        assert (
            claim_next_job(conn, replica_id="privacy-test", scope_gpu_models=[native.jid]) is None
        )
    assert job_row(native)[:3] == ("stopping", "running", "stopped")


def test_provider_deletion_drains_hosts_without_projection(
    privacy_subject, without_projection_triggers
):
    from privacy_sinks import delete_authoritative_subject

    pool = without_projection_triggers
    pid = f"native-provider-{uuid.uuid4().hex}"
    hids = [f"native-deletion-{uuid.uuid4().hex}" for _ in range(3)]
    try:
        with pool.connection() as conn:
            conn.execute(
                "UPDATE users SET provider_id=%s WHERE user_id=%s",
                (pid, privacy_subject["user_id"]),
            )
            for hid, status, admitted, admin in zip(
                hids,
                ["active", "active", "disabled"],
                ["admitted", "pending", "admitted"],
                ["admitted", "pending", "disabled"],
                strict=True,
            ):
                conn.execute(
                    """INSERT INTO hosts(host_id,status,registered_at,payload,provider_id,
                    admission_state,administrative_state,availability_state)
                    VALUES(%s,%s,100,'{}',%s,%s,%s,'unknown')""",
                    (hid, status, pid, admitted, admin),
                )
        outcome = delete_authoritative_subject(privacy_subject["request"], {})
        assert outcome.status == "completed"
        assert outcome.evidence["rows_affected"]["hosts_drained"] == 2
        with pool.connection() as conn:
            for hid, expected in zip(
                hids,
                [
                    ("draining", "draining", "unknown"),
                    ("draining", "pending", "unknown"),
                    ("disabled", "disabled", "unknown"),
                ],
                strict=True,
            ):
                assert (
                    conn.execute(
                        "SELECT status,administrative_state,availability_state FROM hosts WHERE host_id=%s",
                        (hid,),
                    ).fetchone()
                    == expected
                )
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM hosts WHERE host_id=ANY(%s)", (hids,))


@pytest.mark.parametrize("intent", [None, "cancel", "terminate", "restart"])
def test_non_serverless_expiry_preserves_intent_and_requeues_with_new_time(native, intent):
    with native.pool.connection() as conn:
        # A distinct admitted batch workload, not mutable edits to an admitted
        # serverless spec. No worker is bound for this reservation.
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
        DatabaseOps.upsert_job(
            conn,
            dict(
                job_id=native.jid,
                status="queued",
                gpu_model=native.jid,
                image="test/image:1",
                num_gpus=1,
                submitted_at=100,
            ),
            backend="postgres",
        )
        admit_fixture_job(conn, native.jid)
    native.reserve()
    if intent:
        run_transaction(
            lambda conn: claim_lease(
                conn,
                lease_id=native.r.lease_id,
                job_id=native.jid,
                attempt_id=native.r.attempt_id,
                host_id=native.hid,
                fencing_token=native.r.fencing_token,
                worker_session_id="native-projection-test",
            )
        )
        assert request_fenced_stop_remove(
            job_id=native.jid, intent=intent, created_by="native-test"
        ).ok
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET claim_deadline=clock_timestamp()-interval '1 hour',expires_at=clock_timestamp()-interval '1 hour' WHERE lease_id=%s",
            (native.r.lease_id,),
        )
    run_transaction(lambda conn: expire_stale_leases(conn))
    row = job_row(native)
    if intent in {None, "restart"}:
        assert row[:3] == ("queued", "pending", "running")
        assert abs(float(row[5]) - row[6]) < 0.000001 and row[6] > 100
    else:
        assert row[:3] == (
            "cancelled" if intent == "cancel" else "terminated",
            "stopped",
            "stopped",
        )


def test_host_inventory_and_operator_decisions_need_no_projection(without_projection_triggers):
    import host_admission as ha

    pool = without_projection_triggers
    hid = f"native-host-{uuid.uuid4().hex}"
    actor = dict(
        user_id="native-operator", email="native@example.test", role="admin", is_admin=True
    )
    host = dict(
        host_id=hid,
        status="active",
        owner="native-provider",
        gpu_model="RTX 4090",
        total_vram_gb=24,
        gpu_count=1,
        cost_per_hour=0.75,
        admitted=True,
        admission_state="admitted",
        admission_version=999,
        admission_decision_id=str(uuid.uuid4()),
    )

    def state():
        with pool.connection() as conn:
            return conn.execute(
                """SELECT administrative_state,availability_state,
                admission_state,admission_version,admission_decision_id::text,payload
                FROM hosts WHERE host_id=%s""",
                (hid,),
            ).fetchone()

    try:
        with pool.connection() as conn:
            DatabaseOps.upsert_host(conn, host, backend="postgres")
        row = state()
        assert row[:5] == ("pending", "ready", "pending", 0, None)
        assert row[5]["admitted"] is False and row[5]["admission_decision_id"] is None
        for typ in ha.REQUIRED_AUTHORITATIVE_EVIDENCE:
            ha.record_authoritative_evidence(
                host_id=hid,
                actor=actor,
                evidence_type=typ,
                verdict="pass",
                summary={"note": "test"},
                idempotency_key=f"native-{typ}-{hid}",
            )
        ha.decide_admission(
            host_id=hid,
            actor=actor,
            action="admit",
            reason="verified",
            idempotency_key=f"native-admit-{hid}",
            expected_version=0,
        )
        admitted = state()
        assert admitted[:4] == ("admitted", "ready", "admitted", 1)
        for status, admin, available in [
            ("draining", "draining", "unknown"),
            ("dead", "admitted", "not_ready"),
            ("disabled", "disabled", "unknown"),
            ("active", "admitted", "ready"),
        ]:
            with pool.connection() as conn:
                DatabaseOps.upsert_host(
                    conn, {**host, "status": status, "admitted": False}, backend="postgres"
                )
            row = state()
            assert row[:5] == (admin, available, "admitted", 1, admitted[4])
            assert row[5]["admitted"] is True
            assert row[5]["admission_version"] == 1
            assert row[5]["admission_decision_id"] == admitted[4]
            if status in ("draining", "disabled"):
                # Only an explicit operator lifecycle request reopens a host;
                # an incoming observation must preserve the restriction.
                from control_plane.host_lifecycle import set_draining

                run_transaction(lambda c: set_draining(c, host_id=hid, draining=False))
        ha.decide_admission(
            host_id=hid,
            actor=actor,
            action="revoke",
            reason="operator stop",
            idempotency_key=f"native-revoke-{hid}",
            expected_version=1,
        )
        assert state()[:4] == ("disabled", "unknown", "revoked", 2)
        with pool.connection() as conn:
            DatabaseOps.upsert_host(conn, host, backend="postgres")
        assert state()[:4] == ("disabled", "unknown", "revoked", 2)
        assert state()[5]["admitted"] is False
    finally:
        with pool.connection() as conn:
            for table in [
                "host_admission_decisions",
                "host_admission_evidence",
                "gpu_offers",
                "hosts",
            ]:
                conn.execute(
                    sql.SQL("DELETE FROM {} WHERE host_id=%s").format(sql.Identifier(table)), (hid,)
                )
