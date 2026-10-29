"""Missing fields must be corrected at their real authority, never invented."""

import uuid
from contextlib import contextmanager

import pytest
from psycopg.types.json import Jsonb

from db import DatabaseOps, _get_pg_pool
from control_plane.observations import ingest_observation, latest_observation
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.test_serverless_native_lifecycle import native  # noqa: F401
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401


@pytest.fixture
def connection():
    with _get_pg_pool().connection() as conn:
        try:
            yield conn
        finally:
            conn.rollback()


def device(uuid="GPU-physical-proof", index=0):
    return dict(
        gpu_uuid=uuid,
        device_index=index,
        pci_bus_id="0000:01:00.0",
        model="RTX 2060",
        total_vram_mb=6144,
    )


def test_physical_observations_populate_native_fields_and_retire_synthetic_slots(connection):
    conn = connection
    host = uuid.uuid4().hex
    DatabaseOps.upsert_host(conn, dict(host_id=host), backend="postgres")
    conn.execute("INSERT INTO host_gpu_devices(host_id,gpu_uuid) VALUES(%s,'slot:0')", (host,))
    ingest_observation(
        conn,
        host_id=host,
        session_id="first",
        observation_generation=1,
        workloads=[],
        capabilities={"egress_isolation": True},
        conditions={"gpu_inventory": "reported"},
        gpu_inventory=[device()],
    )
    row = conn.execute(
        "SELECT observation_session_id,capabilities,conditions,last_observed_at,administrative_state FROM hosts WHERE host_id=%s",
        (host,),
    ).fetchone()
    assert row[:3] == ("first", {"egress_isolation": True}, {"gpu_inventory": "reported"})
    assert row[3] is not None and row[4] == "pending"
    row = conn.execute(
        "SELECT gpu_uuid,pci_bus_id,model,total_vram_mb,last_observed_at FROM host_gpu_devices WHERE host_id=%s AND retired_at IS NULL",
        (host,),
    ).fetchone()
    assert (
        row[:4] == ("GPU-physical-proof", "0000:01:00.0", "RTX 2060", 6144) and row[4] is not None
    )
    assert conn.execute(
        "SELECT retired_at IS NOT NULL FROM host_gpu_devices WHERE host_id=%s AND gpu_uuid='slot:0'",
        (host,),
    ).fetchone() == (True,)
    generation = conn.execute(
        "SELECT inventory_generation FROM hosts WHERE host_id=%s", (host,)
    ).fetchone()
    ingest_observation(
        conn,
        host_id=host,
        session_id="first",
        observation_generation=2,
        workloads=[],
        gpu_inventory=[device()],
    )
    assert (
        conn.execute("SELECT inventory_generation FROM hosts WHERE host_id=%s", (host,)).fetchone()
        == generation
    )
    ingest_observation(
        conn,
        host_id=host,
        session_id="second",
        observation_generation=1,
        workloads=[],
        gpu_inventory=[device("GPU-new-proof")],
        # A new session takes over only once the current one is silent; this
        # models the restarted worker after that silence.
        handoff_silence_sec=0,
    )
    ingest_observation(
        conn,
        host_id=host,
        session_id="first",
        observation_generation=3,
        workloads=[],
        gpu_inventory=[device("GPU-stale-proof")],
    )
    assert latest_observation(conn, host)["session_id"] == "second"
    assert conn.execute(
        "SELECT gpu_uuid FROM host_gpu_devices WHERE host_id=%s AND retired_at IS NULL", (host,)
    ).fetchone() == ("GPU-new-proof",)
    ingest_observation(
        conn,
        host_id=host,
        session_id="second",
        observation_generation=0,
        workloads=[],
        gpu_inventory=[],
    )
    assert latest_observation(conn, host)["observation_generation"] == 1
    assert conn.execute(
        "SELECT gpu_uuid FROM host_gpu_devices WHERE host_id=%s AND retired_at IS NULL", (host,)
    ).fetchone() == ("GPU-new-proof",)

    ingest_observation(
        conn,
        host_id=host,
        session_id="second",
        observation_generation=2,
        workloads=[],
        gpu_inventory=[],
    )
    from control_plane.inventory import sync_host_gpu_inventory

    sync_host_gpu_inventory(conn, {"host_id": host, "gpu_count": 8, "total_vram_gb": 128})
    assert conn.execute(
        "SELECT count(*) FROM host_gpu_devices WHERE host_id=%s AND retired_at IS NULL", (host,)
    ).fetchone() == (0,)
    assert conn.execute(
        "SELECT availability_state FROM hosts WHERE host_id=%s", (host,)
    ).fetchone() == ("not_ready",)
    DatabaseOps.upsert_host(conn, {"host_id": host, "status": "active"}, backend="postgres")
    assert conn.execute(
        "SELECT availability_state FROM hosts WHERE host_id=%s", (host,)
    ).fetchone() == ("not_ready",)


@pytest.mark.parametrize(
    "inventory",
    [
        [device(), device()],
        [device(index=-1)],
        [device(uuid="slot:0")],
        [{**device(), "total_vram_mb": 0}],
    ],
)
def test_ambiguous_inventory_is_refused_before_any_observation_write(connection, inventory):
    host = uuid.uuid4().hex
    DatabaseOps.upsert_host(connection, dict(host_id=host), backend="postgres")
    with pytest.raises(ValueError, match="inventory"):
        ingest_observation(
            connection,
            host_id=host,
            session_id="proof",
            observation_generation=1,
            workloads=[],
            gpu_inventory=inventory,
        )
    assert connection.execute(
        "SELECT count(*) FROM host_observations WHERE host_id=%s", (host,)
    ).fetchone() == (0,)


def test_native_owner_and_phase_win_over_conflicting_observations():
    from routes.instances import _count_active_instances

    job, owner = uuid.uuid4().hex, uuid.uuid4().hex
    pool = _get_pg_pool()
    try:
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, dict(job_id=job, owner=owner), backend="postgres")
            conn.execute(
                "UPDATE jobs SET status='assigned',phase='scheduled',payload=payload || %s WHERE job_id=%s",
                (
                    Jsonb(
                        {"owner": "forged-owner", "owner_id": "forged-owner", "status": "completed"}
                    ),
                    job,
                ),
            )
        with pool.connection() as conn:
            assert DatabaseOps.get_job(conn, job, backend="postgres")["owner"] == owner
            assert (
                next(
                    item
                    for item in DatabaseOps.load_jobs(conn, backend="postgres")
                    if item["job_id"] == job
                )["owner"]
                == owner
            )
        assert _count_active_instances(owner) == 1
        assert _count_active_instances("forged-owner") == 0
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job,))


def test_concurrency_does_not_fall_back_to_process_state_on_database_failure(monkeypatch):
    import db
    from fastapi import HTTPException
    from routes.instances import _count_active_instances

    def unavailable():
        raise RuntimeError("offline")

    monkeypatch.setattr(db, "_get_pg_pool", unavailable)
    with pytest.raises(HTTPException) as exc:
        _count_active_instances("customer")
    assert exc.value.status_code == 503


def test_general_upserts_cannot_transfer_known_identity_or_duplicate_it_in_payload(connection):
    marker = uuid.uuid4().hex
    conn = connection
    DatabaseOps.upsert_job(
        conn, {"job_id": marker, "owner": "customer", "team_id": "team"}, backend="postgres"
    )
    DatabaseOps.upsert_job(
        conn,
        {"job_id": marker, "owner": "forged", "tenant_id": "forged", "team_id": "forged"},
        backend="postgres",
    )
    assert conn.execute(
        "SELECT owner_id,tenant_id,team_id,payload ?| ARRAY['owner','owner_id','tenant_id','team_id'] FROM jobs WHERE job_id=%s",
        (marker,),
    ).fetchone() == ("customer", "customer", "team", False)
    DatabaseOps.upsert_host(conn, {"host_id": marker, "owner": "provider"}, backend="postgres")
    DatabaseOps.upsert_host(
        conn,
        {
            "host_id": marker,
            "owner": "forged",
            "tenant_id": "forged",
            "provider_id": "forged",
            "user_id": "forged",
        },
        backend="postgres",
    )
    assert conn.execute(
        "SELECT owner_id,tenant_id,payload ?| ARRAY['owner','owner_id','tenant_id','provider_id','user_id'] FROM hosts WHERE host_id=%s",
        (marker,),
    ).fetchone() == ("provider", "provider", False)
    assert DatabaseOps.get_host(conn, marker, backend="postgres")["owner"] == "provider"
    assert (
        next(
            item
            for item in DatabaseOps.load_hosts(conn, backend="postgres")
            if item["host_id"] == marker
        )["owner"]
        == "provider"
    )


def test_docs_refresh_works_under_real_api_role_without_truncate_rights(
    empty_db, provisioned_roles, monkeypatch
):
    import ai_assistant
    from psycopg.rows import dict_row
    from tests.test_migration_from_production_snapshot import _alembic, _connect

    _alembic(empty_db, "upgrade", "head")
    with _connect(empty_db) as conn:
        conn.execute("GRANT SELECT,INSERT,UPDATE,DELETE ON ai_docs TO xcelsior_api")
        conn.execute("GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO xcelsior_api")
        assert conn.execute(
            "SELECT has_table_privilege('xcelsior_api','ai_docs','TRUNCATE')"
        ).fetchone() == (False,)
        conn.execute("SET LOCAL ROLE xcelsior_api")
        conn.row_factory = dict_row

        @contextmanager
        def isolated_corpus():
            yield conn

        monkeypatch.setattr(ai_assistant, "_ai_db", isolated_corpus)
        monkeypatch.setattr(
            ai_assistant,
            "collect_public_docs",
            lambda: [("support_docs/proof.md", "physical GPU inventory proof")],
        )
        assert ai_assistant.ingest_docs() == 1
        assert ai_assistant.search_docs("physical GPU")[0]["source"] == "support_docs/proof.md"


def test_active_allocation_keeps_its_device_through_inventory_replacement(native):
    native.reserve()
    with native.pool.connection() as conn:
        before = conn.execute(
            "SELECT gpu_device_id,gpu_uuid FROM host_gpu_devices WHERE host_id=%s",
            (native.hid,),
        ).fetchall()
        ingest_observation(
            conn,
            host_id=native.hid,
            session_id="inventory-proof",
            observation_generation=1,
            workloads=[],
            gpu_inventory=[device()],
        )
        # The allocated device keeps its row (the attempt's authority points at
        # it) but, missing from a complete report, is never offered again. The
        # reported device is added alongside it rather than replacing it.
        busy = before[0][0]
        assert conn.execute(
            "SELECT gpu_uuid,retired_at IS NULL,health FROM host_gpu_devices WHERE gpu_device_id=%s",
            (busy,),
        ).fetchone() == (before[0][1], True, "unhealthy")
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE gpu_device_id=%s AND released_at IS NULL",
            (busy,),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT gpu_uuid FROM host_gpu_devices WHERE host_id=%s AND gpu_device_id<>%s "
            "AND retired_at IS NULL",
            (native.hid, busy),
        ).fetchall() == [(device()["gpu_uuid"],)]
        assert conn.execute(
            "SELECT gpu_inventory FROM host_observations WHERE host_id=%s AND session_id='inventory-proof'",
            (native.hid,),
        ).fetchone() == ([device()],)
        conn.rollback()


def test_nvml_inventory_failure_never_publishes_a_partial_device_list(monkeypatch):
    import nvml_telemetry
    from types import SimpleNamespace
    from unittest.mock import Mock

    nvml = Mock()
    nvml.nvmlDeviceGetCount.return_value = 2
    nvml.nvmlDeviceGetHandleByIndex.side_effect = [
        "first",
        RuntimeError("second device unavailable"),
    ]
    nvml.nvmlDeviceGetUUID.return_value = b"GPU-first"
    nvml.nvmlDeviceGetName.return_value = b"RTX 2060"
    nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=6 * 1024**3)
    monkeypatch.setattr(nvml_telemetry, "pynvml", nvml)
    monkeypatch.setattr(nvml_telemetry, "_nvml_initialized", True)
    monkeypatch.setattr(nvml_telemetry, "get_gpu_pci_info", lambda _: {"bus_id": "0000:01:00.0"})
    assert nvml_telemetry.collect_gpu_inventory() is None


def test_worker_reports_measured_inventory_and_capabilities(monkeypatch):
    import nvml_telemetry
    import worker_agent
    import egress_isolation

    posted = []
    monkeypatch.setattr(worker_agent, "_v2_enabled", True)
    monkeypatch.setattr(worker_agent, "is_nvml_available", lambda: True)
    monkeypatch.setattr(nvml_telemetry, "collect_gpu_inventory", lambda: [device()])
    monkeypatch.setattr(egress_isolation, "isolation_ready", lambda: True)
    monkeypatch.setattr(worker_agent, "_collect_local_workloads", lambda: [])
    monkeypatch.setattr(worker_agent, "_resume_pending_log_delivery", lambda: None)
    monkeypatch.setattr(worker_agent, "_v2_retry_resource_cleanup", lambda **_: None)
    monkeypatch.setattr(
        worker_agent, "_v2_post", lambda path, body, **_: (posted.append(body) or 200, {})
    )
    assert worker_agent.report_observations_v2(force=True)
    assert posted[0]["gpu_inventory"] == [device()]
    assert posted[0]["capabilities"] == {"egress_isolation": True}
    assert posted[0]["conditions"] == {"gpu_inventory": "reported"}


def test_native_terminal_report_records_observed_exit_without_allowing_stale_overwrite():
    from tests.fixtures.native_runtime import running_instance
    from control_plane.attempts import report_attempt_status
    from control_plane.db import run_transaction
    from control_plane.leases import FencingViolation

    with running_instance() as instance:
        auth = dict(job_id=instance["job_id"], host_id=instance["host_id"], **instance["authority"])
        run_transaction(
            lambda conn: report_attempt_status(
                conn, status="succeeded", detail={"exit_code": 0}, **auth
            )
        )
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT exit_code FROM jobs WHERE job_id=%s", (instance["job_id"],)
            ).fetchone() == (0,)
        with pytest.raises(FencingViolation):
            run_transaction(
                lambda conn: report_attempt_status(
                    conn, status="failed", detail={"exit_code": 1}, **auth
                )
            )
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT exit_code FROM jobs WHERE job_id=%s", (instance["job_id"],)
            ).fetchone() == (0,)


def test_worker_exit_report_carries_the_measured_code(monkeypatch):
    import worker_agent as worker
    from unittest.mock import Mock

    auth = {
        "job_id": "job",
        "attempt_id": "attempt",
        "fencing_token": 1,
        "container_name": "xcl-job-attempt",
    }
    report = Mock(return_value="ok")
    monkeypatch.setattr(worker, "_v2_attempts", {"job": auth})
    monkeypatch.setattr(worker, "v2_report_attempt_status", report)
    monkeypatch.setattr(worker, "_v2_forget_attempt", lambda *a: None)
    assert worker.report_job_status("job", "completed", expected_v2_auth=auth, observed_exit_code=0)
    assert report.call_args.args[3] == {"exit_code": 0}
