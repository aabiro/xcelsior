"""Restricted logins must support real workflows without extra domain authority."""

from __future__ import annotations

import os
import uuid

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege
from psycopg import sql
import pytest

import db
from control_plane.db import run_transaction
from control_plane.db_roles import SERVICE_ROLES, service_scope, service_dsn_env_var
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind
from serverless.repo import EndpointCreate, ServerlessRepo
from tests.test_db_service_roles import provisioned_roles
from tests.test_serverless_worker_attempt_binding import (
    scratch,
    _mkhost,
    _mk_serverless_job,
    _worker_row,
)


@pytest.fixture
def restricted_logins(provisioned_roles, monkeypatch):
    admin_dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    # Restore the ordinary pool before autouse DB teardown runs. Otherwise
    # it tries to reconnect through the temporary login after NOLOGIN.
    with monkeypatch.context() as identity_patch:
        identity_patch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
        identity_patch.setenv("XCELSIOR_SERVICE", "api")
        identity_patch.setattr(db, "_pg_pool", None)
        identity_patch.setattr(db, "_pg_pool_identity", None)
        identity_patch.setattr(db, "_scoped_pg_pools", {})
        dsns = {}
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            for service in ["api", "scheduler", "reconciler", "billing", "projector"]:
                name = SERVICE_ROLES[service].name
                admin.execute(
                    sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                        sql.Identifier(name), sql.Literal("isolated-role-workflow-password")
                    )
                )
                dsns[service] = make_conninfo(
                    admin_dsn, user=name, password="isolated-role-workflow-password"
                )
                identity_patch.setenv(service_dsn_env_var(service), dsns[service])
        try:
            yield dsns
        finally:
            for pool in {db._pg_pool, *db._scoped_pg_pools.values()} - {None}:
                pool.close()
            with psycopg.connect(admin_dsn, autocommit=True) as admin:
                for service in dsns:
                    admin.execute(
                        sql.SQL("ALTER ROLE {} NOLOGIN PASSWORD NULL").format(
                            sql.Identifier(SERVICE_ROLES[service].name)
                        )
                    )


@pytest.mark.parametrize("serverless", [False, True])
def test_scheduler_login_reserves_real_work_and_preserves_serverless_authority(
    scratch, restricted_logins, serverless
):
    repo = ServerlessRepo()
    endpoint = repo.create_endpoint(
        EndpointCreate(
            owner_id="role-proof-" + uuid.uuid4().hex,
            name="role-proof",
            mode="preset",
            model_ref="m",
            min_workers=0,
        )
    )
    scratch["endpoints"].append(endpoint["endpoint_id"])
    worker = repo.create_worker(endpoint["endpoint_id"], gpu_count=1)
    scratch["workers"].append(worker["worker_id"])
    job_id = _mk_serverless_job(scratch, worker["worker_id"], serverless=serverless)
    host_id = _mkhost(scratch)
    with service_scope("scheduler"):
        with db.pg_transaction() as conn:
            assert conn.execute("SELECT current_user,session_user").fetchone() == (
                "xcelsior_scheduler",
                "xcelsior_scheduler",
            )
            claimed = claim_next_job(conn, replica_id="role-workflow", scope_gpu_models=[job_id])
        assert claimed and claimed.job_id == job_id
        reservation = run_transaction(
            lambda conn: reserve_and_bind(
                conn,
                job_id=job_id,
                claim_token=claimed.claim_token,
                replica_id="role-workflow",
                host_id=host_id,
                num_gpus=1,
            )
        )
    row = _worker_row(worker["worker_id"])
    assert row is not None
    assert (
        str(row["attempt_id"]) == str(reservation.attempt_id)
        if serverless
        else row["attempt_id"] is None
    )
    with psycopg.connect(restricted_logins["scheduler"]) as scheduler:
        with pytest.raises(InsufficientPrivilege):
            scheduler.execute(
                "UPDATE serverless_workers SET state=state WHERE worker_id=%s",
                (worker["worker_id"],),
            )
        scheduler.rollback()
        with pytest.raises(InsufficientPrivilege):
            scheduler.execute(
                "UPDATE serverless_endpoints SET name=name WHERE endpoint_id=%s",
                (endpoint["endpoint_id"],),
            )
        scheduler.rollback()


def test_provisioning_removes_unqualified_column_grants(provisioned_roles):
    from scripts.provision_db_roles import apply_plan, build_plan

    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as admin:
        admin.execute("GRANT UPDATE(balance_micros) ON wallets TO xcelsior_readonly")
        assert admin.execute(
            "SELECT has_column_privilege('xcelsior_readonly','wallets','balance_micros','UPDATE')"
        ).fetchone() == (True,)
        apply_plan(admin, build_plan(admin, login_services=set(), passwords={}))
        assert admin.execute(
            "SELECT has_column_privilege('xcelsior_readonly','wallets','balance_micros','UPDATE')"
        ).fetchone() == (False,)
        admin.execute("SET LOCAL ROLE xcelsior_readonly")
        with pytest.raises(InsufficientPrivilege):
            admin.execute("UPDATE wallets SET balance_micros=balance_micros WHERE false")
        admin.rollback()


def test_scheduler_column_exception_does_not_hide_extra_authority(provisioned_roles):
    from control_plane.db_role_audit import role_violations
    from scripts.provision_db_roles import apply_plan, build_plan

    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as admin:
        assert role_violations(admin, "scheduler", check_session=False) == []
        admin.execute("GRANT UPDATE(state) ON serverless_workers TO xcelsior_scheduler")
        assert any(
            "serverless_workers" in issue
            for issue in role_violations(admin, "scheduler", check_session=False)
        )
        apply_plan(admin, build_plan(admin, login_services=set(), passwords={}))
        assert role_violations(admin, "scheduler", check_session=False) == []
        for column in ["attempt_id", "host_id", "allocated_at", "updated_at"]:
            assert admin.execute(
                "SELECT has_column_privilege('xcelsior_scheduler','serverless_workers',%s,'UPDATE')",
                (column,),
            ).fetchone() == (True,)
        assert admin.execute(
            "SELECT has_column_privilege('xcelsior_scheduler','serverless_workers','state','UPDATE')"
        ).fetchone() == (False,)
        admin.rollback()


def test_strict_maintenance_startup_registers_tasks_without_runtime_ddl(
    restricted_logins, monkeypatch
):
    import signal
    import threading
    from types import SimpleNamespace
    from psycopg.types.json import Jsonb
    import bg_worker
    from control_plane import scheduled_tasks
    from control_plane.task_identities import DDL_TASKS, TASK_SERVICES

    names = sorted(set(TASK_SERVICES) | set(DDL_TASKS))
    admin_dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(admin_dsn) as admin:
        before = [
            row[0]
            for row in admin.execute(
                "SELECT to_jsonb(t) FROM scheduled_tasks t WHERE task_name=ANY(%s)", (names,)
            ).fetchall()
        ]
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(bg_worker, "_stop", stop)
    monkeypatch.setattr(
        bg_worker,
        "signal",
        SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=lambda *args: None),
    )
    monkeypatch.setattr(scheduled_tasks, "_registry", {})
    monkeypatch.setattr(scheduled_tasks, "_task_services", {})
    monkeypatch.setenv("XCELSIOR_EXTERNAL_PARTITION_MAINTENANCE", "1")
    monkeypatch.setenv("XCELSIOR_BG_TASKS", "false")
    try:
        with service_scope("reconciler"):
            bg_worker.main()
        assert scheduled_tasks._registry
        assert all(task in scheduled_tasks._task_services for task in scheduled_tasks._registry)
        with psycopg.connect(admin_dsn) as admin:
            assert admin.execute(
                "SELECT enabled FROM scheduled_tasks WHERE task_name='audit_partition_maintenance'"
            ).fetchone() == (False,)
            assert admin.execute(
                "SELECT enabled FROM scheduled_tasks WHERE task_name='telemetry_partition_maintenance'"
            ).fetchone() == (False,)
        assert scheduled_tasks._task_services["billing_cycle"] == "billing"
        assert scheduled_tasks._task_services["billing_control_cycle"] == "api"
        assert scheduled_tasks._task_services["stripe_meter_outbox"] == "billing"
        assert scheduled_tasks._task_services["projection_deliver_audit_log"] == "projector"
    finally:
        with psycopg.connect(admin_dsn) as admin:
            admin.execute("DELETE FROM scheduled_tasks WHERE task_name=ANY(%s)", (names,))
            if before:
                admin.execute(
                    "INSERT INTO scheduled_tasks SELECT * FROM jsonb_populate_recordset(NULL::scheduled_tasks,%s)",
                    (Jsonb(before),),
                )
