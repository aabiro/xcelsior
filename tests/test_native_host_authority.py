"""Admission, listing and SSH cannot recover authority from JSONB aliases."""

import os

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.test_host_admission import host, helper_key  # noqa: F401


def test_marketplace_provider_comes_from_native_host_identity(host):
    import host_admission as ha
    from tests.test_host_admission import TestDecisions, _operator, _version

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"owner": "forged-owner", "provider_id": "forged-provider"}), host),
        )
    TestDecisions()._authorise(host)
    ha.decide_admission(
        host_id=host,
        actor=_operator(),
        action="admit",
        reason="canonical provider proof",
        idempotency_key="canonical-provider",
        expected_version=_version(host),
    )
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT DISTINCT provider_id FROM gpu_offers WHERE host_id=%s AND available", (host,)
        ).fetchall() == [("tenant-a",)]


def test_admission_refuses_payload_only_ownership(host, helper_key):
    import host_admission as ha
    from tests.test_host_admission import _open_session

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE hosts SET tenant_id=NULL,owner_id=NULL,payload=payload||%s WHERE host_id=%s",
            (Jsonb({"owner": "unverified"}), host),
        )
    with pytest.raises(ha.AdmissionPreconditionFailed, match="canonical tenant"):
        _open_session(host, helper_key[1])
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM host_compatibility_sessions WHERE host_id=%s", (host,)
        ).fetchone() == (0,)


@pytest.fixture
def restricted_gateway(provisioned_roles, monkeypatch):
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "0")
    monkeypatch.setenv(
        "XCELSIOR_POSTGRES_DSN_SSH_GATEWAY", make_conninfo(dsn, user="xcelsior_ssh_gateway")
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        from tests._db_helpers import allow_test_login

        allow_test_login(conn, "xcelsior_ssh_gateway", dsn)
    try:
        yield
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("ALTER ROLE xcelsior_ssh_gateway NOLOGIN PASSWORD NULL")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", [None, "expired", "requested_port_only", "suspended", "stopping"]
)
async def test_restricted_gateway_requires_native_route_authority(restricted_gateway, invalid):
    import ssh_gateway

    with running_instance(owner="native-route-owner") as instance:
        job = instance["job_id"]
        namespace = "security.suspended." + job
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE jobs SET owner_id=%s,payload=payload||%s WHERE job_id=%s",
                (
                    job,
                    Jsonb(
                        {
                            "host_id": "forged-host",
                            "owner": "forged-owner",
                            "ssh_port": 22000,
                            "public_ssh_port": 12000 if invalid != "requested_port_only" else None,
                        }
                    ),
                    job,
                ),
            )
            if invalid == "expired":
                conn.execute(
                    "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s",
                    (instance["authority"]["attempt_id"],),
                )
            if invalid == "stopping":
                conn.execute("UPDATE jobs SET desired_state='stopped' WHERE job_id=%s", (job,))
            if invalid == "suspended":
                conn.execute(
                    "INSERT INTO state(namespace,payload) VALUES(%s,'{\"active\":true}')",
                    (namespace,),
                )
        try:
            with pytest.MonkeyPatch.context() as scoped:
                scoped.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
                routes = [
                    route
                    for route in await ssh_gateway.query_running_instances()
                    if route.job_id == job
                ]
            if invalid is None:
                assert len(routes) == 1
                assert routes[0].host_id == instance["host_id"]
                assert routes[0].host_ip == "100.64.1.42" and routes[0].ssh_port == 12000
            else:
                assert routes == []
        finally:
            with _get_pg_pool().connection() as conn:
                conn.execute("DELETE FROM state WHERE namespace=%s", (namespace,))
