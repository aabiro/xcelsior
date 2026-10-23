"""Attempt credentials work through real middleware and lose authority on exit."""

import hashlib
import os

import pytest
from fastapi.testclient import TestClient

from control_plane.db import run_transaction
from control_plane.leases import renew_lease
from control_plane.serverless_credentials import authenticate_callback
from tests.test_serverless_native_lifecycle import native, report  # noqa: F401


def claim_callback(x, client):
    r = x.reserve()
    response = client.post(
        "/agent/v2/leases/claim",
        json={
            "lease_id": r.lease_id,
            "job_id": r.job_id,
            "attempt_id": r.attempt_id,
            "host_id": r.host_id,
            "fencing_token": r.fencing_token,
            "worker_session_id": "callback-proof",
        },
        headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
    )
    assert response.status_code == 200, response.text
    return response.json()["callback_token"]


def test_current_callback_token_is_hash_only_worker_bound_and_not_an_api_principal(
    native, monkeypatch
):
    import api

    monkeypatch.setattr(api, "AUTH_REQUIRED", True)
    with TestClient(api.app) as client:
        token = claim_callback(native, client)
        report(native, "running")
        headers = {"Authorization": f"Bearer {token}"}
        with native.pool.connection() as conn:
            row = conn.execute(
                "SELECT token_hash,tenant_id FROM serverless_attempt_credentials WHERE attempt_id=%s",
                (native.r.attempt_id,),
            ).fetchone()
            assert row[0] == hashlib.sha256(token.encode()).hexdigest()
            assert row[1] == native.owner
            assert authenticate_callback(conn, token, native.wid)
            assert not authenticate_callback(conn, token, "another-worker")
            conn.execute(
                "UPDATE serverless_attempt_credentials SET tenant_id='another-tenant' WHERE attempt_id=%s",
                (native.r.attempt_id,),
            )
            assert not authenticate_callback(conn, token, native.wid)
            conn.rollback()
        assert (
            client.post(
                f"/api/v2/serverless/workers/{native.wid}/heartbeat", headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v2/serverless/workers/another-worker/heartbeat", headers=headers
            ).status_code
            == 401
        )
        assert client.get("/api/billing/wallets", headers=headers).status_code in (401, 403, 404)
        assert client.get("/agent/v2/commands/claim", headers=headers).status_code in (
            401,
            403,
            404,
            405,
        )
        report(native, "failed")
        assert (
            client.post(
                f"/api/v2/serverless/workers/{native.wid}/heartbeat", headers=headers
            ).status_code
            == 401
        )


def test_callback_authority_renews_with_lease_and_fails_at_expiry(native):
    import api

    with TestClient(api.app) as client:
        token = claim_callback(native, client)
    r = native.r
    run_transaction(
        lambda c: renew_lease(
            c,
            lease_id=r.lease_id,
            attempt_id=r.attempt_id,
            host_id=r.host_id,
            fencing_token=r.fencing_token,
            worker_session_id="callback-proof",
        )
    )
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT c.expires_at=l.expires_at FROM serverless_attempt_credentials c JOIN placement_leases l USING(attempt_id) WHERE c.attempt_id=%s",
            (r.attempt_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE serverless_attempt_credentials SET expires_at=clock_timestamp()-interval '1 minute' WHERE attempt_id=%s",
            (r.attempt_id,),
        )
        assert not authenticate_callback(conn, token, native.wid)


@pytest.mark.parametrize("credential", ["", "Bearer invalid", "Bearer master", "Basic xsw_invalid"])
def test_callbacks_have_no_test_mode_or_shared_bearer_bypass(native, credential):
    import api

    if credential == "Bearer master":
        credential = f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"
    with TestClient(api.app) as client:
        response = client.post(
            f"/api/v2/serverless/workers/{native.wid}/heartbeat",
            headers={"Authorization": credential},
        )
    assert response.status_code == 401


def test_credential_pruning_removes_only_expired_terminal_digests(native):
    import api
    from control_plane.serverless_credentials import prune_callback_credentials

    with TestClient(api.app) as client:
        claim_callback(native, client)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE serverless_attempt_credentials SET expires_at=clock_timestamp()-interval '2 days' WHERE attempt_id=%s",
            (native.r.attempt_id,),
        )
        assert prune_callback_credentials(conn) == 0
    report(native, "failed")
    assert run_transaction(prune_callback_credentials) == 1
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM job_attempts WHERE attempt_id=%s", (native.r.attempt_id,)
            ).fetchone()[0]
            == 1
        )


def test_worker_container_receives_only_the_attempt_credential(monkeypatch):
    import worker_agent

    monkeypatch.setattr(worker_agent, "API_TOKEN", "host-wide-secret")
    env = worker_agent.build_platform_env(
        {
            "job_type": "serverless_worker",
            "serverless_worker_id": "worker",
            "_serverless_callback_token": "xsw_scoped-proof",
        }
    )
    assert env["XCELSIOR_API_TOKEN"] == ""
    assert env["XCELSIOR_WORKER_CALLBACK_TOKEN"] == "xsw_scoped-proof"
    assert "host-wide-secret" not in env.values()


def test_sdk_does_not_fall_back_to_a_host_or_platform_token(monkeypatch):
    from serverless.worker_sdk.client import WorkerClient

    monkeypatch.delenv("XCELSIOR_WORKER_CALLBACK_TOKEN", raising=False)
    monkeypatch.setenv("XCELSIOR_API_TOKEN", "host-wide-secret")
    client = WorkerClient(worker_id="worker")
    try:
        assert "Authorization" not in client._headers()
    finally:
        client.close()


@pytest.mark.parametrize(
    "revision,compatible",
    [("125", False), ("126", False), ("127", False), ("131", False), ("132", False), ("133", True)],
)
def test_runtime_requires_the_current_native_history_floor(revision, compatible, monkeypatch):
    import control_plane.schema_compat as schema

    monkeypatch.delenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", raising=False)
    monkeypatch.delenv("XCELSIOR_DB_SCHEMA_MAX_REVISION", raising=False)
    monkeypatch.setattr(schema, "current_revision", lambda conn: revision)
    assert schema.check_schema_compatible(None).compatible is compatible
