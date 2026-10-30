"""A persistent account suspension must defeat new IPs and all token types."""

import time
import uuid
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

import abuse_policy
import oauth_service as oauth
from db import get_engine
from tests._db_helpers import PRICED_GPU, fund_test_owner


@pytest.fixture
def suspended():
    uid = "security-test-" + uuid.uuid4().hex
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_account_suspension(
            conn, user_id=uid, active=True, actor="admin-test", reason="test", backend=backend
        )
    oauth.reset_auth_cache_for_tests()
    yield uid
    with get_engine().transaction() as (conn, backend):
        conn.execute("DELETE FROM state WHERE namespace = %s", (f"security.suspended.{uid}",))
    oauth.reset_auth_cache_for_tests()


def test_cached_bearer_stops_without_waiting_for_expiry(suspended):
    for uid in (suspended, "unrelated-account"):
        token = "xoa_" + uid
        oauth._cache_set_json(
            "access_token", token, {"user_id": uid, "expires_at": time.time() + 3600}, 3600
        )
    assert oauth.resolve_opaque_access_token("xoa_" + suspended) is None
    assert oauth.resolve_opaque_access_token("xoa_unrelated-account") is not None


def test_new_login_and_agent_key_cannot_bypass_hold(suspended, monkeypatch):
    store = Mock()
    monkeypatch.setattr(oauth.UserStore, "create_session", store)
    for mint in (
        lambda: oauth.issue_user_tokens({"user_id": suspended}, Mock()),
        lambda: oauth.issue_agent_api_key(user={"user_id": suspended}, client_id="test", scopes=[]),
        lambda: oauth._issue_opaque_access_token(
            {"user_id": suspended},
            client_id="test",
            scopes=[],
            session_token="new",
            session_type="browser",
        ),
    ):
        with pytest.raises(oauth.OAuthGrantError) as exc:
            mint()
        assert exc.value.status_code == 403
    store.assert_not_called()


def test_legacy_session_uses_fresh_account_policy(suspended):
    from routes._deps import _merge_auth_user

    with pytest.raises(HTTPException) as exc:
        _merge_auth_user({"user_id": suspended, "role": "submitter"})
    assert exc.value.status_code == 403


def test_machine_client_creator_and_workspace_are_checked(suspended):
    assert abuse_policy.account_suspended({"created_by_email": suspended})
    assert abuse_policy.account_suspended({"workspace_customer_id": suspended})
    assert abuse_policy.account_suspended({"client_id": suspended})
    assert abuse_policy.account_suspended({"sub": suspended})
    assert not abuse_policy.account_suspended({"created_by_email": "someone-else"})


def test_store_failure_does_not_become_permission_to_mint(monkeypatch):
    monkeypatch.setattr(abuse_policy, "get_engine", Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError):
        oauth.issue_user_tokens({"user_id": "must-check"}, Mock())


def test_account_suspended_handles_string_and_invalid():
    assert not abuse_policy.account_suspended(None)
    assert not abuse_policy.account_suspended("")
    assert not abuse_policy.account_suspended([])


def test_account_suspended_accepts_direct_string(suspended):
    assert abuse_policy.account_suspended(suspended)


def test_suspend_and_unsuspend_account_links(monkeypatch):
    fake_user = {
        "email": "miner-link-test@example.com",
        "user_id": "usr-link-test-123",
        "customer_id": "cust-link-test-456",
    }
    from db import UserStore

    monkeypatch.setattr(
        UserStore, "get_user", lambda email: fake_user if email == fake_user["email"] else None
    )
    monkeypatch.setattr(
        UserStore, "get_user_by_id", lambda uid: fake_user if uid == fake_user["user_id"] else None
    )
    monkeypatch.setattr(
        UserStore,
        "get_user_by_customer_id",
        lambda cid: fake_user if cid == fake_user["customer_id"] else None,
    )

    suspended_ids = abuse_policy.suspend_account(
        identifier="miner-link-test@example.com",
        actor="security_team",
        reason="incident_containment",
    )
    assert set(suspended_ids) == {
        "miner-link-test@example.com",
        "usr-link-test-123",
        "cust-link-test-456",
    }

    try:
        assert abuse_policy.account_suspended("usr-link-test-123")
        assert abuse_policy.account_suspended("cust-link-test-456")
        assert abuse_policy.account_suspended({"email": "miner-link-test@example.com"})

        # Unsuspend
        abuse_policy.unsuspend_account(
            identifier="miner-link-test@example.com",
            actor="security_team",
            reason="cleared",
        )
        assert not abuse_policy.account_suspended("usr-link-test-123")
        assert not abuse_policy.account_suspended("cust-link-test-456")
    finally:
        with get_engine().transaction() as (conn, backend):
            for uid in suspended_ids:
                conn.execute(
                    "DELETE FROM state WHERE namespace = %s", (f"security.suspended.{uid}",)
                )


def test_scheduler_rejects_suspended_account(suspended):
    from scheduler import submit_job

    with pytest.raises(ValueError, match="suspended by security policy"):
        submit_job(name="test-hold", vram_needed_gb=1, gpu_model=PRICED_GPU, owner=suspended)


def test_requeue_rejects_suspended_account(suspended):
    from scheduler import requeue_job, submit_job, get_job

    # Normal user job can be submitted
    job = submit_job(
        name="test-requeue",
        vram_needed_gb=1,
        gpu_model=PRICED_GPU,
        owner=fund_test_owner("normal-user"),
    )
    job_id = job["job_id"]
    try:
        # Status and owner are authority columns a payload patch may not set.
        from psycopg.types.json import Jsonb

        with get_engine().transaction() as (conn, backend):
            conn.execute(
                "UPDATE jobs SET status='failed', owner_id=%s, payload=payload||%s WHERE job_id=%s",
                (suspended, Jsonb({"status": "failed", "owner": suspended}), job_id),
            )
        result = requeue_job(job_id, user_initiated=True)
        assert result is None
    finally:
        with get_engine().transaction() as (conn, backend):
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def test_submit_instance_rejects_suspended_account(suspended, monkeypatch):
    from routes.instances import api_submit_instance, JobIn
    from unittest.mock import MagicMock

    req = MagicMock()
    monkeypatch.setattr(
        "routes.instances._require_auth",
        lambda r: {"user_id": suspended, "email": "susp@example.com"},
    )
    monkeypatch.setattr("routes.instances._require_scope", lambda u, s: None)
    monkeypatch.setattr("routes.instances._require_team_instance_write", lambda u: None)

    job_in = JobIn(name="test-susp-launch", vram_needed_gb=0)
    with pytest.raises(HTTPException) as exc:
        api_submit_instance(job_in, req)
    assert exc.value.status_code == 403
    assert "suspended" in exc.value.detail.lower()


def test_authorize_instance_mutation_rejects_suspended_account_but_permits_admin(
    suspended, monkeypatch
):
    from routes.instances import _authorize_instance_mutation
    from unittest.mock import MagicMock

    req = MagicMock()
    fake_job = {"job_id": "test-mut-job", "owner": suspended, "status": "running"}
    monkeypatch.setattr("scheduler.get_job", lambda jid: fake_job)
    monkeypatch.setattr("routes.instances._require_scope", lambda u, s: None)

    # 1. Suspended caller is forbidden
    monkeypatch.setattr(
        "routes.instances._require_auth",
        lambda r: {"user_id": suspended, "email": "susp@example.com"},
    )
    with pytest.raises(HTTPException) as exc:
        _authorize_instance_mutation(req, "test-mut-job", "stop")
    assert exc.value.status_code == 403

    # 2. Platform admin can mutate a suspended account's instance
    admin_user = {
        "user_id": "admin-1",
        "email": "admin@xcelsior.ca",
        "is_admin": True,
        "role": "admin",
    }
    monkeypatch.setattr("routes.instances._require_auth", lambda r: admin_user)
    u, j = _authorize_instance_mutation(req, "test-mut-job", "stop")
    assert u == admin_user
    assert j == fake_job


def test_validate_ws_auth_rejects_suspended_account(suspended, monkeypatch):
    from routes._deps import _validate_ws_auth
    from unittest.mock import MagicMock

    ws = MagicMock()
    ws.cookies = {}
    ws.query_params = {"token": "xoa_susp_ws"}

    # Mock UserStore
    monkeypatch.setattr("routes._deps._USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr("routes._deps.AUTH_REQUIRED", True)
    monkeypatch.setattr(
        "routes._deps.UserStore.get_session",
        lambda tok: {"email": "susp_ws@example.com", "user_id": suspended},
    )
    monkeypatch.setattr(
        "routes._deps.UserStore.get_user",
        lambda email: {
            "email": "susp_ws@example.com",
            "user_id": suspended,
            "customer_id": suspended,
        },
    )

    assert _validate_ws_auth(ws) is None
