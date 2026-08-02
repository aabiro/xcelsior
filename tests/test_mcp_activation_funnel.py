"""Connector activation funnel and per-surface attribution (X7.33/X7.34).

The funnel exists to answer one question honestly: between "a user approved the
connector" and "they ran a paid workload", where do people stop? These tests
pin that the stages are computed from durable records, that a surface is
attributed rather than guessed, and that the endpoint is not a cross-tenant
data leak wearing an analytics hat.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

import scheduler

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app  # noqa: E402

client = TestClient(app)


def _register(email: str, *, admin: bool = False) -> tuple[str, str]:
    password = "FunnelTest123!"
    reg = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg.status_code == 200, reg.text
    body = reg.json()
    token = body.get("access_token")
    if not token and body.get("email_verification_required"):
        import routes._deps as _deps_mod
        from db import auth_connection

        if _deps_mod._USE_PERSISTENT_AUTH:
            with auth_connection() as conn:
                row = conn.execute(
                    "SELECT email_verification_token FROM users WHERE email = %s", (email,)
                ).fetchone()
            verification = row["email_verification_token"] if row else None
        else:
            verification = _deps_mod._users_db.get(email, {}).get("email_verification_token")
        verified = client.post("/api/auth/verify-email", json={"token": verification})
        token = verified.json().get("access_token")
    if not token:
        token = client.post(
            "/api/auth/login", json={"email": email, "password": password}
        ).json()["access_token"]
    if admin:
        import routes._deps as _deps_mod
        from db import UserStore

        # Which store backs auth depends on `_USE_PERSISTENT_AUTH`, and this
        # suite runs with the in-memory one. Update both, then re-login so the
        # token carries the flag — an access token's principal is resolved from
        # the auth cache entry written at issuance, not re-read per request.
        UserStore.update_user(email, {"is_admin": 1, "role": "admin"})
        if email in _deps_mod._users_db:
            _deps_mod._users_db[email].update({"is_admin": 1, "role": "admin"})
        client.cookies.clear()
        token = client.post(
            "/api/auth/login", json={"email": email, "password": password}
        ).json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    body = me.json()
    tenant = body.get("customer_id") or body.get("user", {}).get("customer_id")
    assert tenant, me.text
    return token, str(tenant)


def _grant(tenant: str, surface: str, client_id: str = "xcelsior-connector") -> None:
    from db import ConsentStore

    ConsentStore.record(
        grant_id=f"cg_{uuid.uuid4().hex}",
        tenant_id=tenant,
        user_id=f"user-{tenant}",
        email=f"{tenant}@example.test",
        client_id=client_id,
        scopes=["instances:read"],
        resource="https://mcp.xcelsior.ca/mcp",
        surface=surface,
    )


def _audit(bearer: str, tenant: str, tool: str, outcome: str = "success") -> None:
    response = client.post(
        "/api/v1/mcp/tool-audit",
        headers={"Authorization": f"Bearer {bearer}"},
        json={
            "tool_name": tool,
            "tool_version": "2.0.0",
            "transport": "streamable_http",
            "tenant_id": tenant,
            "scopes_evaluated": ["instances:read"],
            "redacted_args_hash": uuid.uuid4().hex + uuid.uuid4().hex,
            "latency_ms": 5,
            "trace_id": uuid.uuid4().hex,
            "outcome": outcome,
        },
    )
    assert response.status_code == 202, response.text


@pytest.fixture(autouse=True)
def _clean():
    from db import _get_pg_pool, auth_connection
    from oauth_service import reset_auth_cache_for_tests
    import routes._deps as _deps_mod

    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM state")
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM mcp_tool_audit")
        conn.commit()
    with auth_connection() as conn:
        conn.execute("DELETE FROM oauth_consent_grants")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")
    reset_auth_cache_for_tests()
    client.cookies.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    yield


def test_funnel_reports_each_stage_and_attributes_the_surface():
    admin_token, admin_tenant = _register("funnel-admin@xcelsior.ca", admin=True)

    # Connected from Claude, called a tool, and it succeeded.
    _grant(admin_tenant, "claude")
    _audit(admin_token, admin_tenant, "list_instances", "success")

    # Connected from ChatGPT, never called anything.
    _grant("tenant-chatgpt-only", "chatgpt")

    body = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    stages = {stage["stage"]: stage for stage in body["funnel"]}

    assert stages["authorized"]["tenants"] == 2
    assert stages["authorized"]["by_surface"] == {"chatgpt": 1, "claude": 1}
    assert stages["first_tool_call"]["tenants"] == 1
    assert stages["first_tool_call"]["by_surface"] == {"claude": 1}
    assert stages["first_success"]["tenants"] == 1
    # Nobody wrote anything, so the write stage is empty rather than absent.
    assert stages["first_write"]["tenants"] == 0


def test_funnel_names_the_biggest_cliff():
    admin_token, admin_tenant = _register("funnel-cliff@xcelsior.ca", admin=True)
    _grant(admin_tenant, "claude")
    for index in range(4):
        _grant(f"tenant-never-called-{index}", "claude")
    _audit(admin_token, admin_tenant, "list_instances")

    body = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    # The point of the funnel is naming the step that loses people.
    assert body["biggest_drop"]["between"] == "authorized → first_tool_call"
    assert body["biggest_drop"]["tenants_lost"] == 4


def test_write_stage_counts_only_tools_that_change_something():
    admin_token, admin_tenant = _register("funnel-write@xcelsior.ca", admin=True)
    _grant(admin_tenant, "claude")
    _audit(admin_token, admin_tenant, "list_instances")
    body = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    assert {s["stage"]: s["tenants"] for s in body["funnel"]}["first_write"] == 0

    _audit(admin_token, admin_tenant, "create_instance")
    body = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    assert {s["stage"]: s["tenants"] for s in body["funnel"]}["first_write"] == 1


def test_funnel_is_not_readable_by_an_ordinary_tenant():
    # Platform-wide data. A tenant's own activity is the audit export; this
    # endpoint must not become a cross-tenant view with an analytics label.
    tenant_token, tenant_id = _register("funnel-tenant@xcelsior.ca")
    _grant(tenant_id, "claude")
    response = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {tenant_token}"},
    )
    assert response.status_code == 404, response.text
    assert "funnel" not in response.text


def test_funnel_is_explicit_about_what_it_cannot_measure():
    admin_token, _ = _register("funnel-notes@xcelsior.ca", admin=True)
    body = client.get(
        "/api/v1/mcp/activation-funnel",
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    # Discovery hits and 401 challenges are Prometheus counters, not rows —
    # saying so beats reporting a number we would have to invent.
    assert "Prometheus" in body["notes"]["top_of_funnel"]
    assert body["window_days"] == 30
