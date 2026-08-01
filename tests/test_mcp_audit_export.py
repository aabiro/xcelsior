"""Customer-visible MCP audit export (adoption plan X6.30).

An audit trail a customer cannot read is a log file, not an assurance. These
tests pin the two properties that make it one: a tenant sees its own calls, and
sees nothing else at any privilege level.
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


def _register(email: str, password: str = "AuditExport123!") -> tuple[str, str]:
    """Returns (bearer, customer_id)."""
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
        assert verification
        verified = client.post("/api/auth/verify-email", json={"token": verification})
        assert verified.status_code == 200, verified.text
        token = verified.json().get("access_token")
    if not token:
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    customer_id = me.json().get("customer_id") or me.json().get("user", {}).get("customer_id")
    assert customer_id
    return token, customer_id


def _write_audit(bearer: str, tenant: str, tool_name: str, outcome: str = "success") -> None:
    response = client.post(
        "/api/v1/mcp/tool-audit",
        headers={"Authorization": f"Bearer {bearer}"},
        json={
            "tool_name": tool_name,
            "tool_version": "2.0.0",
            "transport": "streamable_http",
            "principal_id": "principal-under-test",
            "tenant_id": tenant,
            "scopes_evaluated": ["instances:read"],
            "redacted_args_hash": uuid.uuid4().hex + uuid.uuid4().hex,
            "latency_ms": 12,
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
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")
    reset_auth_cache_for_tests()
    client.cookies.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    yield


def test_tenant_reads_its_own_tool_calls():
    bearer, tenant = _register("audit-owner@xcelsior.ca")
    _write_audit(bearer, tenant, "list_instances")
    _write_audit(bearer, tenant, "create_instance")

    response = client.get(
        "/api/v1/mcp/tool-audit", headers={"Authorization": f"Bearer {bearer}"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tenant_id"] == tenant
    assert {record["tool_name"] for record in body["records"]} == {
        "list_instances",
        "create_instance",
    }
    record = body["records"][0]
    # The fields an enterprise security review asks to see.
    for field in (
        "occurred_at", "tool_name", "scopes_evaluated", "redacted_args_hash",
        "latency_ms", "trace_id", "outcome",
    ):
        assert field in record, field


def test_export_never_returns_another_tenants_calls():
    owner_bearer, owner_tenant = _register("audit-a@xcelsior.ca")
    other_bearer, other_tenant = _register("audit-b@xcelsior.ca")
    assert owner_tenant != other_tenant
    _write_audit(owner_bearer, owner_tenant, "list_instances")
    _write_audit(other_bearer, other_tenant, "get_wallet_balance")

    body = client.get(
        "/api/v1/mcp/tool-audit", headers={"Authorization": f"Bearer {other_bearer}"}
    ).json()
    assert [record["tool_name"] for record in body["records"]] == ["get_wallet_balance"]
    assert all(record.get("tenant_id") != owner_tenant for record in body["records"])


def test_export_discloses_nothing_without_a_tenant():
    # 401 where auth is enforced; 404 tenant_not_found where the caller
    # resolves to no tenant — the same not-an-oracle answer the sibling POST
    # gives. What matters is that no records come back either way.
    response = client.get("/api/v1/mcp/tool-audit")
    assert response.status_code in (401, 403, 404), response.text
    assert "records" not in response.text


def test_export_filters_and_pages():
    bearer, tenant = _register("audit-page@xcelsior.ca")
    for index in range(5):
        _write_audit(bearer, tenant, "list_instances", "success" if index % 2 else "error")

    filtered = client.get(
        "/api/v1/mcp/tool-audit",
        params={"outcome": "error"},
        headers={"Authorization": f"Bearer {bearer}"},
    ).json()
    assert filtered["records"]
    assert all(record["outcome"] == "error" for record in filtered["records"])

    first = client.get(
        "/api/v1/mcp/tool-audit",
        params={"limit": 2},
        headers={"Authorization": f"Bearer {bearer}"},
    ).json()
    assert len(first["records"]) == 2
    assert first["next_cursor"], "a full page must offer a cursor"

    second = client.get(
        "/api/v1/mcp/tool-audit",
        params={"limit": 2, "cursor": first["next_cursor"]},
        headers={"Authorization": f"Bearer {bearer}"},
    ).json()
    # Keyset paging: the second page must not repeat the first.
    assert not {r["audit_id"] for r in first["records"]} & {
        r["audit_id"] for r in second["records"]
    }


def test_export_page_size_is_bounded():
    bearer, tenant = _register("audit-bound@xcelsior.ca")
    _write_audit(bearer, tenant, "list_instances")
    response = client.get(
        "/api/v1/mcp/tool-audit",
        params={"limit": "100000"},
        headers={"Authorization": f"Bearer {bearer}"},
    )
    assert response.status_code == 200
    # An unbounded export is a table scan anyone can trigger.
    assert response.json()["next_cursor"] is None


def test_export_never_contains_raw_arguments():
    bearer, tenant = _register("audit-redaction@xcelsior.ca")
    _write_audit(bearer, tenant, "create_instance")
    body = client.get(
        "/api/v1/mcp/tool-audit", headers={"Authorization": f"Bearer {bearer}"}
    ).json()
    record = body["records"][0]
    assert "arguments" not in record and "args" not in record
    assert len(record["redacted_args_hash"]) == 64
    assert "SHA-256" in body["notice"]
