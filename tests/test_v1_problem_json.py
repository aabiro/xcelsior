"""Track B B2.8 — RFC 9457 (application/problem+json) errors on the v1 surface.

Every typed control-plane failure on `/api/v1/*` is a problem document with the
full field set (§18.5): `type`, `title`, `status`, `detail`, `code`,
`retryable`, `retry_after_ms`, `trace_id`, `errors`. The legacy API shape is
untouched (the handler is scoped to typed errors), which a companion assertion
here pins.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from routes.problem import PROBLEM_MEDIA_TYPE, ProblemException, problem_response

client = TestClient(app)

_REQUIRED_FIELDS = {
    "type",
    "title",
    "status",
    "detail",
    "code",
    "retryable",
    "retry_after_ms",
    "trace_id",
    "errors",
}


def _token(email: str) -> str:
    reg = client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()
    return reg["access_token"]


# ── Unit: the builder produces a complete, correctly-typed problem ──────


def test_problem_response_has_all_rfc9457_fields():
    resp = problem_response(status=409, code="plan_revoked", detail="a revoked plan")
    assert resp.media_type == PROBLEM_MEDIA_TYPE
    import json

    body = json.loads(bytes(resp.body))
    assert set(body) >= _REQUIRED_FIELDS
    assert body["status"] == 409
    assert body["code"] == "plan_revoked"
    assert body["type"].endswith("/plan_revoked")
    assert isinstance(body["retryable"], bool)
    assert isinstance(body["trace_id"], str) and body["trace_id"]


def test_retryable_codes_are_marked_and_retry_after_sets_header():
    retry = problem_response(
        status=429, code="rate_limited", detail="slow down", retry_after_ms=2000
    )
    import json

    body = json.loads(bytes(retry.body))
    assert body["retryable"] is True
    assert retry.headers["Retry-After"] == "2"
    # A non-retryable code stays false unless explicitly overridden.
    other = problem_response(status=409, code="plan_revoked", detail="x")
    assert json.loads(bytes(other.body))["retryable"] is False


def test_problem_exception_round_trips():
    exc = ProblemException(status=402, code="insufficient_funds", detail="broke")
    import json

    body = json.loads(bytes(exc.to_response().body))
    assert body["status"] == 402 and body["code"] == "insufficient_funds"


# ── HTTP: typed control-plane errors surface as problem+json ────────────


def test_approve_unknown_plan_is_problem_json_404():
    token = _token("b28-404@xcelsior.ca")
    resp = client.post(
        f"/api/v1/launch-plans/{uuid.uuid4()}/approve",
        headers={"Authorization": f"Bearer {token}"},
        json={"confirm": True},
    )
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = resp.json()
    assert set(body) >= _REQUIRED_FIELDS
    assert body["code"] == "plan_not_found"
    assert body["status"] == 404
    assert body["trace_id"]


def test_execute_unapproved_plan_is_problem_json_409():
    token = _token("b28-409@xcelsior.ca")
    headers = {"Authorization": f"Bearer {token}"}
    # Create a plan (quoted, not approved) …
    created = client.post(
        "/api/v1/launch-plans",
        headers=headers,
        json={"name": "b28", "interactive": True},
    )
    assert created.status_code == 200, created.text
    plan_id = created.json()["plan_id"]
    try:
        # … executing an unapproved plan is a typed 409 problem.
        resp = client.post(f"/api/v1/launch-plans/{plan_id}/execute", headers=headers)
        assert resp.status_code == 409
        assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
        body = resp.json()
        assert body["code"] == "plan_not_approved"
        assert set(body) >= _REQUIRED_FIELDS
    finally:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (plan_id,))
            conn.commit()


def test_legacy_error_shape_is_unchanged():
    """A non-v1 404 keeps the legacy {"ok": false, "error": {...}} envelope —
    the problem handler must not leak onto the rest of the API."""
    resp = client.get("/instance/does-not-exist-xyz")
    assert resp.status_code == 404
    assert not resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert resp.json().get("ok") is False
