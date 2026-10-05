"""The public status page must be able to read the status it publishes.

`GET /api/status` documented itself as "intentionally unauthenticated (the
wizard calls this before sign-in)". It was never in `PUBLIC_PATHS`, so
`TokenAuthMiddleware` answered 401 to every anonymous caller. The status page
renders any failure as:

    Status check unreachable — We could not reach the status endpoint from your
    browser … treat it as unknown rather than healthy.

which is what it showed every visitor on 2026-10-05 while production was
healthy. Found by driving real production with Playwright, not by any test
here — the in-process suite runs with `AUTH_REQUIRED` off, where the
middleware never runs, so the 401 could not appear.

Two conditions came with making it public. Its detail strings carry raw
exception text, so anonymous callers get state only. And public paths skip the
rate limiter, while one computation runs a DB query, an auth-cache probe and an
HTTP probe of the MCP process with a 3s timeout — so anonymous callers get a
short-lived cached answer.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

import routes.health as health  # noqa: E402
from routes._deps import PUBLIC_PATHS, PUBLIC_PATH_PREFIXES  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    health._public_status_cache.update(at=0.0, payload=None)
    yield
    health._public_status_cache.update(at=0.0, payload=None)


class _Req:
    def __init__(self):
        self.headers = {}
        self.cookies = {}


def _fake_probe(detail="unreachable: connection to server at \"10.0.0.5\", port 5432, user \"xcelsior\""):
    return {
        "ok": True,
        "verdict": "degraded",
        "services": [
            {"name": "API", "state": "operational", "detail": "Control plane reachable", "required": True},
            {"name": "Database", "state": "down", "detail": detail, "required": True},
        ],
    }


def test_the_path_is_reachable_without_a_token() -> None:
    """The half the in-process suite could never see: the middleware's allowlist."""
    assert "/api/status" in PUBLIC_PATHS or "/api/status".startswith(PUBLIC_PATH_PREFIXES)


def test_an_anonymous_caller_gets_state_and_no_detail(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(health, "_get_current_user", lambda _r: None)
    monkeypatch.setattr(health, "_compute_service_status", _fake_probe)
    body = health.service_status(_Req())
    assert [s["state"] for s in body["services"]] == ["operational", "down"]
    assert body["verdict"] == "degraded"
    leaked = [s["detail"] for s in body["services"] if s["detail"]]
    assert not leaked, f"exception text reached an anonymous caller: {leaked}"


def test_an_admin_still_gets_the_detail(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(health, "_get_current_user", lambda _r: {"is_admin": True, "role": "admin"})
    monkeypatch.setattr(health, "_is_platform_admin", lambda u: True)
    monkeypatch.setattr(health, "_compute_service_status", _fake_probe)
    body = health.service_status(_Req())
    assert "10.0.0.5" in body["services"][1]["detail"]


def test_a_burst_of_anonymous_requests_costs_one_probe(monkeypatch) -> None:  # noqa: ANN001
    """Public paths are not rate-limited; this is what bounds the work instead."""
    calls = []
    monkeypatch.setattr(health, "_get_current_user", lambda _r: None)
    monkeypatch.setattr(health, "_compute_service_status", lambda: calls.append(1) or _fake_probe())
    for _ in range(50):
        health.service_status(_Req())
    assert len(calls) == 1, f"{len(calls)} probes for 50 anonymous requests"


def test_the_cached_answer_expires(monkeypatch) -> None:  # noqa: ANN001
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(health, "_get_current_user", lambda _r: None)
    monkeypatch.setattr(health, "_compute_service_status", lambda: calls.append(1) or _fake_probe())
    monkeypatch.setattr(health.time, "monotonic", lambda: clock[0])
    health.service_status(_Req())
    clock[0] += health._PUBLIC_STATUS_TTL_SEC + 1
    health.service_status(_Req())
    assert len(calls) == 2, "a stale status was served past its TTL"


def test_the_real_handler_answers_an_anonymous_request() -> None:
    """End to end through the app, unauthenticated, real probes."""
    from fastapi.testclient import TestClient

    from api import app

    response = TestClient(app).get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in {"operational", "degraded", "blocked"}
    assert all(s["detail"] == "" for s in body["services"]), body["services"]
