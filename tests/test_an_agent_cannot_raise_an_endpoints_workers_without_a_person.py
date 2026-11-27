"""An agent cannot raise an endpoint's workers without a person.

Creating a serverless endpoint refuses agent keys, MCP connector tokens and any
credential under a spend envelope (`_launch_plan_required`): they create one
only through an endpoint plan their owner approved. `PATCH` did not hold the
same rule, so such a credential could create a one-worker endpoint through an
approved plan and then raise it to 32 always-on workers, billing worker time
nobody approved. Found while building the Terraform provider (2026-11-27).

The handler runs for real; the endpoint lookup and the service are stand-ins,
so the assertions are about the rule and what reaches the service.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

ENDPOINT = {
    "endpoint_id": "ep-patch-gate",
    "owner_id": "demo-user",
    "name": "chat",
    "min_workers": 0,
    "max_workers": 4,
    "idle_timeout_sec": 300,
    "keep_warm": False,
}

AGENT = {
    "email": "demo@xcelsior.ca",
    "user_id": "demo-user",
    "role": "user",
    "auth_type": "agent_api_key",
    "client_id": "client-test",
    "scopes": ["inference:read", "inference:write"],
}

PERSON = {
    "email": "demo@xcelsior.ca",
    "user_id": "demo-user",
    "role": "user",
    "auth_type": "session",
    "session_type": "browser",
}


@pytest.fixture
def patch(monkeypatch):
    from fastapi.testclient import TestClient

    import api as api_mod
    from routes import _deps
    from routes import serverless as route

    applied: list[dict] = []

    class Service:
        def patch_endpoint(self, endpoint_id, owner_id, fields):
            applied.append(dict(fields))
            return {**ENDPOINT, **fields}

    monkeypatch.setattr(route, "_get_endpoint_for_user", lambda endpoint_id, user: dict(ENDPOINT))
    monkeypatch.setattr(route, "_require_serverless_endpoint_write", lambda user, ep: None)
    monkeypatch.setattr(route, "_svc", lambda: Service())
    client = TestClient(api_mod.app)

    def send(principal: dict, body: dict):
        monkeypatch.setattr(_deps, "_get_current_user", lambda request: dict(principal))
        return client.patch(f"/api/v2/serverless/endpoints/{ENDPOINT['endpoint_id']}", json=body)

    send.applied = applied  # type: ignore[attr-defined]
    return send


@pytest.mark.parametrize(
    "body, raised",
    [
        ({"max_workers": 32}, ["max_workers"]),
        ({"min_workers": 2, "keep_warm": True}, ["min_workers", "keep_warm"]),
        ({"idle_timeout_sec": 86400}, ["idle_timeout_sec"]),
    ],
)
def test_an_agent_raising_what_bills_is_sent_to_a_person(patch, body, raised):
    r = patch(AGENT, body)
    assert r.status_code == 403, r.text[:400]
    problem = r.json()
    assert problem.get("code") == "human_required", problem
    assert problem.get("fields") == raised, problem
    assert patch.applied == [], "nothing reached the service"


def test_an_agent_may_still_lower_rename_and_tune(patch):
    body = {"max_workers": 1, "name": "chat-small", "max_concurrency": 8, "keep_warm": False}
    r = patch(AGENT, body)
    assert r.status_code == 200, r.text[:400]
    assert patch.applied == [body]


def test_setting_the_same_value_is_not_a_raise(patch):
    r = patch(AGENT, {"max_workers": 4, "idle_timeout_sec": 300})
    assert r.status_code == 200, r.text[:400]


def test_a_person_in_the_dashboard_raises_workers_directly(patch):
    r = patch(PERSON, {"max_workers": 32, "min_workers": 1})
    assert r.status_code == 200, r.text[:400]
    assert patch.applied == [{"max_workers": 32, "min_workers": 1}]
