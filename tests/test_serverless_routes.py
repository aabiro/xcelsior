"""Smoke coverage for routes/serverless.py management + queue API."""

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore
from serverless.repo import ServerlessRepo, WORKER_STATE_READY

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _register_and_fund() -> dict:
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    deps._USE_PERSISTENT_AUTH = True
    auth._USE_PERSISTENT_AUTH = True
    api_mod._USE_PERSISTENT_AUTH = True

    email = f"slvr-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Slvr Cov"},
    )
    assert reg.status_code == 200, reg.text[:300]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    customer_id = user["customer_id"]
    assert UserStore.get_user(email) is not None
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "serverless route test credits"},
        headers=headers,
    )
    assert deposit.status_code == 200, deposit.text[:200]
    return headers


@pytest.fixture(scope="module")
def user_headers():
    return _register_and_fund()


@pytest.fixture(scope="module")
def endpoint_id(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": "test-llama",
            "mode": "preset",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
            "queue_timeout_sec": 1,
            "request_timeout_sec": 10,
        },
    )
    assert r.status_code == 200, r.text[:300]
    return r.json()["endpoint"]["endpoint_id"]


def test_serverless_list_without_session_token():
    """In test env anonymous access is allowed when AUTH_REQUIRED=false."""
    r = client.get("/api/v2/serverless/endpoints")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_serverless_list(user_headers):
    r = client.get("/api/v2/serverless/endpoints", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_serverless_get_endpoint(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json()["endpoint"]["endpoint_id"] == endpoint_id
    assert r.json()["endpoint"]["invoke_path"].endswith("/test-llama")


def test_serverless_endpoint_health(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}/health",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "health" in r.json()


def test_serverless_endpoint_usage(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}/usage",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "usage" in r.json()


def test_serverless_get_not_found(user_headers):
    r = client.get(
        "/api/v2/serverless/endpoints/sep-nonexistent",
        headers=user_headers,
    )
    assert r.status_code == 404


def test_serverless_create_patch_execution_mode_fields(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": f"mode-{uuid.uuid4().hex[:6]}",
            "mode": "preset",
            "model_name": "Qwen/Qwen3-8B",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
            "execution_mode": "async",
            "queue_timeout_sec": 77,
        },
    )
    assert r.status_code == 200, r.text[:300]
    ep = r.json()["endpoint"]
    assert ep["execution_mode"] == "async"
    assert ep["queue_timeout_sec"] == 77

    p = client.patch(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}",
        headers=user_headers,
        json={"execution_mode": "sync", "queue_timeout_sec": 180},
    )
    assert p.status_code == 200, p.text[:300]
    assert p.json()["endpoint"]["execution_mode"] == "sync"
    assert p.json()["endpoint"]["queue_timeout_sec"] == 180
    client.delete(f"/api/v2/serverless/endpoints/{ep['endpoint_id']}", headers=user_headers)


def test_serverless_warm_endpoint_returns_status(user_headers, endpoint_id, monkeypatch):
    import routes.serverless as serverless_routes

    class FakeService:
        def warm_endpoint(self, endpoint_id_arg: str, *args, **kwargs) -> dict:
            return {
                "endpoint_id": endpoint_id_arg,
                "state": "starting",
                "ready_count": 0,
                "booting_count": 1,
                "active_count": 1,
                "workers": [],
            }

    monkeypatch.setattr(serverless_routes, "_svc", lambda: FakeService())
    r = client.post(
        f"/api/v2/serverless/endpoints/{endpoint_id}/warm",
        headers=user_headers,
    )
    assert r.status_code == 200, r.text[:300]
    assert r.json()["warm"]["state"] == "starting"


def test_serverless_worker_telemetry_normalizes_bars(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": f"telemetry-{uuid.uuid4().hex[:6]}",
            "mode": "preset",
            "model_name": "Qwen/Qwen3-8B",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
        },
    )
    assert r.status_code == 200, r.text[:300]
    endpoint_id = r.json()["endpoint"]["endpoint_id"]
    repo = ServerlessRepo()
    worker = repo.create_worker(endpoint_id, scheduler_job_id=f"job-{uuid.uuid4().hex[:8]}")
    worker = repo.update_worker(
        str(worker["worker_id"]),
        state=WORKER_STATE_READY,
        host_id="host-telemetry-test",
    )
    assert worker is not None

    import routes.agent as agent_routes

    agent_routes._host_telemetry["host-telemetry-test"] = {
        "received_at": time.time(),
        "metrics": {
            "utilization": 42,
            "memory_used_gb": 20,
            "memory_total_gb": 80,
            "cpu_util_pct": 17,
            "system_memory_pct": 63,
        },
    }
    try:
        tr = client.get(
            f"/api/v2/serverless/endpoints/{endpoint_id}/workers/{worker['worker_id']}/telemetry",
            headers=user_headers,
        )
        assert tr.status_code == 200, tr.text[:300]
        telemetry = tr.json()["telemetry"]
        assert telemetry["gpu_util_pct"] == 42
        assert telemetry["gpu_memory_pct"] == 25
        assert telemetry["cpu_util_pct"] == 17
        assert telemetry["system_memory_pct"] == 63
        assert telemetry["stale"] is False
    finally:
        agent_routes._host_telemetry.pop("host-telemetry-test", None)
        client.delete(f"/api/v2/serverless/endpoints/{endpoint_id}", headers=user_headers)


def test_serverless_run_job(user_headers, endpoint_id):
    r = client.post(
        f"/v1/serverless/{endpoint_id}/run",
        headers=user_headers,
        json={"input": {"prompt": "hello"}},
    )
    assert r.status_code == 200
    assert r.json().get("status") == "IN_QUEUE"
    assert r.json().get("id")
    assert r.json().get("warm")



def test_serverless_openai_models(user_headers, endpoint_id):
    r = client.get(
        f"/v1/serverless/{endpoint_id}/openai/v1/models",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("object") == "list"

    vanity = client.get(
        f"/v1/serverless/{endpoint_id}/test-llama/openai/v1/models",
        headers=user_headers,
    )
    assert vanity.status_code == 200
    assert vanity.json().get("object") == "list"


def test_dashboard_test_job_is_rate_limited_and_billing_exempt(
    user_headers, endpoint_id, monkeypatch
):
    import routes.serverless as serverless_routes

    class FakeDispatcher:
        def dispatch_for_endpoint(self, _ep):
            return None

    class FakeService:
        dispatcher = FakeDispatcher()

        def log_job_enqueued(self, *args, **kwargs):
            return None

        def warm_endpoint(self, endpoint_id_arg: str, *, billable: bool = True) -> dict:
            assert billable is False
            return {"endpoint_id": endpoint_id_arg, "state": "starting", "workers": []}

    monkeypatch.setattr(serverless_routes, "_svc", lambda: FakeService())
    r = client.post(
        f"/api/v2/serverless/endpoints/{endpoint_id}/test/run",
        headers=user_headers,
        json={"input": {"prompt": "dashboard test"}},
    )
    assert r.status_code == 200, r.text[:300]
    assert r.headers["x-ratelimit-limit"] == "10"
    body = r.json()
    assert body["billing_exempt"] is True
    job = ServerlessRepo().get_job(body["id"], endpoint_id=endpoint_id)
    assert job is not None
    assert job["billing_exempt"] is True


def test_serverless_endpoint_with_lora_adapters(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": "test-llama-lora",
            "mode": "preset",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
            "lora_adapters": [
                {"name": "sql-helper", "source": "acme/llama3-sql-lora"},
            ],
        },
    )
    assert r.status_code == 200, r.text[:300]
    ep = r.json()["endpoint"]
    assert ep["lora_adapters"] == [{"name": "sql-helper", "source": "acme/llama3-sql-lora"}]
    assert "--enable-lora" in ep["startup_command"]
    assert "sql-helper=acme/llama3-sql-lora" in ep["startup_command"]

    models = client.get(
        f"/v1/serverless/{ep['endpoint_id']}/openai/v1/models",
        headers=user_headers,
    )
    assert models.status_code == 200
    ids = {m["id"] for m in models.json()["data"]}
    assert ids == {"meta-llama/Llama-3.1-8B-Instruct", "sql-helper"}

    client.delete(f"/api/v2/serverless/endpoints/{ep['endpoint_id']}", headers=user_headers)


def test_serverless_openai_chat_requires_worker(user_headers, endpoint_id):
    r = client.post(
        f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
        headers=user_headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code in (503, 502, 504)
    assert r.status_code != 500


def test_serverless_delete_endpoint(user_headers, endpoint_id):
    r = client.delete(
        f"/api/v2/serverless/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
