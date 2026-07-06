"""Captures usage/health API sample for goal verification evidence."""

import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from serverless.repo import ServerlessRepo

client = TestClient(app)
SCRATCH = os.environ.get(
    "XCELSIOR_GOAL_SCRATCH",
    "/tmp/grok-goal-6f86c7cfe9c2/implementer",
)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _headers():
    email = f"usage-ev-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Usage Ev"},
    )
    assert reg.status_code == 200
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    body = login.json()
    user = reg.json().get("user") or body.get("user") or {}
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    dep = client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 50.0},
        headers=headers,
    )
    assert dep.status_code == 200
    return headers, user["customer_id"]


def test_usage_api_sample_json(tmp_path, monkeypatch):
    headers, cust = _headers()
    created = client.post(
        "/api/v2/serverless/endpoints",
        headers=headers,
        json={
            "name": f"usage-{uuid.uuid4().hex[:4]}",
            "mode": "preset",
            "model_name": "Qwen/Qwen3-8B",
            "min_workers": 0,
        },
    )
    assert created.status_code == 200
    ep_id = created.json()["endpoint"]["endpoint_id"]
    repo = ServerlessRepo()
    worker = repo.create_worker(ep_id, scheduler_job_id="sched-evidence")
    for i in range(5):
        job = repo.enqueue_job(ep_id, cust, {"prompt": f"p{i}"})
        repo.claim_next_job(ep_id, worker["worker_id"])
        repo.complete_job(
            job["job_id"],
            output={"ok": True},
            input_tokens=1000 + i * 100,
            output_tokens=200 + i * 20,
            cached_tokens=500 + i * 50,
            ttft_ms=120 + i * 10,
            gpu_seconds=2,
        )

    r1 = client.get(f"/api/v2/serverless/endpoints/{ep_id}/usage", headers=headers)
    r2 = client.get(f"/api/v2/serverless/endpoints/{ep_id}/usage", headers=headers)
    health = client.get(f"/api/v2/serverless/endpoints/{ep_id}/health", headers=headers)
    assert r1.status_code == 200
    sample = {
        "usage_run1": r1.json(),
        "usage_run2": r2.json(),
        "health_pricing": health.json().get("health", {}).get("pricing"),
    }
    u = sample["usage_run1"]["usage"]
    assert u["pricing"]["input_price_cad_per_m"] > 0
    assert "rate_cents_per_second_per_worker" in u["pricing"]
    assert u["last_24h"]["ttft_p95_ms"] > 0
    assert u["last_24h"]["kv_cache_hit_rate"] > 0
    assert sample["usage_run1"] == sample["usage_run2"]

    os.makedirs(SCRATCH, exist_ok=True)
    out = os.path.join(SCRATCH, "usage-sample.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)