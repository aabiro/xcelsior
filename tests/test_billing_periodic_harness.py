"""E2E: proxy accrual → BillingEngine.auto_billing_cycle → max(gpu, token) debit."""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")
os.environ.setdefault("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")

from api import app
from billing import get_billing_engine
from serverless.billing_context import endpoint_billing_context
from serverless.metering import (
    blended_billing_for_endpoint,
    estimate_cost_cad,
    get_gpu_rate_per_hour,
)
from serverless.repo import EndpointCreate, ServerlessRepo

SCRATCH = os.environ.get(
    "XCELSIOR_GOAL_SCRATCH",
    "/tmp/grok-goal-6f86c7cfe9c2/implementer",
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


@pytest.fixture(autouse=True)
def billing_env(monkeypatch):
    monkeypatch.delenv("XCELSIOR_SERVERLESS_BLENDED_BILLING", raising=False)


def _funded_headers() -> tuple[dict[str, str], str]:
    email = f"periodic-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Periodic"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    user = reg.json().get("user") or login.json().get("user") or {}
    owner = str(user["customer_id"])
    get_billing_engine().deposit(owner, 100.0, description="periodic-harness")
    return headers, owner


def _write_evidence(payload: dict) -> None:
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, "billing-periodic-harness.json")
    cases: list[dict] = []
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            cases = existing if isinstance(existing, list) else [existing]
        except (OSError, json.JSONDecodeError):
            cases = []
    cases.append(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, default=str)


class TestBillingPeriodicHarness:
    def test_auto_billing_cycle_charges_max_gpu_when_token_small(
        self, fake_vllm_port,
    ):
        """Production path with gpu_tier set: blended = max(gpu_slice, token_accrual)."""
        headers, owner = _funded_headers()
        repo = ServerlessRepo()
        billing = get_billing_engine()

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="periodic-cycle",
                mode="preset",
                model_ref="Qwen/Qwen3-8B",
                managed_engine="vllm",
                gpu_tier="RTX 4090",
                region="ca-east",
                gpu_count=1,
                min_workers=0,
            )
        )
        ep_id = str(ep["endpoint_id"])
        sched = f"sched-{uuid.uuid4().hex[:8]}"
        worker = repo.create_worker(ep_id, scheduler_job_id=sched)
        worker_id = str(worker["worker_id"])
        repo.update_worker(worker_id, state="ready", allocated_at=time.time() - 130.0)

        chat = client.post(
            f"/v1/serverless/{ep_id}/openai/v1/chat/completions",
            headers={**headers, "idempotency-key": f"pc-{uuid.uuid4().hex[:8]}"},
            json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "ping"}]},
        )
        assert chat.status_code == 200, chat.text[:300]
        usage = chat.json().get("usage") or {}
        token_cost = float(repo.peek_endpoint_token_cost(ep_id) or 0.0)
        assert token_cost > 0

        ctx = endpoint_billing_context(repo, worker)
        assert blended_billing_for_endpoint(ctx) is True
        rate = get_gpu_rate_per_hour("RTX 4090", "ca-east")
        gpu_amount = estimate_cost_cad(130, rate, 1)
        expected = max(gpu_amount, token_cost)
        assert gpu_amount > token_cost, "fixture must exercise max(gpu, token) with gpu winning"

        wallet_before = float(billing.get_wallet(owner)["balance_cad"])
        cycle = billing.auto_billing_cycle()
        wallet_after = float(billing.get_wallet(owner)["balance_cad"])
        debited = wallet_before - wallet_after

        assert int(cycle.get("inference_billed") or 0) >= 1
        assert debited == pytest.approx(expected, rel=0.03)
        assert debited >= gpu_amount * 0.95
        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(0.0)

        _write_evidence(
            {
                "case": "auto_billing_cycle_max_gpu",
                "path": "proxy → accrual → auto_billing_cycle → bill_active_serverless_workers",
                "usage": usage,
                "token_cost_cad": token_cost,
                "gpu_amount_cad": gpu_amount,
                "expected_blended_cad": expected,
                "debited_cad": debited,
                "inference_billed": cycle.get("inference_billed"),
                "gpu_tier": ctx.get("gpu_tier"),
            }
        )
        repo.soft_delete_endpoint(ep_id, owner)

    def test_auto_billing_cycle_charges_max_token_when_token_larger(self):
        """When token accrual exceeds GPU slice, blended charges token side."""
        repo = ServerlessRepo()
        billing = get_billing_engine()
        owner = f"cust-{uuid.uuid4().hex[:12]}"
        billing.deposit(owner, 100.0)

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="periodic-token-win",
                mode="preset",
                model_ref="Qwen/Qwen3-8B",
                managed_engine="vllm",
                gpu_tier="RTX 4090",
                region="ca-east",
                gpu_count=1,
                min_workers=0,
            )
        )
        ep_id = str(ep["endpoint_id"])
        worker = repo.create_worker(ep_id, scheduler_job_id=f"sched-{uuid.uuid4().hex[:8]}")
        repo.update_worker(
            str(worker["worker_id"]), state="ready", allocated_at=time.time() - 130.0
        )
        token_cost = 2.50
        repo.accrue_endpoint_token_cost(ep_id, token_cost)

        rate = get_gpu_rate_per_hour("RTX 4090", "ca-east")
        gpu_amount = estimate_cost_cad(130, rate, 1)
        expected = max(gpu_amount, token_cost)

        wallet_before = float(billing.get_wallet(owner)["balance_cad"])
        cycle = billing.auto_billing_cycle()
        debited = wallet_before - float(billing.get_wallet(owner)["balance_cad"])

        assert int(cycle.get("inference_billed") or 0) >= 1
        assert debited == pytest.approx(expected, rel=0.03)
        assert debited >= token_cost
        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(0.0)

        _write_evidence(
            {
                "case": "auto_billing_cycle_max_token",
                "token_cost_cad": token_cost,
                "gpu_amount_cad": gpu_amount,
                "debited_cad": debited,
            }
        )
        repo.soft_delete_endpoint(ep_id, owner)

    def test_deprovision_worker_uses_full_billing_context(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")

        repo = ServerlessRepo()
        billing = get_billing_engine()
        owner = f"cust-{uuid.uuid4().hex[:12]}"
        billing.deposit(owner, 50.0)

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="deprov-llm",
                mode="preset",
                model_ref="Qwen/Qwen3-8B",
                managed_engine="vllm",
                gpu_tier="RTX 4090",
                region="ca-east",
                gpu_count=1,
                min_workers=0,
            )
        )
        ep_id = str(ep["endpoint_id"])
        worker = repo.create_worker(ep_id, scheduler_job_id=f"sched-{uuid.uuid4().hex[:8]}")
        worker_id = str(worker["worker_id"])
        repo.update_worker(worker_id, state="ready", allocated_at=time.time() - 130.0)
        repo.accrue_endpoint_token_cost(ep_id, 1.25)

        from unittest.mock import patch

        from serverless.service import ServerlessService

        svc = ServerlessService(repo)
        wallet_before = float(billing.get_wallet(owner)["balance_cad"])
        with patch.object(svc.dispatcher, "terminate_worker"):
            svc.deprovision_worker(worker_id, charge=True)
        wallet_after = float(billing.get_wallet(owner)["balance_cad"])

        assert wallet_before - wallet_after >= 1.25
        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(0.0)
        repo.soft_delete_endpoint(ep_id, owner)