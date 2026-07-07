"""Production billing tick: bill_active_serverless_workers with real PG + wallet."""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from billing import get_billing_engine
from serverless.billing_context import endpoint_from_worker_row
from serverless.metering import (
    bill_active_serverless_workers,
    blended_billing_for_endpoint,
    estimate_cost_cad,
    get_gpu_rate_per_hour,
)
from serverless.repo import EndpointCreate, ServerlessRepo

SCRATCH = os.environ.get(
    "XCELSIOR_GOAL_SCRATCH",
    "/tmp/grok-goal-6f86c7cfe9c2/implementer",
)


@pytest.fixture
def repo():
    return ServerlessRepo()


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
def billing_tick_env(monkeypatch):
    monkeypatch.delenv("XCELSIOR_SERVERLESS_BLENDED_BILLING", raising=False)
    monkeypatch.setenv("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")


def _write_tick_evidence(payload: dict) -> None:
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, "billing-tick-integration.json")
    cases: list[dict] = []
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                cases = existing
            elif isinstance(existing, dict):
                cases = [existing]
        except (OSError, json.JSONDecodeError):
            cases = []
    cases.append(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, default=str)


class TestBillingTickIntegration:
    def test_preset_llm_tick_charges_max_gpu_token_and_consumes_accrual(
        self, repo: ServerlessRepo, owner_id: str
    ):
        """Shipped periodic path must blend preset LLM charges and zero accrual."""
        billing = get_billing_engine()
        billing.deposit(owner_id, 200.0, description="tick-integration")

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner_id,
                name="tick-llm",
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
        sched_job = f"sched-{uuid.uuid4().hex[:8]}"
        worker = repo.create_worker(ep_id, scheduler_job_id=sched_job)
        worker_id = str(worker["worker_id"])
        allocated = time.time() - 130.0
        repo.update_worker(worker_id, state="ready", allocated_at=allocated)

        token_cost = 2.50
        repo.accrue_endpoint_token_cost(ep_id, token_cost)
        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(token_cost)

        joined = repo.get_endpoint(ep_id) or {}
        ctx = endpoint_from_worker_row({**joined, **worker})
        assert blended_billing_for_endpoint(ctx) is True

        rate = get_gpu_rate_per_hour("RTX 4090", "ca-east")
        gpu_amount = estimate_cost_cad(130, rate, 1)
        expected = max(gpu_amount, token_cost)

        wallet_before = float(billing.get_wallet(owner_id)["balance_cad"])
        billed_count = bill_active_serverless_workers(billing, now=time.time())
        wallet_after = float(billing.get_wallet(owner_id)["balance_cad"])
        debited = wallet_before - wallet_after

        assert billed_count >= 1
        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(0.0)
        assert debited == pytest.approx(expected, rel=0.02)
        assert debited >= token_cost

        _write_tick_evidence(
            {
                "case": "preset_llm_tick",
                "endpoint": ctx,
                "token_cost_cad": token_cost,
                "gpu_amount_cad": gpu_amount,
                "debited_cad": debited,
                "billed_workers": billed_count,
                "accrual_after": repo.peek_endpoint_token_cost(ep_id),
            }
        )
        repo.soft_delete_endpoint(ep_id, owner_id)

    def test_custom_endpoint_tick_gpu_only_preserves_token_accrual(
        self, repo: ServerlessRepo, owner_id: str
    ):
        billing = get_billing_engine()
        billing.deposit(owner_id, 200.0, description="tick-custom")

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner_id,
                name="tick-custom",
                mode="custom",
                image_ref="xcelsior/serverless-base:cuda12.4",
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

        token_cost = 1.75
        repo.accrue_endpoint_token_cost(ep_id, token_cost)

        joined = repo.get_endpoint(ep_id) or {}
        ctx = endpoint_from_worker_row({**joined, **worker})
        assert blended_billing_for_endpoint(ctx) is False

        rate = get_gpu_rate_per_hour("RTX 4090", "ca-east")
        gpu_amount = estimate_cost_cad(130, rate, 1)

        wallet_before = float(billing.get_wallet(owner_id)["balance_cad"])
        bill_active_serverless_workers(billing, now=time.time())
        wallet_after = float(billing.get_wallet(owner_id)["balance_cad"])

        assert repo.peek_endpoint_token_cost(ep_id) == pytest.approx(token_cost)
        assert (wallet_before - wallet_after) == pytest.approx(gpu_amount, rel=0.05)

        _write_tick_evidence(
            {
                "case": "custom_gpu_only_tick",
                "debited_cad": wallet_before - wallet_after,
                "gpu_amount_cad": gpu_amount,
                "accrual_preserved": repo.peek_endpoint_token_cost(ep_id),
            }
        )
        repo.soft_delete_endpoint(ep_id, owner_id)