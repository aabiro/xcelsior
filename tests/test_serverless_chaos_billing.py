"""Chaos / fault-injection: worker dies mid-inference — requeue, idempotency, no double billing."""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

os.environ["XCELSIOR_ENV"] = "test"
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.dispatcher import ServerlessDispatcher
from serverless.metering import charge_serverless_execution
from serverless.openai_proxy import accrue_proxy_token_usage
from serverless.repo import (
    EndpointCreate,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_QUEUED,
    ServerlessRepo,
)
from serverless.service import ServerlessService


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repo():
    return ServerlessRepo()


@pytest.fixture
def preset_endpoint(repo: ServerlessRepo, owner_id: str):
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner_id,
            name="chaos-token",
            mode="preset",
            model_ref="Qwen/Qwen3-8B",
            managed_engine="vllm",
            gpu_tier="RTX 4090",
            region="ca-east",
            gpu_count=1,
            min_workers=0,
        )
    )
    yield ep
    repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)


class TestWorkerDeathMidInference:
    """Simulate worker crash during an in-flight preset job with token accrual."""

    def test_requeue_after_worker_lost(
        self, repo: ServerlessRepo, preset_endpoint: dict, owner_id: str
    ):
        ep_id = preset_endpoint["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id=f"sched-{uuid.uuid4().hex[:8]}")
        worker_id = str(worker["worker_id"])
        job = repo.enqueue_job(ep_id, owner_id, {"messages": [{"role": "user", "content": "hi"}]})
        claimed = repo.claim_next_job(ep_id, worker_id)
        assert claimed is not None
        assert claimed["status"] == JOB_STATUS_IN_PROGRESS

        dispatcher = ServerlessDispatcher(repo)
        requeued = dispatcher.handle_worker_lost(worker_id)
        assert requeued == 1

        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_QUEUED
        assert refreshed.get("worker_id") is None

    def test_token_accrual_idempotent_after_worker_death(
        self, repo: ServerlessRepo, preset_endpoint: dict, owner_id: str
    ):
        ep_id = preset_endpoint["endpoint_id"]
        idem_key = f"req-{uuid.uuid4().hex[:16]}"
        usage = {"input_tokens": 512, "output_tokens": 128, "cached_tokens": 256}

        first = accrue_proxy_token_usage(
            repo, preset_endpoint, usage, idempotency_key=idem_key, ttft_ms=120, latency_ms=800
        )
        assert first["accrued"] is True

        # Client retries same request after worker died — must not double-accrue.
        second = accrue_proxy_token_usage(
            repo, preset_endpoint, usage, idempotency_key=idem_key, ttft_ms=120, latency_ms=800
        )
        assert second["accrued"] is False
        assert second["reason"] == "duplicate_idempotency_key"

        ledger = repo.list_token_ledger(ep_id, limit=10)
        matching = [r for r in ledger if r.get("idempotency_key") == idem_key]
        assert len(matching) == 1
        assert matching[0]["input_tokens"] == 512
        assert matching[0]["output_tokens"] == 128

    def test_gpu_billing_no_double_charge_same_period(
        self, repo: ServerlessRepo, preset_endpoint: dict, monkeypatch
    ):
        """Mid-inference tick then worker death: re-billing same period must not double-charge."""
        monkeypatch.setenv("XCELSIOR_SERVERLESS_BLENDED_BILLING", "1")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")

        ep_id = preset_endpoint["endpoint_id"]
        sched_job = f"sched-{uuid.uuid4().hex[:8]}"
        worker = repo.create_worker(ep_id, scheduler_job_id=sched_job)
        worker_id = str(worker["worker_id"])
        allocated = time.time() - 90.0
        repo.update_worker(worker_id, allocated_at=allocated)

        billing = MagicMock()
        billing.charge.return_value = {"charged": True, "balance_cad": 50.0}

        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
            "serverless.metering.last_billed_period_end", return_value=None
        ):
            period_end = allocated + 60.0
            result1 = charge_serverless_execution(
                billing,
                repo,
                {**worker, "allocated_at": allocated},
                preset_endpoint,
                period_end=period_end,
                final=True,
                token_cost_cad=0.05,
            )
            assert result1["charged"] is True
            first_amount = result1["amount_cad"]
            assert first_amount > 0

            # Same period_end after cycle recorded — must not bill again.
            with patch(
                "serverless.metering.last_billed_period_end", return_value=period_end
            ):
                result2 = charge_serverless_execution(
                    billing,
                    repo,
                    {**worker, "allocated_at": allocated},
                    preset_endpoint,
                    period_end=period_end,
                    final=True,
                    token_cost_cad=0.05,
                )
            assert result2["charged"] is False
            assert result2["reason"] == "zero_duration"
            assert billing.charge.call_count == 1

    def test_full_chaos_worker_death_then_retry_single_bill(
        self, repo: ServerlessRepo, preset_endpoint: dict, owner_id: str, monkeypatch
    ):
        """End-to-end: in-flight job → token accrual → worker lost → retry → one ledger row."""
        monkeypatch.setenv("XCELSIOR_SERVERLESS_BLENDED_BILLING", "1")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")

        ep_id = preset_endpoint["endpoint_id"]
        sched_job = f"sched-{uuid.uuid4().hex[:8]}"
        worker1 = repo.create_worker(ep_id, scheduler_job_id=sched_job)
        w1 = str(worker1["worker_id"])
        idem_key = f"chaos-{uuid.uuid4().hex[:12]}"

        job = repo.enqueue_job(
            ep_id,
            owner_id,
            {"messages": [{"role": "user", "content": "analyze trade"}]},
            idempotency_key=idem_key,
        )
        repo.claim_next_job(ep_id, w1)

        usage = {"input_tokens": 1024, "output_tokens": 256, "cached_tokens": 512}
        accrue_proxy_token_usage(
            repo, preset_endpoint, usage, idempotency_key=idem_key, ttft_ms=200, latency_ms=1500
        )

        dispatcher = ServerlessDispatcher(repo)
        assert dispatcher.handle_worker_lost(w1) == 1

        # Retry on fresh worker — same client idempotency key.
        worker2 = repo.create_worker(ep_id, scheduler_job_id=sched_job)
        w2 = str(worker2["worker_id"])
        retry_claim = repo.claim_next_job(ep_id, w2)
        assert retry_claim is not None
        assert retry_claim["job_id"] == job["job_id"]

        retry_accrual = accrue_proxy_token_usage(
            repo, preset_endpoint, usage, idempotency_key=idem_key, ttft_ms=180, latency_ms=1200
        )
        assert retry_accrual["accrued"] is False

        ledger = repo.list_token_ledger(ep_id, limit=20)
        idem_rows = [r for r in ledger if r.get("idempotency_key") == idem_key]
        assert len(idem_rows) == 1

        billing = MagicMock()
        billing.charge.return_value = {"charged": True, "balance_cad": 100.0}
        allocated = time.time() - 120.0
        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
            "serverless.metering.last_billed_period_end", return_value=None
        ):
            bill = charge_serverless_execution(
                billing,
                repo,
                {**worker2, "allocated_at": allocated, "scheduler_job_id": sched_job},
                preset_endpoint,
                period_end=time.time(),
                final=True,
            )
        assert bill["charged"] is True
        assert billing.charge.call_count == 1

        svc = ServerlessService(repo)
        with patch.object(svc, "_broadcast"):
            completed = svc.worker_complete_job(
                w2,
                job["job_id"],
                output={"choices": [{"message": {"content": "ok"}}]},
                input_tokens=1024,
                output_tokens=256,
            )
        assert completed is not None
        assert float(completed.get("cost_cad") or 0) == 0.0  # wallet debit via periodic meter only