"""Phase 7 — serverless GPU-seconds billing (Novita-aligned worker uptime)."""

import os
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.metering import (
    charge_serverless_execution,
    estimate_cost_cad,
    estimate_worker_cost_for_duration_sec,
    last_billed_period_end,
    pricing_for_endpoint,
    token_cost_metadata,
    worker_rate_cad_per_second,
)
from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.service import ServerlessService, WalletPreflightError


class TestNovitaPricingQuote:
    def test_rate_cents_per_second_per_worker(self):
        # $3.60/hr single GPU → $0.001/s → 0.1 ¢/s
        rate_sec = worker_rate_cad_per_second(3.60, 1)
        assert rate_sec == pytest.approx(0.001, rel=1e-3)
        quote = pricing_for_endpoint(
            {"gpu_tier": "RTX 4090", "region": "ca-east", "gpu_count": 2}
        )
        assert quote["billing_model"] == "worker_running_seconds"
        assert quote["gpu_count"] == 2
        assert "rate_cents_per_second_per_worker" in quote

    def test_estimate_matches_novita_formula(self):
        # 100s × ($3.60/hr × 1 GPU) / 3600 = $0.10
        cost = estimate_worker_cost_for_duration_sec(100, 3.60, 1)
        assert cost == pytest.approx(0.1, rel=1e-3)


class TestIncrementalBilling:
    def test_no_double_charge_when_period_already_billed(self):
        repo = MagicMock()
        billing = MagicMock()
        billing.charge.return_value = {"charged": True, "balance_cad": 10.0}

        worker = {
            "worker_id": "swk-1",
            "scheduler_job_id": "job-abc",
            "allocated_at": 1000.0,
            "host_id": "h1",
        }
        endpoint = {
            "endpoint_id": "sep-1",
            "owner_id": "cust-1",
            "gpu_tier": "RTX 4090",
            "region": "ca-east",
            "gpu_count": 1,
            "name": "test",
        }

        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
            "serverless.metering.last_billed_period_end", return_value=1100.0
        ), patch("serverless.metering.MIN_BILLING_INTERVAL_SEC", 60):
            result = charge_serverless_execution(
                billing,
                repo,
                worker,
                endpoint,
                period_end=1150.0,
                final=True,
            )
        assert result["charged"] is True
        assert result["gpu_seconds"] == 50
        billing.charge.assert_called_once()
        args = billing.charge.call_args[0]
        expected = estimate_cost_cad(50, 3.60, 1)
        assert args[1] == pytest.approx(expected, rel=1e-4)

    def test_skips_sub_minimum_interval_unless_final(self):
        repo = MagicMock()
        billing = MagicMock()
        worker = {"scheduler_job_id": "j1", "allocated_at": time.time() - 30}
        endpoint = {"endpoint_id": "e1", "owner_id": "c1", "gpu_tier": "x", "region": "ca-east"}

        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=1.0), patch(
            "serverless.metering.last_billed_period_end", return_value=None
        ), patch("serverless.metering.MIN_BILLING_INTERVAL_SEC", 60):
            result = charge_serverless_execution(
                billing, repo, worker, endpoint, final=False
            )
        assert result["charged"] is False
        assert result["reason"] == "below_minimum_interval"
        billing.charge.assert_not_called()


class TestTokenMetadataOnly:
    def test_tokens_not_primary_debit(self):
        meta = token_cost_metadata(1000, 2000)
        assert meta["total_token_cost_cad"] > 0
        assert "input_tokens" in meta


class TestWalletPreflight:
    def test_suspended_wallet_rejected(self):
        mock_wallet = {"status": "suspended", "balance_cad": 10.0, "grace_until": 0}
        with patch("billing.get_billing_engine") as mock_be:
            mock_be.return_value.get_wallet.return_value = mock_wallet
            with pytest.raises(WalletPreflightError) as exc:
                ServerlessService.wallet_preflight("cust-suspended")
            assert exc.value.status_code == 402


class TestJobCompleteNoWalletDebit:
    def test_worker_complete_does_not_set_billed_cost(self):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="bill", mode="preset", model_ref="m", min_workers=0)
        )
        worker = repo.create_worker(str(ep["endpoint_id"]), scheduler_job_id="sched-1")
        repo.update_worker(str(worker["worker_id"]), state="ready")
        job = repo.enqueue_job(str(ep["endpoint_id"]), owner, {"x": 1})
        repo.claim_next_job(str(ep["endpoint_id"]), str(worker["worker_id"]))

        svc = ServerlessService(repo)
        with patch.object(svc, "_broadcast"):
            completed = svc.worker_complete_job(
                str(worker["worker_id"]),
                str(job["job_id"]),
                output={"ok": True},
                input_tokens=100,
                output_tokens=50,
            )
        assert completed is not None
        assert float(completed.get("cost_cad") or 0) == 0.0
        assert completed.get("input_tokens") == 100


class TestTeamBillingOwner:
    def test_endpoint_create_uses_billing_customer(self):
        from routes._deps import _serverless_scope_owner_id

        user = {
            "email": "u@x.com",
            "customer_id": "personal-cust",
            "team_id": "team-abc",
            "team_role": "member",
        }
        with patch("routes._deps._effective_billing_customer_id", return_value="team-wallet-cust"):
            owner = _serverless_scope_owner_id(user)
        assert owner == "team-wallet-cust"