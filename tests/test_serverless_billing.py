"""Phase 7 — serverless GPU-seconds billing (Novita-aligned worker uptime)."""

import os
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.metering import (
    blended_period_amount,
    cached_input_price,
    charge_serverless_execution,
    estimate_cost_cad,
    estimate_worker_cost_for_duration_sec,
    infer_model_params_b,
    last_billed_period_end,
    pricing_for_endpoint,
    token_cost_metadata,
    token_prices_for_model,
    token_pricing_quote,
    worker_rate_cad_per_second,
)
from serverless.openai_proxy import accrue_proxy_token_usage, extract_usage_from_response
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


class TestSizeTieredTokenPricing:
    def test_param_inference(self):
        assert infer_model_params_b("meta-llama/Llama-3.1-8B-Instruct") == 8.0
        assert infer_model_params_b("meta-llama/Llama-3.3-70B-Instruct") == 70.0
        assert infer_model_params_b("google/gemma-2-9b-it") == 9.0
        # MoE / large families map to the top band even without an explicit size.
        assert infer_model_params_b("deepseek-ai/DeepSeek-R1") == float("inf")
        assert infer_model_params_b("mistralai/Mixtral-8x7B") == float("inf")
        assert infer_model_params_b(None) is None
        assert infer_model_params_b("some-unknown-model") is None

    def test_bands(self):
        # ≤ 9B band
        assert token_prices_for_model("gemma-2-9b-it") == (0.15, 0.45)
        # 10–34B band
        assert token_prices_for_model("Qwen2.5-32B") == (0.35, 1.05)
        # 35–80B band
        assert token_prices_for_model("Llama-3.3-70B") == (0.70, 2.10)
        # 80B+ / MoE band
        assert token_prices_for_model("DeepSeek-V3") == (1.10, 3.30)

    def test_small_model_cheaper_than_large(self):
        small = token_cost_metadata(1_000_000, 1_000_000, model_ref="Mistral-7B")
        large = token_cost_metadata(1_000_000, 1_000_000, model_ref="Llama-3.3-70B")
        assert small["total_token_cost_cad"] < large["total_token_cost_cad"]
        # 7B → 0.15 + 0.45 = 0.60 CAD per 1M+1M
        assert small["total_token_cost_cad"] == pytest.approx(0.60, rel=1e-6)

    def test_unknown_model_falls_back_to_flat(self):
        meta = token_cost_metadata(1_000_000, 1_000_000)  # no model_ref
        assert meta["input_price_cad_per_m"] == token_prices_for_model(None)[0]


class TestBlendedMeter:
    def test_charges_the_higher(self):
        assert blended_period_amount(0.10, 0.25) == 0.25  # token cost wins
        assert blended_period_amount(0.40, 0.05) == 0.40  # gpu cost wins
        assert blended_period_amount(0.0, 0.0) == 0.0
        assert blended_period_amount(0.30, 0.30) == 0.30

    def test_charge_consumes_accrual_and_bills_blended_when_enabled(self):
        repo = MagicMock()
        repo.consume_endpoint_token_cost.return_value = 5.00  # token cost ≫ gpu cost
        billing = MagicMock()
        billing.charge.return_value = {"charged": True, "balance_cad": 100.0}
        worker = {"worker_id": "w1", "scheduler_job_id": "j1", "allocated_at": 1000.0, "host_id": "h1"}
        endpoint = {
            "endpoint_id": "ep1", "owner_id": "c1", "gpu_tier": "RTX 4090",
            "region": "ca-east", "gpu_count": 1, "name": "t",
        }
        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
            "serverless.metering.last_billed_period_end", return_value=None
        ), patch("serverless.metering.BLENDED_BILLING_ENABLED", True), patch(
            "serverless.metering.MIN_BILLING_INTERVAL_SEC", 60
        ):
            # 120s @ 3.60/hr = 0.12 CAD GPU; token cost 5.00 → blended bills 5.00
            result = charge_serverless_execution(
                billing, repo, worker, endpoint, period_end=1000.0 + 120, final=True
            )
        repo.consume_endpoint_token_cost.assert_called_once_with("ep1")
        assert result["token_cost_cad"] == 5.00
        assert result["blended_amount_cad"] == 5.00
        assert result["amount_cad"] == 5.00
        assert billing.charge.call_args[0][1] == 5.00  # charged the blended (token) amount

    def test_charge_defaults_to_gpu_when_flag_off(self):
        repo = MagicMock()
        repo.consume_endpoint_token_cost.return_value = 5.00
        billing = MagicMock()
        billing.charge.return_value = {"charged": True, "balance_cad": 100.0}
        worker = {"worker_id": "w2", "scheduler_job_id": "j2", "allocated_at": 1000.0, "host_id": "h1"}
        endpoint = {
            "endpoint_id": "ep2", "owner_id": "c1", "gpu_tier": "RTX 4090",
            "region": "ca-east", "gpu_count": 1, "name": "t",
        }
        with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
            "serverless.metering.last_billed_period_end", return_value=None
        ), patch("serverless.metering.BLENDED_BILLING_ENABLED", False), patch(
            "serverless.metering.MIN_BILLING_INTERVAL_SEC", 60
        ):
            result = charge_serverless_execution(
                billing, repo, worker, endpoint, period_end=1000.0 + 120, final=True
            )
        # Flag off → GPU-seconds billed; token cost recorded (dual-write) but not charged.
        assert result["amount_cad"] == pytest.approx(0.12, rel=1e-3)
        assert result["token_cost_cad"] == 5.00
        assert billing.charge.call_args[0][1] == pytest.approx(0.12, rel=1e-3)


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


class TestCachedTokenPricing:
    def test_cached_cheaper_than_computed(self):
        meta = token_cost_metadata(1_000_000, 0, model_ref="Mistral-7B", cached_tokens=800_000)
        full = token_cost_metadata(1_000_000, 0, model_ref="Mistral-7B", cached_tokens=0)
        assert meta["total_token_cost_cad"] < full["total_token_cost_cad"]
        assert meta["cached_tokens"] == 800_000
        assert meta["computed_input_tokens"] == 200_000

    def test_cached_price_is_half_input_band(self):
        in_price, _ = token_prices_for_model("gemma-2-9b-it")
        assert cached_input_price(in_price) == pytest.approx(in_price * 0.5, rel=1e-6)

    def test_pricing_quote_includes_token_rates_for_preset(self):
        quote = pricing_for_endpoint(
            {
                "mode": "preset",
                "model_ref": "Qwen/Qwen3-8B",
                "gpu_tier": "RTX 4090",
                "region": "ca-east",
                "gpu_count": 1,
            }
        )
        assert quote.get("token_billing") is True
        assert quote["input_price_cad_per_m"] == 0.15
        assert quote["output_price_cad_per_m"] == 0.45
        assert quote["cached_input_price_cad_per_m"] == pytest.approx(0.075, rel=1e-6)
        assert "rate_cents_per_second_per_worker" in quote

    def test_custom_endpoint_has_no_token_billing(self):
        quote = pricing_for_endpoint(
            {"mode": "custom", "image_ref": "img", "gpu_tier": "RTX 4090", "region": "ca-east"}
        )
        assert "token_billing" not in quote


class TestOpenAIProxyAccrual:
    def test_extract_usage_with_cached_tokens(self):
        usage = extract_usage_from_response(
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 600},
                }
            }
        )
        assert usage["input_tokens"] == 1000
        assert usage["cached_tokens"] == 600

    def test_proxy_accrual_idempotent(self):
        repo = MagicMock()
        repo.token_usage_already_recorded.return_value = False
        ep = {"endpoint_id": "ep-x", "model_ref": "Qwen/Qwen3-8B"}
        usage = {"input_tokens": 500, "output_tokens": 100, "cached_tokens": 200}
        r1 = accrue_proxy_token_usage(repo, ep, usage, idempotency_key="req-1")
        assert r1["accrued"] is True
        repo.accrue_endpoint_token_cost.assert_called_once()
        repo.token_usage_already_recorded.return_value = True
        r2 = accrue_proxy_token_usage(repo, ep, usage, idempotency_key="req-1")
        assert r2["accrued"] is False
        assert r2["reason"] == "duplicate_idempotency_key"


class TestDualWriteMatrix:
    def test_7b_vs_70b_token_costs(self):
        small = token_cost_metadata(1_000_000, 1_000_000, model_ref="Qwen/Qwen3-8B")
        large = token_cost_metadata(1_000_000, 1_000_000, model_ref="Llama-3.3-70B")
        assert small["total_token_cost_cad"] < large["total_token_cost_cad"]

    def test_blended_matrix_cases(self):
        # GPU wins when utilization is high relative to token throughput
        assert blended_period_amount(0.40, 0.12) == 0.40
        # Token wins when throughput is high on a cheap GPU slice
        assert blended_period_amount(0.05, 0.25) == 0.25
        # Cached tokens lower the token side
        cached = token_cost_metadata(1_000_000, 1_000_000, model_ref="Mistral-7B", cached_tokens=900_000)
        uncached = token_cost_metadata(1_000_000, 1_000_000, model_ref="Mistral-7B")
        assert cached["total_token_cost_cad"] < uncached["total_token_cost_cad"]

    def test_token_pricing_quote_matches_metadata(self):
        q = token_pricing_quote("Qwen/Qwen3-8B")
        meta = token_cost_metadata(0, 0, model_ref="Qwen/Qwen3-8B")
        assert q["input_price_cad_per_m"] == meta["input_price_cad_per_m"]


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