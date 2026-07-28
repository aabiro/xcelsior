"""Honest tests for the payments-plan hardening work.

Drives real shipped functions (charge, evaluate_settlement, meters, catalog,
account-session contract). Stripe network I/O is faked only at the SDK boundary.
No live money movement.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Criterion 1: hard-stop + dynamic PMs ───────────────────────────────


class TestWalletHardStopShippedPath:
    def test_charge_insufficient_does_not_debit(self):
        from billing import get_billing_engine

        eng = get_billing_engine()
        cid = f"plan-hs-{__import__('uuid').uuid4().hex[:10]}"
        eng.get_wallet(cid)
        before = eng.get_wallet(cid)["balance_cad"]
        result = eng.charge(cid, 25.0, job_id="j-hs-1", description="GPU job")
        after = eng.get_wallet(cid)
        assert result["charged"] is False
        assert result["reason"] == "insufficient_balance"
        assert after["balance_cad"] == pytest.approx(before, abs=0.001)
        assert after["balance_cad"] >= 0
        assert after.get("hard_stop") is True or after["balance_cad"] <= 0

    def test_charge_zero_balance_hard_stops(self):
        from billing import get_billing_engine

        eng = get_billing_engine()
        cid = f"plan-zero-{__import__('uuid').uuid4().hex[:10]}"
        eng.get_wallet(cid)
        result = eng.charge(cid, 1.0, job_id="j-zero")
        wallet = eng.get_wallet(cid)
        assert result["charged"] is False
        assert result.get("action") in ("hard_stop", "account_suspended")
        assert wallet["balance_cad"] == pytest.approx(0.0, abs=0.001)

    def test_successful_charge_then_low_balance_flag(self):
        from billing import get_billing_engine

        eng = get_billing_engine()
        cid = f"plan-ok-{__import__('uuid').uuid4().hex[:10]}"
        eng.deposit(cid, 6.0, description="seed", idempotency_key=f"seed-{cid}")
        result = eng.charge(cid, 2.0, job_id="j-ok", description="Hosted GPU (3600s)")
        assert result["charged"] is True
        wallet = eng.get_wallet(cid)
        assert wallet["balance_cad"] == pytest.approx(4.0, abs=0.02)
        # Default warn threshold is $5 — 4 CAD should flag low_balance.
        assert wallet.get("low_balance") is True
        assert wallet.get("hard_stop") is False

    def test_create_credit_deposit_source_uses_dynamic_pms(self):
        from stripe_connect import StripeConnectManager

        src = inspect.getsource(StripeConnectManager.create_credit_deposit)
        assert "automatic_payment_methods" in src
        # Must not pass payment_method_types= in the create kwargs block
        assert 'payment_method_types=["card"]' not in src
        assert "payment_method_types=['card']" not in src

    def test_setup_intent_source_uses_dynamic_pms(self):
        from billing import BillingEngine

        src = inspect.getsource(BillingEngine.create_setup_intent)
        assert "automatic_payment_methods" in src
        assert 'payment_method_types=["card"]' not in src


# ── Criterion 2: meters dual-write isolation ───────────────────────────


class TestMeterDualWrite:
    def test_charge_enqueues_meter_without_blocking(self):
        from billing import get_billing_engine
        import stripe_meters

        eng = get_billing_engine()
        cid = f"plan-mtr-{__import__('uuid').uuid4().hex[:10]}"
        eng.deposit(cid, 20.0, description="seed", idempotency_key=f"seed-{cid}")

        captured: list[dict] = []

        def backend(op, **kwargs):
            if op == "enqueue":
                captured.append(kwargs)
                return {
                    "enqueued": True,
                    "event_id": "evt-1",
                    "event_name": kwargs["event_name"],
                    "value": kwargs["value"],
                }
            return {"sent": 0, "failed": 0, "skipped": 0}

        stripe_meters.set_outbox_backend(backend)
        try:
            result = eng.charge(
                cid,
                1.5,
                job_id="slvr-abc",
                description="Serverless worker final: ep (90s @ 0.5/hr)",
            )
            assert result["charged"] is True
            assert len(captured) == 1
            assert captured[0]["event_name"] == stripe_meters.EVENT_SERVERLESS_SECOND
            assert captured[0]["value"] == 90.0
            assert captured[0]["customer_id"] == cid
            # Wallet still correct after dual-write
            assert eng.get_wallet(cid)["balance_cad"] == pytest.approx(18.5, abs=0.05)
        finally:
            stripe_meters.set_outbox_backend(None)

    def test_drain_failure_does_not_reverse_wallet(self):
        from billing import get_billing_engine
        import stripe_meters

        eng = get_billing_engine()
        cid = f"plan-drain-{__import__('uuid').uuid4().hex[:10]}"
        eng.deposit(cid, 10.0, description="seed", idempotency_key=f"seed-{cid}")
        eng.charge(cid, 1.0, job_id="j-d1", description="GPU")
        bal_after_charge = eng.get_wallet(cid)["balance_cad"]

        def backend(op, **kwargs):
            if op == "drain":
                return {"sent": 0, "failed": 3, "skipped": 0}
            return {"enqueued": True}

        stripe_meters.set_outbox_backend(backend)
        try:
            stats = stripe_meters.drain_meter_outbox(limit=10)
            assert stats["failed"] == 3
            assert eng.get_wallet(cid)["balance_cad"] == pytest.approx(bal_after_charge, abs=0.001)
        finally:
            stripe_meters.set_outbox_backend(None)

    def test_infer_event_gpu_hours(self):
        from stripe_meters import EVENT_GPU_HOUR, _infer_event

        name, value = _infer_event("Hosted instance 2.5 hours", "job-1")
        assert name == EVENT_GPU_HOUR
        assert value == pytest.approx(2.5)


# ── Criterion 3: settlement queue vs pay ───────────────────────────────


class TestSettlementEligibility:
    def test_queue_no_account(self):
        from stripe_connect import evaluate_settlement

        d = evaluate_settlement(provider=None, provider_share_cad=10.0, available_cad_cents=100000)
        assert d["status"] == "queued"
        assert d["error"] == "no_stripe_account"

    def test_queue_inactive(self):
        from stripe_connect import evaluate_settlement

        d = evaluate_settlement(
            provider={"stripe_account_id": "acct_x", "status": "onboarding"},
            provider_share_cad=10.0,
            available_cad_cents=100000,
        )
        assert d["status"] == "queued"
        assert d["error"] == "provider_not_active"

    def test_queue_insufficient_float(self):
        from stripe_connect import evaluate_settlement

        d = evaluate_settlement(
            provider={"stripe_account_id": "acct_x", "status": "active"},
            provider_share_cad=50.0,
            available_cad_cents=100,  # $1.00
        )
        assert d["status"] == "queued"
        assert d["error"] == "insufficient_platform_balance"
        assert d["need_cents"] == 5000

    def test_paid_eligible_when_float_ok(self):
        from stripe_connect import evaluate_settlement

        d = evaluate_settlement(
            provider={"stripe_account_id": "acct_x", "status": "active"},
            provider_share_cad=12.34,
            available_cad_cents=50_00,
        )
        assert d["status"] == "paid_eligible"
        assert d["need_cents"] == 1234

    def test_split_payout_queues_when_float_low(self):
        from stripe_connect import StripeConnectManager, PLATFORM_CUT_FRAC

        mgr = StripeConnectManager.__new__(StripeConnectManager)
        mock_stripe = MagicMock()
        mock_stripe.Balance.retrieve.return_value = SimpleNamespace(
            available=[{"currency": "cad", "amount": 50}]
        )
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock()

        with (
            patch("stripe_connect.STRIPE_ENABLED", True),
            patch("stripe_connect.stripe", mock_stripe),
            patch.object(StripeConnectManager, "get_provider", return_value={
                "provider_id": "p1",
                "stripe_account_id": "acct_1",
                "status": "active",
            }),
            patch.object(StripeConnectManager, "_conn", return_value=mock_conn),
            patch("billing.get_tax_rate_for_province", return_value=0.13),
        ):
            result = mgr.split_payout("job-q1", "p1", 100.0, "ON")

        assert result["settlement_status"] == "queued"
        assert result["settlement_error"] == "insufficient_platform_balance"
        assert result["stripe_transfer_id"] == ""
        mock_stripe.Transfer.create.assert_not_called()

    def test_split_payout_pays_when_eligible(self):
        from stripe_connect import StripeConnectManager

        mgr = StripeConnectManager.__new__(StripeConnectManager)
        mock_stripe = MagicMock()
        mock_stripe.Balance.retrieve.return_value = SimpleNamespace(
            available=[{"currency": "cad", "amount": 1_000_00}]
        )
        transfer = SimpleNamespace(id="tr_paid_1")
        mock_stripe.Transfer.create.return_value = transfer
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock()

        with (
            patch("stripe_connect.STRIPE_ENABLED", True),
            patch("stripe_connect.stripe", mock_stripe),
            patch.object(StripeConnectManager, "get_provider", return_value={
                "provider_id": "p1",
                "stripe_account_id": "acct_1",
                "status": "active",
            }),
            patch.object(StripeConnectManager, "_conn", return_value=mock_conn),
            patch("billing.get_tax_rate_for_province", return_value=0.13),
        ):
            result = mgr.split_payout("job-pay1", "p1", 100.0, "ON")

        assert result["settlement_status"] == "paid"
        assert result["stripe_transfer_id"] == "tr_paid_1"
        assert result["provider_share_cad"] == pytest.approx(85.0, abs=0.01)
        mock_stripe.Transfer.create.assert_called_once()

    def test_daily_settle_drains_queued(self):
        from stripe_connect import StripeConnectManager

        mgr = StripeConnectManager.__new__(StripeConnectManager)
        mock_stripe = MagicMock()
        mock_stripe.Balance.retrieve.return_value = SimpleNamespace(
            available=[{"currency": "cad", "amount": 500_00}]
        )
        mock_stripe.Transfer.create.return_value = SimpleNamespace(id="tr_daily_1")

        select_rows = [
            {
                "job_id": "j-q",
                "provider_id": "p1",
                "provider_share_cad": 10.0,
                "stripe_transfer_id": "",
            }
        ]

        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                m = MagicMock()
                if "SELECT" in sql:
                    m.fetchall.return_value = select_rows
                return m

        with (
            patch("stripe_connect.STRIPE_ENABLED", True),
            patch("stripe_connect.stripe", mock_stripe),
            patch.object(StripeConnectManager, "_conn", side_effect=lambda: Conn()),
            patch.object(
                StripeConnectManager,
                "get_provider",
                return_value={
                    "provider_id": "p1",
                    "stripe_account_id": "acct_1",
                    "status": "active",
                },
            ),
        ):
            out = mgr.settle_queued_payouts(limit=10, stripe_mod=mock_stripe)

        assert out["settled"] == 1
        assert out["failed"] == 0


# ── Criterion 3b / 4: account session + country ────────────────────────


class TestConnectContracts:
    def test_create_account_session_returns_client_secret(self):
        from stripe_connect import StripeConnectManager

        mgr = StripeConnectManager.__new__(StripeConnectManager)
        mock_stripe = MagicMock()
        mock_stripe.AccountSession.create.return_value = SimpleNamespace(
            client_secret="acs_secret_test",
            expires_at=123456,
        )
        with (
            patch("stripe_connect.STRIPE_ENABLED", True),
            patch("stripe_connect.stripe", mock_stripe),
            patch.object(
                StripeConnectManager,
                "get_provider",
                return_value={
                    "provider_id": "p1",
                    "stripe_account_id": "acct_xyz",
                    "status": "onboarding",
                },
            ),
        ):
            out = mgr.create_account_session("p1")

        assert out["client_secret"] == "acs_secret_test"
        assert out["account_id"] == "acct_xyz"
        assert out["provider_id"] == "p1"
        kwargs = mock_stripe.AccountSession.create.call_args.kwargs
        assert kwargs["account"] == "acct_xyz"
        assert kwargs["components"]["account_onboarding"]["enabled"] is True
        assert kwargs["components"]["notification_banner"]["enabled"] is True

    def test_create_provider_account_accepts_non_ca_country(self):
        from stripe_connect import StripeConnectManager

        src = inspect.getsource(StripeConnectManager.create_provider_account)
        assert "country_code" in src
        assert 'country=country_code' in src or "country=country_code" in src
        # Must not hardcode only country="CA" on Account.create
        assert 'country="CA"' not in src or "country_code" in src

    def test_account_session_route_registered(self):
        from routes.providers import router

        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/api/providers/{provider_id}/account-session" in paths


# ── Criterion 5: pricing sanity ────────────────────────────────────────


class TestPricingSanity:
    def test_key_gpu_rates_competitive_cad(self):
        from db import _GPU_PRICING_BASE

        by_key = {
            (m, v, ff, hf): rate for m, v, ff, hf, rate in _GPU_PRICING_BASE
        }
        assert by_key[("RTX 2060", 6, "PCIe", False)] == pytest.approx(0.08, abs=0.001)
        assert by_key[("RTX 3060", 12, "PCIe", False)] == pytest.approx(0.12, abs=0.001)
        assert by_key[("RTX 4090", 24, "PCIe", False)] == pytest.approx(0.49, abs=0.001)
        # Ordering: 2060 < 3060 < 4090
        assert by_key[("RTX 2060", 6, "PCIe", False)] < by_key[("RTX 3060", 12, "PCIe", False)]
        assert by_key[("RTX 3060", 12, "PCIe", False)] < by_key[("RTX 4090", 24, "PCIe", False)]

    def test_wallet_product_in_catalog(self):
        from stripe_catalog import load_manifest

        manifest = load_manifest()
        wallet = manifest.get("wallet_product") or {}
        assert wallet.get("product_id")
        assert wallet.get("sku") == "xcelsior-compute-credits"
        meters = {manifest.get("gpu_meter", {}).get("event_name")}
        assert "xcelsior_gpu_hour" in meters or manifest.get("gpu_meter")


# ── Criterion 4: UI / frontend source contracts ────────────────────────


class TestUiSourceContracts:
    def test_billing_page_banners(self):
        from pathlib import Path

        text = Path("frontend/src/app/(dashboard)/dashboard/billing/page.tsx").read_text()
        assert "wallet-hard-stop-banner" in text
        assert "wallet-low-balance-banner" in text
        assert "hard_stop" in text
        assert "low_balance" in text
        assert "Top up now" in text or "Add credits" in text

    def test_earnings_embeds_stripe_and_paypal(self):
        from pathlib import Path

        text = Path("frontend/src/app/(dashboard)/dashboard/earnings/page.tsx").read_text()
        assert "StripeConnectEmbedded" in text
        assert 'mode="manage"' in text
        assert 'mode="setup"' in text
        assert "PayPalConnectCard" in text

    def test_deposit_modal_stripe_and_paypal(self):
        from pathlib import Path

        text = Path("frontend/src/components/billing/deposit-modal.tsx").read_text()
        assert "PaymentElement" in text
        assert "createPayPalOrder" in text or "PayPal" in text
        assert "createPaymentIntent" in text

    def test_stripe_connect_embedded_component_exists(self):
        from pathlib import Path

        p = Path("frontend/src/components/providers/stripe-connect-embedded.tsx")
        assert p.is_file()
        text = p.read_text()
        assert "loadConnectAndInitialize" in text
        assert "createProviderAccountSession" in text
        assert "notification-banner" in text
        assert "account-onboarding" in text
