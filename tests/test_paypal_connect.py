"""PayPal Connect manager unit tests."""

from __future__ import annotations

import time
import uuid

import paypal_connect as pc


def test_auth_assertion_jwt_shape():
    pc._PAYPAL_CLIENT_ID = "test-client-id"
    assertion = pc.auth_assertion(merchant_id="BEC6DEHNQBV32")
    parts = assertion.split(".")
    assert len(parts) == 3
    assert parts[2] == ""


def test_platform_payee_sandbox_prefers_email(monkeypatch):
    monkeypatch.setattr(pc, "_PAYPAL_MODE", "sandbox")
    monkeypatch.setattr(pc, "_PAYPAL_PLATFORM_MERCHANT_ID", "BEC6DEHNQBV32")
    monkeypatch.setattr(pc, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "payee@example.com")
    assert pc.platform_payee() == {"email_address": "payee@example.com"}


def test_marketplace_purchase_unit_includes_platform_fee(monkeypatch):
    mgr = pc.PayPalConnectManager()
    monkeypatch.setattr(
        pc.PayPalConnectManager,
        "seller_payee",
        lambda _self, _provider_id: (
            {"email_address": "seller@example.com"},
            "",
            "seller@example.com",
        ),
    )
    unit = mgr.marketplace_purchase_unit(
        "prov-1",
        "job-9",
        {
            "currency": "CAD",
            "total_micros": 100_000_000,
            "platform_share_micros": 15_000_000,
        },
    )
    assert unit["payee"] == {"email_address": "seller@example.com"}
    assert unit["custom_id"] == "prov-1:job-9"
    fees = unit["payment_instruction"]["platform_fees"][0]["amount"]
    assert fees["value"] == "15.00"


def test_split_amounts_respects_platform_cut(monkeypatch):
    monkeypatch.setattr(pc, "PLATFORM_CUT_FRAC", 0.15)
    import billing

    monkeypatch.setattr(billing, "get_tax_rate_for_province", lambda _p: 0.13)
    splits = pc._split_amounts(100.0, "ON")
    assert splits["platform_share_cad"] == 15.0
    assert splits["provider_share_cad"] == 85.0


def test_capture_marketplace_order_idempotent(monkeypatch):
    """Second capture for the same job must return existing payout_splits row."""
    import billing

    monkeypatch.setattr(pc, "PLATFORM_CUT_FRAC", 0.15)
    monkeypatch.setattr(billing, "get_tax_rate_for_province", lambda _p: 0.13)
    monkeypatch.setattr(pc, "_access_token", lambda: "test-token")

    provider_id = f"prov-idem-{uuid.uuid4().hex[:8]}"
    customer_id = f"customer-idem-{uuid.uuid4().hex[:8]}"
    job_id = f"job-idem-{uuid.uuid4().hex[:8]}"
    order_id = "ORDER-IDEM-TEST"
    now = time.time()
    mgr = pc.PayPalConnectManager()

    with mgr._conn() as conn:
        conn.execute(
            """INSERT INTO provider_accounts
               (provider_id, provider_type, stripe_account_id, status, email, legal_name,
                country, province, created_at, paypal_status, paypal_merchant_id)
               VALUES (%s, 'individual', '', 'active', %s, 'Idem Provider',
                       'CA', 'ON', %s, 'active', 'MERCHANT-IDEM')
               ON CONFLICT (provider_id) DO NOTHING""",
            (provider_id, f"{provider_id}@xcelsior.ca", now),
        )
        conn.execute(
            """INSERT INTO payout_splits
               (job_id, provider_id, tenant_id, total_micros,
                source_total_micros, provider_share_micros,
                platform_share_micros, gst_hst_micros, stripe_transfer_id,
                paypal_capture_id, paypal_order_id, payment_rail,
                settlement_status, created_at)
               VALUES (%s, %s, %s, 0, 0, 0, 0, 0, '', '', %s, 'paypal',
                       'legacy_conflict', to_timestamp(%s))""",
            (f"legacy-{job_id}", provider_id, provider_id, order_id, now - 1),
        )
        conn.execute(
            """INSERT INTO payout_splits
               (job_id, provider_id, customer_id, currency,
                stripe_transfer_id, paypal_capture_id, paypal_order_id,
                payment_rail, settlement_status, settlement_error,
                source_total_micros, total_micros, provider_share_micros,
                platform_share_micros, gst_hst_micros, rounding_adjustment_micros,
                platform_cut_bps, tax_rate_bps, settlement_key,
                rail_idempotency_key, tenant_id, created_at, updated_at, settled_at,
                legacy_imported)
               VALUES (%s, %s, %s, 'CAD',
                       '', %s, %s,
                       'paypal', 'paid', '',
                       100000000, 100000000, 85000000,
                       15000000, 13000000, 0,
                       1500, 1300, %s,
                       %s, %s, to_timestamp(%s), clock_timestamp(),
                       clock_timestamp(), FALSE)""",
            (
                job_id,
                provider_id,
                customer_id,
                "CAP-EXISTING",
                order_id,
                f"provider-job:{job_id}",
                f"provider-settlement:{job_id}",
                customer_id,
                now,
            ),
        )

    def unexpected_get(*args, **kwargs):
        raise AssertionError("paid settlement replay must not call PayPal")

    monkeypatch.setattr(pc.httpx, "get", unexpected_get)
    monkeypatch.setattr(
        pc.PayPalConnectManager,
        "seller_payee",
        lambda self, pid: ({"email_address": "seller@example.com"}, "", "seller@example.com"),
    )

    try:
        result = mgr.capture_marketplace_order(
            provider_id,
            order_id,
            expected_customer_id=customer_id,
        )
        assert result["job_id"] == job_id
        assert result["provider_share_cad"] == 85.0
        assert result["capture_id"] == "CAP-EXISTING"

        with mgr._conn() as conn:
            count = conn.execute(
                """SELECT COUNT(*) AS n
                     FROM payout_splits
                    WHERE settlement_key=%s""",
                (f"provider-job:{job_id}",),
            ).fetchone()["n"]
        assert count == 1
    finally:
        with mgr._conn() as conn:
            conn.execute(
                "DELETE FROM payout_splits WHERE paypal_order_id=%s",
                (order_id,),
            )
            conn.execute(
                "DELETE FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            )
