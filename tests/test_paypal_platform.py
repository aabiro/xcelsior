"""PayPal platform Complete Payments helpers."""

from __future__ import annotations

import routes.billing as billing_mod


def test_paypal_purchase_unit_without_payee(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_MERCHANT_ID", "")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "")
    unit = billing_mod._paypal_purchase_unit("cust-1", 25.5)
    assert unit["amount"] == {"currency_code": "CAD", "value": "25.50"}
    assert unit["custom_id"] == "cust-1"
    assert "payee" not in unit


def test_paypal_purchase_unit_payee_email(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_MERCHANT_ID", "")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "payee@example.com")
    unit = billing_mod._paypal_purchase_unit("cust-2", 10.0)
    assert unit["payee"] == {"email_address": "payee@example.com"}


def test_paypal_purchase_unit_sandbox_prefers_email(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_MODE", "sandbox")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_MERCHANT_ID", "BEC6DEHNQBV32")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "payee@example.com")
    unit = billing_mod._paypal_purchase_unit("cust-3", 10.0)
    assert unit["payee"] == {"email_address": "payee@example.com"}


def test_paypal_purchase_unit_live_prefers_merchant_id(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_MODE", "live")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_MERCHANT_ID", "BEC6DEHNQBV32")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "payee@example.com")
    unit = billing_mod._paypal_purchase_unit("cust-4", 10.0)
    assert unit["payee"] == {"merchant_id": "BEC6DEHNQBV32"}


def test_paypal_headers_partner_attribution(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_PARTNER_ATTRIBUTION_ID", "BN-CODE-123")
    headers = billing_mod._paypal_headers("tok")
    assert headers["PayPal-Partner-Attribution-Id"] == "BN-CODE-123"


def test_paypal_enabled_platform_mode(monkeypatch):
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "id")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "secret")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PARTNER_ATTRIBUTION_ID", "4819757953265067229")
    monkeypatch.setattr(billing_mod, "_PAYPAL_PLATFORM_PAYEE_EMAIL", "payee@example.com")
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    r = client.get("/api/billing/paypal/enabled")
    assert r.status_code == 200
    data = r.json()
    assert data["enabled"] is True
    assert data["platform_mode"] is True