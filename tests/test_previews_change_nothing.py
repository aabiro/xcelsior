"""A preview runs the real checks and changes nothing.

The MCP tools `top_up_wallet` and `create_volume` preview before they act.
Both previews come from the server, not from the tool: the card a top-up
would charge is whatever `resolve_payment_method` picks, and whether a volume
can be created depends on the owner's existing volumes. A preview that guessed
either could name a different card, or promise a volume the real call
refuses.
"""

import os
import time
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

import routes.billing as billing_routes
from api import app
from db import _get_pg_pool

client = TestClient(app)


@pytest.fixture(scope="module")
def account():
    email = f"preview-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Preview"},
    )
    assert reg.status_code == 200, reg.text[:300]
    body = reg.json()
    return {
        "customer_id": body["user"]["customer_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


# ── top-up ─────────────────────────────────────────────────────────────


class _Engine:
    """Two saved cards; charging is a test failure."""

    CARDS = [
        {"id": "pm_visa", "brand": "visa", "last4": "4242", "exp_month": 1, "exp_year": 2030},
        {"id": "pm_mc", "brand": "mastercard", "last4": "4444", "exp_month": 2, "exp_year": 2031},
    ]

    def resolve_payment_method(self, customer_id, *, payment_method_id="", last4="", brand=""):
        matches = [
            c
            for c in self.CARDS
            if (not payment_method_id or c["id"] == payment_method_id)
            and (not last4 or c["last4"] == last4)
            and (not brand or c["brand"] == brand.lower())
        ]
        if len(matches) == 1:
            return {"ok": True, "payment_method": matches[0]}
        return {
            "ok": False,
            "reason": "ambiguous" if matches else "no_match",
            "message": "Which card?",
            "candidates": matches,
        }

    def get_wallet(self, customer_id):
        return {"stripe_customer_id": "cus_preview"}

    def charge_saved_card(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("a preview charged the card")


@pytest.fixture()
def engine(monkeypatch):
    fake = _Engine()
    monkeypatch.setattr(billing_routes, "get_billing_engine", lambda: fake)
    return fake


def test_a_topup_preview_names_the_card_the_server_would_charge(account, engine):
    r = client.post(
        "/api/v2/billing/top-up",
        json={"amount_cad": 25, "card_brand": "mastercard", "preview": True},
        headers=account["headers"],
    )
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["preview"] is True and body["charged"] is False
    assert body["amount_cad"] == 25
    assert body["payment_method"]["last4"] == "4444"


def test_a_topup_preview_refuses_an_ambiguous_card_as_the_charge_would(account, engine):
    r = client.post(
        "/api/v2/billing/top-up",
        json={"amount_cad": 25, "preview": True},
        headers=account["headers"],
    )
    assert r.status_code == 409, r.text[:300]
    # The app's error handler may reshape the detail; the refusal itself is
    # what matters: same 409 and reason the real charge would give.
    assert "ambiguous" in r.text


# ── volumes ────────────────────────────────────────────────────────────


def _volumes(owner: str) -> int:
    with _get_pg_pool().connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM volumes WHERE owner_id=%s AND status!='deleted'", (owner,)
        ).fetchone()[0]


def test_a_volume_preview_quotes_the_price_and_creates_nothing(account):
    before = _volumes(account["customer_id"])
    r = client.post(
        "/api/v2/volumes",
        json={"name": "preview-weights", "size_gb": 40, "preview": True},
        headers=account["headers"],
    )
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["preview"] is True and body["created"] is False
    assert body["estimated_monthly_cost_cad"] == round(40 * body["price_per_gb_month_cad"], 2)
    assert _volumes(account["customer_id"]) == before


def test_a_volume_preview_refuses_what_a_create_would_refuse(account):
    """The owner already has a volume of that name: preview and create agree."""
    owner = account["customer_id"]
    name = f"taken-{uuid.uuid4().hex[:6]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb, region, "
            "province, encrypted, status, created_at) "
            "VALUES (%s,%s,%s,'nfs',10,'','',true,'available',%s)",
            (f"vol-{uuid.uuid4().hex[:12]}", owner, name, time.time()),
        )
        conn.commit()
    r = client.post(
        "/api/v2/volumes",
        json={"name": name, "size_gb": 10, "preview": True},
        headers=account["headers"],
    )
    assert r.status_code == 400, r.text[:300]
    assert "already exists" in r.json().get("detail", r.text)
