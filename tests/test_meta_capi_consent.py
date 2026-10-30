"""Conversions API events follow the cookie banner's marketing choice.

The browser states the choice on every API request (X-Marketing-Consent). A
signed-in visitor's choice is also saved to the account, so the Stripe
webhook and auto top-ups, which have no browser, follow it. No choice is no
consent. The Graph API call is mocked; nothing here reaches Meta.
"""

from __future__ import annotations

import threading
import uuid

import pytest
from fastapi.testclient import TestClient

import meta_capi


class _Req:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.cookies: dict[str, str] = {}
        self.client = None


FBP = "fb.1.1710000000000.1234567890"


@pytest.fixture
def sent(monkeypatch):
    """Turn the sender on and record what it would deliver, synchronously."""
    out: list[dict] = []
    monkeypatch.setattr(meta_capi, "configured", lambda: True)
    monkeypatch.setattr(meta_capi, "deliver", lambda events: out.extend(events) or {"ok": True})

    class _Inline:
        def __init__(self, target, **_kw):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(threading, "Thread", _Inline)
    return out


def _saved(marketing: bool | None) -> dict:
    prefs = {} if marketing is None else {"ad_consent": {"marketing": marketing}}
    return {"email": "ada@xcelsior.ca", "user_id": "u_1", "preferences": prefs}


def test_request_consent_reads_only_explicit_values():
    assert meta_capi.request_consent(_Req({"X-Marketing-Consent": "granted"})) is True
    assert meta_capi.request_consent(_Req({"X-Marketing-Consent": "denied"})) is False
    assert meta_capi.request_consent(_Req({"X-Marketing-Consent": "yes"})) is None
    assert meta_capi.request_consent(_Req()) is None
    assert meta_capi.request_consent(None) is None


def test_no_choice_anywhere_sends_nothing(sent):
    meta_capi.track_event("CompleteRegistration", "reg_1", user=_saved(None), request=_Req())
    assert sent == []


def test_denied_header_sends_nothing(sent):
    meta_capi.track_event(
        "CompleteRegistration",
        "reg_1",
        user=_saved(True),
        request=_Req({"X-Marketing-Consent": "denied", "X-FB-FBP": FBP}),
    )
    assert sent == []


def test_granted_header_sends_with_the_shared_event_id(sent):
    meta_capi.track_event(
        "CompleteRegistration",
        "reg_1",
        user=_saved(None),
        request=_Req({"X-Marketing-Consent": "granted", "X-FB-FBP": FBP}),
    )
    assert [e["event_id"] for e in sent] == ["reg_1"]
    assert sent[0]["user_data"]["fbp"] == FBP


def test_request_without_a_header_falls_back_to_the_account(sent):
    meta_capi.track_event("StartTrial", "first_gpu_u_1", user=_saved(True), request=_Req())
    meta_capi.track_event("StartTrial", "first_gpu_u_2", user=_saved(False), request=_Req())
    assert [e["event_id"] for e in sent] == ["first_gpu_u_1"]


def test_explicit_consent_argument_wins(sent):
    meta_capi.track_event("Purchase", "purchase_a", user=_saved(True), consent=False)
    meta_capi.track_event("Purchase", "purchase_b", user=_saved(None), consent=True)
    assert [e["event_id"] for e in sent] == ["purchase_b"]


def test_click_ids_are_not_stored_without_consent(monkeypatch):
    stored: list[dict] = []

    class _Store:
        @staticmethod
        def get_user(_email):
            return {}

        @staticmethod
        def update_user(_email, updates):
            stored.append(updates)

    import db

    monkeypatch.setattr(db, "UserStore", _Store)
    user = {"email": f"ads-{uuid.uuid4().hex[:8]}@xcelsior.ca"}
    meta_capi.capture_request_attribution(user, _Req({"X-FB-FBP": FBP}))
    meta_capi.capture_request_attribution(
        user, _Req({"X-Marketing-Consent": "denied", "X-FB-FBP": FBP})
    )
    assert stored == []
    meta_capi.capture_request_attribution(
        user, _Req({"X-Marketing-Consent": "granted", "X-FB-FBP": FBP})
    )
    assert stored and stored[0]["preferences"]["meta_ads"]["fbp"] == FBP


def test_checkout_metadata_carries_the_choice_and_click_ids_only_with_consent():
    assert meta_capi.ads_metadata(_Req({"X-FB-FBP": FBP})) == {}
    assert meta_capi.ads_metadata(_Req({"X-Marketing-Consent": "denied", "X-FB-FBP": FBP})) == {
        "mkt_consent": "denied"
    }
    granted = meta_capi.ads_metadata(_Req({"X-Marketing-Consent": "granted", "X-FB-FBP": FBP}))
    assert granted["mkt_consent"] == "granted"
    assert granted["fbp"] == FBP


def test_account_consent_ignores_malformed_values():
    assert meta_capi.account_consent(_saved(True)) is True
    assert meta_capi.account_consent(_saved(False)) is False
    assert meta_capi.account_consent(_saved(None)) is None
    assert meta_capi.account_consent({"preferences": {"ad_consent": {"marketing": "true"}}}) is None
    assert meta_capi.account_consent({"preferences": '{"ad_consent": {"marketing": true}}'}) is True
    assert meta_capi.account_consent(None) is None


# ── Webhook: account choice first, checkout metadata second ──────────────


@pytest.mark.needs_db
@pytest.mark.parametrize(
    ("saved", "checkout", "expected"),
    [
        (True, "denied", True),
        (False, "granted", False),
        (None, "granted", True),
        (None, "denied", False),
        (None, None, False),
    ],
)
def test_webhook_purchase_follows_the_account_then_checkout(
    monkeypatch, sent, saved, checkout, expected
):
    import db
    from stripe_connect import StripeConnectManager

    customer_id = f"meta-consent-{uuid.uuid4().hex[:8]}"
    payer = {**_saved(saved), "customer_id": customer_id}
    monkeypatch.setattr(db.UserStore, "get_user_by_customer_id", staticmethod(lambda _c: payer))
    metadata = {"customer_id": customer_id, "product_type": "wallet_deposit"}
    if checkout:
        metadata["mkt_consent"] = checkout
    intent_id = f"pi_{uuid.uuid4().hex[:12]}"

    StripeConnectManager()._handle_payment_succeeded(
        {"id": intent_id, "amount_received": 2500, "metadata": metadata},
        f"evt_{uuid.uuid4().hex[:12]}",
    )

    purchases = [e for e in sent if e["event_name"] == "Purchase"]
    assert bool(purchases) is expected
    if expected:
        assert purchases[0]["event_id"] == f"purchase_{intent_id}"


# ── PUT /api/meta/consent ────────────────────────────────────────────────


@pytest.mark.needs_db
def test_consent_route_saves_the_choice_and_drops_click_ids_on_opt_out():
    from api import app
    from db import UserStore

    client = TestClient(app)
    email = f"consent-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Consent Tester"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text
    auth = {"Authorization": f"Bearer {login.json()['access_token']}"}

    anonymous = TestClient(app).put("/api/meta/consent", json={"marketing": True})
    assert anonymous.status_code in (401, 403)

    first = client.put("/api/meta/consent", json={"marketing": True}, headers=auth)
    assert first.status_code == 200, first.text
    assert first.json()["changed"] is True
    again = client.put("/api/meta/consent", json={"marketing": True}, headers=auth)
    assert again.json()["changed"] is False

    stored = UserStore.get_user(email) or {}
    prefs = meta_capi._user_prefs(stored)
    assert prefs["ad_consent"]["marketing"] is True
    assert prefs["ad_consent"]["source"] == "banner"
    prefs["meta_ads"] = {"fbp": FBP}
    UserStore.update_user(email, {"preferences": prefs})

    off = client.put(
        "/api/meta/consent", json={"marketing": False, "source": "settings"}, headers=auth
    )
    assert off.status_code == 200, off.text
    prefs = meta_capi._user_prefs(UserStore.get_user(email) or {})
    assert prefs["ad_consent"]["marketing"] is False
    assert prefs["ad_consent"]["source"] == "settings"
    assert "meta_ads" not in prefs
    assert meta_capi.account_consent(UserStore.get_user(email)) is False

    bad = client.put("/api/meta/consent", json={"marketing": "maybe"}, headers=auth)
    assert bad.status_code == 422
