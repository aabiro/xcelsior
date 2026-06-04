"""Smoke coverage for routes/stripe_connect_v2.py (UNTESTED_ENDPOINTS.md)."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault(
    "XCELSIOR_STRIPE_SECRET_KEY", "sk_test_ci_placeholder_not_for_production"
)

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

OK_OR_HANDLED = {200, 400, 404, 502, 503}


class _FakeStripeClient:
    """Minimal Stripe client stub — avoids live API calls in CI."""

    class _Products:
        def create(self, params=None):
            class _Product:
                id = "prod_cov_fake"
                default_price = "price_cov_fake"

            return _Product()

    class _CheckoutSessions:
        def create(self, params=None):
            class _Session:
                url = "https://checkout.stripe.com/cov-fake"
                id = "cs_cov_fake"

            return _Session()

    class _V2Core:
        class accounts:
            @staticmethod
            def create(params=None):
                class _Account:
                    id = "acct_cov_fake123"

                return _Account()

            @staticmethod
            def retrieve(account_id, params=None):
                class _Capability:
                    status = "active"

                class _StripeBalance:
                    stripe_transfers = _Capability()

                class _Capabilities:
                    stripe_balance = _StripeBalance()

                class _Recipient:
                    capabilities = _Capabilities()

                class _Configuration:
                    recipient = _Recipient()

                class _MinimumDeadline:
                    status = "complete"

                class _Summary:
                    minimum_deadline = _MinimumDeadline()

                class _Requirements:
                    summary = _Summary()

                class _Account:
                    id = account_id
                    configuration = _Configuration()
                    requirements = _Requirements()

                return _Account()

        class account_links:
            @staticmethod
            def create(params=None):
                class _Link:
                    url = "https://connect.stripe.com/setup/cov-fake"

                return _Link()

        class events:
            @staticmethod
            def retrieve(event_id):
                class _Related:
                    id = "acct_cov_fake123"

                class _Event:
                    type = "v2.core.account[requirements].updated"
                    related_object = _Related()

                return _Event()

    def __init__(self):
        self.products = self._Products()
        self.checkout = type("Checkout", (), {"sessions": self._CheckoutSessions()})()
        self.v2 = type("V2", (), {"core": self._V2Core})()

    def parse_event_notification(self, payload, sig_header, secret):
        class _Thin:
            type = "v2.core.account[requirements].updated"
            id = "evt_cov_fake"

        return _Thin()


@pytest.fixture
def fake_stripe(monkeypatch):
    import routes.stripe_connect_v2 as connect_mod

    fake = _FakeStripeClient()
    monkeypatch.setattr(connect_mod, "_stripe_client", fake)
    monkeypatch.setattr(connect_mod, "_get_stripe_client", lambda: fake)
    monkeypatch.setattr(connect_mod, "STRIPE_WEBHOOK_SECRET", "whsec_cov_test_secret")
    return fake


def test_connect_html_pages():
    for path in ("/connect/dashboard", "/connect/storefront", "/connect/success"):
        r = client.get(path)
        assert r.status_code == 200
        assert "text/html" in (r.headers.get("content-type") or "")


def test_connect_list_accounts():
    r = client.get("/api/connect/accounts")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("accounts"), list)


def test_connect_list_products():
    r = client.get("/api/connect/products")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("products"), list)


def test_connect_create_account(fake_stripe):
    r = client.post(
        "/api/connect/accounts",
        json={"display_name": "Cov Seller", "contact_email": "seller@xcelsior.ca"},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("account_id") == "acct_cov_fake123"


def test_connect_onboarding_link(fake_stripe):
    r = client.get("/api/connect/accounts/acct_cov_fake123/onboarding-link")
    assert r.status_code == 200
    assert r.json().get("url")


def test_connect_account_status(fake_stripe):
    r = client.get("/api/connect/accounts/acct_cov_fake123/status")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "onboarding_complete" in r.json()


def test_connect_create_product(fake_stripe):
    r = client.post(
        "/api/connect/products",
        json={
            "name": "Cov GPU Hour",
            "description": "Test",
            "price_cents": 500,
            "currency": "usd",
            "account_id": "acct_cov_fake123",
        },
    )
    assert r.status_code == 200
    assert r.json().get("product_id") == "prod_cov_fake"


def test_connect_checkout_missing_product(fake_stripe):
    r = client.post(
        "/api/connect/checkout",
        json={"product_id": "prod_nonexistent", "quantity": 1},
    )
    assert r.status_code == 404


def test_connect_checkout_ok(fake_stripe, monkeypatch):
    import routes.stripe_connect_v2 as connect_mod

    pool = connect_mod._get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connect_products
                (stripe_product_id, stripe_price_id, name, description,
                 price_cents, currency, account_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stripe_product_id) DO NOTHING
            """,
            (
                "prod_cov_checkout",
                "price_cov_checkout",
                "Checkout Cov",
                "desc",
                1000,
                "usd",
                "acct_cov_fake123",
            ),
        )
        conn.commit()

    r = client.post(
        "/api/connect/checkout",
        json={"product_id": "prod_cov_checkout", "quantity": 1},
    )
    assert r.status_code == 200
    assert r.json().get("checkout_url")


def test_connect_webhook_missing_signature(fake_stripe):
    r = client.post("/api/connect/webhooks", content=b"{}")
    assert r.status_code == 400


def test_connect_webhook_invalid_signature(monkeypatch):
    import routes.stripe_connect_v2 as connect_mod

    monkeypatch.setattr(connect_mod, "_stripe_client", None)
    monkeypatch.setattr(connect_mod, "STRIPE_WEBHOOK_SECRET", "whsec_cov_test_secret")
    r = client.post(
        "/api/connect/webhooks",
        content=b'{"id":"evt_1"}',
        headers={"stripe-signature": "bad-sig", "Content-Type": "application/json"},
    )
    assert r.status_code in (400, 502)


def test_connect_webhook_handled(fake_stripe, monkeypatch):
    import routes.stripe_connect_v2 as connect_mod

    monkeypatch.setattr(connect_mod, "STRIPE_WEBHOOK_SECRET", "whsec_cov_test_secret")

    class _Thin:
        type = "v2.core.account[requirements].updated"
        id = "evt_cov_fake"

    fake = connect_mod._get_stripe_client()
    monkeypatch.setattr(
        fake,
        "parse_event_notification",
        lambda payload, sig, secret: _Thin(),
    )

    r = client.post(
        "/api/connect/webhooks",
        content=b'{"id":"evt_cov_fake"}',
        headers={
            "stripe-signature": "t=1,v1=fake",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True