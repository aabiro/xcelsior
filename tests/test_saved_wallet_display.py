from types import SimpleNamespace
from unittest.mock import Mock

from billing import BillingEngine
import stripe
import stripe_connect


def test_link_wallet_records_are_visible_but_remain_separate_from_saved_cards(monkeypatch):
    method = stripe.PaymentMethod.construct_from({"id": "pm-link", "type": "link", "link": {"email": "wallet@example.test"}}, "test-key")
    listing = Mock(return_value=SimpleNamespace(data=[method]))
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", SimpleNamespace(PaymentMethod=SimpleNamespace(list=listing)))
    engine = BillingEngine.__new__(BillingEngine)
    monkeypatch.setattr(engine, "get_wallet", lambda customer: {"stripe_customer_id": "cus-owned"})
    assert engine.list_wallet_payment_methods("owner") == [{"id": "pm-link", "type": "link", "email": "wallet@example.test"}]
    listing.assert_called_once_with(customer="cus-owned", type="link", limit=100)
