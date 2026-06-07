"""PayPal marketplace vs wallet webhook routing."""

from __future__ import annotations

import routes.billing as billing_mod


def test_marketplace_custom_id_detected():
    assert billing_mod._paypal_is_marketplace_custom_id("host-1:job-42") is True
    assert billing_mod._paypal_is_marketplace_custom_id("user@example.com") is False