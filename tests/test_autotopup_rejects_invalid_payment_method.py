"""Auto-top-up must not send unusable stored cards to the live Stripe API.

Production wallets carry test-fixture payment methods — `pm_x` on `topup-test-*`
customers was live in the database. Every sweep sent those to Stripe, got back
`No such PaymentMethod`, and counted it as a payment failure, walking the wallet
through its 3-strike budget until auto-top-up was disabled. That end state is
identical to a genuinely declined card, so bad data was indistinguishable from a
bad card.
"""

import pytest

from billing import is_plausible_payment_method_id


class TestPlausiblePaymentMethodId:
    @pytest.mark.parametrize(
        "pm",
        [
            "pm_1MqLiJLkdIwHu7ixUEgbFdYF",  # real-shaped PaymentMethod
            "card_1MqLiJLkdIwHu7ixUEgbFdYF",  # legacy card id
            "src_1MqLiJLkdIwHu7ixUEgbFdYF",  # legacy source id
            # Readable stand-ins used by tests against a mocked Stripe. These
            # must pass: an over-tight guard that forces test doubles to be
            # rewritten is one that gets loosened later by someone in a hurry.
            # `pm_replayprobe` is a real fixture in test_funding_replay_is_one_charge.
            "pm_replayprobe",
            "pm_testcard1",
        ],
    )
    def test_accepts_stripe_shaped_ids(self, pm):
        assert is_plausible_payment_method_id(pm) is True

    @pytest.mark.parametrize(
        "pm",
        [
            "pm_x",  # the fixture value found on production wallets
            "pm_",
            "",
            None,
            "tok_visa",  # a token is not a payment method
            "cus_1MqLiJLkdIwHu7ix",  # a customer id in the wrong column
            "not-an-id-at-all",
            "pm_short",  # right prefix, impossible length
        ],
    )
    def test_rejects_everything_stripe_could_not_recognise(self, pm):
        assert is_plausible_payment_method_id(pm) is False


class TestChargeSavedCardGuard:
    def test_refuses_to_charge_an_impossible_payment_method(self):
        """The guard lives in the shared charge path, so the manual top-up
        route is protected as well as the sweep."""
        import billing

        mgr = billing.BillingEngine.__new__(billing.BillingEngine)

        with pytest.raises(ValueError, match="not a valid Stripe payment method id"):
            mgr.charge_saved_card(
                "topup-test-e6161ec9",
                50_000_000,
                stripe_customer_id="cus_test",
                payment_method_id="pm_x",
                idempotency_key="test-key",
                description="Automatic wallet top-up",
            )

    def test_rejection_happens_before_any_network_call(self, monkeypatch):
        """A guard that fires after the API call would still burn a live
        request and a rate-limit slot, which is most of the harm."""
        import billing

        called = []

        class _Boom:
            class PaymentIntent:
                @staticmethod
                def create(*a, **kw):
                    called.append(kw)
                    raise AssertionError("Stripe must not be reached")

        monkeypatch.setattr(billing, "_get_pg_pool", lambda: None, raising=False)
        mgr = billing.BillingEngine.__new__(billing.BillingEngine)

        with pytest.raises(ValueError):
            mgr.charge_saved_card(
                "topup-test-55d64b09",
                50_000_000,
                stripe_customer_id="cus_test",
                payment_method_id="pm_x",
                idempotency_key="test-key",
                description="Automatic wallet top-up",
            )

        assert called == [], "Stripe was contacted despite an impossible payment method"
