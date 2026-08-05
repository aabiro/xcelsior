"""Card data and payment credentials must not survive into a log line.

Gate P1: *"card data, `client_secret`, and processor tokens appear in no tool
result, log, trace, audit row, or error string. Canary-tested with fake PANs."*

The scrubber already covered emails, `cus_` ids, JWTs and Bearer tokens. It did
not cover the things the billing surface actually handles:

* **Stripe client secrets** — `pi_..._secret_...` is a bearer credential for
  *confirming a payment*. Anyone holding one can complete or cancel that intent.
  They arrive inside exception strings, which is precisely where a handler logs
  them, and the SCA path added this week produces one on every declined charge.
* **`sk_live_` / `whsec_`** — the API key and the webhook signing secret. A
  traceback from a misconfigured client is the likely carrier.
* **`xcel_ai_` / `xoa_`** — this platform's own credentials. Agent keys are
  pasted into editor configs and never expire, so one in a log is a durable
  credential sitting in a file other people can read.
* **PANs** — the thing the gate names explicitly.

**Fake material only.** Every value here is either a Stripe-published test
number or a random string in the right shape. The card numbers are the standard
test PANs, which are valid under Luhn — that matters, because Luhn is what
distinguishes a card number from a timestamp.

**Both directions are asserted.** A scrubber that redacted every long digit
string would pass every leak test and make the logs useless, so ordinary numbers
— timestamps, job ids, byte counts — must survive untouched. That is the
failure mode that gets a scrubber switched off, and a switched-off scrubber
redacts nothing.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

from log_pii_filter import _scrub  # noqa: E402

#: Stripe's published test card numbers. Real shape, Luhn-valid, not real cards.
TEST_PANS = [
    "4242424242424242",  # Visa
    "5555555555554444",  # Mastercard
    "378282246310005",  # Amex, 15 digits
    "6011111111111117",  # Discover
    "4000002500003155",  # the 3DS-required test card
]

#: Written the way a human or a form would.
FORMATTED_PANS = ["4242 4242 4242 4242", "4242-4242-4242-4242"]

SECRETS = [
    ("pi_3SCAprobe000000_secret_abc123XYZ456", "a PaymentIntent client secret"),
    ("seti_1Nprobe0000000_secret_def789ABC012", "a SetupIntent client secret"),
    ("sk_live_51Habcdefghijklmnopqrstuvwx", "a live Stripe secret key"),
    ("sk_test_51Habcdefghijklmnopqrstuvwx", "a test Stripe secret key"),
    ("whsec_abcdefghijklmnopqrstuvwxyz0123", "a webhook signing secret"),
    ("xcel_ai_abcdefghijklmnopqrstuvwxyz01", "an agent API key"),
    ("xoa_abcdefghijklmnopqrstuvwxyz012345", "an OAuth access token"),
]


@pytest.mark.parametrize("pan", TEST_PANS)
def test_a_card_number_never_survives_scrubbing(pan):
    """The canary the gate names."""
    scrubbed = _scrub(f"charge failed for card {pan} on customer acct")
    assert pan not in scrubbed, f"the PAN {pan[:4]}... survived into the log line"
    assert pan[-4:] in scrubbed, (
        "the last four digits should remain — they identify the card to a human "
        "and are what Stripe itself stores for display"
    )


@pytest.mark.parametrize("pan", FORMATTED_PANS)
def test_a_spaced_or_hyphenated_card_number_is_caught(pan):
    """People and forms write them with separators; regexes often miss that."""
    scrubbed = _scrub(f"user pasted {pan} into the chat")
    assert pan not in scrubbed
    assert "4242424242424242" not in scrubbed.replace(" ", "").replace("-", "")


@pytest.mark.parametrize("secret,what", SECRETS)
def test_no_payment_credential_survives_scrubbing(secret, what):
    """Each is a credential, not merely sensitive-looking."""
    scrubbed = _scrub(f"request failed: {secret} rejected by the processor")
    assert secret not in scrubbed, f"{what} survived into the log line"


def test_a_client_secret_inside_an_exception_string_is_caught():
    """The realistic carrier, and the one this week's work added.

    An SCA decline raises a `CardError` whose message can carry the declined
    intent, and the handler logs `str(exc)`. That is the exact path — not a
    deliberate log of a secret, but a secret riding inside an error.
    """
    message = (
        "CardError: authentication required for "
        "pi_3SCAprobe000000_secret_abc123XYZ456 (declined)"
    )
    scrubbed = _scrub(message)
    assert "_secret_" not in scrubbed
    assert "authentication required" in scrubbed, (
        "the diagnostic text was destroyed along with the secret; a scrubbed "
        "log still has to be readable or it will be turned off"
    )


def test_ordinary_long_numbers_are_left_alone():
    """The calibration control, and the reason for the Luhn check.

    A scrubber that redacts every long digit string passes every leak test and
    makes the logs useless — which is how it ends up disabled, redacting
    nothing. Luhn is what separates a card number from a timestamp.
    """
    benign = (
        "job 1785859874797 finished after 1234567890123 ns, "
        "wrote 987654321098765 bytes, request 20260804195512"
    )
    assert _scrub(benign) == benign, (
        "ordinary numbers were redacted; this scrubber will be switched off"
    )


def test_the_filter_scrubs_a_real_log_record():
    """End to end through `logging`, not just the helper.

    The helper being correct is not the claim — the claim is that a record
    emitted by application code comes out clean.
    """
    from log_pii_filter import PIIScrubFilter

    record = logging.LogRecord(
        name="xcelsior.billing",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="top-up failed for card %s using %s",
        args=("4242424242424242", "sk_live_51Habcdefghijklmnopqrstuvwx"),
        exc_info=None,
    )
    PIIScrubFilter().filter(record)
    emitted = record.getMessage()
    assert "4242424242424242" not in emitted
    assert "sk_live_51Habcdefghijklmnopqrstuvwx" not in emitted
    assert "top-up failed" in emitted
