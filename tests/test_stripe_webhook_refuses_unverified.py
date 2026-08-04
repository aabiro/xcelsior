"""An event whose signature cannot be verified must not be answered 200.

`POST /api/providers/webhook` returned `{"ok": True, **result}` unconditionally,
so a body with a bad signature — or none — got HTTP 200.

Stripe reads any 2xx as delivered. Two consequences, and the second is the one
that makes this unrecoverable rather than merely delayed:

1. The event is never retried. Stripe otherwise retries for ~3 days with
   exponential backoff in live mode.
2. The event never appears in `GET /v1/events?delivery_success=false`, which is
   the documented way to find events you missed. Answering 200 removes it from
   the only recovery mechanism there is. Events are retained 30 days; after that
   there is nothing to find.

So a rotated or misrouted signing secret does not slow these events down, it
loses them, invisibly. Stripe's own reference implementation returns 400 on
`SignatureVerificationError`.

This matters more than it looks. Auto-top-up completion, SCA recovery, and
Connect payout onboarding all take their completion signal from a webhook and
from nowhere else — the agent-native plan is explicit that returning from a
`return_url` proves nothing. A silently dropped event is a top-up that never
credits, or a provider stuck in `pending_requirements` forever.

**Statuses are distinguished deliberately**, because they mean different things
to the sender:

* `400` — the signature is wrong. The request is bad; Stripe retries and the
  event stays visible in the undelivered sweep.
* `503` — no secret is configured, or Stripe is disabled here. The *deployment*
  is wrong, not the request. Still retryable, so events arriving during a
  misconfiguration land once it is fixed.

A test that accepted "400 or 503" could not tell those apart, and would pass
with the secret permanently missing.
"""

from __future__ import annotations

import json
import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)

WEBHOOK_PATH = "/api/providers/webhook"


def _body() -> bytes:
    return json.dumps(
        {
            "id": f"evt_{uuid.uuid4().hex[:16]}",
            "type": "payment_intent.succeeded",
            "created": int(time.time()),
            "data": {"object": {"id": f"pi_{uuid.uuid4().hex[:16]}"}},
        }
    ).encode()


def _signed_with(secret: str, payload: bytes) -> str:
    """A syntactically valid `stripe-signature` computed with the wrong secret.

    Malformed headers are rejected by a different code path — the point here is
    a *well-formed* signature that simply does not verify, which is what a
    rotated secret or a misrouted destination actually produces.
    """
    import hashlib
    import hmac

    timestamp = int(time.time())
    signed_payload = f"{timestamp}.".encode() + payload
    digest = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


@pytest.fixture
def stripe_enabled_with_a_secret(monkeypatch):
    """Stripe on, with a signing secret that will not match the test's signature."""
    import stripe_connect as sc

    if not getattr(sc, "STRIPE_ENABLED", False) or sc.stripe is None:
        pytest.skip("stripe SDK not enabled in this environment")
    monkeypatch.setattr(
        sc, "_webhook_secret_candidates", lambda: ["whsec_the_configured_one"]
    )
    return sc


def test_a_wrong_signature_is_refused_with_400(stripe_enabled_with_a_secret):
    """The rule. Signed properly, with the wrong key."""
    payload = _body()
    r = client.post(
        WEBHOOK_PATH,
        content=payload,
        headers={
            "stripe-signature": _signed_with("whsec_an_entirely_different_secret", payload),
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 400, (
        f"an event whose signature did not verify was answered {r.status_code}. "
        "Stripe reads 2xx as delivered: it will not retry, and the event will "
        f"not appear in delivery_success=false. Body: {r.text[:200]}"
    )


def test_a_missing_signature_header_is_refused_with_400(stripe_enabled_with_a_secret):
    """No header at all is the same failure with less effort."""
    r = client.post(
        WEBHOOK_PATH, content=_body(), headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400, (
        f"an unsigned body was answered {r.status_code}: {r.text[:200]}"
    )


def test_a_missing_secret_is_503_not_400(monkeypatch):
    """The deployment is wrong, not the request — and it stays retryable.

    Asserted separately from the 400 cases so a single "4xx or 5xx" tolerance
    cannot hide a permanently unconfigured deployment.
    """
    import stripe_connect as sc

    if not getattr(sc, "STRIPE_ENABLED", False) or sc.stripe is None:
        pytest.skip("stripe SDK not enabled in this environment")
    monkeypatch.setattr(sc, "_webhook_secret_candidates", lambda: [])

    payload = _body()
    r = client.post(
        WEBHOOK_PATH,
        content=payload,
        headers={
            "stripe-signature": _signed_with("whsec_anything", payload),
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 503, (
        f"a deployment with no signing secret answered {r.status_code}; 503 keeps "
        f"the event retryable once the secret is set. Body: {r.text[:200]}"
    )


def test_the_startup_gate_refuses_a_live_integration_with_no_secret(monkeypatch):
    """Retrying only helps if someone notices. The boot is where they notice."""
    import stripe_connect as sc
    from control_plane import startup_validation as sv

    if not getattr(sc, "STRIPE_ENABLED", False) or sc.stripe is None:
        pytest.skip("stripe SDK not enabled in this environment")

    monkeypatch.setattr(sc, "_webhook_secret_candidates", lambda: [])
    finding = sv._check_stripe_webhook_secret()
    assert finding is not None and finding.severity == "error", (
        "Stripe is enabled with no webhook signing secret and the startup gate "
        "did not object; the deployment would boot and drop every event"
    )
    assert finding.code == "stripe_webhook_secret_missing"

    # The calibration control: a configured secret must not produce a finding,
    # or the gate would refuse every correct deployment and be turned off.
    monkeypatch.setattr(sc, "_webhook_secret_candidates", lambda: ["whsec_ok"])
    assert sv._check_stripe_webhook_secret() is None


def test_the_v2_connect_webhook_already_agrees():
    """Two webhooks, one contract.

    `routes/stripe_connect_v2.py` already refused an unverifiable signature with
    400 and a missing secret with 503; `routes/providers.py` did not. They now
    behave the same way, and this pins that so the pair cannot drift back apart.
    """
    import inspect

    from routes import stripe_connect_v2

    source = inspect.getsource(stripe_connect_v2)
    assert 'HTTPException(400, "Invalid webhook signature")' in source
    assert "503" in source
