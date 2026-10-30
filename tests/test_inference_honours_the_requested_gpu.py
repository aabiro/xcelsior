"""`gpu_model` reached the API and stopped there.

`InferenceRequest.gpu_model` is documented in the schema as "Preferred GPU
model or 'any'", so it is advertised, validated, and shown in the generated
OpenAPI. `api_inference_submit` never passed it to `submit_job`, so every
request — including one that names an H100 — was scheduled against whatever
host won the ordinary scoring.

The scheduler had already decided this must not happen.
`scheduler._gpu_model_candidates` narrows to exact matches for an explicit
request, and its docstring says callers that will accept any sufficiently
capable card "must omit ``gpu_model`` instead of silently substituting
hardware". This route substituted hardware for every caller that named one, and
the caller is billed for whatever it got.

Asserted at the `submit_job` boundary rather than end to end: what went wrong
was an argument that was never passed, and that is the thing to pin.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import routes.inference as inference_routes
from api import app

client = TestClient(app)


@pytest.fixture
def auth_headers():
    email = f"gpu-pref-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "GPU Pref"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def captured_submit(monkeypatch):
    """Record what the route asks the scheduler for.

    The wallet pre-flight runs before scheduling and 402s on a zero balance, so
    a funded wallet is stubbed in — the subject here is the argument passed to
    `submit_job`, not the billing gate, which has its own tests.
    """
    calls: list[dict] = []

    def _fake_submit_job(**kwargs):
        calls.append(kwargs)
        return {"job_id": f"job-{uuid.uuid4().hex[:8]}"}

    class _FundedWallet:
        def get_wallet(self, _customer_id):
            return {"status": "active", "balance_cad": 100.0, "grace_until": 0}

    import billing

    monkeypatch.setattr(billing, "get_billing_engine", lambda: _FundedWallet())
    monkeypatch.setattr(inference_routes, "submit_job", _fake_submit_job)
    monkeypatch.setattr(inference_routes, "store_inference_job", lambda **_: None)
    monkeypatch.setattr(inference_routes, "broadcast_sse", lambda *_a, **_k: None)
    return calls


def _submit(headers, **extra):
    return client.post(
        "/api/inference",
        json={"model": "distilbert-base-uncased", "inputs": "hello", **extra},
        headers=headers,
    )


def test_a_named_gpu_reaches_the_scheduler(auth_headers, captured_submit) -> None:
    r = _submit(auth_headers, gpu_model="RTX4090")
    assert r.status_code == 200, r.text[:300]
    assert captured_submit, "submit_job was never called"
    assert captured_submit[-1].get("gpu_model") == "RTX4090", (
        "the requested GPU was dropped between the request model and the "
        f"scheduler: submit_job got gpu_model={captured_submit[-1].get('gpu_model')!r}. "
        "The caller is billed for whatever card it silently ran on instead."
    )


def _assert_resolved_to_a_priced_gpu(call: dict, value) -> None:
    """No preference resolves to a real Stripe-priced variant, never "any".

    Every customer launch binds a Stripe Price, and a Price belongs to one GPU
    variant, so "I don't care" cannot reach the scheduler as an omitted model
    any more. It must not reach it as a model literally named "any" either:
    that would match no host and strand the job.
    """
    from db import pg_connection

    model, price_id = call.get("gpu_model"), call.get("service_tier_price_id")
    assert model and model.strip().lower() != "any", (
        f"gpu_model={value!r} became {model!r}; an exact-match filter on that "
        "name selects nothing"
    )
    with pg_connection() as conn:
        row = conn.execute(
            """SELECT 1 FROM gpu_pricing p
                 JOIN stripe_compute_prices s ON s.stripe_price_id=p.stripe_price_id
                WHERE p.gpu_model=%s AND p.stripe_price_id=%s AND p.active AND s.active
                  AND p.tier='standard' AND p.pricing_mode='on_demand'""",
            (model, price_id),
        ).fetchone()
    assert row, f"{model!r} with price {price_id!r} is not an active Standard Stripe Price"


@pytest.mark.parametrize("value", ["any", "ANY", " any ", ""])
def test_no_preference_resolves_to_a_priced_gpu(auth_headers, captured_submit, value) -> None:
    """ "any" is the documented way to say "I don't care"; it is bound to the
    cheapest active Standard Stripe Price instead of a model named "any"."""
    r = _submit(auth_headers, gpu_model=value)
    assert r.status_code == 200, r.text[:300]
    _assert_resolved_to_a_priced_gpu(captured_submit[-1], value)


def test_the_default_resolves_to_a_priced_gpu(auth_headers, captured_submit) -> None:
    """The field defaults to "any", so omitting it must behave the same way."""
    r = _submit(auth_headers)
    assert r.status_code == 200, r.text[:300]
    _assert_resolved_to_a_priced_gpu(captured_submit[-1], None)
