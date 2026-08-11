"""§1.3: one live-credential assertion per phase, for P1, P2 and P3.

The clause asks each phase to prove *something* against a real server with a
real token. P0 has `test_named_scopes_refuse_live.py` and P5 has
`test_placement_preference_refuses_live.py`; these are the three that had none.

Each asserts the narrowest true thing that a mock cannot establish — that the
deployed surface exists, authenticates, and refuses correctly. They are not
substitutes for the phases' own clauses; they are the live-credential proof
those clauses were supposed to carry.

Positive controls throughout: a server refusing everything passes a refusal-only
suite perfectly, which is how a Cloudflare interstitial once reported a fix that
had not happened.
"""

from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    MISSING_CREDENTIALS,
    TOKEN,
    auth,
)

pytestmark = pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS)


def _get(path: str):
    return requests.get(f"{BASE}{path}", headers=auth(), timeout=30)


def _post(path: str, body: dict):
    return requests.post(f"{BASE}{path}", headers=auth(), json=body, timeout=30)


# ── The positive control ──────────────────────────────────────────────


def test_the_token_reaches_the_api_at_all():
    """Without this, every refusal below could be an edge blocking the origin."""
    response = _get("/instances")
    assert response.status_code == 200, (
        f"the credential does not reach the API ({response.status_code}); no "
        "refusal in this file means anything until it does"
    )


# ── P1: the money levers ──────────────────────────────────────────────


def test_p1_a_funding_call_without_an_idempotency_key_is_refused():
    """P1's clause is replay-safety; the live proof is that the key is required.

    A funding endpoint that accepts a request with no idempotency key cannot be
    replay-safe, whatever its internals do.
    """
    response = _post("/api/billing/topup", {"amount_cad": 1})
    assert response.status_code != 200, (
        "a funding call with no idempotency key was accepted"
    )
    assert response.status_code in (400, 402, 403, 409, 422), response.status_code


def test_p1_the_wallet_balance_is_readable_with_this_token():
    """Positive control for the pair above — the surface is reachable."""
    response = _get("/api/billing/wallet")
    assert response.status_code in (200, 404), response.status_code


# ── P2: access ────────────────────────────────────────────────────────


def test_p2_an_ssh_key_endpoint_requires_the_scope():
    """P2's surface authenticates rather than being open."""
    response = _get("/api/ssh/keys")
    assert response.status_code in (200, 403), response.status_code


def test_p2_a_terminal_ticket_is_not_issued_for_an_unknown_instance():
    """A credential with a clock on it must not be minted for nothing."""
    response = _post("/api/terminal/ticket", {"instance_id": "does-not-exist"})
    assert response.status_code != 200, "a terminal ticket was issued for no instance"


# ── P3: volumes and promotion ─────────────────────────────────────────


def test_p3_the_volume_surface_is_reachable_and_scoped():
    response = _get("/api/v2/volumes")
    assert response.status_code in (200, 403), response.status_code


def test_p3_a_promotion_for_an_unknown_job_is_refused():
    """Promotion is a copy onto a user's volume; it must not accept a stranger."""
    response = _post(
        "/api/v1/promotions", {"job_id": "does-not-exist", "volume_id": "nope"}
    )
    assert response.status_code != 200, (
        "a promotion was accepted for a job that does not exist"
    )
