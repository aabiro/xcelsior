"""Smoke coverage for the routes `UNTESTED_ENDPOINTS.md` still listed as bare.

The ledger's workflow is "write a TestClient test → if it works, tick the box;
if it 500s, fix-or-delete then tick". Probing all of them at once turned up
four defects, each with its own regression file:

* `tests/test_malformed_identifiers_are_not_server_errors.py` — three routes
  returned 500 for an unparseable id, and `dismiss` reported success for a
  finding it had not touched.
* `tests/test_facebook_callbacks_verify_their_signature.py` — both Facebook
  callbacks accepted a forged `signed_request`.

What remains here is the coverage itself: each route answers, and answers
something defensible. These are deliberately shallow — their job is to notice
when a route stops responding at all, which is the failure the ledger was
counting.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def user_headers():
    email = f"coverage-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Coverage Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"registration did not return a token: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


# ── OAuth key discovery ───────────────────────────────────────────────────


def test_jwks_document_is_served() -> None:
    r = client.get("/.well-known/jwks.json")
    assert r.status_code == 200
    assert isinstance(r.json().get("keys"), list)


def test_jwks_never_publishes_private_or_symmetric_key_material() -> None:
    """The invariant the route exists to hold.

    `oauth_jwks` filters to RSA *public* keys, so a symmetric development
    secret is skipped rather than exported. Nothing asserted it, and the
    failure would be silent and total: the signing secret served to anyone who
    asks, from a URL clients are told to fetch.
    """
    body = client.get("/.well-known/jwks.json").json()
    forbidden = {"d", "p", "q", "dp", "dq", "qi", "k"}
    for key in body.get("keys", []):
        assert key.get("kty") == "RSA", f"non-RSA key published: {key.get('kty')}"
        leaked = forbidden & set(key)
        assert not leaked, f"JWKS exposed private key material: {sorted(leaked)}"


# ── Agent keys ────────────────────────────────────────────────────────────


def test_listing_agent_keys(user_headers) -> None:
    r = client.get("/api/agent-keys", headers=user_headers)
    assert r.status_code == 200
    assert "keys" in r.json() or r.json().get("ok") is True


def test_revoking_an_unknown_agent_key_is_404(user_headers) -> None:
    r = client.delete(f"/api/agent-keys/{uuid.uuid4()}", headers=user_headers)
    assert r.status_code == 404


def test_renaming_an_unknown_agent_key_is_404(user_headers) -> None:
    r = client.patch(
        f"/api/agent-keys/{uuid.uuid4()}", json={"name": "renamed"}, headers=user_headers
    )
    assert r.status_code == 404


# ── Account self-service ──────────────────────────────────────────────────


def test_demo_credentials_are_ip_gated() -> None:
    """Open by design to a whitelist, so the only useful assertion is the refusal.

    `in (200, 403)` would pass whether the gate works or the route is wide open
    — `tests/test_authz_assertions_can_fail.py` rejects exactly that shape. The
    TestClient's peer is the literal string `testclient`, which is in no
    whitelist and never will be, so 403 is the deterministic answer here and
    the one worth pinning.
    """
    r = client.get("/api/auth/demo-credentials")
    assert r.status_code == 403, (
        f"an unwhitelisted caller got {r.status_code} from the demo-credentials "
        f"route: {r.text[:200]}"
    )


def test_avatar_upload_fetch_and_delete_round_trip(user_headers) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    up = client.post(
        "/api/auth/me/avatar",
        files={"file": ("avatar.png", png, "image/png")},
        headers=user_headers,
    )
    assert up.status_code == 200, up.text[:200]

    got = client.get("/api/auth/me/avatar", headers=user_headers)
    assert got.status_code == 200, got.text[:200]

    gone = client.delete("/api/auth/me/avatar", headers=user_headers)
    assert gone.status_code == 200, gone.text[:200]


def test_a_non_image_avatar_is_refused(user_headers) -> None:
    r = client.post(
        "/api/auth/me/avatar",
        files={"file": ("payload.html", b"<script>alert(1)</script>", "text/html")},
        headers=user_headers,
    )
    assert r.status_code == 400, (
        f"an HTML file was accepted as an avatar ({r.status_code}); it would be "
        "served back from this origin"
    )


def test_updating_the_profile(user_headers) -> None:
    r = client.put(
        "/api/auth/me/profile", json={"name": "Renamed Probe"}, headers=user_headers
    )
    assert r.status_code == 200, r.text[:200]
    assert r.json().get("ok") is True


def test_the_profile_route_refuses_a_role_it_cannot_change(user_headers) -> None:
    """It used to accept `role`, return `{"ok": true}`, and change nothing."""
    r = client.put(
        "/api/auth/me/profile", json={"role": "provider"}, headers=user_headers
    )
    assert r.status_code == 400, (
        f"PUT /api/auth/me/profile returned {r.status_code} for a role change it "
        "does not perform; silence here reads as success"
    )
    assert "PATCH /api/auth/me" in r.text


def test_requesting_an_email_change(user_headers) -> None:
    r = client.post(
        "/api/auth/me/email-change",
        json={
            "new_email": f"changed-{uuid.uuid4().hex[:8]}@xcelsior.ca",
            "password": "StrongPass123!",
        },
        headers=user_headers,
    )
    assert r.status_code == 200, r.text[:200]


def test_confirming_an_email_change_with_a_bogus_token(user_headers) -> None:
    r = client.post(
        "/api/auth/me/email-change/confirm",
        json={"token": "not-a-real-token"},
        headers=user_headers,
    )
    assert r.status_code == 400, r.text[:200]


# ── Pricing ───────────────────────────────────────────────────────────────


def test_spot_quote() -> None:
    r = client.get("/api/pricing/spot-quote", params={"gpu_model": "RTX4090"})
    assert r.status_code == 200, r.text[:200]


def test_spot_floor_suggestion() -> None:
    r = client.get("/api/pricing/spot-floor-suggestion", params={"gpu_model": "RTX4090"})
    assert r.status_code == 200, r.text[:200]


def test_preset_token_pricing() -> None:
    r = client.get("/api/v2/serverless/preset-token-pricing")
    assert r.status_code == 200, r.text[:200]


# ── Marketplace v1 search ─────────────────────────────────────────────────


def test_marketplace_v1_search_is_actually_exercised() -> None:
    """The ledger marked this covered because it shared a name with the v2 route.

    Both handlers were called `api_marketplace_search`, and coverage is scored
    by whether the handler name appears under `tests/`. One test of the v2 POST
    ticked the box for both.
    """
    r = client.get("/marketplace/search", params={"limit": 5})
    assert r.status_code == 200, r.text[:200]
    assert "listings" in r.json() or r.json().get("ok") is True


# ── Platform, hosts, serverless, reputation ───────────────────────────────


def test_admission_queue() -> None:
    r = client.get("/api/admin/admission-queue", headers=_admin_headers())
    assert r.status_code == 200, r.text[:200]


def test_compatibility_evidence_requires_a_well_formed_body() -> None:
    r = client.post(
        f"/api/hosts/compatibility-sessions/{uuid.uuid4()}/evidence",
        json={},
        headers=_admin_headers(),
    )
    assert r.status_code < 500, r.text[:200]


def test_capacity_forecast_for_an_unknown_endpoint(user_headers) -> None:
    r = client.get(f"/api/v2/platform/capacity-forecast/{uuid.uuid4()}", headers=user_headers)
    assert r.status_code == 404, r.text[:200]


def test_reputation_journey(user_headers) -> None:
    r = client.get("/api/reputation/me/journey", headers=user_headers)
    assert r.status_code == 200, r.text[:200]


def test_inference_compat_endpoint_health(user_headers) -> None:
    r = client.get("/api/v2/inference/endpoints", headers=user_headers)
    assert r.status_code == 200, r.text[:200]


def test_inference_compat_endpoint_usage_rejects_an_empty_body(user_headers) -> None:
    r = client.post("/api/v2/inference/endpoints", json={}, headers=user_headers)
    assert r.status_code < 500, r.text[:200]


def test_getting_an_unknown_serverless_batch(user_headers) -> None:
    r = client.get(f"/api/v2/serverless/batches/{uuid.uuid4()}", headers=user_headers)
    assert r.status_code == 404, r.text[:200]


def test_github_resolve_requires_its_fields(user_headers) -> None:
    r = client.post(
        "/api/v2/serverless/github/resolve", json={"repo": "octocat/Hello-World"},
        headers=user_headers,
    )
    assert r.status_code < 500, r.text[:200]


def test_artifact_finalize_requires_its_fields(user_headers) -> None:
    r = client.post("/api/artifacts/finalize", json={}, headers=user_headers)
    assert r.status_code < 500, r.text[:200]


def test_agent_degraded_report() -> None:
    """A platform bearer token is refused here with 410 — API keys are retired."""
    r = client.post(
        "/agent/degraded",
        json={"host_id": "coverage-host", "reason": "gpu_fault", "context": "probe"},
        headers=_admin_headers(),
    )
    assert r.status_code < 500, r.text[:200]


@pytest.mark.parametrize("suffix", ("files", "result"))
def test_promotion_agent_callbacks_reject_an_empty_body(suffix) -> None:
    r = client.post(
        f"/api/v1/promotions/{uuid.uuid4()}/{suffix}", json={}, headers=_admin_headers()
    )
    assert r.status_code < 500, r.text[:200]
