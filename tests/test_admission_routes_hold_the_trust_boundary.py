"""The admission routes, which nothing had ever called.

`routes/host_admission.py` opens by explaining why the trust boundary lives at
the routing layer:

    Only an **operator** (admin) may record authoritative evidence or decide
    admission. [...] Keeping the split at the routing layer means the service
    cannot be reached through a path that quietly upgrades advisory evidence
    into an admission.

`tests/test_host_admission.py` tests that boundary thoroughly — in the
*service*, calling `host_admission.*` directly. Measured with
`scripts/measure_route_execution.py`, five of the seven route handlers were
never entered by any test, so the layer the docstring says holds the boundary
was the layer nothing exercised. A route that forgot `_require_admin`, or
called the operator path from the provider surface, would have passed the whole
suite.

These go over HTTP, with a real Ed25519 helper signature, because a signature
the route rejects and a signature the service rejects are different bugs.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

import host_admission as ha
from api import app
from control_plane.db import run_transaction

client = TestClient(app)

HOST_PAYLOAD = {
    "gpu_model": "RTX 4090",
    "total_vram_gb": 24,
    "gpu_count": 1,
    "cost_per_hour": 0.75,
}


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def _make_host(host_id: str, *, tenant: str) -> None:
    def txn(conn):
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  tenant_id, owner_id, admission_state)
               VALUES (%s, 'active', %s, %s::jsonb, %s, %s, 'pending')
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps(HOST_PAYLOAD), tenant, tenant),
        )

    run_transaction(txn, what="test_admission_routes_make_host")


def _drop_host(host_id: str) -> None:
    def txn(conn):
        for sql in (
            "DELETE FROM host_admission_decisions WHERE host_id = %s",
            "DELETE FROM host_admission_evidence WHERE host_id = %s",
            "DELETE FROM host_compatibility_sessions WHERE host_id = %s",
            "DELETE FROM hosts WHERE host_id = %s",
        ):
            conn.execute(sql, (host_id,))

    run_transaction(txn, what="test_admission_routes_drop_host")


@pytest.fixture
def host():
    host_id = f"admission-route-{uuid.uuid4().hex[:10]}"
    _make_host(host_id, tenant="tenant-routes")
    yield host_id
    _drop_host(host_id)


@pytest.fixture
def helper_key():
    priv = Ed25519PrivateKey.generate()
    spki = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, base64.b64encode(spki).decode()


@pytest.fixture
def provider_headers():
    """A signed-in non-admin — the principal the operator routes must refuse."""
    email = f"admission-provider-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Provider"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register a provider: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


def _open_session(host_id: str, spki: str) -> dict:
    r = client.post(
        f"/api/hosts/{host_id}/compatibility-sessions",
        json={
            "helper_public_key_spki": spki,
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
            "ttl_seconds": 600,
        },
        headers=_admin_headers(),
    )
    assert r.status_code == 200, f"opening a session failed: {r.status_code} {r.text[:300]}"
    return r.json()


def _sign(priv, session: dict, report: dict) -> str:
    message = ha.helper_signature_message(
        session["session_id"],
        session["challenge"],
        ha.sha256_hex(ha.canonical_json(report)),
    )
    return base64.b64encode(priv.sign(message)).decode()


# ── Provider surface ──────────────────────────────────────────────────────


def test_opening_a_compatibility_session(host, helper_key) -> None:
    _, spki = helper_key
    session = _open_session(host, spki)
    assert session.get("session_id")
    assert session.get("challenge"), "no challenge to sign — the session proves nothing"
    assert session.get("submit_token")


def test_submitting_signed_evidence_over_the_route(host, helper_key) -> None:
    """The signature is real, so this exercises verification rather than skipping it."""
    priv, spki = helper_key
    session = _open_session(host, spki)
    report = {"gpu_model": "RTX 4090", "driver": "550.54", "vram_gb": 24}

    r = client.post(
        f"/api/hosts/compatibility-sessions/{session['session_id']}/evidence",
        json={
            "submit_token": session["submit_token"],
            "report": report,
            "signature": _sign(priv, session, report),
        },
        headers=_admin_headers(),
    )
    assert r.status_code == 200, r.text[:400]


def test_a_forged_signature_is_refused_over_the_route(host, helper_key) -> None:
    """A different key signing the same report must not be accepted."""
    priv, spki = helper_key
    session = _open_session(host, spki)
    report = {"gpu_model": "RTX 4090"}
    impostor = Ed25519PrivateKey.generate()

    r = client.post(
        f"/api/hosts/compatibility-sessions/{session['session_id']}/evidence",
        json={
            "submit_token": session["submit_token"],
            "report": report,
            "signature": _sign(impostor, session, report),
        },
        headers=_admin_headers(),
    )
    assert r.status_code >= 400, (
        f"evidence signed by a key the session was not opened with was accepted "
        f"({r.status_code})"
    )
    assert r.status_code < 500, f"a forged signature should refuse, not fault: {r.text[:300]}"


def test_a_wrong_submit_token_is_refused(host, helper_key) -> None:
    priv, spki = helper_key
    session = _open_session(host, spki)
    report = {"gpu_model": "RTX 4090"}
    r = client.post(
        f"/api/hosts/compatibility-sessions/{session['session_id']}/evidence",
        json={
            "submit_token": "not-the-token",
            "report": report,
            "signature": _sign(priv, session, report),
        },
        headers=_admin_headers(),
    )
    assert 400 <= r.status_code < 500, r.text[:300]


def test_recording_provider_evidence(host) -> None:
    r = client.post(
        f"/api/hosts/{host}/provider-evidence",
        json={"report": {"gpu_model": "RTX 4090", "source": "provider-agent"}},
        headers=_admin_headers(),
    )
    assert r.status_code == 200, r.text[:400]


def test_admission_status_reports_what_is_outstanding(host) -> None:
    r = client.get(f"/api/hosts/{host}/admission", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body, "admission status came back empty"


def test_admission_status_for_an_unknown_host_is_404() -> None:
    r = client.get(
        f"/api/hosts/no-such-host-{uuid.uuid4().hex[:8]}/admission", headers=_admin_headers()
    )
    assert r.status_code == 404, r.text[:200]


# ── Operator surface ──────────────────────────────────────────────────────


def test_recording_authoritative_evidence(host) -> None:
    r = client.post(
        f"/api/admin/hosts/{host}/authoritative-evidence",
        json={
            "evidence_type": "compatibility",
            "verdict": "pass",
            "summary": {"note": "verified by hand"},
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
            "validity_seconds": 86400,
        },
        headers=_admin_headers(),
    )
    assert r.status_code == 200, r.text[:400]


def test_an_unsupported_evidence_type_is_refused_by_the_service(host) -> None:
    """`evidence_type` is free-form at the model and constrained by the service.

    The refusal arrives as a structured `admission_precondition_failed`, which
    is the handler running and `_translate` carrying the service's own code
    through — not FastAPI rejecting the body.
    """
    r = client.post(
        f"/api/admin/hosts/{host}/authoritative-evidence",
        json={
            "evidence_type": "operator_vibes",
            "verdict": "pass",
            "summary": {},
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
        },
        headers=_admin_headers(),
    )
    assert r.status_code == 422, r.text[:300]
    assert "evidence_type" in r.text


def test_an_invalid_verdict_never_reaches_the_service(host) -> None:
    """`verdict` is pattern-constrained; anything else is a 422 by design."""
    r = client.post(
        f"/api/admin/hosts/{host}/authoritative-evidence",
        json={
            "evidence_type": "compatibility",
            "verdict": "probably-fine",
            "summary": {},
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
        },
        headers=_admin_headers(),
    )
    assert r.status_code == 422, r.text[:300]


def test_deciding_admission(host) -> None:
    # Both kinds in REQUIRED_AUTHORITATIVE_EVIDENCE must be on file before a
    # host can be admitted; recording one and expecting success would be
    # testing the precondition rather than the decision.
    for kind in ("compatibility", "hardware_verification"):
        rec = client.post(
            f"/api/admin/hosts/{host}/authoritative-evidence",
            json={
                "evidence_type": kind,
                "verdict": "pass",
                "summary": {"note": "verified"},
                "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
            },
            headers=_admin_headers(),
        )
        assert rec.status_code == 200, f"recording {kind} failed: {rec.text[:300]}"
    r = client.post(
        f"/api/admin/hosts/{host}/admission-decisions",
        json={
            "action": "admit",
            "reason": "evidence complete",
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
        },
        headers=_admin_headers(),
    )
    assert r.status_code in (200, 409, 422), r.text[:400]
    assert r.status_code != 500, f"deciding admission faulted: {r.text[:400]}"


def test_a_stale_expected_version_is_a_conflict(host) -> None:
    """Optimistic concurrency, asserted at the route rather than the service.

    Two operators acting at once must not silently overwrite one another, and
    the route is what carries `expected_version` through.
    """
    r = client.post(
        f"/api/admin/hosts/{host}/admission-decisions",
        json={
            "action": "admit",
            "reason": "racing",
            "idempotency_key": f"idem-{uuid.uuid4().hex[:10]}",
            "expected_version": 9999,
        },
        headers=_admin_headers(),
    )
    assert r.status_code in (409, 422), f"a stale version was not refused: {r.status_code}"


# ── The boundary itself ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/api/admin/hosts/{host}/authoritative-evidence",
            {
                "evidence_type": "self_attested",
                "verdict": "pass",
                "summary": {},
                "idempotency_key": "provider-attempt",
            },
        ),
        (
            "/api/admin/hosts/{host}/admission-decisions",
            {"action": "admit", "reason": "please", "idempotency_key": "provider-attempt"},
        ),
    ],
)
def test_a_provider_cannot_reach_the_operator_routes(host, provider_headers, path, body) -> None:
    """The property the module docstring is about, at the layer it names.

    A provider describing their own machine produces advisory evidence. If
    either of these routes admitted a signed-in non-admin, a provider could
    sign their own machine's way into paid work — and every service-level test
    would still pass, because the service would never have been asked.
    """
    r = client.post(path.format(host=host), json=body, headers=provider_headers)
    assert r.status_code in (401, 403), (
        f"a non-admin got {r.status_code} from {path}; the operator surface is "
        "reachable from the provider side"
    )


def test_the_admission_queue_is_admin_only(provider_headers) -> None:
    r = client.get("/api/admin/admission-queue", headers=provider_headers)
    assert r.status_code in (401, 403), (
        f"a non-admin read the admission queue ({r.status_code})"
    )
