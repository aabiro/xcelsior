"""The MFA routes the suite never entered.

Measured with `scripts/measure_route_execution.py`, nine of the fifteen MFA
handlers were never reached by any test — every one of them scored "covered" by
`UNTESTED_ENDPOINTS.md`, because the ledger matches on the `/api/auth/mfa/`
path prefix and two tests mention it.

That is the second-factor surface: the code that decides whether a password
alone is enough to sign in. `tests/test_mfa_flow.py` and
`tests/test_mfa_removal_needs_a_human.py` between them touched only
`/api/auth/mfa/methods` and `/api/auth/mfa/all`.

The flows here are real rather than shaped to reach a line: setup returns a
`test_code` in the test environment and TOTP secrets are verifiable with
`pyotp`, so enrolment and login are driven end to end through the same requests
a browser makes. The passkey handlers are the exception — a genuine WebAuthn
attestation cannot be forged here, so those are driven to their own validation
and lookup paths, which is still the handler running and refusing rather than
FastAPI refusing on its behalf. Each says which it is.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

import api as _api_mod
import routes._deps as _deps_mod
import routes.auth as _auth_mod

from api import app
from db import MfaStore, auth_connection

PASSWORD = "MfaSurface123!"


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    """MFA is DB-backed; the unit-test default stores users in a process dict."""
    monkeypatch.setattr(_api_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_deps_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_auth_mod, "_USE_PERSISTENT_AUTH", True)
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    monkeypatch.setattr(_deps_mod, "_AUTH_RATE_LIMIT_REQUESTS", 5000)
    yield
    with auth_connection() as conn:
        conn.execute("DELETE FROM mfa_backup_codes")
        conn.execute("DELETE FROM mfa_methods")
        conn.execute("DELETE FROM mfa_challenges")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM oauth_refresh_tokens WHERE email LIKE 'mfa-surface-%'")
        conn.execute("DELETE FROM users WHERE email LIKE 'mfa-surface-%'")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def account(client):
    """A registered user and a bearer token for it."""
    email = f"mfa-surface-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD, "name": "MFA Surface"}
    )
    token = r.json().get("access_token")
    assert token, f"registration failed: {r.status_code} {r.text[:200]}"
    return {"email": email, "headers": {"Authorization": f"Bearer {token}"}}


def _enable_totp(client, account) -> str:
    """Run the real setup → verify enrolment. Returns the TOTP secret."""
    import pyotp

    setup = client.post("/api/auth/mfa/totp/setup", headers=account["headers"])
    assert setup.status_code == 200, setup.text[:300]
    secret = setup.json()["secret"]

    verify = client.post(
        "/api/auth/mfa/totp/verify",
        json={"code": pyotp.TOTP(secret).now()},
        headers=account["headers"],
    )
    assert verify.status_code == 200, f"TOTP enrolment failed: {verify.text[:300]}"
    return secret


# ── TOTP enrolment ────────────────────────────────────────────────────────


def test_totp_verify_enables_the_method_and_issues_backup_codes(client, account) -> None:
    import pyotp

    setup = client.post("/api/auth/mfa/totp/setup", headers=account["headers"])
    assert setup.status_code == 200, setup.text[:300]
    secret = setup.json()["secret"]

    r = client.post(
        "/api/auth/mfa/totp/verify",
        json={"code": pyotp.TOTP(secret).now()},
        headers=account["headers"],
    )
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["ok"] is True
    assert body["backup_codes"], (
        "enabling the first factor issued no backup codes — a user who loses the "
        "authenticator then has no way back into the account"
    )

    methods = client.get("/api/auth/mfa/methods", headers=account["headers"]).json()
    # The listing renames `method_type` to `type` on the way out.
    totp = [m for m in methods.get("methods", []) if m["type"] == "totp"]
    assert totp and totp[0]["enabled"], f"TOTP did not end up enabled: {methods}"
    assert methods["mfa_enabled"] is True


def test_totp_verify_refuses_a_wrong_code(client, account) -> None:
    client.post("/api/auth/mfa/totp/setup", headers=account["headers"])
    r = client.post(
        "/api/auth/mfa/totp/verify", json={"code": "000000"}, headers=account["headers"]
    )
    assert r.status_code == 400, (
        f"a wrong TOTP code returned {r.status_code}; the factor would be no factor"
    )


def test_totp_verify_without_setup_is_refused(client, account) -> None:
    r = client.post(
        "/api/auth/mfa/totp/verify", json={"code": "123456"}, headers=account["headers"]
    )
    assert r.status_code == 400
    assert "setup" in r.text.lower()


# ── SMS enrolment ─────────────────────────────────────────────────────────


def test_sms_setup_and_verify_round_trip(client, account) -> None:
    setup = client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165551234"},
        headers=account["headers"],
    )
    assert setup.status_code == 200, setup.text[:300]
    code = setup.json().get("test_code")
    assert code, "the test environment did not hand back the code to verify with"

    r = client.post(
        "/api/auth/mfa/sms/verify", json={"code": code}, headers=account["headers"]
    )
    assert r.status_code == 200, r.text[:300]


def test_sms_setup_refuses_a_non_e164_number(client, account) -> None:
    r = client.post(
        "/api/auth/mfa/sms/setup", json={"phone_number": "416-555-1234"}, headers=account["headers"]
    )
    assert r.status_code == 400, f"a non-E.164 number was accepted: {r.text[:200]}"


def test_sms_setup_twice_is_refused(client, account) -> None:
    """Otherwise a second enrolment silently replaces the registered number."""
    first = client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165551234"},
        headers=account["headers"],
    )
    assert first.status_code == 200, first.text[:200]
    client.post(
        "/api/auth/mfa/sms/verify",
        json={"code": first.json()["test_code"]},
        headers=account["headers"],
    )
    second = client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165559999"},
        headers=account["headers"],
    )
    assert second.status_code == 409, (
        f"a second SMS enrolment returned {second.status_code}; it would point the "
        "second factor at a different phone without the first being removed"
    )


def test_sms_verify_refuses_a_wrong_code(client, account) -> None:
    client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165551234"},
        headers=account["headers"],
    )
    r = client.post(
        "/api/auth/mfa/sms/verify", json={"code": "000000"}, headers=account["headers"]
    )
    assert r.status_code == 400


# ── Login challenge ───────────────────────────────────────────────────────


def _login_challenge(client, email: str) -> str:
    r = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body.get("mfa_required") is True, (
        f"login did not demand a second factor after enrolment: {body}"
    )
    return body["challenge_id"]


def test_login_with_totp_completes_through_the_challenge(client, account) -> None:
    """The whole point of the surface: password alone must not be enough."""
    import pyotp

    secret = _enable_totp(client, account)
    challenge_id = _login_challenge(client, account["email"])

    r = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "totp", "code": pyotp.TOTP(secret).now()},
    )
    assert r.status_code == 200, r.text[:300]
    assert r.json().get("access_token"), f"MFA completed without issuing a session: {r.text[:200]}"


def test_login_with_a_wrong_totp_code_is_refused(client, account) -> None:
    _enable_totp(client, account)
    challenge_id = _login_challenge(client, account["email"])
    r = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "totp", "code": "000000"},
    )
    assert r.status_code == 400
    assert "access_token" not in r.text


def test_sms_send_then_verify_completes_login(client, account) -> None:
    setup = client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165551234"},
        headers=account["headers"],
    )
    client.post(
        "/api/auth/mfa/sms/verify",
        json={"code": setup.json()["test_code"]},
        headers=account["headers"],
    )
    challenge_id = _login_challenge(client, account["email"])

    sent = client.post("/api/auth/mfa/sms/send", json={"challenge_id": challenge_id})
    assert sent.status_code == 200, sent.text[:300]
    login_code = sent.json().get("test_code")
    assert login_code

    r = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "sms", "code": login_code},
    )
    assert r.status_code == 200, r.text[:300]
    assert r.json().get("access_token")


def test_sms_verify_without_a_sent_code_is_refused(client, account) -> None:
    """The branch that would otherwise accept any code for an SMS factor."""
    setup = client.post(
        "/api/auth/mfa/sms/setup",
        json={"phone_number": "+14165551234"},
        headers=account["headers"],
    )
    client.post(
        "/api/auth/mfa/sms/verify",
        json={"code": setup.json()["test_code"]},
        headers=account["headers"],
    )
    challenge_id = _login_challenge(client, account["email"])
    r = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "sms", "code": "123456"},
    )
    assert r.status_code == 400
    assert "request one first" in r.text.lower() or "no sms code" in r.text.lower()


def test_sms_send_requires_an_enrolled_sms_factor(client, account) -> None:
    _enable_totp(client, account)  # TOTP only — no SMS
    challenge_id = _login_challenge(client, account["email"])
    r = client.post("/api/auth/mfa/sms/send", json={"challenge_id": challenge_id})
    assert r.status_code == 400
    assert "sms" in r.text.lower()


def test_sms_send_rejects_an_unknown_challenge(client) -> None:
    r = client.post("/api/auth/mfa/sms/send", json={"challenge_id": "no-such-challenge"})
    assert r.status_code == 400


def test_a_backup_code_completes_login_once(client, account) -> None:
    """Single-use is the property that matters — a reusable one is a password."""
    import pyotp

    setup = client.post("/api/auth/mfa/totp/setup", headers=account["headers"])
    secret = setup.json()["secret"]
    enrol = client.post(
        "/api/auth/mfa/totp/verify",
        json={"code": pyotp.TOTP(secret).now()},
        headers=account["headers"],
    )
    backup_code = enrol.json()["backup_codes"][0]

    challenge_id = _login_challenge(client, account["email"])
    first = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "backup", "code": backup_code},
    )
    assert first.status_code == 200, first.text[:300]

    second_challenge = _login_challenge(client, account["email"])
    second = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": second_challenge, "method": "backup", "code": backup_code},
    )
    assert second.status_code == 400, (
        "the same backup code was accepted twice; a backup code that survives use "
        "is a second password"
    )


def test_an_unsupported_method_is_refused(client, account) -> None:
    _enable_totp(client, account)
    challenge_id = _login_challenge(client, account["email"])
    r = client.post(
        "/api/auth/mfa/verify",
        json={"challenge_id": challenge_id, "method": "carrier-pigeon", "code": "123456"},
    )
    assert r.status_code == 400


# ── Passkeys ──────────────────────────────────────────────────────────────
#
# A real WebAuthn attestation is signed by an authenticator this process does
# not have, so these drive the handlers to their own lookup and validation
# paths. That is the handler running and refusing — not FastAPI refusing on its
# behalf, which is the distinction the coverage measurement exists to make.


def test_passkey_register_complete_rejects_an_unknown_session(client, account) -> None:
    r = client.post(
        "/api/auth/mfa/passkey/register-complete",
        json={"state_id": f"no-such-session-{uuid.uuid4().hex}", "credential": {}},
        headers=account["headers"],
    )
    assert r.status_code == 400, r.text[:300]
    assert "registration session" in r.text.lower()


def test_passkey_register_complete_rejects_another_users_session(client, account) -> None:
    """The ownership check: a session belongs to the account that opened it.

    Without it, anyone holding a state_id could bind a passkey of their own to
    somebody else's account — a second factor the attacker controls.
    """
    options = client.post(
        "/api/auth/mfa/passkey/register-options",
        json={"device_name": "Probe Key"},
        headers=account["headers"],
    )
    assert options.status_code == 200, options.text[:300]
    state_id = options.json()["state_id"]

    other_email = f"mfa-surface-other-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": other_email, "password": PASSWORD, "name": "Other"},
    )
    other_headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    r = client.post(
        "/api/auth/mfa/passkey/register-complete",
        json={"state_id": state_id, "credential": {}},
        headers=other_headers,
    )
    assert r.status_code == 400, (
        f"another account completed this registration session ({r.status_code}); "
        "the passkey would be bound to the wrong user"
    )


def test_passkey_delete_rejects_an_unknown_method(client, account) -> None:
    r = client.post(
        "/api/auth/mfa/passkey/delete",
        json={"method_id": 99999999},
        headers=account["headers"],
    )
    assert r.status_code in (400, 404), r.text[:300]


def test_passkey_authenticate_options_rejects_an_unknown_challenge(client) -> None:
    r = client.post(
        "/api/auth/mfa/passkey/authenticate-options",
        json={"challenge_id": f"no-such-challenge-{uuid.uuid4().hex}"},
    )
    assert r.status_code == 400
    assert "challenge" in r.text.lower()


def test_passkey_authenticate_options_needs_a_registered_passkey(client, account) -> None:
    """A TOTP-only account must not be offered a passkey challenge."""
    _enable_totp(client, account)
    challenge_id = _login_challenge(client, account["email"])
    r = client.post(
        "/api/auth/mfa/passkey/authenticate-options", json={"challenge_id": challenge_id}
    )
    assert r.status_code == 400
    assert "passkey" in r.text.lower()


def test_passkey_authenticate_complete_rejects_an_unknown_session(client) -> None:
    """`state_id`, not `challenge_id`.

    Written with the wrong field name first, which produced a 422 — FastAPI
    refusing before the handler ran. That is the exact failure this file exists
    to avoid, so the assertion now pins it explicitly.
    """
    r = client.post(
        "/api/auth/mfa/passkey/authenticate-complete",
        json={"state_id": f"no-such-{uuid.uuid4().hex}", "credential": {}},
    )
    assert r.status_code != 422, f"the handler was never reached: {r.text[:300]}"
    assert r.status_code == 400, r.text[:300]
    assert "authentication session" in r.text.lower()
