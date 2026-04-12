"""Routes: mfa."""

import hashlib
import json
import os
import re
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from routes._deps import (
    SESSION_EXPIRY,
    XCELSIOR_ENV,
    _check_auth_rate_limit,
    _get_current_user as _deps_get_current_user,
    _is_platform_admin,
    _require_user_grant as _deps_require_user_grant,
    _set_session_cookies,
    log,
)
from scheduler import (
    log,
)
from db import MfaStore, UserStore
from oauth_service import issue_user_tokens
import base64 as _b64

router = APIRouter()

# WebAuthn configuration
_WEBAUTHN_RP_ID = os.environ.get("XCELSIOR_WEBAUTHN_RP_ID", "xcelsior.ca")
_WEBAUTHN_RP_NAME = "Xcelsior"


def _get_current_user(request: Request) -> dict | None:
    """Wrapper that rejects machine/API-key principals for MFA routes.

    MFA management is inherently an interactive-user operation.
    """
    return _deps_require_user_grant(request)


# ── Helper: _verify_totp_code ──

def _verify_totp_code(secret: str, code: str) -> bool:
    """Verify a TOTP code against the current time slice only."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=0)


# ── Helper: _hash_backup_code ──

def _hash_backup_code(code: str) -> str:
    """Hash a backup code for storage."""
    import hashlib
    return hashlib.sha256(code.encode()).hexdigest()


# ── Helper: _generate_backup_codes ──

def _generate_backup_codes(count: int = 10) -> list[str]:
    """Generate a set of backup codes."""
    codes = []
    for _ in range(count):
        part1 = secrets.token_hex(2).upper()
        part2 = secrets.token_hex(2).upper()
        codes.append(f"{part1}-{part2}")
    return codes


# ── Helper: _complete_mfa_login ──

def _complete_mfa_login(email: str, challenge_id: str, request: Request | None = None) -> JSONResponse:
    """Complete login after successful MFA verification."""
    challenge = MfaStore.get_challenge(challenge_id)
    if not challenge or challenge["email"] != email:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(challenge_id)
        raise HTTPException(400, "MFA challenge expired")

    MfaStore.delete_challenge(challenge_id)

    user = UserStore.get_user(email)
    if not user:
        raise HTTPException(400, "User not found")

    token_bundle = issue_user_tokens(user, request, client_id="xcelsior-web", session_type="browser")
    body = {
        "ok": True,
        "access_token": token_bundle["access_token"],
        "token_type": "Bearer",
        "expires_in": token_bundle.get("expires_in", SESSION_EXPIRY),
        "user": {
            "user_id": user["user_id"],
            "email": email,
            "name": user["name"],
            "role": user["role"],
            "is_admin": True if _is_platform_admin(user) else False,
            "customer_id": user["customer_id"],
            "provider_id": user.get("provider_id"),
        },
    }
    resp = JSONResponse(content=body)
    _set_session_cookies(resp, token_bundle)
    return resp


# ── Helper: _refresh_mfa_enabled ──

def _refresh_mfa_enabled(email: str) -> None:
    """Recalculate mfa_enabled flag for user based on active methods."""
    methods = MfaStore.list_methods(email)
    enabled = any(m.get("enabled") for m in methods)
    UserStore.update_user(email, {"mfa_enabled": 1 if enabled else 0})


# ── Helper: _send_sms ──

def _send_sms(phone_number: str, message: str) -> None:
    """Send an SMS via Twilio. Raises on failure."""
    from twilio.rest import Client
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_phone = os.environ.get("TWILIO_PHONE_NUMBER", "")
    if not sid or not token or not from_phone:
        raise HTTPException(500, "SMS service is not configured")
    try:
        client = Client(sid, token)
        client.messages.create(body=message, from_=from_phone, to=phone_number)
        log.info("SMS sent to %s", phone_number[-4:])
    except Exception as e:
        log.error("SMS send failed to %s: %s", phone_number[-4:], e)
        raise HTTPException(502, "Failed to send SMS. Please try again.")

@router.get("/api/auth/mfa/methods", tags=["Auth - MFA"])
def api_mfa_list_methods(request: Request):
    """List the user's configured MFA methods."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:read")
    methods = MfaStore.list_methods(user["email"])
    backup_codes = MfaStore.list_backup_codes(user["email"])
    enabled = any(bool(m.get("enabled")) for m in methods)
    if enabled != bool(user.get("mfa_enabled")):
        _refresh_mfa_enabled(user["email"])
    return {
        "ok": True,
        "mfa_enabled": enabled,
        "methods": [
            {
                "id": m["id"],
                "type": m["method_type"],
                "enabled": bool(m["enabled"]),
                "device_name": m.get("device_name"),
                "phone_number": m.get("phone_number", "")[-4:] if m.get("phone_number") else None,
                "created_at": m["created_at"],
            }
            for m in methods
        ],
        "backup_codes_remaining": sum(1 for c in backup_codes if not c["used"]),
    }

@router.post("/api/auth/mfa/totp/setup", tags=["Auth - MFA"])
def api_mfa_totp_setup(request: Request):
    """Generate a TOTP secret and QR code URI for setup."""
    import pyotp
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    # Check if TOTP already enabled
    existing = MfaStore.get_method_by_type(user["email"], "totp")
    if existing:
        raise HTTPException(409, "Authenticator app is already set up. Manage it from Settings.")

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user["email"],
        issuer_name="Xcelsior",
    )

    # Store secret temporarily (not enabled until verified)
    method_id = MfaStore.create_method({
        "email": user["email"],
        "method_type": "totp",
        "secret": secret,
        "enabled": 0,
        "created_at": time.time(),
    })

    return {
        "ok": True,
        "secret": secret,
        "provisioning_uri": provisioning_uri,
        "method_id": method_id,
    }
# ── Model: TotpVerifyRequest ──

class TotpVerifyRequest(BaseModel):
    code: str
    method_id: int | None = None

@router.post("/api/auth/mfa/totp/verify", tags=["Auth - MFA"])
def api_mfa_totp_verify(request: Request, req: TotpVerifyRequest):
    """Verify a TOTP code to complete setup and enable TOTP."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    methods = MfaStore.list_methods(user["email"])
    totp_method = None
    for m in methods:
        if m["method_type"] == "totp":
            if req.method_id and m["id"] == req.method_id:
                totp_method = m
                break
            if not m["enabled"]:
                totp_method = m
                break
            totp_method = m

    if not totp_method or not totp_method.get("secret"):
        raise HTTPException(400, "No TOTP method found. Run setup first.")

    if not _verify_totp_code(totp_method["secret"], req.code):
        raise HTTPException(400, "Invalid code. Please try again.")

    # Enable the method
    from db import auth_connection
    with auth_connection() as conn:
        conn.execute("UPDATE mfa_methods SET enabled = 1 WHERE id = %s", (totp_method["id"],))

    _refresh_mfa_enabled(user["email"])

    # Generate backup codes if none exist yet
    existing_codes = MfaStore.list_backup_codes(user["email"])
    backup_codes = None
    if not existing_codes:
        backup_codes = _generate_backup_codes()
        code_hashes = [_hash_backup_code(c) for c in backup_codes]
        MfaStore.create_backup_codes(user["email"], code_hashes)

    return {
        "ok": True,
        "message": "TOTP enabled successfully",
        "backup_codes": backup_codes,
    }

@router.delete("/api/auth/mfa/totp", tags=["Auth - MFA"])
def api_mfa_totp_disable(request: Request):
    """Disable and remove TOTP."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")
    MfaStore.delete_methods_by_type(user["email"], "totp")
    _refresh_mfa_enabled(user["email"])
    return {"ok": True, "message": "TOTP disabled"}


# ── Model: SmsSetupRequest ──

class SmsSetupRequest(BaseModel):
    phone_number: str  # E.164 format, e.g. +14165551234

@router.post("/api/auth/mfa/sms/setup", tags=["Auth - MFA"])
def api_mfa_sms_setup(request: Request, req: SmsSetupRequest):
    """Register a phone number for SMS MFA. Sends a verification code."""
    import re
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    phone = req.phone_number.strip()
    if not re.match(r"^\+[1-9]\d{6,14}$", phone):
        raise HTTPException(400, "Invalid phone number. Use E.164 format (e.g. +14165551234)")

    existing = MfaStore.get_method_by_type(user["email"], "sms")
    if existing:
        raise HTTPException(409, "SMS verification is already set up. Manage it from Settings.")

    # Generate 6-digit verification code
    code = f"{secrets.randbelow(1000000):06d}"
    code_hash = _hash_backup_code(code)  # reuse SHA-256 helper for code storage
    sms_challenge_id = f"sms-setup:{user['email']}"
    # Remove any existing SMS setup challenge
    MfaStore.delete_challenge(sms_challenge_id)
    MfaStore.create_challenge({
        "challenge_id": sms_challenge_id,
        "email": user["email"],
        "challenge_data": json.dumps({"code_hash": code_hash, "phone": phone}),
        "created_at": time.time(),
        "expires_at": time.time() + 600,
    })

    if os.environ.get("XCELSIOR_ENV") == "test":
        return {"ok": True, "message": "Verification code sent", "test_code": code}

    _send_sms(phone, f"Your Xcelsior verification code is: {code}")
    return {"ok": True, "message": "Verification code sent"}


# ── Model: SmsVerifyRequest ──

class SmsVerifyRequest(BaseModel):
    code: str

@router.post("/api/auth/mfa/sms/verify", tags=["Auth - MFA"])
def api_mfa_sms_verify(request: Request, req: SmsVerifyRequest):
    """Verify the SMS code to complete SMS MFA setup."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    sms_challenge_id = f"sms-setup:{user['email']}"
    pending = MfaStore.get_challenge(sms_challenge_id)
    if not pending:
        raise HTTPException(400, "No pending SMS verification. Run setup first.")
    if time.time() > pending["expires_at"]:
        MfaStore.delete_challenge(sms_challenge_id)
        raise HTTPException(400, "Verification code expired")
    pending_data = json.loads(pending["challenge_data"])
    if _hash_backup_code(req.code) != pending_data["code_hash"]:
        raise HTTPException(400, "Invalid verification code")

    MfaStore.delete_challenge(sms_challenge_id)

    MfaStore.create_method({
        "email": user["email"],
        "method_type": "sms",
        "phone_number": pending_data["phone"],
        "enabled": 1,
        "created_at": time.time(),
    })
    _refresh_mfa_enabled(user["email"])

    # Generate backup codes if none exist
    existing_codes = MfaStore.list_backup_codes(user["email"])
    backup_codes = None
    if not existing_codes:
        backup_codes = _generate_backup_codes()
        code_hashes = [_hash_backup_code(c) for c in backup_codes]
        MfaStore.create_backup_codes(user["email"], code_hashes)

    return {
        "ok": True,
        "message": "SMS MFA enabled successfully",
        "backup_codes": backup_codes,
    }

@router.delete("/api/auth/mfa/sms", tags=["Auth - MFA"])
def api_mfa_sms_disable(request: Request):
    """Disable and remove SMS MFA."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")
    MfaStore.delete_methods_by_type(user["email"], "sms")
    _refresh_mfa_enabled(user["email"])
    return {"ok": True, "message": "SMS MFA disabled"}


# ── Helper: _get_fido2_server ──

def _get_fido2_server():
    from fido2.server import Fido2Server
    from fido2.webauthn import PublicKeyCredentialRpEntity
    rp = PublicKeyCredentialRpEntity(id=_WEBAUTHN_RP_ID, name=_WEBAUTHN_RP_NAME)
    return Fido2Server(rp)


# ── Helper: _b64url_encode ──

def _b64url_encode(data: bytes) -> str:
    return _b64.urlsafe_b64encode(data).rstrip(b"=").decode()


# ── Helper: _b64url_decode ──

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return _b64.urlsafe_b64decode(s)


# ── Helper: _webauthn_options_to_json ──

def _webauthn_options_to_json(options) -> dict:
    """Recursively convert WebAuthn options to JSON-safe dict (bytes → base64url)."""
    if isinstance(options, bytes):
        return _b64url_encode(options)
    if isinstance(options, dict):
        return {k: _webauthn_options_to_json(v) for k, v in options.items()}
    if isinstance(options, (list, tuple)):
        return [_webauthn_options_to_json(v) for v in options]
    return options


# ── Model: PasskeyRegisterRequest ──

class PasskeyRegisterRequest(BaseModel):
    device_name: str = "Security Key"

@router.post("/api/auth/mfa/passkey/register-options", tags=["Auth - MFA"])
def api_mfa_passkey_register_options(req: PasskeyRegisterRequest, request: Request):
    """Generate WebAuthn registration options for adding a new passkey."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    from fido2.webauthn import PublicKeyCredentialUserEntity, PublicKeyCredentialDescriptor, PublicKeyCredentialType

    server = _get_fido2_server()
    user_entity = PublicKeyCredentialUserEntity(
        id=user["user_id"].encode(),
        name=user["email"],
        display_name=user.get("name") or user["email"],
    )

    # Exclude existing passkeys
    existing = MfaStore.list_methods(user["email"])
    exclude_creds = []
    for m in existing:
        if m["method_type"] == "passkey" and m.get("credential_id"):
            exclude_creds.append(
                PublicKeyCredentialDescriptor(
                    type=PublicKeyCredentialType.PUBLIC_KEY,
                    id=_b64url_decode(m["credential_id"]),
                )
            )

    options, state = server.register_begin(user_entity, exclude_creds)
    options_dict = _webauthn_options_to_json(dict(options))

    # Store state in challenge
    state_id = f"passkey-reg:{secrets.token_urlsafe(16)}"
    MfaStore.create_challenge({
        "challenge_id": state_id,
        "email": user["email"],
        "challenge_data": json.dumps({
            "state": _webauthn_options_to_json(state),
            "device_name": req.device_name,
        }),
        "created_at": time.time(),
        "expires_at": time.time() + 300,
    })

    return {"ok": True, "options": options_dict, "state_id": state_id}


# ── Model: PasskeyRegisterCompleteRequest ──

class PasskeyRegisterCompleteRequest(BaseModel):
    state_id: str
    credential: dict

@router.post("/api/auth/mfa/passkey/register-complete", tags=["Auth - MFA"])
def api_mfa_passkey_register_complete(req: PasskeyRegisterCompleteRequest, request: Request):
    """Complete passkey registration with the browser's attestation response."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    challenge = MfaStore.get_challenge(req.state_id)
    if not challenge or challenge["email"] != user["email"]:
        raise HTTPException(400, "Invalid registration session")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.state_id)
        raise HTTPException(400, "Registration session expired")

    challenge_data = json.loads(challenge["challenge_data"])
    stored_state = challenge_data["state"]
    device_name = challenge_data.get("device_name", "Security Key")

    # Reconstruct state — fido2 v2.x expects base64url string for challenge
    state = {
        "challenge": stored_state["challenge"],
        "user_verification": stored_state.get("user_verification"),
    }

    # Pass credential response as-is — fido2 v2.x from_dict handles b64url decoding
    cred = req.credential
    response_data = {
        "id": cred["id"],
        "rawId": cred["rawId"],
        "response": {
            "clientDataJSON": cred["response"]["clientDataJSON"],
            "attestationObject": cred["response"]["attestationObject"],
        },
        "type": cred.get("type", "public-key"),
    }

    server = _get_fido2_server()
    try:
        auth_data = server.register_complete(state, response_data)
    except Exception as e:
        log.warning("Passkey registration failed: %s", e)
        error_text = str(e).lower()
        if "already" in error_text and ("registered" in error_text or "credential" in error_text):
            raise HTTPException(409, "This passkey is already added to your account. Use it to sign in or add a different device.")
        raise HTTPException(400, "Passkey registration verification failed")

    MfaStore.delete_challenge(req.state_id)

    # Store credential
    cred_data = auth_data.credential_data
    credential_id_b64 = _b64url_encode(cred_data.credential_id)
    public_key_b64 = _b64url_encode(bytes(cred_data))

    existing_passkey = MfaStore.get_passkey_by_credential(credential_id_b64)
    if existing_passkey:
        raise HTTPException(409, "This passkey is already added to your account. Use it to sign in or add a different device.")

    method_id = MfaStore.create_method({
        "email": user["email"],
        "method_type": "passkey",
        "credential_id": credential_id_b64,
        "public_key": public_key_b64,
        "sign_count": 0,
        "device_name": device_name,
        "enabled": 1,
        "created_at": time.time(),
    })
    _refresh_mfa_enabled(user["email"])

    # Generate backup codes if none exist
    existing_codes = MfaStore.list_backup_codes(user["email"])
    backup_codes = None
    if not existing_codes:
        backup_codes = _generate_backup_codes()
        code_hashes = [_hash_backup_code(c) for c in backup_codes]
        MfaStore.create_backup_codes(user["email"], code_hashes)

    return {
        "ok": True,
        "message": "Passkey registered successfully",
        "method_id": method_id,
        "device_name": device_name,
        "backup_codes": backup_codes,
    }


# ── Model: PasskeyDeleteRequest ──

class PasskeyDeleteRequest(BaseModel):
    method_id: int

@router.post("/api/auth/mfa/passkey/delete", tags=["Auth - MFA"])
def api_mfa_passkey_delete(req: PasskeyDeleteRequest, request: Request):
    """Remove a registered passkey."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")

    method = MfaStore.get_method(req.method_id)
    if not method or method["email"] != user["email"] or method["method_type"] != "passkey":
        raise HTTPException(404, "Passkey not found")

    MfaStore.delete_method(req.method_id, user["email"])
    _refresh_mfa_enabled(user["email"])
    return {"ok": True, "message": "Passkey removed"}


# ── Model: PasskeyAuthenticateOptionsRequest ──

class PasskeyAuthenticateOptionsRequest(BaseModel):
    challenge_id: str

@router.post("/api/auth/mfa/passkey/authenticate-options", tags=["Auth - MFA"])
def api_mfa_passkey_authenticate_options(req: PasskeyAuthenticateOptionsRequest):
    """Generate WebAuthn authentication options during login MFA challenge."""
    challenge = MfaStore.get_challenge(req.challenge_id)
    if not challenge:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.challenge_id)
        raise HTTPException(400, "MFA challenge expired")

    email = challenge["email"]
    from fido2.webauthn import PublicKeyCredentialDescriptor, PublicKeyCredentialType

    methods = MfaStore.list_methods(email)
    allow_creds = []
    for m in methods:
        if m["method_type"] == "passkey" and m.get("enabled") and m.get("credential_id"):
            allow_creds.append(
                PublicKeyCredentialDescriptor(
                    type=PublicKeyCredentialType.PUBLIC_KEY,
                    id=_b64url_decode(m["credential_id"]),
                )
            )

    if not allow_creds:
        raise HTTPException(400, "No passkeys registered")

    server = _get_fido2_server()
    options, state = server.authenticate_begin(allow_creds)
    options_dict = _webauthn_options_to_json(dict(options))

    # Store WebAuthn state
    state_id = f"passkey-auth:{secrets.token_urlsafe(16)}"
    MfaStore.create_challenge({
        "challenge_id": state_id,
        "email": email,
        "challenge_data": json.dumps({
            "state": _webauthn_options_to_json(state),
            "login_challenge_id": req.challenge_id,
        }),
        "created_at": time.time(),
        "expires_at": time.time() + 300,
    })

    return {"ok": True, "options": options_dict, "state_id": state_id}


# ── Model: PasskeyAuthenticateCompleteRequest ──

class PasskeyAuthenticateCompleteRequest(BaseModel):
    state_id: str
    credential: dict

@router.post("/api/auth/mfa/passkey/authenticate-complete", tags=["Auth - MFA"])
def api_mfa_passkey_authenticate_complete(req: PasskeyAuthenticateCompleteRequest, request: Request):
    """Verify passkey authentication to complete MFA login."""
    challenge = MfaStore.get_challenge(req.state_id)
    if not challenge:
        raise HTTPException(400, "Invalid authentication session")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.state_id)
        raise HTTPException(400, "Authentication session expired")

    challenge_data = json.loads(challenge["challenge_data"])
    stored_state = challenge_data["state"]
    login_challenge_id = challenge_data["login_challenge_id"]
    email = challenge["email"]

    # Verify the login challenge is still valid
    login_challenge = MfaStore.get_challenge(login_challenge_id)
    if not login_challenge or login_challenge["email"] != email:
        raise HTTPException(400, "Invalid MFA login challenge")

    # Reconstruct state — fido2 v2.x expects base64url string for challenge
    state = {
        "challenge": stored_state["challenge"],
        "user_verification": stored_state.get("user_verification"),
    }

    # Pass credential response as-is — fido2 v2.x from_dict handles b64url decoding
    cred = req.credential
    response_data = {
        "id": cred["id"],
        "rawId": cred["rawId"],
        "response": {
            "authenticatorData": cred["response"]["authenticatorData"],
            "clientDataJSON": cred["response"]["clientDataJSON"],
            "signature": cred["response"]["signature"],
        },
        "type": cred.get("type", "public-key"),
    }
    if cred["response"].get("userHandle"):
        response_data["response"]["userHandle"] = cred["response"]["userHandle"]

    # Reconstruct stored credentials for verification
    from fido2.webauthn import AttestedCredentialData

    methods = MfaStore.list_methods(email)
    stored_creds = []
    for m in methods:
        if m["method_type"] == "passkey" and m.get("enabled") and m.get("public_key"):
            stored_creds.append(AttestedCredentialData(_b64url_decode(m["public_key"])))

    if not stored_creds:
        raise HTTPException(400, "No passkeys found")

    server = _get_fido2_server()
    try:
        result = server.authenticate_complete(state, stored_creds, response_data)
    except Exception as e:
        log.warning("Passkey authentication failed: %s", e)
        raise HTTPException(400, "Passkey authentication failed")

    MfaStore.delete_challenge(req.state_id)

    # Update sign count for the matched credential
    authenticated_cred_id = _b64url_encode(result.credential_id)
    passkey_method = MfaStore.get_passkey_by_credential(authenticated_cred_id)
    if passkey_method:
        MfaStore.update_passkey_sign_count(passkey_method["id"], passkey_method["sign_count"] + 1)

    return _complete_mfa_login(email, login_challenge_id, request)


# ── Model: MfaVerifyLogin ──

class MfaVerifyLogin(BaseModel):
    challenge_id: str
    method: str  # totp | sms | backup
    code: str

@router.post("/api/auth/mfa/verify", tags=["Auth - MFA"])
def api_mfa_verify_login(req: MfaVerifyLogin, request: Request):
    """Verify MFA code during login to complete authentication."""
    _check_auth_rate_limit(request)
    challenge = MfaStore.get_challenge(req.challenge_id)
    if not challenge:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.challenge_id)
        raise HTTPException(400, "MFA challenge expired. Please sign in again.")

    email = challenge["email"]

    if req.method == "totp":
        method = MfaStore.get_method_by_type(email, "totp")
        if not method or not method.get("secret"):
            raise HTTPException(400, "TOTP not configured")
        if not _verify_totp_code(method["secret"], req.code):
            raise HTTPException(400, "Invalid code")
        return _complete_mfa_login(email, req.challenge_id, request)

    elif req.method == "sms":
        # For login SMS verification, generate and send code
        method = MfaStore.get_method_by_type(email, "sms")
        if not method:
            raise HTTPException(400, "SMS MFA not configured")

        # Check if code matches the one we sent
        sms_login_id = f"sms-login:{email}"
        pending = MfaStore.get_challenge(sms_login_id)
        if not pending:
            raise HTTPException(400, "No SMS code sent. Request one first.")
        if time.time() > pending["expires_at"]:
            MfaStore.delete_challenge(sms_login_id)
            raise HTTPException(400, "Code expired")
        pending_data = json.loads(pending["challenge_data"])
        if _hash_backup_code(req.code) != pending_data["code_hash"]:
            raise HTTPException(400, "Invalid code")
        MfaStore.delete_challenge(sms_login_id)
        return _complete_mfa_login(email, req.challenge_id, request)

    elif req.method == "backup":
        code_hash = _hash_backup_code(req.code)
        if not MfaStore.use_backup_code(email, code_hash):
            raise HTTPException(400, "Invalid backup code")
        return _complete_mfa_login(email, req.challenge_id, request)

    else:
        raise HTTPException(400, f"Unsupported MFA method: {req.method}")


# ── Model: MfaSendSmsRequest ──

class MfaSendSmsRequest(BaseModel):
    challenge_id: str

@router.post("/api/auth/mfa/sms/send", tags=["Auth - MFA"])
def api_mfa_sms_send_login(req: MfaSendSmsRequest, request: Request):
    """Send an SMS verification code during MFA login challenge."""
    _check_auth_rate_limit(request)
    challenge = MfaStore.get_challenge(req.challenge_id)
    if not challenge:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.challenge_id)
        raise HTTPException(400, "MFA challenge expired")

    email = challenge["email"]
    method = MfaStore.get_method_by_type(email, "sms")
    if not method:
        raise HTTPException(400, "SMS MFA not configured")

    code = f"{secrets.randbelow(1000000):06d}"
    code_hash = _hash_backup_code(code)
    sms_login_id = f"sms-login:{email}"
    MfaStore.delete_challenge(sms_login_id)
    MfaStore.create_challenge({
        "challenge_id": sms_login_id,
        "email": email,
        "challenge_data": json.dumps({"code_hash": code_hash}),
        "created_at": time.time(),
        "expires_at": time.time() + 600,
    })

    if os.environ.get("XCELSIOR_ENV") == "test":
        return {"ok": True, "message": "Code sent", "test_code": code}

    _send_sms(method.get("phone_number", ""), f"Your Xcelsior login code is: {code}")
    return {"ok": True, "message": "Code sent"}

@router.post("/api/auth/mfa/backup-codes/regenerate", tags=["Auth - MFA"])
def api_mfa_regenerate_backup_codes(request: Request):
    """Regenerate backup recovery codes. Invalidates all previous codes."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")
    if not any(bool(m.get("enabled")) for m in MfaStore.list_methods(user["email"])):
        raise HTTPException(400, "MFA is not enabled")

    codes = _generate_backup_codes()
    code_hashes = [_hash_backup_code(c) for c in codes]
    MfaStore.create_backup_codes(user["email"], code_hashes)

    return {"ok": True, "backup_codes": codes}

@router.delete("/api/auth/mfa/all", tags=["Auth - MFA"])
def api_mfa_disable_all(request: Request):
    """Disable all MFA methods for the user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "mfa:write")
    from db import auth_connection
    with auth_connection() as conn:
        conn.execute("DELETE FROM mfa_methods WHERE email = %s", (user["email"],))
        conn.execute("DELETE FROM mfa_backup_codes WHERE email = %s", (user["email"],))

    UserStore.update_user(user["email"], {"mfa_enabled": 0})
    return {"ok": True, "message": "All MFA methods disabled"}
