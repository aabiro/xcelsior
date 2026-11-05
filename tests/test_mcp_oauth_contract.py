"""MCP audience, asymmetric signing, JWKS, and revocation gates."""

import json
import hashlib
import os
from types import SimpleNamespace

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import pytest

import oauth_service


def _rsa_key_config() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return json.dumps(
        {
            "active_kid": "mcp-test-1",
            "keys": [
                {
                    "kid": "mcp-test-1",
                    "private_key_pem": private,
                    "public_key_pem": public,
                }
            ],
        }
    )


def test_mcp_token_is_rs256_audience_bound_and_revocable(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", _rsa_key_config())
    client = {
        "client_id": "mcp-client-test",
        "client_secret_salt": "salt",
        "status": "active",
        "workspace_customer_id": "tenant-test",
        "team_id": "team-test",
    }
    monkeypatch.setattr(oauth_service.OAuthStore, "get_client", lambda _: client)
    monkeypatch.setattr(oauth_service.OAuthStore, "update_last_used", lambda _: None)
    monkeypatch.setattr(
        oauth_service.OAuthStore,
        "client_credential_revoked",
        staticmethod(lambda _jti: False),
    )

    bundle = oauth_service.issue_client_credentials_jwt(
        client,
        ["instances:read"],
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    header = json.loads(oauth_service._base64url_decode(bundle["access_token"].split(".")[0]))
    assert header["alg"] == "RS256"
    principal = oauth_service.validate_client_credentials_jwt(bundle["access_token"])
    assert principal["audience"] == oauth_service.MCP_RESOURCE_AUDIENCE
    assert principal["tenant_id"] == "tenant-test"
    assert oauth_service.oauth_jwks()["keys"][0]["kid"] == "mcp-test-1"

    client["status"] = "disabled"
    assert oauth_service.validate_client_credentials_jwt(bundle["access_token"]) is None


def test_one_machine_token_revokes_without_disabling_the_client(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", _rsa_key_config())
    client = {
        "client_id": "mcp-client-test",
        "client_secret_salt": "salt",
        "status": "active",
        "workspace_customer_id": "tenant-test",
    }
    revoked: set[str] = set()
    monkeypatch.setattr(oauth_service.OAuthStore, "get_client", lambda _: client)
    monkeypatch.setattr(oauth_service.OAuthStore, "update_last_used", lambda _: None)
    monkeypatch.setattr(
        oauth_service.OAuthStore,
        "client_credential_revoked",
        staticmethod(lambda jti: jti in revoked),
    )
    monkeypatch.setattr(
        oauth_service.OAuthStore,
        "revoke_client_credential",
        staticmethod(lambda jti, tenant_id, client_id, expires_at: revoked.add(jti)),
    )

    def _resign(payload: dict) -> str:
        key = json.loads(os.environ["XCELSIOR_OAUTH_JWT_KEYS_JSON"])["keys"][0]
        header = {"alg": "RS256", "typ": "JWT", "kid": key["kid"]}
        signing = ".".join(
            [
                oauth_service._base64url_encode(oauth_service._json_dumps(header).encode()),
                oauth_service._base64url_encode(oauth_service._json_dumps(payload).encode()),
            ]
        )
        private = serialization.load_pem_private_key(key["private_key_pem"].encode(), password=None)
        signature = private.sign(signing.encode(), padding.PKCS1v15(), hashes.SHA256())
        return f"{signing}.{oauth_service._base64url_encode(signature)}"

    first = oauth_service.issue_client_credentials_jwt(client, ["instances:read"])
    second = oauth_service.issue_client_credentials_jwt(client, ["instances:read"])
    assert oauth_service.revoke_client_credentials_jwt(first["access_token"]) is True
    assert oauth_service.validate_client_credentials_jwt(first["access_token"]) is None
    assert oauth_service.validate_client_credentials_jwt(second["access_token"]) is not None
    assert client["status"] == "active"
    unsigned = json.loads(oauth_service._base64url_decode(second["access_token"].split(".")[1]))
    unsigned.pop("jti")
    assert oauth_service.validate_client_credentials_jwt(_resign(unsigned)) is None


def test_revocation_lookup_failure_rejects_the_machine_token(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", _rsa_key_config())
    client = {
        "client_id": "mcp-client-test",
        "client_secret_salt": "salt",
        "status": "active",
        "workspace_customer_id": "tenant-test",
    }
    monkeypatch.setattr(oauth_service.OAuthStore, "get_client", lambda _: client)
    monkeypatch.setattr(oauth_service.OAuthStore, "update_last_used", lambda _: None)

    def unavailable(_jti):
        raise RuntimeError("revocation store unavailable")

    monkeypatch.setattr(
        oauth_service.OAuthStore,
        "client_credential_revoked",
        staticmethod(unavailable),
    )
    bundle = oauth_service.issue_client_credentials_jwt(client, ["instances:read"])
    assert oauth_service.validate_client_credentials_jwt(bundle["access_token"]) is None


def test_production_refuses_symmetric_machine_token(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", raising=False)
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_SECRET", "legacy-shared-secret")
    client = {"client_id": "legacy", "client_secret_salt": "salt"}
    try:
        oauth_service.issue_client_credentials_jwt(client, ["instances:read"])
    except oauth_service.OAuthGrantError as exc:
        assert "asymmetric" in str(exc)
    else:  # pragma: no cover - hard security gate
        raise AssertionError("production accepted symmetric OAuth signing")


def test_authorization_code_binds_mcp_audience_into_access_and_refresh(monkeypatch):
    cache: dict[tuple[str, str], dict] = {}
    refresh_records: list[dict] = []
    sessions: list[dict] = []
    client = {
        "client_id": "interactive-mcp",
        "scopes": ["instances:read", "offline_access"],
    }
    monkeypatch.setattr(
        oauth_service,
        "_cache_set_json",
        lambda namespace, key, value, ttl, **_: cache.__setitem__((namespace, key), dict(value))
        is None,
    )
    monkeypatch.setattr(
        oauth_service,
        "_cache_getdel_json",
        lambda namespace, key: cache.pop((namespace, key), None),
    )
    monkeypatch.setattr(oauth_service, "get_client", lambda _: client)
    monkeypatch.setattr(
        oauth_service.UserStore, "create_session", lambda value: sessions.append(dict(value))
    )
    monkeypatch.setattr(
        oauth_service.OAuthStore,
        "create_refresh_token",
        lambda value: refresh_records.append(dict(value)),
    )
    request = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        headers={"user-agent": "mcp-contract-test"},
    )
    verifier = "v" * 64
    challenge = oauth_service._base64url_encode(hashlib.sha256(verifier.encode()).digest())
    code = oauth_service.issue_authorization_code(
        client=client,
        user={"email": "user@example.test", "user_id": "user-1"},
        redirect_uri="https://agent.example.test/callback",
        code_challenge=challenge,
        code_challenge_method="S256",
        scopes=["instances:read", "offline_access"],
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    bundle = oauth_service.exchange_authorization_code(
        client=client,
        code=code,
        redirect_uri="https://agent.example.test/callback",
        code_verifier=verifier,
        request=request,
        resource=oauth_service.MCP_RESOURCE_AUDIENCE,
    )

    assert bundle["resource"] == oauth_service.MCP_RESOURCE_AUDIENCE
    assert refresh_records[0]["audience"] == oauth_service.MCP_RESOURCE_AUDIENCE
    access = cache[("access_token", bundle["access_token"])]
    assert access["audience"] == oauth_service.MCP_RESOURCE_AUDIENCE
    assert sessions


def test_authorization_code_rejects_resource_substitution(monkeypatch):
    cache: dict[tuple[str, str], dict] = {}
    client = {"client_id": "interactive-mcp", "scopes": ["instances:read"]}
    monkeypatch.setattr(
        oauth_service,
        "_cache_set_json",
        lambda namespace, key, value, ttl, **_: cache.__setitem__((namespace, key), dict(value))
        is None,
    )
    monkeypatch.setattr(
        oauth_service,
        "_cache_getdel_json",
        lambda namespace, key: cache.pop((namespace, key), None),
    )
    verifier = "v" * 64
    code = oauth_service.issue_authorization_code(
        client=client,
        user={"email": "user@example.test"},
        redirect_uri="https://agent.example.test/callback",
        code_challenge=oauth_service._base64url_encode(hashlib.sha256(verifier.encode()).digest()),
        code_challenge_method="S256",
        scopes=["instances:read"],
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    try:
        oauth_service.exchange_authorization_code(
            client=client,
            code=code,
            redirect_uri="https://agent.example.test/callback",
            code_verifier=verifier,
            request=None,
            resource="https://attacker.example.test",
        )
    except oauth_service.OAuthGrantError as exc:
        assert exc.error == "invalid_target"
    else:  # pragma: no cover
        raise AssertionError("authorization code accepted a substituted resource")


@pytest.mark.parametrize("identity_scopes", [[], ["profile"], ["email"], ["profile", "email"]])
def test_mcp_user_jwt_has_signed_consent_scoped_identity_and_cached_revocation(
    monkeypatch, identity_scopes
):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    key_config = _rsa_key_config()
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", key_config)
    oauth_service.reset_auth_cache_for_tests()
    user = {"user_id": "identity-user", "email": "identity@example.test", "name": "Identity User"}
    scopes = ["instances:read", *identity_scopes]
    bundle = oauth_service._issue_cached_access_token(
        user,
        client_id="identity-connector",
        scopes=scopes,
        session_token="identity-session",
        session_type="browser",
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    token = bundle["access_token"]
    header_b64, payload_b64, signature_b64 = token.split(".")
    header = json.loads(oauth_service._base64url_decode(header_b64))
    claims = json.loads(oauth_service._base64url_decode(payload_b64))
    public = serialization.load_pem_public_key(
        json.loads(key_config)["keys"][0]["public_key_pem"].encode()
    )
    public.verify(
        oauth_service._base64url_decode(signature_b64),
        f"{header_b64}.{payload_b64}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    assert header == {"alg": "RS256", "typ": "at+jwt", "kid": "mcp-test-1"}
    assert claims["sub"] == user["user_id"]
    assert claims["aud"] == oauth_service.MCP_RESOURCE_AUDIENCE
    assert claims["iss"] == oauth_service.OAUTH_ISSUER
    assert claims["client_id"] == "identity-connector"
    assert claims["exp"] - claims["iat"] == bundle["expires_in"]
    assert set(claims["scope"].split()) == set(scopes)
    assert ("name" in claims) == ("profile" in identity_scopes)
    assert ("email" in claims) == ("email" in identity_scopes)
    assert ("preferred_username" in claims) == ("email" in identity_scopes)
    assert not {"session_token", "is_admin", "role", "customer_id"} & claims.keys()
    assert oauth_service.is_oauth_access_token(token)
    assert oauth_service.validate_client_credentials_jwt(token) is None
    assert oauth_service.resolve_opaque_access_token(token)["user_id"] == user["user_id"]

    # A valid signature alone is insufficient; the exact issued bearer must
    # still exist in the auth cache. Neither forged nor re-signed claims work.
    unissued = oauth_service._sign_oauth_jwt({**claims, "jti": "not-issued"}, token_type="at+jwt")
    assert oauth_service.is_oauth_access_token(unissued)
    assert oauth_service.resolve_opaque_access_token(unissued) is None
    oauth_service.revoke_access_token(token)
    assert oauth_service.resolve_opaque_access_token(token) is None

    replacement = oauth_service._issue_cached_access_token(
        user,
        client_id="identity-connector",
        scopes=scopes,
        session_token="identity-session",
        session_type="browser",
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )["access_token"]
    oauth_service.revoke_session("identity-session")
    assert oauth_service.resolve_opaque_access_token(replacement) is None


def test_mcp_user_jwt_requires_production_asymmetric_signing(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", raising=False)
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_SECRET", "development-only-secret")
    with pytest.raises(oauth_service.OAuthGrantError, match="asymmetric"):
        oauth_service._issue_cached_access_token(
            {"user_id": "u1", "email": "identity@example.test"},
            client_id="identity-connector",
            scopes=["email"],
            session_token="identity-session",
            session_type="browser",
            audience=oauth_service.MCP_RESOURCE_AUDIENCE,
        )
