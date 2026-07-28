"""MCP audience, asymmetric signing, JWKS, and revocation gates."""

import json
import hashlib
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import oauth_service


def _rsa_key_config() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
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
        lambda namespace, key, value, ttl, **_: cache.__setitem__(
            (namespace, key), dict(value)
        )
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
    challenge = oauth_service._base64url_encode(
        hashlib.sha256(verifier.encode()).digest()
    )
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
        lambda namespace, key, value, ttl, **_: cache.__setitem__(
            (namespace, key), dict(value)
        )
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
        code_challenge=oauth_service._base64url_encode(
            hashlib.sha256(verifier.encode()).digest()
        ),
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
