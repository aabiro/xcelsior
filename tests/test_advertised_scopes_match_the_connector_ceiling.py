"""A connector is only told to ask for scopes it can be granted.

An MCP connector requests every scope it is shown. Two documents show it
scopes: the resource metadata at `mcp.xcelsior.ca/.well-known/oauth-protected-
resource`, which the MCP authorization spec points at, and the authorization
server metadata, which VS Code reads instead. What a connector can actually hold
is `CONNECTOR_ALLOWED_SCOPES`, and anything above it is dropped from the grant
without an error (RFC 6749 §3.3).

The resource metadata advertised `billing:write`, `providers:write`,
`reputation:write` and `ssh:read`, and the ceiling held none of them. So every
connector asked the user for four permissions it never received, and nine
tools refused it afterwards. Plan §8.3 (Headmesh) settled each one:

  * `ssh:read` joined the ceiling. It is read-only, and it is what lets a
    connector see the key it registered and remove it. Its consent copy names
    SSH keys, because a public key identifies its owner elsewhere.
  * the three writes, which move money, a payout destination or reputation
    badges, stay off connectors and left the advertisement. Their tools still
    exist for credentials that carry them.

The resource metadata is served by the TypeScript MCP service, so this reads
the copy that `npm run surface:update` records in `api/mcp/tool-surface.json`.
`api/mcp/tests/unit/surface.test.ts` fails if that copy is stale.
"""

from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

from oauth_registration import CONNECTOR_ALLOWED_SCOPES, CONNECTOR_DEFAULT_SCOPES  # noqa: E402
from oauth_service import (  # noqa: E402
    MCP_QUICK_CONNECT_SCOPES,
    SCOPE_DESCRIPTIONS,
    summarize_consent_scopes,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SURFACE = ROOT / "api" / "mcp" / "tool-surface.json"

#: In the ceiling, and rightly absent from the resource metadata, which lists
#: what the MCP tools require. The first three are OpenID Connect scopes that
#: belong to the authorization server; no tool requires `mcp_actions:approve`.
NOT_RESOURCE_SCOPES = {"profile", "email", "offline_access", "mcp_actions:approve"}

#: On the authorization server's metadata for the clients it serves besides
#: connectors: the first-party apps and providers' worker agents. A connector
#: that asks for them has them dropped, which `test_connector_scope_ceiling.py`
#: exercises end to end.
AUTHORIZATION_SERVER_ONLY = {"billing:write", "hosts:read", "hosts:write"}

#: Tenant scopes connectors never hold, and the tools that need them.
CONNECTOR_WITHHELD = {"billing:write", "providers:write", "reputation:write"}
TOOLS_NEEDING_ANOTHER_CREDENTIAL = {
    "configure_auto_topup",
    "create_crypto_deposit",
    "create_lightning_deposit",
    "top_up_wallet",
    "register_provider",
    "request_provider_payout",
    "claim_reputation_milestones",
}


def _surface() -> dict:
    return json.loads(SURFACE.read_text(encoding="utf-8"))


def _resource_metadata_scopes() -> set[str]:
    surface = _surface()
    assert "scopesSupported" in surface, (
        "api/mcp/tool-surface.json does not record the resource metadata's "
        "scopes_supported; run `npm run surface:update` in api/mcp"
    )
    return set(surface["scopesSupported"])


def _authorization_server_scopes() -> set[str]:
    from fastapi.testclient import TestClient

    from api import app

    r = TestClient(app).get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200, r.text
    return set(r.json()["scopes_supported"])


def test_the_resource_metadata_advertises_nothing_a_connector_is_refused():
    unreachable = sorted(_resource_metadata_scopes() - set(CONNECTOR_ALLOWED_SCOPES))
    assert not unreachable, (
        f"the MCP resource metadata advertises {unreachable}, which no connector "
        "can be granted. Either add them to CONNECTOR_ALLOWED_SCOPES with consent "
        "copy, or to CONNECTOR_WITHHELD_SCOPES in api/mcp/src/tools/profiles.ts."
    )


def test_the_resource_metadata_advertises_every_resource_scope_in_the_ceiling():
    """The other direction: a scope a connector may hold but is never told about
    is one it never asks for, so the tools behind it refuse it all the same."""
    expected = set(CONNECTOR_ALLOWED_SCOPES) - NOT_RESOURCE_SCOPES
    assert _resource_metadata_scopes() == expected


def test_the_authorization_server_advertises_the_ceiling():
    advertised = _authorization_server_scopes()
    assert advertised - AUTHORIZATION_SERVER_ONLY == set(CONNECTOR_ALLOWED_SCOPES)


def test_ssh_read_is_reachable_by_a_connector_and_consent_names_it():
    assert "ssh:read" in CONNECTOR_ALLOWED_SCOPES
    assert "ssh:read" in _resource_metadata_scopes()
    # Only when a connector asks for it, and never on the Quick Connect key,
    # which is issued without the consent card.
    assert "ssh:read" not in CONNECTOR_DEFAULT_SCOPES
    assert "ssh:read" not in MCP_QUICK_CONNECT_SCOPES

    assert "public key" in SCOPE_DESCRIPTIONS["ssh:read"].lower()
    for requested in (["ssh:read"], sorted(_resource_metadata_scopes())):
        card = " ".join(summarize_consent_scopes(requested))
        assert "SSH keys" in card, f"consent for {requested} does not name SSH keys: {card}"
    without = summarize_consent_scopes(["profile", "instances:read"])
    assert not any("SSH" in line for line in without), without


def test_only_the_money_moving_tools_need_a_credential_other_than_a_connector():
    tools = {tool["name"]: set(tool["requiredScopes"]) for tool in _surface()["tools"]}
    blocked = {name for name, scopes in tools.items() if scopes & CONNECTOR_WITHHELD}
    assert blocked == TOOLS_NEEDING_ANOTHER_CREDENTIAL
    assert not CONNECTOR_WITHHELD & set(CONNECTOR_ALLOWED_SCOPES)
    # The two the ceiling change opened up.
    for name in ("list_ssh_keys", "delete_ssh_key"):
        assert tools[name] <= set(CONNECTOR_ALLOWED_SCOPES), name
