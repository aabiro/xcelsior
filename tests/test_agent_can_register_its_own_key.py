"""The SSH-key routes must be reachable by the credential the tools carry.

`register_ssh_key` shipped as an MCP tool, correctly scoped to `ssh:write` and
unit-tested at the MCP layer, against a route that could not serve it.
`POST /api/ssh/keys` guarded with `_require_user_grant`, which raises 403 for
`auth_type == "client_credentials"` before any scope is consulted — and a Quick
Connect token is exactly that. The tool was reachable, authorised, and answered
403 from the backend.

That is the second time in one day the same shape got through: a capability
promised to a credential that could not reach it. The first was
`instances:connect` missing from `MCP_QUICK_CONNECT_SCOPES` while the routes
enforced it. Both were invisible to every existing test because each layer was
correct in isolation.

So this file asserts the join, from the route's own source: **for every SSH-key
route, the auth helper it uses must admit machine credentials, and the scope it
demands must be one Quick Connect carries** (or, if not carried, be absent by a
decision recorded here rather than by accident).

Read by AST rather than by calling the app, deliberately: the failure being
guarded is a *helper choice* on a specific handler, and an integration test that
happened to authenticate as a session user would pass while the defect stood.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SSH_ROUTES = ROOT / "routes" / "ssh.py"

#: Helpers that refuse `client_credentials` outright, before any scope check.
#: `_require_user_grant` documents this in its own docstring: "Rejects
#: client_credentials (machine) tokens outright".
HUMAN_ONLY_HELPERS = {"_require_user_grant"}

#: The handlers an agent must be able to reach for P2's journey, and the scope
#: each one is expected to demand.
AGENT_REACHABLE = {
    "api_add_ssh_key": "ssh:write",
    "api_delete_ssh_key": "ssh:write",
    "api_list_ssh_keys": "ssh:read",
}


def _handlers() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(SSH_ROUTES.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _calls_in(node: ast.AST) -> set[str]:
    """Every function *called* in this handler, by name.

    `ast.walk` over the function body only — a name that appears in the
    docstring does not count, which matters here because the docstring on
    `api_add_ssh_key` explains at length why `_require_user_grant` is wrong for
    it. A text scan would read that explanation as the defect it describes.
    """
    return {
        getattr(call.func, "id", "") or getattr(call.func, "attr", "")
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def _scopes_required_by(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        name = getattr(call.func, "id", "") or getattr(call.func, "attr", "")
        if name != "_require_scope":
            continue
        for arg in call.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
    return found


def test_the_handlers_are_still_where_this_thinks_they_are():
    """Prove the reach — a rename makes every assertion below vacuous."""
    handlers = _handlers()
    missing = sorted(set(AGENT_REACHABLE) - set(handlers))
    assert not missing, f"handlers not found in routes/ssh.py: {missing}"


def test_the_docstring_scan_would_be_wrong_here():
    """Calibration for the AST choice above.

    `api_add_ssh_key`'s docstring names `_require_user_grant` while its body
    does not call it. A grep-based version of this file would fail on correct
    code — so this asserts the two disagree, which is the whole reason the
    check reads calls rather than text.
    """
    handler = _handlers()["api_add_ssh_key"]
    source = ast.get_docstring(handler) or ""
    assert "_require_user_grant" in source, (
        "the docstring no longer explains the guard that was replaced; if it "
        "was rewritten deliberately, delete this test with it"
    )
    assert "_require_user_grant" not in _calls_in(handler)


@pytest.mark.parametrize("handler_name,expected_scope", sorted(AGENT_REACHABLE.items()))
def test_an_agent_credential_is_not_refused_before_its_scope_is_read(
    handler_name: str, expected_scope: str
):
    """The defect, asserted directly."""
    handler = _handlers()[handler_name]
    human_only = HUMAN_ONLY_HELPERS & _calls_in(handler)
    assert not human_only, (
        f"{handler_name} guards with {sorted(human_only)}, which refuses "
        "client_credentials before any scope is consulted. An MCP tool calling "
        "this route gets 403 no matter what scope its token holds."
    )
    assert expected_scope in _scopes_required_by(handler), (
        f"{handler_name} no longer requires {expected_scope}; opening the "
        "handler to machine credentials is only safe because the scope check "
        "is what restricts it"
    )


def test_registering_a_key_records_which_credential_did_it():
    """Migration 099's columns are written, not merely present.

    Allowing a machine credential to register a key is only defensible with
    attribution: revoking an OAuth client has to identify which keys it added.
    A migration that adds two columns nothing writes is worse than no migration,
    because the dashboard would show every agent-added key as human-added.
    """
    handler = _handlers()["api_add_ssh_key"]
    written = {
        node.value
        for node in ast.walk(handler)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for column in ("registered_by_client_id", "registered_by_auth_type"):
        assert column in written, (
            f"{column} is never set by api_add_ssh_key, so migration 099's "
            "column stays NULL for agent-registered keys and the attribution "
            "the narrowing depends on does not exist"
        )


def test_the_store_persists_both_columns():
    """The other end of the same wire.

    Setting the fields in the route means nothing if the INSERT ignores them,
    and that INSERT listed its columns explicitly, so adding a key to the dict
    without touching the SQL would have silently dropped them.
    """
    import inspect

    from db import UserStore

    source = inspect.getsource(UserStore.add_ssh_key)
    for column in ("registered_by_client_id", "registered_by_auth_type"):
        assert column in source, f"UserStore.add_ssh_key does not persist {column}"


def test_quick_connect_can_reach_the_registration_route():
    """The credential the product tells people to paste, against the route."""
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    required = _scopes_required_by(_handlers()["api_add_ssh_key"])
    missing = sorted(required - set(MCP_QUICK_CONNECT_SCOPES))
    assert not missing, (
        f"a pasted Quick Connect token cannot register an SSH key: {missing}. "
        "This is what a user sees as '403 Insufficient scope' at the exact "
        "moment they try to open a shell."
    )


def test_the_migration_that_makes_this_defensible_is_in_the_repository():
    """099 lived only on a closed branch while its columns existed in a database.

    A migration applied to a developer's database but absent from the
    repository is the worst of both: `alembic current` reports a revision the
    tree cannot explain, and every other environment silently lacks the columns
    this route now writes.
    """
    migration = ROOT / "migrations" / "versions" / "099_ssh_key_client_binding.py"
    assert migration.exists(), "migration 099 is missing from migrations/versions/"
    text = migration.read_text(encoding="utf-8")
    assert 'down_revision = "098"' in text, "099 no longer chains onto 098"
    for column in ("registered_by_client_id", "registered_by_auth_type"):
        assert column in text, f"099 does not add {column}"
