"""An agent API key is an API key, and all three SSH-key routes are scoped.

Two defects, one shape: a rule stated once and spelled two ways.

**`_require_user_grant` tested the literal `"api_key"`.** Agent keys are minted
with `auth_type: "agent_api_key"` (`oauth_service.py`), so the literal never
matched one. Thirty call sites use the default `allow_api_key=False` — MFA,
password change, privacy consent, account deletion, team management — and every
one of them admitted an agent key, which is the exact opposite of what the
function's docstring promises. A Quick Connect key, of the kind the quickstarts
tell users to paste, reached all of them.

**`DELETE /api/ssh/keys/{key_id}` enforced no scope.** `POST` requires
`ssh:write` and `GET` requires `ssh:read`; the delete required nothing, so the
only one of the three reachable by a narrowed machine credential was the
destructive one — and it calls `_trigger_reinject_for_user`, revoking the key
from running instances. The verb asymmetry `test_conditional_scope_guard.py`
exists to catch, arriving through a missing line rather than a wrong one.

Both are asserted here against the definitions rather than the current text, so
a third spelling fails.
"""

from __future__ import annotations

import ast
import inspect

import pytest
from fastapi import HTTPException

from routes import _deps


class _FakeRequest:
    """`_require_user_grant` only passes this to `_get_current_user`, which is patched."""


def _as(auth_type: str, **extra) -> dict:
    return {"auth_type": auth_type, "email": "probe@example.com", **extra}


@pytest.fixture
def as_principal(monkeypatch):
    def _install(principal: dict | None):
        monkeypatch.setattr(_deps, "_get_current_user", lambda _request: principal)

    return _install


# ── An agent key is an API key ─────────────────────────────────────────


def test_agent_key_is_refused_where_an_interactive_session_is_required(as_principal):
    """The defect: thirty routes take the default and meant to refuse this."""
    as_principal(_as("agent_api_key", scopes=["instances:read"]))
    with pytest.raises(HTTPException) as caught:
        _deps._require_user_grant(_FakeRequest())
    assert caught.value.status_code == 403


def test_the_bare_api_key_literal_is_still_refused(as_principal):
    """The case that always worked keeps working — the fix widens, not replaces."""
    as_principal(_as("api_key"))
    with pytest.raises(HTTPException) as caught:
        _deps._require_user_grant(_FakeRequest())
    assert caught.value.status_code == 403


@pytest.mark.parametrize("auth_type", ["api_key", "agent_api_key"])
def test_both_key_types_are_admitted_when_the_caller_opts_in(as_principal, auth_type):
    """`allow_api_key=True` must still admit both, or fourteen callers break.

    The fix is about the routes that opted *out*. A route that deliberately
    accepts keys keeps accepting them — and enforces scopes separately.
    """
    as_principal(_as(auth_type, scopes=["ssh:write"]))
    user = _deps._require_user_grant(_FakeRequest(), allow_api_key=True)
    assert user["auth_type"] == auth_type


def test_an_interactive_session_is_unaffected(as_principal):
    as_principal(_as("oauth_access_token"))
    assert _deps._require_user_grant(_FakeRequest())["auth_type"] == "oauth_access_token"


def test_client_credentials_are_still_refused_even_with_allow_api_key(as_principal):
    """`allow_api_key` opts into keys, not into machine tokens generally."""
    as_principal(_as("client_credentials", scopes=["ssh:write"]))
    with pytest.raises(HTTPException) as caught:
        _deps._require_user_grant(_FakeRequest(), allow_api_key=True)
    assert caught.value.status_code == 403


def test_the_key_types_are_a_named_set_not_a_literal():
    """Guards the shape, not the spelling.

    The bug was a literal in one place disagreeing with a set in another. If a
    future edit reintroduces an inline `== "api_key"` comparison in the grant
    check, the definition and its consumer can disagree again silently.
    """
    assert "agent_api_key" in _deps._API_KEY_AUTH_TYPES
    assert "api_key" in _deps._API_KEY_AUTH_TYPES
    source = inspect.getsource(_deps._require_user_grant)
    assert '== "api_key"' not in source, (
        "the grant check compares against a literal again — use "
        "_API_KEY_AUTH_TYPES so the definition has one spelling"
    )


# ── Every SSH-key route enforces a scope ───────────────────────────────

#: Route handler → the scope it must require. Deletion is `ssh:write` because it
#: is a write, and because leaving it unscoped made it the only one of the three
#: a narrowed credential could reach.
_SSH_KEY_ROUTE_SCOPES = {
    "api_add_ssh_key": "ssh:write",
    "api_list_ssh_keys": "ssh:read",
    "api_delete_ssh_key": "ssh:write",
}


def _scopes_required_by(function_name: str) -> list[str]:
    """Every literal passed to `_require_scope` inside one handler, via AST."""
    import routes.ssh as ssh_module

    tree = ast.parse(inspect.getsource(ssh_module))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        found: list[str] = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "_require_scope":
                continue
            found += [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        return found
    pytest.fail(f"{function_name} not found in routes/ssh.py — was it renamed?")


@pytest.mark.parametrize("handler,scope", sorted(_SSH_KEY_ROUTE_SCOPES.items()))
def test_every_ssh_key_route_requires_its_scope(handler, scope):
    required = _scopes_required_by(handler)
    assert scope in required, (
        f"{handler} does not require {scope!r} (requires {required}). An SSH key "
        "route reachable by a machine credential with no scope check is how "
        "deletion became the only one of the three an under-scoped agent key "
        "could call."
    )


def test_the_check_fails_on_a_handler_that_enforces_nothing():
    """The failing arm: the delete handler as it stood before this commit."""
    assert _scopes_required_by("api_delete_ssh_key"), "delete is scoped now"
    # A handler with no _require_scope call must read as empty, which is what the
    # assertion above would have caught.
    assert _scopes_required_by("_trigger_reinject_for_user") == []
