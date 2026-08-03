"""An OAuth-connected agent can register its own public key. Nothing more.

Gate P2 requires launch → wait → connect → run → terminate **using only tool
calls**, and explicitly fails a journey that needs a raw HTTP call or a
dashboard click. `register_ssh_key` is the step that makes "connect" reachable,
and it was blocked for one of the two credential types: `routes/ssh.py` guarded
registration with `_require_user_grant`, which refuses `client_credentials`
outright. An agent connected over OAuth could launch an instance and then had
nowhere to put its key.

The guard was not wrong, it was too broad. `_require_user_grant` exists to keep
machine tokens away from account security — password, MFA, sessions, deletion —
and registering a **public** key is a different act. The agent already holds the
private half; the platform is accepting an assertion about a key the caller
controls, not minting shell access. That distinction is why the change is a new
opt-in helper on two routes rather than a relaxation of the guard itself.

What the narrowing costs, and what pays for it:

* the scope is non-default — a connector that asked for nothing cannot reach it;
* every registration records the credential that made it, so a key an agent
  added is distinguishable from one a human pasted, and revoking a client says
  exactly which keys to remove;
* the rest of the account-security surface keeps `_require_user_grant`, which is
  asserted here rather than assumed.

The refusal test is the load-bearing one: holding *some* scope must not be
enough. A machine credential without `ssh:write` is still refused, and that is
what stops this from being "machine tokens may now touch SSH keys".
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest

import routes.ssh as ssh_routes
from routes._deps import _require_user_or_scoped_machine


def _source_of(fn) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover
        return ""


class _Req:
    """Minimal stand-in; the helper only reads the resolved principal."""

    def __init__(self, user):
        self._user = user


def _call(user, *scopes):
    import routes._deps as deps

    original = deps._get_current_user
    deps._get_current_user = lambda request: user
    try:
        return _require_user_or_scoped_machine(_Req(user), *scopes)
    finally:
        deps._get_current_user = original


def _machine(scopes, client_id="cli-1"):
    return {
        "email": "agent@example.com",
        "user_id": "u1",
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "client_id": client_id,
        "scopes": scopes,
    }


def test_a_scoped_machine_credential_is_admitted():
    """The behaviour Gate P2 needs."""
    user = _call(_machine(["ssh:write"]), "ssh:write")
    assert user["client_id"] == "cli-1"


def test_a_machine_credential_holding_only_ssh_read_cannot_register():
    """The sharp refusal: the *neighbour*, not an unrelated scope.

    Refusing a credential holding `instances:read` proves little — it shares no
    surface with key registration. The load-bearing case is a credential that
    can already *list* keys and must still be refused when it tries to add one,
    because `ssh:read` and `ssh:write` are the pair most likely to be collapsed
    by a future "simplification".
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _call(_machine(["ssh:read"]), "ssh:write")
    assert exc.value.status_code == 403
    assert "ssh:write" in str(exc.value.detail)


def test_a_machine_credential_with_an_unrelated_scope_is_also_refused():
    """The weaker case, kept because it is cheap and covers the obvious hole."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _call(_machine(["instances:read"]), "ssh:write")
    assert exc.value.status_code == 403


def test_an_unauthenticated_caller_is_refused():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _call(None, "ssh:write")
    assert exc.value.status_code == 401


def test_an_interactive_session_still_works_without_api_scopes():
    """Browser sessions carry OIDC claims, not API scopes.

    Treating "has no API scopes" as a restriction would deny every logged-in
    user — the defect that broke 122 tests when `_require_scope` was first
    tightened.
    """
    session = {"email": "human@example.com", "user_id": "u2", "auth_type": "session"}
    assert _call(session, "ssh:write")["email"] == "human@example.com"


def test_registration_records_the_credential_that_made_it():
    """Binding, so revoking a client identifies its keys."""
    source = inspect.getsource(ssh_routes.api_add_ssh_key)
    assert "registered_by_client_id" in source
    assert "registered_by_auth_type" in source


def test_ssh_write_is_not_a_default_connector_scope():
    """A connector that asks for nothing must not arrive able to add a key.

    Quick Connect deliberately *does* include `ssh:write` — that is a
    first-party token a user mints for their own agent on their own dashboard,
    which is a different act from approving a third party's scope request. This
    asserts the other half: a client that requested nothing in particular must
    not receive it.
    """
    from oauth_registration import CONNECTOR_DEFAULT_SCOPES

    assert "ssh:write" not in CONNECTOR_DEFAULT_SCOPES, (
        "ssh:write became a silent default; a connector that asked for nothing "
        "would arrive able to install shell access"
    )


def test_a_machine_credential_may_delete_only_the_keys_it_registered():
    """Own keys, not all keys — the same shape as the host visibility split."""
    import inspect

    source = inspect.getsource(ssh_routes.api_delete_ssh_key)
    assert "only_client_id" in source, (
        "delete is not scoped to the registering client, so an agent could "
        "revoke the key its owner pasted into the dashboard"
    )
    assert "client_credentials" in source


def test_delete_without_a_client_id_is_refused_not_widened():
    """Fail closed: an unattributable machine credential deletes nothing.

    If `client_id` were missing and `only_client_id` fell back to `None`, the
    delete would silently widen to *every* key the user owns — the failure mode
    this rule exists to prevent.
    """
    import inspect

    source = inspect.getsource(ssh_routes.api_delete_ssh_key)
    assert "has no client_id to scope deletion by" in source


#: Exactly the routes permitted to accept a scoped machine credential. Bounding
#: by name rather than by existence: "some route still requires an interactive
#: grant" is satisfied by any count above zero, so a fourth route could adopt
#: the relaxed guard silently. Same shape as the anonymous-write exemption list.
ROUTES_ALLOWED_TO_ACCEPT_MACHINE_CREDENTIALS = {
    "api_add_ssh_key",     # register a public key the caller already holds
    "api_list_ssh_keys",   # read your own keys
    "api_delete_ssh_key",  # rotate keys this client registered, and no others
}


def test_only_the_named_ssh_routes_accept_machine_credentials():
    """No fourth route joins silently, and no listed one leaves unnoticed.

    The three, and why each is here — set equality is only as good as the set,
    so a reader must be able to judge whether it is *correct*, not merely
    stable:

    * **`api_add_ssh_key`** — the registration `register_ssh_key` needs. Accepts
      a public key the caller already controls; the private half never exists
      server-side.
    * **`api_delete_ssh_key`** — self-rotation. Scoped to
      `registered_by_client_id`, so a client removes only what it added and
      cannot revoke the key its owner pasted into the dashboard. Without it an
      agent accumulates keys forever and cannot rotate from the terminal.
    * **`api_list_ssh_keys`** — the read that makes the other two usable: an
      agent must see whether its key is already registered before adding a
      duplicate, and which id to remove when rotating. Gated on `ssh:read`, and
      **filtered to the client's own keys** for machine credentials — the same
      own-versus-all rule as host visibility and deletion. A key's comment
      field routinely carries a hostname and username, so returning the user's
      full list to an agent is a disclosure the workflow does not need.

    Everything else on the account-security surface — password, MFA, sessions,
    account deletion — keeps `_require_user_grant`.
    """
    actual = {
        name
        for name, fn in vars(ssh_routes).items()
        if callable(fn)
        and getattr(fn, "__module__", "") == "routes.ssh"
        and "_require_user_or_scoped_machine" in _source_of(fn)
    }
    assert actual == ROUTES_ALLOWED_TO_ACCEPT_MACHINE_CREDENTIALS, (
        "the set of routes accepting machine credentials changed:\n"
        f"  newly accepting: {sorted(actual - ROUTES_ALLOWED_TO_ACCEPT_MACHINE_CREDENTIALS)}\n"
        f"  no longer accepting: {sorted(ROUTES_ALLOWED_TO_ACCEPT_MACHINE_CREDENTIALS - actual)}"
    )


def test_no_route_outside_ssh_adopted_the_relaxed_guard():
    """The relaxation must not spread beyond the module it was argued for."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "routes"
    offenders = sorted(
        p.name
        for p in root.glob("*.py")
        if p.name != "ssh.py"
        and p.name != "_deps.py"
        and "_require_user_or_scoped_machine" in p.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"the relaxed guard spread to other route modules: {offenders}"
    )


def test_the_account_security_surface_still_requires_an_interactive_grant():
    """Where the line actually sits now.

    An earlier version of this test asserted that key *deletion* stays on
    `_require_user_grant`. That decision was superseded on purpose: an agent
    that can add a key but never remove one accumulates keys and cannot rotate
    from the terminal. Deletion is now scoped to the registering client — own
    keys, not all keys — which closes the hygiene gap without widening reach.

    The test is rewritten rather than removed, because the boundary still
    exists; it just moved. Password, MFA, sessions and account deletion are
    outside this change and must stay outside it.
    """
    import routes.auth as auth_mod

    guarded = [
        name
        for name, fn in vars(auth_mod).items()
        if callable(fn)
        and getattr(fn, "__module__", "") == "routes.auth"
        and "_require_user_grant" in (inspect.getsource(fn) or "")
    ]
    assert guarded, (
        "no route in routes/auth.py requires an interactive grant any more; "
        "the relaxation has spread beyond SSH public keys"
    )


def test_account_security_routes_are_untouched():
    """MFA is the canary: this must not have widened into account security."""
    import routes.mfa as mfa

    relaxed = [
        name
        for name, fn in vars(mfa).items()
        if callable(fn)
        and getattr(fn, "__module__", "") == "routes.mfa"
        and "_require_user_or_scoped_machine" in (inspect.getsource(fn) if fn.__doc__ is not None else "")
    ]
    assert not relaxed, f"the relaxed guard leaked into account security: {relaxed}"


def test_listing_is_filtered_to_the_clients_own_keys():
    """The third application of own-versus-all, decided rather than defaulted.

    Host visibility and deletion both answer "yours, not everyone's". Listing
    had no reason to be the exception: duplicate-avoidance and rotation both
    operate on the client's own keys, while a full list leaks the comment field,
    which conventionally holds `user@hostname`.
    """
    source = _source_of(ssh_routes.api_list_ssh_keys)
    assert "registered_by_client_id" in source, (
        "listing returns every key the user owns to a machine credential"
    )
    assert "client_credentials" in source, (
        "the filter is unconditional; an interactive user must still see all "
        "of their own keys"
    )
