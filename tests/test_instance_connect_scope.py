"""`instances:connect` is enforced, not merely advertised.

The scope existed in every place except the one that matters. It is defined,
carries a consent description, and is in the delegable set — so a user is shown

    "Open a terminal on your running instances, and publish their ports to the
     internet"

and asked to approve it. Nothing checked it. `grep -r '_require_scope(user,
"instances:connect")'` returned nothing across the whole repository, while the
three endpoints that sentence describes sat behind `_require_auth` alone.

That is worse than an unscoped endpoint. An unscoped endpoint is a gap; a scope
that is displayed at the moment of consent and enforced nowhere is a statement
to the user that is not true. A token granted `instances:read` for monitoring
reached all three, and a token *denied* `instances:connect` reached them too —
the grant decision had no effect either way.

Each of the three is one clause of that sentence:

* `api_instance_stream_ticket` — mints the WebSocket ticket for the browser
  terminal. *"Open a terminal."*
* `api_instances_auto_launch_get` — returns each auto-launched service's public
  URL **with its access token in the query string**. Reading this is equivalent
  to being handed the Jupyter session.
* `api_instances_expose` — publishes a container port on a public HTTPS
  hostname. *"Publish their ports to the internet."*

**Ownership was already enforced and is not what this fixes.** All three check
that the caller owns the instance. The question this answers is different: given
that it *is* your instance, does this particular credential get to open a shell
on it? That is what the user was asked, and until now the answer was always yes.

This is P0 in the implementation plan — §0.1, *"the access endpoints are
authenticated but not authorized"* — and the plan is explicit that it comes
first "or the access phase builds on sand." P1's billing work was built while
this was still open.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

import routes.instances as instance_routes  # noqa: E402

#: Handler -> the scope it must require. Listed one per line so that deleting an
#: entry shows up in review as a removed guarantee rather than a changed count.
CONNECT_ROUTES = {
    "api_instance_stream_ticket": "instances:connect",
    "api_instances_auto_launch_get": "instances:connect",
    "api_instances_expose": "instances:connect",
}


@pytest.mark.parametrize("handler,scope", sorted(CONNECT_ROUTES.items()))
def test_each_connection_route_requires_its_scope(handler, scope):
    """The defect: advertised at consent, enforced nowhere."""
    fn = getattr(instance_routes, handler, None)
    assert fn is not None, f"{handler} no longer exists — was it renamed?"

    source = inspect.getsource(fn)
    assert f'_require_scope(user, "{scope}")' in source, (
        f"{handler} does not require {scope!r}. The consent screen tells users "
        "this scope controls opening a terminal and publishing ports; if no "
        "route enforces it, approving or refusing it changes nothing"
    )


def test_the_worker_callback_is_not_given_a_user_scope():
    """The refusal, and the reason this is three routes rather than four.

    `POST /instances/{id}/auto-launch/report` looks like a fourth connection
    endpoint and is not one. It is the worker agent reporting which ports it
    published, authenticated by the shared agent secret and bound to the
    reporting host_id — a different principal entirely, with no user, no OAuth
    token and therefore no scopes.

    Adding `instances:connect` there would not tighten it. It would break
    auto-launch reporting on every host, because `_require_scope` would be
    handed an agent principal that cannot carry user scopes. A scope applied to
    the wrong principal is not a smaller version of the right fix.
    """
    source = inspect.getsource(instance_routes.api_instances_auto_launch_report)
    assert "_require_agent_auth" in source, (
        "the auto-launch report no longer authenticates as the worker agent; if "
        "it became a user route, it needs a user scope after all"
    )
    assert "instances:connect" not in source, (
        "the worker callback was given a user scope — it has no user principal"
    )


def test_the_scope_is_described_where_it_is_granted():
    """A scope a user is asked to approve has to say what it does.

    This is the half that was already right, asserted so the fix cannot be
    "delete the description" if the two ever disagree again.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    text = SCOPE_DESCRIPTIONS.get("instances:connect", "")
    assert text, "instances:connect has no consent description"
    lowered = text.lower()
    assert "terminal" in lowered, "the description no longer mentions terminal access"
    assert "port" in lowered, "the description no longer mentions publishing ports"


def test_the_scope_can_actually_be_granted():
    """Enforcing a scope nobody can hold would lock every agent out.

    `hosts:fleet` and the transparency scopes were each enforced while ungrantable,
    which produces a refusal that reads as a bug and gets "fixed" by removing the
    check. `instances:connect` is listed under *MCP quick connect*, so the token
    this product tells people to paste already carries it and the endpoints keep
    working for the flow they exist to serve.
    """
    from oauth_delegation import SYSTEM_ALLOWED_SCOPES

    assert "instances:connect" in SYSTEM_ALLOWED_SCOPES, (
        "instances:connect is enforced but no seeded client can grant it, so no "
        "agent credential can ever satisfy it"
    )


def test_connecting_to_your_own_instance_is_not_operator_authority():
    """Guards the shape of the eventual fix, not just its presence.

    If enforcement ever produces an unexpected refusal, the cheap repair is to
    promote the scope into `OPERATOR_SCOPES` so admins sail through. That would
    make opening a shell on your own instance require platform-operator
    authority, which is both wrong and a privilege escalation dressed as a fix.
    """
    from oauth_delegation import OPERATOR_SCOPES

    assert "instances:connect" not in OPERATOR_SCOPES, (
        "instances:connect became operator authority — it governs a user's own "
        "instances, not the platform"
    )


def test_an_interactive_session_is_unaffected():
    """Why this is safe to add: it changes nothing for a browser user.

    A dashboard session carries OIDC identity scopes, which say nothing about
    API authority, and `_require_scope` passes them through. If that were not
    true, adding this to three routes would break the instance view and the fix
    would get reverted rather than corrected.
    """
    from routes._deps import _require_scope

    session = {
        "email": "human@example.com",
        "user_id": "u1",
        "auth_type": "oauth_access_token",
        "scopes": ["openid", "profile", "email"],
    }
    _require_scope(session, "instances:connect")  # must not raise


def test_a_narrowed_machine_credential_is_refused():
    """The calibration control.

    Without it, `_require_scope` silently becoming a no-op for every principal
    would satisfy every assertion above — which is precisely the state this file
    was written to end.
    """
    from fastapi import HTTPException

    from routes._deps import _require_scope

    narrowed = {
        "email": "agent@example.com",
        "user_id": "u1",
        "auth_type": "agent_api_key",
        "scopes": ["instances:read"],
    }
    with pytest.raises(HTTPException) as excinfo:
        _require_scope(narrowed, "instances:connect")
    assert excinfo.value.status_code == 403
