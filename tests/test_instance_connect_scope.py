"""Connecting to a running instance requires `instances:connect`.

§0.1 of the plan: *`auto-launch`, `expose`, `stream-ticket` and `ssh/keygen` all
sit behind `_require_auth` — authentication, not authorization.* Any
authenticated principal could open a terminal ticket, expose a container port to
the public internet, or read the credentials auto-launch generated for Jupyter,
regardless of what its credential was granted.

That matters most for a **narrowed** credential. A Quick Connect token issued
with `instances:read` so an agent can watch a job could also open a shell on it,
because reading and connecting were the same authority: none.

`instances:connect` separates them. It is granted by default in Quick Connect —
connecting from the terminal is the workflow the whole plan exists for — but it
is now a scope a credential either holds or does not, which is what makes
"issue a read-only agent token" mean something.

**`/ssh/keygen` is deliberately not on this list.** It mints the *platform's*
host-access private key server-side. That is infrastructure, not a user
capability, and it is admin-only rather than scoped — putting it behind a scope
an agent is meant to hold would hand an agent the ability to mint platform
credentials. `docs/mcp-agent-native-implementation-plan.md` records the split.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest

import routes.health as health_routes
import routes.instances as instance_routes

#: Endpoints that let a caller reach *into* a running instance. Named
#: individually so removing one is visible in review rather than showing up as
#: a count change.
CONNECT_HANDLERS = [
    "api_instance_stream_ticket",
    "api_instances_expose",
    "api_instances_auto_launch_get",
]


@pytest.mark.parametrize("handler", CONNECT_HANDLERS)
def test_connect_endpoints_require_the_connect_scope(handler):
    """Authentication is not authorization — the defect §0.1 names."""
    source = inspect.getsource(getattr(instance_routes, handler))
    assert '_require_scope(user, "instances:connect")' in source, (
        f"{handler} does not require `instances:connect`, so any authenticated "
        "principal can reach into a running instance regardless of what its "
        "credential was granted"
    )


@pytest.mark.parametrize("handler", CONNECT_HANDLERS)
def test_connect_endpoints_still_check_ownership(handler):
    """The scope is additional to ownership, never a replacement for it.

    A caller holding `instances:connect` must still only reach *their own*
    instances. If a scope check ever replaced the ownership check, every holder
    of the scope could open a shell on anyone's job.
    """
    source = inspect.getsource(getattr(instance_routes, handler))
    assert any(
        marker in source
        for marker in ("_check_job_access", "_canonical_owner_id", "owner")
    ), f"{handler} no longer establishes ownership"


def test_the_connect_scope_is_grantable():
    """A scope enforced but not issuable seals the endpoint instead of guarding it."""
    import oauth_service

    assert "instances:connect" in oauth_service.SCOPE_DESCRIPTIONS


def test_connect_is_not_implied_by_read():
    """Reading a job and entering it are different authorities.

    This is the property that makes a read-only agent token meaningful. If
    `instances:read` ever satisfies a connect endpoint, the separation is
    decorative.
    """
    from routes._deps import _require_scope
    from fastapi import HTTPException

    reader = {
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "scopes": ["instances:read"],
    }
    with pytest.raises(HTTPException) as exc:
        _require_scope(reader, "instances:connect")
    assert exc.value.status_code == 403


def test_ssh_keygen_is_admin_only_and_not_scoped():
    """It mints the platform's private key. No agent scope may reach it.

    Asserted as *both* halves: admin is required, and no `instances:connect` or
    `ssh:*` scope grants it. A future change that "makes keygen usable by
    agents" has to delete this test to do so.
    """
    source = inspect.getsource(health_routes.api_generate_ssh_key)
    assert "_require_admin" in source, (
        "/ssh/keygen is not admin-only; it generates the platform's "
        "host-access private key"
    )
    assert "_require_scope" not in source, (
        "/ssh/keygen is scope-gated, which would let a credential holding an "
        "agent scope mint platform key material"
    )


def test_ssh_pubkey_read_stays_open():
    """The public half is meant to be distributed to hosts.

    Pinned so that tightening keygen does not sweep this up by association —
    it would break host provisioning for no security gain.
    """
    source = inspect.getsource(health_routes.api_get_pubkey)
    assert "_require_admin" not in source
