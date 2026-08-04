"""`GET /hosts` returned the whole fleet to anyone holding `hosts:read`.

`hosts:read` is dual-use, and that is why it could not simply be reclassified.
It gates a provider's own admission status and heartbeat — every worker agent is
registered by a **non-admin** provider, so admin-gating it breaks onboarding —
and it also gated `GET /hosts`, which returns every host on the platform:
capacity, GPU models, owners, admission state.

So a credential that a provider legitimately needs in order to run their own rig
could enumerate the entire fleet, including competitors' capacity.

Reclassifying the scope is the wrong fix. The right one is additive, and splits
the two uses apart:

* **`hosts:read`** stays freely grantable and now answers *your* hosts.
* **`hosts:fleet`** is a new operator scope for platform-wide visibility. It is
  in `oauth_delegation.OPERATOR_SCOPES`, so the delegation guard refuses to let
  a non-admin mint it — the same rule that closed the `hosts:evict` escalation.

Admins keep fleet visibility without holding the scope, because the operator
gate has always been "admin, or a machine principal with the scope".

The failure this prevents is silent: no error, no refusal, just more rows than
the caller should see. That is why most assertions below are about the *absence*
of another provider's host rather than about a status code.

**Single-host reads are covered too.** Filtering only the list endpoint leaves
`GET /host/{id}` and `GET /api/hosts/{id}/spot-preview` returning any row by id,
and the spot preview discloses a provider's pricing floor, which is commercially
sensitive on its own. Both now refuse with 404 rather than 403 — a 403 confirms
the host exists, which is most of what enumeration wanted.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

from routes.hosts import FLEET_READ_SCOPE, visible_hosts  # noqa: E402


def _host(owner: str, host_id: str | None = None) -> dict:
    return {
        "host_id": host_id or f"h-{uuid.uuid4().hex[:8]}",
        "owner": owner,
        "gpu_model": "RTX 4090",
        "status": "active",
    }


def _user(customer_id: str, *, admin: bool = False, scopes: list[str] | None = None) -> dict:
    return {
        "customer_id": customer_id,
        "user_id": customer_id,
        "email": f"{customer_id}@example.com",
        "is_admin": admin,
        "role": "admin" if admin else "user",
        "scopes": scopes or [],
        "grant_type": "client_credentials" if scopes else "",
    }


FLEET = [_host("provider-a"), _host("provider-a"), _host("provider-b"), _host("unowned-host")]
FLEET[3]["owner"] = ""


def test_a_provider_sees_only_their_own_hosts():
    """The defect, stated as the behaviour that replaces it."""
    visible = visible_hosts(_user("provider-a", scopes=["hosts:read"]), FLEET)
    owners = {h["owner"] for h in visible}
    assert owners == {"provider-a"}, f"a provider credential saw hosts it does not own: {owners}"
    assert len(visible) == 2


def test_a_provider_cannot_see_a_competitors_capacity():
    """Named separately because this is the disclosure that mattered."""
    visible = visible_hosts(_user("provider-a", scopes=["hosts:read"]), FLEET)
    assert not [h for h in visible if h["owner"] == "provider-b"]


def test_the_fleet_scope_restores_platform_wide_visibility():
    """The capability is not removed, only moved behind an operator scope."""
    visible = visible_hosts(_user("ops", scopes=["hosts:read", FLEET_READ_SCOPE]), FLEET)
    assert len(visible) == len(FLEET)


def test_an_admin_keeps_fleet_visibility_without_the_scope():
    """Operator gates here have always been 'admin, or the scope'.

    Requiring admins to also hold it would break the dashboard, which is how
    this would get reverted.
    """
    assert len(visible_hosts(_user("root", admin=True), FLEET)) == len(FLEET)


def test_an_interactive_user_without_scopes_sees_only_their_own():
    """Browser sessions carry no API scopes, so they must not default to fleet.

    `_require_scope` is a no-op for interactive sessions — if visibility keyed
    on "has no scopes" it would hand every logged-in user the whole fleet, which
    is the defect with extra steps.
    """
    assert {h["owner"] for h in visible_hosts(_user("provider-b"), FLEET)} == {"provider-b"}


def test_unowned_hosts_are_not_visible_to_an_arbitrary_provider():
    """A row with no owner must not become everyone's host."""
    visible = visible_hosts(_user("provider-a", scopes=["hosts:read"]), FLEET)
    assert not [h for h in visible if not h.get("owner")]


def test_a_worker_credential_sees_the_host_it_was_created_for(monkeypatch):
    """Provisioning must keep working — this is why the scope stayed grantable.

    A worker agent's OAuth client is created by the provider, so ownership
    resolves through the creator identity, exactly as `_require_host_operator`
    already does. What is asserted here is that `visible_hosts` *consults* it;
    without that, a worker credential whose own identity is the client rather
    than the provider would see nothing and provisioning would silently stop.
    """
    import routes.hosts as hosts_mod

    monkeypatch.setattr(
        hosts_mod,
        "_oauth_client_creator",
        lambda user: {"customer_id": "provider-a", "user_id": "provider-a"},
    )
    worker = _user("worker-client", scopes=["hosts:read", "hosts:write"])
    assert {h["owner"] for h in visible_hosts(worker, FLEET)} == {"provider-a"}


def test_a_worker_credential_without_a_resolvable_creator_sees_nothing():
    """Fail closed: an unresolvable creator must not widen visibility."""
    assert visible_hosts(_user("worker-orphan", scopes=["hosts:read"]), FLEET) == []


def test_fleet_read_is_operator_authority():
    """A non-admin must not be able to grant themselves fleet visibility.

    Without this the split is decorative: any user could register a client
    holding `hosts:fleet` and read the fleet anyway — the `hosts:evict`
    escalation in a different costume.
    """
    from oauth_delegation import OPERATOR_SCOPES, ScopeDelegationError, assert_delegable

    assert FLEET_READ_SCOPE in OPERATOR_SCOPES

    with pytest.raises(ScopeDelegationError):
        assert_delegable(
            [FLEET_READ_SCOPE],
            actor={"email": "t@example.com", "is_admin": False, "role": "user"},
        )


def test_fleet_read_is_grantable_by_an_admin():
    """Guarded is not the same as sealed.

    `hosts:fleet` was in OPERATOR_SCOPES but absent from SCOPE_DESCRIPTIONS, so
    it was enforced, ungrantable and unused all at once — a scope no credential
    could ever hold. It has to be describable for the split to be usable.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    assert FLEET_READ_SCOPE in SCOPE_DESCRIPTIONS, (
        "hosts:fleet is guarded but has no description, so no consent screen can "
        "render it and no admin can knowingly grant it"
    )
    assert "(operator)" in SCOPE_DESCRIPTIONS[FLEET_READ_SCOPE]


def test_hosts_read_is_no_longer_annotated_as_operator():
    """The annotation and the guard have to agree.

    `hosts:read` was described "(operator)" while *not* being in
    OPERATOR_SCOPES. Whichever a future reader believed, deriving one from the
    other would have silently changed behaviour.
    """
    from oauth_delegation import OPERATOR_SCOPES
    from oauth_service import SCOPE_DESCRIPTIONS

    assert "hosts:read" not in OPERATOR_SCOPES
    assert "(operator)" not in SCOPE_DESCRIPTIONS["hosts:read"], (
        "hosts:read is annotated as operator authority but is not guarded as "
        "such; the description and the guard must not contradict each other"
    )


def test_the_single_host_routes_apply_the_same_filter():
    """`GET /host/{id}` and the spot preview, not only the list.

    Asserted through `_require_host_visible` rather than over HTTP, because the
    route body resolves the row before the check and the interesting question is
    what the check does with a row the caller does not own.
    """
    from fastapi import HTTPException

    from routes.hosts import _require_host_visible

    stranger = _user("provider-a", scopes=["hosts:read"])
    with pytest.raises(HTTPException) as excinfo:
        _require_host_visible(stranger, _host("provider-b"))
    assert excinfo.value.status_code == 404, (
        "a host the caller cannot see must read as absent; 403 confirms it exists"
    )

    # And the owner is unaffected.
    _require_host_visible(stranger, _host("provider-a"))
