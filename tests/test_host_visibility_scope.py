"""`GET /hosts` returned the whole fleet to anyone holding `hosts:read`.

`hosts:read` is dual-use, and that is why it could not simply be reclassified.
It gates a provider's own admission status and heartbeat — every worker agent is
registered by a **non-admin** provider, so admin-gating it breaks onboarding —
and it also gated `GET /hosts`, which returns every host on the platform:
capacity, GPU models, owners, admission state.

So a credential that a provider legitimately needs in order to run their own
rig could enumerate the entire fleet, including competitors' capacity.

Reclassifying the scope was the wrong fix and was reverted once already. The
right one is additive, and splits the two uses apart:

* **`hosts:read`** stays freely grantable and now answers *your* hosts.
* **`hosts:fleet`** is a new operator scope for platform-wide visibility.
  Marked "(operator)" in `SCOPE_DESCRIPTIONS`, so `assert_scopes_delegable`
  refuses to let a non-admin mint it — the same rule that closed the
  `hosts:evict` escalation.

Admins keep fleet visibility without holding the scope, because the operator
gate has always been "admin, or a machine principal with the scope".

The failure this prevents is silent: no error, no refusal, just more rows than
the caller should see. That is why the assertions below are about *absence* of
another provider's host, not about a status code.
"""

from __future__ import annotations

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"

import pytest

from routes.hosts import visible_hosts


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
    assert owners == {"provider-a"}, (
        f"a provider credential saw hosts it does not own: {owners}"
    )
    assert len(visible) == 2


def test_a_provider_cannot_see_a_competitors_capacity():
    """Named separately because this is the disclosure that mattered.

    Fleet-wide listing exposes competitors' GPU models, counts, and admission
    state to anyone who registered a rig.
    """
    visible = visible_hosts(_user("provider-a", scopes=["hosts:read"]), FLEET)
    assert not [h for h in visible if h["owner"] == "provider-b"]


def test_the_fleet_scope_restores_platform_wide_visibility():
    """The capability is not removed, only moved behind an operator scope."""
    visible = visible_hosts(
        _user("ops", scopes=["hosts:read", "hosts:fleet"]), FLEET
    )
    assert len(visible) == len(FLEET)


def test_an_admin_keeps_fleet_visibility_without_the_scope():
    """Operator gates here have always been 'admin, or the scope'.

    Requiring admins to also hold the scope would break the dashboard, which is
    how this would get reverted.
    """
    visible = visible_hosts(_user("root", admin=True), FLEET)
    assert len(visible) == len(FLEET)


def test_an_interactive_user_without_scopes_sees_only_their_own():
    """Browser sessions carry no API scopes, so they must not default to fleet.

    `_require_scope` is a no-op for interactive sessions — if visibility keyed
    on "has no scopes" it would hand every logged-in user the whole fleet,
    which is the defect with extra steps.
    """
    visible = visible_hosts(_user("provider-b"), FLEET)
    assert {h["owner"] for h in visible} == {"provider-b"}


def test_unowned_hosts_are_not_visible_to_an_arbitrary_provider():
    """A row with no owner must not become everyone's host."""
    visible = visible_hosts(_user("provider-a", scopes=["hosts:read"]), FLEET)
    assert not [h for h in visible if not h.get("owner")]


def test_a_worker_credential_sees_the_host_it_was_created_for(monkeypatch):
    """Provisioning must keep working — this is why the scope stayed grantable.

    A worker agent's OAuth client is created by the provider, so ownership
    resolves through the creator identity, exactly as `_require_host_operator`
    already does.

    The creator lookup itself hits the OAuth client table and is covered by the
    worker-provisioning tests; what is asserted here is that `visible_hosts`
    *consults* it. Without that, a worker credential whose own identity is the
    client rather than the provider would see nothing and provisioning would
    silently stop working.
    """
    import routes.hosts as hosts_mod

    worker = _user("worker-client", scopes=["hosts:read", "hosts:write"])
    monkeypatch.setattr(
        hosts_mod,
        "_oauth_client_creator",
        lambda user: {"customer_id": "provider-a", "user_id": "provider-a"},
    )
    visible = visible_hosts(worker, FLEET)
    assert {h["owner"] for h in visible} == {"provider-a"}


def test_a_worker_credential_without_a_resolvable_creator_sees_nothing():
    """Fail closed: an unresolvable creator must not widen visibility."""
    worker = _user("worker-orphan", scopes=["hosts:read"])
    assert visible_hosts(worker, FLEET) == []


def test_fleet_read_is_operator_authority():
    """A non-admin must not be able to grant themselves fleet visibility.

    Without this the split is decorative: any user could register a client
    holding `hosts:fleet` and read the fleet anyway — exactly the
    `hosts:evict` escalation in a different costume.
    """
    from oauth_service import OAuthGrantError, assert_scopes_delegable

    with pytest.raises(OAuthGrantError):
        assert_scopes_delegable(
            ["hosts:fleet"],
            creator={"email": "t@example.com", "is_admin": False, "role": "user"},
        )
