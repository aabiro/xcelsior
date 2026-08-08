"""Who may grant which scopes — one definition, enforced where scopes are stored.

Issue #16: *the guard is on the door, not the lock.* `assert_scopes_delegable` was
called at `POST /api/oauth/clients`, so the invariant "no non-admin holds an
operator scope" depended on registration being the only path that writes
`oauth_clients.scopes`. It was not: `PATCH /api/oauth/clients/{id}` writes it too,
which is how a two-request escalation survived the first fix. Any future path —
admin tooling, a seeding script, a migration backfill, a flow that provisions a
service client — reintroduces it with every test still green.

This module is the lock. `db.OAuthStore.create_client`, `update_client` and
`update_client_in_workspace` call `assert_delegable` themselves, so a caller cannot
omit it by not knowing about it.

Deliberately dependency-free. `oauth_service` imports `db`, so anything `db`
imports must not reach back into `oauth_service` — a module with no imports of its
own cannot create that cycle.

## Why `actor` has no default

`assert_delegable(scopes, *, actor)` — a required keyword. If the default were
`SYSTEM_PRINCIPAL`, a new caller that forgets the argument would be silently
exempted, and "the permissive value is what you get when you don't specify" is the
defect class this whole review sequence has been about. A missing argument is a
`TypeError` at the call site instead.

## Why the system path is not simply exempt

Two callers legitimately create clients with no user behind them: the seeded
default clients, and MCP quick-connect. They pass `SYSTEM_PRINCIPAL`, and that
still runs a check — against `SYSTEM_ALLOWED_SCOPES`, which is exactly the scopes
those two paths use and nothing else. They are currently safe by their *contents*,
and contents go stale; a future system-managed caller asking for `hosts:evict` is
refused rather than trusted for being a system caller.
"""

from __future__ import annotations

#: Sentinel for "no user is behind this call". A distinct object compared with
#: `is`, not a string: a string sentinel compared with `==` would be a fifth
#: instance of the one-definition-two-spellings defect that produced #16 in the
#: first place.
SYSTEM_PRINCIPAL = object()


class ScopeDelegationError(Exception):
    """A principal asked to grant a scope it may not grant."""


#: Scopes conferring authority over the *platform* rather than the caller's own
#: resources. `control_plane_v1._require_host_operator` authorises a machine
#: principal on scope alone — correctly, since a machine has no role to inspect —
#: which makes the write of this column the only place the authority can be
#: withheld.
#:
#: A literal, not derived from `SCOPE_DESCRIPTIONS`'s `(operator)` annotations,
#: and that is a deliberate choice recorded rather than a shortcut. On this branch
#: the annotations name five scopes and this set names seven; three of these
#: (`hosts:fleet`, `transparency:read`, `transparency:write`) are absent from
#: `SCOPE_DESCRIPTIONS` entirely. Deriving would silently narrow the guard to five
#: and reopen the other two. `tests/test_scope_delegation.py` asserts the
#: relationship in both directions so the divergence is visible and must be
#: resolved deliberately.
OPERATOR_SCOPES = frozenset(
    {
        "control_plane:operate",
        "control_plane:read",
        "hosts:evict",
        "hosts:fleet",
        "hosts:operate",
        "transparency:read",
        "transparency:write",
        # Added when every enforced scope was given a consent description and
        # the annotation ratchet asked what `(operator)` meant for these three.
        # Each is platform authority by evidence rather than by name:
        #
        #   autoscale:write  routes/autoscale.py guards it with
        #                    `_require_scope(_require_admin(request), ...)` —
        #                    the route already demands admin
        #   sla:write        same shape in routes/sla.py
        #   admin            guards POST /api/v2/admin/volumes/reopen-encrypted;
        #                    a scope literally named `admin`, on an admin route
        #
        # `reputation:write` was annotated `(operator)` at the same time and is
        # *not* here: it guards `/api/reputation/me/claim`, where a provider
        # claims milestones they earned. The annotation was wrong, and the
        # description was corrected rather than the set widened to match it.
        "autoscale:write",
        "sla:write",
        "admin",
    }
)

#: What `SYSTEM_PRINCIPAL` may grant: the union of what the two system callers
#: actually use. Narrow on purpose — see the module docstring. Extending this is a
#: decision, and a visible one.
SYSTEM_ALLOWED_SCOPES = frozenset(
    {
        # Seeded first-party clients and the web/desktop apps.
        "instances:read",
        "instances:write",
        "instances:operate",
        "billing:read",
        "billing:write",
        "gpu:read",
        "marketplace:read",
        "inference:read",
        "inference:write",
        "events:read",
        "mcp_actions:approve",
        "hosts:read",
        # MCP quick connect.
        "instances:connect",
        "ssh:read",
        "ssh:write",
        # P3 durable state. Added here *and* to MCP_QUICK_CONNECT_SCOPES, which
        # are two different questions: this list is what the system principal
        # may grant, that one is what a minted token carries. Granting in one
        # without the other is the defect that produced `instances:connect`
        # answering 403 in production — the scope was enforced on the routes and
        # missing from the held set. This time `tests/test_scope_delegation.py`
        # caught it before a deploy, which is what that guard is for.
        "volumes:read",
        "volumes:write",
        "artifacts:read",
        "openid",
        "profile",
        "email",
        "offline_access",
        # Legacy blanket scope carried by seeded machine clients. Not operator
        # authority, and not in OPERATOR_SCOPES — but broad, so it is listed
        # explicitly rather than arriving by a wildcard.
        "api",
    }
)


def is_platform_admin(actor: object) -> bool:
    """True for a principal the platform treats as an admin.

    Mirrors `routes._deps._is_platform_admin` rather than importing it, because
    that module imports `db` and this one must import nothing. The two are pinned
    together by `tests/test_scope_delegation.py`.
    """
    if not isinstance(actor, dict):
        return False
    flag = actor.get("is_admin")
    return flag in (True, 1, "1", "true", "True") or actor.get("role") == "admin"


def known_scopes() -> frozenset[str]:
    """Every scope this platform defines, from the places that define them.

    Built from the three existing sources rather than restated, so a scope
    added to one of them cannot be missing here — a fourth hand-kept list is
    how a vocabulary check ends up rejecting something legitimate.

    Imported lazily because `oauth_service` imports this module; at module
    scope it would be a cycle.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    return frozenset(SCOPE_DESCRIPTIONS) | OPERATOR_SCOPES | SYSTEM_ALLOWED_SCOPES


def assert_delegable(scopes: object, *, actor: object) -> None:
    """Raise `ScopeDelegationError` if *actor* may not grant *scopes*.

    `scopes is None` means the caller did not mention scopes — a `PATCH` that
    changes only a client's name must not be refused. An empty list is a real
    request to hold nothing and passes trivially.
    """
    if scopes is None:
        return
    requested = {str(s).strip() for s in scopes if str(s).strip()}
    if not requested:
        return

    # Vocabulary first, and for every actor including admins.
    #
    # This checked `OPERATOR_SCOPES` — a denylist — so anything that was not an
    # operator scope passed and was stored. A non-admin could register a client
    # holding `totally:invented`, or, more to the point,
    # `"Full access to your account - this is safe and standard"`.
    #
    # An invented scope grants nothing: enforcement is membership, so it can
    # never satisfy a check. The damage is at consent. `describe_scope` falls
    # back to rendering the scope *as itself* when it has no description, so an
    # unknown scope becomes attacker-chosen prose on a first-party consent page,
    # presented as a permission the user is about to grant. Dynamic
    # registration was already closed against this (`CONNECTOR_ALLOWED_SCOPES`);
    # the authenticated client-creation path was not.
    #
    # Admins are included deliberately. Adding a scope means giving it a
    # description a user can consent to, which is a code change — not something
    # typed into a form. An admin typo should fail here rather than become a
    # phantom scope nobody can grant and nothing enforces.
    unknown = sorted(requested - known_scopes())
    if unknown:
        raise ScopeDelegationError(
            "Not scopes this platform defines: "
            + ", ".join(repr(u) for u in unknown)
            + ". A scope must have a consent description before it can be "
            "granted — otherwise the authorization screen shows the user raw "
            "text as if it were a permission."
        )

    if actor is SYSTEM_PRINCIPAL:
        outside = sorted(requested - SYSTEM_ALLOWED_SCOPES)
        if outside:
            raise ScopeDelegationError(
                "A system-managed client may not be created with "
                f"{', '.join(outside)}. If a platform path now needs these, add "
                "them to SYSTEM_ALLOWED_SCOPES deliberately — this refusal exists "
                "so a new system caller cannot inherit operator authority by "
                "being a system caller."
            )
        return

    if is_platform_admin(actor):
        return

    operator = sorted(requested & OPERATOR_SCOPES)
    if operator:
        raise ScopeDelegationError(
            "These scopes may only be granted by a platform administrator: "
            + ", ".join(operator)
        )
