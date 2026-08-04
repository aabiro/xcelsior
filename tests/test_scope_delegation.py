"""The scope-delegation guard is on the lock, and every writer inherits it.

Issue #16. The check lived at `POST /api/oauth/clients`, so "no non-admin holds an
operator scope" held only while registration was the sole writer of
`oauth_clients.scopes`. It wasn't — `PATCH` writes it too, which is how a
two-request escalation survived the first fix.

The guard now lives in `oauth_delegation.assert_delegable` and is called by the
three `OAuthStore` methods that write the column, so a caller cannot omit it by not
knowing it exists. These tests assert that arrangement rather than the routes:
**no route is involved below**, which is the point — a refusal that only a route
can produce is a refusal a new caller bypasses.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from oauth_delegation import (
    OPERATOR_SCOPES,
    SYSTEM_ALLOWED_SCOPES,
    SYSTEM_PRINCIPAL,
    ScopeDelegationError,
    assert_delegable,
    is_platform_admin,
)

ADMIN = {"email": "admin@xcelsior.ca", "is_admin": True}
USER = {"email": "user@xcelsior.ca", "role": "submitter"}


# ── The policy ─────────────────────────────────────────────────────────


def test_a_non_admin_may_not_grant_an_operator_scope():
    with pytest.raises(ScopeDelegationError, match="hosts:evict"):
        assert_delegable(["instances:read", "hosts:evict"], actor=USER)


def test_a_non_admin_may_grant_ordinary_scopes():
    """The refusal is narrow. A blunt one is its own outage."""
    assert_delegable(["instances:read", "billing:read"], actor=USER) is None


def test_an_admin_may_grant_an_operator_scope():
    """Delegation, not prohibition — an admin *may*, which is why the check is
    about who is asking rather than about the scope alone."""
    assert_delegable(["hosts:evict"], actor=ADMIN) is None


@pytest.mark.parametrize("flag", [True, 1, "1", "true", "True"])
def test_every_truthy_admin_spelling_is_honoured(flag):
    """The database has stored this flag as an int, a bool and a string."""
    assert_delegable(["hosts:evict"], actor={"is_admin": flag}) is None


def test_role_admin_is_also_admin():
    assert_delegable(["hosts:evict"], actor={"role": "admin"}) is None


def test_none_means_the_request_did_not_mention_scopes():
    """A PATCH renaming a client must not be refused for scopes it never sent."""
    assert_delegable(None, actor=USER) is None


def test_an_empty_list_is_a_real_request_and_passes():
    assert_delegable([], actor=USER) is None


def test_an_unknown_actor_shape_is_treated_as_non_admin():
    """Fail closed on anything that is not recognisably an admin."""
    for actor in (None, "admin", 1, object()):
        with pytest.raises(ScopeDelegationError):
            assert_delegable(["hosts:evict"], actor=actor)


# ── The system sentinel ────────────────────────────────────────────────


def test_the_system_principal_is_checked_not_exempt():
    """The narrowing #16 asked for: system paths are safe by contents today, and
    contents go stale."""
    assert_delegable(["instances:read"], actor=SYSTEM_PRINCIPAL) is None
    with pytest.raises(ScopeDelegationError, match="hosts:evict"):
        assert_delegable(["hosts:evict"], actor=SYSTEM_PRINCIPAL)


def test_the_sentinel_is_an_object_not_a_string():
    """A string sentinel compared with `==` would be a fifth instance of the
    one-definition-two-spellings defect that produced #16."""
    assert not isinstance(SYSTEM_PRINCIPAL, str)
    assert SYSTEM_PRINCIPAL is not None


def test_the_system_allowlist_grants_no_operator_authority():
    overlap = sorted(SYSTEM_ALLOWED_SCOPES & OPERATOR_SCOPES)
    assert not overlap, (
        f"SYSTEM_ALLOWED_SCOPES contains operator scopes {overlap} — a system "
        "caller would inherit platform authority for being a system caller"
    )


# ── Every writer inherits it (the set-equality #16 specifies) ──────────

DB = pathlib.Path(__file__).resolve().parent.parent / "db.py"

#: The `OAuthStore` methods that write `oauth_clients.scopes`. Derived below from
#: the source rather than trusted, so a fourth writer fails this file.
_KNOWN_SCOPE_WRITERS = {"create_client", "update_client", "update_client_in_workspace"}


def _oauthstore_methods_writing_scopes() -> set[str]:
    """Methods of `OAuthStore` whose body mentions the scopes column."""
    tree = ast.parse(DB.read_text(encoding="utf-8"))
    store = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "OAuthStore"
    )
    writing = set()
    for fn in store.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(fn)
        mentions_scopes = '"scopes"' in body or "'scopes'" in body
        # `oauth_clients`, specifically. `create_refresh_token` also stores a
        # `scopes` column, on `oauth_refresh_tokens` — a token's scopes are derived
        # from the client that minted it, so it is not a grant path for platform
        # authority and #16 is not about it. (Whether a refresh token can be minted
        # with scopes exceeding its client's is a separate question, and not one
        # this gate answers.)
        touches_client_row = "oauth_clients" in body
        if mentions_scopes and touches_client_row:
            writing.add(fn.name)
    return writing


def _methods_calling_the_guard() -> set[str]:
    tree = ast.parse(DB.read_text(encoding="utf-8"))
    store = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "OAuthStore"
    )
    calling = set()
    for fn in store.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
                if name == "assert_delegable":
                    calling.add(fn.name)
    return calling


def test_every_scope_writer_calls_the_guard():
    writers = _oauthstore_methods_writing_scopes()
    guarded = _methods_calling_the_guard()
    unguarded = sorted(writers - guarded)
    assert not unguarded, (
        f"{unguarded} write oauth_clients.scopes without calling assert_delegable. "
        "That is exactly #16: a new writer inherits the escalation with every test "
        "still green. Call the guard, or explain here why the method cannot."
    )


def test_the_writer_inventory_has_not_drifted():
    """A fourth writer must be a conscious addition, not a discovery."""
    writers = _oauthstore_methods_writing_scopes()
    assert writers == _KNOWN_SCOPE_WRITERS, (
        f"the set of scope writers changed: found {sorted(writers)}, expected "
        f"{sorted(_KNOWN_SCOPE_WRITERS)}. Update this set *and* confirm the new "
        "member calls assert_delegable."
    )


@pytest.mark.parametrize("method", sorted(_KNOWN_SCOPE_WRITERS))
def test_actor_is_required_with_no_default(method):
    """A forgotten argument must be a TypeError, not a silent exemption.

    If `actor` defaulted to `SYSTEM_PRINCIPAL`, a new caller that omitted it would
    be exempted — "the permissive value is what you get when you don't specify",
    which is the defect class this repository has spent a week removing.
    """
    import db

    sig = inspect.signature(getattr(db.OAuthStore, method))
    param = sig.parameters["actor"]
    assert param.default is inspect.Parameter.empty, (
        f"OAuthStore.{method} gives `actor` a default; omitting it would silently "
        "exempt the caller"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"OAuthStore.{method}'s `actor` should be keyword-only so it cannot be "
        "supplied positionally by accident"
    )


# ── The disagreement, recorded rather than resolved ────────────────────

#: `SCOPE_DESCRIPTIONS`'s `(operator)` annotations and `OPERATOR_SCOPES` do not
#: agree, and neither side is obviously right. Recorded here so the divergence is
#: visible and bounded instead of silently inherited by whoever next derives one
#: from the other.
#:
#: `hosts:read` is annotated `(operator)` yet deliberately absent from the guard:
#: providers — who are not platform admins — need it, and a previous
#: reclassification of `hosts:read` broke provider onboarding. Adding it here
#: without checking that would break it again.
_ANNOTATED_BUT_NOT_GUARDED = {"hosts:read"}

#: Guarded but absent from `SCOPE_DESCRIPTIONS` entirely, so nothing can grant
#: them and the refusal is currently moot — the same "enforced but ungrantable"
#: shape as #27.
_GUARDED_BUT_UNDESCRIBED = {"hosts:fleet", "transparency:read", "transparency:write"}


def test_the_annotation_divergence_has_not_grown():
    from oauth_service import SCOPE_DESCRIPTIONS

    annotated = {s for s, text in SCOPE_DESCRIPTIONS.items() if "(operator)" in text}

    unexpected_gap = sorted((annotated - OPERATOR_SCOPES) - _ANNOTATED_BUT_NOT_GUARDED)
    assert not unexpected_gap, (
        f"{unexpected_gap} are annotated `(operator)` but not guarded. Either they "
        "confer platform authority and belong in OPERATOR_SCOPES, or the annotation "
        "is wrong. Do not resolve it by deriving one from the other."
    )

    undescribed = sorted(OPERATOR_SCOPES - set(SCOPE_DESCRIPTIONS) - _GUARDED_BUT_UNDESCRIBED)
    assert not undescribed, (
        f"{undescribed} are guarded but absent from SCOPE_DESCRIPTIONS, so no "
        "client can be granted them and no consent screen can describe them"
    )


def test_is_platform_admin_matches_the_deps_implementation():
    """Two spellings of one rule is what #16 is about; pin them together."""
    from routes._deps import _is_platform_admin

    for actor in (ADMIN, USER, {"role": "admin"}, {"is_admin": 1}, {}, None):
        assert is_platform_admin(actor) == _is_platform_admin(actor), actor


def test_the_system_allowlist_covers_what_the_system_paths_actually_use():
    """Derived-by-check rather than derived-by-import.

    `oauth_delegation` imports nothing, so it cannot read these constants at
    runtime without creating the cycle it exists to avoid. So the literal stays and
    this asserts it covers the union — a system path that starts using a new scope
    fails here instead of at the moment it runs in production.

    The first version of that allowlist was hand-written and already wrong: it
    omitted `api`, which a seeded machine client uses.
    """
    from oauth_registration import CONNECTOR_ALLOWED_SCOPES
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    actually_used = set(MCP_QUICK_CONNECT_SCOPES) | set(CONNECTOR_ALLOWED_SCOPES) | {
        # the seeded first-party clients
        "profile", "email", "offline_access",
        # the legacy blanket scope on seeded machine clients
        "api",
    }
    missing = sorted(actually_used - SYSTEM_ALLOWED_SCOPES)
    assert not missing, (
        f"the system paths use {missing}, which SYSTEM_ALLOWED_SCOPES does not "
        "permit — a system-managed client creation will be refused at runtime"
    )
