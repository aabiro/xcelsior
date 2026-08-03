"""Every scope the API enforces must be a scope a client can actually hold.

`_require_scope(user, "ssh:write")` refuses a machine credential whose granted
set lacks `ssh:write`. That is correct — but it is only *reachable* if some
client can be issued `ssh:write` in the first place. If the string appears at an
enforcement site and nowhere in the grant path, the endpoint is not protected;
it is **sealed**. No token can ever open it, and no amount of granting fixes it,
because the scope is not offered by the authorization server.

That is the state `ssh:read` and `ssh:write` were in: enforced in `routes/ssh.py`
and absent from `SCOPE_DESCRIPTIONS`, `CONNECTOR_ALLOWED_SCOPES`,
`MCP_QUICK_CONNECT_SCOPES`, and the `McpScope` union. Every agent credential was
permanently locked out of SSH key registration — the exact endpoint the
`register_ssh_key` tool depends on.

The failure is silent from both directions. Server-side it looks like sound
authorization: a scope is named and checked. Client-side it looks like a
permissions problem the user could fix by re-consenting, except the scope never
appears on the consent screen. Nothing logs "this scope cannot be granted".

So this guard closes the loop the other scope tests leave open. They assert that
holding a scope grants access and lacking it does not; this asserts that the
vocabulary is *coherent* — that enforcement, consent, registration, and the
TypeScript union describe one set of scopes rather than four overlapping ones.

Related: `tests/test_scope_enforcement.py` (enforcement is real, no wildcard).
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# OIDC identity scopes. Enforced nowhere as API authority, granted everywhere.
IDENTITY_SCOPES = frozenset({"profile", "email", "offline_access"})

# `_require_scope(user, "a:b")` / `_require_scope(user, "a:b", "c:d")`.
# Only literal arguments are matched; a computed scope would not be checkable
# here, and `test_no_computed_scope_arguments` below forbids that instead.
_ENFORCE_CALL = re.compile(r"_require_scope\s*\(\s*[^,)]+((?:\s*,\s*\"[a-z_]+:[a-z_]+\")+)\s*\)")
_SCOPE_LITERAL = re.compile(r"\"([a-z_]+:[a-z_]+)\"")

# Any `_require_scope` call whose scope arguments are not all string literals.
_ENFORCE_ANY = re.compile(r"_require_scope\s*\(")


def _py_sources() -> list[pathlib.Path]:
    skip = {"venv", ".venv", "node_modules", "__pycache__", "migrations", "tests"}
    return [
        p
        for p in ROOT.rglob("*.py")
        if not set(p.relative_to(ROOT).parts) & skip
    ]


def enforced_scopes() -> dict[str, list[str]]:
    """Scope -> the files that enforce it."""
    found: dict[str, list[str]] = {}
    for path in _py_sources():
        body = path.read_text(encoding="utf-8", errors="ignore")
        for match in _ENFORCE_CALL.finditer(body):
            for scope in _SCOPE_LITERAL.findall(match.group(1)):
                found.setdefault(scope, []).append(path.relative_to(ROOT).as_posix())
    return found


def _load(name: str):
    import importlib

    return importlib.import_module(name)


def grantable_scopes() -> set[str]:
    """Scopes the authorization server will actually put in a token.

    `SCOPE_DESCRIPTIONS` is the authority. `assert_scopes_delegable` refuses any
    scope absent from it at client registration, and `_normalize_scopes` then
    confines a token to what its client was registered with — so a described
    scope is reachable by some client, and an undescribed one by none.

    The narrower allowlists are *policy on top of* this vocabulary, not the
    vocabulary itself: `CONNECTOR_ALLOWED_SCOPES` limits self-registered
    connectors, and `OPERATOR_SCOPES` limits who may delegate platform
    authority. A scope missing from those is restricted; a scope missing from
    here cannot be granted at all.
    """
    return set(_load("oauth_service").SCOPE_DESCRIPTIONS)


def test_every_enforced_scope_can_be_granted():
    """The load-bearing assertion: no sealed endpoints."""
    enforced = enforced_scopes()
    grantable = grantable_scopes()
    sealed = {s: files for s, files in enforced.items() if s not in grantable}
    assert not sealed, (
        "These scopes are enforced but cannot be granted to any client, so no "
        "machine credential can ever satisfy them — the endpoints are sealed, "
        "not protected. Add each to SCOPE_DESCRIPTIONS (oauth_service.py) and "
        "to CONNECTOR_ALLOWED_SCOPES (oauth_registration.py) or "
        "MCP_QUICK_CONNECT_SCOPES:\n"
        + "\n".join(
            f"  {scope} — enforced in {', '.join(sorted(set(files)))}"
            for scope, files in sorted(sealed.items())
        )
    )


def test_every_grantable_scope_is_described():
    """A consent screen showing a raw scope string is not informed consent."""
    oauth_service = _load("oauth_service")
    oauth_registration = _load("oauth_registration")
    described = set(oauth_service.SCOPE_DESCRIPTIONS)
    offered = set(oauth_registration.CONNECTOR_ALLOWED_SCOPES) | set(
        oauth_service.MCP_QUICK_CONNECT_SCOPES
    )
    undescribed = sorted(offered - described)
    assert not undescribed, (
        "These scopes can be requested but have no plain-language description, "
        f"so the consent screen would show the raw string: {undescribed}"
    )


def test_connector_defaults_are_a_subset_of_what_connectors_may_hold():
    """Defaults must be grantable, or a silent client gets an unissuable token."""
    oauth_registration = _load("oauth_registration")
    allowed = set(oauth_registration.CONNECTOR_ALLOWED_SCOPES)
    defaults = set(oauth_registration.CONNECTOR_DEFAULT_SCOPES)
    assert defaults <= allowed, (
        f"default scopes not in the allowlist: {sorted(defaults - allowed)}"
    )


def test_connector_defaults_carry_no_write_authority():
    """A connector that asks for nothing must not arrive able to spend money.

    Pins the documented intent of CONNECTOR_DEFAULT_SCOPES ("read-biased on
    purpose") so a later scope addition cannot quietly widen the silent default.
    """
    oauth_registration = _load("oauth_registration")
    defaults = set(oauth_registration.CONNECTOR_DEFAULT_SCOPES) - IDENTITY_SCOPES
    writes = sorted(
        s for s in defaults if not s.endswith(":read")
    )
    assert not writes, f"default connector scopes grant non-read authority: {writes}"


def test_typescript_and_python_agree_on_the_scope_vocabulary():
    """`McpScope` is the tool layer's copy of the same list.

    Two layers enforcing different vocabularies is how `api` survived in one
    after being removed from the other.
    """
    scopes_ts = (ROOT / "mcp" / "src" / "auth" / "scopes.ts").read_text(encoding="utf-8")
    union = scopes_ts.split("export type McpScope =", 1)[1].split(";", 1)[0]
    ts_scopes = set(re.findall(r"\"([a-z_]+:[a-z_]+)\"", union))

    oauth_service = _load("oauth_service")
    py_scopes = {s for s in oauth_service.SCOPE_DESCRIPTIONS if ":" in s}

    missing_in_ts = sorted(py_scopes - ts_scopes)
    assert not missing_in_ts, (
        "Scopes the API describes but `McpScope` does not list, so the tool "
        f"layer cannot express a requirement for them: {missing_in_ts}"
    )
    unknown_in_ts = sorted(ts_scopes - py_scopes)
    assert not unknown_in_ts, (
        "`McpScope` lists scopes the authorization server will never issue; a "
        f"tool requiring one is unreachable: {unknown_in_ts}"
    )


def test_quick_connect_screen_matches_the_token_it_mints():
    """`McpAgentSetup.tsx` shows the user what their Quick Connect token can do.

    The two lists are maintained by hand in different languages, and the comment
    asking for them to be kept in sync has no enforcement behind it. A scope on
    the screen but not in the issued token is a promise the token cannot keep;
    the reverse is authority the user was never shown.
    """
    tsx = (
        ROOT / "frontend" / "src" / "components" / "settings" / "McpAgentSetup.tsx"
    ).read_text(encoding="utf-8")
    block = tsx.split("const MCP_SCOPES = [", 1)[1].split("]", 1)[0]
    shown = set(re.findall(r"\"([a-z_]+:[a-z_]+)\"", block))
    issued = set(_load("oauth_service").MCP_QUICK_CONNECT_SCOPES)
    assert shown == issued, (
        "Quick Connect screen and issued token disagree: "
        f"shown-only={sorted(shown - issued)}, issued-only={sorted(issued - shown)}"
    )


def test_no_wildcard_scope_in_the_vocabulary():
    """`api` short-circuited every check. It must not come back by any name."""
    oauth_service = _load("oauth_service")
    oauth_registration = _load("oauth_registration")
    banned = {"api", "*", "all", "admin", "full"}
    for name, values in (
        ("SCOPE_DESCRIPTIONS", set(oauth_service.SCOPE_DESCRIPTIONS)),
        ("CONNECTOR_ALLOWED_SCOPES", set(oauth_registration.CONNECTOR_ALLOWED_SCOPES)),
        ("CONNECTOR_DEFAULT_SCOPES", set(oauth_registration.CONNECTOR_DEFAULT_SCOPES)),
        ("MCP_QUICK_CONNECT_SCOPES", set(oauth_service.MCP_QUICK_CONNECT_SCOPES)),
    ):
        offending = sorted(values & banned)
        assert not offending, f"{name} carries a wildcard scope: {offending}"


def test_no_computed_scope_arguments():
    """Scope arguments must be literals, or this guard cannot see them.

    A `_require_scope(user, scope_var)` would pass every check in this file
    while being invisible to it. Rather than let the guard quietly narrow, the
    pattern is forbidden outright.
    """
    offenders: dict[str, list[str]] = {}
    for path in _py_sources():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "routes/_deps.py":  # the definition itself
            continue
        body = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(body.splitlines(), 1):
            if not _ENFORCE_ANY.search(line):
                continue
            args = line.split("_require_scope", 1)[1]
            # Every scope-shaped argument on the line must be a quoted literal.
            if ":" in args and not _SCOPE_LITERAL.search(args):
                offenders.setdefault(rel, []).append(f"{line_no}: {line.strip()[:100]}")
    assert not offenders, (
        "`_require_scope` called with a non-literal scope; this guard cannot "
        f"verify it is grantable: {offenders}"
    )


@pytest.mark.parametrize(
    "scope",
    [
        "instances:connect",
        "ssh:read",
        "ssh:write",
    ],
)
def test_access_scopes_exist(scope):
    """The scopes P0 introduces for the connect-from-the-terminal workflow.

    Named individually so removing one is visible in review rather than showing
    up as a count change.
    """
    assert scope in grantable_scopes(), f"{scope} is not grantable"
