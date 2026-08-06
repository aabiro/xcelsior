"""A scope the code enforces must be a scope a consent screen can explain.

`_require_scope` guards 41 distinct scopes across `routes/`. Fifteen had an
entry in `SCOPE_DESCRIPTIONS`. The other twenty-six rendered on the
authorization page as their own identifier, because the renderer falls back to
the raw string:

    def describe_scope(scope: str) -> str:
        return SCOPE_DESCRIPTIONS.get(scope, scope)

So a user was asked to approve `volumes:write` and `mfa:write` as literal text.
`0891e4c` fixed exactly this for `ssh:write`, `ssh:read` and `instances:connect`
— the three Quick Connect carries — and the same defect stayed in place for
everything else, because nothing compared the two lists.

That is what this test is: the comparison. It is not a style rule. An
undescribed scope is one of two things, and both are bad:

* **A real capability the user cannot evaluate.** `mfa:write` removes the second
  factors protecting their account. Shown as `mfa:write`, the consent screen is
  asking for informed agreement while withholding what is being agreed to.
* **A phantom.** A typo enforced in one route and granted nowhere, which fails
  closed and silently — the hardest kind of bug to find from the outside.

Discovered by parsing `_require_scope(...)` calls rather than from a list, so a
scope introduced tomorrow is covered without anyone remembering to add it here.
"""

from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTES = ROOT / "routes"


def enforced_scopes() -> dict[str, set[str]]:
    """Every literal scope passed to `_require_scope`, and where."""
    found: dict[str, set[str]] = {}
    for path in sorted(ROUTES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name != "_require_scope":
                continue
            for arg in node.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.setdefault(arg.value, set()).add(path.stem)
    return found


def test_the_scan_finds_the_enforcement_it_claims_to_read():
    """Prove the reach.

    If `_require_scope` were renamed or the call shape changed, this would find
    nothing and every assertion below would pass while comparing empty sets —
    the guard reporting clean because its subject moved.
    """
    found = enforced_scopes()
    assert len(found) > 25, (
        f"only {len(found)} enforced scopes found by parsing routes/; the scan "
        "has lost sight of _require_scope and the checks below are vacuous"
    )
    assert "billing:write" in found, "a known-enforced scope was not found"


def test_every_enforced_scope_has_a_consent_description():
    """The gate.

    A scope that reaches a consent screen as its own identifier is asking for
    agreement without saying to what.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    found = enforced_scopes()
    undescribed = sorted(set(found) - set(SCOPE_DESCRIPTIONS))
    assert not undescribed, (
        "these scopes are enforced but have no consent description, so the "
        "authorization screen shows the user a raw identifier and asks them to "
        "approve it: "
        + ", ".join(f"{s} ({', '.join(sorted(found[s]))})" for s in undescribed)
    )


def test_every_enforced_scope_is_grantable():
    """The other half: enforced, described, and impossible to hold.

    `assert_delegable` refuses scopes outside `known_scopes()`. A scope a route
    enforces but no client may be granted fails closed for everyone, which looks
    like a permissions bug rather than a vocabulary one.
    """
    from oauth_delegation import known_scopes

    found = enforced_scopes()
    ungrantable = sorted(set(found) - known_scopes())
    assert not ungrantable, (
        "these scopes are enforced but cannot be granted to any client, so the "
        f"routes behind them are unreachable: {ungrantable}"
    )


def test_a_description_says_more_than_the_scope_name():
    """A description that restates the identifier is not one.

    "volumes:write" -> "Volumes write" would satisfy the gate above while
    telling the user nothing. The check is deliberately loose — it catches the
    degenerate case without trying to grade prose.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    lazy = []
    for scope, text in SCOPE_DESCRIPTIONS.items():
        words = text.strip().split()
        if len(words) < 4:
            lazy.append(f"{scope}: {text!r}")
    assert not lazy, (
        "these descriptions are too short to explain anything to the person "
        f"approving them: {lazy}"
    )
