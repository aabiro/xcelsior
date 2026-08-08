"""OAuth sign-in records what the provider just told us.

Issue #25. The OAuth callback set `email_verified` only when the account was
**new**. An account that had registered by password and never verified stayed at
`0` forever — refused on the password door with *"Please verify your email
address before logging in"* while signing in perfectly well through Google or
GitHub.

The severity is worth stating precisely rather than inflating: the provider
genuinely verified the address, so admitting the sign-in was correct. The defect
is that the platform **discarded a fact it had just been given**. The flag was
not a stale cache of something unknowable; anything reading it afterwards was
reasoning from a value the system knew to be wrong.

Both branches of the callback now set it, for the same reason and with the same
comment.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

REPO = pathlib.Path(__file__).resolve().parent.parent


def _callback_branches() -> tuple[str, str]:
    """(new-account branch, existing-account branch) of the OAuth callback.

    Reading the whole function and grepping for `email_verified` would pass
    while only one branch set it — which *is* the bug — so the two branches have
    to be separated before anything is asserted.

    Found by AST rather than by splitting on `else:`. The first draft did the
    latter and picked up an earlier, unrelated `else` in the provider-dispatch
    code, so the "new account" half contained none of the account creation and
    the reach check caught it. Text near the thing you want is not the thing you
    want.
    """
    import routes.auth as auth

    source = inspect.getsource(auth.api_auth_oauth_callback)
    tree = ast.parse(inspect.cleandoc(source))

    def creates_user(node: ast.AST) -> bool:
        return any(
            getattr(c.func, "attr", "") == "create_user"
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        )

    for node in ast.walk(tree):
        # The branch we want is the `if`/`else` whose *if* side creates the
        # account and whose *else* side handles one that already exists.
        if isinstance(node, ast.If) and node.orelse:
            if any(creates_user(stmt) for stmt in node.body):
                new_account = "\n".join(ast.unparse(s) for s in node.body)
                existing = "\n".join(ast.unparse(s) for s in node.orelse)
                return new_account, existing
    raise AssertionError(
        "no if/else in the OAuth callback creates a user on one side — the "
        "callback's shape changed and this file is asserting nothing"
    )


def test_the_callback_still_has_the_shape_this_asserts_against():
    """Prove the reach. If the split stops working, everything below is vacuous."""
    new_account, existing_account = _callback_branches()
    assert "create_user" in new_account, "the new-account branch is not where expected"
    assert "oauth_provider" in existing_account, "the existing-account branch is not where expected"


def test_a_new_account_is_recorded_as_verified():
    """Unchanged behaviour, asserted so the fix cannot be made by removing it."""
    new_account, _ = _callback_branches()
    assert "email_verified" in new_account


def test_an_existing_account_is_also_recorded_as_verified():
    """The defect. This branch updated name and provider and nothing else."""
    _, existing_account = _callback_branches()
    assert "email_verified" in existing_account, (
        "OAuth sign-in on an existing account still leaves email_verified "
        "untouched — the account stays refused on the password door forever "
        "while signing in fine through this one"
    )


def test_both_storage_paths_are_covered():
    """`_USE_PERSISTENT_AUTH` splits every write into two, and both must set it.

    Covering only the persistent path would leave the in-memory path — the one
    tests themselves run on — disagreeing with production about a security flag.
    """
    _, existing_account = _callback_branches()
    branches_setting_it = sum(
        1
        for node in ast.walk(ast.parse(existing_account))
        if isinstance(node, ast.Constant) and node.value == "email_verified"
    )
    assert branches_setting_it >= 2, (
        f"email_verified is set on only {branches_setting_it} of the two storage "
        "paths in the existing-account branch"
    )


def test_the_password_door_still_checks_it():
    """The other half of issue #25: the flag has to mean something.

    If this check were ever removed, setting the flag correctly would stop
    mattering and the fix above would be decorative.
    """
    import routes.auth as auth

    source = inspect.getsource(auth)
    assert 'if not user.get("email_verified"):' in source, (
        "nothing refuses an unverified account any more, so recording the "
        "verification no longer has a consumer"
    )
