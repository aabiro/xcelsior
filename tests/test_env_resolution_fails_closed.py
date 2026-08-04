"""An unset `XCELSIOR_ENV` must not relax anything.

Ten security decisions were independently keyed on
`os.environ.get("XCELSIOR_ENV", "dev")` or on an exact match against
`"production"`. Every one of them treated an *absent or misspelled* variable as
development:

| site | what an unset variable did |
|---|---|
| `routes/_deps.py` | `AUTH_REQUIRED = False`; `_require_auth` returned `is_admin: True` for anonymous callers |
| `security.py` ×2 | Fernet key derived from a constant in this source tree |
| `oauth_service.py` ×3 | signing secret `xcelsior-dev-jwt-secret`, symmetric signing permitted, HS256 tokens accepted |
| `routes/terminal.py` | SSH host-key pinning off — no MITM protection |
| `serverless/limits.py` | rate limiting on the development path |
| `routes/auth.py` ×2 | a failed Facebook signed-request signature merely logged |
| `privacy_deletion.py` | deletion-reference secret derived rather than required |
| `host_admission.py` | the development compatibility secret used |

They also disagreed with each other. `AUTH_REQUIRED` asked `not in {dev,
development, test}` while `security.py` asked `in ("production", "prod")`, so
`prodution` was production for authentication and *not* production for
encryption, simultaneously.

`env_config` is the single answer to "may we relax anything", and it answers no
unless explicitly told otherwise. The tests below drive the resolver through the
`raw` parameter rather than mutating `os.environ`: anything that reads the
variable at call time and caches — the cache-key namespace, for one — sees a
different value mid-suite otherwise, and unrelated tests fail only in a full run.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import env_config
from tests._source_tree import iter_source_files, read_source

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Everything that must NOT be treated as development. Unset and empty are the
#: ones that actually happened; the typos are the shape that made them possible.
NOT_RELAXED = [None, "", "   ", "\t", "production", "prod", "PROD", "staging",
               "preprod", "prodution", "producton", "Production ", "live", "wat"]

#: The only values that may relax a security control.
RELAXED = ["dev", "development", "test", "local", "DEV", " test "]


@pytest.mark.parametrize("raw", NOT_RELAXED)
def test_nothing_unrecognised_is_treated_as_development(raw):
    """The load-bearing rule."""
    value = "" if raw is None else raw
    assert env_config.is_relaxed_env(value) is False, (
        f"{raw!r} was treated as a development environment, so every insecure "
        "fallback keyed on this resolver would be selected"
    )


@pytest.mark.parametrize("raw", RELAXED)
def test_an_explicitly_named_development_environment_still_relaxes(raw):
    """The calibration control.

    Without it, a resolver that returned False for everything would satisfy the
    rule above and break every developer machine — which is how this gets
    reverted rather than fixed.
    """
    assert env_config.is_relaxed_env(raw) is True, (
        f"{raw!r} is a development environment and must still relax; a resolver "
        "that refuses everything is not fail-closed, it is broken"
    )


def test_an_unset_variable_resolves_to_production():
    assert env_config.resolve_env("") == "production"
    assert env_config.resolve_env(None) == env_config.resolve_env()


def test_staging_is_neither_relaxed_nor_production():
    """Three states, and the reason the two booleans are not complements.

    Staging holds real data, so it may not use a dev fallback; it is also not
    the production deployment. Any call site written as
    `if is_production(): strict() else: relaxed()` sends staging down the
    relaxed branch — which is why every call site is written as "relaxed, or
    not" instead.
    """
    assert env_config.is_relaxed_env("staging") is False
    assert env_config.is_production("staging") is False
    assert env_config.resolve_env("staging") == "staging"


def test_the_environment_sets_do_not_overlap():
    """Overlap would make the answer depend on evaluation order."""
    assert env_config.RELAXED_ENVS.isdisjoint(env_config.PRODUCTION_ENVS)
    assert env_config.PREPROD_ENVS.isdisjoint(env_config.RELAXED_ENVS)
    assert env_config.PREPROD_ENVS.isdisjoint(env_config.PRODUCTION_ENVS)
    assert env_config.PREPROD_ENVS, "removing staging would silently widen RELAXED_ENVS' complement"


def test_auth_required_is_derived_from_the_resolver():
    """`AUTH_REQUIRED` is the decision with the widest blast radius.

    Asserted structurally rather than by importing the value, because the value
    depends on the environment the suite happens to run in — and `conftest`
    pins it to False, so reading it here would prove nothing.
    """
    source = (ROOT / "routes" / "_deps.py").read_text(encoding="utf-8")
    assert "AUTH_REQUIRED = not env_config.is_relaxed_env()" in source, (
        "AUTH_REQUIRED is no longer derived from env_config; an unset "
        "XCELSIOR_ENV may once again disable authentication"
    )


#: Spellings of "this is a real deployment". Comparing against any one of them
#: misses the others, plus staging, plus every typo.
_PRODUCTION_SPELLINGS = {"production", "prod"}


def _fail_open_comparisons(tree: ast.AST) -> list[str]:
    """Environment comparisons where an unset or misspelled value relaxes.

    Two shapes qualify, and only two:

    1. **A `"dev"` default** — `os.environ.get("XCELSIOR_ENV", "dev")`. An unset
       variable becomes a development environment, so every fallback keyed on
       it is selected. This is how the committed JWT secret, the deterministic
       Fernet key, and `AUTH_REQUIRED = False` were all reachable at once.

    2. **Equality against a production spelling** — `== "production"`. `prod`,
       `staging`, and every typo take the other branch, which is the branch that
       skips the protection.

    Deliberately *not* flagged: equality against `"test"` with no `"dev"`
    default. An unset variable makes that comparison False, so the affordance
    stays off — it fails closed, and flagging it would push authors to rewrite
    correct code to satisfy a guard. The rule is about which way the default
    falls, not about comparing strings.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        src = ast.unparse(node)
        if "XCELSIOR_ENV" not in src:
            continue
        if "'dev'" in src or '"dev"' in src:
            found.append(src)
            continue
        if isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            compared = {
                c.value.strip().lower()
                for c in node.comparators
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            }
            if compared & _PRODUCTION_SPELLINGS:
                found.append(src)
    return found


def test_no_security_decision_compares_the_environment_string_directly():
    """The rule, applied to the tree rather than to the sites known today.

    Walks via `tests._source_tree.iter_source_files()` rather than its own
    `rglob`. This gate's first draft did roll its own, hit exactly the
    `UnicodeDecodeError` on a macOS AppleDouble sidecar that the shared iterator
    exists to prevent, and hand-patched a `._` skip — the fifth gate to learn
    that lesson separately, which is why the shared iterator was written.
    `tests/test_source_tree_is_shared.py` is what caught it.
    """
    offenders: dict[str, list[str]] = {}
    for path, rel in iter_source_files(exclude_prefixes=("migrations/", "wizard/")):
        if rel == "env_config.py":  # the resolver is allowed to read the variable
            continue
        try:
            tree = ast.parse(read_source(path))
        except SyntaxError:  # pragma: no cover
            continue
        hits = _fail_open_comparisons(tree)
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "environment comparisons that fail open — an unset value, a typo, "
        "`prod`, or staging all skip the protection. Ask "
        f"`env_config.is_relaxed_env()` instead:\n{offenders}"
    )


def test_the_guard_catches_a_planted_fail_open_comparison():
    """Prove the reach rather than trusting the silence.

    A scanner that matches nothing reports clean, and clean is exactly what a
    broken scanner looks like. Both historical shapes are planted.
    """
    equality = ast.parse('if os.environ.get("XCELSIOR_ENV") == "production": strict()')
    assert _fail_open_comparisons(equality), "the guard no longer sees `== \"production\"`"

    dev_default = ast.parse(
        'if os.environ.get("XCELSIOR_ENV", "dev").lower() in {"prod", "production"}: strict()'
    )
    assert _fail_open_comparisons(dev_default), 'the guard no longer sees the "dev" default'


def test_the_guard_ignores_the_compliant_form():
    """And does not flag the shape it is steering people towards."""
    compliant = ast.parse("if not env_config.is_relaxed_env(): strict()")
    assert not _fail_open_comparisons(compliant)


def test_the_guard_ignores_a_fail_closed_test_affordance():
    """`== "test"` with no dev default is correct and must not be flagged.

    An unset variable makes it False, so the affordance stays off. A guard that
    flagged this would be demanding a rewrite of code that already fails closed,
    and the usual response to that is to weaken the guard.
    """
    fail_closed = ast.parse('if os.environ.get("XCELSIOR_ENV") == "test": seed_fixture()')
    assert not _fail_open_comparisons(fail_closed)
