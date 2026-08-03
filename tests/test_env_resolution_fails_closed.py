"""A missing or misspelled `XCELSIOR_ENV` must not disable security.

Four independent security decisions were keyed on `os.environ.get(
"XCELSIOR_ENV", "dev")`, and every one of them treated *absence* as
development:

* `routes/_deps.AUTH_REQUIRED` — false in dev, and `_require_auth` then returns
  a synthetic principal with `is_admin: True` to an anonymous caller.
* `oauth_service` JWT secret — falls back to the literal
  `"xcelsior-dev-jwt-secret"`, which is in the source tree.
* `oauth_service` signing algorithm — permits symmetric HS256 instead of
  requiring an asymmetric key.
* `security.py` secrets key — falls back to a deterministic Fernet key derived
  from a constant, so every encrypted secret is recoverable from the repo.

So an unset variable does not degrade one control, it removes authentication,
lets anyone forge a token, and makes stored secrets readable.

The deployed path sets it twice over — `.env` pins `production` and
`docker-compose.yml` uses `${XCELSIOR_ENV:-production}` — so this was latent
rather than live. It becomes real the first time the app runs outside that path:
a systemd unit, a migration script, a one-off container, a new target.

**Unrecognised values matter as much as absence.** `"prodution"` and
`"Production "` with a trailing space both failed the
`in {"dev", "development", "test"}` test and were therefore treated as
production by `AUTH_REQUIRED` — but `security.py` asked the opposite question,
`env in ("production", "prod")`, so the same typo produced the insecure dev key.
Two checks disagreeing about what counts as production is how one value ends up
production for authentication and not production for encryption at once.

The rule enforced here: **only an explicitly recognised development value
relaxes anything. Everything else — unset, misspelled, unknown — is
production.**

Staging is the one case that is neither, and the split is deliberate. It must
not use a dev fallback, because it holds real data; it must also keep the
escape hatch `routes/agent.py` grants it and refuses to production. So
`is_relaxed_env()` and `is_production()` are not complements, and
`test_staging_is_neither_relaxed_nor_production` pins the gap.
"""

from __future__ import annotations

import os

import pytest

import env_config


@pytest.mark.parametrize(
    "raw",
    [
        None,          # unset
        "",            # set but empty
        "   ",         # whitespace
        "prodution",   # typo
        "Production",  # case
        "PRODUCTION",
        "prod",
        "unknown",
    ],
)
def test_unrecognised_values_are_treated_as_production(raw):
    """Absence and typos must both fail closed.

    Driven by argument, not by `os.environ`. Swapping the variable to exercise
    each branch leaked: modules that read it at call time and cache — the
    cache-key namespace among them — saw a foreign value mid-suite, and eleven
    unrelated tests failed in the full run while passing in isolation.
    """
    assert env_config.is_relaxed_env("" if raw is None else raw) is False, (
        f"XCELSIOR_ENV={raw!r} relaxed a security control"
    )


@pytest.mark.parametrize("raw", ["dev", "development", "test", "DEV", " test "])
def test_recognised_development_values_relax(raw):
    """The escape hatch must still work when asked for explicitly.

    Case and surrounding whitespace are normalised — a `.env` line with a
    trailing space should not silently become production and break local work.
    """
    assert env_config.is_relaxed_env(raw) is True


def test_auth_is_required_when_the_variable_is_absent():
    """The consequence that matters most: no anonymous admin.

    Asserted without reloading `routes._deps`. An earlier version of this test
    called `importlib.reload` on it, which swapped the module object for every
    other module already holding a reference — four unrelated tests failed in
    the full suite while passing in isolation. `AUTH_REQUIRED` is a module
    constant evaluated at import, so the honest way to test its *derivation* is
    to evaluate the same expression, and to check separately that the module
    still derives it that way.
    """
    assert (not env_config.is_relaxed_env("")) is True, (
        "an unset XCELSIOR_ENV would disable authentication; `_require_auth` "
        "would hand an anonymous caller a synthetic admin principal"
    )


def test_auth_required_is_still_derived_from_the_resolver():
    """Pair to the test above: the constant must come from `env_config`.

    Evaluating the expression proves the resolver is right; this proves
    `_deps` uses it, so the two together cover what a reload would have.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "routes" / "_deps.py"
    ).read_text(encoding="utf-8")
    assert "AUTH_REQUIRED = not env_config.is_relaxed_env()" in source, (
        "routes/_deps no longer derives AUTH_REQUIRED from env_config; a "
        "permissive default may have returned"
    )


def test_the_dev_jwt_secret_is_not_reachable_without_an_explicit_dev_env(monkeypatch):
    """A signing secret committed to the repo must never be a fallback."""
    monkeypatch.delenv("XCELSIOR_ENV", raising=False)
    monkeypatch.delenv("XCELSIOR_OAUTH_JWT_SECRET", raising=False)
    monkeypatch.delenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", raising=False)
    import oauth_service

    with pytest.raises(Exception) as exc:
        oauth_service._active_signing_material()
    assert "xcelsior-dev-jwt-secret" not in str(exc.value)


def test_the_dev_secrets_key_is_not_reachable_without_an_explicit_dev_env(monkeypatch):
    """Same for the Fernet key protecting stored secrets."""
    monkeypatch.delenv("XCELSIOR_ENV", raising=False)
    monkeypatch.setattr("security._SECRETS_KEY", "", raising=False)
    monkeypatch.setattr("security._fernet", None, raising=False)
    import security

    with pytest.raises(RuntimeError, match="XCELSIOR_SECRETS_KEY"):
        security._get_fernet()


SECURITY_MODULES = [
    "routes/_deps.py",
    "oauth_service.py",
    "security.py",
    "routes/auth.py",
    "routes/agent.py",
    "control_plane/startup_validation.py",
]

#: The shapes that produced every defect in this file's docstring.
_PERMISSIVE = (
    'environ.get("XCELSIOR_ENV", "dev")',
    "environ.get('XCELSIOR_ENV', 'dev')",
    'environ.get("XCELSIOR_ENV", "")',
    'environ.get("XCELSIOR_ENV") or ""',
)


def _permissive_lines(text: str) -> list[str]:
    return [
        f"{n}: {line.strip()[:110]}"
        for n, line in enumerate(text.splitlines(), 1)
        if "XCELSIOR_ENV" in line
        and not line.lstrip().startswith("#")
        and any(shape in line for shape in _PERMISSIVE)
    ]


def test_no_security_module_still_defaults_the_env_to_dev():
    """Catch the next one at the source rather than by its consequence.

    Security-relevant modules must resolve the environment through
    `env_config`, which fails closed, rather than re-deriving it with a
    permissive default.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = {
        rel: found
        for rel in SECURITY_MODULES
        if (root / rel).exists()
        and (found := _permissive_lines((root / rel).read_text(encoding="utf-8")))
    }
    assert not offenders, (
        "security-relevant modules resolve XCELSIOR_ENV with a permissive "
        f"default instead of env_config: {offenders}"
    )


@pytest.mark.parametrize("shape", _PERMISSIVE)
def test_the_guard_catches_a_planted_regression(shape):
    """Prove the guard's reach instead of trusting its silence.

    A scanner that matches nothing reports clean, and clean is exactly what a
    broken scanner looks like — the repo-wide vocabulary guard reported zero
    while four blog posts and a published docs page were full of what it was
    hunting, because it read the wrong file types.

    So each permissive shape is planted in a synthetic module and the guard is
    required to find it. If someone narrows `_PERMISSIVE`, this fails.
    """
    planted = f"env = os.{shape}.lower()\n"
    assert _permissive_lines(planted), f"guard does not detect {shape!r}"


def test_the_guard_ignores_comments_and_the_compliant_form():
    """It must not fire on prose about the defect, or on correct code."""
    assert not _permissive_lines('# os.environ.get("XCELSIOR_ENV", "dev") was the bug\n')
    assert not _permissive_lines("env = env_config.resolve_env()\n")


@pytest.mark.parametrize("raw", ["staging", "preprod"])
def test_staging_is_neither_relaxed_nor_production(raw):
    """Staging is a real deployment, and also not the production VPS.

    Collapsing it into either direction loses something. Treated as relaxed, it
    would fall back to the committed signing secret and the deterministic
    encryption key while holding real data. Treated as production, it would lose
    the audited escape hatch in `routes/agent.py`, whose test states the intent
    outright: *a production VPS must not silently allow unauth even if
    XCELSIOR_ALLOW_UNAUTH_AGENT is accidentally set; staging and other non-test
    non-prod envs honor the escape hatch.*

    So `is_relaxed_env()` and `is_production()` are not complements, and this
    pins the gap between them.
    """
    assert env_config.is_relaxed_env(raw) is False, "staging must not use dev fallbacks"
    assert env_config.is_production(raw) is False, "staging must keep its escape hatch"
    assert env_config.resolve_env(raw) == raw


def test_relaxed_and_production_are_not_complements():
    """State the invariant directly, so a future simplification cannot erase it."""
    assert env_config.RELAXED_ENVS.isdisjoint(env_config.PRODUCTION_ENVS)
    assert env_config.PREPROD_ENVS.isdisjoint(env_config.RELAXED_ENVS)
    assert env_config.PREPROD_ENVS.isdisjoint(env_config.PRODUCTION_ENVS)
    assert env_config.PREPROD_ENVS, (
        "the gap between relaxed and production is load-bearing; emptying it "
        "collapses staging into one side or the other"
    )
