"""The production configuration gate must actually gate.

`control_plane/startup_validation.py` collects findings — SQLite backend,
unauthenticated agent mode, no asymmetric signing key, missing audit key — and
`validate_startup()` raises `StartupValidationError` on any `error` severity so
the deploy fails rather than serving traffic misconfigured. `api.lifespan` calls
it before anything else.

Whether it raises was decided by:

    def is_production() -> bool:
        return (os.environ.get("XCELSIOR_ENV") or "").strip().lower() == "production"

An exact string match. `prod`, `staging`, a typo, or an unset variable all
returned False, and every error finding degraded to a log line nobody reads. The
gate designed to catch the other fail-open defaults was itself fail-open on the
same variable.

`_check_oauth_signing` compounded it with `os.environ.get("XCELSIOR_ENV",
"dev")`, the same permissive default found in four other modules.

Two decisions are pinned here:

* **Enforcement is the default.** Only an explicitly relaxed environment
  (`dev`/`test`/`local`) may boot with error findings. Staging enforces —
  it holds real data, and the P2/P6 acceptance journeys run against it, so
  discovering that it lacks real signing keys belongs at boot rather than
  mid-gate.
* **Missing secrets are boot failures, not lazy failures.** A missing Fernet
  key currently surfaces at first use, which can be long after traffic starts
  and far from the cause.
"""

from __future__ import annotations

import pytest

import env_config
from control_plane import startup_validation


@pytest.mark.parametrize(
    "raw",
    [None, "", "prod", "production", "prodution", "staging", "preprod", "unknown"],
)
def test_enforcement_is_on_everywhere_except_explicit_dev(raw):
    """Absence, typos, and staging must all still enforce.

    Argument-driven so the suite's environment is never mutated — see the note
    in `test_env_resolution_fails_closed`.
    """
    assert startup_validation.enforcement_enabled("" if raw is None else raw) is True, (
        f"XCELSIOR_ENV={raw!r} turned the production configuration gate into "
        "warnings; every error finding would be logged and ignored"
    )


@pytest.mark.parametrize("raw", ["dev", "development", "test", "local"])
def test_developer_machines_still_boot(raw):
    """The gate must not make local work impossible."""
    assert startup_validation.enforcement_enabled(raw) is False


def test_an_error_finding_actually_raises_when_enforced(monkeypatch):
    """The consequence, not just the predicate."""
    monkeypatch.setenv("XCELSIOR_ENV", "staging")
    monkeypatch.delenv("XCELSIOR_SKIP_STARTUP_VALIDATION", raising=False)
    monkeypatch.setattr(
        startup_validation,
        "collect_findings",
        lambda: [
            startup_validation.Finding(
                code="probe",
                severity="error",
                message="planted",
                remediation="none",
            )
        ],
    )
    with pytest.raises(startup_validation.StartupValidationError):
        startup_validation.validate_startup()


def test_a_missing_secrets_key_is_an_error_finding(monkeypatch):
    """Fail at boot, not at the first encrypt/decrypt.

    A deployment without `XCELSIOR_SECRETS_KEY` used to start cleanly and raise
    the first time something touched an encrypted column — long after traffic
    started, and far from the cause.
    """
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_SECRETS_KEY", raising=False)
    finding = startup_validation._check_secrets_key()
    assert finding is not None and finding.severity == "error"
    assert "XCELSIOR_SECRETS_KEY" in finding.remediation


def test_a_present_secrets_key_passes(monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_SECRETS_KEY", Fernet.generate_key().decode())
    assert startup_validation._check_secrets_key() is None


def test_a_malformed_secrets_key_is_caught_at_boot(monkeypatch):
    """A key that exists but cannot build a Fernet is as bad as none."""
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_SECRETS_KEY", "not-a-valid-fernet-key")
    finding = startup_validation._check_secrets_key()
    assert finding is not None and finding.severity == "error"


def test_startup_validation_does_not_rederive_the_environment(monkeypatch):
    """It must ask `env_config`, not re-implement the comparison.

    Two of the defects this file exists for were re-derivations: an exact
    `== "production"` here, and `get("XCELSIOR_ENV", "dev")` in
    `_check_oauth_signing`.
    """
    import pathlib

    source = (
        pathlib.Path(startup_validation.__file__)
    ).read_text(encoding="utf-8")
    offenders = [
        f"{n}: {line.strip()[:100]}"
        for n, line in enumerate(source.splitlines(), 1)
        if "XCELSIOR_ENV" in line
        and not line.lstrip().startswith("#")
        and "env_config" not in line
    ]
    assert not offenders, (
        f"startup_validation re-derives the environment: {offenders}"
    )


def test_env_config_and_startup_validation_agree():
    """One answer to 'are we relaxed', across both modules."""
    for raw in ("dev", "test", "staging", "production", "prodution", ""):
        assert startup_validation.enforcement_enabled(raw) is (
            not env_config.is_relaxed_env(raw)
        ), f"disagreement at XCELSIOR_ENV={raw!r}"
