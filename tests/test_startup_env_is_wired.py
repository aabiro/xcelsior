"""Every variable production startup validation reads must reach the container.

`docker-compose.yml` declares **no `env_file:`** on any service. A variable
reaches a container only by being named in the `x-api-environment` anchor, so a
value in `.env` that nothing maps is set on the host and unset in the process
that reads it — with no error anywhere, because `.env` is still consumed for
`${...}` interpolation and the file looks correctly configured.

That is not hypothetical. On 2026-08-04 the production deploy applied its
migrations and then failed to boot:

    StartupValidationError: production startup validation failed —
      compat_session_secret_missing: XCELSIOR_COMPAT_SESSION_SECRET is unset …
      audit_signing_key_default: Neither XCELSIOR_AUDIT_SIGNING_KEYS nor
        XCELSIOR_AUDIT_SIGNING_KEY is set …

Both were set in `.env`, correctly, with high-entropy values. Neither was named
in the anchor. The API had been down for hours and the secrets were sitting in
the file the whole time.

The rule this gate enforces is total on purpose: if
`control_plane/startup_validation.py` reads a variable, the container must
receive it. No exemption for variables whose default happens to be safe —
`XCELSIOR_VOLUME_PRIVILEGE` defaults to `host_ssh` and passes, but an operator
setting it to anything else would have been ignored the same way. An exemption
list here would be a list of variables allowed to lie about being configured.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
VALIDATION = PROJECT_ROOT / "control_plane" / "startup_validation.py"

#: Services whose environment comes from the shared `*api-env` anchor. Each one
#: runs code that calls `validate_startup` or the modules it validates, so a
#: variable missing from any of them is the same defect in a different process.
ANCHOR_SERVICES = ("api", "api-blue", "bg-worker")

_ENV_READ = re.compile(r'os\.environ\.get\(\s*"(XCELSIOR_[A-Z0-9_]+)"')


def _vars_read_by_validation() -> set[str]:
    return set(_ENV_READ.findall(VALIDATION.read_text()))


def _service_env(service: str) -> set[str]:
    compose = yaml.safe_load(COMPOSE.read_text())
    return set(compose["services"][service]["environment"])


def _unwired(read: set[str], mapped: set[str]) -> list[str]:
    return sorted(read - mapped)


def test_startup_validation_reads_nothing_the_container_cannot_see():
    read = _vars_read_by_validation()
    assert read, "found no XCELSIOR_* reads — the regex stopped matching"

    for service in ANCHOR_SERVICES:
        unwired = _unwired(read, _service_env(service))
        assert not unwired, (
            f"{service} does not receive {unwired}, which "
            "control_plane/startup_validation.py reads. There is no env_file: on "
            "this service, so setting these in .env has no effect on the "
            "container — the value is present on the host and absent in the "
            f"process that checks it. Add them to the x-api-environment anchor."
        )


def test_the_check_fails_on_the_configuration_that_caused_the_outage():
    """The failing arm: the anchor as it stood before 2026-08-04.

    Without this, the test above passes for as long as nobody breaks it and has
    never been observed to fail — which is the property that let the original
    gap ship.
    """
    read = {"XCELSIOR_COMPAT_SESSION_SECRET", "XCELSIOR_AUDIT_SIGNING_KEYS", "XCELSIOR_ENV"}
    mapped_before = {"XCELSIOR_ENV"}
    unwired = _unwired(read, mapped_before)
    assert unwired == [
        "XCELSIOR_AUDIT_SIGNING_KEYS",
        "XCELSIOR_COMPAT_SESSION_SECRET",
    ], unwired


def test_no_service_reintroduces_env_file_as_a_substitute():
    """`env_file:` would make this gate vacuous rather than satisfied.

    Adding `env_file: .env` would wire everything at once and pass the check
    above — while also handing every container the full production secret set,
    including services that have no business holding it. If that trade is ever
    made deliberately, it should be a deliberate edit here too.
    """
    compose = yaml.safe_load(COMPOSE.read_text())
    with_env_file = sorted(
        name for name, svc in compose["services"].items() if isinstance(svc, dict) and "env_file" in svc
    )
    assert not with_env_file, (
        f"services now use env_file: {with_env_file} — that grants the whole "
        "secret set to every listed service. Map variables explicitly, or "
        "update this test with the reasoning if the trade is intended."
    )
