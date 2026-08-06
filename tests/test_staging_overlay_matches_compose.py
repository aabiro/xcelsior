"""The staging overlay must name services the base compose actually defines.

`docker-compose.staging.yml` exists to force `XCELSIOR_ENV=staging` on the
services that make enforcement decisions. It was written on
`feat/mcp-p0-scopes`, whose pull request was closed, and it named a `scheduler`
service. The base compose calls it `scheduler-worker`.

**Compose does not error on that.** An overlay naming a service the base does
not define *creates* it. So staging would have started a phantom `scheduler`
with no image, and the real `scheduler-worker` would have run **without**
`XCELSIOR_ENV=staging` — the single thing this file exists to set, silently not
set, on the service that decides whether enforcement is relaxed.

That is the failure mode worth a test: not a crash, but an overlay that appears
to apply and does not.
"""

from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "docker-compose.yml"
OVERLAY = ROOT / "docker-compose.staging.yml"


def _services(path: pathlib.Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("services", {})


def test_both_compose_files_parse():
    """Prove the reach — an unparseable file would make the rest vacuous."""
    assert _services(BASE), "no services parsed from docker-compose.yml"
    assert _services(OVERLAY), "no services parsed from docker-compose.staging.yml"


def test_the_overlay_only_names_services_the_base_defines():
    """The defect: a renamed service silently becomes a new, empty one."""
    base, overlay = set(_services(BASE)), set(_services(OVERLAY))
    phantom = sorted(overlay - base)
    assert not phantom, (
        f"the staging overlay names services docker-compose.yml does not define: "
        f"{phantom}. Compose creates these rather than failing, so the real "
        "service runs without the staging environment the overlay is for."
    )


def test_every_overlaid_service_is_pinned_to_staging():
    """The overlay's whole purpose, asserted rather than assumed.

    A service listed here but not given `XCELSIOR_ENV: staging` inherits the
    default — and `env_config.resolve_env` treats an unknown value as
    production, which is the fail-closed direction but the wrong environment.
    """
    unpinned = [
        name
        for name, spec in _services(OVERLAY).items()
        if (spec or {}).get("environment", {}).get("XCELSIOR_ENV") != "staging"
    ]
    assert not unpinned, (
        f"these staging services do not set XCELSIOR_ENV=staging: {unpinned}"
    )


def test_the_services_that_decide_enforcement_are_covered():
    """The API and the workers are where environment is read.

    Named explicitly so dropping one is visible in review rather than showing
    up as a staging box quietly behaving like production.
    """
    overlay = set(_services(OVERLAY))
    for required in ("api", "api-blue", "bg-worker", "scheduler-worker"):
        assert required in overlay, (
            f"{required} is not pinned to staging by the overlay; it would run "
            "with whatever XCELSIOR_ENV the base environment supplies"
        )
