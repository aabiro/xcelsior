"""The staging API must not put the worker protocol on the network.

These services run with `network_mode: host`, so gunicorn's production default
of `0.0.0.0` binds a *developer machine's* interfaces — LAN and tailnet — not a
container network. A staging stack then serves `/agent/*` and `/host` to every
peer that can reach the box.

That is a latent exposure independent of any ingress setting, and it is why
flipping staging's `XCELSIOR_AGENT_PUBLIC_INGRESS` to `allow` would have been
the wrong fix: it would have removed the only thing standing between those
peers and the worker protocol.

The bind is configurable with the production default unchanged, and the staging
runner opts into loopback. Both halves are asserted, because either alone is
undone by an edit to the other.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_bind_address_is_configurable_and_defaults_to_the_production_value():
    """Changing production's posture was never the goal — making staging opt out was."""
    conf = (ROOT / "gunicorn.conf.py").read_text(encoding="utf-8")
    assert 'os.getenv("XCELSIOR_API_BIND"' in conf, (
        "gunicorn no longer reads XCELSIOR_API_BIND; staging cannot opt out of "
        "binding every interface"
    )
    default = re.search(r'os\.getenv\(\s*"XCELSIOR_API_BIND"\s*,\s*"([^"]+)"', conf)
    assert default and default.group(1) == "0.0.0.0", (
        "the default bind changed. Production terminates TLS at nginx and "
        "proxies to this socket; narrowing the default silently is a deploy "
        "outage, and widening staging's is the exposure this file exists for."
    )
    assert 'bind = f"{_host}:{_port}"' in conf, (
        "the bind is no longer assembled from the configurable host"
    )


def test_the_staging_runner_pins_loopback():
    runner = (ROOT / "scripts" / "run_staging_compose.sh").read_text(encoding="utf-8")
    pinned = re.search(r'XCELSIOR_API_BIND="\$\{XCELSIOR_API_BIND:-([^}"]+)\}"', runner)
    assert pinned, (
        "scripts/run_staging_compose.sh no longer pins XCELSIOR_API_BIND; a "
        "staging stack would bind every interface again"
    )
    assert pinned.group(1) in ("127.0.0.1", "localhost"), (
        f"the staging runner binds {pinned.group(1)!r} rather than loopback"
    )


def test_compose_passes_the_bind_through_to_the_api_service():
    """A pinned variable the container never sees is a pin that does nothing.

    This is the half that was missing on the first attempt: the runner exported
    it, the container reported it, and gunicorn still listened on 0.0.0.0
    because the image had not been rebuilt. The passthrough is what makes the
    export reach the process.
    """
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "XCELSIOR_API_BIND: ${XCELSIOR_API_BIND:-0.0.0.0}" in compose, (
        "docker-compose.yml no longer passes XCELSIOR_API_BIND to the api "
        "service, so the staging runner's pin never reaches gunicorn"
    )
