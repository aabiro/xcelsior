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
    assert 'bind = [f"{_host}:{_port}" for _host in _hosts]' in conf, (
        "the bind is no longer assembled from the configurable host list"
    )


def test_the_bind_is_an_allowlist_rather_than_a_wildcard_switch():
    """Reaching staging from one peer must not mean accepting every peer.

    Before this, the only way to serve another host was `0.0.0.0` — the exact
    exposure this file exists for — so "let the Mac reach staging" and "put the
    worker protocol on the LAN" were the same edit. Splitting on commas makes
    the interfaces explicit: `127.0.0.1,100.64.0.6` binds loopback and the
    tailnet address and leaves the LAN interface alone.

    Loopback stays in the list deliberately: the compose healthcheck curls
    `localhost`, so a tailnet-only bind reports the container unhealthy while it
    is serving perfectly well.
    """
    conf = (ROOT / "gunicorn.conf.py").read_text(encoding="utf-8")
    assert '.split(",")' in conf, (
        "XCELSIOR_API_BIND no longer accepts a list, so the only way to reach "
        "staging from another host is the wildcard again"
    )

    namespace: dict = {}
    import os

    # Both pinned. `XCELSIOR_API_PORT` is set in the ambient environment here
    # (9501, the blue slot), and a test that reads it asserts whatever the shell
    # happened to export rather than the property.
    previous = {k: os.environ.get(k) for k in ("XCELSIOR_API_BIND", "XCELSIOR_API_PORT")}
    try:
        os.environ["XCELSIOR_API_BIND"] = "127.0.0.1,100.64.0.6"
        os.environ["XCELSIOR_API_PORT"] = "9500"
        exec(compile(conf, "gunicorn.conf.py", "exec"), namespace)
        assert namespace["bind"] == ["127.0.0.1:9500", "100.64.0.6:9500"], namespace["bind"]
        # The point of an allowlist: naming interfaces must not quietly include
        # the wildcard, or it is a wildcard with extra steps.
        assert not any(b.startswith("0.0.0.0") for b in namespace["bind"])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def test_staging_keeps_agent_ingress_denied():
    """The other half of the pairing, and the reason widening the bind is safe.

    An allowlist bind and a denied ingress are not two independent settings —
    they are the two halves of one property. `XCELSIOR_API_BIND=127.0.0.1,
    100.64.0.6` puts the staging API on the tailnet so a peer can reach it, and
    the only thing that then keeps `/agent/*` and `/host` from that peer is
    `deny`. Flipping this to `allow` while the bind is widened hands the worker
    protocol to every tailnet node, which is the exposure the whole file exists
    for — and it is the fix that looks obvious when an agent cannot register.

    Verified live on 2026-08-25 from the Mac (`100.64.0.3`), a genuine remote
    peer: `/healthz` answered 200, `/agent/register` and `/host` answered
    **410**, and the LAN address `192.168.1.127:9502` refused the connection
    outright because that interface is not in the allowlist.
    """
    env = (ROOT / ".env.staging").read_text(encoding="utf-8")
    match = re.search(r"^XCELSIOR_AGENT_PUBLIC_INGRESS=(\S+)", env, re.M)
    assert match, (
        ".env.staging no longer sets XCELSIOR_AGENT_PUBLIC_INGRESS, so it falls "
        "back to the api.py default of `allow` — and the worker protocol is "
        "served to every peer the bind allowlist admits"
    )
    assert match.group(1).strip().lower() == "deny", (
        f"staging sets agent ingress to {match.group(1)!r}. With a bind "
        "allowlist that includes a tailnet address, this is the difference "
        "between a reachable staging API and an open worker protocol."
    )
