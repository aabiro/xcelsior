"""The compose file must provide the cache its own defaults demand.

`XCELSIOR_AUTH_CACHE_BACKEND` defaults to `redis` and `XCELSIOR_AUTH_REDIS_URL`
to `localhost:6379` — and nothing in `docker-compose.yml` provided one. A plain
`docker compose up` therefore produced an API whose every authentication call
answered `auth_cache_unavailable` (503), while `/healthz` returned
`{"ok": true}`.

Found by trying to use the product as a user: register worked, login did not,
and the health endpoint said the service was fine. A dependency that auth
cannot work without, absent from the file that is supposed to bring the system
up, is the kind of gap that only shows when someone actually runs it.

The API is host-networked and reaches the cache at `localhost`, so the service
must be host-networked too — a bridged service name is unreachable from there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"


def _services() -> dict:
    return (yaml.safe_load(COMPOSE.read_text(encoding="utf-8")) or {}).get("services", {})


def _raw() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_the_auth_cache_backend_still_defaults_to_redis() -> None:
    """If this default changes, the rest of this file is asserting the wrong thing."""
    assert re.search(r"XCELSIOR_AUTH_CACHE_BACKEND:\s*\$\{XCELSIOR_AUTH_CACHE_BACKEND:-redis\}", _raw()), (
        "the auth cache backend no longer defaults to redis — re-check whether a "
        "redis service is still the right requirement before deleting this file"
    )


def test_compose_defines_a_redis_service() -> None:
    assert "redis" in _services(), (
        "docker-compose.yml defines no redis service, but the API defaults to a "
        "redis auth cache at localhost:6379. `docker compose up` would bring up "
        "an API whose every auth call 503s while /healthz reports ok."
    )


def test_redis_is_host_networked_like_the_api() -> None:
    """A bridged redis is unreachable from a host-networked API."""
    svc = _services().get("redis", {})
    assert svc.get("network_mode") == "host", (
        f"redis network_mode is {svc.get('network_mode')!r}; the API is "
        "host-networked and reaches it at localhost, which a bridged service "
        "name cannot satisfy"
    )


def test_redis_is_not_in_an_opt_in_profile() -> None:
    """A profiled service does not start on a plain `docker compose up`."""
    assert not _services().get("redis", {}).get("profiles"), (
        "redis sits behind a compose profile, so the default bring-up still has "
        "no auth cache — which is the defect this service exists to fix"
    )


def test_redis_does_not_bind_every_interface_by_default() -> None:
    """With host networking, 0.0.0.0 publishes the auth cache to the LAN and tailnet."""
    cmd = " ".join(_services().get("redis", {}).get("command", "").split())
    assert "--bind ${XCELSIOR_REDIS_BIND:-127.0.0.1}" in cmd, (
        f"redis bind is not loopback-by-default: {cmd!r}"
    )


def test_the_api_waits_for_the_cache() -> None:
    dep = _services().get("api", {}).get("depends_on", {})
    assert "redis" in dep, "api does not depend on redis; auth 503s during startup"
    assert dep["redis"].get("condition") == "service_healthy", (
        f"api waits on redis with condition {dep['redis'].get('condition')!r}; "
        "started-but-not-ready still 503s"
    )


def test_redis_can_honour_a_configured_password() -> None:
    """A cache that ignores the password its clients send is not a working cache.

    The first version of this service shipped without `--requirepass`. A
    deployment whose `XCELSIOR_AUTH_REDIS_URL` is
    `redis://:<secret>@localhost:6379/0` then authenticates against a server
    with no password set, and redis answers *"Client sent AUTH, but no password
    is set"* — so the cache fails in precisely the deployments that configured
    it most carefully, while working in the ones that did not.
    """
    cmd = " ".join(_services().get("redis", {}).get("command", "").split())
    assert "${XCELSIOR_REDIS_REQUIREPASS:-}" in cmd, (
        f"redis takes no password from the environment: {cmd!r}. It will reject "
        "the AUTH that a password-carrying client sends."
    )


def test_the_healthcheck_can_authenticate() -> None:
    """Otherwise a password-protected redis never reports healthy.

    `api` waits on `service_healthy`, so an unauthenticated probe against a
    password-protected server holds the API down permanently — a worse failure
    than the missing cache this service was added to fix.
    """
    test = _services().get("redis", {}).get("healthcheck", {}).get("test", [])
    joined = " ".join(test) if isinstance(test, list) else str(test)
    assert "XCELSIOR_REDIS_CLI_AUTH" in joined, (
        f"the redis healthcheck cannot authenticate: {joined!r}. With a password "
        "set it would never pass, and depends_on would never release the API."
    )
