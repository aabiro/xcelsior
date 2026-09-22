"""`/internal/route` is loopback-and-tailnet only, and "tailnet" is a range.

The guard was `client_host.startswith("100.")`. The comment beside it said
"Allow Tailscale range too", and the Tailscale range is `100.64.0.0/10` —
100.64.0.0 through 100.127.255.255. A string prefix matches `100.0.0.0/8`,
which is four times as large, and the extra three quarters are publicly
routable space allocated to real networks. A caller at 100.1.2.3 passed a check
written to admit only the tailnet.

It was wrong in the other direction too: an IPv4-mapped IPv6 peer
(`::ffff:100.64.0.1`) *is* inside the tailnet and failed the string test
outright, so a multi-homed box reached over IPv6 was refused.

This is the nginx `auth_request` target for per-instance HTTP routing. Its
docstring calls the restriction "loopback-only to prevent external
enumeration", so the range is the whole defence.

`demo_account.is_ip_whitelisted` already did this properly with `ipaddress`;
the fix follows it, and these are the cases a string prefix cannot express.
"""

from __future__ import annotations

import pytest

from routes.instances import _is_internal_caller

INSIDE = [
    "127.0.0.1",
    "::1",
    "localhost",
    "testclient",
    "100.64.0.1",        # first address of the tailnet
    "100.127.255.254",   # last usable address of the tailnet
    "100.100.100.100",
    "::ffff:100.64.0.1", # IPv4-mapped IPv6, inside the range
    "[::ffff:100.64.0.1]",
    "100.64.0.1%eth0",   # zone id
]

OUTSIDE = [
    "100.0.0.1",         # 100.0.0.0/8 but BELOW the CGNAT range — public
    "100.63.255.255",    # one address below the range
    "100.128.0.0",       # one address above the range
    "100.200.1.1",       # public
    "10.0.0.5",
    "192.168.1.5",
    "8.8.8.8",
    "",
    "not-an-ip",
    "1000.1.1.1",
]


@pytest.mark.parametrize("host", INSIDE)
def test_internal_callers_are_admitted(host) -> None:
    assert _is_internal_caller(host), f"{host} should be internal"


@pytest.mark.parametrize("host", OUTSIDE)
def test_everyone_else_is_refused(host) -> None:
    assert not _is_internal_caller(host), (
        f"{host} was treated as internal; the nginx auth_request target is "
        "reachable from outside"
    )


def test_the_public_neighbours_of_the_tailnet_are_refused() -> None:
    """The specific addresses the old prefix test let through.

    Stated on its own because it is the whole bug: these four are in
    `100.0.0.0/8` and not in `100.64.0.0/10`, and every one of them passed
    `startswith("100.")`.
    """
    for host in ("100.0.0.1", "100.63.255.255", "100.128.0.0", "100.255.255.255"):
        assert host.startswith("100."), "precondition: the old check would admit this"
        assert not _is_internal_caller(host), f"{host} is public and was admitted"
