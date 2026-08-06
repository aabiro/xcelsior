"""The platform's own host key is not a tenant surface.

`POST /ssh/keygen` generates the keypair *the platform* presents to provider
hosts — the one whose public half `/ssh/pubkey` publishes for `authorized_keys`.
It is not a customer's key and there is no journey in which a customer touches
it.

It sat behind `_require_auth` between two `_require_admin` neighbours
(`/alerts/config` above, `/token/generate` below), so any signed-in user could
call it and read back `key_path` — a filesystem path on the server. That is
server topology handed to whoever asks, from a route with no tenant purpose.

**Deliberately not overstated.** The first reading of this was "any user can
rotate the fleet's access key", which would be severe and is wrong:
`generate_ssh_keypair` returns early when the key already exists, so the route
cannot replace a live key. What it leaked was a path, and what it lacked was a
guard consistent with everything around it. Both worth fixing; neither worth a
scare.

`GET /ssh/pubkey` is left unauthenticated on purpose. A public key is public —
its whole function is to be copied into `authorized_keys` — and requiring auth
to read one would be ceremony, not security.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")


def _source(name: str) -> str:
    import routes.health as health

    return inspect.getsource(getattr(health, name))


def test_generating_the_platform_key_requires_admin():
    """The fix."""
    source = _source("api_generate_ssh_key")
    assert "_require_admin(request)" in source, (
        "POST /ssh/keygen no longer requires admin; any signed-in user can call "
        "a platform infrastructure route and read back a server path"
    )
    assert "_require_auth(request)" not in source, (
        "the route still admits any authenticated user"
    )


def test_it_matches_the_guard_its_neighbours_use():
    """Consistency is the actual argument here.

    An infrastructure route between `/alerts/config` and `/token/generate`,
    both admin-only, is the odd one out — and being the odd one out is how it
    escaped notice.
    """
    for neighbour in ("api_configure_alerts", "api_generate_token"):
        try:
            source = _source(neighbour)
        except AttributeError:  # pragma: no cover - route renamed
            continue
        assert "_require_admin(request)" in source, (
            f"{neighbour} no longer requires admin; the comparison this test "
            "relies on has moved"
        )


def test_the_public_key_read_stays_open():
    """The inverse, so nobody 'fixes' the wrong half.

    A public key exists to be copied into `authorized_keys`. Putting auth in
    front of reading one adds a step and protects nothing.
    """
    source = _source("api_get_pubkey")
    assert "_require_admin" not in source and "_require_auth" not in source, (
        "reading the platform's *public* key now requires auth; that is "
        "ceremony, and it breaks host setup for whoever has to paste it"
    )


def test_the_route_cannot_replace_a_live_key():
    """Pins the reason this is a disclosure fix and not an incident.

    If `generate_ssh_keypair` ever starts overwriting, this route becomes able
    to cut the platform off from every host that trusts the old key, and it
    should be re-reviewed rather than left on the strength of today's reading.
    """
    import scheduler

    source = inspect.getsource(scheduler.generate_ssh_keypair)
    assert "if os.path.exists(path):" in source, (
        "generate_ssh_keypair no longer short-circuits on an existing key; "
        "POST /ssh/keygen may now be able to rotate the fleet's access key"
    )
