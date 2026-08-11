"""Exactly one vhost owns unmatched HTTPS traffic, and it is not the agent block.

## What the first version of this test got wrong

It asserted that `agent-xcelsior.conf` never *declares* `default_server`. That
is not the risk. nginx falls back to the **first server block in config order**
when nothing declares a default, and the production includes are globs —
`sites-enabled/*`, `conf.d/*.conf` — so config order is **filename order**.
`agent-xcelsior.conf` sorted ahead of every other vhost here and ahead of every
PixelEnhance hostname on that box. Symlinking it would have made an
`ssl_verify_client on` block the implicit default, which refuses *every*
unmatched hostname across both estates, and the old test stayed green the whole
way through.

**The fix is not a filename.** `unmatched-hosts.conf` declares `default_server`
explicitly, so ordering stops deciding anything — which is why no assertion here
depends on sort order. Winning the race by naming a file `00-` would have been
the same fragile mechanism wearing a different hat.

Sixth instance of one shape this phase: a mechanism that does not cover the case
it was written for. Same question catches all of them — *does this actually
cover the case I cited as its reason?*

## What it asserts now

Against the **assembled set** of vhosts rather than one file, so ordering cannot
decide anything:

* exactly one block declares `default_server`;
* it is not the agent block;
* it does not require client certificates.

That last one matters on its own: a default that demanded a client certificate
would reject unmatched traffic during the TLS handshake, which is the failure
being prevented rather than a stricter form of preventing it.
"""

from __future__ import annotations

import re
from pathlib import Path

NGINX = Path(__file__).resolve().parent.parent / "nginx"
AGENT = NGINX / "agent-xcelsior.conf"

_LISTEN = re.compile(r"^\s*listen\s+([^;]+);", re.M)
_VERIFY_ON = re.compile(r"^\s*ssl_verify_client\s+on\s*;", re.M)


def _vhosts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(NGINX.glob("*.conf"))}


def _declares_default(text: str) -> bool:
    return any("default_server" in directive for directive in _LISTEN.findall(text))


def test_the_scan_sees_the_vhosts():
    """Calibration — an empty set would make every assertion below vacuous."""
    vhosts = _vhosts()
    assert len(vhosts) >= 4, f"only {len(vhosts)} vhost files found"
    assert AGENT.name in vhosts


def test_exactly_one_vhost_owns_unmatched_traffic():
    """Ordering-independent. This is the assertion the first version lacked.

    With none declared, the default is whichever file sorts first — and the
    agent block sorts first. With one declared, filename order stops mattering
    for both estates and for every hostname added later.
    """
    owners = [name for name, text in _vhosts().items() if _declares_default(text)]
    assert owners, (
        "no vhost declares default_server, so unmatched HTTPS traffic goes to "
        "whichever file sorts first — today that is agent-xcelsior.conf, which "
        "requires client certificates and would refuse every unmatched hostname "
        "on the box"
    )
    assert len(owners) == 1, f"{owners} all claim default_server; nginx takes the first"


def test_the_agent_block_is_not_the_owner():
    owners = [name for name, text in _vhosts().items() if _declares_default(text)]
    assert AGENT.name not in owners, (
        "the agent vhost claims default_server; with ssl_verify_client on it "
        "would reject every unmatched hostname at the TLS handshake"
    )


def test_the_owner_does_not_require_client_certificates():
    """A default that demands a client cert *is* the failure, not a stricter fix."""
    for name, text in _vhosts().items():
        if _declares_default(text):
            assert not _VERIFY_ON.search(text), (
                f"{name} owns unmatched traffic and requires client certificates, "
                "so every unmatched hostname is refused during the handshake"
            )


def test_the_agent_block_still_requires_client_certificates():
    """The guards above are only interesting while this stays true."""
    text = AGENT.read_text(encoding="utf-8")
    assert _VERIFY_ON.search(text), (
        "the agent vhost no longer requires client certs, so it is not the mTLS "
        "terminator the cutover plan assumes"
    )
    assert "ssl_client_certificate" in text
