"""The agent vhost must never become the box's default server.

`149.28.121.61` answers for the whole PixelEnhance estate as well as xcelsior's,
and **no server block on it declares `default_server`**. nginx therefore falls
back to the *first* matching block in config order for any unmatched hostname —
which is why `agent.xcelsior.ca` currently gets served the `docs.xcelsior.ca`
certificate.

That makes include order load-bearing and invisible. A vhost that claimed
`default_server` would capture every unmatched hostname on a shared box, and an
mTLS vhost doing so would refuse them all: `ssl_verify_client on` rejects any
client without a certificate, so the blast radius is "every site here stops
answering", not "one site misroutes".

Static, so it runs without a server. The `nginx -T` diff in
`docs/.ingress-prep/` is what covers the ordering half, which no static check
can see.
"""

from __future__ import annotations

import re
from pathlib import Path

VHOST = Path(__file__).resolve().parent.parent / "nginx" / "agent-xcelsior.conf"


def test_the_vhost_exists_where_the_runbook_expects_it():
    assert VHOST.exists(), f"{VHOST} is missing; the prepared cutover references it"


def test_it_never_claims_default_server():
    listens = re.findall(r"^\s*listen\s+([^;]+);", VHOST.read_text(), re.M)
    assert listens, "no listen directive found — the regex or the file changed"
    for directive in listens:
        assert "default_server" not in directive, (
            f"`listen {directive.strip()}` claims default_server. On a box with no "
            "explicit default, this vhost would capture every unmatched hostname — "
            "and with ssl_verify_client on it would refuse them all, taking the "
            "PixelEnhance estate down with it."
        )


def test_it_still_requires_client_certificates():
    """The guard above is only interesting while this is true."""
    text = VHOST.read_text()
    assert re.search(r"^\s*ssl_verify_client\s+on\s*;", text, re.M), (
        "the vhost no longer requires client certs, so it is no longer the "
        "mTLS terminator the cutover plan assumes"
    )
    assert "ssl_client_certificate" in text
