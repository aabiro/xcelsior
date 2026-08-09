"""Gate P2 clause 2: connection material is short-lived and single-use.

The truth table recorded this as PARTIAL with an unusual reason — *"the
implementation is genuinely correct… no test asserts the replay refusal. The
property is true and unguarded, so a regression would be silent."* This is the
guard.

A WebSocket ticket is the credential that turns an authenticated HTTP request
into a terminal session on someone's instance. If a replayed ticket were
accepted, anyone who observed one — a proxy log, a shared screen, a browser
history — could open a shell. Single-use is what makes the exposure window the
length of one connection instead of the length of the TTL.

## Both paths, deliberately

`_consume_ws_ticket` has two implementations: a shared-state path used when
`_USE_SHARED_RUNTIME_LIMITS` is on (so several API replicas agree about which
tickets are spent) and an in-process dict otherwise. **A property that holds in
one and not the other is exactly the kind of thing that hides**, because the
suite runs one path and production runs the other. Every test here is
parametrized across both.

## The four pins, not just the pop

Single-use is the headline, but the implementation also pins purpose, target and
client IP, and each of those is a separate way a stolen ticket could still be
useful. They are asserted individually so a regression names itself rather than
arriving as "some ticket test failed".
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")


class _FakeWebSocket:
    """Enough of a WebSocket for `_get_ws_client_ip`."""

    def __init__(self, ip: str = "203.0.113.7"):
        self.client = type("C", (), {"host": ip})()
        self.headers = {}
        self.query_params = {}


@pytest.fixture(params=["memory", "shared"])
def path(request, monkeypatch):
    """Run every assertion against both consume implementations."""
    from routes import _deps

    if request.param == "memory":
        monkeypatch.setattr(_deps, "_USE_SHARED_RUNTIME_LIMITS", False)
        _deps._WS_TICKETS.clear()
    else:
        # The shared path is only meaningful if it is actually reachable; if the
        # backing store is unavailable, skip rather than silently exercising the
        # in-memory path twice and reporting two passes.
        monkeypatch.setattr(_deps, "_USE_SHARED_RUNTIME_LIMITS", True)
        ok, _ = _deps._shared_state_update(
            _deps._WS_TICKET_STATE_NAMESPACE, lambda: {"tickets": {}}, lambda s: (s, None)
        )
        if not ok:
            pytest.skip("shared runtime state unavailable; not double-counting the memory path")
    return request.param


def _issue(user=None, **kw):
    from routes._deps import _issue_ws_ticket

    return _issue_ws_ticket(
        user or {"email": "demo@xcelsior.ca", "user_id": "u1"},
        purpose="terminal",
        target="job-1",
        client_ip="203.0.113.7",
        **kw,
    )


def _consume(ticket: str, *, purpose="terminal", target="job-1", ip="203.0.113.7"):
    from routes._deps import _consume_ws_ticket

    return _consume_ws_ticket(ticket, _FakeWebSocket(ip), purpose=purpose, target=target)


def test_a_fresh_ticket_is_accepted(path):
    """Calibration. If nothing is ever accepted, every refusal below is vacuous."""
    issued = _issue()
    user = _consume(issued["ticket"])
    assert user is not None, f"[{path}] a freshly issued ticket was refused"
    assert user.get("email") == "demo@xcelsior.ca"


def test_a_replayed_ticket_is_refused(path):
    """The clause. This is the assertion that did not exist."""
    issued = _issue()
    assert _consume(issued["ticket"]) is not None, f"[{path}] first use failed"
    assert _consume(issued["ticket"]) is None, (
        f"[{path}] a ticket was accepted twice — anyone who observed it in a "
        "proxy log or a shared screen could open a shell on the instance"
    )


def test_an_expired_ticket_is_refused(path):
    """"Short-lived" is the other half of the clause."""
    issued = _issue(ttl_sec=1)
    time.sleep(1.2)
    assert _consume(issued["ticket"]) is None, f"[{path}] an expired ticket was accepted"


def test_a_ticket_for_another_purpose_is_refused(path):
    """A ticket minted to stream logs must not open a terminal."""
    issued = _issue()
    assert _consume(issued["ticket"], purpose="logs") is None, (
        f"[{path}] a terminal ticket was accepted for a different purpose"
    )


def test_a_ticket_for_another_target_is_refused(path):
    """One instance's ticket must not reach another instance."""
    issued = _issue()
    assert _consume(issued["ticket"], target="job-2") is None, (
        f"[{path}] a ticket issued for job-1 was accepted for job-2"
    )


def test_a_ticket_from_another_address_is_refused(path):
    """The pin that limits a leaked ticket to the network that requested it."""
    issued = _issue()
    assert _consume(issued["ticket"], ip="198.51.100.4") is None, (
        f"[{path}] a ticket was accepted from an address it was not issued to"
    )


def test_an_unknown_ticket_is_refused(path):
    assert _consume("not-a-real-ticket") is None


def test_a_refused_replay_does_not_consume_a_second_ticket(path):
    """Two live tickets must stay independent.

    A shared purge or a mis-keyed pop could invalidate an unrelated session, and
    that failure would look like a flake rather than a bug.
    """
    first = _issue()
    second = _issue()
    assert _consume(first["ticket"]) is not None
    assert _consume(first["ticket"]) is None
    assert _consume(second["ticket"]) is not None, (
        f"[{path}] consuming one ticket invalidated another"
    )
