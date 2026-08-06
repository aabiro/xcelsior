"""The terminal door, which the first fix missed.

`406c0a1` enforced `instances:connect` on three routes and called one of them
"mints the WebSocket ticket for the browser terminal". That was wrong, and the
error is worth stating precisely because the commit reads convincingly:

* `/api/instances/{job_id}/stream-ticket` issues `purpose="instance_stream"`
* `/api/terminal/ticket` issues `purpose="terminal"`
* `/ws/terminal/{instance_id}` accepts **only** `purpose="terminal"`

So the scope whose consent text is *"Open a terminal on your running instances,
and publish their ports to the internet"* was enforced on port publishing, on
auto-launch discovery, on the streaming ticket — and not on the one route that
opens a terminal. A credential narrowed to exclude it was refused at the
streaming door and admitted at the terminal door.

**Ownership was never the gap.** `_check_terminal_access` has always required
the caller to own the instance and hold team instance-write. The question it
did not ask is the other one, and it is the same question `406c0a1` was written
to answer: given that it *is* your instance, may **this credential** open a
shell on it?

The direct WebSocket path is deliberately not changed. `/ws/terminal/{id}`
falls back to `_validate_ws_auth(..., allow_query_token=False)` when no ticket
is presented, which accepts only a session cookie — a browser, where
`_require_scope` correctly no-ops. Machine credentials cannot take that door,
so the ticket route is the whole of it.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")


def _source(module, name: str) -> str:
    return inspect.getsource(getattr(module, name))


def test_the_terminal_ticket_requires_the_connect_scope():
    """The fix."""
    import routes.terminal as terminal

    source = _source(terminal, "api_terminal_ticket")
    assert '_require_scope(user, "instances:connect")' in source, (
        "POST /api/terminal/ticket does not require instances:connect, so a "
        "credential narrowed to exclude it can still open a shell"
    )


def test_both_ticket_routes_ask_for_the_same_scope():
    """The two doors must agree, or the narrower one is decorative.

    Asserted against both rather than one, because the whole defect was that
    they diverged while looking equivalent.
    """
    import routes.instances as instances
    import routes.terminal as terminal

    stream = _source(instances, "api_instance_stream_ticket")
    ticket = _source(terminal, "api_terminal_ticket")
    for name, source in (("stream-ticket", stream), ("terminal ticket", ticket)):
        assert '_require_scope(user, "instances:connect")' in source, (
            f"{name} no longer requires instances:connect; the other door is "
            "now the only guarded one, which is how this defect arose"
        )


def test_ownership_is_still_checked_as_well():
    """Scope did not replace ownership.

    `instances:connect` says what this credential may do; `_check_terminal_access`
    says whose machine it is. Losing the second while adding the first would
    trade one hole for a worse one.
    """
    import routes.terminal as terminal

    source = _source(terminal, "api_terminal_ticket")
    assert "_check_terminal_access(" in source, (
        "the terminal ticket route no longer checks that the caller owns the "
        "instance"
    )


def test_the_terminal_socket_still_only_accepts_a_terminal_ticket():
    """Pins the fact the whole finding rests on.

    If `/ws/terminal/{id}` ever accepted the streaming purpose, the two ticket
    routes would become interchangeable and the scope check would need
    re-deriving from scratch.

    Scoped to the handler, not the module: the first version scanned all of
    `routes/terminal.py` and failed on this file's own explanation of the
    difference. Ninth time a text-scanning guard in this suite has flagged the
    prose about the thing it checks.
    """
    import routes.terminal as terminal

    source = _source(terminal, "ws_terminal")
    assert 'purpose="terminal"' in source, (
        "the terminal websocket no longer pins the ticket purpose"
    )
    assert "instance_stream" not in source, (
        "the terminal websocket now accepts the streaming purpose; the two "
        "ticket types are no longer distinct"
    )
