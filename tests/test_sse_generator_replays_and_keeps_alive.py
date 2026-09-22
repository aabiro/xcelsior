"""The SSE generator, driven directly instead of over HTTP.

`scripts/measure_route_execution.py` lists three streaming handlers as never
entered, and the reason recorded there is real: `httpx.ASGITransport` buffers a
response before returning it, so a handler that never completes never returns
to the caller. That is a fact about the transport.

It is not a reason to leave the *logic* untested. `_sse_generator` is an async
generator; an `async for` over it exercises every branch without a socket, a
transport, or a server. What HTTP would add is the wiring around it, which is
four lines of `StreamingResponse`.

The logic underneath is not trivial and had no test at all:

* a reconnecting client sends `Last-Event-ID` and must receive the durable gap
  *before* any live event, or it silently loses every transition that happened
  while it was disconnected (B4.6, §16.3);
* a replay against a store that is down must degrade to live-only rather than
  failing the stream, since a broken outbox should not take the dashboard with
  it;
* an idle connection must emit a keepalive, or an intermediary closes it;
* the subscriber queue must be removed on exit, or every disconnect leaks one
  and `broadcast_sse` fans out to queues nobody reads.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

import routes._deps as deps
from routes.health import _sse_generator


class _FakeRequest:
    """Stands in for the Request the generator polls for disconnection."""

    def __init__(self, disconnect_after: int = 10**6):
        self._checks = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._checks += 1
        return self._checks > self._disconnect_after


async def _take(gen, count: int, timeout: float = 5.0) -> list[str]:
    """Pull `count` chunks, failing loudly rather than hanging the suite."""
    out: list[str] = []

    async def pump():
        async for chunk in gen:
            out.append(chunk)
            if len(out) >= count:
                return

    await asyncio.wait_for(pump(), timeout=timeout)
    return out


@pytest.fixture(autouse=True)
def clean_subscribers():
    """The fan-out list is module state; a leak here would corrupt other tests."""
    with deps._sse_lock:
        deps._sse_subscribers.clear()
    yield
    with deps._sse_lock:
        deps._sse_subscribers.clear()


# ── Handshake ─────────────────────────────────────────────────────────────


async def test_the_stream_opens_with_a_retry_hint_and_a_connected_event() -> None:
    """A client that reconnects without `retry:` hammers the server."""
    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    chunks = await _take(gen, 2)
    await gen.aclose()

    assert chunks[0].startswith("retry:"), f"no reconnect hint first: {chunks[0]!r}"
    assert "event: connected" in chunks[1]
    assert json.loads(chunks[1].split("data: ", 1)[1].strip())["status"] == "connected"


async def test_opening_registers_exactly_one_subscriber() -> None:
    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    await _take(gen, 1)
    assert len(deps._sse_subscribers) == 1, (
        f"{len(deps._sse_subscribers)} subscribers registered for one stream"
    )
    await gen.aclose()


async def test_closing_removes_the_subscriber() -> None:
    """Otherwise every disconnect leaks a queue that `broadcast_sse` still fills."""
    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    await _take(gen, 1)
    await gen.aclose()
    assert deps._sse_subscribers == [], (
        "the subscriber outlived the stream; broadcast_sse will keep pushing into "
        "a queue nobody reads until it fills and is reaped"
    )


# ── Live fan-out ──────────────────────────────────────────────────────────


async def test_a_broadcast_reaches_a_live_subscriber() -> None:
    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    await _take(gen, 2)  # drain the handshake

    deps.broadcast_sse("host_update", {"host_id": "h-1", "status": "active"})
    chunk = (await _take(gen, 1))[0]
    await gen.aclose()

    assert "event: host_update" in chunk, chunk
    assert json.loads(chunk.split("data: ", 1)[1].strip())["host_id"] == "h-1"


async def test_the_stream_stops_when_the_client_disconnects() -> None:
    """The loop polls `is_disconnected`; without it a dropped client streams forever."""
    gen = _sse_generator(_FakeRequest(disconnect_after=0), last_event_id=None)
    collected = []
    async def drain():
        async for chunk in gen:
            collected.append(chunk)
    await asyncio.wait_for(drain(), timeout=5.0)
    assert collected, "the stream produced nothing at all"
    assert deps._sse_subscribers == [], "a disconnected client left its queue behind"


# ── Durable replay on reconnect ───────────────────────────────────────────


class _Event:
    def __init__(self, cursor, event_type, payload):
        self.cursor, self.event_type, self.payload = cursor, event_type, payload


async def test_a_reconnect_replays_the_gap_before_any_live_event(monkeypatch) -> None:
    """The property the cursor exists for.

    A client that lost its connection must receive what it missed, in order,
    before anything new — otherwise it resumes with a hole it cannot detect.
    """
    import control_plane.event_stream as event_stream

    replayed = [
        _Event("c-1", "job_started", {"job_id": "j-1"}),
        _Event("c-2", "job_finished", {"job_id": "j-1"}),
    ]
    monkeypatch.setattr(event_stream, "resume_after", lambda conn, cur, limit=1000: replayed)

    gen = _sse_generator(_FakeRequest(), last_event_id="c-0")
    chunks = await _take(gen, 4)  # retry, connected, then both replayed events
    await gen.aclose()

    assert "id: c-1" in chunks[2] and "event: job_started" in chunks[2], chunks[2]
    assert "id: c-2" in chunks[3] and "event: job_finished" in chunks[3], chunks[3]


async def test_a_fresh_connect_replays_nothing(monkeypatch) -> None:
    """No cursor means "start live" — replaying history to a new client is noise."""
    import control_plane.event_stream as event_stream

    called = []
    monkeypatch.setattr(
        event_stream, "resume_after", lambda *a, **k: called.append(1) or []
    )
    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    await _take(gen, 2)
    await gen.aclose()
    assert not called, "a first connection triggered a durable replay"


async def test_a_broken_event_store_degrades_to_live_only(monkeypatch) -> None:
    """A dead outbox must not take the dashboard down with it."""
    import control_plane.event_stream as event_stream

    def _explode(*_args, **_kwargs):
        raise RuntimeError("event store unavailable")

    monkeypatch.setattr(event_stream, "resume_after", _explode)

    gen = _sse_generator(_FakeRequest(), last_event_id="c-0")
    await _take(gen, 2)  # handshake still arrives
    deps.broadcast_sse("host_update", {"host_id": "h-2"})
    chunk = (await _take(gen, 1))[0]
    await gen.aclose()

    assert "event: host_update" in chunk, (
        f"replay failure killed the live stream instead of degrading: {chunk!r}"
    )


# ── Keepalive ─────────────────────────────────────────────────────────────


async def test_an_idle_stream_emits_a_keepalive(monkeypatch) -> None:
    """30s of silence is long enough for an intermediary to close the connection.

    The real timeout is patched down rather than waited out — the assertion is
    that a *timeout* produces a comment frame instead of ending the stream.
    """
    # Shadow *this module's view* of asyncio, not the asyncio module itself.
    # `monkeypatch.setattr("routes.health.asyncio.wait_for", ...)` rewrites the
    # attribute on the shared module object, so it also rebinds the
    # `asyncio.wait_for` this test file uses for its own deadline — the helper
    # below then inherited the 0.05s timeout and failed before the generator
    # could yield anything.
    class _ImpatientAsyncio:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        async def wait_for(self, awaitable, timeout):
            return await asyncio.wait_for(awaitable, 0.05)

    monkeypatch.setattr("routes.health.asyncio", _ImpatientAsyncio())

    gen = _sse_generator(_FakeRequest(), last_event_id=None)
    chunks = await _take(gen, 3)
    await gen.aclose()

    assert chunks[2].startswith(":"), (
        f"an idle stream sent {chunks[2]!r} rather than a keepalive comment"
    )
