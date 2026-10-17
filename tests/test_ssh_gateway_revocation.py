"""SSH relays end when their route disappears or the inventory cannot be read."""

import asyncio

import pytest

import ssh_gateway
from ssh_gateway import InstanceRoute, ListenerState, SSHGateway


class _Server:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _Writer:
    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True

    def close(self):
        self.aborted = True


def _route(port=12000):
    return InstanceRoute("job-12345678", port, "100.64.0.9", "host-1", "running", "box")


@pytest.mark.asyncio
async def test_lost_route_aborts_open_relays_without_draining():
    gateway = SSHGateway()
    writer = _Writer()
    state = ListenerState(route=_route(), server=_Server(), active_connections=1, writers=[writer])
    gateway.listeners[12000] = state

    await gateway.close_listener(12000, drain=False)

    assert 12000 not in gateway.listeners
    assert state.server.closed
    assert writer.aborted


@pytest.mark.asyncio
async def test_inventory_outage_closes_established_relays(monkeypatch):
    gateway = SSHGateway()
    writer = _Writer()
    gateway.listeners[12001] = ListenerState(
        route=_route(12001), server=_Server(), active_connections=1, writers=[writer]
    )

    async def unavailable():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(ssh_gateway, "query_running_instances", unavailable)
    await gateway.sync_routes()

    assert gateway.listeners == {}
    assert writer.aborted


def test_route_query_excludes_suspended_accounts():
    import inspect

    source = inspect.getsource(ssh_gateway.query_running_instances)
    sync = inspect.getsource(ssh_gateway.SSHGateway.sync_routes)
    assert "ssh_account_suspensions" in source
    assert "drain=False" in sync
