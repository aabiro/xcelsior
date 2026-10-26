"""SSH relays end when their route disappears. An inventory blip keeps them."""

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
async def test_inventory_outage_keeps_established_relays(monkeypatch):
    gateway = SSHGateway()
    writer = _Writer()
    server = _Server()
    gateway.listeners[12001] = ListenerState(
        route=_route(12001), server=server, active_connections=1, writers=[writer]
    )

    async def unavailable():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(ssh_gateway, "query_running_instances", unavailable)
    await gateway.sync_routes()

    assert 12001 in gateway.listeners
    assert writer.aborted is False
    assert server.closed is False


@pytest.mark.asyncio
async def test_gateway_requires_a_configured_database(monkeypatch):
    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN_SSH_GATEWAY", raising=False)
    monkeypatch.delenv("XCELSIOR_PG_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "0")

    with pytest.raises(RuntimeError, match="XCELSIOR_POSTGRES_DSN"):
        await ssh_gateway._get_pg_conn()


def test_route_query_excludes_suspended_accounts():
    import inspect

    source = inspect.getsource(ssh_gateway.query_running_instances)
    sync = inspect.getsource(ssh_gateway.SSHGateway.sync_routes)
    assert "ssh_account_suspensions" in source
    assert "drain=False" in sync


# ── Port authority: the host reports a port, the job id decides it ─────────


def _job_route(job_id, port=None, host="host-1", ip="100.64.0.9"):
    if port is None:
        port = 10000 + int(job_id[:4], 16) % 55000  # the worker's mapping
    return InstanceRoute(job_id, port, ip, host, "running", "box")


async def _sync(monkeypatch, gateway, routes):
    opened = []

    async def inventory():
        return list(routes)

    async def record_open(route):
        opened.append(route)
        gateway.listeners[route.ssh_port] = ListenerState(route=route, server=_Server())

    monkeypatch.setattr(ssh_gateway, "query_running_instances", inventory)
    monkeypatch.setattr(gateway, "open_listener", record_open)
    await gateway.sync_routes()
    return opened


def test_expected_port_mirrors_the_worker_mapping():
    import worker_agent

    for job_id in ("abcd1234", "0000ffff", "ffff0000", "2a8b-uuid", "29ff0000"):
        assert ssh_gateway.expected_public_ssh_port(job_id) == (
            worker_agent._compute_public_ssh_port(job_id)
        )
    # No hex prefix, no derivable port: nothing to relay.
    for job_id in ("job-1a2b3c", "", "ab", "0xab12"):
        assert ssh_gateway.expected_public_ssh_port(job_id) is None


@pytest.mark.asyncio
async def test_a_host_cannot_claim_another_tenants_port(monkeypatch):
    victim = _job_route("1111aaaa", host="victim-host", ip="100.64.0.2")
    # The attacker's host reports the victim's port for its own job.
    attacker = _job_route("2222bbbb", port=victim.ssh_port, host="evil-host", ip="100.64.0.66")
    gateway = SSHGateway()

    opened = await _sync(monkeypatch, gateway, [victim, attacker])

    assert [r.job_id for r in opened] == ["1111aaaa"]
    assert gateway.listeners[victim.ssh_port].route.host_ip == "100.64.0.2"


@pytest.mark.asyncio
async def test_a_contested_port_is_routed_for_nobody(monkeypatch):
    # Two live jobs whose ids share a 4-hex prefix derive the same port.
    first = _job_route("abcd0001", host="host-a", ip="100.64.0.2")
    second = _job_route("abcd0002", host="host-b", ip="100.64.0.3")
    assert first.ssh_port == second.ssh_port
    gateway = SSHGateway()
    writer = _Writer()
    server = _Server()
    gateway.listeners[first.ssh_port] = ListenerState(
        route=first, server=server, active_connections=1, writers=[writer]
    )

    opened = await _sync(monkeypatch, gateway, [first, second])

    assert opened == []
    assert first.ssh_port not in gateway.listeners
    assert server.closed and writer.aborted


@pytest.mark.asyncio
async def test_a_port_handed_to_another_job_ends_the_old_relays(monkeypatch):
    old = _job_route("abcd0001", host="host-a", ip="100.64.0.2")
    new = _job_route("abcd0002", host="host-b", ip="100.64.0.3")
    gateway = SSHGateway()
    writer = _Writer()
    gateway.listeners[old.ssh_port] = ListenerState(
        route=old, server=_Server(), active_connections=1, writers=[writer]
    )

    opened = await _sync(monkeypatch, gateway, [new])

    assert [r.job_id for r in opened] == ["abcd0002"]
    assert writer.aborted
    assert gateway.listeners[new.ssh_port].route.host_ip == "100.64.0.3"
