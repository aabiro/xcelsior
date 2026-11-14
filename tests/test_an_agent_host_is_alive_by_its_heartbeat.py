"""A host that runs an agent is alive when its agent heartbeats, not when the machine pings.

On 2026-11-13 a worker had been crash-looping for 55 hours. Its host answered
pings on the mesh, so the health check kept it "active" and refreshed its
last-seen time; a job placed there would never have started.
"""

import time

import pytest

import scheduler


class _Stop(Exception):
    pass


def _pinged(monkeypatch, host):
    """Addresses check_hosts pings for one host; stops before the database."""
    pinged = []
    monkeypatch.setattr(scheduler, "ping_host", lambda ip: pinged.append(ip) or True)
    monkeypatch.setattr(scheduler, "load_hosts", lambda active_only=False: [host])

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(scheduler, "_atomic_mutation", stop)
    with pytest.raises(_Stop):
        scheduler.check_hosts()
    return pinged


def _decision(monkeypatch, host, *, ping=True):
    """'alive' or 'dead' as check_hosts computes it, read from its own update list."""
    monkeypatch.setattr(scheduler, "ping_host", lambda ip: ping)
    monkeypatch.setattr(scheduler, "load_hosts", lambda active_only=False: [host])
    captured = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(scheduler, "_atomic_mutation", lambda: _Conn())
    monkeypatch.setattr(scheduler, "_get_host_by_id_conn", lambda conn, hid: dict(host))
    monkeypatch.setattr(scheduler, "_upsert_host_row", lambda conn, row: captured.update(row))
    monkeypatch.setattr(scheduler, "alert_host_dead", lambda *a: None)
    monkeypatch.setattr(scheduler, "_requeue_dead_host_jobs", lambda *a: None)
    monkeypatch.setattr(scheduler, "_host_first_miss", {})
    return scheduler.check_hosts()[host["host_id"]], captured


MESH = "100.64.0.4"


def test_a_stale_agent_heartbeat_is_dead_even_though_the_machine_pings(monkeypatch):
    host = {
        "host_id": "tower",
        "ip": MESH,
        "status": "active",
        "last_seen": time.time(),  # kept fresh by the health check itself
        "last_agent_heartbeat": time.time() - 3600,
    }
    result, _ = _decision(monkeypatch, host, ping=True)
    assert result == "dead"


def test_a_fresh_agent_heartbeat_is_alive_even_when_ping_fails(monkeypatch):
    host = {
        "host_id": "tower",
        "ip": MESH,
        "status": "active",
        "last_agent_heartbeat": time.time() - 5,
    }
    result, row = _decision(monkeypatch, host, ping=False)
    assert result == "alive"
    assert row["status"] == "active"


def test_the_health_check_never_refreshes_the_agents_heartbeat(monkeypatch):
    stamp = time.time() - 5
    host = {"host_id": "tower", "ip": MESH, "status": "active", "last_agent_heartbeat": stamp}
    _, row = _decision(monkeypatch, host, ping=True)
    assert row["last_agent_heartbeat"] == stamp, "only the agent may move its own heartbeat"


def test_an_agent_host_is_not_pinged_at_all(monkeypatch):
    host = {"host_id": "tower", "ip": MESH, "status": "active", "last_agent_heartbeat": time.time()}
    assert _pinged(monkeypatch, host) == []


def test_a_host_that_never_ran_an_agent_is_still_judged_by_ping(monkeypatch):
    host = {"host_id": "legacy", "ip": MESH, "status": "active", "last_seen": 0}
    assert _decision(monkeypatch, dict(host), ping=True)[0] == "alive"
    assert _decision(monkeypatch, dict(host), ping=False)[0] == "dead"


def test_a_stale_agent_host_keeps_its_status_through_the_grace_period(monkeypatch):
    host = {
        "host_id": "tower",
        "ip": MESH,
        "status": "active",
        "last_agent_heartbeat": time.time() - 3600,
    }
    _, row = _decision(monkeypatch, host)
    assert row["status"] == "active", "the first miss starts the grace period; it does not kill"
    assert scheduler.AGENT_HEARTBEAT_STALE_SEC >= 60


def test_the_heartbeat_route_records_the_agents_own_word():
    import inspect

    from routes import hosts

    body = inspect.getsource(hosts.api_register_host)
    assert 'entry["last_agent_heartbeat"] = time.time()' in body
    block = body[body.index("if h.agent_sha256:") : body.index('entry["last_agent_heartbeat"]')]
    assert (
        "\n    if " not in block[5:]
    ), "the heartbeat stamp must sit under the agent-hash condition"
