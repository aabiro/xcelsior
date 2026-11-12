"""A ping answered by someone else's network is not a heartbeat.

Two hosts registered through the public site were recorded with the CDN edge
address their request came through. The edge answers pings, so the health check
kept them "active" for months although no agent ever ran on them.
"""

import pytest

import scheduler


@pytest.mark.parametrize(
    "ip", ["100.64.0.6", "100.64.0.4", "192.168.1.20", "10.0.0.5", "172.30.243.2"]
)
def test_mesh_and_private_addresses_are_the_machine(ip):
    assert scheduler.ping_is_evidence_of_a_host(ip)


@pytest.mark.parametrize(
    "ip", ["104.23.176.7", "162.158.106.153", "172.69.130.189", "8.8.8.8", "", "not-an-ip"]
)
def test_public_or_malformed_addresses_are_not(ip):
    assert not scheduler.ping_is_evidence_of_a_host(ip)


def test_the_health_check_does_not_ping_a_public_address(monkeypatch):
    pinged = []
    monkeypatch.setattr(scheduler, "ping_host", lambda ip: pinged.append(ip) or True)
    hosts = [
        {"host_id": "edge", "ip": "104.23.176.7", "status": "active", "last_seen": 0},
        {"host_id": "mesh", "ip": "100.64.0.6", "status": "active", "last_seen": 0},
    ]
    monkeypatch.setattr(scheduler, "load_hosts", lambda active_only=False: hosts)

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(scheduler, "_atomic_mutation", stop)
    with pytest.raises(_Stop):
        scheduler.check_hosts()
    assert pinged == ["100.64.0.6"]
