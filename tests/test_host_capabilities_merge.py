"""A host's heartbeat and its native observation each keep the other's capabilities.

The heartbeat (PUT /host) reports egress isolation and checkpoint support; the
observation reports physical isolation (`gpu_uuid_binding`, ...). Both used to
replace `hosts.capabilities` wholesale, so the binding flag flapped between
writes and paid placement (which requires it) was refused most of the time.
"""

import uuid

import pytest

from control_plane.observations import ingest_observation
from db import DatabaseOps, _get_pg_pool


@pytest.fixture
def host():
    with _get_pg_pool().connection() as conn:
        name = "capability-merge-" + uuid.uuid4().hex
        DatabaseOps.upsert_host(conn, {"host_id": name}, backend="postgres")
        try:
            yield conn, name
        finally:
            conn.rollback()


def capabilities(conn, host_id):
    return conn.execute("SELECT capabilities FROM hosts WHERE host_id=%s", (host_id,)).fetchone()[0]


def heartbeat(conn, host_id, caps):
    DatabaseOps.upsert_host(
        conn, {"host_id": host_id, "status": "active", "capabilities": caps}, backend="postgres"
    )


def observe(conn, host_id, generation, caps):
    ingest_observation(
        conn,
        host_id=host_id,
        session_id="capability-session",
        observation_generation=generation,
        workloads=[],
        capabilities=caps,
    )


def test_each_writer_keeps_the_others_keys(host):
    conn, host_id = host
    heartbeat(conn, host_id, {"egress_isolation": True, "criu": True})
    observe(conn, host_id, 1, {"egress_isolation": True, "gpu_uuid_binding": True})
    heartbeat(conn, host_id, {"egress_isolation": True, "criu": True})

    caps = capabilities(conn, host_id)
    assert caps["gpu_uuid_binding"] is True
    assert caps["criu"] is True

    observe(conn, host_id, 2, {"egress_isolation": True, "gpu_uuid_binding": True})
    assert capabilities(conn, host_id)["criu"] is True


def test_an_explicit_false_still_revokes(host):
    conn, host_id = host
    observe(conn, host_id, 1, {"gpu_uuid_binding": True, "egress_isolation": True})
    observe(conn, host_id, 2, {"gpu_uuid_binding": False, "egress_isolation": True})
    heartbeat(conn, host_id, {"egress_isolation": False})

    caps = capabilities(conn, host_id)
    assert caps["gpu_uuid_binding"] is False
    assert caps["egress_isolation"] is False


def test_a_heartbeat_without_capabilities_changes_none(host):
    conn, host_id = host
    observe(conn, host_id, 1, {"gpu_uuid_binding": True})
    DatabaseOps.upsert_host(conn, {"host_id": host_id, "status": "active"}, backend="postgres")
    assert capabilities(conn, host_id) == {"gpu_uuid_binding": True}
