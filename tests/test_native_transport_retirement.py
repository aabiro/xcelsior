"""No scheduler mode can revive v1 execution or destructively drain commands."""

import pytest
from fastapi.testclient import TestClient

from api import app
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance


@pytest.mark.parametrize("mode", ["active", "canary", "paused", "shadow", "typo"])
@pytest.mark.parametrize("transport", ["work", "commands", "status"])
def test_every_v1_execution_transport_is_a_state_preserving_tombstone(monkeypatch, mode, transport):
    from routes import agent, instances
    from control_plane.legacy_execution import legacy_execution_allowed

    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    monkeypatch.setattr(agent, "_require_agent_auth", lambda *a, **k: {"host_id": "verified-host"})
    monkeypatch.setattr(
        instances, "_require_worker_status_update", lambda *a, **k: {"user_id": "verified-worker"}
    )
    assert legacy_execution_allowed() is False
    with running_instance() as native:
        job, host = native["job_id"], native["host_id"]

        def snapshot():
            with _get_pg_pool().connection() as c:
                return c.execute(
                    "SELECT md5(to_jsonb(j)::text),(SELECT count(*) FROM agent_commands cmd WHERE cmd.job_id=j.job_id) FROM jobs j WHERE j.job_id=%s",
                    (job,),
                ).fetchone()

        before = snapshot()
        client = TestClient(app)
        if transport == "status":
            response = client.patch(
                "/instance/" + job, json={"status": "completed", "host_id": host}
            )
        else:
            response = client.get("/agent/" + transport + "/" + host)
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "v2_protocol_required"
        assert snapshot() == before
