"""P3/C4+C5 — enqueue_agent_command args size + per-host queue cap."""

import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

SRC_AGENT = Path(__file__).resolve().parent.parent / "api" / "routes" / "agent.py"


def _mock_pool_factory(pending_count: int = 0, inserted_id: int = 42):
    """Build a nested-mock psycopg pool that returns controllable counts."""
    cur = MagicMock()
    # First SELECT returns pending_count; subsequent INSERT returns id.
    cur.fetchone.side_effect = [(pending_count,), (inserted_id,)]

    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = None

    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    pool.connection.return_value.__exit__.return_value = None
    return pool, cur


# ---------- C4 ----------


@contextmanager
def _native_directive_target():
    """A job directive needs a running, admitted native attempt to target."""
    from db import _get_pg_pool
    from tests.fixtures.native_runtime import running_instance

    with running_instance(owner=f"enqueue-cap-{uuid.uuid4().hex[:8]}") as instance:
        try:
            yield instance
        finally:
            with _get_pg_pool().connection() as conn:
                conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (instance["host_id"],))


def _fill_queue(host_id: str, count: int) -> None:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            """INSERT INTO agent_commands (host_id, command, args, created_by, expires_at)
               SELECT %s, 'health_probe', '{}'::jsonb, 'cap-test',
                      EXTRACT(EPOCH FROM NOW()) + 900
                 FROM generate_series(1, %s)""",
            (host_id, count),
        )


def test_c4_args_under_16kb_accepted():
    from routes.agent import enqueue_agent_command

    with _native_directive_target() as instance:
        cid = enqueue_agent_command(
            instance["host_id"],
            "snapshot_container",
            args={"job_id": instance["job_id"], "image_ref": "foo/bar:baz"},
        )
    assert isinstance(cid, int) and cid > 0


def test_c4_args_over_16kb_rejected():
    from routes.agent import enqueue_agent_command

    huge = {"blob": "A" * 20000}
    pool, _ = _mock_pool_factory()
    with patch("routes.agent._get_pg_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command("host-1", "snapshot_container", args=huge)
    assert exc.value.status_code == 413
    assert "too large" in exc.value.detail.lower()


def test_c4_env_var_adjusts_limit(monkeypatch):
    from routes.agent import enqueue_agent_command

    monkeypatch.setenv("XCELSIOR_AGENT_ARGS_MAX_BYTES", "100")
    pool, _ = _mock_pool_factory()
    with patch("routes.agent._get_pg_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command(
                "host-1",
                "snapshot_container",
                args={"x": "A" * 200},
            )
    assert exc.value.status_code == 413


# ---------- C5 ----------


def test_c5_under_cap_accepted():
    from routes.agent import enqueue_agent_command

    with _native_directive_target() as instance:
        _fill_queue(instance["host_id"], 10)
        cid = enqueue_agent_command(
            instance["host_id"], "snapshot_container", args={"job_id": instance["job_id"]}
        )
    assert isinstance(cid, int) and cid > 0


def test_c5_at_or_above_cap_rejected():
    from routes.agent import enqueue_agent_command

    with _native_directive_target() as instance:
        _fill_queue(instance["host_id"], 1000)
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command(
                instance["host_id"], "snapshot_container", args={"job_id": instance["job_id"]}
            )
    assert exc.value.status_code == 503
    assert "queue full" in exc.value.detail.lower()


def test_c5_env_var_lowers_cap(monkeypatch):
    from routes.agent import enqueue_agent_command

    monkeypatch.setenv("XCELSIOR_AGENT_QUEUE_MAX", "5")
    with _native_directive_target() as instance:
        _fill_queue(instance["host_id"], 5)
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command(
                instance["host_id"], "snapshot_container", args={"job_id": instance["job_id"]}
            )
    assert exc.value.status_code == 503


def test_c5_cap_is_per_host():
    """Different hosts queried separately — source grep guard."""
    src = SRC_AGENT.read_text()
    idx = src.find("def enqueue_agent_command(")
    end = src.find("\n# ── Admission failure", idx)
    body = src[idx:end]
    assert "WHERE host_id=%s" in body, "queue-depth query must be scoped to host_id"
    assert "status='pending'" in body


def test_c5_cap_ignores_expired_commands():
    """Expired rows shouldn't count against the cap."""
    src = SRC_AGENT.read_text()
    idx = src.find("SELECT COUNT(*) FROM agent_commands")
    end = src.find(")", idx + 100)
    body = src[idx:end]
    assert "expires_at >" in body, "cap must ignore expired rows"
