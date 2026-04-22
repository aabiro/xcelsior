"""P3/C4+C5 — enqueue_agent_command args size + per-host queue cap."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


SRC_AGENT = Path(__file__).resolve().parent.parent / "routes" / "agent.py"


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

def test_c4_args_under_16kb_accepted():
    from routes.agent import enqueue_agent_command

    pool, _ = _mock_pool_factory(pending_count=0)
    with patch("routes.agent._get_pg_pool", return_value=pool):
        cid = enqueue_agent_command(
            "host-1", "snapshot_container", args={"image_ref": "foo/bar:baz"},
        )
    assert cid == 42


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
                "host-1", "snapshot_container",
                args={"x": "A" * 200},
            )
    assert exc.value.status_code == 413


# ---------- C5 ----------

def test_c5_under_cap_accepted():
    from routes.agent import enqueue_agent_command

    pool, _ = _mock_pool_factory(pending_count=10)
    with patch("routes.agent._get_pg_pool", return_value=pool):
        cid = enqueue_agent_command("host-1", "snapshot_container", args={})
    assert cid == 42


def test_c5_at_or_above_cap_rejected():
    from routes.agent import enqueue_agent_command

    pool, _ = _mock_pool_factory(pending_count=1000)
    with patch("routes.agent._get_pg_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command("host-1", "snapshot_container", args={})
    assert exc.value.status_code == 503
    assert "queue full" in exc.value.detail.lower()


def test_c5_env_var_lowers_cap(monkeypatch):
    from routes.agent import enqueue_agent_command

    monkeypatch.setenv("XCELSIOR_AGENT_QUEUE_MAX", "5")
    pool, _ = _mock_pool_factory(pending_count=5)
    with patch("routes.agent._get_pg_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            enqueue_agent_command("host-1", "snapshot_container", args={})
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
