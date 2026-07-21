import pytest
import os
import threading
from unittest.mock import patch, MagicMock

import bg_worker
from control_plane.scheduled_tasks import register_task, claim_and_run_tasks
from db import pg_connection

# Task names this module owns. Scoping the cleanup to these matters:
# `DELETE FROM scheduled_tasks` also removes the rows migrations seed
# (telemetry_partition_maintenance, wallet_hold_expiry,
# host_agent_token_expiry), which silently breaks every later test in the
# same process that asserts those durable sweeps are registered.
_OWNED_TASKS = ("test_dummy", "test_fail")


@pytest.fixture
def clean_tasks():
    def _purge():
        with pg_connection() as conn:
            conn.execute(
                "DELETE FROM scheduled_tasks WHERE task_name = ANY(%s)",
                (list(_OWNED_TASKS),),
            )
            conn.commit()

    _purge()
    yield
    _purge()

def test_task_registration_and_execution(clean_tasks):
    # Setup test task
    executed = []
    def my_task():
        executed.append(True)
        
    register_task("test_dummy", my_task, 60)
    
    # Run claim and execute
    count = claim_and_run_tasks("test-worker")
    assert count == 1
    assert len(executed) == 1
    
    # Second run should do nothing (next_run_at is in future)
    count = claim_and_run_tasks("test-worker")
    assert count == 0
    assert len(executed) == 1

def test_task_failure_handling(clean_tasks):
    def failing_task():
        raise RuntimeError("boom")
        
    register_task("test_fail", failing_task, 60)
    
    # Claim will catch exception and update last_error
    count = claim_and_run_tasks("test-worker")
    assert count == 0  # 0 successful
    
    with pg_connection() as conn:
        row = conn.execute("SELECT last_status, last_error FROM scheduled_tasks WHERE task_name='test_fail'").fetchone()
        assert row[0] == 'failed'
        assert 'boom' in row[1]
