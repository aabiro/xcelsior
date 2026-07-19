import pytest
import os
import threading
from unittest.mock import patch, MagicMock

import bg_worker
from control_plane.scheduled_tasks import register_task, claim_and_run_tasks
from db import pg_connection

@pytest.fixture
def clean_tasks():
    with pg_connection() as conn:
        conn.execute("DELETE FROM scheduled_tasks")
        conn.commit()

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
