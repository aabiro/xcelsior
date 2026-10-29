import logging
import time
from typing import Callable, Dict, Any, Optional
from psycopg.rows import dict_row
from db import pg_connection, pg_transaction

log = logging.getLogger("xcelsior.scheduled_tasks")

_registry: Dict[str, Callable] = {}

def register_task(task_name: str, func: Callable, default_interval_sec: int, enabled: bool = True):
    """Register a scheduled task for durable periodic execution."""
    _registry[task_name] = func
    
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO scheduled_tasks 
                (task_name, enabled, interval_seconds, next_run_at)
            VALUES (%s, %s, %s, clock_timestamp())
            ON CONFLICT (task_name) DO UPDATE 
            SET interval_seconds = EXCLUDED.interval_seconds,
                enabled = EXCLUDED.enabled
            """,
            (task_name, enabled, default_interval_sec)
        )

def claim_and_run_tasks(worker_id: str, limit: int = 10) -> int:
    """Claim and execute due tasks."""
    claimed_tasks = []
    
    with pg_transaction() as conn:
        conn.row_factory = dict_row
        # Claim SKIP LOCKED
        rows = conn.execute(
            """
            UPDATE scheduled_tasks
            SET claim_owner = %s,
                claim_expires_at = clock_timestamp() + interval '5 minutes',
                updated_at = clock_timestamp()
            WHERE task_name IN (
                SELECT task_name FROM scheduled_tasks
                WHERE enabled
                  AND next_run_at <= clock_timestamp()
                  AND (claim_owner IS NULL OR claim_expires_at < clock_timestamp())
                ORDER BY next_run_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            RETURNING task_name, interval_seconds, payload
            """,
            (worker_id, limit)
        ).fetchall()
        
        claimed_tasks = [dict(r) for r in rows]

    executed_count = 0
    for task in claimed_tasks:
        task_name = task["task_name"]
        interval = task["interval_seconds"]
        
        func = _registry.get(task_name)
        if not func:
            log.error("Scheduled task '%s' not found in registry", task_name)
            with pg_transaction() as conn:
                conn.execute(
                    """
                    UPDATE scheduled_tasks
                    SET last_status = 'failed',
                        last_error = 'Task function not found in registry',
                        claim_owner = NULL,
                        claim_expires_at = NULL,
                        next_run_at = clock_timestamp() + make_interval(secs => %s)
                    WHERE task_name = %s
                    """,
                    (interval, task_name)
                )
            continue

        try:
            func()
            with pg_transaction() as conn:
                conn.execute(
                    """
                    UPDATE scheduled_tasks
                    SET last_status = 'succeeded',
                        last_error = NULL,
                        last_run_at = clock_timestamp(),
                        claim_owner = NULL,
                        claim_expires_at = NULL,
                        next_run_at = clock_timestamp() + make_interval(secs => %s)
                    WHERE task_name = %s
                    """,
                    (interval, task_name)
                )
            executed_count += 1
        except Exception as e:
            log.exception("Task '%s' failed", task_name)
            with pg_transaction() as conn:
                conn.execute(
                    """
                    UPDATE scheduled_tasks
                    SET last_status = 'failed',
                        last_error = %s,
                        last_run_at = clock_timestamp(),
                        claim_owner = NULL,
                        claim_expires_at = NULL,
                        next_run_at = clock_timestamp() + interval '60 seconds'
                    WHERE task_name = %s
                    """,
                    (str(e), task_name)
                )

    return executed_count

