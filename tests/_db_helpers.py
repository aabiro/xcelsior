"""Shared DB helpers for tests (sqlite + postgres safe)."""


def admit_test_host(host_id: str, *, active: bool = False) -> None:
    """Mark a host admitted (and optionally active) via DatabaseOps."""
    import scheduler

    backend = scheduler._active_backend()
    with scheduler._atomic_mutation() as conn:
        data = scheduler.DatabaseOps.get_host(conn, host_id, backend=backend)
        if not data:
            return
        data["admitted"] = True
        if active:
            data["status"] = "active"
        scheduler.DatabaseOps.upsert_host(conn, data, backend=backend)


from contextlib import contextmanager


@contextmanager
def foreign_scheduled_tasks_deferred(owned_tasks, defer_minutes: int = 30):
    """Make claim_and_run_tasks deterministic for a test's own durable tasks.

    The shared test database accumulates scheduled_tasks rows from app-stack
    runs and other modules. claim_and_run_tasks claims oldest-next_run_at
    first with a small batch limit, so stale due rows can crowd a freshly
    registered task out of the batch entirely. Claiming with a huge limit is
    NOT the answer: any foreign task with a function registered in this
    process would actually execute (one such task SSH-provisions autoscale
    hosts — real connect timeouts and state mutations mid-suite).

    Instead: push foreign due rows just outside the claim window, run the
    test's claims with the default batch, and restore the exact timestamps.
    No foreign task function runs; no durable state is left modified.
    """
    from db import pg_connection

    with pg_connection() as conn:
        rows = conn.execute(
            "SELECT task_name, next_run_at FROM scheduled_tasks "
            "WHERE task_name != ALL(%s) "
            "AND next_run_at <= clock_timestamp() + interval '1 minute'",
            (list(owned_tasks),),
        ).fetchall()
        deferred = [(r[0], r[1]) for r in rows]
        if deferred:
            conn.execute(
                "UPDATE scheduled_tasks "
                "SET next_run_at = clock_timestamp() + make_interval(mins => %s) "
                "WHERE task_name = ANY(%s)",
                (defer_minutes, [name for name, _ in deferred]),
            )
        conn.commit()
    try:
        yield
    finally:
        with pg_connection() as conn:
            for name, ts in deferred:
                conn.execute(
                    "UPDATE scheduled_tasks SET next_run_at = %s WHERE task_name = %s",
                    (ts, name),
                )
            conn.commit()
