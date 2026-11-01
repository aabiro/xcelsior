"""Shared DB helpers for tests (sqlite + postgres safe)."""


def admit_test_host(
    host_id: str,
    *,
    active: bool = False,
    egress_isolated: bool = True,
    report_inventory: bool = False,
) -> None:
    """Mark a host admitted (and optionally active) via DatabaseOps.

    Admitted production hosts also report verified tenant egress isolation
    (egress_isolation.py), without which restricted jobs are never placed.
    ``report_inventory`` additionally has the host's worker report its
    physical GPUs through the v2 observation channel, which is what makes an
    admitted host schedulable natively (see ``report_test_host_inventory``).
    """
    import scheduler

    backend = scheduler._active_backend()
    with scheduler._atomic_mutation() as conn:
        data = scheduler.DatabaseOps.get_host(conn, host_id, backend=backend)
        if not data:
            return
        data["admitted"] = True
        if egress_isolated:
            data["capabilities"] = {**(data.get("capabilities") or {}), "egress_isolation": True}
        if active:
            data["status"] = "active"
        if backend == "postgres":
            # Trusted fixture authority first; the real native persistence
            # writer then sets scheduling state and the compatibility mirror.
            conn.execute(
                "UPDATE hosts SET admission_state='admitted' WHERE host_id=%s",
                (host_id,),
            )
        scheduler.DatabaseOps.upsert_host(conn, data, backend=backend)
    if report_inventory and backend == "postgres":
        report_test_host_inventory(host_id)


def report_test_host_inventory(host_id: str, *, gpu_count: int | None = None) -> None:
    """The host's worker reports its physical inventory (``/agent/v2/observations``).

    Native placement needs what production hosts provide every 60 s: a
    DB-clock observation (freshness), physical ``GPU-`` device identities and
    verified egress isolation. Re-reports the host's existing physical devices,
    or derives NVML-shaped ones from the registration payload's model/VRAM.
    """
    import time
    import uuid

    import scheduler
    from control_plane.db import run_transaction
    from control_plane.observations import ingest_observation

    with scheduler._atomic_mutation() as conn:
        host = scheduler.DatabaseOps.get_host(conn, host_id, backend="postgres") or {}
        rows = conn.execute(
            "SELECT gpu_uuid, device_index, model, total_vram_mb FROM host_gpu_devices "
            "WHERE host_id=%s AND retired_at IS NULL AND gpu_uuid LIKE 'GPU-%%' "
            "ORDER BY device_index",
            (host_id,),
        ).fetchall()
    if rows and gpu_count is None:
        inventory = [
            {
                "gpu_uuid": r["gpu_uuid"] if isinstance(r, dict) else r[0],
                "device_index": r["device_index"] if isinstance(r, dict) else r[1],
                "model": r["model"] if isinstance(r, dict) else r[2],
                "total_vram_mb": r["total_vram_mb"] if isinstance(r, dict) else r[3],
            }
            for r in rows
        ]
    else:
        count = gpu_count or int(host.get("gpu_count") or 1)
        vram_mb = int(float(host.get("total_vram_gb") or 24) * 1024)
        inventory = [
            {
                "gpu_uuid": f"GPU-{uuid.uuid5(uuid.NAMESPACE_URL, host_id)}-{i}",
                "device_index": i,
                "model": str(host.get("gpu_model") or "RTX 4090"),
                "total_vram_mb": vram_mb,
            }
            for i in range(count)
        ]
    capabilities = dict(host.get("capabilities") or {})
    run_transaction(
        lambda c: ingest_observation(
            c,
            host_id=host_id,
            session_id=f"fixture-{host_id}",
            observation_generation=time.monotonic_ns(),
            workloads=[],
            capabilities=capabilities,
            conditions={"gpu_inventory": "reported"},
            gpu_inventory=inventory,
        )
    )


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
