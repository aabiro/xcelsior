"""Shared DB helpers for tests (sqlite + postgres safe)."""

import functools
import os


@functools.cache
def postgres_is_reachable() -> bool:
    """One short connect attempt, remembered for the rest of the session.

    `db._get_pg_pool()` against an absent Postgres does not fail fast: it
    retries five times with backoff, about two and a half minutes, and caches
    nothing on failure. Every caller that probes with it pays that again. On
    the sandboxed runner, which has no database by design, two module-level
    probes and an autouse fixture that probed around every test turned the
    pure-parse gates into a job that hung until it was cancelled.

    True for a sqlite run, where there is nothing remote to reach.
    """
    if (os.environ.get("XCELSIOR_DB_BACKEND") or "").lower() not in ("postgres", "dual"):
        return True
    try:
        import psycopg
        from db import resolve_postgres_dsn

        psycopg.connect(resolve_postgres_dsn(), connect_timeout=3).close()
    except Exception:
        return False
    return True


#: A GPU the imported Stripe catalog prices (conftest imports it once per
#: session). Every customer submission quotes and funds a Stripe Price, so a
#: test that submits for an owner names a priced GPU and funds that owner.
PRICED_GPU = "RTX 4090"


def fund_test_owner(owner: str, amount_cad: float = 50.0) -> str:
    """Deposit enough for a service-tier hold; returns ``owner`` for inline use."""
    from billing import get_billing_engine

    get_billing_engine().deposit(owner, amount_cad, description="test funding for a priced launch")
    return owner


def price_test_gpu(
    gpu_model: str, *, vram_gb: int = 24, modes: tuple[str, ...] = ("on_demand", "spot")
) -> str:
    """Import Standard Stripe Prices for a test-only GPU model; returns the model.

    Tests that isolate their claims behind a unique model name still need a
    priced variant, because every customer submission quotes one. Imported
    through the deploy importer, exactly as ``paid_fleet`` does, never derived.
    """
    import importlib.util
    import uuid
    from pathlib import Path

    from db import _get_pg_pool

    marker = uuid.uuid4().hex[:12]
    prices = [
        {
            "variant": {
                "gpu_model": gpu_model,
                "vram_gb": vram_gb,
                "form_factor": "pcie",
                "high_frequency": False,
            },
            "mode": mode,
            "tier": "standard",
            "price": {
                "id": f"price_test_{marker}_{mode}",
                "product": f"prod_test_{marker}",
                "currency": "cad",
                "billing_scheme": "per_unit",
                "unit_amount_decimal": "49",
                "livemode": False,
                "lookup_key": f"test:{marker}:standard:{mode}",
            },
        }
        for mode in modes
    ]
    script = Path(__file__).resolve().parents[1] / "api/scripts/sync_stripe_service_tiers.py"
    spec = importlib.util.spec_from_file_location("test_gpu_price_import", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _get_pg_pool().connection() as conn:
        module.import_catalog(conn, {"catalog_version": f"test:{marker}", "prices": prices})
    return gpu_model


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


def report_test_host_inventory(host_id: str, *, gpu_count: int | None = None, conn=None) -> None:
    """Report a fixture worker's physical inventory through native ingest.

    Raw fixture declarations supply the emulated NVML model and device count;
    the production fleet projection correctly exposes no capacity before this
    report, so it cannot supply the inputs to its own first observation.
    """
    import json
    import time
    import uuid
    import scheduler
    from control_plane.observations import ingest_observation

    if conn is None:
        with scheduler._atomic_mutation() as transaction:
            return report_test_host_inventory(host_id, gpu_count=gpu_count, conn=transaction)
    row = conn.execute(
        "SELECT payload, capabilities FROM hosts WHERE host_id=%s", (host_id,)
    ).fetchone()
    if row is None:
        raise ValueError("fixture host must exist before its worker reports")
    raw = row["payload"] if isinstance(row, dict) else row[0]
    host = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    capabilities = row["capabilities"] if isinstance(row, dict) else row[1]
    rows = conn.execute(
        "SELECT gpu_uuid, device_index, model, total_vram_mb FROM host_gpu_devices "
        "WHERE host_id=%s AND retired_at IS NULL AND gpu_uuid LIKE 'GPU-%%' ORDER BY device_index",
        (host_id,),
    ).fetchall()
    if rows and gpu_count is None:
        inventory = [
            {
                "gpu_uuid": r["gpu_uuid"] if isinstance(r, dict) else r[0],
                "device_index": (
                    (r["device_index"] if isinstance(r, dict) else r[1])
                    if (r["device_index"] if isinstance(r, dict) else r[1]) is not None
                    else i
                ),
                "model": r["model"] if isinstance(r, dict) else r[2],
                "total_vram_mb": r["total_vram_mb"] if isinstance(r, dict) else r[3],
            }
            for i, r in enumerate(rows)
        ]
    else:
        count = int(host.get("gpu_count") or 1) if gpu_count is None else gpu_count
        inventory = [
            {
                "gpu_uuid": f"GPU-{uuid.uuid5(uuid.NAMESPACE_URL, host_id)}-{i}",
                "device_index": i,
                "model": str(host.get("gpu_model") or "RTX 4090"),
                "total_vram_mb": int(float(host.get("total_vram_gb") or 24) * 1024),
            }
            for i in range(count)
        ]
    ingest_observation(
        conn,
        host_id=host_id,
        session_id=f"fixture-{host_id}",
        observation_generation=time.monotonic_ns(),
        workloads=[],
        # A current worker that can read its inventory reports GPU UUID
        # binding (hardware_isolation.qualified_capabilities); paid placement
        # refuses a host that has not.
        capabilities={**dict(capabilities or {}), "gpu_uuid_binding": True},
        conditions={"gpu_inventory": "reported", "workloads": "reported"},
        gpu_inventory=inventory,
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


def allow_test_login(admin, role: str, dsn: str) -> None:
    """Let a service role log in with the suite's own DSN password.

    Tests connect as a restricted role with ``make_conninfo(dsn, user=role)``,
    which keeps the DSN's password. That only worked on a server that trusts
    local connections; under password auth (CI's postgres image) the role
    needs the same password. Restore with ``NOLOGIN PASSWORD NULL``.
    """
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict

    password = conninfo_to_dict(dsn).get("password")
    if password:
        admin.execute(
            sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
    else:
        admin.execute(sql.SQL("ALTER ROLE {} LOGIN").format(sql.Identifier(role)))
