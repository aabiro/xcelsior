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