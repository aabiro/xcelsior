"""C7 — PG-backed regression test for registry_health.refresh_prometheus_gauges().

Validates the fix from commit 89686ce: every Prometheus scrape MUST read
``registry_health_cache`` directly (not the in-process ``_state`` dict),
because gunicorn workers each have a separate interpreter and the probe
runs in a different process (``bg_worker``).

Without this contract, /metrics emits HELP/TYPE only with no sample
values — the bug observed before 89686ce.
"""

from __future__ import annotations

import time
import uuid

import pytest


@pytest.fixture
def cleanup_registries():
    keys: list[str] = []
    yield keys
    if not keys:
        return
    try:
        from db import _get_pg_pool

        pool = _get_pg_pool()
        with pool.connection() as conn:
            for k in keys:
                conn.execute("DELETE FROM registry_health_cache WHERE registry=%s", (k,))
            conn.commit()
    except Exception:
        pass


def _insert_probe(registry: str, *, reachable: bool, latency_ms: float, ts: float) -> None:
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO registry_health_cache
                (registry, reachable, last_probe_at, latency_ms, status_code, error)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (registry) DO UPDATE SET
                reachable     = EXCLUDED.reachable,
                last_probe_at = EXCLUDED.last_probe_at,
                latency_ms    = EXCLUDED.latency_ms,
                status_code   = EXCLUDED.status_code,
                error         = EXCLUDED.error
            """,
            (registry, reachable, ts, latency_ms, 200 if reachable else None, None),
        )
        conn.commit()


def _gauge_value(gauge, **labels) -> float | None:
    """Read a labelled prometheus Gauge sample value, or None if unset."""
    try:
        child = gauge.labels(**labels)
    except Exception:
        return None
    val = getattr(child, "_value", None)
    if val is None:
        return None
    try:
        return val.get()
    except Exception:
        return None


def test_refresh_prometheus_gauges_reads_from_db(cleanup_registries):
    """89686ce regression — gauges MUST be populated from the DB row,
    not from the per-process ``_state`` cache.
    """
    import registry_health

    reg = f"ghcr.io/test-{uuid.uuid4().hex[:8]}"
    cleanup_registries.append(reg)

    now = time.time()
    _insert_probe(reg, reachable=True, latency_ms=42.5, ts=now)

    # Sanity check: in-process state has NEVER been touched for this fake
    # registry. If the gauge were sourced from `_state`, the assertion
    # below would observe a stale or empty value.
    registry_health.refresh_prometheus_gauges()

    assert _gauge_value(registry_health._reachable_gauge, registry=reg) == 1.0
    assert _gauge_value(registry_health._probe_latency_gauge, registry=reg) == 42.5
    last_ts = _gauge_value(registry_health._last_probe_gauge, registry=reg)
    assert last_ts is not None and abs(last_ts - now) < 1.0


def test_refresh_prometheus_gauges_reflects_unreachable_state(cleanup_registries):
    """Flipping reachable=False on disk must propagate to the gauge."""
    import registry_health

    reg = f"ghcr.io/test-{uuid.uuid4().hex[:8]}"
    cleanup_registries.append(reg)

    now = time.time()
    _insert_probe(reg, reachable=True, latency_ms=10.0, ts=now)
    registry_health.refresh_prometheus_gauges()
    assert _gauge_value(registry_health._reachable_gauge, registry=reg) == 1.0

    # Flip the row, refresh again — gauge MUST drop to 0.
    _insert_probe(reg, reachable=False, latency_ms=99.0, ts=now + 1.0)
    registry_health.refresh_prometheus_gauges()
    assert _gauge_value(registry_health._reachable_gauge, registry=reg) == 0.0
    assert _gauge_value(registry_health._probe_latency_gauge, registry=reg) == 99.0


def test_refresh_prometheus_gauges_handles_empty_table():
    """No rows → no exception. Pre-89686ce code path that crashed on empty
    tables would emit HELP/TYPE only; after the fix the no-op is silent.
    """
    import registry_health

    # Don't insert anything; just make sure the call is safe.
    registry_health.refresh_prometheus_gauges()  # MUST NOT raise
