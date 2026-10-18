"""Rolling KV-cache hit KPI for token SKU GA (row 1: ≥30% within launch month)."""

from __future__ import annotations

import os
import time
from typing import Any

from serverless.slo import KV_CACHE_HIT_TARGET


def _window_sec() -> int:
    return int(os.environ.get("XCELSIOR_KV_KPI_WINDOW_SEC", str(30 * 86400)))


def record_token_cache_sample(
    *,
    input_tokens: int,
    cached_tokens: int,
    ts: float | None = None,
) -> None:
    """Append one served-request sample for rolling KV hit-rate."""
    from db import _get_pg_pool

    inp = max(0, int(input_tokens or 0))
    cached = max(0, min(int(cached_tokens or 0), inp))
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO serverless_kv_cache_samples (sample_ts, input_tokens, cached_tokens)
            VALUES (%s, %s, %s)
            """,
            (float(ts or time.time()), inp, cached),
        )
        conn.commit()


def rolling_kv_cache_hit_rate(*, since_ts: float | None = None) -> float:
    """Fraction of input tokens served from prefix cache over the KPI window."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    window_start = (since_ts if since_ts is not None else time.time() - _window_sec())
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0) AS total_in,
                COALESCE(SUM(cached_tokens), 0) AS total_cached
            FROM serverless_kv_cache_samples
            WHERE sample_ts >= %s
            """,
            (window_start,),
        ).fetchone()
    total_in = int((row or {}).get("total_in") or 0)
    total_cached = int((row or {}).get("total_cached") or 0)
    if total_in <= 0:
        return 0.0
    return round(total_cached / total_in, 4)


def kv_cache_kpi_status() -> dict[str, Any]:
    rate = rolling_kv_cache_hit_rate()
    return {
        "kv_cache_hit_rate": rate,
        "target": KV_CACHE_HIT_TARGET,
        "kpi_met": rate >= KV_CACHE_HIT_TARGET,
        "window_sec": _window_sec(),
    }