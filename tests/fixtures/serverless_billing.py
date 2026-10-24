"""Explicit priced-slice fixtures; production authority is tested separately."""

import time
import uuid

from psycopg.rows import dict_row


def charge_priced_fixture_slice(billing, repo, worker, endpoint, *, paid_end=None, **kwargs):
    """Exercise billing arithmetic with an explicit fixture-approved quote.

    These calculator tests intentionally have no real worker/attempt. They
    cannot use the public debit entry point, which must reject such callers.
    Real execution, wallet, rollback and idempotency gates use that entry point.
    """
    from db import _get_pg_pool
    from serverless.billing_authority import capture_quote
    from serverless.metering import _charge_serverless_execution

    quote = capture_quote(endpoint)
    quote.update(
        execution_start=float(worker["allocated_at"]),
        execution_end_cap=float(kwargs.get("period_end") or time.time()),
    )
    with _get_pg_pool().connection() as conn:
        conn.row_factory = dict_row
        # Calculator fixtures share simple names, but never leak cycles into
        # another test. Real debit/cursor authority has separate native gates.
        with conn.transaction(force_rollback=True):
            if paid_end is not None:
                conn.execute(
                    """INSERT INTO billing_cycles(cycle_id,job_id,customer_id,host_id,
                    resource_type,period_start,period_end,duration_seconds,rate_per_hour,
                    amount_micros,status,created_at)
                    VALUES(%s,%s,%s,%s,'serverless_gpu',%s,%s,%s,%s,%s,'charged',%s)""",
                    (
                        "calc-paid-" + uuid.uuid4().hex,
                        worker["scheduler_job_id"],
                        endpoint["owner_id"],
                        worker["host_id"],
                        quote["execution_start"],
                        paid_end,
                        paid_end - quote["execution_start"],
                        quote["rate_per_gpu_micros"] / 1_000_000,
                        round(
                            (paid_end - quote["execution_start"])
                            * quote["rate_per_gpu_micros"]
                            / 3600
                        ),
                        time.time(),
                    ),
                )
            return _charge_serverless_execution(
                billing, repo, worker, endpoint, _conn=conn, _quote=quote, **kwargs
            )
