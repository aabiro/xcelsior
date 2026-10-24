"""Explicit priced-slice fixtures; production authority is tested separately."""

from psycopg.rows import dict_row


def charge_priced_fixture_slice(billing, repo, worker, endpoint, **kwargs):
    """Exercise billing arithmetic with an explicit fixture-approved quote.

    These calculator tests intentionally have no real worker/attempt. They
    cannot use the public debit entry point, which must reject such callers.
    Real execution, wallet, rollback and idempotency gates use that entry point.
    """
    from db import _get_pg_pool
    from serverless.billing_authority import capture_quote
    from serverless.metering import _charge_serverless_execution

    quote = capture_quote(endpoint)
    with _get_pg_pool().connection() as conn:
        conn.row_factory = dict_row
        return _charge_serverless_execution(
            billing, repo, worker, endpoint, _conn=conn, _quote=quote, **kwargs
        )
