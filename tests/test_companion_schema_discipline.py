"""Schema discipline from the data-architecture companion, §4.4.

These are the rules the companion states as invariants, checked against the
live schema rather than against intent. They exist because the schema drifted
from them once already: `agent_api_keys` (083) was modelled on a pre-companion
table and shipped with float timestamps and no tenant column, while the `082`
tables written days earlier followed both rules.

Scoped to tables added from migration 080 onward. The companion acknowledges
the pre-existing schema violates these ("many TEXT, floating-point timestamps,
floating-point currency values") and treats fixing it as staged work — so this
guards new tables rather than failing on inherited debt.
"""

import pytest

from db import _get_pg_pool

# Tables introduced by migrations 080-088, which are held to the companion's
# rules in full. Adding a table here is how a new migration opts in.
GOVERNED_TABLES = (
    "payout_splits",
    "privacy_deletion_requests",
    "privacy_deletion_sink_status",
    "host_compatibility_sessions",
    "host_admission_evidence",
    "host_admission_decisions",
    "agent_api_keys",
    "casl_consent",
    "user_encryption_keys",
)

# No exemptions. An earlier version skipped the privacy deletion tables on the
# grounds that a deletion subject must stay unlinkable — but that conflated
# tenant with identity. The companion keeps the tenant and pseudonymises the
# identity (2.1: a row "must own identity, tenant, checksum, state, retention,
# region, and deletion status"; 11.2: pseudonymous keys with "direct identity
# only in restricted mapping"). The tenant is the workspace, not the person.
NOT_TENANT_SCOPED: dict[str, str] = {}

def _columns(table: str) -> dict[str, str]:
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            """SELECT column_name, data_type
                 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s""",
            (table,),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


@pytest.mark.parametrize("table", GOVERNED_TABLES)
def test_time_columns_are_timestamptz(table):
    """§4.4.5 — typed timestamps, not epoch floats.

    Float time is lossy at second granularity in the 2020s and sorts
    incorrectly against NULL, which is how "never used" became "used in 1970".
    """
    cols = _columns(table)
    if not cols:
        pytest.skip(f"{table} not present in this database")
    offenders = {
        name: dtype
        for name, dtype in cols.items()
        if (name.endswith("_at") or name in {"expires", "observed"})
        and dtype != "timestamp with time zone"
    }
    assert not offenders, (
        f"{table} stores time as something other than TIMESTAMPTZ "
        f"(companion §4.4.5): {offenders}"
    )


@pytest.mark.parametrize("table", GOVERNED_TABLES)
def test_tenant_owned_tables_carry_tenant_id(table):
    """§4.4.10 — a non-null tenant_id, so cross-tenant denial is provable."""
    cols = _columns(table)
    if not cols:
        pytest.skip(f"{table} not present in this database")
    if table in NOT_TENANT_SCOPED:
        pytest.skip(f"{table}: {NOT_TENANT_SCOPED[table]}")
    assert "tenant_id" in cols, (
        f"{table} has no tenant_id; a tenant-scoped query would have to join "
        f"back through users (companion §4.4.10)"
    )
    with _get_pg_pool().connection() as conn:
        nullable = conn.execute(
            """SELECT is_nullable FROM information_schema.columns
                WHERE table_schema='public' AND table_name=%s
                  AND column_name='tenant_id'""",
            (table,),
        ).fetchone()[0]
    assert nullable == "NO", f"{table}.tenant_id must be NOT NULL (§4.4.10)"


@pytest.mark.parametrize("table", GOVERNED_TABLES)
def test_no_float_money_columns(table):
    """§4.4.6 — money is integer minor units or NUMERIC, never binary float."""
    cols = _columns(table)
    if not cols:
        pytest.skip(f"{table} not present in this database")
    offenders = {
        name: dtype
        for name, dtype in cols.items()
        if name.endswith("_cad") and dtype == "double precision"
    }
    assert not offenders, (
        f"{table} stores money as binary float (companion §4.4.6): {offenders}"
    )


def test_tenant_scoped_tables_index_tenant_first():
    """§4.4.10 — an index beginning with tenant_id for common access paths."""
    missing = []
    for table in GOVERNED_TABLES:
        if table in NOT_TENANT_SCOPED or not _columns(table):
            continue
        with _get_pg_pool().connection() as conn:
            defs = [
                r[0]
                for r in conn.execute(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname='public' AND tablename=%s",
                    (table,),
                ).fetchall()
            ]
        if not any("(tenant_id" in d.replace(" ", "") for d in defs):
            missing.append(table)
    assert not missing, (
        f"tables with no tenant_id-leading index (companion §4.4.10): {missing}"
    )
