"""Schema discipline from the data-architecture companion, §4.4.

These are the rules the companion states as invariants, checked against the
live schema rather than against intent. They exist because the schema drifted
from them once already: `agent_api_keys` (083) was modelled on a pre-companion
table and shipped with float timestamps and no tenant column, while the `082`
tables written days earlier followed both rules.

Scoped to tables created by migration 080 onward, plus older tables a later
migration has since brought up to standard. The companion acknowledges the
pre-existing schema violates these ("many TEXT, floating-point timestamps,
floating-point currency values") and treats fixing it as staged work — so this
guards new and remediated tables rather than failing on inherited debt.

The governed set is read out of the migrations themselves, not maintained by
hand here, so a new table is governed the moment it is created.
"""

import pathlib
import re

import pytest

from db import _get_pg_pool

# Every assertion here reads the live schema, so without a database there is
# nothing to check. Probed once at import and skipped as a module, matching
# `tests/test_no_runtime_ddl.py` — the pattern already proven in this suite.
#
# Without it the file does not fail, it *stalls*: each test retries the pool
# until pytest's 180-second timeout, nine times over. In the sandboxed CI
# runner, which has no database by design, that consumed the entire
# twenty-minute job budget and the run was cancelled rather than reported —
# the slowest possible way to learn a test cannot run here.
try:
    _probe_pool = _get_pg_pool()
    with _probe_pool.connection() as _probe_conn:
        _probe_conn.execute("SELECT 1").fetchone()
except Exception as _probe_error:  # pragma: no cover - environment dependent
    pytestmark = pytest.mark.skip(f"no pg pool available: {_probe_error}")

# Governance starts here: the companion predates the schema, so tables created
# from this revision onward had no excuse to diverge.
FIRST_GOVERNED_REVISION = "080"

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations" / "versions"

# The trailing `(` is what keeps prose out of the result — several migrations
# discuss `CREATE TABLE IF NOT EXISTS` in their docstrings.
_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_]*)\s*\(", re.I
)


def _tables_created_since(revision: str) -> frozenset[str]:
    """Every table any migration at or after `revision` creates.

    Derived rather than hand-listed on purpose. The previous version of this
    module kept a literal tuple and said "adding a table here is how a new
    migration opts in" — which means a migration that forgets to opt in is
    silently ungoverned, the exact drift these tests exist to catch. Governance
    now attaches to the migration itself, so it cannot be skipped by omission.
    """
    tables: set[str] = set()
    for path in sorted(MIGRATIONS_DIR.glob("*.py")):
        rev = path.name.split("_", 1)[0]
        if rev.isdigit() and rev >= revision:
            tables |= set(_CREATE_TABLE.findall(path.read_text()))
    return frozenset(tables)


# Pre-existing tables a later migration brought up to the companion's standard.
# They are not caught by the rule above because they were created earlier, so
# they are named explicitly. This is a ratchet: once a table is remediated it
# is listed here and can never regress. Removing an entry is the only way to
# lower a standard, which makes that visible in review.
ADOPTED_LEGACY_TABLES = frozenset(
    {
        "payout_splits",  # 080 hardened it, 089 finished it
    }
)

GOVERNED_TABLES = tuple(
    sorted(_tables_created_since(FIRST_GOVERNED_REVISION) | ADOPTED_LEGACY_TABLES)
)

# There is deliberately no exemption list. An earlier version of this module
# carried one and used it to skip the privacy deletion tables, on the grounds
# that a deletion subject must stay unlinkable — but that conflated tenant with
# identity. The companion keeps the tenant and pseudonymises the identity (2.1:
# a row "must own identity, tenant, checksum, state, retention, region, and
# deletion status"; 11.2: pseudonymous keys with "direct identity only in
# restricted mapping"). The tenant is the workspace, not the person. An empty
# exemption map would just be an invitation to refill it.


def test_governed_set_is_derived_and_populated():
    """Canary: a regex regression must not silently empty the governed set.

    Every other test in this module is parametrised over GOVERNED_TABLES, so if
    that collection ever came back empty the whole file would report green
    while checking nothing.
    """
    assert len(GOVERNED_TABLES) >= 9, (
        f"governed set collapsed to {GOVERNED_TABLES}; _CREATE_TABLE or "
        f"MIGRATIONS_DIR is likely wrong"
    )
    # Anchors from three different migrations, so a single file moving or being
    # renamed does not go unnoticed.
    for anchor in ("agent_api_keys", "host_admission_decisions", "casl_consent"):
        assert anchor in GOVERNED_TABLES, f"{anchor} is no longer governed"


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
def test_no_binary_float_columns(table):
    """§4.4.6 — money is integer minor units or NUMERIC, never binary float.

    Checked as "no binary float at all" rather than "no `_cad` column that is a
    float". Matching on the suffix only catches money that admits to being
    money: `total_micros` stored as a double would have passed, and so would
    any future column that holds an amount under a different name. No governed
    table has a legitimate use for binary float, so the absolute rule is both
    stricter and simpler than guessing from names.
    """
    cols = _columns(table)
    if not cols:
        pytest.skip(f"{table} not present in this database")
    offenders = {
        name: dtype for name, dtype in cols.items() if dtype in ("double precision", "real")
    }
    assert not offenders, (
        f"{table} has binary float columns (companion §4.4.6): {offenders}"
    )


def test_tenant_scoped_tables_index_tenant_first():
    """§4.4.10 — an index beginning with tenant_id for common access paths."""
    missing = []
    for table in GOVERNED_TABLES:
        if not _columns(table):
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


# ── Inherited float money: a ratchet, not a rule ──────────────────────────
#
# Migration 087 removed every `_cad` column that duplicated a `_micros` column
# — those were projections, and a float projection of an integer source is a
# rounding bug waiting to be reconciled against. What remains is different:
# 30 `_cad` columns across 18 legacy tables where the float *is* the only
# representation of that amount. Converting them is a real migration per table
# with code changes on both sides, not a mechanical sweep, and several sit on
# financially authoritative tables (invoices, payout_ledger, billing_cycles).
#
# So this is not asserted to be zero. It is pinned, so the number can only go
# down: a new float money column fails here, and finishing a table's conversion
# requires lowering the bound in the same commit. That makes the debt a visible
# countdown instead of a comment nobody revisits.
# Zero, and it stays zero. 095 mirrored the last 26 float CAD columns into
# integer micros, every read and write moved, and 097 dropped the floats after
# verifying the two representations agreed on every row. This is no longer a
# budget to be negotiated: a new float money column is a straight failure.
MAX_LEGACY_FLOAT_CAD_COLUMNS = 0


def test_legacy_float_money_only_shrinks():
    """§4.4.6, applied to inherited debt as a downward ratchet."""
    with _get_pg_pool().connection() as conn:
        offenders = [
            f"{r[0]}.{r[1]}"
            for r in conn.execute(
                r"""SELECT table_name, column_name
                      FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND column_name LIKE '%\_cad'
                       AND data_type IN ('double precision', 'real')
                     ORDER BY table_name, column_name"""
            ).fetchall()
        ]
    assert len(offenders) <= MAX_LEGACY_FLOAT_CAD_COLUMNS, (
        f"{len(offenders)} float CAD columns, budget is "
        f"{MAX_LEGACY_FLOAT_CAD_COLUMNS} (companion §4.4.6). New money must be "
        f"integer micros or NUMERIC:\n  " + "\n  ".join(offenders)
    )
    assert len(offenders) == MAX_LEGACY_FLOAT_CAD_COLUMNS, (
        f"only {len(offenders)} float CAD columns remain but the budget is "
        f"still {MAX_LEGACY_FLOAT_CAD_COLUMNS}. Lower "
        f"MAX_LEGACY_FLOAT_CAD_COLUMNS to {len(offenders)} so the ratchet "
        f"cannot slip back."
    )


def test_no_float_cad_column_shadows_a_micros_column():
    """The 087 defect class, asserted as gone rather than assumed.

    A `_cad` float alongside a `_micros` integer for the same amount means two
    sources of truth for one number, and they drift. 087 dropped all twelve;
    this fails if one is ever reintroduced.
    """
    with _get_pg_pool().connection() as conn:
        shadows = [
            f"{r[0]}.{r[1]}"
            for r in conn.execute(
                r"""SELECT c.table_name, c.column_name
                      FROM information_schema.columns c
                     WHERE c.table_schema = 'public'
                       AND c.column_name LIKE '%\_cad'
                       AND EXISTS (
                           SELECT 1 FROM information_schema.columns m
                            WHERE m.table_schema = 'public'
                              AND m.table_name = c.table_name
                              AND m.column_name IN (
                                  regexp_replace(c.column_name, '_cad$', '') || '_micros',
                                  regexp_replace(c.column_name, '_cad$', '') || '_minor'
                              )
                       )
                     ORDER BY 1, 2"""
            ).fetchall()
        ]
    # 095 deliberately creates 26 such pairs as a transition: micros is mirrored
    # from the float by a BEFORE INSERT OR UPDATE trigger so the two cannot
    # diverge, which is precisely what made the 087 pairs dangerous. A shadow is
    # therefore tolerated *only* while its mirror trigger exists. 096 drops the
    # floats and the triggers together, and this list empties itself.
    with _get_pg_pool().connection() as conn:
        mirrored = {
            r[0]
            for r in conn.execute(
                """SELECT REPLACE(tgname, 'trg_mirror_', '')
                     FROM pg_trigger WHERE tgname LIKE 'trg_mirror_%'"""
            ).fetchall()
        }
    unguarded = [
        s for s in shadows
        if s.split(".", 1)[1].removesuffix("_cad") + "_micros" not in mirrored
    ]
    assert not unguarded, (
        "a _cad column shadows a _micros/_minor column with no mirror trigger, "
        f"so the two can drift — one is a stale projection: {unguarded}"
    )


# The companion records this conformance state in prose (§4.4, "Conformance
# state as implemented"). A number written in a document goes stale silently,
# which is how the schema drifted from the companion in the first place — so
# the document is checked against the database like everything else here.
COMPANION_DOC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs"
    / "xcelsior-production-data-architecture-companion.md"
)

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty",
}


def _spell(n: int) -> str:
    if n in _NUMBER_WORDS:
        return _NUMBER_WORDS[n]
    tens, units = divmod(n, 10)
    if tens * 10 in _NUMBER_WORDS and units:
        return f"{_NUMBER_WORDS[tens * 10]}-{_NUMBER_WORDS[units]}"
    return str(n)


def test_companion_conformance_prose_matches_the_database():
    """The companion's stated float-money debt must be the real debt."""
    if not COMPANION_DOC.exists():
        pytest.skip("companion document not present in this checkout")
    with _get_pg_pool().connection() as conn:
        columns, tables = conn.execute(
            r"""SELECT count(*), count(DISTINCT table_name)
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND column_name LIKE '%\_cad'
                   AND data_type IN ('double precision', 'real')"""
        ).fetchone()

    # Guard the guard: MAX_LEGACY_FLOAT_CAD_COLUMNS and the prose must not
    # drift apart from each other either.
    assert columns == MAX_LEGACY_FLOAT_CAD_COLUMNS

    # At zero the debt sentence would read "0 columns across 0 tables", which is
    # true and useless. The cutover is finished, so the prose has to say that
    # instead — and this still fails if a float money column ever reappears,
    # because the branch above it flips back to the counted sentence.
    if columns == 0:
        sentence = "No float money columns remain"
    else:
        sentence = (
            f"{_spell(columns).capitalize()} `_cad` columns across "
            f"{_spell(tables)} legacy tables"
        )
    assert sentence in COMPANION_DOC.read_text(), (
        f"the companion's §4.4 conformance prose no longer matches the "
        f"database. Expected it to say:\n  {sentence!r}\n"
        f"Update the sentence in {COMPANION_DOC.name} in the same commit as "
        f"the migration that changed the count."
    )
