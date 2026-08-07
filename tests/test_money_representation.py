"""Money is stored and mutated as integer micros, never as float.

Binary floating point cannot represent most decimal amounts exactly, so every
arithmetic step on a float money column accumulates error.

**The schema no longer carries a single `_cad` column** — `095`–`097` removed the
float money, and `100` removed the last two (`provider_accounts`, which were
`NUMERIC` and so escaped a sweep aimed at floats). `test_the_schema_holds_no_cad_column`
below turns that into a floor rather than a milestone.

The source scan is kept, and is not redundant with it. A column can be
reintroduced by a migration *and* written by application code, and the two
failures want catching in different places: the schema check fails on the
migration that adds it, the source scan fails on the code that writes it. The
names below are the historical projection pairs — where `wallets_project_money`
derived a float from the integer column on write, and a stray application write
would have inverted the projection and made the float the source of truth.

This guards the invariant rather than a single call site, because the failure
is silent — a float write produces a plausible-looking number.
"""

import os
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# The float half of a dual-representation pair, keyed by the table it lives on.
# Scoped deliberately: a `_cad` column is only a projection where a trigger
# derives it. Columns with no `_micros` twin (serverless_endpoints.total_cost_cad)
# are ordinary float columns and out of scope here.
#
# payout_splits is excluded on purpose: it carries all four pairs but has NO
# projection trigger, so its `_cad` and `_micros` columns are independently
# written and can silently diverge. Writing only micros there would leave the
# float stale, so stripe_connect correctly writes both. Giving payout_splits a
# trigger (or dropping its float columns outright) is tracked in HANDOFF.
PROJECTED_FLOAT_COLUMNS = {
    "wallets": (
        "balance_cad",
        "total_deposited_cad",
        "total_spent_cad",
        "total_refunded_cad",
    ),
    "wallet_transactions": ("amount_cad", "balance_after_cad"),
    "wallet_holds": ("amount_cad",),
    "usage_meters": ("total_cost_cad",),
}
FLOAT_MONEY_COLUMNS = tuple(
    sorted({c for cols in PROJECTED_FLOAT_COLUMNS.values() for c in cols})
)

SKIP_DIRS = {".venv", "venv", "node_modules", ".next", "migrations", "tests", "__pycache__"}


def _application_sources():
    for path in REPO.rglob("*.py"):
        if SKIP_DIRS & set(path.parts):
            continue
        yield path


def test_no_application_code_writes_a_float_money_column():
    """A SQL assignment to a float money column inverts the projection."""
    offenders = []
    for path in _application_sources():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Only look inside SQL string literals. A Python keyword argument
        # named amount_cad is fine — the API contract speaks CAD; what must
        # not happen is a float column being assigned in an UPDATE or INSERT.
        for block in re.finditer(r'"""(.*?)"""', text, re.S):
            sql = block.group(1)
            if not re.search(r"\b(UPDATE|INSERT)\b", sql, re.I):
                continue
            tables = [t for t in PROJECTED_FLOAT_COLUMNS if re.search(rf"\b{t}\b", sql)]
            if not tables:
                continue
            columns = {c for t in tables for c in PROJECTED_FLOAT_COLUMNS[t]}
            for column in sorted(columns):
                if re.search(rf"(?:SET\s+|,\s*){re.escape(column)}\s*=(?!=)", sql):
                    line = text[: block.start()].count("\n") + 1
                    offenders.append(
                        f"{path.relative_to(REPO)}:~{line} writes {column} in SQL"
                    )
    assert not offenders, (
        "Money must be written as integer micros; the _cad columns are "
        "projections maintained by wallets_project_money. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_the_money_helpers_round_rather_than_truncate():
    """int(x * 1_000_000) truncates; 10.07 CAD would land a micro short."""
    from money import cad_to_micros, micros_to_cad

    assert cad_to_micros("10.07") == 10_070_000
    assert cad_to_micros(10.07) == 10_070_000
    assert cad_to_micros("0.1") + cad_to_micros("0.2") == cad_to_micros("0.3")
    assert micros_to_cad(10_070_000) == 10.07


def test_float_arithmetic_would_have_drifted():
    """Documents why this matters, and fails if the helpers regress to floats."""
    from money import cad_to_micros

    # The classic float result is 0.30000000000000004.
    assert 0.1 + 0.2 != 0.3
    assert cad_to_micros(0.1) + cad_to_micros(0.2) == 300_000


@pytest.mark.needs_db
def test_the_schema_holds_no_cad_column():
    """Zero, and it stays zero.

    A floor rather than a milestone: reaching zero is worth nothing if the next
    migration can add `total_earned_cad` back without anything noticing. The
    same discipline as `MAX_UNCLASSIFIED = 0` — once the number is at the
    bottom, the guard is what keeps it there.

    Excluded from the sandboxed gates by `needs_db`, since it reads a live
    catalogue; `tests/test_migration_ledger.py` covers the migrations there.
    """
    import psycopg
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    assert dsn, "no database DSN configured; this test cannot verify anything"

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        offenders = conn.execute(
            """SELECT table_schema || '.' || table_name || '.' || column_name
                 FROM information_schema.columns
                WHERE column_name LIKE %s
                  AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY 1""",
            ("%%_cad",),
        ).fetchall()

    assert not offenders, (
        "these columns store money as CAD rather than integer micros: "
        f"{[o[0] for o in offenders]}. Money is integer micros throughout; CAD "
        "is a presentation unit the API converts to at its boundary, not a "
        "storage type."
    )
