"""Every money column a SQL literal names must exist on the table it selects from.

Five production queries named `_cad` columns that migrations `095`–`100` had
dropped. Each one raised `UndefinedColumn` every time it ran, and each was
inside a `try` that turned the failure into a plausible answer:

- `/api/billing/gst-threshold/{provider_id}` summed `provider_payout_cad` from
  `payout_ledger`, caught the error, and computed the threshold from `0.0` — so
  it told **every** provider they were below the $30k GST registration
  threshold, which is a tax-compliance answer.
- `sla.auto_issue_credits` filtered `WHERE credit_cad > 0` on `sla_monthly`, so
  the query that pays out SLA credits matched nothing and no credit has ever
  been issued by that path.
- Three AI-assistant tools (provider earnings, SLA history, billing cycles)
  named dropped columns and returned an error string to the user instead.

`test_money_representation.py` did not catch these: it scans for *writes* to a
float money column, and these are reads. The failure mode is also the opposite
one — there the column exists and holds a stale float, here it does not exist at
all. So this checks the other direction: for each single-table SQL literal in
application source, every `_cad`/`_micros` identifier it references must be a
real column on that table.

Only money identifiers are checked, and only where the statement reads from
exactly one table, because that is the set this can resolve without a SQL
parser. Output aliases (`... / 1000000.0 AS amount_cad`) are excluded — they
name a result field, not a column.
"""

import os
import re

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from db import _get_pg_pool

from tests._source_tree import iter_source_files

MONEY_IDENT = re.compile(r"\b([a-z_][a-z0-9_]*_(?:cad|micros))\b", re.IGNORECASE)
# `<expr> AS name_cad` — the alias names an output field, not a column.
ALIASED = re.compile(r"\bAS\s+([a-z_][a-z0-9_]*_(?:cad|micros))\b", re.IGNORECASE)
FROM_TABLE = re.compile(r"\b(?:FROM|UPDATE|INTO)\s+([a-z_][a-z0-9_]*)\b", re.IGNORECASE)
# A CTE or subquery makes "the one table" ambiguous; so does an explicit JOIN.
MULTI_TABLE = re.compile(r"\b(?:JOIN|WITH)\b", re.IGNORECASE)
# A `--` comment inside the literal is prose. `billing.py` explains next to the
# INSERT that "087 dropped total_cost_cad" — which a text scan reads as a
# reference to the dropped column, the same recursion that has flagged this
# suite's own docstrings before.
SQL_COMMENT = re.compile(r"--[^\n]*")


def _schema_columns() -> dict[str, set[str]]:
    pool = _get_pg_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public'"
        ).fetchall()
    out: dict[str, set[str]] = {}
    for r in rows:
        table = r["table_name"] if isinstance(r, dict) else r[0]
        column = r["column_name"] if isinstance(r, dict) else r[1]
        out.setdefault(table, set()).add(column)
    return out


def _sql_literals():
    """Yield (rel_path, lineno, sql) for every string literal that reads a table."""
    import ast

    # `migrations/` is excluded on purpose: a migration operates on the schema
    # *at its own revision*, so 097 naming `amount_cad` is the drop itself, and
    # 043 naming `base_rate_cad` predates its removal. Checking them against
    # head would forbid every migration that has ever touched a money column.
    for path, rel in iter_source_files(exclude_prefixes=("migrations/",)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                # f-string: only the literal parts are inspectable, which is
                # enough — a column name is never interpolated.
                text = "".join(
                    v.value
                    for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
            if not text:
                continue
            text = SQL_COMMENT.sub(" ", text)
            if not FROM_TABLE.search(text):
                continue
            if not MONEY_IDENT.search(text):
                continue
            yield rel, getattr(node, "lineno", 0), text


def test_money_columns_named_in_sql_exist_on_their_table():
    schema = _schema_columns()
    if not schema:
        pytest.skip("no PostgreSQL schema available")

    bad: list[str] = []
    for rel, lineno, sql in _sql_literals():
        tables = {t.lower() for t in FROM_TABLE.findall(sql)}
        tables = {t for t in tables if t in schema}
        if len(tables) != 1 or MULTI_TABLE.search(sql):
            continue
        table = next(iter(tables))
        columns = schema[table]
        aliases = {a.lower() for a in ALIASED.findall(sql)}
        for ident in {i.lower() for i in MONEY_IDENT.findall(sql)}:
            if ident in aliases or ident in columns:
                continue
            bad.append(f"{rel}:{lineno}: {table}.{ident} does not exist")

    assert not bad, "SQL names money columns that do not exist:\n  " + "\n  ".join(sorted(bad))
