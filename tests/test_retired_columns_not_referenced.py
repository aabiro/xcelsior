"""No live SQL may reference a column a migration dropped.

This exists because two did, and neither failed a test.

Migration `087` dropped `wallets.balance_cad` and `usage_meters.total_cost_cad`
as float projections of integer micros. `routes/admin.py` kept selecting
`balance_cad` and `ai_assistant.py` kept selecting `total_cost_cad`. Both sites
wrap their query in `except Exception`, so neither raised — the admin user list
silently reported a $0.00 balance for every user, and the assistant silently
reported no recent usage. A swallowed `UndefinedColumn` looks exactly like an
empty result, which is the failure mode the companion warns about (§22.7: do
not add fallbacks that hide failed execution).

The retired set is read out of the migrations themselves, so this cannot drift
from what was actually dropped.

That claim was not true until now. The loader named exactly two files — `085`
and `087` — so every column dropped after them was invisible here, and eight
queries against columns `095`, `097` and `100` removed sat in the tree for
months doing exactly what the two above did: failing inside `except Exception`
and rendering as data. `/api/billing/gst-threshold/{provider_id}` told every
provider they were below the $30k GST registration threshold, computed from a
sum that had never once executed.

So the loader now reads **every** migration, in all three shapes they use to
drop a column — a module-level tuple list (`DEAD_COLUMNS`, `DERIVED`,
`COLUMNS`), an `op.drop_column(...)` call, and a raw
`ALTER TABLE ... DROP COLUMN` inside `op.execute(...)`, which is what `100`
uses and what a text scan for the first two forms would miss.

A column is then only retired if the **live schema** also lacks it. Migrations
add columns back, and rename cycles drop-then-recreate; without that check this
would forbid naming a column that exists. The catalogue is the arbiter, the
migrations are only how candidates are found.
"""

import ast
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS = REPO / "migrations" / "versions"

# Not part of the running application. `migrations` is excluded because a
# migration must be able to name the column it drops.
SKIP_PREFIXES = ("tests/", "migrations/", "scripts/")


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # alembic's op is unavailable at import time; the
        pass  # module-level tuples we want are already bound.
    return module


#: `ALTER TABLE <t> DROP COLUMN [IF EXISTS] <c>` inside an `op.execute` string.
_ALTER_DROP = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?([a-z_][a-z0-9_]*)\s+"
    r"DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)


def _live_columns() -> set[tuple[str, str]] | None:
    """(table, column) for the migrated database, or None if unavailable."""
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            rows = conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public'"
            ).fetchall()
    except Exception:
        return None
    return {(r[0], r[1]) for r in rows}


def _dropped_anywhere() -> dict[str, set[str]]:
    """{column: {tables}} every migration drops, in all three shapes."""
    dropped: dict[str, set[str]] = {}

    def note(table: str, column: str) -> None:
        dropped.setdefault(str(column), set()).add(str(table))

    for path in sorted(MIGRATIONS.glob("*.py")):
        module = _load(path)

        # Shape 1: module-level tuple lists.
        for entry in getattr(module, "DEAD_COLUMNS", ()):
            note(entry[0], entry[1])
        for entry in getattr(module, "DERIVED", ()):
            table, columns = entry[0], entry[-1]
            for column in columns:
                note(table, column)
        for entry in getattr(module, "COLUMNS", ()):
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                note(entry[0], entry[1])

        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue

        # Shape 2: op.drop_column("table", "column").
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "drop_column"
                and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) for a in node.args[:2])
            ):
                note(node.args[0].value, node.args[1].value)

        # Shape 3: raw DDL in op.execute(...) — how `100` drops its columns.
        for table, column in _ALTER_DROP.findall(source):
            note(table, column)

    return dropped


def _retired_columns() -> dict[str, set[str]]:
    """{column_name: {tables it was dropped from}} — dropped, and still gone.

    The live catalogue filters the candidates: a migration that drops a column
    and a later one that adds it back leaves a name this must not forbid.
    """
    dropped = _dropped_anywhere()
    live = _live_columns()
    if live is None:
        return dropped

    retired: dict[str, set[str]] = {}
    for column, tables in dropped.items():
        gone = {t for t in tables if (t, column) not in live}
        if gone:
            retired[column] = gone
    return retired


def _python_sources() -> list[pathlib.Path]:
    """Tracked Python files only.

    Asking git rather than walking the tree keeps vendored and ignored
    directories out by construction — a checkout here carries both a `.venv`
    and a stray `venv`, together nearly 14,000 files, and a hand-maintained
    skip list is one rename away from either missing them or missing real code.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return [
        REPO / rel
        for rel in listed
        if rel and not rel.startswith(SKIP_PREFIXES) and (REPO / rel).is_file()
    ]


# `... AS balance_cad` inside a query defines an alias; the column it derives
# from is a live one. That is a definition, not a reference.
_ALIAS = re.compile(r"\bAS\s+$", re.I)
#: Every name the statement defines with `AS`. A name aliased anywhere in a
#: statement is an alias *everywhere* in it — `ORDER BY base_rate_cad` after
#: `... AS base_rate_cad` sorts by the projection, and reads as a bare column
#: reference to a scan that only looks immediately leftward.
_ALIAS_NAMES = re.compile(r"\bAS\s+([a-z_][a-z0-9_]*)", re.I)
#: A name inside SQL quotes is a value or a JSON key, never a column reference:
#: `payload->>'spot_rate_cad'` reads a key out of jsonb on a table whose
#: `spot_rate_cad` column is long gone, and is correct.
_SQL_STRING = re.compile(r"'[^']*'")
_SQL_VERB = re.compile(r"\b(SELECT|UPDATE|INSERT\s+INTO|DELETE\s+FROM|RETURNING)\b", re.I)
# `-- total_cost_cad is derived by ...` is commentary, not a column reference.
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def _sql_literals(text: str):
    """Yield (lineno, sql) for every string constant that looks like SQL.

    Working from the AST rather than from raw lines is what makes this precise.
    `billing.py` has a local named `amount_cad` — a perfectly live variable —
    inside functions that also run queries against `wallet_transactions`. Any
    line-window heuristic flags it, and seven such false positives is enough to
    make people delete the test. A column reference lives inside the query
    string or it is not a column reference.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, ast.JoinedStr):
            value = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
        else:
            continue
        if _SQL_VERB.search(value):
            yield node.lineno, _SQL_COMMENT.sub(" ", value)


def _offending_references(extra: tuple[pathlib.Path, ...] = ()) -> list[str]:
    retired = _retired_columns()
    assert retired, "no retired columns parsed; the migration tuples moved"

    any_retired = re.compile(r"\b(" + "|".join(sorted(map(re.escape, retired))) + r")\b")

    findings: list[str] = []
    for path in [*_python_sources(), *extra]:
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if not any_retired.search(text):
            continue
        try:
            shown = path.relative_to(REPO)
        except ValueError:  # a probe from outside the repo
            shown = path
        for lineno, sql in _sql_literals(text):
            aliased = {n.lower() for n in _ALIAS_NAMES.findall(sql)}
            scanned = _SQL_STRING.sub(" '' ", sql)
            for match in any_retired.finditer(scanned):
                column = match.group(1)
                if column.lower() in aliased:
                    continue
                if _ALIAS.search(scanned[: match.start()]):
                    continue
                # Adjacent literals fold into one constant, so a statement and
                # its table name are almost always in the same string.
                tables = sorted(t for t in retired[column] if re.search(rf"\b{t}\b", sql))
                if not tables:
                    continue
                findings.append(
                    f"{shown}:{lineno} references {column!r}, dropped from "
                    f"{tables}: {' '.join(sql.split())[:110]}"
                )
                break
    return findings


def test_no_live_sql_references_a_dropped_column():
    offenders = _offending_references()
    assert not offenders, (
        "live SQL references columns that no longer exist. These fail at "
        "runtime, and where the query is wrapped in `except Exception` they "
        "fail silently as an empty result:\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_reintroduced_reference(tmp_path):
    """The guard above passes trivially if the scan is broken; prove it isn't.

    This is the exact query `routes/admin.py` shipped with — the one that
    returned a $0.00 balance for every user for as long as it ran.
    """
    planted = tmp_path / "probe.py"
    planted.write_text(
        "def probe(conn):\n"
        '    return conn.execute("SELECT customer_id, balance_cad FROM wallets").fetchall()\n'
    )
    offenders = _offending_references(extra=(planted,))
    assert any("probe.py" in o for o in offenders), (
        "the scan did not flag a planted reference to wallets.balance_cad, so "
        "a green result from the test above means nothing"
    )


def test_retired_set_covers_the_known_drops():
    """Canary against the migration tuples being renamed or moved."""
    retired = _retired_columns()
    for column, table in (
        ("balance_cad", "wallets"),
        ("total_cost_cad", "usage_meters"),
        ("amount_cad", "wallet_transactions"),
    ):
        assert table in retired.get(column, set()), (
            f"{table}.{column} is no longer in the retired set; 085/087 "
            f"module-level tuples have probably been renamed"
        )
    if not retired:
        pytest.fail("retired set is empty")
