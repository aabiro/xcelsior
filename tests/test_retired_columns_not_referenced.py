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
        pass          # module-level tuples we want are already bound.
    return module


def _retired_columns() -> dict[str, set[str]]:
    """{column_name: {tables it was dropped from}}, from the migrations."""
    retired: dict[str, set[str]] = {}

    dead = _load(MIGRATIONS / "085_drop_dead_columns.py")
    for entry in getattr(dead, "DEAD_COLUMNS", ()):
        table, column = entry[0], entry[1]
        retired.setdefault(column, set()).add(table)

    derived = _load(MIGRATIONS / "087_drop_derived_cad_columns.py")
    for table, _trigger, _fn, columns in getattr(derived, "DERIVED", ()):
        for column in columns:
            retired.setdefault(column, set()).add(table)

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
        capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    return [
        REPO / rel
        for rel in listed
        if rel and not rel.startswith(SKIP_PREFIXES) and (REPO / rel).is_file()
    ]


# `... AS balance_cad` inside a query defines an alias; the column it derives
# from is a live one. That is a definition, not a reference.
_ALIAS = re.compile(r"\bAS\s+$", re.I)
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
            for match in any_retired.finditer(sql):
                column = match.group(1)
                if _ALIAS.search(sql[: match.start()]):
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
