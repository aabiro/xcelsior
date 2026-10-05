"""The compatibility floor must cover every column the application writes.

`control_plane.schema_compat.REQUIRED_MIN_REVISION` is what `/readyz` refuses to
serve below. It is the one thing standing between a partially-migrated database
and a service that starts, reports healthy, and then fails every query naming a
column the schema does not have yet.

It said `"057"` for 59 revisions after the code moved past it. The constant's own
comment says to raise it alongside code that needs a newer schema, and that was
simply never done — so the gate certified as compatible every schema from 057
upward, including ones missing the `_micros` columns added by `095` and
`wallets.low_balance_warned_at` added by `116`. A database at 079 passed, and
then `UndefinedColumn` was raised on every money path and swallowed by the
surrounding `except Exception`: no `billing_cycles` row written, so each job
re-billed from its start every five minutes; the GST threshold computed from
`0.0`; SLA credits never issued.

So this derives the floor instead of trusting it. For every migration, take the
columns it adds — `ADD COLUMN`, `op.add_column(...)`, and the `_cad`→`_micros`
pairs `095` generates from its `COLUMNS` list — and look for them **inside SQL
string literals** in application code. Matching on SQL literals rather than
anywhere in the file is what keeps `attempt_id` and `tenant_id` from counting
every time they appear as a local variable.

The highest migration with a hit is the lowest schema this code can run on.
`REQUIRED_MIN_REVISION` must be at least that.

It is deliberately not asserted *equal* to head. `117` drops redundant indexes,
which nothing reads, and pinning the floor to head would force a bump for every
migration whether or not the code depends on it — which is how a floor stops
meaning anything.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess

from control_plane.schema_compat import REQUIRED_MIN_REVISION

REPO = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS = REPO / "migrations" / "versions"

_ADD_COLUMN_DDL = re.compile(r"ADD COLUMN(?:\s+IF NOT EXISTS)?\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_ADD_COLUMN_OP = re.compile(
    r"add_column\(\s*[\"']([a-z_0-9]+)[\"']\s*,\s*sa\.Column\(\s*[\"']([a-z_0-9]+)[\"']",
    re.IGNORECASE,
)
#: `("billing_cycles", "amount_cad")` in 095's COLUMNS list — the migration
#: creates the `_micros` twin of each entry in a loop, so the name never appears
#: literally in the file.
_CAD_PAIR = re.compile(r"\(\"([a-z_]+)\",\s*\"([a-z_]+_cad)\"\)")
_SQL_VERB = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|RETURNING)\b", re.IGNORECASE)


def _application_sql() -> str:
    """Every SQL string literal in tracked application code, concatenated."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")

    chunks: list[str] = []
    for rel in listed:
        if not rel or rel.startswith(("tests/", "migrations/", "scripts/")):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                text = "".join(
                    v.value
                    for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
            if text and _SQL_VERB.search(text):
                chunks.append(text)
    return "\n".join(chunks)


def _columns_added_by(source: str) -> set[str]:
    columns = set(_ADD_COLUMN_DDL.findall(source))
    columns |= {column for _table, column in _ADD_COLUMN_OP.findall(source)}
    columns |= {column.replace("_cad", "_micros") for _table, column in _CAD_PAIR.findall(source)}
    return columns


def _required_floor() -> tuple[str, str, list[str]]:
    """(revision, filename, the columns that make it required)."""
    sql = _application_sql()
    assert sql, "no SQL literals found in application code — the scan is broken"

    highest = ("000", "", [])
    for path in sorted(MIGRATIONS.glob("*.py")):
        revision = path.name.split("_", 1)[0]
        if not revision.isdigit():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        used = sorted(
            column
            for column in _columns_added_by(source)
            if re.search(rf"\b{re.escape(column)}\b", sql)
        )
        if used and revision > highest[0]:
            highest = (revision, path.name, used)
    return highest


def test_the_floor_is_not_below_what_the_code_writes():
    revision, filename, columns = _required_floor()
    assert REQUIRED_MIN_REVISION >= revision, (
        f"REQUIRED_MIN_REVISION is {REQUIRED_MIN_REVISION!r}, but application SQL "
        f"names {', '.join(columns)} — added by {filename}. A database between "
        f"{REQUIRED_MIN_REVISION} and {revision} passes the readiness check and "
        "then raises UndefinedColumn on those queries, which the handlers around "
        "them swallow. Raise the floor in the same commit as the code that needs it."
    )


def test_the_floor_names_a_migration_that_exists():
    """A floor pointing at nothing would pass the check above vacuously."""
    revisions = {
        p.name.split("_", 1)[0]
        for p in MIGRATIONS.glob("*.py")
        if p.name.split("_", 1)[0].isdigit()
    }
    assert REQUIRED_MIN_REVISION in revisions, (
        f"REQUIRED_MIN_REVISION={REQUIRED_MIN_REVISION!r} is not a migration in "
        "migrations/versions/"
    )
