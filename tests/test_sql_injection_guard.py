"""Static-analysis guard against SQL injection via dynamic identifier interpolation.

psycopg parameterization (%s) only protects *values*, never identifiers
(table names, column names, ORDER BY clauses, JSONB paths). Any f-string or
.format() that injects a Python variable into those positions is a potential
SQL injection unless the variable is allowlisted upstream.

Every match below has been manually audited and confirmed safe. If this test
fails, a new occurrence was introduced — either:
  1. Add it to KNOWN_SAFE after verifying the variable is allowlisted, OR
  2. Rewrite to use psycopg.sql.Identifier / literal_column / %s parameters.

This keeps SQL injection hygiene as a one-liner regression check rather than
a quarterly audit.
"""

from __future__ import annotations

import re
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Files audited and verified safe. Each entry is (relative_path, variable_names_allowlist_source).
KNOWN_SAFE = {
    # f"ORDER BY {order}" — order is mapped through _SORT_MAP allowlist.
    "marketplace.py",
    # f"... SET {set_clause} ..." — keys come from ALLOWED_FIELDS set.
    "db.py",
    # f"... WHERE {where} ..." — where_parts are static literals, values via %s.
    "events.py",
    # f"... WHERE {where_clause} ..." / {group_sql} — static parts or _GROUP_SQL map.
    "routes/admin.py",
    "routes/billing.py",
    # f"... {placeholders}" — %s count for IN clause (no identifiers).
    "ai_assistant.py",
    # Dev-only migration script — reads table names from a hardcoded TABLES list.
    "scripts/migrate_sqlite_to_pg.py",
    # Pre-split legacy file, kept only for regression tests (not served). Shares
    # the same allowlist pattern as routes/billing.py.
    "api_old.py",
}

# SQL keywords must be UPPERCASE — this is the codebase convention for all real
# SQL, and it prevents false positives from Python `from X import Y` statements
# and prose that happens to contain words like "from" or "set".
SCAN_PATTERNS = [
    # f-string with UPPERCASE SQL keyword + brace-interpolated Python variable.
    re.compile(
        r"""f["'][^"']*\b(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|ORDER\s+BY|GROUP\s+BY|JOIN|SET)\b[^"']*\{[a-zA-Z_][\w]*\}""",
    ),
    # .format() called directly on an UPPERCASE-SQL string literal.
    re.compile(
        r"""["'][^"']*\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^"']*["']\s*\.\s*format\(""",
    ),
    # execute("..." + variable) style concatenation.
    re.compile(
        r"""\bexecute\w*\s*\(\s*["'][^"']*["']\s*\+\s*[a-zA-Z_]""",
    ),
]


def _source_files():
    for path in REPO.rglob("*.py"):
        parts = set(path.parts)
        # Exclude venv, caches, third-party, tests themselves.
        if any(
            p in parts
            for p in (
                "venv",
                ".venv",
                "env",
                "__pycache__",
                "site-packages",
                "tests",
                ".hypothesis",
            )
        ):
            continue
        yield path


def test_no_new_dynamic_sql_identifier_interpolation():
    """Fail if a new file introduces dynamic SQL identifier interpolation.

    Adding a match in a file already in KNOWN_SAFE is allowed (a human already
    vetted that file's interpolations). Adding such a match to a new file is
    a finding — either audit it and extend KNOWN_SAFE, or rewrite the query.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}

    for path in _source_files():
        rel = str(path.relative_to(REPO))
        if rel in KNOWN_SAFE:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in SCAN_PATTERNS:
                if pat.search(line):
                    offenders.setdefault(rel, []).append((lineno, line.strip()[:200]))
                    break

    if offenders:
        msg = ["New dynamic-SQL interpolation detected (possible SQL injection):"]
        for rel, hits in offenders.items():
            msg.append(f"  {rel}:")
            for lineno, line in hits:
                msg.append(f"    L{lineno}: {line}")
        msg.append("")
        msg.append("Fix options:")
        msg.append(
            "  1. Rewrite the query to use psycopg %s placeholders or psycopg.sql.Identifier()."
        )
        msg.append(
            "  2. If the interpolated variable is rigorously allowlisted, audit this test and"
        )
        msg.append("     add the file to KNOWN_SAFE with a comment naming the allowlist source.")
        raise AssertionError("\n".join(msg))


def test_known_safe_files_still_exist():
    """Every KNOWN_SAFE entry must still be a real file."""
    for rel in KNOWN_SAFE:
        assert (REPO / rel).exists(), f"KNOWN_SAFE references missing file: {rel}"
