"""Schema-revision compatibility contract (blueprint §13.8 / ADR-009).

Every control-plane service declares the Alembic revision range it can
run against. Readiness (`/readyz`) must call
:func:`assert_schema_compatible` and fail — refusing traffic — when the
database is outside that range, instead of limping along against a schema
it does not understand.

Revisions in this repository are zero-padded numeric strings
(``"001"``…``"057"``), so ordering is numeric. A non-numeric revision
(e.g. an unexpected branch) is treated as incompatible, never silently
accepted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

from psycopg import Connection

# The minimum schema this code requires: migration 057 completed the
# Track A expand set (attempts/allocations/leases/commands/outbox/
# observations). Raise this only alongside code that needs the newer
# schema; the maximum is open-ended until a breaking contract migration
# (060) defines one.
REQUIRED_MIN_REVISION = "057"
REQUIRED_MAX_REVISION: str | None = None


class SchemaIncompatibleError(RuntimeError):
    """Database revision is outside this service's supported range."""


@dataclass(frozen=True)
class SchemaCompat:
    current: str | None
    minimum: str
    maximum: str | None
    compatible: bool
    reason: str


def _revision_ord(revision: str) -> int | None:
    try:
        return int(revision, 10)
    except (TypeError, ValueError):
        return None


def configured_range() -> tuple[str, str | None]:
    """Supported range, env-overridable per §30 for staged deploys."""
    minimum = os.environ.get("XCELSIOR_DB_SCHEMA_MIN_REVISION", REQUIRED_MIN_REVISION)
    maximum = os.environ.get("XCELSIOR_DB_SCHEMA_MAX_REVISION") or REQUIRED_MAX_REVISION
    return minimum, maximum


def current_revision(conn: Connection) -> str | None:
    """The database's Alembic head, or None if migrations never ran."""
    exists = conn.execute("SELECT to_regclass('alembic_version')").fetchone()
    if exists is None or exists[0] is None:
        return None
    row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    if row is None:
        return None
    # Pool row_factory may yield tuples or dicts; normalize either shape.
    if isinstance(row, dict):
        return str(cast("dict[str, object]", row)["version_num"])
    return str(row[0])


def check_schema_compatible(conn: Connection) -> SchemaCompat:
    minimum, maximum = configured_range()
    current = current_revision(conn)
    if current is None:
        return SchemaCompat(
            current, minimum, maximum, False,
            "alembic_version missing or empty — migrations have not run",
        )
    cur_ord = _revision_ord(current)
    min_ord = _revision_ord(minimum)
    if cur_ord is None or min_ord is None:
        return SchemaCompat(
            current, minimum, maximum, False,
            f"non-numeric revision (current={current!r}, min={minimum!r}) — "
            "unknown migration branch",
        )
    if cur_ord < min_ord:
        return SchemaCompat(
            current, minimum, maximum, False,
            f"database at {current}, service requires >= {minimum} — "
            "run alembic upgrade before deploying this build",
        )
    if maximum is not None:
        max_ord = _revision_ord(maximum)
        if max_ord is None or cur_ord > max_ord:
            return SchemaCompat(
                current, minimum, maximum, False,
                f"database at {current} exceeds supported maximum {maximum} — "
                "deploy newer service binaries first",
            )
    return SchemaCompat(current, minimum, maximum, True, "compatible")


def assert_schema_compatible(conn: Connection) -> SchemaCompat:
    """Readiness-check form: raise on incompatibility, return details."""
    compat = check_schema_compatible(conn)
    if not compat.compatible:
        raise SchemaIncompatibleError(compat.reason)
    return compat
