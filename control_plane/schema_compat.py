"""Schema-revision compatibility contract (blueprint §13.8 / ADR-009).

Every control-plane service declares the Alembic revision range it can
run against. Readiness (`/readyz`) must call
:func:`assert_schema_compatible` and fail — refusing traffic — when the
database is outside that range, instead of limping along against a schema
it does not understand.

Most revisions are zero-padded numeric strings (``"001"``…``"061"``).
Non-numeric ids (e.g. hash revisions in the chain) are ordered via the
Alembic script graph so a legitimate head is never rejected as an
"unknown branch".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from psycopg import Connection

# The minimum schema this code requires: migration 057 completed the
# Track A expand set (attempts/allocations/leases/commands/outbox/
# observations). Raise this only alongside code that needs the newer
# schema; the maximum is open-ended until a breaking contract migration
# defines one.
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
    """Numeric order for zero-padded revision ids; None for hash ids."""
    try:
        return int(revision, 10)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _alembic_script():
    """Load the project's Alembic ScriptDirectory (once per process)."""
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
    except ImportError:  # pragma: no cover
        return None
    root = Path(__file__).resolve().parent.parent
    ini = root / "alembic.ini"
    if not ini.is_file():
        return None
    try:
        return ScriptDirectory.from_config(Config(str(ini)))
    except Exception:  # pragma: no cover
        return None


def _revision_at_least(current: str, minimum: str) -> bool | None:
    """True if ``current`` is ``minimum`` or a descendant in the Alembic graph.

    Returns None when the graph cannot be consulted (missing alembic
    package / config) so the caller can fall back to numeric-only logic.
    """
    if current == minimum:
        return True
    script = _alembic_script()
    if script is None:
        return None
    try:
        # walk from current toward base; if we see minimum, current >= min.
        for rev in script.walk_revisions(base="base", head=current):
            if rev.revision == minimum:
                return True
        return False
    except Exception:
        return None


def _revision_at_most(current: str, maximum: str) -> bool | None:
    """True if ``current`` is ``maximum`` or an ancestor of ``maximum``."""
    if current == maximum:
        return True
    script = _alembic_script()
    if script is None:
        return None
    try:
        for rev in script.walk_revisions(base="base", head=maximum):
            if rev.revision == current:
                return True
        return False
    except Exception:
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

    # Prefer numeric compare when both sides are numeric (fast path).
    if cur_ord is not None and min_ord is not None:
        if cur_ord < min_ord:
            return SchemaCompat(
                current, minimum, maximum, False,
                f"database at {current}, service requires >= {minimum} — "
                "run alembic upgrade before deploying this build",
            )
    else:
        # Hash / mixed revisions: walk the Alembic graph.
        at_least = _revision_at_least(current, minimum)
        if at_least is False:
            return SchemaCompat(
                current, minimum, maximum, False,
                f"database at {current}, service requires >= {minimum} — "
                "run alembic upgrade before deploying this build",
            )
        if at_least is None and (cur_ord is None or min_ord is None):
            return SchemaCompat(
                current, minimum, maximum, False,
                f"non-numeric revision (current={current!r}, min={minimum!r}) "
                "and Alembic graph unavailable — unknown migration branch",
            )

    if maximum is not None:
        max_ord = _revision_ord(maximum)
        if cur_ord is not None and max_ord is not None:
            if cur_ord > max_ord:
                return SchemaCompat(
                    current, minimum, maximum, False,
                    f"database at {current} exceeds supported maximum {maximum} — "
                    "deploy newer service binaries first",
                )
        else:
            at_most = _revision_at_most(current, maximum)
            if at_most is False:
                return SchemaCompat(
                    current, minimum, maximum, False,
                    f"database at {current} exceeds supported maximum {maximum} — "
                    "deploy newer service binaries first",
                )
            if at_most is None and (cur_ord is None or max_ord is None):
                return SchemaCompat(
                    current, minimum, maximum, False,
                    f"non-numeric revision (current={current!r}, max={maximum!r}) "
                    "and Alembic graph unavailable — unknown migration branch",
                )

    return SchemaCompat(current, minimum, maximum, True, "compatible")


def assert_schema_compatible(conn: Connection) -> SchemaCompat:
    """Readiness-check form: raise on incompatibility, return details."""
    compat = check_schema_compatible(conn)
    if not compat.compatible:
        raise SchemaIncompatibleError(compat.reason)
    return compat
