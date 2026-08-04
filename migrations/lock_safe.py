"""Apply per-table work without holding every lock for the whole migration.

Deploys are blue-green (`scripts/deploy.sh` runs `alembic upgrade head` while
the live API, scheduler and background workers keep serving), so a migration
here competes with application traffic by design. Two properties follow from
that, and neither is optional:

1. **A migration must not accumulate `ACCESS EXCLUSIVE` locks across many
   tables inside one transaction.** Application transactions touch those same
   tables in their own order, so the moment a migration holds table A and wants
   table B while a request holds B and wants A, PostgreSQL kills one of them.
   That is not a hypothetical: migration `095` died exactly this way on the
   2026-08-04 deploy, on `ALTER TABLE jobs ADD COLUMN spot_rate_micros`, after
   it had already locked fourteen other tables in the same transaction.

2. **A blocked `ALTER TABLE` must give up rather than wait.** A lock request
   queues *ahead* of every later request on that table, so a migration waiting
   behind one long-running read stalls all traffic to the table for as long as
   it waits. `lock_timeout` converts that outage into a retry.

This module is what `migrations/README.md` rule 5 ("`lock_timeout` set") means
in practice: each unit of work runs in its own transaction, on its own
connection, with a short `lock_timeout`, retried on the two SQLSTATEs that mean
*someone else held the lock* rather than *this statement is wrong*.

Because each unit commits independently, a migration using this module is
**resumable, not atomic**: if the process dies half way, the units that
committed stay committed and `alembic_version` is not advanced, so the next
`alembic upgrade` re-runs the whole migration. Every unit must therefore be
idempotent — `ADD COLUMN IF NOT EXISTS`, `CREATE OR REPLACE`, `DROP ... IF
EXISTS`, and backfills predicated on the target still being NULL. That is a
condition on the caller, and it is why the trade is worth making: an
unresumable migration that deadlocks is strictly worse than a resumable one
that does not.

Requires `transaction_per_migration=True` in `migrations/env.py`. Without it
Alembic holds one transaction open across the whole `upgrade`, an earlier
migration's locks are still held on the Alembic connection, and this module's
separate connection would block against its own run. That precondition is
checked rather than assumed — see `_refuse_if_alembic_holds_locks`.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence

from alembic import op
from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

log = logging.getLogger("alembic.lock_safe")

#: `deadlock_detected` and `lock_not_available`. Both mean the statement was
#: fine and the contention was not; anything else is a real error and is
#: re-raised so a broken migration still fails loudly.
RETRYABLE_SQLSTATES = frozenset({"40P01", "55P03"})

#: Long enough that an ordinary short query does not cause a retry, short
#: enough that traffic to the table is not stalled while we wait.
DEFAULT_LOCK_TIMEOUT = os.environ.get("XCELSIOR_MIGRATION_LOCK_TIMEOUT", "5s")

#: With exponential backoff capped at 8s this waits ~40s in the worst case per
#: unit before giving up and failing the deploy, which is the right outcome:
#: something is holding a long transaction and a human should look.
DEFAULT_ATTEMPTS = int(os.environ.get("XCELSIOR_MIGRATION_LOCK_ATTEMPTS", "10"))

Unit = tuple[str, Callable[[Connection], None]]

#: `SET` takes no bind parameters, so the value is interpolated and therefore
#: validated first. A PostgreSQL interval literal is all that is ever wanted
#: here, and anything else is a configuration mistake worth failing on.
_TIMEOUT_RE = re.compile(r"^\d+(ms|s|min)?$")


def checked_timeout(value: str) -> str:
    if not _TIMEOUT_RE.match(value.strip()):
        raise ValueError(
            f"lock_timeout must look like '5s', '500ms' or '2min', got {value!r}"
        )
    return value.strip()


def _sqlstate(exc: BaseException) -> str | None:
    orig = getattr(exc, "orig", None)
    for attr in ("sqlstate", "pgcode"):
        code = getattr(orig, attr, None)
        if code:
            return str(code)
    return None


def _refuse_if_alembic_holds_locks(bind: Connection, tables: Sequence[str]) -> None:
    """Fail loudly if Alembic's own transaction has locked our tables.

    A separate connection cannot take `ACCESS EXCLUSIVE` on a table that the
    Alembic connection already holds any lock on — it would wait for a
    transaction that cannot commit until we return, i.e. block forever (or
    until `lock_timeout`, presenting as a mysterious timeout rather than the
    configuration error it is). The only way that happens is a runner sharing
    one transaction across migrations, so name that in the message.
    """
    if not tables:
        return
    held = (
        bind.execute(
            text(
                """
                SELECT DISTINCT c.relname
                FROM pg_locks l
                JOIN pg_class c ON c.oid = l.relation
                WHERE l.pid = pg_backend_pid()
                  AND l.locktype = 'relation'
                  AND c.relname = ANY(:names)
                """
            ),
            {"names": list(tables)},
        )
        .scalars()
        .all()
    )
    if held:
        raise RuntimeError(
            "Alembic's transaction already holds locks on "
            f"{sorted(held)}, so per-table transactions would block against "
            "this same migration run. Set transaction_per_migration=True in "
            "migrations/env.py (see migrations/lock_safe.py)."
        )


def _run_unit(
    conn: Connection,
    label: str,
    apply: Callable[[Connection], None],
    lock_timeout: str,
    attempts: int,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            # SET LOCAL, so it is scoped to this unit's transaction and cannot
            # leak a short timeout onto anything that follows.
            conn.execute(text(f"SET LOCAL lock_timeout = '{lock_timeout}'"))
            apply(conn)
            conn.commit()
            return
        except DBAPIError as exc:
            conn.rollback()
            state = _sqlstate(exc)
            if state not in RETRYABLE_SQLSTATES or attempt == attempts:
                raise
            delay = min(8.0, 0.5 * 2 ** (attempt - 1))
            log.warning(
                "lock_safe: %s hit %s (attempt %d/%d), retrying in %.1fs",
                label,
                state,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)


def apply_in_own_transactions(
    units: Iterable[Unit],
    *,
    tables: Sequence[str] = (),
    lock_timeout: str | None = None,
    attempts: int | None = None,
    bind: Connection | None = None,
) -> None:
    """Run each `(label, apply)` unit in its own transaction and connection.

    `tables` is every table the units touch; it is used only for the
    precondition check above, so an over-broad list is harmless and a missing
    entry only costs the clear error message.

    `bind` defaults to Alembic's connection, which is what a migration wants.
    It is a parameter so the contention behaviour can be tested against real
    PostgreSQL without an Alembic runtime — see
    `tests/test_migration_lock_discipline.py`.
    """
    units = list(units)
    if not units:
        return
    bind = bind if bind is not None else op.get_bind()
    _refuse_if_alembic_holds_locks(bind, tables)

    lock_timeout = checked_timeout(lock_timeout or DEFAULT_LOCK_TIMEOUT)
    attempts = attempts or DEFAULT_ATTEMPTS

    # A distinct connection, deliberately: work committed per unit is the whole
    # point, and Alembic owns the transaction on its own connection.
    engine = create_engine(bind.engine.url, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            for label, apply in units:
                _run_unit(conn, label, apply, lock_timeout, attempts)
    finally:
        engine.dispose()
