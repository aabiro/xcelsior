"""Transaction helpers for lock-sensitive control-plane operations.

Blueprint §2.4 / §23.1: the placement, lease, command, and outbox
transactions coordinate multiple replicas through PostgreSQL row locks.
These helpers give every such transaction the same hardened envelope:

- per-transaction ``SET LOCAL`` statement/lock timeouts, so a stuck lock
  fails fast instead of queueing invisibly (and the settings can never
  leak back into the shared pool — SET LOCAL dies with the transaction);
- bounded, jittered retry of *known transient* SQLSTATEs only
  (serialization failure ``40001``, deadlock ``40P01``) and pre-commit
  connection failures — never validation, policy, or constraint errors;
- explicit ambiguity handling: an error raised *while committing* means
  the commit may or may not have landed. That is never retried blindly —
  it surfaces as :class:`AmbiguousCommitError` so the caller resolves it
  through its idempotency key.

Also provides the transaction-scoped advisory-lock helper mandated by
§2.5 (stable key derivation, never Python's process-randomized ``hash``)
that replaces the leaky pooled session locks in ``serverless/repo.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import psycopg
from psycopg import Connection
from psycopg.errors import DeadlockDetected, SerializationFailure

log = logging.getLogger("xcelsior.control_plane.db")

T = TypeVar("T")

# SQLSTATEs that are safe to retry with a fresh transaction (§2.4).
TRANSIENT_SQLSTATES = frozenset({"40001", "40P01"})

_DEFAULT_STATEMENT_TIMEOUT_MS = int(
    os.environ.get("XCELSIOR_PG_STATEMENT_TIMEOUT_MS", "30000")
)
_DEFAULT_LOCK_TIMEOUT_MS = int(os.environ.get("XCELSIOR_PG_LOCK_TIMEOUT_MS", "5000"))


class RetryBudgetExceeded(Exception):
    """A transient conflict persisted past the bounded retry budget.

    Carries the final attempt's error as ``__cause__``. Callers treat this
    as a retryable-later condition (e.g. requeue the scheduling claim),
    not a bug.
    """


class AmbiguousCommitError(Exception):
    """The COMMIT itself failed in a way that may still have applied.

    Retrying the transaction could double-apply its effects, so this is
    never retried here. Callers must resolve through their idempotency
    record (repeat the request; the idempotency key returns the original
    outcome either way).
    """


def is_transient_error(exc: BaseException) -> bool:
    """True only for errors that are provably safe to retry pre-commit."""
    if isinstance(exc, (SerializationFailure, DeadlockDetected)):
        return True
    if isinstance(exc, psycopg.Error):
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate in TRANSIENT_SQLSTATES:
            return True
        # Connection lost before commit: transaction definitely rolled
        # back server-side, safe to retry on a fresh connection.
        if isinstance(exc, psycopg.OperationalError) and sqlstate is None:
            return True
    return False


def _pool():
    # Reuse the application's shared psycopg pool (db.py owns sizing and
    # DSN resolution). Imported lazily so importing this module never
    # forces a connection.
    from db import _get_pg_pool

    return _get_pg_pool()


@contextmanager
def control_plane_transaction(
    *,
    statement_timeout_ms: int | None = None,
    lock_timeout_ms: int | None = None,
) -> Iterator[Connection]:
    """One explicit transaction on a pooled connection, with local timeouts.

    Commits on clean exit, rolls back on any exception. A failure during
    the COMMIT itself is re-raised as :class:`AmbiguousCommitError` —
    see the class docstring for why that must not be retried.
    """
    stmt_ms = _DEFAULT_STATEMENT_TIMEOUT_MS if statement_timeout_ms is None else statement_timeout_ms
    lock_ms = _DEFAULT_LOCK_TIMEOUT_MS if lock_timeout_ms is None else lock_timeout_ms
    with _pool().connection() as conn:
        try:
            conn.execute(f"SET LOCAL statement_timeout = {int(stmt_ms)}")
            conn.execute(f"SET LOCAL lock_timeout = {int(lock_ms)}")
            yield conn
        except BaseException:
            conn.rollback()
            raise
        try:
            conn.commit()
        except psycopg.OperationalError as exc:
            # The commit message may have reached the server before the
            # connection died — outcome unknown.
            raise AmbiguousCommitError(
                "connection failed during COMMIT; transaction outcome unknown"
            ) from exc


def run_transaction(
    fn: Callable[[Connection], T],
    *,
    max_attempts: int = 5,
    base_backoff_ms: float = 20.0,
    max_backoff_ms: float = 1000.0,
    statement_timeout_ms: int | None = None,
    lock_timeout_ms: int | None = None,
    what: str = "control_plane_txn",
) -> T:
    """Run ``fn(conn)`` in a transaction, retrying transient conflicts.

    Each attempt gets a fresh transaction (and, after a connection error,
    a fresh pooled connection). Backoff is full-jitter exponential and
    strictly bounded; non-transient errors and :class:`AmbiguousCommitError`
    propagate immediately on the first occurrence.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    attempt = 0
    while True:
        attempt += 1
        try:
            with control_plane_transaction(
                statement_timeout_ms=statement_timeout_ms,
                lock_timeout_ms=lock_timeout_ms,
            ) as conn:
                return fn(conn)
        except AmbiguousCommitError:
            raise
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            if attempt >= max_attempts:
                raise RetryBudgetExceeded(
                    f"{what}: transient conflict persisted through "
                    f"{max_attempts} attempts"
                ) from exc
            # Full jitter: uniform over [0, min(cap, base * 2^(n-1))].
            ceiling_ms = min(max_backoff_ms, base_backoff_ms * (2 ** (attempt - 1)))
            delay_s = random.uniform(0, ceiling_ms) / 1000.0
            log.warning(
                "%s: transient %s on attempt %d/%d; retrying in %.0f ms",
                what,
                getattr(exc, "sqlstate", None) or type(exc).__name__,
                attempt,
                max_attempts,
                delay_s * 1000,
            )
            time.sleep(delay_s)


def stable_advisory_key(namespace: str, resource_id: str) -> int:
    """Derive the signed 64-bit advisory-lock key for a resource.

    §2.5: the key must be a documented, stable mapping — identical across
    processes, hosts, and Python releases (``hash()`` is process-seeded
    and forbidden here). Uses the first 8 bytes of
    ``sha256(namespace ":" resource_id)`` reinterpreted as a signed
    big-endian 64-bit integer, which is what ``pg_try_advisory_xact_lock``
    expects.
    """
    digest = hashlib.sha256(f"{namespace}:{resource_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def try_advisory_xact_lock(conn: Connection, namespace: str, resource_id: str) -> bool:
    """Take a transaction-scoped advisory lock; never blocks.

    Returns False if another transaction holds the resource. The lock
    releases automatically at commit/rollback on *this* connection, so it
    is safe with pooled connections and PgBouncer transaction mode —
    unlike session-level ``pg_try_advisory_lock`` on a pool.
    """
    row = conn.execute(
        "SELECT pg_try_advisory_xact_lock(%s)",
        (stable_advisory_key(namespace, resource_id),),
    ).fetchone()
    if row is None:  # pg_try_advisory_xact_lock always returns one row
        raise RuntimeError("pg_try_advisory_xact_lock returned no row")
    # psycopg may return tuple or dict rows depending on the pool's
    # row_factory contamination; handle both like the rest of the codebase.
    value = row[0] if isinstance(row, (tuple, list)) else next(iter(dict(row).values()))
    return bool(value)
