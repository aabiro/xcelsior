"""Per-item isolation for the scheduler's maintenance sweeps.

The scheduler tick runs its lease, preemption and host-eviction sweeps before
placement. Each sweep walks its rows in a deterministic order, so one row that
fails deterministically (a held billing-period lock, a plan that no longer
validates) used to fail first on every tick, roll back the whole sweep and,
because the tick did not isolate its sweeps, stop all placement with it.

``SweepQuarantine.run`` applies one item inside a savepoint. A failure rolls
back only that item, is logged with the item's identity, and puts the item in
an in-process exponential back-off so the rest of the batch (and later
batches) keep moving. Back-off is per replica and deliberately not durable:
it only decides how soon this process tries the row again, never whether the
row's state transition is allowed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

log = logging.getLogger("xcelsior.control_plane.sweeps")


class SweepQuarantine:
    """Savepoint-isolated item execution with bounded per-item back-off."""

    def __init__(
        self,
        name: str,
        *,
        base_backoff_sec: float = 5.0,
        max_backoff_sec: float = 300.0,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, int]] = {}

    def skipped(self) -> list[str]:
        """Item ids still backing off; callers exclude them from their batch."""
        now = self._clock()
        with self._lock:
            return sorted(key for key, (until, _) in self._entries.items() if until > now)

    def failures(self, item_id: str) -> int:
        with self._lock:
            return self._entries.get(str(item_id), (0.0, 0))[1]

    def run(self, conn: Any, item_id: str, fn: Callable[[], T]) -> tuple[bool, T | None]:
        """Apply ``fn`` in a savepoint; (True, result) or (False, None) on failure."""
        item_id = str(item_id)
        try:
            with conn.transaction():
                result = fn()
        except Exception as exc:
            if getattr(conn, "closed", False) or getattr(conn, "broken", False):
                raise
            self._record_failure(item_id, exc)
            return False, None
        with self._lock:
            self._entries.pop(item_id, None)
        return True, result

    def _record_failure(self, item_id: str, exc: BaseException) -> None:
        now = self._clock()
        with self._lock:
            failures = self._entries.get(item_id, (0.0, 0))[1] + 1
            delay = min(self.max_backoff_sec, self.base_backoff_sec * (2 ** (failures - 1)))
            self._entries[item_id] = (now + delay, failures)
            if len(self._entries) > self.max_entries:
                # Forget the entries closest to retrying; never grow without bound.
                for key, _ in sorted(self._entries.items(), key=lambda kv: kv[1][0])[
                    : len(self._entries) - self.max_entries
                ]:
                    self._entries.pop(key, None)
        log.warning(
            "%s sweep: item %s failed (%s: %s); quarantined for %.0fs after %d failure(s)",
            self.name,
            item_id,
            type(exc).__name__,
            str(exc)[:300],
            delay,
            failures,
        )
