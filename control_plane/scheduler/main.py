"""Standalone shadow-scheduler replica (blueprint Phase 3).

``python -m control_plane.scheduler.main`` runs the shadow runner as its
own process/container — the deployment shape the Phase 4 authoritative
scheduler will inherit. The in-process alternative is
``scheduler.start_shadow_runner()``, which the legacy ``scheduler_main``
loop starts as a daemon thread when ``XCELSIOR_SCHEDULER_MODE=shadow``.

Refuses to start when the database lacks the shadow-decision table
(migration 058) rather than failing every cycle.
"""

from __future__ import annotations

import logging
import time

from control_plane.db import run_transaction
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.shadow import ShadowRunner

log = logging.getLogger("xcelsior.control_plane.scheduler.main")


def shadow_schema_ready() -> bool:
    """True when migration 058's decision table exists."""
    row = run_transaction(
        lambda conn: conn.execute(
            "SELECT to_regclass('scheduler_shadow_decisions')"
        ).fetchone(),
        what="shadow_schema_check",
    )
    if row is None:
        return False
    value = row[0] if not isinstance(row, dict) else next(iter(row.values()))
    return value is not None


def run_shadow_loop(config: SchedulerConfig | None = None) -> None:
    """Blocking shadow loop; one cycle per interval, failures isolated."""
    cfg = config or SchedulerConfig.from_env()
    if cfg.mode is not SchedulerMode.SHADOW:
        log.info(
            "scheduler mode is %s — shadow runner not starting", cfg.mode.value
        )
        return
    if not shadow_schema_ready():
        raise RuntimeError(
            "scheduler_shadow_decisions table missing — run alembic upgrade "
            "to migration 058 before enabling shadow mode"
        )
    runner = ShadowRunner(cfg)
    log.info(
        "shadow scheduler replica %s: interval=%ds grace=%ds retention=%dd",
        cfg.replica_id,
        cfg.shadow_interval_sec,
        cfg.shadow_compare_grace_sec,
        cfg.shadow_retention_days,
    )
    while True:
        try:
            runner.run_once()
        except Exception:
            log.exception("shadow cycle failed; continuing")
        time.sleep(cfg.shadow_interval_sec)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_shadow_loop()


if __name__ == "__main__":
    main()
