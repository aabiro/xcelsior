"""Shadow-mode scheduler runner (blueprint Phase 3).

Runs the new placement pipeline — snapshot → order → Stage C filters →
Stage D scoring — against a consistent read-only snapshot, simulates
capacity consumption across the cycle in memory, and persists one
explained decision row per queued job into ``scheduler_shadow_decisions``.
It writes *nothing* to jobs, hosts, attempts, or allocations: the legacy
scheduler remains fully authoritative until the Phase 4 cutover.

A second step compares aged decisions (older than a grace window that
gives the legacy scheduler time to act on the same queue state) against
what actually happened to each job, and records a typed agreement/
mismatch class. Mismatch-rate summaries over these rows are the Phase 3
exit-gate evidence.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction
from control_plane.scheduler.config import SchedulerConfig
from control_plane.scheduler.explain import build_explanation
from control_plane.scheduler.filters import (
    FILTER_POLICY_VERSION,
    FilterContext,
    filter_hosts,
)
from control_plane.scheduler.scoring import rank_candidates
from control_plane.scheduler.snapshot import SchedulerSnapshot, take_snapshot

log = logging.getLogger("xcelsior.control_plane.scheduler.shadow")

_COMPARE_BATCH = 500
_COMPARE_MAX_BATCHES = 40
_PRUNE_BATCH = 1000

COMPARISON_CLASSES = (
    "match_place",
    "match_queue",
    "host_mismatch",
    "shadow_placed_legacy_queued",
    "legacy_placed_shadow_queued",
    "job_missing",
    "indeterminate",
)

# Order of checks matters: a job whose *status* says queued is queued,
# even if a stale host_id lingers from a previous attempt.
_COMPARE_SQL = """
WITH due AS (
    SELECT decision_id, job_id, outcome, selected_host_id
      FROM scheduler_shadow_decisions
     WHERE compared_at IS NULL
       AND snapshot_at < clock_timestamp() - make_interval(secs => %(grace)s)
     ORDER BY snapshot_at
     LIMIT %(limit)s
       FOR UPDATE SKIP LOCKED
)
UPDATE scheduler_shadow_decisions d
   SET compared_at = clock_timestamp(),
       legacy_status = j.status,
       legacy_host_id = j.host_id,
       comparison = CASE
           WHEN j.job_id IS NULL THEN 'job_missing'
           WHEN j.status = 'queued' AND q.outcome = 'queue' THEN 'match_queue'
           WHEN j.status = 'queued' THEN 'shadow_placed_legacy_queued'
           WHEN j.host_id IS NOT NULL AND q.outcome = 'place'
                AND j.host_id = q.selected_host_id THEN 'match_place'
           WHEN j.host_id IS NOT NULL AND q.outcome = 'place'
                THEN 'host_mismatch'
           WHEN j.host_id IS NOT NULL THEN 'legacy_placed_shadow_queued'
           ELSE 'indeterminate'
       END
  FROM due q
  LEFT JOIN jobs j ON j.job_id = q.job_id
 WHERE d.decision_id = q.decision_id
RETURNING d.comparison
"""


@dataclass(frozen=True)
class ShadowDecision:
    job_id: str
    outcome: str  # 'place' | 'queue'
    selected_host_id: str | None
    placement_score: int | None
    queue_reason_code: str | None
    eligible_host_count: int
    host_count: int
    explanation: dict[str, Any]


@dataclass(frozen=True)
class ShadowCycleReport:
    cycle_id: str
    jobs_considered: int = 0
    placed: int = 0
    queued: int = 0
    comparisons: dict[str, int] = field(default_factory=dict)
    pruned: int = 0


def _row_get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def _job_gpus(job: dict[str, Any]) -> int:
    try:
        return max(1, int(job.get("num_gpus", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _job_vram_gb(job: dict[str, Any]) -> float:
    try:
        return max(0.0, float(job.get("vram_needed_gb", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def simulate_cycle(
    snapshot: SchedulerSnapshot,
    *,
    max_rejections: int = 25,
    max_ranked: int = 10,
) -> list[ShadowDecision]:
    """Pure placement simulation over one snapshot.

    Walks jobs in claim order and charges each simulated placement
    against in-memory host capacity, so later jobs in the same cycle see
    the fleet as it would look after earlier placements — mirroring what
    sequential reservation transactions would observe.
    """
    hosts = [dict(h) for h in snapshot.hosts]
    ctx = FilterContext(stale_host_ids=snapshot.stale_host_ids)
    decisions: list[ShadowDecision] = []

    for job in snapshot.jobs:
        eligible, rejections = filter_hosts(job, hosts, ctx)
        ranked = rank_candidates(job, eligible) if eligible else []
        if ranked:
            winner = ranked[0]
            explanation = build_explanation(
                job=job,
                host_count=len(hosts),
                rejections=rejections,
                ranked=ranked,
                selected_host_id=winner.host_id,
                max_rejections=max_rejections,
                max_ranked=max_ranked,
            )
            decisions.append(
                ShadowDecision(
                    job_id=str(job.get("job_id")),
                    outcome="place",
                    selected_host_id=winner.host_id,
                    placement_score=winner.total,
                    queue_reason_code=None,
                    eligible_host_count=len(eligible),
                    host_count=len(hosts),
                    explanation=explanation,
                )
            )
            # Charge the simulated placement against fleet capacity.
            for host in hosts:
                if str(host.get("host_id")) == winner.host_id:
                    free_gpus = int(host.get("free_gpu_count", host.get("gpu_count", 1)) or 0)
                    host["free_gpu_count"] = max(0, free_gpus - _job_gpus(job))
                    try:
                        free_vram = float(host.get("free_vram_gb", 0) or 0)
                    except (TypeError, ValueError):
                        free_vram = 0.0
                    host["free_vram_gb"] = max(0.0, free_vram - _job_vram_gb(job))
                    break
        else:
            reason_code = "no_hosts" if not hosts else "no_eligible_host"
            explanation = build_explanation(
                job=job,
                host_count=len(hosts),
                rejections=rejections,
                ranked=[],
                selected_host_id=None,
                queue_reason_code=reason_code,
                max_rejections=max_rejections,
                max_ranked=max_ranked,
            )
            decisions.append(
                ShadowDecision(
                    job_id=str(job.get("job_id")),
                    outcome="queue",
                    selected_host_id=None,
                    placement_score=None,
                    queue_reason_code=reason_code,
                    eligible_host_count=0,
                    host_count=len(hosts),
                    explanation=explanation,
                )
            )
    return decisions


def persist_decisions(
    conn: Connection,
    decisions: list[ShadowDecision],
    *,
    cycle_id: str,
    replica_id: str,
    snapshot_at: Any,
) -> int:
    for d in decisions:
        conn.execute(
            """
            INSERT INTO scheduler_shadow_decisions
                (cycle_id, replica_id, job_id, snapshot_at, engine,
                 policy_version, outcome, queue_reason_code,
                 selected_host_id, placement_score,
                 eligible_host_count, host_count, explanation)
            VALUES (%s, %s, %s, %s, 'v2', %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cycle_id, replica_id, d.job_id, snapshot_at,
                FILTER_POLICY_VERSION, d.outcome, d.queue_reason_code,
                d.selected_host_id, d.placement_score,
                d.eligible_host_count, d.host_count, Jsonb(d.explanation),
            ),
        )
    return len(decisions)


def compare_due_decisions(
    conn: Connection,
    *,
    grace_sec: int,
    limit: int = _COMPARE_BATCH,
) -> dict[str, int]:
    """Settle aged, uncompared decisions against legacy reality."""
    rows = conn.execute(
        _COMPARE_SQL, {"grace": grace_sec, "limit": limit}
    ).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        cls = str(_row_get(row, "comparison", 0))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def prune_old_decisions(
    conn: Connection, *, retention_days: int, limit: int = _PRUNE_BATCH
) -> int:
    result = conn.execute(
        """
        DELETE FROM scheduler_shadow_decisions
         WHERE decision_id IN (
             SELECT decision_id
               FROM scheduler_shadow_decisions
              WHERE created_at <
                    clock_timestamp() - make_interval(days => %(days)s)
              LIMIT %(limit)s
         )
        """,
        {"days": retention_days, "limit": limit},
    )
    return result.rowcount


def summarize_comparisons(conn: Connection, *, since_hours: int = 24) -> dict[str, Any]:
    """Mismatch-rate rollup for dashboards and the Phase 3 sign-off."""
    rows = conn.execute(
        """
        SELECT comparison, count(*) AS n
          FROM scheduler_shadow_decisions
         WHERE compared_at IS NOT NULL
           AND snapshot_at > clock_timestamp() - make_interval(hours => %s)
         GROUP BY comparison
        """,
        (since_hours,),
    ).fetchall()
    counts = {
        str(_row_get(r, "comparison", 0)): int(_row_get(r, "n", 1)) for r in rows
    }
    total = sum(counts.values())
    matches = counts.get("match_place", 0) + counts.get("match_queue", 0)
    return {
        "since_hours": since_hours,
        "compared": total,
        "matches": matches,
        "match_rate": (matches / total) if total else None,
        "by_class": counts,
    }


class ShadowRunner:
    """One shadow scheduling replica: compare, snapshot, decide, persist."""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig.from_env()

    def run_once(self) -> ShadowCycleReport:
        cfg = self.config
        cycle_id = str(uuid.uuid4())

        # 1. Settle decisions whose grace window has passed. Drain in
        # bounded batches: one cycle over a deep queue can produce far
        # more decisions than a single batch, and leaving them uncompared
        # would starve the mismatch evidence the phase exists to collect.
        comparisons: dict[str, int] = {}
        for _ in range(_COMPARE_MAX_BATCHES):
            batch = run_transaction(
                lambda conn: compare_due_decisions(
                    conn, grace_sec=cfg.shadow_compare_grace_sec
                ),
                what="shadow_compare",
            )
            for cls, n in batch.items():
                comparisons[cls] = comparisons.get(cls, 0) + n
            if sum(batch.values()) < _COMPARE_BATCH:
                break

        # 2. Consistent read-only snapshot.
        snapshot = run_transaction(
            lambda conn: take_snapshot(
                conn, host_freshness_timeout_sec=cfg.host_freshness_timeout_sec
            ),
            what="shadow_snapshot",
        )

        # 3. Pure simulation + 4. persist the decision set.
        decisions = simulate_cycle(
            snapshot,
            max_rejections=cfg.explain_max_rejections,
            max_ranked=cfg.explain_max_ranked,
        )
        if decisions:
            run_transaction(
                lambda conn: persist_decisions(
                    conn,
                    decisions,
                    cycle_id=cycle_id,
                    replica_id=cfg.replica_id,
                    snapshot_at=snapshot.taken_at,
                ),
                what="shadow_persist",
            )

        # 5. Bounded retention.
        pruned = run_transaction(
            lambda conn: prune_old_decisions(
                conn, retention_days=cfg.shadow_retention_days
            ),
            what="shadow_prune",
        )

        placed = sum(1 for d in decisions if d.outcome == "place")
        report = ShadowCycleReport(
            cycle_id=cycle_id,
            jobs_considered=len(decisions),
            placed=placed,
            queued=len(decisions) - placed,
            comparisons=comparisons,
            pruned=pruned,
        )
        if report.jobs_considered or report.comparisons or report.pruned:
            mismatches = sum(
                n for cls, n in comparisons.items()
                if cls not in ("match_place", "match_queue")
            )
            log.info(
                "shadow cycle %s: %d decisions (%d place / %d queue), "
                "%d compared (%d mismatch), %d pruned",
                cycle_id[:8],
                report.jobs_considered,
                report.placed,
                report.queued,
                sum(comparisons.values()),
                mismatches,
                pruned,
            )
        return report
