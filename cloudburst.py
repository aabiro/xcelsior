# Xcelsior Cloud Burst Auto-Scaler
# Extends the existing autoscale system with cloud provider bursting.
#
# When community GPU supply is insufficient, automatically provisions
# GPU instances from AWS/GCP as overflow, then drains them when
# community supply recovers.
#
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md:
# - Budget-capped cloud burst (never overspend)
# - Prefer community GPUs (lower cost, sovereignty)
# - Drain cloud instances when community catches up
# - Track cloud spend for billing transparency

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("xcelsior.cloudburst")

# ── Configuration ─────────────────────────────────────────────────────

BURST_ENABLED = os.environ.get("XCELSIOR_BURST_ENABLED", "").lower() in ("1", "true", "yes")
BURST_BUDGET_CAD = float(os.environ.get("XCELSIOR_BURST_BUDGET_CAD", "500.0"))
BURST_MAX_INSTANCES = int(os.environ.get("XCELSIOR_BURST_MAX_INSTANCES", "10"))
BURST_QUEUE_THRESHOLD = int(os.environ.get("XCELSIOR_BURST_QUEUE_THRESHOLD", "5"))
BURST_DRAIN_IDLE_SEC = int(os.environ.get("XCELSIOR_BURST_DRAIN_IDLE_SEC", "900"))  # 15 min

# Cloud provider instance types and pricing
CLOUD_INSTANCE_TYPES = {
    "aws": {
        "g5.xlarge": {"gpu_model": "A10G", "gpu_count": 1, "cost_per_hour_cad": 1.80, "region": "ca-central-1"},
        "g5.2xlarge": {"gpu_model": "A10G", "gpu_count": 1, "cost_per_hour_cad": 2.10, "region": "ca-central-1"},
        "p4d.24xlarge": {"gpu_model": "A100", "gpu_count": 8, "cost_per_hour_cad": 55.0, "region": "ca-central-1"},
    },
    "gcp": {
        "a2-highgpu-1g": {"gpu_model": "A100", "gpu_count": 1, "cost_per_hour_cad": 5.50, "region": "northamerica-northeast1"},
        "g2-standard-4": {"gpu_model": "L4", "gpu_count": 1, "cost_per_hour_cad": 1.20, "region": "northamerica-northeast1"},
    },
}


class CloudBurstEngine:
    """Manages cloud burst instances for overflow GPU demand."""

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def evaluate_burst_need(self) -> dict:
        """Check if cloud bursting is needed based on queue pressure."""
        if not BURST_ENABLED:
            return {"action": "disabled"}

        with self._conn() as conn:
            # Count queued jobs waiting
            queued = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE status = 'queued'",
            ).fetchone()

            # Count active community hosts
            community = conn.execute(
                "SELECT COUNT(*) as cnt FROM hosts WHERE status = 'active'",
            ).fetchone()

            # Count active burst instances
            burst = conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(budget_spent_cad), 0) as spent FROM cloud_burst_instances WHERE status IN ('provisioning', 'running')",
            ).fetchone()

        queue_depth = queued["cnt"] or 0
        active_community = community["cnt"] or 0
        active_burst = burst["cnt"] or 0
        total_spent = float(burst["spent"] or 0)

        if queue_depth < BURST_QUEUE_THRESHOLD:
            return {
                "action": "none",
                "queue_depth": queue_depth,
                "threshold": BURST_QUEUE_THRESHOLD,
            }

        if active_burst >= BURST_MAX_INSTANCES:
            return {"action": "at_max", "active_burst": active_burst}

        if total_spent >= BURST_BUDGET_CAD:
            return {"action": "budget_exceeded", "spent": total_spent, "budget": BURST_BUDGET_CAD}

        # Need to burst
        instances_needed = min(
            queue_depth - active_community,
            BURST_MAX_INSTANCES - active_burst,
            3,  # Max 3 at a time
        )
        if instances_needed <= 0:
            return {"action": "none", "reason": "sufficient_supply"}

        return {
            "action": "burst",
            "instances_needed": instances_needed,
            "queue_depth": queue_depth,
            "active_burst": active_burst,
            "budget_remaining": BURST_BUDGET_CAD - total_spent,
        }

    def provision_burst_instance(
        self,
        cloud_provider: str,
        instance_type: str,
    ) -> dict:
        """Provision a cloud GPU instance for bursting.

        NOTE: Actual cloud API calls would go here. This records the intent
        and the worker_agent on that instance registers back via heartbeat.
        """
        type_info = CLOUD_INSTANCE_TYPES.get(cloud_provider, {}).get(instance_type)
        if not type_info:
            raise ValueError(f"Unknown instance type: {cloud_provider}/{instance_type}")

        now = time.time()
        instance_id = f"burst-{uuid.uuid4().hex[:12]}"

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO cloud_burst_instances
                   (instance_id, cloud_provider, instance_type, region,
                    gpu_model, gpu_count, cost_per_hour_cad,
                    status, started_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 'provisioning', %s)""",
                (instance_id, cloud_provider, instance_type,
                 type_info["region"], type_info["gpu_model"],
                 type_info["gpu_count"], type_info["cost_per_hour_cad"], now),
            )

        log.info("BURST PROVISION: %s %s/%s gpu=%s cost=$%.2f/hr",
                 instance_id, cloud_provider, instance_type,
                 type_info["gpu_model"], type_info["cost_per_hour_cad"])

        return {
            "instance_id": instance_id,
            "cloud_provider": cloud_provider,
            "instance_type": instance_type,
            "gpu_model": type_info["gpu_model"],
            "status": "provisioning",
        }

    def mark_running(self, instance_id: str, host_id: str = "", cloud_instance_id: str = ""):
        """Mark a burst instance as running (after it registers via heartbeat)."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE cloud_burst_instances
                   SET status = 'running', host_id = %s, cloud_instance_id = %s
                   WHERE instance_id = %s""",
                (host_id, cloud_instance_id, instance_id),
            )

    def drain_idle_instances(self) -> int:
        """Drain burst instances that are idle (no jobs assigned)."""
        cutoff = time.time() - BURST_DRAIN_IDLE_SEC
        drained = 0

        with self._conn() as conn:
            running = conn.execute(
                "SELECT * FROM cloud_burst_instances WHERE status = 'running'",
            ).fetchall()

            for inst in running:
                host_id = inst.get("host_id", "")
                if not host_id:
                    continue

                # Check if any jobs assigned
                job = conn.execute(
                    "SELECT job_id FROM jobs WHERE host_id = %s AND status = 'running'",
                    (host_id,),
                ).fetchone()

                if job:
                    continue  # Still busy

                # Check last job completed time
                last_job = conn.execute(
                    "SELECT MAX(completed_at) as last FROM jobs WHERE host_id = %s",
                    (host_id,),
                ).fetchone()

                last_active = float(last_job["last"] or inst["started_at"])
                if last_active < cutoff:
                    conn.execute(
                        "UPDATE cloud_burst_instances SET status = 'draining' WHERE instance_id = %s",
                        (inst["instance_id"],),
                    )
                    drained += 1
                    log.info("BURST DRAIN: %s idle for >%ds", inst["instance_id"], BURST_DRAIN_IDLE_SEC)

        return drained

    def terminate_instance(self, instance_id: str):
        """Terminate a burst instance and record final spending."""
        now = time.time()
        with self._conn() as conn:
            inst = conn.execute(
                "SELECT * FROM cloud_burst_instances WHERE instance_id = %s",
                (instance_id,),
            ).fetchone()
            if not inst:
                return

            runtime_hours = (now - inst["started_at"]) / 3600
            total_cost = round(runtime_hours * inst["cost_per_hour_cad"], 2)

            conn.execute(
                """UPDATE cloud_burst_instances
                   SET status = 'terminated', terminated_at = %s, budget_spent_cad = %s
                   WHERE instance_id = %s""",
                (now, total_cost, instance_id),
            )

        log.info("BURST TERMINATED: %s runtime=%.1fh cost=$%.2f", instance_id, runtime_hours, total_cost)

    def update_burst_spending(self) -> float:
        """Update running burst instance spend tracking. Returns total spent."""
        now = time.time()
        total = 0.0
        with self._conn() as conn:
            running = conn.execute(
                "SELECT * FROM cloud_burst_instances WHERE status IN ('running', 'draining')",
            ).fetchall()
            for inst in running:
                hours = (now - inst["started_at"]) / 3600
                cost = round(hours * inst["cost_per_hour_cad"], 2)
                conn.execute(
                    "UPDATE cloud_burst_instances SET budget_spent_cad = %s WHERE instance_id = %s",
                    (cost, inst["instance_id"]),
                )
                total += cost
        return total

    def get_burst_status(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT status, COUNT(*) as cnt, COALESCE(SUM(budget_spent_cad), 0) as spent
                   FROM cloud_burst_instances GROUP BY status""",
            ).fetchall()

        status = {r["status"]: {"count": r["cnt"], "spent": float(r["spent"])} for r in rows}
        total_spent = sum(s["spent"] for s in status.values())
        return {
            "enabled": BURST_ENABLED,
            "budget_cad": BURST_BUDGET_CAD,
            "total_spent_cad": round(total_spent, 2),
            "budget_remaining_cad": round(BURST_BUDGET_CAD - total_spent, 2),
            "max_instances": BURST_MAX_INSTANCES,
            "by_status": status,
        }


# ── Singleton ─────────────────────────────────────────────────────────

_burst_engine: Optional[CloudBurstEngine] = None


def get_burst_engine() -> CloudBurstEngine:
    global _burst_engine
    if _burst_engine is None:
        _burst_engine = CloudBurstEngine()
    return _burst_engine
