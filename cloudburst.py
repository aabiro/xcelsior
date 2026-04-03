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

import json as _json
import logging
import os
import shlex
import subprocess
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

# AWS configuration
BURST_AWS_AMI = os.environ.get("XCELSIOR_BURST_AWS_AMI", "")  # GPU AMI
BURST_AWS_KEY_PAIR = os.environ.get("XCELSIOR_BURST_AWS_KEY_PAIR", "")
BURST_AWS_SECURITY_GROUP = os.environ.get("XCELSIOR_BURST_AWS_SECURITY_GROUP", "")
BURST_AWS_SUBNET = os.environ.get("XCELSIOR_BURST_AWS_SUBNET", "")

# GCP configuration
BURST_GCP_PROJECT = os.environ.get("XCELSIOR_BURST_GCP_PROJECT", "")
BURST_GCP_IMAGE_FAMILY = os.environ.get("XCELSIOR_BURST_GCP_IMAGE_FAMILY", "pytorch-latest-gpu")
BURST_GCP_ZONE = os.environ.get("XCELSIOR_BURST_GCP_ZONE", "northamerica-northeast1-a")

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
        """Provision a cloud GPU instance via the provider CLI.

        Uses `aws ec2 run-instances` or `gcloud compute instances create`
        to actually launch a VM. The worker_agent on bootstraps itself
        and registers back via heartbeat.
        """
        type_info = CLOUD_INSTANCE_TYPES.get(cloud_provider, {}).get(instance_type)
        if not type_info:
            raise ValueError(f"Unknown instance type: {cloud_provider}/{instance_type}")

        now = time.time()
        instance_id = f"burst-{uuid.uuid4().hex[:12]}"

        # Actually provision the cloud VM
        cloud_instance_id = self._launch_cloud_vm(cloud_provider, instance_type, instance_id, type_info)

        with self._conn() as conn:
            status = "provisioning" if cloud_instance_id else "error"
            conn.execute(
                """INSERT INTO cloud_burst_instances
                   (instance_id, cloud_provider, instance_type, region,
                    gpu_model, gpu_count, cost_per_hour_cad,
                    status, cloud_instance_id, started_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (instance_id, cloud_provider, instance_type,
                 type_info["region"], type_info["gpu_model"],
                 type_info["gpu_count"], type_info["cost_per_hour_cad"],
                 status, cloud_instance_id or "", now),
            )

        if not cloud_instance_id:
            raise ValueError(f"Failed to launch {cloud_provider}/{instance_type} — check CLI configuration")

        log.info("BURST PROVISION: %s %s/%s gpu=%s cost=$%.2f/hr cloud_id=%s",
                 instance_id, cloud_provider, instance_type,
                 type_info["gpu_model"], type_info["cost_per_hour_cad"], cloud_instance_id)

        return {
            "instance_id": instance_id,
            "cloud_provider": cloud_provider,
            "instance_type": instance_type,
            "gpu_model": type_info["gpu_model"],
            "cloud_instance_id": cloud_instance_id,
            "status": "provisioning",
        }

    def _launch_cloud_vm(self, provider: str, instance_type: str,
                         xcelsior_id: str, type_info: dict) -> Optional[str]:
        """Launch a VM via the cloud provider CLI. Returns the cloud instance ID."""
        if provider == "aws":
            return self._launch_aws(instance_type, xcelsior_id, type_info)
        elif provider == "gcp":
            return self._launch_gcp(instance_type, xcelsior_id, type_info)
        else:
            log.error("Unsupported cloud provider: %s", provider)
            return None

    def _launch_aws(self, instance_type: str, xcelsior_id: str, type_info: dict) -> Optional[str]:
        """Launch an AWS EC2 GPU instance."""
        if not BURST_AWS_AMI:
            log.error("XCELSIOR_BURST_AWS_AMI not configured — cannot launch AWS instance")
            return None

        region = type_info.get("region", "ca-central-1")
        cmd = [
            "aws", "ec2", "run-instances",
            "--image-id", BURST_AWS_AMI,
            "--instance-type", instance_type,
            "--region", region,
            "--count", "1",
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value=xcelsior-{xcelsior_id}}},"
            f"{{Key=xcelsior-id,Value={xcelsior_id}}}]",
            "--output", "json",
        ]
        if BURST_AWS_KEY_PAIR:
            cmd.extend(["--key-name", BURST_AWS_KEY_PAIR])
        if BURST_AWS_SECURITY_GROUP:
            cmd.extend(["--security-group-ids", BURST_AWS_SECURITY_GROUP])
        if BURST_AWS_SUBNET:
            cmd.extend(["--subnet-id", BURST_AWS_SUBNET])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                log.error("AWS launch failed: %s", result.stderr)
                return None
            data = _json.loads(result.stdout)
            cloud_id = data.get("Instances", [{}])[0].get("InstanceId", "")
            log.info("AWS instance launched: %s", cloud_id)
            return cloud_id
        except FileNotFoundError:
            log.error("AWS CLI not installed — cannot launch burst instance")
            return None
        except subprocess.TimeoutExpired:
            log.error("AWS launch timed out")
            return None
        except Exception as e:
            log.error("AWS launch error: %s", e)
            return None

    def _launch_gcp(self, instance_type: str, xcelsior_id: str, type_info: dict) -> Optional[str]:
        """Launch a GCP Compute Engine GPU instance."""
        if not BURST_GCP_PROJECT:
            log.error("XCELSIOR_BURST_GCP_PROJECT not configured — cannot launch GCP instance")
            return None

        vm_name = f"xcelsior-{xcelsior_id}"
        cmd = [
            "gcloud", "compute", "instances", "create", vm_name,
            "--project", BURST_GCP_PROJECT,
            "--zone", BURST_GCP_ZONE,
            "--machine-type", instance_type,
            "--image-family", BURST_GCP_IMAGE_FAMILY,
            "--image-project", "deeplearning-platform-release",
            "--maintenance-policy", "TERMINATE",
            "--labels", f"xcelsior-id={xcelsior_id}",
            "--format", "json",
        ]

        # Add accelerator config based on GPU model
        gpu_model = type_info.get("gpu_model", "")
        gpu_count = type_info.get("gpu_count", 1)
        accelerator_map = {
            "A100": "nvidia-tesla-a100",
            "L4": "nvidia-l4",
            "T4": "nvidia-tesla-t4",
            "V100": "nvidia-tesla-v100",
        }
        accel_type = accelerator_map.get(gpu_model)
        if accel_type:
            cmd.extend(["--accelerator", f"count={gpu_count},type={accel_type}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                log.error("GCP launch failed: %s", result.stderr)
                return None
            data = _json.loads(result.stdout)
            # GCP returns a list; get the self link or name as identifier
            cloud_id = data[0].get("name", vm_name) if isinstance(data, list) and data else vm_name
            log.info("GCP instance launched: %s", cloud_id)
            return cloud_id
        except FileNotFoundError:
            log.error("gcloud CLI not installed — cannot launch burst instance")
            return None
        except subprocess.TimeoutExpired:
            log.error("GCP launch timed out")
            return None
        except Exception as e:
            log.error("GCP launch error: %s", e)
            return None

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

        # Actually terminate drained instances
        for inst in running:
            # Re-check in case it was just marked draining
            pass
        with self._conn() as conn:
            draining = conn.execute(
                "SELECT * FROM cloud_burst_instances WHERE status = 'draining'",
            ).fetchall()
        for inst in draining:
            self.terminate_instance(inst["instance_id"])

        return drained

    def terminate_instance(self, instance_id: str):
        """Terminate a burst instance: kill the cloud VM and record final spending."""
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

            # Actually terminate the cloud VM
            cloud_id = inst.get("cloud_instance_id", "")
            provider = inst.get("cloud_provider", "")
            if cloud_id:
                self._terminate_cloud_vm(provider, cloud_id, inst)

            conn.execute(
                """UPDATE cloud_burst_instances
                   SET status = 'terminated', terminated_at = %s, budget_spent_cad = %s
                   WHERE instance_id = %s""",
                (now, total_cost, instance_id),
            )

        log.info("BURST TERMINATED: %s runtime=%.1fh cost=$%.2f", instance_id, runtime_hours, total_cost)

    def _terminate_cloud_vm(self, provider: str, cloud_id: str, inst: dict):
        """Terminate a cloud VM via provider CLI."""
        try:
            if provider == "aws":
                region = inst.get("region", "ca-central-1")
                result = subprocess.run(
                    ["aws", "ec2", "terminate-instances",
                     "--instance-ids", cloud_id,
                     "--region", region],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    log.error("AWS terminate failed for %s: %s", cloud_id, result.stderr)
                else:
                    log.info("AWS instance terminated: %s", cloud_id)

            elif provider == "gcp":
                result = subprocess.run(
                    ["gcloud", "compute", "instances", "delete", cloud_id,
                     "--project", BURST_GCP_PROJECT,
                     "--zone", BURST_GCP_ZONE,
                     "--quiet"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    log.error("GCP terminate failed for %s: %s", cloud_id, result.stderr)
                else:
                    log.info("GCP instance terminated: %s", cloud_id)
        except FileNotFoundError:
            log.error("Cloud CLI not installed — cannot terminate %s/%s", provider, cloud_id)
        except Exception as e:
            log.error("Cloud terminate error for %s: %s", cloud_id, e)

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
