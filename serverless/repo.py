# Xcelsior — Serverless inference repository (Postgres)
# Typed CRUD + SKIP LOCKED queue claim. Business logic lives in service.py.

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("xcelsior.serverless.repo")

JOB_STATUS_QUEUED = "IN_QUEUE"
JOB_STATUS_IN_PROGRESS = "IN_PROGRESS"
JOB_STATUS_COMPLETED = "COMPLETED"
JOB_STATUS_FAILED = "FAILED"
JOB_STATUS_CANCELLED = "CANCELLED"

ENDPOINT_STATUS_PROVISIONING = "provisioning"
ENDPOINT_STATUS_ACTIVE = "active"
ENDPOINT_STATUS_SCALED_DOWN = "scaled_down"
ENDPOINT_STATUS_ERROR = "error"
ENDPOINT_STATUS_DELETED = "deleted"

WORKER_STATE_BOOTING = "booting"
WORKER_STATE_READY = "ready"
WORKER_STATE_IDLE = "idle"
WORKER_STATE_DRAINING = "draining"
WORKER_STATE_ERROR = "error"
WORKER_STATE_TERMINATED = "terminated"


@dataclass
class EndpointCreate:
    owner_id: str
    name: str = ""
    mode: str = "preset"
    managed_engine: str = "vllm"
    model_ref: str = ""
    model_revision: str = "main"
    image_ref: str = ""
    startup_command: str = ""
    http_port: int = 8080
    health_check_path: str = "/health"
    cuda_version: str = "12.4"
    registry_auth_ref: str = ""
    gpu_tier: str = ""
    gpu_count: int = 1
    vram_required_gb: float = 0.0
    min_workers: int = 0
    max_workers: int = 4
    max_concurrency: int = 4
    idle_timeout_sec: int = 300
    scaling_policy_type: str = "queue_request_count"
    scaling_policy_value: int = 1
    request_timeout_sec: int = 120
    max_request_bytes: int = 10_485_760
    max_queue_size: int = 100
    keep_warm: bool = False
    cache_volume_id: str | None = None
    region: str = "ca-east"
    source_type: str = ""
    source_ref: str = ""
    source_ref_branch: str = "main"
    env: dict[str, Any] = field(default_factory=dict)
    lora_adapters: list[dict[str, str]] = field(default_factory=list)


class ServerlessRepo:
    """Postgres repository for serverless endpoints, workers, jobs, keys, and streams."""

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

    @staticmethod
    def _jsonb(value: Any):
        from psycopg.types.json import Jsonb

        return Jsonb(value if value is not None else {})

    # ── Endpoints ─────────────────────────────────────────────────────

    def create_endpoint(self, spec: EndpointCreate) -> dict:
        now = time.time()
        endpoint_id = f"sep-{uuid.uuid4().hex[:12]}"
        status = (
            ENDPOINT_STATUS_PROVISIONING
            if spec.min_workers >= 1
            else ENDPOINT_STATUS_SCALED_DOWN
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO serverless_endpoints (
                    endpoint_id, owner_id, name, mode, managed_engine,
                    model_ref, model_revision, image_ref, startup_command,
                    http_port, health_check_path, cuda_version, registry_auth_ref,
                    gpu_tier, gpu_count, vram_required_gb,
                    min_workers, max_workers, max_concurrency, idle_timeout_sec,
                    scaling_policy_type, scaling_policy_value,
                    request_timeout_sec, max_request_bytes, max_queue_size, keep_warm,
                    cache_volume_id, region, env, status, created_at, updated_at,
                    lora_adapters
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s
                )
                """,
                (
                    endpoint_id,
                    spec.owner_id,
                    spec.name,
                    spec.mode,
                    spec.managed_engine,
                    spec.model_ref,
                    spec.model_revision,
                    spec.image_ref,
                    spec.startup_command,
                    spec.http_port,
                    spec.health_check_path,
                    spec.cuda_version,
                    spec.registry_auth_ref,
                    spec.gpu_tier,
                    spec.gpu_count,
                    spec.vram_required_gb,
                    spec.min_workers,
                    spec.max_workers,
                    spec.max_concurrency,
                    spec.idle_timeout_sec,
                    spec.scaling_policy_type,
                    spec.scaling_policy_value,
                    spec.request_timeout_sec,
                    spec.max_request_bytes,
                    spec.max_queue_size,
                    spec.keep_warm,
                    spec.cache_volume_id,
                    spec.region,
                    self._jsonb(spec.env),
                    status,
                    now,
                    now,
                    self._jsonb(spec.lora_adapters),
                ),
            )
        row = self.get_endpoint(endpoint_id)
        assert row is not None
        return row

    def get_endpoint(self, endpoint_id: str, *, owner_id: str | None = None) -> dict | None:
        with self._conn() as conn:
            if owner_id:
                row = conn.execute(
                    """
                    SELECT * FROM serverless_endpoints
                    WHERE endpoint_id = %s AND owner_id = %s AND deleted_at = 0
                    """,
                    (endpoint_id, owner_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM serverless_endpoints
                    WHERE endpoint_id = %s AND deleted_at = 0
                    """,
                    (endpoint_id,),
                ).fetchone()
        return dict(row) if row else None

    def list_endpoints(self, owner_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_endpoints
                WHERE owner_id = %s AND deleted_at = 0
                ORDER BY created_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def patch_endpoint(self, endpoint_id: str, owner_id: str, fields: dict[str, Any]) -> dict | None:
        allowed = {
            "name",
            "min_workers",
            "max_workers",
            "max_concurrency",
            "idle_timeout_sec",
            "scaling_policy_type",
            "scaling_policy_value",
            "request_timeout_sec",
            "max_request_bytes",
            "max_queue_size",
            "keep_warm",
            "env",
            "status",
            "lora_adapters",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_endpoint(endpoint_id, owner_id=owner_id)
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values())
        if "env" in updates:
            idx = list(updates.keys()).index("env")
            values[idx] = self._jsonb(updates["env"])
        if "lora_adapters" in updates:
            idx = list(updates.keys()).index("lora_adapters")
            values[idx] = self._jsonb(updates["lora_adapters"])
        values.extend([time.time(), endpoint_id, owner_id])
        with self._conn() as conn:
            conn.execute(
                f"""
                UPDATE serverless_endpoints
                SET {set_clause}, updated_at = %s
                WHERE endpoint_id = %s AND owner_id = %s AND deleted_at = 0
                """,
                values,
            )
        return self.get_endpoint(endpoint_id, owner_id=owner_id)

    def soft_delete_endpoint(self, endpoint_id: str, owner_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE serverless_endpoints
                SET status = %s, deleted_at = %s, updated_at = %s
                WHERE endpoint_id = %s AND owner_id = %s AND deleted_at = 0
                """,
                (ENDPOINT_STATUS_DELETED, now, now, endpoint_id, owner_id),
            )
        return cur.rowcount > 0

    # ── Workers ─────────────────────────────────────────────────────

    def create_worker(
        self,
        endpoint_id: str,
        *,
        scheduler_job_id: str | None = None,
        gpu_count: int = 1,
        allocated_at: float | None = None,
    ) -> dict:
        now = time.time()
        worker_id = f"swk-{uuid.uuid4().hex[:12]}"
        alloc = allocated_at if allocated_at is not None else now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO serverless_workers (
                    worker_id, endpoint_id, scheduler_job_id, state, gpu_count,
                    allocated_at, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    worker_id,
                    endpoint_id,
                    scheduler_job_id,
                    WORKER_STATE_BOOTING,
                    gpu_count,
                    alloc,
                    now,
                    now,
                ),
            )
        row = self.get_worker(worker_id)
        assert row is not None
        return row

    def get_worker(self, worker_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM serverless_workers WHERE worker_id = %s",
                (worker_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_worker_by_scheduler_job_id(self, scheduler_job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM serverless_workers
                WHERE scheduler_job_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (scheduler_job_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_workers(self, endpoint_id: str, *, states: list[str] | None = None) -> list[dict]:
        with self._conn() as conn:
            if states:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_workers
                    WHERE endpoint_id = %s AND state = ANY(%s)
                    ORDER BY created_at ASC
                    """,
                    (endpoint_id, states),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_workers
                    WHERE endpoint_id = %s
                    ORDER BY created_at ASC
                    """,
                    (endpoint_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def update_worker(
        self,
        worker_id: str,
        *,
        state: str | None = None,
        host_id: str | None = None,
        current_concurrency: int | None = None,
        released_at: float | None = None,
        last_heartbeat_at: float | None = None,
        error_message: str | None = None,
    ) -> dict | None:
        fields: dict[str, Any] = {"updated_at": time.time()}
        if state is not None:
            fields["state"] = state
        if host_id is not None:
            fields["host_id"] = host_id
        if current_concurrency is not None:
            fields["current_concurrency"] = current_concurrency
        if released_at is not None:
            fields["released_at"] = released_at
        if last_heartbeat_at is not None:
            fields["last_heartbeat_at"] = last_heartbeat_at
        if error_message is not None:
            fields["error_message"] = error_message
        set_clause = ", ".join(f"{k} = %s" for k in fields)
        values = list(fields.values()) + [worker_id]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE serverless_workers SET {set_clause} WHERE worker_id = %s",
                values,
            )
        return self.get_worker(worker_id)

    def count_active_workers(self, endpoint_id: str) -> int:
        active = {
            WORKER_STATE_BOOTING,
            WORKER_STATE_READY,
            WORKER_STATE_IDLE,
            WORKER_STATE_DRAINING,
        }
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM serverless_workers
                WHERE endpoint_id = %s AND state = ANY(%s)
                """,
                (endpoint_id, list(active)),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    # ── Jobs (queue) ──────────────────────────────────────────────────

    def enqueue_job(
        self,
        endpoint_id: str,
        owner_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        webhook_url: str | None = None,
    ) -> dict:
        if idempotency_key:
            existing = self.find_job_by_idempotency(endpoint_id, idempotency_key)
            if existing:
                return existing
        now = time.time()
        job_id = f"sjob-{uuid.uuid4().hex[:12]}"
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO serverless_jobs (
                        job_id, endpoint_id, owner_id, status, payload,
                        idempotency_key, webhook_url, queued_at, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        endpoint_id,
                        owner_id,
                        JOB_STATUS_QUEUED,
                        self._jsonb(payload),
                        idempotency_key,
                        webhook_url,
                        now,
                        now,
                        now,
                    ),
                )
        except Exception as exc:
            if idempotency_key and self._is_idempotency_violation(exc):
                existing = self.find_job_by_idempotency(endpoint_id, idempotency_key)
                if existing:
                    return existing
            raise
        row = self.get_job(job_id)
        assert row is not None
        return row

    @staticmethod
    def _is_idempotency_violation(exc: Exception) -> bool:
        name = type(exc).__name__
        if name == "UniqueViolation":
            return True
        cause = getattr(exc, "__cause__", None)
        return cause is not None and type(cause).__name__ == "UniqueViolation"

    def get_job(self, job_id: str, *, endpoint_id: str | None = None) -> dict | None:
        with self._conn() as conn:
            if endpoint_id:
                row = conn.execute(
                    """
                    SELECT * FROM serverless_jobs
                    WHERE job_id = %s AND endpoint_id = %s
                    """,
                    (job_id, endpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM serverless_jobs WHERE job_id = %s",
                    (job_id,),
                ).fetchone()
        return dict(row) if row else None

    def find_job_by_idempotency(self, endpoint_id: str, idempotency_key: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM serverless_jobs
                WHERE endpoint_id = %s AND idempotency_key = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (endpoint_id, idempotency_key),
            ).fetchone()
        return dict(row) if row else None

    def claim_next_job(self, endpoint_id: str, worker_id: str) -> dict | None:
        """Atomically claim the oldest queued job for an endpoint (SKIP LOCKED)."""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                """
                UPDATE serverless_jobs
                SET status = %s,
                    worker_id = %s,
                    started_at = %s,
                    updated_at = %s
                WHERE job_id = (
                    SELECT job_id FROM serverless_jobs
                    WHERE endpoint_id = %s AND status = %s
                    ORDER BY queued_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (
                    JOB_STATUS_IN_PROGRESS,
                    worker_id,
                    now,
                    now,
                    endpoint_id,
                    JOB_STATUS_QUEUED,
                ),
            ).fetchone()
        return dict(row) if row else None

    def complete_job(
        self,
        job_id: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        gpu_seconds: int = 0,
        cold_start_seconds: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        ttft_ms: int = 0,
        cost_cad: float = 0.0,
    ) -> dict | None:
        now = time.time()
        status = JOB_STATUS_FAILED if error else JOB_STATUS_COMPLETED
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_jobs
                SET status = %s,
                    output = %s,
                    error = %s,
                    finished_at = %s,
                    gpu_seconds = %s,
                    cold_start_seconds = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    cached_tokens = %s,
                    ttft_ms = %s,
                    cost_cad = %s,
                    updated_at = %s
                WHERE job_id = %s
                """,
                (
                    status,
                    self._jsonb(output) if output is not None else None,
                    error,
                    now,
                    gpu_seconds,
                    cold_start_seconds,
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                    ttft_ms,
                    cost_cad,
                    now,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def set_job_cold_start_seconds(self, job_id: str, cold_start_seconds: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_jobs
                SET cold_start_seconds = %s, updated_at = %s
                WHERE job_id = %s
                """,
                (max(0, int(cold_start_seconds)), time.time(), job_id),
            )

    def cancel_job(self, job_id: str, endpoint_id: str) -> dict | None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_jobs
                SET status = %s, finished_at = %s, updated_at = %s
                WHERE job_id = %s AND endpoint_id = %s
                  AND status IN (%s, %s)
                """,
                (
                    JOB_STATUS_CANCELLED,
                    now,
                    now,
                    job_id,
                    endpoint_id,
                    JOB_STATUS_QUEUED,
                    JOB_STATUS_IN_PROGRESS,
                ),
            )
        return self.get_job(job_id, endpoint_id=endpoint_id)

    def update_job_webhook_status(
        self,
        job_id: str,
        *,
        status: str,
        attempts: int | None = None,
        last_error: str | None = None,
        next_retry_at: float | None = None,
    ) -> None:
        fields: list[str] = ["webhook_status = %s", "updated_at = %s"]
        values: list[Any] = [status, time.time()]
        if attempts is not None:
            fields.append("webhook_attempts = %s")
            values.append(attempts)
        if last_error is not None:
            fields.append("webhook_last_error = %s")
            values.append(last_error[:500])
        if next_retry_at is not None:
            fields.append("webhook_next_retry_at = %s")
            values.append(next_retry_at)
        values.append(job_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE serverless_jobs SET {', '.join(fields)} WHERE job_id = %s",
                tuple(values),
            )

    def list_jobs_pending_webhook_retry(
        self,
        *,
        now: float,
        limit: int = 50,
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_jobs
                WHERE webhook_url IS NOT NULL
                  AND webhook_url != ''
                  AND status IN ('COMPLETED', 'FAILED', 'CANCELLED')
                  AND webhook_status = 'pending'
                  AND webhook_next_retry_at > 0
                  AND webhook_next_retry_at <= %s
                ORDER BY webhook_next_retry_at ASC
                LIMIT %s
                """,
                (now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def queue_depth(self, endpoint_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM serverless_jobs
                WHERE endpoint_id = %s AND status = %s
                """,
                (endpoint_id, JOB_STATUS_QUEUED),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def max_queue_wait_sec(self, endpoint_id: str) -> float:
        """Seconds since oldest queued job was enqueued (0 if queue empty)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT MIN(queued_at) AS oldest FROM serverless_jobs
                WHERE endpoint_id = %s AND status = %s
                """,
                (endpoint_id, JOB_STATUS_QUEUED),
            ).fetchone()
        if not row or row["oldest"] is None:
            return 0.0
        return max(0.0, time.time() - float(row["oldest"]))

    def list_jobs(
        self,
        endpoint_id: str,
        *,
        limit: int = 50,
        since_finished_at: float | None = None,
    ) -> list[dict]:
        with self._conn() as conn:
            if since_finished_at is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_jobs
                    WHERE endpoint_id = %s AND finished_at >= %s
                    ORDER BY finished_at DESC
                    LIMIT %s
                    """,
                    (endpoint_id, since_finished_at, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_jobs
                    WHERE endpoint_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (endpoint_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Stream events ─────────────────────────────────────────────────

    def append_stream_event(
        self,
        job_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        seq_no: int | None = None,
    ) -> dict:
        now = time.time()
        with self._conn() as conn:
            if seq_no is None:
                row = conn.execute(
                    """
                    SELECT COALESCE(MAX(seq_no), 0) + 1 AS next_seq
                    FROM serverless_job_stream_events
                    WHERE job_id = %s
                    """,
                    (job_id,),
                ).fetchone()
                seq_no = int(row["next_seq"]) if row else 1
            conn.execute(
                """
                INSERT INTO serverless_job_stream_events
                    (job_id, seq_no, event_type, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (job_id, seq_no) DO NOTHING
                """,
                (job_id, seq_no, event_type, self._jsonb(payload), now),
            )
        return {
            "job_id": job_id,
            "seq_no": seq_no,
            "event_type": event_type,
            "payload": payload,
            "created_at": now,
        }

    def list_stream_events(self, job_id: str, *, after_seq: int = 0) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_job_stream_events
                WHERE job_id = %s AND seq_no > %s
                ORDER BY seq_no ASC
                """,
                (job_id, after_seq),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── API keys ──────────────────────────────────────────────────────

    def create_api_key(
        self,
        owner_id: str,
        key_prefix: str,
        key_hash: str,
        *,
        endpoint_id: str | None = None,
        name: str = "default",
        scopes: str = "inference:write",
        rate_limit_rpm: int = 60,
    ) -> dict:
        now = time.time()
        key_id = f"skey-{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO serverless_api_keys (
                    key_id, endpoint_id, owner_id, name,
                    key_prefix, key_hash, scopes, rate_limit_rpm, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    key_id,
                    endpoint_id,
                    owner_id,
                    name,
                    key_prefix,
                    key_hash,
                    scopes,
                    rate_limit_rpm,
                    now,
                ),
            )
        return self.get_api_key(key_id) or {}

    def get_api_key(self, key_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM serverless_api_keys WHERE key_id = %s",
                (key_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_api_key_by_hash(self, key_hash: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM serverless_api_keys
                WHERE key_hash = %s AND revoked_at = 0
                """,
                (key_hash,),
            ).fetchone()
        return dict(row) if row else None

    def list_api_keys(self, owner_id: str, *, endpoint_id: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if endpoint_id:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_api_keys
                    WHERE owner_id = %s AND endpoint_id = %s AND revoked_at = 0
                    ORDER BY created_at DESC
                    """,
                    (owner_id, endpoint_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM serverless_api_keys
                    WHERE owner_id = %s AND revoked_at = 0
                    ORDER BY created_at DESC
                    """,
                    (owner_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def revoke_api_key(self, key_id: str, owner_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE serverless_api_keys
                SET revoked_at = %s
                WHERE key_id = %s AND owner_id = %s AND revoked_at = 0
                """,
                (now, key_id, owner_id),
            )
        return cur.rowcount > 0

    def touch_api_key(self, key_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE serverless_api_keys SET last_used_at = %s WHERE key_id = %s",
                (time.time(), key_id),
            )

    # ── Dispatcher / reconcile helpers ────────────────────────────────

    def find_ready_worker_with_capacity(self, endpoint_id: str, max_concurrency: int) -> dict | None:
        """Return a ready/idle worker with a free concurrency slot."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM serverless_workers
                WHERE endpoint_id = %s
                  AND state IN (%s, %s)
                  AND current_concurrency < %s
                ORDER BY current_concurrency ASC, last_heartbeat_at DESC
                LIMIT 1
                """,
                (
                    endpoint_id,
                    WORKER_STATE_READY,
                    WORKER_STATE_IDLE,
                    max_concurrency,
                ),
            ).fetchone()
        return dict(row) if row else None

    def increment_worker_concurrency(self, worker_id: str, delta: int = 1) -> dict | None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_workers
                SET current_concurrency = current_concurrency + %s,
                    state = CASE WHEN state = %s THEN %s ELSE state END,
                    updated_at = %s
                WHERE worker_id = %s
                """,
                (delta, WORKER_STATE_IDLE, WORKER_STATE_READY, time.time(), worker_id),
            )
        return self.get_worker(worker_id)

    def decrement_worker_concurrency(self, worker_id: str, delta: int = 1) -> dict | None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_workers
                SET current_concurrency = GREATEST(0, current_concurrency - %s),
                    state = CASE
                        WHEN current_concurrency - %s <= 0 THEN %s
                        ELSE state
                    END,
                    updated_at = %s
                WHERE worker_id = %s
                """,
                (
                    delta,
                    delta,
                    WORKER_STATE_IDLE,
                    time.time(),
                    worker_id,
                ),
            )
        return self.get_worker(worker_id)

    def requeue_job(self, job_id: str) -> dict | None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_jobs
                SET status = %s,
                    worker_id = NULL,
                    started_at = NULL,
                    updated_at = %s,
                    queued_at = %s
                WHERE job_id = %s AND status = %s
                """,
                (JOB_STATUS_QUEUED, now, now, job_id, JOB_STATUS_IN_PROGRESS),
            )
        return self.get_job(job_id)

    def requeue_worker_jobs(self, worker_id: str) -> int:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE serverless_jobs
                SET status = %s,
                    worker_id = NULL,
                    started_at = NULL,
                    updated_at = %s,
                    queued_at = %s
                WHERE worker_id = %s AND status = %s
                """,
                (JOB_STATUS_QUEUED, now, now, worker_id, JOB_STATUS_IN_PROGRESS),
            )
        return cur.rowcount

    def list_in_progress_jobs(self, worker_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_jobs
                WHERE worker_id = %s AND status = %s
                ORDER BY started_at ASC
                """,
                (worker_id, JOB_STATUS_IN_PROGRESS),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_workers_stale_heartbeat(self, *, heartbeat_before: float) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_workers
                WHERE state IN (%s, %s, %s)
                  AND current_concurrency = 0
                  AND (
                    last_heartbeat_at IS NULL
                    OR last_heartbeat_at < %s
                  )
                ORDER BY last_heartbeat_at ASC NULLS FIRST
                """,
                (
                    WORKER_STATE_READY,
                    WORKER_STATE_IDLE,
                    WORKER_STATE_DRAINING,
                    heartbeat_before,
                ),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_stuck_booting_workers(self, *, booting_before: float) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_workers
                WHERE state = %s AND allocated_at < %s
                ORDER BY allocated_at ASC
                """,
                (WORKER_STATE_BOOTING, booting_before),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_booting_workers(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_workers
                WHERE state = %s
                ORDER BY allocated_at ASC
                """,
                (WORKER_STATE_BOOTING,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_jobs_stale_queued(self, *, queued_before: float) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_jobs
                WHERE status = %s
                  AND queued_at IS NOT NULL
                  AND queued_at < %s
                ORDER BY queued_at ASC
                LIMIT 100
                """,
                (JOB_STATUS_QUEUED, queued_before),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_jobs_past_request_timeout(self, *, now: float) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT j.*, e.request_timeout_sec
                FROM serverless_jobs j
                JOIN serverless_endpoints e ON e.endpoint_id = j.endpoint_id
                WHERE j.status = %s
                  AND j.started_at IS NOT NULL
                  AND j.started_at < %s - COALESCE(e.request_timeout_sec, 120)
                ORDER BY j.started_at ASC
                """,
                (JOB_STATUS_IN_PROGRESS, now),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_endpoints_for_reconcile(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM serverless_endpoints
                WHERE deleted_at = 0
                  AND status NOT IN (%s, %s)
                ORDER BY created_at ASC
                """,
                (ENDPOINT_STATUS_DELETED, ENDPOINT_STATUS_ERROR),
            ).fetchall()
        return [dict(r) for r in rows]

    def increment_endpoint_totals(
        self,
        endpoint_id: str,
        *,
        requests: int = 0,
        gpu_seconds: int = 0,
        cost_cad: float = 0.0,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_endpoints
                SET total_requests = total_requests + %s,
                    total_gpu_seconds = total_gpu_seconds + %s,
                    total_cost_cad = total_cost_cad + %s,
                    updated_at = %s
                WHERE endpoint_id = %s
                """,
                (requests, gpu_seconds, cost_cad, time.time(), endpoint_id),
            )

    def accrue_endpoint_token_cost(self, endpoint_id: str, cost_cad: float) -> None:
        """Add request token cost to the endpoint's unbilled accrual (blended meter)."""
        if not endpoint_id or cost_cad <= 0:
            return
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_endpoints
                   SET unbilled_token_cost_cad = unbilled_token_cost_cad + %s,
                       updated_at = %s
                 WHERE endpoint_id = %s
                """,
                (float(cost_cad), time.time(), endpoint_id),
            )

    def token_usage_already_recorded(self, endpoint_id: str, idempotency_key: str) -> bool:
        if not endpoint_id or not idempotency_key:
            return False
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM serverless_token_ledger
                WHERE endpoint_id = %s AND idempotency_key = %s
                LIMIT 1
                """,
                (endpoint_id, idempotency_key),
            ).fetchone()
        return row is not None

    def record_token_usage_idempotency(
        self,
        endpoint_id: str,
        idempotency_key: str,
        meta: dict[str, Any],
    ) -> None:
        if not endpoint_id or not idempotency_key:
            return
        import uuid

        now = time.time()
        ledger_id = f"stl-{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO serverless_token_ledger
                    (ledger_id, endpoint_id, idempotency_key,
                     input_tokens, output_tokens, cached_tokens, cost_cad, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (endpoint_id, idempotency_key) DO NOTHING
                """,
                (
                    ledger_id,
                    endpoint_id,
                    idempotency_key,
                    int(meta.get("input_tokens") or 0),
                    int(meta.get("output_tokens") or 0),
                    int(meta.get("cached_tokens") or 0),
                    float(meta.get("total_token_cost_cad") or 0.0),
                    now,
                ),
            )

    def consume_endpoint_token_cost(self, endpoint_id: str) -> float:
        """Atomically read and zero the endpoint's accrued unbilled token cost,
        returning the amount consumed (for the blended meter's billing slice)."""
        if not endpoint_id:
            return 0.0
        with self._conn() as conn:
            row = conn.execute(
                """
                WITH old AS (
                    SELECT unbilled_token_cost_cad AS v
                    FROM serverless_endpoints
                    WHERE endpoint_id = %s
                    FOR UPDATE
                )
                UPDATE serverless_endpoints e
                   SET unbilled_token_cost_cad = 0, updated_at = %s
                  FROM old
                 WHERE e.endpoint_id = %s
                RETURNING old.v
                """,
                (endpoint_id, time.time(), endpoint_id),
            ).fetchone()
        if not row:
            return 0.0
        return float(row["v"] or 0.0)

    def get_worker_job_row(self, worker_id: str) -> dict | None:
        """Join serverless_workers → jobs → host IP for proxy routing."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT w.*,
                       j.status AS job_status,
                       j.payload AS job_payload,
                       h.payload->>'ip' AS host_ip
                FROM serverless_workers w
                LEFT JOIN jobs j ON j.job_id = w.scheduler_job_id
                LEFT JOIN hosts h ON h.host_id = j.host_id
                WHERE w.worker_id = %s
                """,
                (worker_id,),
            ).fetchone()
        return dict(row) if row else None

    def try_advisory_lock(self, lock_key: int = 0x534C5652) -> bool:
        """Postgres advisory lock for single-writer reconcile (SLVR)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired",
                (lock_key,),
            ).fetchone()
        return bool(row and row.get("acquired"))

    def release_advisory_lock(self, lock_key: int = 0x534C5652) -> None:
        with self._conn() as conn:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))