# Xcelsior Serverless Inference Engine
# Production OpenAI-compatible inference with worker routing, model caching,
# auto-scaling, and per-token billing.
#
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md:
# - /v1/completions and /v1/chat/completions endpoints
# - Warm worker routing (prefer workers with model already loaded)
# - Automatic scale-to-zero (cold start with model download)
# - Per-token billing with metering
# - Streaming support via SSE
# - Token counting via tiktoken-compatible estimation

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("xcelsior.inference")

# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS = int(os.environ.get("XCELSIOR_INFERENCE_MAX_TOKENS", "2048"))
DEFAULT_TIMEOUT_SEC = int(os.environ.get("XCELSIOR_INFERENCE_TIMEOUT", "120"))
# Price per 1M tokens (input/output)
INPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_INPUT_TOKEN_PRICE", "0.50"))
OUTPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_OUTPUT_TOKEN_PRICE", "1.50"))
SCALEDOWN_WINDOW_SEC = int(os.environ.get("XCELSIOR_INFERENCE_SCALEDOWN", "300"))


@dataclass
class InferenceRequest:
    """OpenAI-compatible inference request."""

    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}")
    endpoint_id: str = ""
    model: str = ""
    messages: list = field(default_factory=list)  # For chat completions
    prompt: str = ""  # For text completions
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: list = field(default_factory=list)
    user: str = ""  # customer_id for billing


@dataclass
class InferenceResponse:
    """OpenAI-compatible inference response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    # Xcelsior-specific
    worker_id: str = ""
    latency_ms: float = 0
    cold_start: bool = False


class InferenceEngine:
    """Serverless inference engine with model caching and warm routing."""

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

    # ── Endpoint Management ───────────────────────────────────────────

    def create_endpoint(
        self,
        owner_id: str,
        model_id: str,
        gpu_type: str = "",
        region: str = "ca-east",
        docker_image: str = "xcelsior/vllm:latest",
        min_workers: int = 0,
        max_workers: int = 4,
        max_batch_size: int = 8,
        max_concurrent: int = 4,
        scaledown_window_sec: int = SCALEDOWN_WINDOW_SEC,
        mode: str = "sync",
        health_endpoint: str = "/health",
        api_format: str = "openai",
    ) -> dict:
        """Create a serverless inference endpoint.

        Validates GPU availability, stores all configuration, and optionally
        provisions initial workers if min_workers >= 1.
        """
        now = time.time()
        endpoint_id = f"ep-{uuid.uuid4().hex[:12]}"

        # Estimate VRAM required
        vram_required = self._estimate_vram_gb(model_id)

        # Validate GPU type if specified
        if gpu_type:
            available = self._check_gpu_available(gpu_type, region)
            if not available:
                raise ValueError(f"No GPUs of type '{gpu_type}' available in region '{region}'")

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO inference_endpoints
                   (endpoint_id, owner_id, model_id, gpu_type, vram_required_gb,
                    max_batch_size, max_concurrent, min_workers, max_workers,
                    scaledown_window_sec, status,
                    docker_image, mode, health_endpoint, api_format, region,
                    total_cost_cad,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active',
                           %s, %s, %s, %s, %s,
                           0.0,
                           %s, %s)""",
                (
                    endpoint_id,
                    owner_id,
                    model_id,
                    gpu_type,
                    vram_required,
                    max_batch_size,
                    max_concurrent,
                    min_workers,
                    max_workers,
                    scaledown_window_sec,
                    docker_image,
                    mode,
                    health_endpoint,
                    api_format,
                    region,
                    now,
                    now,
                ),
            )

        log.info(
            "Inference endpoint created: %s model=%s gpu=%s region=%s owner=%s",
            endpoint_id,
            model_id,
            gpu_type,
            region,
            owner_id,
        )

        # Provision initial workers if min_workers > 0
        worker_job_id = None
        if min_workers >= 1:
            worker_job_id = self.provision_worker(
                endpoint_id,
                model_id,
                gpu_type,
                vram_required,
                region,
                docker_image,
                owner_id=owner_id,
            )

        # Look up cost_per_hour for the GPU type
        cost_per_hour = self._get_gpu_cost_per_hour(gpu_type, region)

        return {
            "endpoint_id": endpoint_id,
            "model_id": model_id,
            "gpu_type": gpu_type,
            "region": region,
            "docker_image": docker_image,
            "mode": mode,
            "status": "active",
            "min_workers": min_workers,
            "max_workers": max_workers,
            "worker_job_id": worker_job_id,
            "cost_per_hour_cad": cost_per_hour,
        }

    def provision_worker(
        self,
        endpoint_id: str,
        model_id: str,
        gpu_type: str,
        vram_gb: float,
        region: str,
        docker_image: str,
        owner_id: str = "",
    ) -> Optional[str]:
        """Provision a GPU worker for an inference endpoint via the scheduler.

        Submits the job, processes the queue to assign a host, and polls
        for container readiness before returning.
        """
        try:
            from scheduler import (
                submit_job,
                process_queue,
                list_jobs,
                check_job_running,
                list_hosts,
            )

            job = submit_job(
                name=f"inference-{endpoint_id}",
                vram_needed_gb=vram_gb,
                priority=5,
                tier="on-demand",
                num_gpus=1,
                image=docker_image,
                command=f"serve --model {model_id}",
                owner=owner_id,
            )
            worker_job_id = job.get("job_id", "")

            # Process queue to assign + start the job
            process_queue()

            # Poll for container readiness (up to 60s, 3s interval)
            started = False
            for _ in range(20):
                time.sleep(3)
                # Re-read job from DB to get updated host_id + container info
                current_job = None
                for j in list_jobs():
                    if j["job_id"] == worker_job_id:
                        current_job = j
                        break
                if not current_job:
                    break
                if current_job.get("status") == "running":
                    started = True
                    break
                if current_job.get("status") in ("failed", "cancelled"):
                    break

            # Store worker_job_id on the endpoint
            with self._conn() as conn:
                new_status = "active" if started else "error"
                conn.execute(
                    """UPDATE inference_endpoints
                       SET worker_job_id = %s, status = %s, updated_at = %s
                       WHERE endpoint_id = %s""",
                    (worker_job_id, new_status, time.time(), endpoint_id),
                )

            if not started:
                log.error(
                    "Worker container did not start within 60s for endpoint %s (job %s)",
                    endpoint_id,
                    worker_job_id,
                )
                return worker_job_id  # Return job_id so caller can inspect

            log.info(
                "Worker provisioned and running for endpoint %s: job_id=%s",
                endpoint_id,
                worker_job_id,
            )
            return worker_job_id
        except Exception as e:
            log.error("Failed to provision worker for %s: %s", endpoint_id, e)
            with self._conn() as conn:
                conn.execute(
                    "UPDATE inference_endpoints SET status = 'error', updated_at = %s WHERE endpoint_id = %s",
                    (time.time(), endpoint_id),
                )
            return None

    def deprovision_worker(self, endpoint_id: str):
        """Stop the worker job for an inference endpoint.

        Kills the running container on the host via the scheduler, then cleans
        up the DB records.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT worker_job_id FROM inference_endpoints WHERE endpoint_id = %s",
                (endpoint_id,),
            ).fetchone()
            if row and row.get("worker_job_id"):
                worker_job_id = row["worker_job_id"]
                # Actually kill the container via scheduler
                try:
                    from scheduler import kill_job as scheduler_kill

                    # Look up the job + host to pass to kill_job
                    job_row = conn.execute(
                        "SELECT j.*, h.payload->>'ip' AS ip FROM jobs j JOIN hosts h ON j.host_id = h.host_id WHERE j.job_id = %s",
                        (worker_job_id,),
                    ).fetchone()
                    if job_row:
                        scheduler_kill(dict(job_row), dict(job_row))
                        log.info(
                            "Killed worker container for endpoint %s (job %s)",
                            endpoint_id,
                            worker_job_id,
                        )
                    else:
                        # Job not found in DB — mark cancelled anyway
                        log.warning(
                            "Worker job %s not found in jobs table, skipping kill", worker_job_id
                        )
                except Exception as e:
                    log.error("Failed to kill worker job %s: %s", worker_job_id, e)

                conn.execute(
                    "UPDATE inference_endpoints SET worker_job_id = NULL, updated_at = %s WHERE endpoint_id = %s",
                    (time.time(), endpoint_id),
                )
                log.info("Worker deprovisioned for endpoint %s", endpoint_id)

    def _check_gpu_available(self, gpu_type: str, region: str) -> bool:
        """Check if a GPU type is available in the given region."""
        with self._conn() as conn:
            # Check gpu_offers first
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM gpu_offers
                   WHERE gpu_model = %s AND region = %s AND available = true""",
                (gpu_type, region),
            ).fetchone()
            if row and row["cnt"] > 0:
                return True
            # Fall back to hosts table
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM hosts
                   WHERE payload->>'gpu_model' = %s AND status = 'active'""",
                (gpu_type,),
            ).fetchone()
            return bool(row and row["cnt"] > 0)

    def _get_gpu_cost_per_hour(self, gpu_type: str, region: str) -> float:
        """Get the cost per hour for a GPU type in a region."""
        if not gpu_type:
            return 0.0
        with self._conn() as conn:
            row = conn.execute(
                """SELECT MIN(ask_cents_per_hour) as price FROM gpu_offers
                   WHERE gpu_model = %s AND region = %s AND available = true""",
                (gpu_type, region),
            ).fetchone()
            if row and row.get("price"):
                return round(float(row["price"]) / 100.0, 2)
            # Fall back to hosts table
            row = conn.execute(
                """SELECT MIN((payload->>'cost_per_hour')::float) as price FROM hosts
                   WHERE payload->>'gpu_model' = %s AND status = 'active'""",
                (gpu_type,),
            ).fetchone()
            if row and row.get("price"):
                return round(float(row["price"]), 2)
        # No pricing data in DB for this GPU type
        log.warning("No pricing data found for GPU %s in region %s", gpu_type, region)
        return 0.0

    def get_endpoint(self, endpoint_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM inference_endpoints WHERE endpoint_id = %s",
                (endpoint_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_endpoints(self, owner_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM inference_endpoints WHERE owner_id = %s AND status != 'deleted' ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_endpoint(self, endpoint_id: str):
        # Deprovision any running workers first
        self.deprovision_worker(endpoint_id)
        with self._conn() as conn:
            conn.execute(
                "UPDATE inference_endpoints SET status = 'deleted', updated_at = %s WHERE endpoint_id = %s",
                (time.time(), endpoint_id),
            )

    def get_endpoint_health(self, endpoint_id: str) -> dict:
        """Get health status for an endpoint including its workers.

        Goes beyond DB state: verifies the worker job is actually running
        via the scheduler.
        """
        with self._conn() as conn:
            ep = conn.execute(
                "SELECT * FROM inference_endpoints WHERE endpoint_id = %s",
                (endpoint_id,),
            ).fetchone()
            if not ep:
                return {"status": "not_found"}

            # Check if the main worker job is actually running
            worker_alive = False
            worker_job_id = ep.get("worker_job_id")
            if worker_job_id:
                try:
                    from scheduler import list_jobs, check_job_running, list_hosts

                    # Look up the job and its host to check container state
                    job_data = None
                    for j in list_jobs():
                        if j["job_id"] == worker_job_id:
                            job_data = j
                            break
                    if job_data and job_data.get("status") == "running" and job_data.get("host_id"):
                        hosts = list_hosts()
                        hmap = {h["host_id"]: h for h in hosts}
                        host = hmap.get(job_data["host_id"])
                        if host:
                            worker_alive = check_job_running(job_data, host)
                    elif job_data and job_data.get("status") == "running":
                        worker_alive = True  # Running but can't verify container
                except Exception as e:
                    log.warning("Could not verify worker job %s: %s", worker_job_id, e)

            # Count active workers for this model
            workers = conn.execute(
                """SELECT wmc.worker_id, wmc.state, wmc.last_used_at, h.payload->>'gpu_model' AS gpu_model, (h.payload->>'free_vram_gb')::float AS free_vram_gb
                   FROM worker_model_cache wmc
                   JOIN hosts h ON h.host_id = wmc.worker_id
                   WHERE wmc.model_id = %s AND wmc.state IN ('ready', 'loading')""",
                (ep["model_id"],),
            ).fetchall()

            warm_workers = [w for w in workers if w["state"] == "ready"]
            loading_workers = [w for w in workers if w["state"] == "loading"]

            return {
                "endpoint_id": endpoint_id,
                "status": ep["status"],
                "model_id": ep["model_id"],
                "worker_alive": worker_alive,
                "warm_workers": len(warm_workers),
                "loading_workers": len(loading_workers),
                "total_workers": len(workers),
                "min_workers": ep.get("min_workers", 0),
                "max_workers": ep.get("max_workers", 3),
                "worker_job_id": worker_job_id,
                "workers": [
                    {
                        "worker_id": w["worker_id"],
                        "state": w["state"],
                        "gpu_model": w.get("gpu_model", ""),
                        "last_used": w.get("last_used_at", 0),
                    }
                    for w in workers
                ],
            }

    def get_endpoint_usage(self, endpoint_id: str) -> dict:
        """Get usage statistics for an endpoint."""
        with self._conn() as conn:
            ep = conn.execute(
                "SELECT * FROM inference_endpoints WHERE endpoint_id = %s",
                (endpoint_id,),
            ).fetchone()
            if not ep:
                return {}

            # Get recent inference jobs for this endpoint's model
            recent_jobs = conn.execute(
                """SELECT COUNT(*) as total, AVG(latency_ms) as avg_latency,
                          SUM(input_tokens) as total_input, SUM(output_tokens) as total_output
                   FROM inference_results
                   WHERE model = %s AND COALESCE(created_at, completed_at) > %s""",
                (ep["model_id"], time.time() - 86400),  # Last 24h
            ).fetchone()

            cost_per_hour = self._get_gpu_cost_per_hour(
                ep.get("gpu_type", ""), ep.get("region", "ca-east")
            )

            return {
                "endpoint_id": endpoint_id,
                "total_requests": ep.get("total_requests", 0),
                "total_tokens_generated": ep.get("total_tokens_generated", 0),
                "total_cost_cad": float(ep.get("total_cost_cad", 0)),
                "cost_per_hour_cad": cost_per_hour,
                "last_24h": {
                    "requests": int(recent_jobs["total"] or 0) if recent_jobs else 0,
                    "avg_latency_ms": (
                        round(float(recent_jobs["avg_latency"] or 0), 1) if recent_jobs else 0
                    ),
                    "input_tokens": int(recent_jobs["total_input"] or 0) if recent_jobs else 0,
                    "output_tokens": int(recent_jobs["total_output"] or 0) if recent_jobs else 0,
                },
                "created_at": ep.get("created_at", 0),
                "updated_at": ep.get("updated_at", 0),
            }

    # ── Inference Routing ─────────────────────────────────────────────

    def route_request(self, req: InferenceRequest) -> Optional[dict]:
        """Find the best worker for an inference request.

        Routing priority (Phase 3.2):
        1. Worker with model already loaded (warm hit)
        2. Worker with most free VRAM (best capacity for cold start)
        3. None (queue the request)
        """
        model_id = req.model
        vram_needed = self._estimate_vram_gb(model_id)

        with self._conn() as conn:
            # Priority 1: Warm worker with model already loaded
            warm = conn.execute(
                """SELECT wmc.worker_id, wmc.last_used_at
                   FROM worker_model_cache wmc
                   JOIN hosts h ON h.host_id = wmc.worker_id
                   WHERE wmc.model_id = %s AND wmc.state = 'ready'
                     AND h.status = 'active'
                   ORDER BY wmc.last_used_at DESC
                   LIMIT 1""",
                (model_id,),
            ).fetchone()

            if warm:
                # Update last_used_at
                conn.execute(
                    "UPDATE worker_model_cache SET last_used_at = %s WHERE worker_id = %s AND model_id = %s",
                    (time.time(), warm["worker_id"], model_id),
                )
                return {"worker_id": warm["worker_id"], "cold_start": False}

            # Priority 2: Worker with most free VRAM (cold start on best-capacity worker)
            cold = conn.execute(
                """SELECT h.host_id, (h.payload->>'free_vram_gb')::float AS free_vram_gb
                   FROM hosts h
                   WHERE h.status = 'active'
                     AND (h.payload->>'free_vram_gb')::float >= %s
                   ORDER BY (h.payload->>'free_vram_gb')::float DESC
                   LIMIT 1""",
                (vram_needed,),
            ).fetchone()

            if cold:
                # Register model loading
                conn.execute(
                    """INSERT INTO worker_model_cache
                       (worker_id, model_id, state, loaded_at, last_used_at)
                       VALUES (%s, %s, 'loading', %s, %s)
                       ON CONFLICT (worker_id, model_id, model_revision) DO UPDATE
                       SET state = 'loading', loaded_at = EXCLUDED.loaded_at""",
                    (cold["host_id"], model_id, time.time(), time.time()),
                )
                return {"worker_id": cold["host_id"], "cold_start": True}

        return None  # No suitable worker

    def mark_model_ready(self, worker_id: str, model_id: str, vram_bytes: int = 0):
        """Called by worker after model is fully loaded and ready."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE worker_model_cache
                   SET state = 'ready', vram_bytes = %s, last_used_at = %s
                   WHERE worker_id = %s AND model_id = %s""",
                (vram_bytes, time.time(), worker_id, model_id),
            )

    def mark_model_error(self, worker_id: str, model_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE worker_model_cache SET state = 'error' WHERE worker_id = %s AND model_id = %s",
                (worker_id, model_id),
            )

    # ── Request Processing ────────────────────────────────────────────

    def submit_request(self, req: InferenceRequest) -> dict:
        """Submit an inference request. Returns request tracking info."""
        from inference_store import store_inference_job

        # Validate
        if not req.model:
            return {"error": "model is required", "status": 400}

        # Store the request
        inputs = req.messages if req.messages else [{"role": "user", "content": req.prompt}]
        store_inference_job(
            job_id=req.request_id,
            customer_id=req.user,
            model=req.model,
            inputs=inputs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            timeout_sec=DEFAULT_TIMEOUT_SEC,
        )

        # Try to route immediately
        route = self.route_request(req)
        if route:
            return {
                "request_id": req.request_id,
                "status": "processing",
                "worker_id": route["worker_id"],
                "cold_start": route["cold_start"],
            }

        return {
            "request_id": req.request_id,
            "status": "queued",
            "message": "No warm workers available, request queued",
        }

    def complete_request(
        self,
        request_id: str,
        outputs: list,
        model: str,
        worker_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> dict:
        """Called when a worker completes an inference request."""
        from inference_store import store_inference_result, get_inference_job

        store_inference_result(request_id, outputs, model, latency_ms)

        # Update endpoint stats
        job = get_inference_job(request_id)
        if job:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE inference_endpoints
                       SET total_requests = total_requests + 1,
                           total_tokens_generated = total_tokens_generated + %s,
                           updated_at = %s
                       WHERE model_id = %s AND status = 'active'""",
                    (output_tokens, time.time(), model),
                )

        # Bill for tokens
        if job and job.get("customer_id"):
            cost = self.compute_token_cost(input_tokens, output_tokens)
            if cost > 0:
                from billing import get_billing_engine

                engine = get_billing_engine()
                engine.charge(
                    job["customer_id"],
                    cost,
                    job_id=request_id,
                    description=f"Inference: {model} ({input_tokens}+{output_tokens} tokens)",
                )

        return {
            "request_id": request_id,
            "status": "completed",
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    # ── Token Billing ─────────────────────────────────────────────────

    def compute_token_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute cost in CAD for a given token count."""
        input_cost = (input_tokens / 1_000_000) * INPUT_TOKEN_PRICE_CAD_PER_M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE_CAD_PER_M
        return round(input_cost + output_cost, 6)

    # ── Auto-Scaling ──────────────────────────────────────────────────

    def scaledown_idle_workers(self) -> int:
        """Evict models from workers that haven't been used recently.

        Respects min_workers — never scales below the configured minimum.
        Called periodically by the background scheduler.
        """
        cutoff = time.time() - SCALEDOWN_WINDOW_SEC
        evicted = 0
        with self._conn() as conn:
            stale = conn.execute(
                """SELECT worker_id, model_id FROM worker_model_cache
                   WHERE state = 'ready' AND last_used_at < %s""",
                (cutoff,),
            ).fetchall()

            for row in stale:
                model_id = row["model_id"]
                # Check if this model has an endpoint with min_workers
                ep = conn.execute(
                    """SELECT min_workers FROM inference_endpoints
                       WHERE model_id = %s AND status = 'active'
                       ORDER BY min_workers DESC LIMIT 1""",
                    (model_id,),
                ).fetchone()
                min_w = int(ep["min_workers"]) if ep else 0

                if min_w > 0:
                    # Count current ready workers for this model
                    count_row = conn.execute(
                        """SELECT COUNT(*) as cnt FROM worker_model_cache
                           WHERE model_id = %s AND state = 'ready'""",
                        (model_id,),
                    ).fetchone()
                    current = int(count_row["cnt"]) if count_row else 0
                    if current <= min_w:
                        continue  # Don't scale below min_workers

                conn.execute(
                    "UPDATE worker_model_cache SET state = 'evicting' WHERE worker_id = %s AND model_id = %s",
                    (row["worker_id"], row["model_id"]),
                )
                evicted += 1

        # Actually stop the evicted worker containers
        if evicted:
            self._kill_evicted_workers()
            log.info("INFERENCE SCALEDOWN: evicted %d idle models", evicted)
        return evicted

    def get_active_workers_for_model(self, model_id: str) -> list[dict]:
        """Get workers currently serving a model."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT wmc.*, h.payload->>'gpu_model' AS gpu_model, (h.payload->>'free_vram_gb')::float AS free_vram_gb
                   FROM worker_model_cache wmc
                   JOIN hosts h ON h.host_id = wmc.worker_id
                   WHERE wmc.model_id = %s AND wmc.state = 'ready'""",
                (model_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def _kill_evicted_workers(self):
        """Kill containers for workers in 'evicting' state, then remove the cache entry."""
        try:
            from scheduler import kill_job as scheduler_kill
        except ImportError:
            log.error("Cannot import scheduler.kill_job — evicted workers not stopped")
            return

        with self._conn() as conn:
            evicting = conn.execute(
                """SELECT wmc.worker_id, wmc.model_id, j.job_id, h.payload->>'ip' AS ip
                   FROM worker_model_cache wmc
                   JOIN hosts h ON h.host_id = wmc.worker_id
                   LEFT JOIN jobs j ON j.host_id = h.host_id
                     AND j.payload->>'name' LIKE 'inference-%' AND j.status = 'running'
                   WHERE wmc.state = 'evicting'""",
            ).fetchall()

        for row in evicting:
            if row.get("job_id") and row.get("ip"):
                try:
                    scheduler_kill(
                        {"job_id": row["job_id"], "name": f"inference-evict-{row['worker_id']}"},
                        {"ip": row["ip"]},
                    )
                    log.info(
                        "Killed evicted worker container: worker=%s job=%s",
                        row["worker_id"],
                        row["job_id"],
                    )
                except Exception as e:
                    log.error("Failed to kill evicted worker %s: %s", row["worker_id"], e)

            # Clean up cache entry
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM worker_model_cache WHERE worker_id = %s AND model_id = %s AND state = 'evicting'",
                    (row["worker_id"], row["model_id"]),
                )

    # ── Helpers ───────────────────────────────────────────────────────

    def _estimate_vram_gb(self, model_id: str) -> float:
        """Rough VRAM estimate for a model. Proper solution: model registry."""
        model_lower = model_id.lower()
        # Common model sizes (FP16)
        if "70b" in model_lower:
            return 40.0
        elif "34b" in model_lower:
            return 20.0
        elif "13b" in model_lower:
            return 10.0
        elif "7b" in model_lower:
            return 6.0
        elif "3b" in model_lower:
            return 3.0
        elif "1b" in model_lower or "1.5b" in model_lower:
            return 2.0
        elif "distilbert" in model_lower or "bert" in model_lower:
            return 1.0
        return 8.0  # Conservative default


# ── Singleton ─────────────────────────────────────────────────────────

_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine


if __name__ == "__main__":
    engine = get_inference_engine()
    print("Inference engine ready:", engine)
