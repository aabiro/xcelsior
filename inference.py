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
        min_workers: int = 0,
        max_workers: int = 4,
        max_batch_size: int = 8,
        max_concurrent: int = 4,
        scaledown_window_sec: int = SCALEDOWN_WINDOW_SEC,
    ) -> dict:
        """Create a serverless inference endpoint."""
        now = time.time()
        endpoint_id = f"ep-{uuid.uuid4().hex[:12]}"

        # Estimate VRAM required
        vram_required = self._estimate_vram_gb(model_id)

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO inference_endpoints
                   (endpoint_id, owner_id, model_id, gpu_type, vram_required_gb,
                    max_batch_size, max_concurrent, min_workers, max_workers,
                    scaledown_window_sec, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)""",
                (endpoint_id, owner_id, model_id, gpu_type, vram_required,
                 max_batch_size, max_concurrent, min_workers, max_workers,
                 scaledown_window_sec, now, now),
            )

        log.info("Inference endpoint created: %s model=%s owner=%s", endpoint_id, model_id, owner_id)
        return {
            "endpoint_id": endpoint_id,
            "model_id": model_id,
            "status": "active",
            "min_workers": min_workers,
            "max_workers": max_workers,
        }

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
        with self._conn() as conn:
            conn.execute(
                "UPDATE inference_endpoints SET status = 'deleted', updated_at = %s WHERE endpoint_id = %s",
                (time.time(), endpoint_id),
            )

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
                """SELECT h.host_id, h.free_vram_gb
                   FROM hosts h
                   WHERE h.status = 'active'
                     AND h.free_vram_gb >= %s
                   ORDER BY h.free_vram_gb DESC
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
                conn.execute(
                    "UPDATE worker_model_cache SET state = 'evicting' WHERE worker_id = %s AND model_id = %s",
                    (row["worker_id"], row["model_id"]),
                )
                evicted += 1

        if evicted:
            log.info("INFERENCE SCALEDOWN: evicting %d idle models", evicted)
        return evicted

    def get_active_workers_for_model(self, model_id: str) -> list[dict]:
        """Get workers currently serving a model."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT wmc.*, h.gpu_model, h.free_vram_gb
                   FROM worker_model_cache wmc
                   JOIN hosts h ON h.host_id = wmc.worker_id
                   WHERE wmc.model_id = %s AND wmc.state = 'ready'""",
                (model_id,),
            ).fetchall()
            return [dict(r) for r in rows]

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
