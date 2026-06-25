# Xcelsior — Serverless inference service layer

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from serverless.autoscaler import (
    DRAIN_GRACE_SEC,
    SCALE_DOWN_COOLDOWN_SEC,
    AutoscalerInput,
    compute_desired_workers,
    scale_down_cooldown_active,
    workers_to_mark_draining,
    workers_to_reap,
)
from serverless.cache import (
    attach_cache_to_endpoint_spec,
    replicate_cache_volumes,
)
from serverless.github_deploy import GitHubSourceError, apply_github_source
from serverless.dispatcher import ServerlessDispatcher
from serverless.metering import (
    charge_serverless_worker,
    pricing_for_endpoint,
    record_cold_start_line_item,
    token_cost_metadata,
)
from serverless.repo import (
    ENDPOINT_STATUS_ACTIVE,
    ENDPOINT_STATUS_ERROR,
    ENDPOINT_STATUS_PROVISIONING,
    ENDPOINT_STATUS_SCALED_DOWN,
    EndpointCreate,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_QUEUED,
    ServerlessRepo,
    WORKER_STATE_BOOTING,
    WORKER_STATE_DRAINING,
    WORKER_STATE_ERROR,
    WORKER_STATE_IDLE,
    WORKER_STATE_READY,
    WORKER_STATE_TERMINATED,
)

log = logging.getLogger("xcelsior.serverless.service")

_QUEUE_DEPTH_HISTORY: dict[str, list[tuple[float, int]]] = {}
_QUEUE_HISTORY_MAX = 20


def _record_queue_depth_sample(endpoint_id: str, depth: int) -> list[tuple[float, int]]:
    hist = _QUEUE_DEPTH_HISTORY.setdefault(endpoint_id, [])
    hist.append((time.time(), depth))
    if len(hist) > _QUEUE_HISTORY_MAX:
        del hist[: len(hist) - _QUEUE_HISTORY_MAX]
    return list(hist)

DEFAULT_VLLM_IMAGE = "xcelsior/serverless-vllm:12.4"

MANAGED_ENGINE_IMAGES: dict[str, str] = {
    "vllm": DEFAULT_VLLM_IMAGE,
    "tgi": "ghcr.io/huggingface/text-generation-inference:latest",
    "sglang": "lmsysorg/sglang:latest",
}
ALLOWED_MANAGED_ENGINES = frozenset(MANAGED_ENGINE_IMAGES.keys())


def _preset_image_for_engine(managed_engine: str) -> str:
    return MANAGED_ENGINE_IMAGES.get(managed_engine, DEFAULT_VLLM_IMAGE)


def _preset_startup_command(managed_engine: str, model_ref: str) -> str:
    if managed_engine == "tgi":
        return f"--model-id {model_ref}"
    if managed_engine == "sglang":
        return f"--model-path {model_ref} --port 8080"
    return (
        f"--model {model_ref} --max-model-len 4096 "
        f"--chat-template-content-format openai"
    )


@dataclass
class WalletPreflightError(Exception):
    message: str
    status_code: int = 402


class ServerlessService:
    def __init__(self, repo: ServerlessRepo | None = None):
        self.repo = repo or ServerlessRepo()
        self.dispatcher = ServerlessDispatcher(self.repo)

    # ── Wallet / validation ───────────────────────────────────────────

    @staticmethod
    def wallet_preflight(customer_id: str) -> None:
        from billing import get_billing_engine

        if not customer_id:
            raise WalletPreflightError("Missing billing customer", 400)
        wallet = get_billing_engine().get_wallet(customer_id)
        if wallet.get("status") == "suspended":
            raise WalletPreflightError(
                "Wallet suspended — please add funds to resume service",
                402,
            )
        if wallet["balance_cad"] <= 0 and float(wallet.get("grace_until") or 0) < time.time():
            raise WalletPreflightError(
                "Insufficient wallet balance — please deposit credits",
                402,
            )

    @staticmethod
    def validate_endpoint_spec(spec: EndpointCreate) -> None:
        if spec.mode not in ("preset", "custom"):
            raise ValueError("mode must be 'preset' or 'custom'")
        if spec.mode == "preset" and not spec.model_ref:
            raise ValueError("model_ref is required for preset endpoints")
        if spec.mode == "preset" and spec.managed_engine not in ALLOWED_MANAGED_ENGINES:
            raise ValueError(
                f"managed_engine must be one of: {', '.join(sorted(ALLOWED_MANAGED_ENGINES))}"
            )
        if spec.mode == "custom" and not spec.image_ref:
            if (spec.source_type or "").strip().lower() != "github" or not (spec.source_ref or "").strip():
                raise ValueError("image_ref is required for custom endpoints")
        if spec.min_workers < 0 or spec.max_workers < spec.min_workers:
            raise ValueError("invalid min_workers / max_workers")
        if spec.scaling_policy_type not in ("queue_request_count", "queue_delay"):
            raise ValueError("scaling_policy_type must be queue_request_count or queue_delay")

    @staticmethod
    def estimate_vram_gb(model_ref: str) -> float:
        m = model_ref.lower()
        if "70b" in m:
            return 40.0
        if "34b" in m:
            return 20.0
        if "13b" in m:
            return 10.0
        if "8b" in m or "7b" in m:
            return 6.0
        if "3b" in m:
            return 3.0
        if "1b" in m or "1.5b" in m:
            return 2.0
        return 8.0

    # ── Endpoint lifecycle ────────────────────────────────────────────

    def create_endpoint(self, spec: EndpointCreate, *, emit_sse: bool = True) -> dict:
        self.validate_endpoint_spec(spec)
        self.wallet_preflight(spec.owner_id)

        if spec.mode == "preset" and not spec.image_ref:
            spec.image_ref = _preset_image_for_engine(spec.managed_engine)
        if spec.mode == "preset" and spec.vram_required_gb <= 0:
            spec.vram_required_gb = self.estimate_vram_gb(spec.model_ref)
        if spec.mode == "preset" and not spec.startup_command:
            spec.startup_command = _preset_startup_command(spec.managed_engine, spec.model_ref)

        if spec.mode == "custom" and (spec.source_type or "").strip().lower() == "github":
            try:
                image, gh_env = apply_github_source(
                    mode=spec.mode,
                    image_ref=spec.image_ref,
                    source_type=spec.source_type,
                    source_ref=spec.source_ref,
                    source_ref_branch=spec.source_ref_branch,
                )
            except GitHubSourceError as exc:
                raise ValueError(str(exc)) from exc
            spec.image_ref = image
            spec.env.update(gh_env)

        cache_by_region = replicate_cache_volumes(
            self.repo,
            spec.owner_id,
            spec.model_ref,
            spec.model_revision,
            spec.region,
        )
        attach_cache_to_endpoint_spec(spec, cache_by_region.get(spec.region))
        peer_regions = [r for r, vid in cache_by_region.items() if r != spec.region and vid]
        if peer_regions:
            spec.env["XCELSIOR_CACHE_REPLICA_REGIONS"] = ",".join(peer_regions)

        ep = self.repo.create_endpoint(spec)
        if int(spec.min_workers) >= 1:
            self.provision_worker(str(ep["endpoint_id"]))
            ep = self.repo.get_endpoint(str(ep["endpoint_id"])) or ep

        if emit_sse:
            self._broadcast("serverless_endpoint.created", ep)
        return ep

    def list_endpoints(self, owner_id: str) -> list[dict]:
        return self.repo.list_endpoints(owner_id)

    def get_endpoint(self, endpoint_id: str, *, owner_id: str | None = None) -> dict | None:
        return self.repo.get_endpoint(endpoint_id, owner_id=owner_id)

    def patch_endpoint(
        self,
        endpoint_id: str,
        owner_id: str,
        fields: dict[str, Any],
        *,
        emit_sse: bool = True,
    ) -> dict | None:
        ep = self.repo.patch_endpoint(endpoint_id, owner_id, fields)
        if ep and emit_sse:
            self._broadcast("serverless_endpoint.updated", {"endpoint_id": endpoint_id})
        return ep

    def delete_endpoint(self, endpoint_id: str, owner_id: str, *, emit_sse: bool = True) -> bool:
        ep = self.repo.get_endpoint(endpoint_id, owner_id=owner_id)
        if not ep:
            return False
        for w in self.repo.list_workers(endpoint_id):
            if w.get("state") != WORKER_STATE_TERMINATED:
                self.deprovision_worker(str(w["worker_id"]), charge=True)
        ok = self.repo.soft_delete_endpoint(endpoint_id, owner_id)
        if ok and emit_sse:
            self._broadcast("serverless_endpoint.deleted", {"endpoint_id": endpoint_id})
        return ok

    # ── Workers ───────────────────────────────────────────────────────

    def provision_worker(self, endpoint_id: str) -> dict | None:
        ep = self.repo.get_endpoint(endpoint_id)
        if not ep:
            return None
        self.wallet_preflight(str(ep["owner_id"]))

        image = str(ep.get("image_ref") or DEFAULT_VLLM_IMAGE)
        command = str(ep.get("startup_command") or "")
        vram = float(ep.get("vram_required_gb") or 8.0)
        gpu_count = int(ep.get("gpu_count") or 1)
        gpu_model = str(ep.get("gpu_tier") or "").strip() or None
        http_port = int(ep.get("http_port") or 8080)
        health_check_path = str(ep.get("health_check_path") or "/health")
        from serverless.env_secrets import decrypt_env_for_worker

        endpoint_env = decrypt_env_for_worker(ep.get("env"))

        volume_ids: list[str] = []
        volume_mounts: dict[str, str] = {}
        cache_vol = str(ep.get("cache_volume_id") or "").strip()
        if cache_vol:
            volume_ids.append(cache_vol)
            volume_mounts[cache_vol] = "/xcelsior-cache"

        try:
            from scheduler import _set_job_fields, submit_job

            job = submit_job(
                name=f"serverless-{endpoint_id}",
                vram_needed_gb=vram,
                priority=5,
                tier="on-demand",
                num_gpus=gpu_count,
                gpu_model=gpu_model,
                image=image,
                command=command,
                owner=str(ep["owner_id"]),
                volume_ids=volume_ids or None,
                exposed_ports=[http_port],
                job_type="serverless_worker",
                region=str(ep.get("region") or ""),
            )
            scheduler_job_id = str(job.get("job_id") or "")
        except Exception as e:
            log.error("provision_worker submit_job failed: %s", e)
            self.repo.patch_endpoint(
                endpoint_id,
                str(ep["owner_id"]),
                {"status": ENDPOINT_STATUS_ERROR},
            )
            return None

        worker = self.repo.create_worker(
            endpoint_id,
            scheduler_job_id=scheduler_job_id,
            gpu_count=gpu_count,
            allocated_at=time.time(),
        )
        try:
            _set_job_fields(
                scheduler_job_id,
                serverless_worker_id=str(worker["worker_id"]),
                serverless_endpoint_id=endpoint_id,
                environment=endpoint_env,
                volume_mounts=volume_mounts,
                http_port=http_port,
                health_check_path=health_check_path,
            )
        except Exception as e:
            log.warning("provision_worker _set_job_fields failed: %s", e)

        self.repo.patch_endpoint(
            endpoint_id,
            str(ep["owner_id"]),
            {"status": ENDPOINT_STATUS_PROVISIONING},
        )
        self._broadcast(
            "serverless_worker.booting",
            {"endpoint_id": endpoint_id, "worker_id": worker["worker_id"]},
        )
        return worker

    def worker_heartbeat(self, worker_id: str) -> dict | None:
        w = self.repo.get_worker(worker_id)
        if not w or w.get("state") == WORKER_STATE_TERMINATED:
            return None
        now = time.time()
        state = str(w.get("state") or WORKER_STATE_BOOTING)
        if state == WORKER_STATE_BOOTING:
            state = WORKER_STATE_READY
        if int(w.get("current_concurrency") or 0) == 0 and state == WORKER_STATE_READY:
            state = WORKER_STATE_IDLE
        return self.repo.update_worker(worker_id, last_heartbeat_at=now, state=state)

    def worker_claim_job(self, worker_id: str) -> dict | None:
        w = self.repo.get_worker(worker_id)
        if not w:
            return None
        state = str(w.get("state") or "")
        if state in (
            WORKER_STATE_DRAINING,
            WORKER_STATE_TERMINATED,
            WORKER_STATE_ERROR,
            WORKER_STATE_BOOTING,
        ):
            return None
        ep = self.repo.get_endpoint(str(w["endpoint_id"]))
        if not ep:
            return None
        max_conc = int(ep.get("max_concurrency") or 1)
        if int(w.get("current_concurrency") or 0) >= max_conc:
            return None
        from routes._deps import otel_span
        from serverless.observability import log_job_event, record_cold_start, refresh_endpoint_gauges

        with otel_span(
            "serverless.job.claim",
            {"endpoint_id": str(ep["endpoint_id"]), "worker_id": worker_id},
        ):
            claimed = self.repo.claim_next_job(str(ep["endpoint_id"]), worker_id)
        if claimed:
            job_id = str(claimed["job_id"])
            first_job_on_worker = int(w.get("current_concurrency") or 0) == 0
            self.repo.increment_worker_concurrency(worker_id)
            if first_job_on_worker:
                alloc = float(w.get("allocated_at") or 0)
                if alloc:
                    cold = max(0, int(time.time() - alloc))
                    if cold:
                        self.repo.set_job_cold_start_seconds(job_id, cold)
                        record_cold_start(str(ep["endpoint_id"]))
            log_job_event(
                "dispatched",
                correlation_id=job_id,
                job_id=job_id,
                endpoint_id=str(ep["endpoint_id"]),
                worker_id=worker_id,
            )
            refresh_endpoint_gauges(self.repo, str(ep["endpoint_id"]))
        return claimed

    def worker_complete_job(
        self,
        worker_id: str,
        job_id: str,
        *,
        output: dict | None = None,
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict | None:
        w = self.repo.get_worker(worker_id)
        if not w:
            return None
        job = self.repo.get_job(job_id)
        if not job or str(job.get("worker_id")) != worker_id:
            return None
        status = str(job.get("status") or "")
        if status in (JOB_STATUS_CANCELLED, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED):
            return job
        ep = self.repo.get_endpoint(str(w["endpoint_id"]))
        now = time.time()
        started = float(job.get("started_at") or now)
        execution_seconds = max(0, int(math.ceil(now - started)))
        cold_start_seconds = int(job.get("cold_start_seconds") or 0)
        token_meta = token_cost_metadata(
            input_tokens, output_tokens, model_ref=str(ep.get("model_ref") or "") if ep else None
        )
        # Accrue this request's token cost so the blended meter can later charge the
        # higher of GPU-seconds vs. token cost. Best-effort: never block completion.
        try:
            self.repo.accrue_endpoint_token_cost(
                str(w["endpoint_id"]), float(token_meta.get("total_token_cost_cad") or 0.0)
            )
        except Exception:
            log.debug("token cost accrual skipped for endpoint %s", w.get("endpoint_id"))

        completed = self.repo.complete_job(
            job_id,
            output=output,
            error=error,
            gpu_seconds=execution_seconds,
            cold_start_seconds=cold_start_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cad=0.0,
        )
        self.dispatcher.release_job(job_id, worker_id)
        if ep:
            self.repo.increment_endpoint_totals(str(ep["endpoint_id"]), requests=1)
            if token_meta.get("total_token_cost_cad"):
                log.debug(
                    "Job %s token metadata (not billed): %s",
                    job_id,
                    token_meta,
                )
        if error:
            self.repo.append_stream_event(job_id, "error", {"message": error})
        else:
            self.repo.append_stream_event(job_id, "result", output or {})
        if completed and ep:
            from serverless.observability import (
                log_job_event,
                record_job_terminal,
                refresh_endpoint_gauges,
            )

            eid = str(ep["endpoint_id"])
            self._broadcast(
                "serverless_job.completed" if not error else "serverless_job.failed",
                {"endpoint_id": eid, "job_id": job_id},
            )
            record_job_terminal(eid, failed=bool(error))
            log_job_event(
                "failed" if error else "completed",
                correlation_id=job_id,
                job_id=job_id,
                endpoint_id=eid,
                worker_id=worker_id,
                gpu_seconds=execution_seconds,
                error=error or "",
            )
            refresh_endpoint_gauges(self.repo, eid)
            self.on_job_terminal(completed)
        return completed

    def on_job_terminal(self, job: dict | None) -> None:
        """Deliver webhook when a job reaches a terminal state."""
        if not job:
            return
        from serverless.webhooks import enqueue_job_webhook

        enqueue_job_webhook(self.repo, job)

    def fail_job_without_worker(self, job_id: str, *, error: str) -> dict | None:
        """Mark a queued/in-flight job failed (no worker callback) and notify webhook."""
        completed = self.repo.complete_job(job_id, error=error)
        if completed:
            from serverless.observability import log_job_event, record_job_terminal, refresh_endpoint_gauges

            eid = str(completed.get("endpoint_id") or "")
            self.repo.append_stream_event(
                job_id,
                "error",
                {"code": "job_failed", "message": error},
            )
            if eid:
                record_job_terminal(eid, failed=True)
                log_job_event(
                    "failed",
                    correlation_id=job_id,
                    job_id=job_id,
                    endpoint_id=eid,
                    error=error,
                )
                self._broadcast(
                    "serverless_job.failed",
                    {"endpoint_id": eid, "job_id": job_id},
                )
                refresh_endpoint_gauges(self.repo, eid)
            self.on_job_terminal(completed)
        return completed

    def cancel_inflight_job(
        self,
        job_id: str,
        endpoint_id: str,
        *,
        reason: str = "cancelled",
    ) -> dict | None:
        """Cancel a queued or in-flight job and release worker concurrency."""
        job = self.repo.get_job(job_id, endpoint_id=endpoint_id)
        if not job:
            return None
        status = str(job.get("status") or "")
        if status not in (JOB_STATUS_QUEUED, JOB_STATUS_IN_PROGRESS):
            return job

        worker_id = str(job.get("worker_id") or "")
        cancelled = self.repo.cancel_job(job_id, endpoint_id)
        if not cancelled:
            return None

        if worker_id and status == JOB_STATUS_IN_PROGRESS:
            self.dispatcher.release_job(job_id, worker_id)

        from serverless.observability import log_job_event, refresh_endpoint_gauges

        self.repo.append_stream_event(
            job_id,
            "error",
            {"code": "job_cancelled", "message": reason},
        )
        log_job_event(
            "cancelled",
            correlation_id=job_id,
            job_id=job_id,
            endpoint_id=endpoint_id,
            reason=reason,
        )
        refresh_endpoint_gauges(self.repo, endpoint_id)
        self.on_job_terminal(cancelled)
        return cancelled

    def worker_get_job(self, worker_id: str, job_id: str) -> dict | None:
        w = self.repo.get_worker(worker_id)
        if not w:
            return None
        job = self.repo.get_job(job_id, endpoint_id=str(w["endpoint_id"]))
        if not job or str(job.get("worker_id")) != worker_id:
            return None
        return job

    @staticmethod
    def normalize_runsync_output(job: dict) -> dict:
        """Defensive output shaping for /runsync — never raises on empty output."""
        output = job.get("output")
        if output is None:
            return {}
        if isinstance(output, dict):
            return output
        return {"output": output}

    def worker_append_event(
        self,
        worker_id: str,
        job_id: str,
        event_type: str,
        payload: dict,
    ) -> dict | None:
        w = self.repo.get_worker(worker_id)
        if not w:
            return None
        job = self.repo.get_job(job_id)
        if not job or str(job.get("worker_id")) != worker_id:
            return None
        return self.repo.append_stream_event(job_id, event_type, payload)

    def mark_worker_ready(self, worker_id: str, host_id: str = "") -> dict | None:
        now = time.time()
        w = self.repo.update_worker(
            worker_id,
            state=WORKER_STATE_READY,
            host_id=host_id or None,
            last_heartbeat_at=now,
        )
        if not w:
            return None
        ep = self.repo.get_endpoint(str(w["endpoint_id"]))
        if ep:
            if ep.get("status") in (ENDPOINT_STATUS_PROVISIONING, ENDPOINT_STATUS_SCALED_DOWN):
                self.repo.patch_endpoint(
                    str(ep["endpoint_id"]),
                    str(ep["owner_id"]),
                    {"status": ENDPOINT_STATUS_ACTIVE},
                )
            alloc = float(w.get("allocated_at") or 0)
            if alloc:
                cold = max(0, int(now - alloc))
                if cold:
                    try:
                        from billing import get_billing_engine

                        record_cold_start_line_item(
                            get_billing_engine(),
                            self.repo,
                            w,
                            ep,
                            cold_start_seconds=cold,
                            ready_at=now,
                        )
                    except Exception as e:
                        log.debug("cold_start line item skipped: %s", e)
        self._broadcast("serverless_worker.ready", {"worker_id": worker_id})
        return self.repo.get_worker(worker_id)

    def worker_exited(
        self,
        worker_id: str,
        *,
        exit_code: int = 0,
        error_message: str = "",
    ) -> dict | None:
        """Worker container exited — requeue in-flight jobs and mark worker lost."""
        w = self.repo.get_worker(worker_id)
        if not w or w.get("state") == WORKER_STATE_TERMINATED:
            return None
        self.dispatcher.handle_worker_lost(worker_id)
        state = WORKER_STATE_ERROR if exit_code != 0 else WORKER_STATE_TERMINATED
        self.repo.update_worker(
            worker_id,
            state=state,
            released_at=time.time(),
            error_message=(error_message or f"exit code {exit_code}")[:500] or None,
        )
        self._broadcast(
            "serverless_worker.exited",
            {"worker_id": worker_id, "exit_code": exit_code},
        )
        return self.repo.get_worker(worker_id)

    def deprovision_worker(self, worker_id: str, *, charge: bool = True) -> None:
        w = self.repo.get_worker(worker_id)
        if not w:
            return
        ep = self.repo.get_endpoint(str(w["endpoint_id"]))
        now = time.time()
        self.dispatcher.terminate_worker(worker_id)

        scheduler_job_id = str(w.get("scheduler_job_id") or "")
        if scheduler_job_id:
            try:
                from scheduler import kill_job, list_jobs, list_hosts

                jobs = {j["job_id"]: j for j in list_jobs()}
                job = jobs.get(scheduler_job_id)
                if job and job.get("host_id"):
                    hosts = {h["host_id"]: h for h in list_hosts()}
                    host = hosts.get(job["host_id"])
                    if host:
                        kill_job(job, host)
            except Exception as e:
                log.warning("kill_job failed for %s: %s", scheduler_job_id, e)

        self.repo.update_worker(worker_id, state=WORKER_STATE_TERMINATED, released_at=now)
        if charge and ep:
            charge_serverless_worker(self.repo, {**w, "released_at": now}, ep, released_at=now)

    # ── Reconcile (autoscaler + dispatch) ─────────────────────────────

    def reconcile_all(self) -> dict:
        """Single-writer reconcile tick — acquire advisory lock first."""
        if os.environ.get("XCELSIOR_SERVERLESS_RECONCILE", "true").lower() == "false":
            return {"skipped": True, "reason": "disabled"}
        if not self.repo.try_advisory_lock():
            return {"skipped": True, "reason": "lock_held"}
        try:
            from serverless.reaper import reap_all

            from serverless.observability import refresh_all_gauges

            reaper_stats = reap_all(self)
            results = []
            for ep in self.repo.list_endpoints_for_reconcile():
                results.append(self.reconcile_endpoint(str(ep["endpoint_id"])))
            refresh_all_gauges(self.repo)
            if any(r.get("scaled_up") for r in results):
                try:
                    from scheduler import process_queue

                    process_queue()
                except Exception as e:
                    log.warning("process_queue after scale-up failed: %s", e)
            return {
                "reconciled": len(results),
                "endpoints": results,
                "reaper": reaper_stats,
            }
        finally:
            self.repo.release_advisory_lock()

    def reconcile_endpoint(self, endpoint_id: str) -> dict:
        ep = self.repo.get_endpoint(endpoint_id)
        if not ep:
            return {"endpoint_id": endpoint_id, "error": "not_found"}

        workers = self.repo.list_workers(endpoint_id)
        queue_depth = self.repo.queue_depth(endpoint_id)
        samples = _record_queue_depth_sample(endpoint_id, queue_depth)
        inp = AutoscalerInput(
            min_workers=int(ep.get("min_workers") or 0),
            max_workers=int(ep.get("max_workers") or 4),
            max_concurrency=int(ep.get("max_concurrency") or 4),
            scaling_policy_type=str(ep.get("scaling_policy_type") or "queue_request_count"),
            scaling_policy_value=int(ep.get("scaling_policy_value") or 1),
            queue_depth=queue_depth,
            max_queue_wait_sec=self.repo.max_queue_wait_sec(endpoint_id),
            workers=workers,
            queue_depth_samples=samples,
        )
        desired = compute_desired_workers(inp)
        active = self.repo.count_active_workers(endpoint_id)
        scaled_up = 0
        while active + scaled_up < desired:
            self.provision_worker(endpoint_id)
            scaled_up += 1
            active = self.repo.count_active_workers(endpoint_id)
            if scaled_up >= max(1, desired - active + 2):
                break

        now = time.time()
        idle_timeout = int(ep.get("idle_timeout_sec") or 300)
        cooldown_active = scale_down_cooldown_active(
            workers, now=now, cooldown_sec=SCALE_DOWN_COOLDOWN_SEC
        )
        drained = 0
        reaped = 0
        if not cooldown_active:
            for wid in workers_to_mark_draining(
                workers,
                desired=desired,
                idle_timeout_sec=idle_timeout,
                now=now,
            ):
                self.repo.update_worker(wid, state=WORKER_STATE_DRAINING)
                drained += 1
            workers = self.repo.list_workers(endpoint_id)
            for wid in workers_to_reap(
                workers,
                desired=desired,
                drain_grace_sec=DRAIN_GRACE_SEC,
                now=now,
            ):
                self.deprovision_worker(wid, charge=True)
                reaped += 1

        for w in self.repo.list_workers(endpoint_id):
            if w.get("state") == WORKER_STATE_BOOTING:
                self._maybe_prepull_worker(w, ep)

        dispatched = self.dispatcher.dispatch_for_endpoint(ep)
        if scaled_up > 0 or reaped > 0 or drained > 0:
            self._broadcast(
                "serverless_worker.scaled",
                {
                    "endpoint_id": endpoint_id,
                    "desired_workers": desired,
                    "active_workers": self.repo.count_active_workers(endpoint_id),
                    "scaled_up": scaled_up,
                    "drained": drained,
                    "reaped": reaped,
                },
            )
        if self.repo.count_active_workers(endpoint_id) == 0 and inp.queue_depth == 0:
            if int(ep.get("min_workers") or 0) == 0 and ep.get("status") == ENDPOINT_STATUS_ACTIVE:
                self.repo.patch_endpoint(
                    endpoint_id,
                    str(ep["owner_id"]),
                    {"status": ENDPOINT_STATUS_SCALED_DOWN},
                )

        return {
            "endpoint_id": endpoint_id,
            "desired_workers": desired,
            "active_workers": active,
            "scaled_up": scaled_up,
            "drained": drained,
            "reaped": reaped,
            "scale_down_cooldown": cooldown_active,
            "dispatched_job": dispatched.get("job_id") if dispatched else None,
        }

    def _endpoint_needs_workers(self, ep: dict) -> bool:
        endpoint_id = str(ep["endpoint_id"])
        inp = AutoscalerInput(
            min_workers=int(ep.get("min_workers") or 0),
            max_workers=int(ep.get("max_workers") or 4),
            max_concurrency=int(ep.get("max_concurrency") or 4),
            scaling_policy_type=str(ep.get("scaling_policy_type") or "queue_request_count"),
            scaling_policy_value=int(ep.get("scaling_policy_value") or 1),
            queue_depth=self.repo.queue_depth(endpoint_id),
            max_queue_wait_sec=self.repo.max_queue_wait_sec(endpoint_id),
            workers=self.repo.list_workers(endpoint_id),
        )
        desired = compute_desired_workers(inp)
        return self.repo.count_active_workers(endpoint_id) < desired

    def _maybe_prepull_worker(self, worker: dict, ep: dict) -> None:
        sched_id = str(worker.get("scheduler_job_id") or "")
        if not sched_id:
            return
        try:
            from scheduler import get_job

            sched_job = get_job(sched_id)
        except Exception:
            return
        host_id = str((sched_job or {}).get("host_id") or "")
        if not host_id:
            return
        image = str(ep.get("image_ref") or DEFAULT_VLLM_IMAGE)
        try:
            from routes.agent import enqueue_agent_command

            enqueue_agent_command(
                host_id,
                "prepull_image",
                {"image": image},
                created_by="serverless_reconcile",
            )
        except Exception as e:
            log.debug("prepull_image enqueue failed host=%s: %s", host_id, e)

    # ── Health / metrics ──────────────────────────────────────────────

    def get_endpoint_health(self, endpoint_id: str) -> dict:
        ep = self.repo.get_endpoint(endpoint_id)
        if not ep:
            return {"status": "not_found"}
        workers = self.repo.list_workers(endpoint_id)
        ready = sum(1 for w in workers if w.get("state") in (WORKER_STATE_READY, WORKER_STATE_IDLE))
        booting = sum(1 for w in workers if w.get("state") == WORKER_STATE_BOOTING)
        pricing = pricing_for_endpoint(ep)
        return {
            "endpoint_id": endpoint_id,
            "status": ep.get("status"),
            "mode": ep.get("mode"),
            "workers_ready": ready,
            "workers_booting": booting,
            "workers_total": len(workers),
            "queue_depth": self.repo.queue_depth(endpoint_id),
            "max_queue_wait_sec": round(self.repo.max_queue_wait_sec(endpoint_id), 2),
            "pricing": pricing,
            "rate_per_hour_cad": pricing["rate_per_hour_cad"],
            "rate_cents_per_second_per_worker": pricing["rate_cents_per_second_per_worker"],
        }

    def log_job_enqueued(
        self,
        job: dict,
        *,
        correlation_id: str | None = None,
    ) -> None:
        from serverless.observability import log_job_event, refresh_endpoint_gauges

        job_id = str(job.get("job_id") or "")
        endpoint_id = str(job.get("endpoint_id") or "")
        cid = correlation_id or job_id
        log_job_event(
            "enqueued",
            correlation_id=cid,
            job_id=job_id,
            endpoint_id=endpoint_id,
            status=str(job.get("status") or ""),
        )
        if endpoint_id:
            refresh_endpoint_gauges(self.repo, endpoint_id)

    def get_endpoint_metrics(self, endpoint_id: str, *, since: float | None = None) -> dict:
        from serverless.observability import compute_endpoint_metrics, refresh_endpoint_gauges

        ep = self.repo.get_endpoint(endpoint_id)
        if not ep:
            return {}
        now = time.time()
        since_ts = since or (now - 86400)
        jobs = self.repo.list_jobs(endpoint_id, since_finished_at=since_ts, limit=500)
        workers = self.repo.list_workers(endpoint_id)
        queue_depth = self.repo.queue_depth(endpoint_id)
        refresh_endpoint_gauges(self.repo, endpoint_id, workers=workers)
        return compute_endpoint_metrics(
            ep,
            jobs,
            workers,
            queue_depth=queue_depth,
            window_sec=now - since_ts,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _broadcast(event_type: str, data: dict) -> None:
        try:
            from routes._deps import broadcast_sse

            broadcast_sse(event_type, data)
        except Exception as e:
            log.debug("SSE broadcast skipped: %s", e)


_service: ServerlessService | None = None


def get_serverless_service() -> ServerlessService:
    global _service
    if _service is None:
        _service = ServerlessService()
    return _service
