"""Routes: serverless inference endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routes._deps import (
    _effective_billing_customer_id,
    _get_current_user,
    _require_auth,
    _require_scope,
    _require_serverless_endpoint_access,
    _require_serverless_endpoint_write,
    _require_serverless_feature,
    _require_team_instance_write,
    _resolve_serverless_endpoint_auth,
    _serverless_owner_ids_readable,
    _serverless_scope_owner_id,
    _user_has_serverless_feature,
    broadcast_sse,
    otel_span,
)
from serverless.observability import resolve_correlation_id
from serverless.keys import create_endpoint_key, validate_key
from serverless.limits import (
    QueueFullError,
    RateLimitExceeded,
    check_key_rate_limit,
    check_queue_capacity,
    rate_limit_headers,
)
from serverless.webhooks import validate_webhook_url
from serverless.openai_proxy import (
    OpenAIProxyError,
    async_proxy_stream_lines,
    proxy_chat_completions,
    proxy_request,
)
from serverless.env_secrets import encrypt_env_for_storage, payload_byte_size
from serverless.repo import EndpointCreate, ServerlessRepo, WORKER_STATE_READY
from serverless.metering import pricing_for_endpoint
from serverless.service import ServerlessService, WalletPreflightError, get_serverless_service
from serverless.streams import (
    SSE_RESPONSE_HEADERS,
    async_live_job_stream,
    format_error_chunk,
    iter_job_stream,
)

router = APIRouter()


def _svc() -> ServerlessService:
    return get_serverless_service()


def _repo() -> ServerlessRepo:
    return ServerlessRepo()


def _serialize_endpoint(ep: dict) -> dict:
    eid = str(ep.get("endpoint_id") or "")
    from serverless.vanity import endpoint_vanity_slug, vanity_invoke_path

    slug = endpoint_vanity_slug(str(ep.get("name") or ""), eid)
    return {
        "endpoint_id": eid,
        "owner_id": ep.get("owner_id"),
        "name": ep.get("name") or "",
        "mode": ep.get("mode"),
        "managed_engine": ep.get("managed_engine") or "vllm",
        "vanity_slug": slug,
        "invoke_path": vanity_invoke_path(eid, slug),
        "model_id": ep.get("model_ref"),
        "model_name": ep.get("model_ref"),
        "model_ref": ep.get("model_ref"),
        "model_revision": ep.get("model_revision"),
        "gpu_type": ep.get("gpu_tier"),
        "gpu_tier": ep.get("gpu_tier"),
        "gpu_count": ep.get("gpu_count"),
        "region": ep.get("region"),
        "docker_image": ep.get("image_ref"),
        "image_ref": ep.get("image_ref"),
        "startup_command": ep.get("startup_command"),
        "http_port": ep.get("http_port"),
        "health_check_path": ep.get("health_check_path"),
        "health_endpoint": ep.get("health_check_path"),
        "status": ep.get("status"),
        "min_workers": ep.get("min_workers"),
        "max_workers": ep.get("max_workers"),
        "max_concurrency": ep.get("max_concurrency"),
        "idle_timeout_sec": ep.get("idle_timeout_sec"),
        "scaling_policy_type": ep.get("scaling_policy_type"),
        "scaling_policy_value": ep.get("scaling_policy_value"),
        "total_requests": int(ep.get("total_requests") or 0),
        "total_gpu_seconds": int(ep.get("total_gpu_seconds") or 0),
        "total_cost_cad": float(ep.get("total_cost_cad") or 0),
        "created_at": ep.get("created_at"),
        "updated_at": ep.get("updated_at"),
        "openai_base_url": f"/v1/serverless/{eid}/openai/v1" if eid else "",
        "pricing": pricing_for_endpoint(ep),
    }


# ── Management models ────────────────────────────────────────────────


class ServerlessEndpointCreate(BaseModel):
    name: str = ""
    mode: str = "preset"
    managed_engine: str = "vllm"
    model_name: str = ""
    model_ref: str = ""
    image_ref: str = ""
    docker_image: str = ""
    source_type: str = ""
    source_ref: str = ""
    source_ref_branch: str = "main"
    gpu_type: str = ""
    gpu_tier: str = ""
    gpu_count: int = Field(1, ge=1, le=8)
    region: str = "ca-east"
    min_workers: int = Field(0, ge=0, le=32)
    max_workers: int = Field(4, ge=1, le=32)
    max_concurrency: int = Field(4, ge=1, le=256)
    idle_timeout_sec: int = Field(300, ge=60, le=86400)
    scaledown_window_sec: int | None = None
    scaling_policy_type: str = "queue_request_count"
    scaling_policy_value: int = Field(1, ge=1)
    startup_command: str = ""
    http_port: int = Field(8080, ge=1, le=65535)
    health_check_path: str = "/health"
    health_endpoint: str = ""
    cuda_version: str = "12.4"
    request_timeout_sec: int = Field(120, ge=10, le=3600)
    max_request_bytes: int = Field(10_485_760, ge=1024, le=52_428_800)
    env: dict[str, str] = Field(default_factory=dict)


class ServerlessEndpointPatch(BaseModel):
    name: str | None = None
    min_workers: int | None = Field(None, ge=0, le=32)
    max_workers: int | None = Field(None, ge=1, le=32)
    max_concurrency: int | None = Field(None, ge=1, le=256)
    idle_timeout_sec: int | None = Field(None, ge=60, le=86400)
    scaling_policy_type: str | None = None
    scaling_policy_value: int | None = Field(None, ge=1)
    request_timeout_sec: int | None = Field(None, ge=10, le=3600)
    max_queue_size: int | None = Field(None, ge=1, le=10_000)
    keep_warm: bool | None = None


class RunJobRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    webhook: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    stream: bool = False


class ApiKeyCreateRequest(BaseModel):
    name: str = "default"
    scopes: str = "inference:write"
    rate_limit_rpm: int = Field(60, ge=1, le=10_000)


def _body_to_endpoint_create(body: ServerlessEndpointCreate, owner_id: str) -> EndpointCreate:
    model_ref = (body.model_ref or body.model_name or "").strip()
    image_ref = (body.image_ref or body.docker_image or "").strip()
    gpu_tier = (body.gpu_tier or body.gpu_type or "").strip()
    health = (body.health_check_path or body.health_endpoint or "/health").strip()
    if health and not health.startswith("/"):
        health = f"/{health}"
    idle = body.idle_timeout_sec
    if body.scaledown_window_sec is not None:
        idle = body.scaledown_window_sec
    managed = (body.managed_engine or "vllm").strip().lower() or "vllm"
    return EndpointCreate(
        owner_id=owner_id,
        name=body.name.strip(),
        mode=body.mode,
        managed_engine=managed,
        model_ref=model_ref,
        image_ref=image_ref,
        source_type=(body.source_type or "").strip(),
        source_ref=(body.source_ref or "").strip(),
        source_ref_branch=(body.source_ref_branch or "main").strip() or "main",
        gpu_tier=gpu_tier,
        gpu_count=body.gpu_count,
        region=body.region,
        min_workers=body.min_workers,
        max_workers=body.max_workers,
        max_concurrency=body.max_concurrency,
        idle_timeout_sec=idle,
        scaling_policy_type=body.scaling_policy_type,
        scaling_policy_value=body.scaling_policy_value,
        startup_command=body.startup_command,
        http_port=body.http_port,
        health_check_path=health,
        cuda_version=body.cuda_version,
        request_timeout_sec=body.request_timeout_sec,
        max_request_bytes=body.max_request_bytes,
        env=encrypt_env_for_storage(body.env),
    )


def _get_endpoint_for_user(endpoint_id: str, user: dict) -> dict:
    ep = _repo().get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(404, "Endpoint not found")
    _require_serverless_feature(user=user, owner_id=str(ep["owner_id"]))
    _require_serverless_endpoint_access(user, ep)
    return ep


def _run_rate_limit(key_row: dict | None) -> dict[str, str]:
    if not key_row:
        return {}
    info = check_key_rate_limit(
        str(key_row["key_id"]),
        int(key_row.get("rate_limit_rpm") or 60),
    )
    return rate_limit_headers(info)


def _check_payload_size(payload: dict[str, Any], ep: dict) -> None:
    limit = int(ep.get("max_request_bytes") or 10_485_760)
    size = payload_byte_size(payload)
    if size > limit:
        raise HTTPException(
            413,
            detail={
                "error": {
                    "code": "payload_too_large",
                    "message": f"Request payload {size} bytes exceeds limit {limit}",
                }
            },
        )


def _enqueue_serverless_job(
    *,
    endpoint_id: str,
    owner_id: str,
    payload: dict[str, Any],
    ep: dict,
    idempotency_key: str | None,
    webhook_url: str | None,
) -> tuple[dict, dict[str, str]]:
    _check_payload_size(payload, ep)
    check_queue_capacity(_repo(), endpoint_id, ep)
    job = _repo().enqueue_job(
        endpoint_id,
        owner_id,
        payload,
        idempotency_key=idempotency_key,
        webhook_url=webhook_url,
    )
    return job, {}


def _warm_worker_row(endpoint_id: str) -> dict | None:
    repo = _repo()
    ep = repo.get_endpoint(endpoint_id)
    if not ep:
        return None
    worker = repo.find_ready_worker_with_capacity(
        endpoint_id, int(ep.get("max_concurrency") or 1)
    )
    if not worker:
        return None
    row = repo.get_worker_job_row(str(worker["worker_id"]))
    return row if row else worker


# ── Management API ───────────────────────────────────────────────────


@router.get("/api/v2/serverless/enabled", tags=["Serverless"])
def api_serverless_enabled(request: Request):
    """Feature-flag probe for dashboard rollout (session optional)."""
    from serverless.feature import serverless_feature_status

    user = _get_current_user(request)
    owner_id = _serverless_scope_owner_id(user) if user else None
    return {"ok": True, **serverless_feature_status(owner_id=owner_id)}


class GitHubResolveRequest(BaseModel):
    source_ref: str
    source_ref_branch: str = "main"


@router.post("/api/v2/serverless/github/resolve", tags=["Serverless"])
def api_serverless_github_resolve(body: GitHubResolveRequest, request: Request):
    """Resolve a GitHub repo URL to the default GHCR image reference."""
    _require_auth(request)
    from serverless.github_deploy import GitHubSourceError, resolve_github_image

    try:
        image = resolve_github_image(body.source_ref, ref=body.source_ref_branch)
    except GitHubSourceError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "image_ref": image, "source_type": "github"}


@router.post("/api/v2/serverless/endpoints", tags=["Serverless"])
def api_serverless_create_endpoint(body: ServerlessEndpointCreate, request: Request):
    user = _require_auth(request)
    _require_serverless_feature(user=user)
    _require_scope(user, "inference:write")
    _require_team_instance_write(user)
    try:
        ep = _svc().create_endpoint(
            _body_to_endpoint_create(body, _serverless_scope_owner_id(user))
        )
        return {"ok": True, "endpoint": _serialize_endpoint(ep)}
    except WalletPreflightError as e:
        raise HTTPException(e.status_code, e.message)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/v2/serverless/endpoints", tags=["Serverless"])
def api_serverless_list_endpoints(request: Request):
    user = _require_auth(request)
    if not _user_has_serverless_feature(user):
        return {"ok": True, "endpoints": []}
    endpoints: list[dict] = []
    seen: set[str] = set()
    for owner_id in sorted(_serverless_owner_ids_readable(user)):
        for ep in _svc().list_endpoints(owner_id):
            eid = str(ep.get("endpoint_id") or "")
            if eid and eid not in seen:
                seen.add(eid)
                endpoints.append(_serialize_endpoint(ep))
    return {"ok": True, "endpoints": endpoints}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}", tags=["Serverless"])
def api_serverless_get_endpoint(endpoint_id: str, request: Request):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    return {"ok": True, "endpoint": _serialize_endpoint(ep)}


@router.patch("/api/v2/serverless/endpoints/{endpoint_id}", tags=["Serverless"])
def api_serverless_patch_endpoint(
    endpoint_id: str, body: ServerlessEndpointPatch, request: Request
):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    _require_serverless_endpoint_write(user, ep)
    fields = body.model_dump(exclude_none=True)
    updated = _svc().patch_endpoint(endpoint_id, str(ep["owner_id"]), fields)
    if not updated:
        raise HTTPException(404, "Endpoint not found")
    return {"ok": True, "endpoint": _serialize_endpoint(updated)}


@router.delete("/api/v2/serverless/endpoints/{endpoint_id}", tags=["Serverless"])
def api_serverless_delete_endpoint(endpoint_id: str, request: Request):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    _require_serverless_endpoint_write(user, ep)
    if not _svc().delete_endpoint(endpoint_id, str(ep["owner_id"])):
        raise HTTPException(404, "Endpoint not found")
    return {"ok": True}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/health", tags=["Serverless"])
def api_serverless_endpoint_health(endpoint_id: str, request: Request):
    user = _require_auth(request)
    _get_endpoint_for_user(endpoint_id, user)
    health = _svc().get_endpoint_health(endpoint_id)
    return {"ok": True, "health": health}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/metrics", tags=["Serverless"])
def api_serverless_endpoint_metrics(endpoint_id: str, request: Request):
    user = _require_auth(request)
    _get_endpoint_for_user(endpoint_id, user)
    since_hours = float(request.query_params.get("since_hours") or 24)
    since_ts = time.time() - max(1.0, since_hours) * 3600.0
    metrics = _svc().get_endpoint_metrics(endpoint_id, since=since_ts)
    return {"ok": True, "metrics": metrics}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/usage", tags=["Serverless"])
def api_serverless_endpoint_usage(endpoint_id: str, request: Request):
    """Alias for metrics — dashboard compatibility."""
    user = _require_auth(request)
    _get_endpoint_for_user(endpoint_id, user)
    metrics = _svc().get_endpoint_metrics(endpoint_id)
    usage = {
        "endpoint_id": endpoint_id,
        "total_requests": metrics.get("total_requests", 0),
        "total_gpu_seconds": metrics.get("total_gpu_seconds", 0),
        "total_cost_cad": metrics.get("total_cost_cad", 0),
        "last_24h": {
            "jobs_completed": metrics.get("jobs_completed", 0),
            "jobs_failed": metrics.get("jobs_failed", 0),
            "avg_gpu_seconds": metrics.get("avg_gpu_seconds", 0),
            "success_rate": metrics.get("success_rate", 0),
            "error_rate": metrics.get("error_rate", 0),
            "avg_queue_ms": metrics.get("avg_queue_ms", 0),
            "avg_execution_ms": metrics.get("avg_execution_ms", 0),
            "tokens_per_sec": metrics.get("tokens_per_sec", 0),
        },
    }
    return {"ok": True, "usage": usage}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/workers", tags=["Serverless"])
def api_serverless_list_workers(endpoint_id: str, request: Request):
    user = _require_auth(request)
    _get_endpoint_for_user(endpoint_id, user)
    workers = _repo().list_workers(endpoint_id)
    return {"ok": True, "workers": workers}


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/jobs", tags=["Serverless"])
def api_serverless_list_jobs(endpoint_id: str, request: Request):
    user = _require_auth(request)
    _get_endpoint_for_user(endpoint_id, user)
    jobs = _repo().list_jobs(endpoint_id, limit=100)
    return {"ok": True, "jobs": jobs}


@router.post("/api/v2/serverless/endpoints/{endpoint_id}/keys", tags=["Serverless"])
def api_serverless_create_key(
    endpoint_id: str, body: ApiKeyCreateRequest, request: Request
):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    _require_serverless_endpoint_write(user, ep)
    raw, row = create_endpoint_key(
        _repo(),
        str(ep["owner_id"]),
        endpoint_id=endpoint_id,
        name=body.name,
        scopes=body.scopes,
        rate_limit_rpm=body.rate_limit_rpm,
    )
    return {
        "ok": True,
        "key": {
            "key_id": row.get("key_id"),
            "name": row.get("name"),
            "key_prefix": row.get("key_prefix"),
            "api_key": raw,
            "scopes": row.get("scopes"),
        },
    }


@router.get("/api/v2/serverless/endpoints/{endpoint_id}/keys", tags=["Serverless"])
def api_serverless_list_keys(endpoint_id: str, request: Request):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    keys = _repo().list_api_keys(str(ep["owner_id"]), endpoint_id=endpoint_id)
    safe = [
        {
            "key_id": k["key_id"],
            "name": k.get("name"),
            "key_prefix": k.get("key_prefix"),
            "scopes": k.get("scopes"),
            "created_at": k.get("created_at"),
            "last_used_at": k.get("last_used_at"),
        }
        for k in keys
    ]
    return {"ok": True, "keys": safe}


@router.delete("/api/v2/serverless/endpoints/{endpoint_id}/keys/{key_id}", tags=["Serverless"])
def api_serverless_revoke_key(endpoint_id: str, key_id: str, request: Request):
    user = _require_auth(request)
    ep = _get_endpoint_for_user(endpoint_id, user)
    _require_serverless_endpoint_write(user, ep)
    if not _repo().revoke_api_key(key_id, str(ep["owner_id"])):
        raise HTTPException(404, "Key not found")
    return {"ok": True}


# ── Queue job API ────────────────────────────────────────────────────


@router.post("/v1/serverless/{endpoint_id}/run", tags=["Serverless"])
def api_serverless_run(endpoint_id: str, body: RunJobRequest, request: Request):
    user, key_row, ep = _resolve_serverless_endpoint_auth(request, endpoint_id, write=True)
    owner_id = str(ep["owner_id"])
    try:
        ServerlessService.wallet_preflight(owner_id)
    except WalletPreflightError as e:
        raise HTTPException(e.status_code, e.message)

    try:
        rl_headers = _run_rate_limit(key_row)
    except RateLimitExceeded as e:
        raise HTTPException(
            429,
            detail={"error": {"code": "rate_limited", "message": "Rate limit exceeded"}},
            headers=rate_limit_headers(e.info),
        )

    idem = request.headers.get("Idempotency-Key") or request.headers.get("idempotency-key")
    webhook_url = validate_webhook_url(body.webhook or "")
    try:
        job, _ = _enqueue_serverless_job(
            endpoint_id=endpoint_id,
            owner_id=owner_id,
            payload=body.input,
            ep=ep,
            idempotency_key=idem,
            webhook_url=webhook_url,
        )
    except QueueFullError as e:
        raise HTTPException(
            429,
            detail={
                "error": {
                    "code": "queue_full",
                    "message": f"Queue full ({e.depth}/{e.limit})",
                }
            },
            headers=rl_headers,
        )

    correlation_id = resolve_correlation_id(
        {k.lower(): v for k, v in request.headers.items()},
        job_id=str(job["job_id"]),
    )
    with otel_span(
        "serverless.job.enqueue",
        {"endpoint_id": endpoint_id, "job_id": job["job_id"]},
    ):
        _svc().log_job_enqueued(job, correlation_id=correlation_id)
        _svc().reconcile_endpoint(endpoint_id)
    broadcast_sse(
        "serverless_job.queued",
        {
            "endpoint_id": endpoint_id,
            "job_id": job["job_id"],
            "correlation_id": correlation_id,
        },
    )
    return JSONResponse(
        content={"id": job["job_id"], "status": "IN_QUEUE"},
        headers=rl_headers,
    )


@router.post("/v1/serverless/{endpoint_id}/runsync", tags=["Serverless"])
def api_serverless_runsync(endpoint_id: str, body: RunJobRequest, request: Request):
    user, key_row, ep = _resolve_serverless_endpoint_auth(request, endpoint_id, write=True)
    owner_id = str(ep["owner_id"])
    try:
        ServerlessService.wallet_preflight(owner_id)
    except WalletPreflightError as e:
        raise HTTPException(e.status_code, e.message)

    try:
        rl_headers = _run_rate_limit(key_row)
    except RateLimitExceeded as e:
        raise HTTPException(
            429,
            detail={"error": {"code": "rate_limited", "message": "Rate limit exceeded"}},
            headers=rate_limit_headers(e.info),
        )

    timeout = int(ep.get("request_timeout_sec") or 120)
    webhook_url = validate_webhook_url(body.webhook or "")
    try:
        job, _ = _enqueue_serverless_job(
            endpoint_id=endpoint_id,
            owner_id=owner_id,
            payload=body.input,
            ep=ep,
            idempotency_key=None,
            webhook_url=webhook_url,
        )
    except QueueFullError as e:
        raise HTTPException(
            429,
            detail={
                "error": {
                    "code": "queue_full",
                    "message": f"Queue full ({e.depth}/{e.limit})",
                }
            },
            headers=rl_headers,
        )
    _svc().reconcile_endpoint(endpoint_id)
    _svc().dispatcher.dispatch_for_endpoint(ep)

    deadline = time.time() + timeout
    while time.time() < deadline:
        current = _repo().get_job(str(job["job_id"]), endpoint_id=endpoint_id)
        if not current:
            break
        status = str(current.get("status") or "")
        if status == "COMPLETED":
            return ServerlessService.normalize_runsync_output(current)
        if status in ("FAILED", "CANCELLED"):
            err = current.get("error") or status
            raise HTTPException(
                500,
                detail={
                    "error": {
                        "code": "job_failed" if status == "FAILED" else "job_cancelled",
                        "message": str(err),
                        "job_id": job["job_id"],
                    }
                },
            )
        time.sleep(0.25)

    raise HTTPException(
        202,
        detail={"status": "IN_QUEUE", "job_id": job["job_id"], "poll": f"/v1/serverless/{endpoint_id}/status/{job['job_id']}"},
    )


@router.get("/v1/serverless/{endpoint_id}/status/{job_id}", tags=["Serverless"])
def api_serverless_job_status(endpoint_id: str, job_id: str, request: Request):
    _resolve_serverless_endpoint_auth(request, endpoint_id, write=False)
    job = _repo().get_job(job_id, endpoint_id=endpoint_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job_id,
        "status": job.get("status"),
        "output": job.get("output") if job.get("output") is not None else None,
        "error": job.get("error"),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "gpu_seconds": job.get("gpu_seconds"),
        "cold_start_seconds": job.get("cold_start_seconds"),
        "cost_cad": job.get("cost_cad"),
    }


@router.get("/v1/serverless/{endpoint_id}/stream/{job_id}", tags=["Serverless"])
async def api_serverless_job_stream(endpoint_id: str, job_id: str, request: Request):
    _resolve_serverless_endpoint_auth(request, endpoint_id, write=False)
    job = _repo().get_job(job_id, endpoint_id=endpoint_id)
    if not job:
        raise HTTPException(404, "Job not found")
    after = int(request.query_params.get("after_seq") or 0)
    svc = _svc()
    repo = _repo()

    def _on_disconnect() -> None:
        svc.cancel_inflight_job(
            job_id,
            endpoint_id,
            reason="client disconnected",
        )

    async def _gen():
        try:
            async for chunk in async_live_job_stream(
                repo,
                job_id,
                after_seq=after,
                request=request,
                on_disconnect=_on_disconnect,
            ):
                yield chunk
        except Exception as e:
            yield format_error_chunk("stream_error", str(e))

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers=SSE_RESPONSE_HEADERS,
    )


@router.post("/v1/serverless/{endpoint_id}/cancel/{job_id}", tags=["Serverless"])
def api_serverless_cancel_job(endpoint_id: str, job_id: str, request: Request):
    _resolve_serverless_endpoint_auth(request, endpoint_id, write=True)
    job = _svc().cancel_inflight_job(job_id, endpoint_id, reason="cancelled by client")
    if not job:
        raise HTTPException(404, "Job not found")
    return {"id": job_id, "status": job.get("status")}


# ── OpenAI-compatible surface ────────────────────────────────────────


@router.post("/v1/serverless/{endpoint_id}/openai/v1/chat/completions", tags=["Serverless"])
async def api_serverless_openai_chat(endpoint_id: str, body: ChatCompletionRequest, request: Request):
    _user, key_row, _ep = _resolve_serverless_endpoint_auth(request, endpoint_id, write=True)
    try:
        _run_rate_limit(key_row)
    except RateLimitExceeded as e:
        raise HTTPException(
            429,
            detail={"error": {"code": "rate_limited", "message": "Rate limit exceeded"}},
            headers=rate_limit_headers(e.info),
        )
    ep = _repo().get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(404, "Endpoint not found")
    try:
        ServerlessService.wallet_preflight(str(ep["owner_id"]))
    except WalletPreflightError as e:
        raise HTTPException(e.status_code, e.message)

    if str(ep.get("mode")) != "preset":
        raise HTTPException(400, "OpenAI proxy requires a preset (managed model) endpoint")

    worker_row = _warm_worker_row(endpoint_id)
    if not worker_row:
        _svc().reconcile_endpoint(endpoint_id)
        worker_row = _warm_worker_row(endpoint_id)
    if not worker_row:
        raise HTTPException(503, f"No warm workers available for endpoint {endpoint_id}")

    payload = body.model_dump()
    with otel_span("serverless.openai.chat", {"endpoint_id": endpoint_id, "model": body.model}):
        try:
            if body.stream:
                timeout = float(ep.get("request_timeout_sec") or 120)

                async def _stream():
                    try:
                        async for line in async_proxy_stream_lines(
                            worker_row,
                            ep,
                            "chat/completions",
                            json_body={**payload, "stream": True},
                            timeout_sec=timeout,
                        ):
                            if await request.is_disconnected():
                                return
                            if line:
                                yield (line + "\n").encode("utf-8")
                        yield b"data: [DONE]\n\n"
                    except OpenAIProxyError as e:
                        err = format_error_chunk(e.code, e.message)
                        yield err.encode("utf-8")
                    except Exception as e:
                        err = format_error_chunk("stream_error", str(e))
                        yield err.encode("utf-8")

                return StreamingResponse(
                    _stream(),
                    media_type="text/event-stream",
                    headers=SSE_RESPONSE_HEADERS,
                )

            result = proxy_chat_completions(worker_row, ep, payload)
            return result
        except OpenAIProxyError as e:
            raise HTTPException(e.status_code, detail={"error": {"code": e.code, "message": e.message}})


@router.get("/v1/serverless/{endpoint_id}/openai/v1/models", tags=["Serverless"])
def api_serverless_openai_models(endpoint_id: str, request: Request):
    _resolve_serverless_endpoint_auth(request, endpoint_id, write=False)
    ep = _repo().get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(404, "Endpoint not found")
    model_id = str(ep.get("model_ref") or "unknown")
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(ep.get("created_at") or time.time()),
                "owned_by": "xcelsior",
            }
        ],
    }


# ── Worker callbacks (agent / scheduler) ─────────────────────────────


class WorkerReadyBody(BaseModel):
    host_id: str = ""


def _require_worker_callback(request: Request) -> None:
    from routes.instances import _require_worker_status_update

    _require_worker_status_update(request)


@router.post("/api/v2/serverless/workers/{worker_id}/ready", tags=["Serverless"])
def api_serverless_worker_ready(worker_id: str, request: Request, body: WorkerReadyBody | None = None):
    _require_worker_callback(request)
    host_id = str((body.host_id if body else "") or "")
    w = _svc().mark_worker_ready(worker_id, host_id=host_id)
    if not w:
        raise HTTPException(404, "Worker not found")
    return {"ok": True, "worker": w}


class JobCompleteBody(BaseModel):
    output: dict[str, Any] | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class JobEventBody(BaseModel):
    event_type: str = "progress"
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/api/v2/serverless/workers/{worker_id}/heartbeat", tags=["Serverless"])
def api_serverless_worker_heartbeat(worker_id: str, request: Request):
    _require_worker_callback(request)
    w = _svc().worker_heartbeat(worker_id)
    if not w:
        raise HTTPException(404, "Worker not found")
    return {"ok": True, "worker": w}


class WorkerExitedBody(BaseModel):
    exit_code: int = 0
    error_message: str = ""


@router.post("/api/v2/serverless/workers/{worker_id}/exited", tags=["Serverless"])
def api_serverless_worker_exited(
    worker_id: str,
    request: Request,
    body: WorkerExitedBody | None = None,
):
    _require_worker_callback(request)
    payload = body or WorkerExitedBody()
    w = _svc().worker_exited(
        worker_id,
        exit_code=int(payload.exit_code),
        error_message=str(payload.error_message or ""),
    )
    if not w:
        raise HTTPException(404, "Worker not found")
    return {"ok": True, "worker": w}


@router.post("/api/v2/serverless/workers/{worker_id}/jobs/claim", tags=["Serverless"])
def api_serverless_worker_claim_job(worker_id: str, request: Request):
    _require_worker_callback(request)
    job = _svc().worker_claim_job(worker_id)
    return {"ok": True, "job": job}


@router.get("/api/v2/serverless/workers/{worker_id}/jobs/{job_id}", tags=["Serverless"])
def api_serverless_worker_get_job(worker_id: str, job_id: str, request: Request):
    _require_worker_callback(request)
    job = _svc().worker_get_job(worker_id, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"ok": True, "job": job}


@router.post("/api/v2/serverless/workers/{worker_id}/jobs/{job_id}/complete", tags=["Serverless"])
def api_serverless_worker_complete_job(
    worker_id: str,
    job_id: str,
    request: Request,
    body: JobCompleteBody,
):
    _require_worker_callback(request)
    job = _svc().worker_complete_job(
        worker_id,
        job_id,
        output=body.output,
        error=body.error,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
    )
    if not job:
        raise HTTPException(404, "Job not found")
    return {"ok": True, "job": job}


@router.post("/api/v2/serverless/workers/{worker_id}/jobs/{job_id}/events", tags=["Serverless"])
def api_serverless_worker_job_event(
    worker_id: str,
    job_id: str,
    request: Request,
    body: JobEventBody,
):
    _require_worker_callback(request)
    ev = _svc().worker_append_event(worker_id, job_id, body.event_type, body.payload)
    if not ev:
        raise HTTPException(404, "Job not found")
    return {"ok": True, "event": ev}


# ── Deprecated /api/v2/inference/* aliases (managed endpoints → serverless) ──


@router.get("/api/v2/inference/endpoints", tags=["Serverless (compat)"])
def api_inference_compat_list_endpoints(request: Request):
    return api_serverless_list_endpoints(request)


@router.post("/api/v2/inference/endpoints", tags=["Serverless (compat)"])
def api_inference_compat_create_endpoint(body: ServerlessEndpointCreate, request: Request):
    return api_serverless_create_endpoint(body, request)


@router.get("/api/v2/inference/endpoints/{endpoint_id}", tags=["Serverless (compat)"])
def api_inference_compat_get_endpoint(endpoint_id: str, request: Request):
    return api_serverless_get_endpoint(endpoint_id, request)


@router.delete("/api/v2/inference/endpoints/{endpoint_id}", tags=["Serverless (compat)"])
def api_inference_compat_delete_endpoint(endpoint_id: str, request: Request):
    return api_serverless_delete_endpoint(endpoint_id, request)


@router.get("/api/v2/inference/endpoints/{endpoint_id}/health", tags=["Serverless (compat)"])
def api_inference_compat_endpoint_health(endpoint_id: str, request: Request):
    return api_serverless_endpoint_health(endpoint_id, request)


@router.get("/api/v2/inference/endpoints/{endpoint_id}/usage", tags=["Serverless (compat)"])
def api_inference_compat_endpoint_usage(endpoint_id: str, request: Request):
    return api_serverless_endpoint_usage(endpoint_id, request)