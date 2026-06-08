"""Routes: inference."""

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routes._deps import (
    _effective_billing_customer_id,
    _get_current_user,
    _inference_owner_ids_readable,
    _inference_scope_owner_id,
    _require_auth,
    _require_inference_endpoint_access,
    _require_inference_endpoint_write,
    _require_inference_job_access,
    _require_scope,
    _require_team_instance_write,
    broadcast_sse,
    otel_span,
)
from scheduler import (
    list_jobs,
    submit_job,
)
from inference_store import (
    get_inference_job,
    get_inference_result,
    store_inference_job,
    store_inference_result,
)
from inference import get_inference_engine

router = APIRouter()


# ── Model: InferenceRequest ──


class InferenceRequest(BaseModel):
    model: str = Field(
        ...,
        description="Model name or HuggingFace repo (e.g. 'distilbert-base-uncased-finetuned-sst-2-english')",
    )
    inputs: list[str] | str = Field(..., description="Text input(s) for inference")
    gpu_model: str = Field("any", description="Preferred GPU model or 'any'")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    timeout_sec: int = Field(300, ge=10, le=3600)


@router.post("/api/inference", tags=["Inference"])
def api_inference_submit(req: InferenceRequest, request: Request):
    """Submit a serverless inference request.

    Schedules a short-lived GPU job that runs the specified model on the
    provided inputs. Returns a job_id to poll for results.
    """
    user = _require_auth(request)
    _require_scope(user, "inference:write")
    _require_team_instance_write(user)
    customer_id = _effective_billing_customer_id(user) or "anon"
    inputs_list = [req.inputs] if isinstance(req.inputs, str) else req.inputs

    # Wallet pre-flight
    from billing import get_billing_engine

    wallet = get_billing_engine().get_wallet(customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(402, detail="Wallet suspended 2 please add funds to resume service")
    if wallet["balance_cad"] <= 0 and (wallet.get("grace_until") or 0) < time.time():
        raise HTTPException(402, detail="Insufficient wallet balance 2 please deposit credits")

    job = submit_job(
        name=f"inference:{req.model.replace('/', '--')}",
        vram_needed_gb=2,
        image=f"xcelsior/inference:{req.model.replace('/', '--')}",
        owner=customer_id,
    )
    job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
    # Persist inference metadata to SQLite (survives API restarts)
    store_inference_job(
        job_id=job_id,
        customer_id=customer_id,
        model=req.model,
        inputs=inputs_list,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        timeout_sec=req.timeout_sec,
    )
    broadcast_sse("inference_submitted", {"job_id": job_id, "model": req.model})
    return {"ok": True, "job_id": job_id, "model": req.model, "status": "queued"}


@router.get("/api/inference/{job_id}", tags=["Inference"])
def api_inference_result(job_id: str, request: Request):
    """Get inference results for a submitted request."""
    user = _require_auth(request)
    _require_scope(user, "inference:read")
    _require_inference_job_access(user, job_id)
    result = get_inference_result(job_id)
    if result:
        return {"ok": True, "status": "completed", **result}
    meta = get_inference_job(job_id)
    if meta:
        elapsed = time.time() - meta["submitted_at"]
        if elapsed > meta["timeout_sec"]:
            return {"ok": False, "status": "timeout", "job_id": job_id}
        return {
            "ok": True,
            "status": meta.get("status", "running"),
            "job_id": job_id,
            "model": meta["model"],
            "elapsed_sec": round(elapsed, 1),
        }
    # Check scheduler
    jobs = list_jobs()
    job = next((j for j in jobs if j.get("job_id") == job_id), None)
    if job:
        return {"ok": True, "status": job.get("status", "unknown"), "job_id": job_id}
    raise HTTPException(404, f"Inference job {job_id} not found")


@router.get("/api/inference/models/available", tags=["Inference"])
def api_inference_models(request: Request):
    """List available inference models and their resource requirements."""
    user = _require_auth(request)
    _require_scope(user, "inference:read")
    models = [
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "task": "sentiment-analysis",
            "min_vram_gb": 1,
            "avg_latency_ms": 50,
        },
        {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "min_vram_gb": 14,
            "avg_latency_ms": 2000,
        },
        {
            "name": "meta-llama/Llama-2-13b-chat-hf",
            "task": "text-generation",
            "min_vram_gb": 26,
            "avg_latency_ms": 4000,
        },
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "task": "image-generation",
            "min_vram_gb": 8,
            "avg_latency_ms": 5000,
        },
        {
            "name": "openai/whisper-large-v3",
            "task": "speech-to-text",
            "min_vram_gb": 4,
            "avg_latency_ms": 3000,
        },
        {
            "name": "BAAI/bge-large-en-v1.5",
            "task": "embeddings",
            "min_vram_gb": 2,
            "avg_latency_ms": 100,
        },
    ]
    return {"ok": True, "models": models}


# ── Model: InferenceResultCallback ──


class InferenceResultCallback(BaseModel):
    outputs: list = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0


@router.post("/api/inference/{job_id}/result", tags=["Inference"])
def api_inference_post_result(job_id: str, body: InferenceResultCallback, request: Request):
    """Worker callback: post inference results. Internal use."""
    from routes.instances import _require_worker_status_update

    _require_worker_status_update(request)
    store_inference_result(
        job_id=job_id,
        outputs=body.outputs,
        model=body.model,
        latency_ms=body.latency_ms,
    )
    broadcast_sse("inference_completed", {"job_id": job_id})
    return {"ok": True}


# ── Model: V1InferenceRequest ──


class V1InferenceRequest(BaseModel):
    """OpenAI-compatible inference request for /v1/inference."""

    model: str = Field(..., description="Model name or HuggingFace repo")
    inputs: list[str] | str = Field(..., description="Text input(s) for inference")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    stream: bool = Field(False, description="Stream response via SSE")


@router.post("/v1/inference", tags=["Inference v2"])
def api_v1_inference_sync(body: V1InferenceRequest, request: Request):
    """Synchronous inference endpoint (OpenAI-compatible path).

    Submits an inference request and polls for results up to 30 seconds.
    If stream=true, returns SSE text/event-stream with token deltas.
    """
    user = _require_auth(request)
    _require_team_instance_write(user)
    customer_id = _effective_billing_customer_id(user) or "anon"
    inputs_list = [body.inputs] if isinstance(body.inputs, str) else body.inputs

    # Wallet pre-flight
    from billing import get_billing_engine

    wallet = get_billing_engine().get_wallet(customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(402, detail="Wallet suspended — please add funds to resume service")
    if wallet["balance_cad"] <= 0 and (wallet.get("grace_until") or 0) < time.time():
        raise HTTPException(402, detail="Insufficient wallet balance — please deposit credits")

    with otel_span("inference.v1.sync", {"model": body.model, "stream": body.stream}):
        job = submit_job(
            name=f"inference:{body.model.replace('/', '--')}",
            vram_needed_gb=2,
            image=f"xcelsior/inference:{body.model.replace('/', '--')}",
            owner=customer_id,
        )
        job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
        store_inference_job(
            job_id=job_id,
            customer_id=customer_id,
            model=body.model,
            inputs=inputs_list,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            timeout_sec=30,
        )

        if body.stream:
            # SSE streaming response (OpenAI delta format)
            async def _sse_stream():
                start = time.time()
                yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                while time.time() - start < 30:
                    result = get_inference_result(job_id)
                    if result:
                        outputs = result.get("outputs", [])
                        for token in outputs:
                            yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                        yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    await asyncio.sleep(0.5)
                yield f"data: {json.dumps({'id': job_id, 'error': 'timeout'})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse_stream(), media_type="text/event-stream")

        # Sync: poll up to 30 seconds
        start = time.time()
        while time.time() - start < 30:
            result = get_inference_result(job_id)
            if result:
                return {
                    "id": job_id,
                    "object": "inference.completion",
                    "created": int(time.time()),
                    "model": body.model,
                    "outputs": result.get("outputs", []),
                    "usage": {
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "total_tokens": result.get("input_tokens", 0)
                        + result.get("output_tokens", 0),
                    },
                    "latency_ms": result.get("latency_ms", 0),
                }
            time.sleep(0.5)

        # Timeout — return pending status
        return JSONResponse(
            status_code=202,
            content={
                "id": job_id,
                "object": "inference.pending",
                "model": body.model,
                "status": "processing",
                "poll_url": f"/v1/inference/{job_id}",
            },
        )


@router.post("/v1/inference/async", tags=["Inference v2"])
def api_v1_inference_async(body: V1InferenceRequest, request: Request):
    """Asynchronous inference — returns job_id immediately for polling."""
    user = _require_auth(request)
    _require_team_instance_write(user)
    customer_id = _effective_billing_customer_id(user) or "anon"
    inputs_list = [body.inputs] if isinstance(body.inputs, str) else body.inputs

    # Wallet pre-flight
    from billing import get_billing_engine

    wallet = get_billing_engine().get_wallet(customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(402, detail="Wallet suspended — please add funds to resume service")
    if wallet["balance_cad"] <= 0 and (wallet.get("grace_until") or 0) < time.time():
        raise HTTPException(402, detail="Insufficient wallet balance — please deposit credits")

    with otel_span("inference.v1.async", {"model": body.model}):
        job = submit_job(
            name=f"inference:{body.model.replace('/', '--')}",
            vram_needed_gb=2,
            image=f"xcelsior/inference:{body.model.replace('/', '--')}",
            owner=customer_id,
        )
        job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
        store_inference_job(
            job_id=job_id,
            customer_id=customer_id,
            model=body.model,
            inputs=inputs_list,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            timeout_sec=300,
        )

    return {
        "id": job_id,
        "object": "inference.async",
        "model": body.model,
        "status": "queued",
        "poll_url": f"/v1/inference/{job_id}",
    }


@router.get("/v1/inference/{job_id}", tags=["Inference v2"])
def api_v1_inference_poll(job_id: str, request: Request):
    """Poll for inference results by job_id."""
    user = _require_auth(request)
    _require_scope(user, "inference:read")
    _require_inference_job_access(user, job_id)
    result = get_inference_result(job_id)
    if result:
        return {
            "id": job_id,
            "object": "inference.completion",
            "created": int(time.time()),
            "model": result.get("model", ""),
            "outputs": result.get("outputs", []),
            "usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
            },
            "latency_ms": result.get("latency_ms", 0),
            "status": "completed",
        }
    meta = get_inference_job(job_id)
    if meta:
        elapsed = time.time() - meta["submitted_at"]
        if elapsed > meta["timeout_sec"]:
            return {"id": job_id, "status": "timeout", "elapsed_sec": round(elapsed, 1)}
        return {"id": job_id, "status": "processing", "elapsed_sec": round(elapsed, 1)}
    raise HTTPException(404, f"Inference job {job_id} not found")
