# Xcelsior — OpenAI-compatible proxy to warm vLLM workers (preset mode)

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("xcelsior.serverless.openai_proxy")

SUPPORTED_ROUTES = frozenset(
    {
        "chat/completions",
        "completions",
        "embeddings",
        "rerank",
        "score",
        "models",
    }
)

# Preset model-task classification by model id. Embedding/reranker presets launch
# vLLM with --task embed / --task score and expose /v1/embeddings or /v1/rerank.
_RERANK_MARKERS = ("rerank", "reranker", "cross-encoder")
_EMBED_MARKERS = (
    "bge-m3", "bge-large", "bge-base", "bge-small", "nomic-embed", "gte-",
    "e5-large", "e5-base", "e5-small", "stella", "snowflake-arctic-embed",
    "embed",
)


def model_task(model_ref: str | None) -> str:
    """Classify a preset model as 'embed', 'rerank', or 'chat' (default)."""
    if not model_ref:
        return "chat"
    s = model_ref.lower()
    if any(m in s for m in _RERANK_MARKERS):
        return "rerank"
    if any(m in s for m in _EMBED_MARKERS):
        return "embed"
    return "chat"


CAPABILITY_REQUIREMENTS: dict[str, set[str]] = {
    "embeddings": {"embeddings"},
    "chat/completions": {"chat"},
    "completions": {"completions"},
    "models": {"models"},
}


class OpenAIProxyError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def normalize_route(path: str) -> str:
    """Strip /openai/v1/ prefix → chat/completions, models, etc."""
    p = path.strip("/")
    for prefix in ("openai/v1/", "v1/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
    return p


def capability_gate(route: str, endpoint: dict, body: dict | None = None) -> None:
    """
    Reject unsupported routes/features with explicit 4xx (§11.2).

    The OpenAI route must match the endpoint's model task: chat models serve
    /v1/chat/completions, embedding models serve /v1/embeddings, reranker models
    serve /v1/rerank|/v1/score.
    """
    norm = normalize_route(route)
    if norm not in SUPPORTED_ROUTES:
        raise OpenAIProxyError(
            404,
            "route_not_supported",
            f"OpenAI route '/{norm}' is not supported on this endpoint",
        )
    if endpoint.get("mode") == "custom" and norm != "models":
        # Custom images use queue API; OpenAI proxy is preset-only in v1.
        raise OpenAIProxyError(
            400,
            "preset_only",
            "OpenAI proxy is only available for preset (managed) endpoints",
        )

    # Route ↔ model-task must agree (skip 'models', which works for any task).
    task = model_task(endpoint.get("model_ref"))
    if norm == "embeddings" and task != "embed":
        raise OpenAIProxyError(
            400,
            "wrong_model_task",
            f"/v1/embeddings requires an embeddings model; this endpoint serves a {task} model.",
        )
    if norm in ("rerank", "score") and task != "rerank":
        raise OpenAIProxyError(
            400,
            "wrong_model_task",
            f"/v1/{norm} requires a reranker model; this endpoint serves a {task} model.",
        )
    if norm in ("chat/completions", "completions") and task in ("embed", "rerank"):
        alt = "embeddings" if task == "embed" else "rerank"
        raise OpenAIProxyError(
            400,
            "wrong_model_task",
            f"This endpoint serves a {task} model — use /v1/{alt} instead.",
        )

    if body and _body_requests_vision(body) and norm == "chat/completions":
        raise OpenAIProxyError(
            400,
            "vision_not_supported",
            "Vision inputs are not supported on this worker",
        )


def _body_requests_vision(body: dict) -> bool:
    messages = body.get("messages") or []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _job_payload_dict(worker_row: dict) -> dict:
    payload = worker_row.get("job_payload") or {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            import json

            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def worker_base_url(worker_row: dict, http_port: int) -> str | None:
    """Build http://host:port from joined worker/job/host row."""
    payload = _job_payload_dict(worker_row)
    ip = str(worker_row.get("host_ip") or "").strip()
    if not ip:
        ip = str(payload.get("host_ip") or payload.get("ip") or "").strip()
    if not ip:
        return None
    host_port = http_port
    http_ports = payload.get("http_ports") or {}
    if isinstance(http_ports, dict):
        mapped = http_ports.get(str(http_port)) or http_ports.get(http_port)
        if mapped is not None:
            host_port = int(mapped)
    return f"http://{ip}:{host_port}"


def build_upstream_url(worker_row: dict, endpoint: dict, route: str) -> str:
    base = worker_base_url(worker_row, int(endpoint.get("http_port") or 8080))
    if not base:
        raise OpenAIProxyError(503, "worker_unreachable", "No warm worker available for proxy")
    norm = normalize_route(route)
    return f"{base}/v1/{norm}"


def proxy_request(
    worker_row: dict,
    endpoint: dict,
    route: str,
    *,
    method: str = "POST",
    json_body: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout_sec: float = 120.0,
) -> httpx.Response:
    capability_gate(route, endpoint, json_body)
    url = build_upstream_url(worker_row, endpoint, route)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update({k: v for k, v in headers.items() if k.lower() != "host"})
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            if method.upper() == "GET":
                resp = client.get(url, headers=hdrs)
            else:
                resp = client.post(url, json=json_body or {}, headers=hdrs)
        return resp
    except httpx.TimeoutException as e:
        raise OpenAIProxyError(504, "upstream_timeout", str(e)) from e
    except httpx.HTTPError as e:
        raise OpenAIProxyError(502, "upstream_error", str(e)) from e


async def async_proxy_stream_lines(
    worker_row: dict,
    endpoint: dict,
    route: str,
    *,
    json_body: dict | None = None,
    timeout_sec: float = 120.0,
):
    """Async line iterator for upstream OpenAI SSE streams."""
    capability_gate(route, endpoint, json_body)
    url = build_upstream_url(worker_row, endpoint, route)
    timeout = httpx.Timeout(timeout_sec, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            json=json_body or {},
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread())[:500].decode("utf-8", errors="replace")
                raise OpenAIProxyError(
                    resp.status_code,
                    "upstream_http_error",
                    body or f"Upstream returned {resp.status_code}",
                )
            async for line in resp.aiter_lines():
                yield line


def proxy_chat_completions(
    worker_row: dict,
    endpoint: dict,
    body: dict[str, Any],
    *,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    timeout = float(timeout_sec or endpoint.get("request_timeout_sec") or 120)
    resp = proxy_request(
        worker_row,
        endpoint,
        "chat/completions",
        json_body=body,
        timeout_sec=timeout,
    )
    if resp.status_code >= 400:
        raise OpenAIProxyError(
            resp.status_code,
            "upstream_http_error",
            resp.text[:500] or f"Upstream returned {resp.status_code}",
        )
    return resp.json()