"""Shared helpers for live vLLM evidence capture scripts."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("/mnt/storage/models/staging/2026H2/hf")
DEFAULT_BASE_URL = os.environ.get("XCELSIOR_LIVE_VLLM_URL", "http://127.0.0.1:8199/v1").rstrip("/")
DEFAULT_HF_MODEL = os.environ.get(
    "XCELSIOR_LIVE_VLLM_MODEL",
    "Qwen/Qwen3-4B-AWQ"
    if (DEFAULT_MODEL_ROOT / "Qwen_Qwen3-4B-AWQ").is_dir()
    else "Qwen/Qwen3-8B",
)


def server_ready(base_url: str = DEFAULT_BASE_URL) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionResetError):
        return False


def resolve_live_model_id(base_url: str = DEFAULT_BASE_URL) -> str:
    """Return the model id the live server expects (usually ``/model`` when mounted)."""
    override = os.environ.get("XCELSIOR_LIVE_VLLM_REQUEST_MODEL", "").strip()
    if override:
        return override
    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode())
        models = payload.get("data") or []
        if models and isinstance(models[0], dict):
            mid = str(models[0].get("id") or "").strip()
            if mid:
                return mid
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    local = DEFAULT_MODEL_ROOT / DEFAULT_HF_MODEL.replace("/", "_")
    return "/model" if local.is_dir() else DEFAULT_HF_MODEL


def resolve_hf_model_name(model_id: str) -> str:
    """Map ``/model`` mount back to a HF id for evidence metadata."""
    if model_id != "/model":
        return model_id
    return DEFAULT_HF_MODEL


def spec_acceptance_from_metrics(base_url: str = DEFAULT_BASE_URL) -> dict[str, float]:
    metrics_url = base_url.removesuffix("/v1") + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, TimeoutError):
        return {}
    draft = accepted = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num_draft_tokens_total{" in line:
            draft = float(line.rsplit(" ", 1)[-1])
        if "spec_decode_num_accepted_tokens_total{" in line:
            accepted = float(line.rsplit(" ", 1)[-1])
    if draft <= 0:
        return {}
    return {
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "acceptance_rate": round(accepted / draft, 4),
    }