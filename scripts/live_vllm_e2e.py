#!/usr/bin/env python3
"""Probe a live vLLM OpenAI server (docker or env URL) for real speculative usage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)
DEFAULT_MODEL_ROOT = Path("/mnt/storage/models/staging/2026H2/hf")
BASE_URL = os.environ.get("XCELSIOR_LIVE_VLLM_URL", "http://127.0.0.1:8199/v1")
MODEL = os.environ.get(
    "XCELSIOR_LIVE_VLLM_MODEL",
    "Qwen/Qwen3-4B-AWQ"
    if (DEFAULT_MODEL_ROOT / "Qwen_Qwen3-4B-AWQ").is_dir()
    else "Qwen/Qwen3-8B",
)
MODEL_PATH = os.environ.get("XCELSIOR_LIVE_VLLM_MODEL_PATH", "").strip()
DRAFT_PATH = os.environ.get("XCELSIOR_LIVE_VLLM_DRAFT_PATH", "").strip()
IMAGE = os.environ.get(
    "XCELSIOR_LIVE_VLLM_IMAGE",
    "vllm/vllm-openai:latest",
)


def _server_ready() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL.rstrip('/')}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def _post_chat(prompt: str = "Say 'fleet ready' in two words.", max_tokens: int = 32) -> dict:
    body = json.dumps(
        {
            "model": _request_model_id(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _resolve_local_paths() -> tuple[str, str | None]:
    """Map HF ids to on-disk staging paths when available."""
    if MODEL_PATH and Path(MODEL_PATH).is_dir():
        model_launch = MODEL_PATH
    else:
        local = DEFAULT_MODEL_ROOT / MODEL.replace("/", "_")
        model_launch = str(local) if local.is_dir() else MODEL
    draft_launch = None
    if DRAFT_PATH and Path(DRAFT_PATH).is_dir():
        draft_launch = DRAFT_PATH
    elif "qwen3-4b" in MODEL.lower():
        draft_local = DEFAULT_MODEL_ROOT / "AngelSlim_Qwen3-4B_eagle3"
        if draft_local.is_dir():
            draft_launch = str(draft_local)
    elif "qwen3" in MODEL.lower():
        for name in ("RedHatAI_Qwen3-8B-speculator.eagle3", "AngelSlim_Qwen3-4B_eagle3"):
            draft_local = DEFAULT_MODEL_ROOT / name
            if draft_local.is_dir():
                draft_launch = str(draft_local)
                break
    return model_launch, draft_launch


def _request_model_id() -> str:
    from scripts.live_vllm_common import resolve_live_model_id

    if _server_ready():
        return resolve_live_model_id(BASE_URL.rstrip("/"))
    return "/model" if MODEL_PATH or (DEFAULT_MODEL_ROOT / MODEL.replace("/", "_")).is_dir() else MODEL


def _spec_metrics() -> dict[str, float]:
    base = BASE_URL.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as resp:
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


def _try_docker_start() -> tuple[bool, str]:
    model_launch, draft_launch = _resolve_local_paths()
    use_eagle = bool(draft_launch) and os.environ.get("XCELSIOR_LIVE_VLLM_EAGLE3", "1") == "1"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--gpus",
        "all",
        "-p",
        "8199:8000",
    ]
    if model_launch.startswith("/"):
        cmd.extend(["-v", f"{model_launch}:/model:ro"])
    if use_eagle and draft_launch and draft_launch.startswith("/"):
        cmd.extend(["-v", f"{draft_launch}:/draft:ro"])
    cmd.append(IMAGE)
    if model_launch.startswith("/"):
        cmd.extend(["--model", "/model"])
    else:
        cmd.extend(["--model", model_launch])
    cmd.extend(
        [
            "--max-model-len",
            os.environ.get("XCELSIOR_LIVE_VLLM_MAX_LEN", "2048"),
            "--dtype",
            os.environ.get("XCELSIOR_LIVE_VLLM_DTYPE", "half"),
            "--gpu-memory-utilization",
            os.environ.get("XCELSIOR_LIVE_VLLM_GPU_UTIL", "0.92"),
            "--enable-prefix-caching",
        ]
    )
    quant = os.environ.get("XCELSIOR_LIVE_VLLM_QUANTIZATION", "").strip()
    if not quant and model_launch.endswith("Qwen_Qwen3-4B-AWQ"):
        quant = "awq"
    if quant:
        cmd.extend(["--quantization", quant])
    if use_eagle and draft_launch:
        spec_cfg = {
            "method": "eagle3",
            "model": "/draft" if draft_launch.startswith("/") else draft_launch,
            "num_speculative_tokens": int(os.environ.get("XCELSIOR_LIVE_VLLM_SPEC_TOKENS", "2")),
        }
        cmd.extend(["--speculative-config", json.dumps(spec_cfg)])
    try:
        cid = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        return True, cid
    except subprocess.CalledProcessError as exc:
        return False, exc.output


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH / "live-vllm-e2e.json"
    model_launch, draft_launch = _resolve_local_paths()
    evidence: dict = {
        "base_url": BASE_URL,
        "model": MODEL,
        "model_launch": model_launch,
        "draft_launch": draft_launch,
        "started_container": False,
        "container_id": "",
        "usage": {},
        "error": None,
        "harness": "docker_or_existing_vllm_openai_server",
    }
    cid = ""
    mesh_hosts = os.environ.get("XCELSIOR_LMCACHE_MESH_HOSTS", "")
    evidence["mesh_host_count"] = len([h for h in mesh_hosts.split(",") if h.strip()])
    evidence["preexisting_server"] = _server_ready()
    start_docker = os.environ.get("XCELSIOR_LIVE_VLLM_START_DOCKER", "0") == "1"
    if start_docker and not evidence["preexisting_server"]:
        ok, msg = _try_docker_start()
        evidence["started_container"] = ok
        if ok:
            cid = msg
            evidence["container_id"] = cid
            time.sleep(45)
        else:
            evidence["docker_start_error"] = msg[:2000]
    elif evidence["preexisting_server"]:
        evidence["started_container"] = False
        evidence["note"] = "reused existing vLLM server at BASE_URL"

    try:
        if not _server_ready():
            raise urllib.error.URLError("vLLM server not ready at BASE_URL")
        samples = int(os.environ.get("XCELSIOR_LIVE_VLLM_SAMPLES", "8"))
        for i in range(samples):
            _post_chat(
                prompt=f"Fleet mesh probe {i}: summarize token SKU readiness.",
                max_tokens=48,
            )
        payload = _post_chat()
        usage = payload.get("usage") or {}
        evidence["usage"] = usage
        evidence["proxy_requests"] = samples + 1
        details = usage.get("completion_tokens_details") or {}
        accepted = int(details.get("accepted_tokens") or usage.get("speculative_accepted_tokens") or 0)
        rejected = int(details.get("rejected_tokens") or usage.get("speculative_rejected_tokens") or 0)
        if accepted + rejected > 0:
            evidence["acceptance_rate"] = round(accepted / (accepted + rejected), 4)
        metrics = _spec_metrics()
        if metrics:
            evidence["spec_metrics"] = metrics
            evidence.setdefault("acceptance_rate", metrics.get("acceptance_rate"))
        evidence["eagle3_enabled"] = bool(draft_launch)
        evidence["upstream_mode"] = "live_vllm"
        evidence["choices_preview"] = (
            payload.get("choices", [{}])[0].get("message", {}).get("content", "")[:200]
        )
        evidence["ok"] = True
        evidence["harness"] = "live_vllm_openai_server"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionResetError) as exc:
        evidence["ok"] = False
        evidence["error"] = str(exc)

    if cid:
        subprocess.run(["docker", "stop", cid], check=False, capture_output=True)

    out_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(out_path)
    return 0 if evidence.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())