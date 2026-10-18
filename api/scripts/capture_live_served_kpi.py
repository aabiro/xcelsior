#!/usr/bin/env python3
"""Capture KV hit rate from live vLLM HTTP (not fake_vllm) for row 1 evidence."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_vllm_common import (
    DEFAULT_BASE_URL,
    resolve_hf_model_name,
    resolve_live_model_id,
    server_ready as _server_ready_common,
)

BASE_URL = DEFAULT_BASE_URL
SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)
SHARED_PREFIX = (
    "You are the Xcelsior fleet status assistant. Answer briefly about GPU queue depth, "
    "worker health, and token SKU launch readiness. "
)


def _chat(user_msg: str, model_id: str) -> dict:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SHARED_PREFIX},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 24,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH / "kv-live-served-kpi.json"
    model_id = resolve_live_model_id(BASE_URL)
    hf_model = resolve_hf_model_name(model_id)
    evidence: dict = {
        "source": "live_vllm_served_traffic",
        "upstream_mode": "live_vllm",
        "base_url": BASE_URL,
        "model": hf_model,
        "model_id": model_id,
        "target": 0.30,
        "kpi_met": False,
        "requests": [],
    }
    if not _server_ready_common(BASE_URL):
        evidence["reason"] = "live vLLM server not ready"
        out_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(out_path)
        return 1

    total_in = 0
    total_cached = 0
    # Warm once, then repeat an identical turn so prefix cache hits accrue.
    warm_msg = "Status check: report fleet readiness in one sentence."
    for i in range(6):
        try:
            payload = _chat(warm_msg, model_id)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError) as exc:
            evidence["error"] = str(exc)
            break
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        total_in += prompt
        total_cached += cached
        evidence["requests"].append(
            {"index": i, "prompt_tokens": prompt, "cached_tokens": cached, "usage": usage}
        )

    kv_rate = round(total_cached / total_in, 4) if total_in > 0 else 0.0
    evidence.update(
        {
            "kv_cache_hit_rate": kv_rate,
            "total_input_tokens": total_in,
            "total_cached_tokens": total_cached,
            "kpi_met": kv_rate >= 0.30,
            "request_count": len(evidence["requests"]),
        }
    )
    out_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(out_path)
    return 0 if evidence["kpi_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())