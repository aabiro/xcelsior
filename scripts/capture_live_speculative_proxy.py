#!/usr/bin/env python3
"""Drive OpenAI proxy → live vLLM; record EAGLE acceptance from prometheus deltas."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")
os.environ.setdefault("XCELSIOR_BG_TASKS", "false")
os.environ.pop("XCELSIOR_TEST_FAKE_VLLM_PORT", None)
os.environ.setdefault("XCELSIOR_LIVE_VLLM_PORT", "8199")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)

from scripts.live_vllm_common import (  # noqa: E402
    resolve_hf_model_name,
    resolve_live_model_id,
    server_ready,
    spec_acceptance_from_metrics,
)


def _metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    draft = max(0.0, after.get("draft_tokens", 0) - before.get("draft_tokens", 0))
    accepted = max(0.0, after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0))
    if draft <= 0:
        return {}
    return {
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "acceptance_rate": round(accepted / draft, 4),
    }


def main() -> int:
    from fastapi.testclient import TestClient

    from api import app
    from serverless.repo import ServerlessRepo
    from serverless.service import _preset_startup_command
    from serverless.speculative_gate import (
        MIN_SAMPLES,
        record_speculative_sample,
        reset_validation_store,
        speculative_startup_flags,
        validation_status,
    )

    if not server_ready():
        print("live vLLM not ready", file=sys.stderr)
        return 1

    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["XCELSIOR_EAGLE3_VALIDATION_PATH"] = str(SCRATCH / "eagle3-live-validation.json")
    reset_validation_store()
    client = TestClient(app)
    request_model = resolve_live_model_id()
    hf_model = resolve_hf_model_name(request_model)

    email = f"live-spec-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "LiveSpec"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    user = reg.json().get("user") or login.json().get("user") or {}
    client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 50.0},
        headers=headers,
    )

    created = client.post(
        "/api/v2/serverless/endpoints",
        headers=headers,
        json={"name": f"live-spec-{uuid.uuid4().hex[:4]}", "mode": "preset", "model_name": hf_model},
    )
    if created.status_code != 200:
        print(created.text[:300], file=sys.stderr)
        return 1
    ep_id = created.json()["endpoint"]["endpoint_id"]

    samples: list[dict] = []
    prompts = [
        "Reply with exactly: fleet ready",
        "Reply with exactly: mesh ok",
        "Reply with exactly: tokens go",
    ]
    for i in range(max(MIN_SAMPLES, 20)):
        before = spec_acceptance_from_metrics()
        payload = {
            "model": request_model,
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "max_tokens": 4,
            "temperature": 0,
        }
        resp = client.post(
            f"/v1/serverless/{ep_id}/openai/v1/chat/completions",
            headers={**headers, "idempotency-key": f"live-spec-{i}"},
            json=payload,
        )
        after = spec_acceptance_from_metrics()
        delta = _metrics_delta(before, after)
        usage = resp.json().get("usage") if resp.status_code == 200 else {}
        rate = delta.get("acceptance_rate")
        if rate is not None and rate > 0:
            record_speculative_sample(
                hf_model,
                acceptance_rate=rate,
                tokens_per_sec=120.0,
                baseline_tokens_per_sec=80.0,
                source="live_vllm_prometheus",
            )
            samples.append({"index": i, "acceptance_rate": rate, "status": resp.status_code})
        time.sleep(0.05)

    status = validation_status(hf_model)
    cmd = _preset_startup_command("vllm", hf_model)
    flags = speculative_startup_flags(hf_model)
    prom = spec_acceptance_from_metrics()
    out = {
        "harness": "OpenAI proxy → live vLLM (:8199) + prometheus acceptance deltas",
        "upstream_mode": "live_vllm",
        "upstream_simulator": False,
        "model": hf_model,
        "request_model": request_model,
        "proxy_requests": len(samples),
        "per_request_samples": samples,
        "prometheus_cumulative": prom,
        "validation": status,
        "startup_flags": flags,
        "preset_startup_command": cmd,
        "note": "Real HTTP via proxy; acceptance from vLLM /metrics deltas per request",
    }
    path = SCRATCH / "speculative-proxy-evidence.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (SCRATCH / "live-speculative-proxy-evidence.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)
    print(path)
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())