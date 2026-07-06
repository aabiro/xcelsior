#!/usr/bin/env python3
"""Drive chat/completions through OpenAI proxy → live vLLM; capture KV KPI."""

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


def main() -> int:
    from fastapi.testclient import TestClient

    from api import app
    from serverless.repo import ServerlessRepo

    SCRATCH.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    email = f"live-kv-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "LiveKV"},
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

    from scripts.live_vllm_common import resolve_hf_model_name, resolve_live_model_id

    request_model = resolve_live_model_id()
    hf_model = resolve_hf_model_name(request_model)
    created = client.post(
        "/api/v2/serverless/endpoints",
        headers=headers,
        json={
            "name": f"live-kv-{uuid.uuid4().hex[:4]}",
            "mode": "preset",
            "model_name": hf_model,
        },
    )
    if created.status_code != 200:
        print(f"endpoint create failed: {created.text[:300]}", file=sys.stderr)
        return 1
    ep_id = created.json()["endpoint"]["endpoint_id"]
    repo = ServerlessRepo()

    total_in = 0
    total_cached = 0
    errors: list[str] = []
    # Short prompts for live vLLM (max_model_len may be 512); warm + repeat for prefix hits.
    shared_system = (
        "You are the Xcelsior fleet status assistant. Reply in one short sentence."
    )
    base_prompts = [
        {
            "messages": [
                {"role": "system", "content": shared_system},
                {"role": "user", "content": msg},
            ],
            "max_tokens": 24,
            "temperature": 0,
        }
        for msg in (
            "live fleet status alpha",
            "live fleet status beta",
            "live fleet status gamma",
        )
    ]
    seq: list[tuple[str, dict]] = []
    for idx, payload in enumerate(base_prompts):
        seq.append((f"live-kv-warm-{idx}", payload))
        seq.append((f"live-kv-hit-{idx}", payload))
    for key, payload in seq:
        payload = dict(payload)
        payload["model"] = request_model
        resp = client.post(
            f"/v1/serverless/{ep_id}/openai/v1/chat/completions",
            headers={**headers, "idempotency-key": key},
            json=payload,
        )
        if resp.status_code != 200:
            errors.append(f"{key}:{resp.status_code}:{resp.text[:120]}")
            continue
        body = resp.json()
        usage = body.get("usage") or {}
        inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        cached = int(usage.get("cached_tokens") or 0)
        if cached == 0 and (
            body.get("xcelsior_cache_hit")
            or body.get("semantic_cache_hit")
            or body.get("x_semantic_cache_hit")
        ):
            cached = inp
        total_in += inp
        total_cached += cached
        time.sleep(0.05)

    kv_rate = round(total_cached / total_in, 4) if total_in > 0 else 0.0
    out = {
        "source": "openai_proxy_served_traffic",
        "upstream_mode": "live_vllm",
        "evidence_tier": "live_proxy_served_traffic",
        "kv_cache_hit_rate": kv_rate,
        "target": 0.30,
        "kpi_met": kv_rate >= 0.30 and total_in > 0,
        "total_input_tokens": total_in,
        "total_cached_tokens": total_cached,
        "proxy_ledger_rows": len(repo.list_token_ledger(ep_id)),
        "endpoint_id": ep_id,
        "live_port": os.environ.get("XCELSIOR_LIVE_VLLM_PORT"),
        "errors": errors,
    }
    path = SCRATCH / "kv-served-traffic-kpi.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)
    print(path)
    return 0 if out["kpi_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())