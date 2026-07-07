"""Shared helpers for token-billing closure tests and SCRATCH artifacts."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

SCRATCH = Path(
    os.environ.get(
        "XCELSIOR_GOAL_SCRATCH",
        "/tmp/grok-goal-6f86c7cfe9c2/implementer",
    )
)


def scratch_path(name: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH / name


def write_json_artifact(name: str, payload: Any) -> Path:
    path = scratch_path(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def register_funded_headers(client: TestClient, prefix: str = "billing") -> tuple[dict, str]:
    email = f"{prefix}-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Billing Test"},
    )
    assert reg.status_code == 200, reg.text
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    body = login.json()
    user = reg.json().get("user") or body.get("user") or {}
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    dep = client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 100.0},
        headers=headers,
    )
    assert dep.status_code == 200
    return headers, user["customer_id"]


def write_metering_matrix() -> Path:
    from unittest.mock import MagicMock, patch

    from serverless.metering import (
        blended_period_amount,
        charge_serverless_execution,
        token_cost_metadata,
    )

    lines: list[str] = []
    for model, inp, out, cached in [
        ("Qwen/Qwen3-8B", 1000, 500, 400),
        ("meta-llama/Llama-3.3-70B-Instruct", 2000, 800, 0),
        ("BAAI/bge-m3", 12000, 0, 4000),
    ]:
        m = token_cost_metadata(inp, out, model_ref=model, cached_tokens=cached)
        gpu = 0.12
        blended = blended_period_amount(gpu, m["total_token_cost_cad"])
        lines.append(
            f"{model}: token_cad={m['total_token_cost_cad']:.6f} "
            f"gpu_cad={gpu:.6f} blended={blended:.6f}"
        )

    os.environ.pop("XCELSIOR_SERVERLESS_BLENDED_BILLING", None)
    os.environ["XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC"] = "1"
    billing = MagicMock()
    billing.charge.return_value = {"charged": True, "balance_cad": 90.0}
    repo = MagicMock()
    repo.consume_endpoint_token_cost.return_value = 5.0
    worker = {"worker_id": "w1", "scheduler_job_id": "j1", "allocated_at": 1000.0, "host_id": "h1"}
    endpoint = {
        "endpoint_id": "ep1",
        "owner_id": "c1",
        "gpu_tier": "RTX 4090",
        "region": "ca-east",
        "gpu_count": 1,
        "name": "t",
        "mode": "preset",
        "model_ref": "Qwen/Qwen3-8B",
    }
    with patch("serverless.metering.get_gpu_rate_per_hour", return_value=3.60), patch(
        "serverless.metering.last_billed_period_end", return_value=None
    ):
        r = charge_serverless_execution(
            billing, repo, worker, endpoint, period_end=1120.0, final=True
        )
    lines.append(f"blended_charge: {r}")

    path = scratch_path("metering-matrix.log")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path