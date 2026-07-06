"""Anchor workloads from project roots — real payloads + HTTP + ledger rows."""

from __future__ import annotations

import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from serverless.anchor_workloads import (
    ara_code_chat_payload,
    configured_project_roots,
    discover_anchor_repos,
    phantom_trades_chat_payload,
    pixelenhance_chat_payload,
    pixelenhance_embed_payload,
)
from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.service import ServerlessService

client = TestClient(app)
SCRATCH = os.environ.get(
    "XCELSIOR_GOAL_SCRATCH",
    "/tmp/grok-goal-6f86c7cfe9c2/implementer",
)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _headers():
    email = f"anchor-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Anchor"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    body = login.json()
    h = {"Authorization": f"Bearer {body['access_token']}"}
    user = reg.json().get("user") or body.get("user") or {}
    client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 50.0},
        headers=h,
    )
    return h


class TestAnchorWorkloadsHTTP:
    def test_pixelenhance_and_phantom_via_proxy(self, fake_vllm_port):
        headers = _headers()
        discovered = discover_anchor_repos()
        evidence: dict = {
            "configured_roots": configured_project_roots(),
            "discovered_repos": discovered,
            "workloads": [],
        }
        assert len(discovered) >= 3, f"expected ≥3 anchor repos, found {discovered}"

        for model_name, workload_fn, route, key, source_repo in (
            ("BAAI/bge-m3", pixelenhance_embed_payload, "embeddings", "pel-embed", "pixelenhance-labs"),
            ("Qwen/Qwen3-8B", pixelenhance_chat_payload, "chat/completions", "pel-chat", "pixelenhance-labs"),
            ("Qwen/Qwen3-8B", phantom_trades_chat_payload, "chat/completions", "pt-signal", "phantom-trades-mvp"),
            ("Qwen/Qwen3-8B", ara_code_chat_payload, "chat/completions", "ara-agent", "ara-code"),
        ):
            created = client.post(
                "/api/v2/serverless/endpoints",
                headers=headers,
                json={"name": f"anc-{key}", "mode": "preset", "model_name": model_name},
            )
            assert created.status_code == 200
            ep_id = created.json()["endpoint"]["endpoint_id"]
            payload = workload_fn()
            path = (
                f"/v1/serverless/{ep_id}/openai/v1/embeddings"
                if route == "embeddings"
                else f"/v1/serverless/{ep_id}/openai/v1/chat/completions"
            )
            resp = client.post(
                path,
                headers={**headers, "idempotency-key": key},
                json=payload,
            )
            assert resp.status_code == 200, resp.text[:400]
            body = resp.json()
            ledger = ServerlessRepo().list_token_ledger(ep_id)
            row = {
                "workload": key,
                "source_repo": source_repo,
                "request": payload,
                "response_usage": body.get("usage"),
                "ledger_rows": ledger,
            }
            evidence["workloads"].append(row)
            assert ledger, f"no ledger for {key}"
            assert float(ledger[0]["cost_cad"]) > 0
            client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)

        os.makedirs(SCRATCH, exist_ok=True)
        log_path = os.path.join(SCRATCH, "anchor-workloads.log")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, default=str)
        with open(os.path.join(SCRATCH, "anchor-workloads.json"), "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, default=str)

    def test_custom_endpoint_skips_token_ledger(self):
        from unittest.mock import patch

        repo = ServerlessRepo()
        owner = f"custom-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="custom-sr",
                mode="custom",
                image_ref="xcelsior/serverless-base:cuda12.4-py3.12",
                min_workers=0,
            )
        )
        worker = repo.create_worker(str(ep["endpoint_id"]), scheduler_job_id="sched-custom")
        repo.update_worker(str(worker["worker_id"]), state="ready")
        job = repo.enqueue_job(str(ep["endpoint_id"]), owner, {"image": "x.png"})
        repo.claim_next_job(str(ep["endpoint_id"]), str(worker["worker_id"]))
        svc = ServerlessService(repo)
        with patch.object(svc, "_broadcast"):
            svc.worker_complete_job(
                str(worker["worker_id"]),
                str(job["job_id"]),
                output={"ok": True},
                input_tokens=9_999,
                output_tokens=9_999,
            )
        assert repo.list_token_ledger(str(ep["endpoint_id"])) == []