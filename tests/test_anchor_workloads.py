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
    execute_anchor_builder,
    execute_anchor_on_mac,
    _mac_projects_reachable,
    phantom_trades_chat_payload,
    pixelenhance_chat_payload,
    pixelenhance_embed_payload,
    post_openai_from_mac,
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
    def test_pixelenhance_and_phantom_via_proxy(self, fake_vllm_port, mac_reachable_api):
        headers = _headers()
        discovered = discover_anchor_repos(include_remote=True)
        evidence: dict = {
            "configured_roots": configured_project_roots(),
            "discovered_repos": discovered,
            "mac_ssh_repos": [d for d in discovered if d.get("reachable") == "ssh_only"],
            "workloads": [],
        }
        assert len(discovered) >= 3, f"expected ≥3 anchor repos, found {discovered}"
        executed = [execute_anchor_builder(k) for k in ("pel-embed", "pel-chat", "pt-signal", "ara-agent")]
        evidence["executed_builders"] = executed
        evidence["local_builders"] = [
            e for e in executed if e.get("executed_on") == "local_import"
        ]
        mac_ssh_builders: list[dict] = []
        if _mac_projects_reachable():
            for workload in ("pt-signal", "ara-agent", "pel-chat"):
                mac = execute_anchor_on_mac(workload)
                if mac:
                    mac_ssh_builders.append(mac)
        evidence["mac_ssh_builders"] = mac_ssh_builders
        assert len(mac_ssh_builders) >= 2, f"expected ≥2 Mac SSH builders, got {mac_ssh_builders}"

        mac_ssh_inference: list[dict] = []

        mac_by_key = {m["workload"]: m for m in mac_ssh_builders}
        builder_by_key = {e["workload"]: e for e in executed}
        workload_specs = (
            ("BAAI/bge-m3", "embeddings", "pel-embed", "pixelenhance-labs"),
            ("Qwen/Qwen3-8B", "chat/completions", "pel-chat", "pixelenhance-labs"),
            ("Qwen/Qwen3-8B", "chat/completions", "pt-signal", "phantom-trades-mvp"),
            ("Qwen/Qwen3-8B", "chat/completions", "ara-agent", "ara-code"),
        )
        for model_name, route, key, source_repo in workload_specs:
            created = client.post(
                "/api/v2/serverless/endpoints",
                headers=headers,
                json={"name": f"anc-{key}", "mode": "preset", "model_name": model_name},
            )
            assert created.status_code == 200
            ep_id = created.json()["endpoint"]["endpoint_id"]
            built = mac_by_key.get(key) or builder_by_key.get(key) or execute_anchor_builder(key)
            payload = built["payload"]
            payload_source = str(built.get("executed_on") or "local_import")
            builder_meta = {
                "builder": built.get("builder"),
                "module": built.get("module"),
                "host": built.get("host"),
            }
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
                "payload_source": payload_source,
                "builder": builder_meta,
                "request": payload,
                "response_usage": body.get("usage"),
                "ledger_rows": ledger,
            }
            evidence["workloads"].append(row)
            if payload_source == "mac_ssh":
                mac_ssh_inference.append(
                    {
                        "workload": key,
                        "host": built.get("host"),
                        "ledger_cost_cad": float(ledger[0]["cost_cad"]) if ledger else 0,
                        "input_tokens": ledger[0].get("input_tokens") if ledger else 0,
                    }
                )
            assert ledger, f"no ledger for {key}"
            assert float(ledger[0]["cost_cad"]) > 0
            client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)

        if _mac_projects_reachable() and mac_ssh_builders:
            mac_builder = mac_ssh_builders[0]
            mac_payload = mac_builder["payload"]
            mac_ep = client.post(
                "/api/v2/serverless/endpoints",
                headers=headers,
                json={"name": "anc-mac-remote", "mode": "preset", "model_name": "Qwen/Qwen3-8B"},
            )
            assert mac_ep.status_code == 200
            mac_ep_id = mac_ep.json()["endpoint"]["endpoint_id"]
            mac_url = (
                f"{mac_reachable_api}/v1/serverless/{mac_ep_id}/openai/v1/chat/completions"
            )
            mac_result = post_openai_from_mac(
                url=mac_url,
                headers={**headers, "idempotency-key": "mac-pt-signal"},
                payload=mac_payload,
                workload=str(mac_builder.get("workload") or "pt-signal"),
            )
            evidence["mac_remote_inference_probe"] = mac_result
            if mac_result.get("ok"):
                mac_ledger = ServerlessRepo().list_token_ledger(mac_ep_id)
                mac_ssh_inference.append(
                    {
                        "workload": mac_result.get("workload"),
                        "executed_on": "mac_ssh",
                        "host": mac_result.get("host"),
                        "usage": mac_result.get("usage"),
                        "ledger_cost_cad": float(mac_ledger[0]["cost_cad"]) if mac_ledger else 0,
                        "input_tokens": mac_ledger[0].get("input_tokens") if mac_ledger else 0,
                    }
                )
                assert mac_ledger and float(mac_ledger[0]["cost_cad"]) > 0
            client.delete(f"/api/v2/serverless/endpoints/{mac_ep_id}", headers=headers)

        evidence["mac_ssh_inference"] = mac_ssh_inference
        evidence["mac_reachable"] = _mac_projects_reachable()
        evidence["mac_driven_workloads"] = len(
            [
                w
                for w in evidence["workloads"]
                if w.get("payload_source") == "mac_ssh"
            ]
        )
        evidence["mac_driven_inference"] = len(mac_ssh_inference)
        local_real = [
            w
            for w in evidence["workloads"]
            if w.get("payload_source") == "local_import"
            and w.get("source_repo") in ("pixelenhance-labs", "phantom-trades-mvp")
            and w.get("builder", {}).get("module")
        ]
        evidence["local_real_builders"] = len(local_real)
        assert len(local_real) >= 2 or len(mac_ssh_inference) >= 1, (
            "need ≥2 local_import workloads from real repos or ≥1 mac_ssh-driven inference"
        )
        if _mac_projects_reachable():
            assert len(mac_ssh_inference) >= 1, evidence.get("mac_remote_inference_probe")
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