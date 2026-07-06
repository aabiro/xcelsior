"""Token billing closure: one suite emits all verification SCRATCH artifacts."""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from billing import get_billing_engine
from serverless.anchor_workloads import (
    ara_code_chat_payload,
    configured_project_roots,
    discover_anchor_repos,
    phantom_trades_chat_payload,
    pixelenhance_chat_payload,
    pixelenhance_embed_payload,
)
from serverless.metering import charge_serverless_execution
from serverless.repo import ServerlessRepo
from serverless.service import ServerlessService, _preset_startup_command
from serverless.speculative_gate import (
    MIN_SAMPLES,
    record_speculative_sample,
    reset_validation_store,
    speculative_startup_flags,
    validation_status,
)
from tests.fixtures.fake_vllm_upstream import _VLLMHandler
from tests.fixtures.token_billing_evidence import (
    register_funded_headers,
    write_json_artifact,
    write_metering_matrix,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    api_mod._RATE_BUCKETS.clear()


@pytest.fixture
def blended_billing_env(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SERVERLESS_BLENDED_BILLING", "1")
    monkeypatch.setenv("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "1")


@pytest.fixture
def speculative_store(tmp_path, monkeypatch):
    store = tmp_path / "eagle3.json"
    monkeypatch.setenv("XCELSIOR_EAGLE3_VALIDATION_PATH", str(store))
    monkeypatch.setenv("XCELSIOR_VLLM_EAGLE3", "1")
    reset_validation_store()
    yield store
    reset_validation_store()


class TestTokenBillingClosure:
    def test_metering_matrix_artifact(self):
        path = write_metering_matrix()
        assert path.is_file()
        assert "blended_charge" in path.read_text(encoding="utf-8")

    def test_anchor_workloads_and_discovery(self, fake_vllm_port):
        discovered = discover_anchor_repos()
        roots_report = configured_project_roots()
        assert len(discovered) >= 3, discovered
        assert any(r["path"] == "/Users/aaryn/Projects" for r in roots_report)
        assert any(r["exists"] for r in roots_report)

        headers, _ = register_funded_headers(client, "anchor")
        evidence: dict = {
            "configured_roots": roots_report,
            "discovered_repos": discovered,
            "workloads": [],
        }

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
            evidence["workloads"].append(
                {
                    "workload": key,
                    "source_repo": source_repo,
                    "request": payload,
                    "response_usage": body.get("usage"),
                    "ledger_rows": ledger,
                }
            )
            assert ledger and float(ledger[0]["cost_cad"]) > 0
            client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)

        write_json_artifact("anchor-workloads.json", evidence)
        write_json_artifact("anchor-workloads.log", evidence)

    def test_usage_api_proxy_traffic(self, fake_vllm_port):
        headers, cust = register_funded_headers(client, "usage")
        repo = ServerlessRepo()
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={
                "name": f"usage-{uuid.uuid4().hex[:4]}",
                "mode": "preset",
                "model_name": "Qwen/Qwen3-8B",
                "gpu_type": "RTX 4090",
            },
        )
        ep_id = created.json()["endpoint"]["endpoint_id"]
        prof = _VLLMHandler.usage_profiles["chat"]

        for i in range(5):
            payload = pixelenhance_chat_payload(user_message=f"fleet status {i}")
            r = client.post(
                f"/v1/serverless/{ep_id}/openai/v1/chat/completions",
                headers={**headers, "idempotency-key": f"usage-proxy-{i}"},
                json=payload,
            )
            assert r.status_code == 200

        w = repo.create_worker(ep_id, scheduler_job_id="sched-usage-job")
        repo.update_worker(str(w["worker_id"]), state="ready")
        job = repo.enqueue_job(ep_id, cust, phantom_trades_chat_payload())
        repo.claim_next_job(ep_id, str(w["worker_id"]))
        svc = ServerlessService(repo)
        with patch.object(svc, "_broadcast"):
            svc.worker_complete_job(
                str(w["worker_id"]),
                str(job["job_id"]),
                output={"analysis": "ok"},
                input_tokens=900,
                output_tokens=220,
                cached_tokens=180,
                ttft_ms=95,
            )

        r1 = client.get(f"/api/v2/serverless/endpoints/{ep_id}/usage", headers=headers)
        r2 = client.get(f"/api/v2/serverless/endpoints/{ep_id}/usage", headers=headers)
        health = client.get(f"/api/v2/serverless/endpoints/{ep_id}/health", headers=headers)
        ledger_rows = repo.list_token_ledger(ep_id)
        sample = {
            "usage_run1": r1.json(),
            "usage_run2": r2.json(),
            "health_pricing": health.json().get("health", {}).get("pricing"),
            "ledger_rows": ledger_rows,
            "worker_job_id": job["job_id"],
        }
        u = sample["usage_run1"]["usage"]
        assert u["pricing"]["token_billing"] is True
        assert u["last_24h"]["ttft_p95_ms"] > 0
        assert u["last_24h"]["tokens_per_sec"] > 0
        assert u["last_24h"]["jobs_completed"] >= 1
        assert sample["usage_run1"] == sample["usage_run2"]
        assert len([r for r in ledger_rows if "usage-proxy" in str(r.get("idempotency_key", ""))]) == 5
        assert u["last_24h"]["total_input_tokens"] >= prof["prompt_tokens"] * 5

        write_json_artifact("usage-sample.json", sample)
        client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)

    def test_proxy_stream_embed_worker_blended_charge(
        self, fake_vllm_port, blended_billing_env
    ):
        headers, cust = register_funded_headers(client, "e2e")
        repo = ServerlessRepo()
        billing = get_billing_engine()
        wallet_before = billing.get_wallet(cust)["balance_cad"]
        transcript: dict = {"cases": [], "upstream_requests": []}

        chat_ep = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={"name": f"e2e-{uuid.uuid4().hex[:4]}", "mode": "preset", "model_name": "Qwen/Qwen3-8B"},
        ).json()["endpoint"]
        chat_id = chat_ep["endpoint_id"]
        embed_ep = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={"name": f"emb-{uuid.uuid4().hex[:4]}", "mode": "preset", "model_name": "BAAI/bge-m3"},
        ).json()["endpoint"]
        embed_id = embed_ep["endpoint_id"]

        r_comp = client.post(
            f"/v1/serverless/{chat_id}/openai/v1/chat/completions",
            headers={**headers, "idempotency-key": "e2e-completion"},
            json=pixelenhance_chat_payload(),
        )
        assert r_comp.status_code == 200
        transcript["cases"].append({"case": "completion", "usage": r_comp.json().get("usage")})

        with client.stream(
            "POST",
            f"/v1/serverless/{chat_id}/openai/v1/chat/completions",
            headers={**headers, "idempotency-key": "e2e-stream"},
            json={**phantom_trades_chat_payload(), "stream": True},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            stream_text = "".join(stream_resp.iter_text())
        stream_row = next(
            (r for r in repo.list_token_ledger(chat_id) if r["idempotency_key"] == "e2e-stream"),
            None,
        )
        assert stream_row is not None
        transcript["cases"].append({"case": "stream", "sse_excerpt": stream_text[:300], "ledger": stream_row})

        r_emb = client.post(
            f"/v1/serverless/{embed_id}/openai/v1/embeddings",
            headers={**headers, "idempotency-key": "e2e-embed"},
            json=pixelenhance_embed_payload(),
        )
        assert r_emb.status_code == 200
        transcript["cases"].append({"case": "embeddings", "usage": r_emb.json().get("usage")})

        w = repo.create_worker(chat_id, scheduler_job_id="sched-blend")
        repo.update_worker(str(w["worker_id"]), state="ready", allocated_at=time.time() - 130)
        job = repo.enqueue_job(chat_id, cust, phantom_trades_chat_payload())
        repo.claim_next_job(chat_id, str(w["worker_id"]))
        svc = ServerlessService(repo)
        with patch.object(svc, "_broadcast"):
            svc.worker_complete_job(
                str(w["worker_id"]),
                str(job["job_id"]),
                output={"ok": True},
                input_tokens=2500,
                output_tokens=800,
                cached_tokens=1200,
                ttft_ms=88,
            )
        ep_row = repo.get_endpoint(chat_id) or {}
        ep_row["owner_id"] = cust
        charge = charge_serverless_execution(
            billing, repo, w, ep_row, period_end=time.time(), final=True
        )
        wallet_after = billing.get_wallet(cust)["balance_cad"]
        transcript["cases"].append(
            {
                "case": "blended_charge",
                "charge": charge,
                "wallet_before_cad": wallet_before,
                "wallet_after_cad": wallet_after,
            }
        )
        assert charge["token_cost_cad"] > 0
        assert charge["amount_cad"] >= charge["gpu_amount_cad"]

        transcript["upstream_requests"] = list(_VLLMHandler.requests_log)
        write_json_artifact("billing-e2e-transcript.json", transcript)

        client.delete(f"/api/v2/serverless/endpoints/{chat_id}", headers=headers)
        client.delete(f"/api/v2/serverless/endpoints/{embed_id}", headers=headers)

    def test_speculative_proxy_records_then_enables_flags(
        self, fake_vllm_port, speculative_store, monkeypatch
    ):
        model = "Qwen/Qwen3-8B"
        assert speculative_startup_flags(model) == []
        assert "EAGLE3" not in _preset_startup_command("vllm", model)

        headers, _ = register_funded_headers(client, "spec")
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={"name": "spec-test", "mode": "preset", "model_name": model},
        )
        ep_id = created.json()["endpoint"]["endpoint_id"]

        for i in range(MIN_SAMPLES):
            client.post(
                f"/v1/serverless/{ep_id}/openai/v1/chat/completions",
                headers={**headers, "idempotency-key": f"spec-{i}"},
                json=pixelenhance_chat_payload(user_message=f"spec probe {i}"),
            )

        status = validation_status(model)
        assert status["validated"] is True
        assert status["sample_count"] >= MIN_SAMPLES
        assert status["mean_acceptance_rate"] >= 0.75
        flags = speculative_startup_flags(model)
        assert "--speculative-algorithm" in flags
        cmd = _preset_startup_command("vllm", model)
        assert "EAGLE3" in cmd

        prof = _VLLMHandler.usage_profiles["chat"]
        acceptance = prof["accepted_tokens"] / (
            prof["accepted_tokens"] + prof["rejected_tokens"]
        )
        spec_evidence = {
            "harness": "fake_vllm_upstream → OpenAI proxy → accrue_proxy_token_usage → speculative_gate",
            "model": model,
            "proxy_requests": MIN_SAMPLES,
            "fake_vllm_acceptance_rate": round(acceptance, 4),
            "validation": status,
            "startup_flags": flags,
            "preset_startup_command": cmd,
            "upstream_requests": list(_VLLMHandler.requests_log)[-MIN_SAMPLES:],
        }
        write_json_artifact("eagle3-gate.log", spec_evidence)
        write_json_artifact("speculative-proxy-evidence.json", spec_evidence)

        client.delete(f"/api/v2/serverless/endpoints/{ep_id}", headers=headers)