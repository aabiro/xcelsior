"""MCP/assistant safety eval suite — spend, isolation, provisioning guardrails (row 25)."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"

from api import app
from serverless.repo import EndpointCreate, ServerlessRepo
from tests.fixtures.token_billing_evidence import register_funded_headers

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


class TestSpendGuardrails:
    def test_pel_guardrail_rejects_zero_wallet(self, monkeypatch):
        import routes._deps as deps

        email = f"safety-{uuid.uuid4().hex[:8]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": "Safety"},
        )
        login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        r = client.post(
            "/api/v2/serverless/should-i-run-this",
            headers=headers,
            json={"model_ref": "Qwen/Qwen3-8B", "estimated_input_tokens": 5000},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["approved"] is False
        assert any("zero" in x.lower() or "balance" in x.lower() for x in body["reasons"])

    def test_tenant_isolation_wrong_owner_endpoint(self):
        headers_a, owner_a = register_funded_headers(client, "iso-a")
        headers_b, _owner_b = register_funded_headers(client, "iso-b")
        repo = ServerlessRepo()
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner_a,
                name="iso",
                mode="preset",
                model_ref="Qwen/Qwen3-8B",
                managed_engine="vllm",
            )
        )
        r = client.post(
            "/api/v2/serverless/should-i-run-this",
            headers=headers_b,
            json={"endpoint_id": ep["endpoint_id"]},
        )
        assert r.status_code == 404
        repo.soft_delete_endpoint(ep["endpoint_id"], owner_a)


class TestProvisioningSafety:
    def test_attestation_schema_public(self):
        r = client.get("/api/v2/platform/attestation-schema")
        assert r.status_code == 200
        schema = r.json()["schema"]
        assert "tee_evidence_jwt" in schema["fields"]

    def test_mcp_scope_map_includes_pel_guardrail(self):
        from pathlib import Path

        scopes = (Path(__file__).resolve().parent.parent / "mcp/src/auth/scopes.ts").read_text()
        assert "should_i_run_pel_job" in scopes


class TestModularizationHygiene:
    def test_no_committed_bak_files_in_repo(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        baks = list(root.rglob("*.bak")) + list(root.rglob("*.bak*"))
        assert baks == [], f"Remove backup files: {baks[:5]}"

    def test_worker_agent_uses_criu_hosts_module(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "worker_agent.py").read_text()
        assert "from criu_hosts import" in src

    def test_ai_assistant_config_module_extracted(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        assert (root / "ai_assistant_config.py").is_file()
        assistant = (root / "ai_assistant.py").read_text()
        assert "from ai_assistant_config import" in assistant