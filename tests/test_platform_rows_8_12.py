"""Rows 8–12: attestation schema, ops plan, host admission, capacity forecast, PEL guardrail."""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"

from api import app
from host_attestation import (
    attestation_schema,
    normalize_attestation,
    platform_ops_plan,
    validate_attestation,
)
from security import admit_node
from serverless.capacity_planner import (
    demand_breach_alert,
    forecast_gpu_demand_14d,
    weekday_covariate,
)

client = TestClient(app)


class TestAttestationSchema:
    def test_schema_has_tee_and_nras_fields(self):
        schema = attestation_schema()
        assert schema["version"] == "1.0"
        assert "tee_evidence_jwt" in schema["fields"]
        assert "nras_verify_status" in schema["fields"]

    def test_validate_attested_requires_jwt(self):
        ok, reasons = validate_attestation({"attestation_tier": "attested"})
        assert not ok
        assert any("tee_evidence_jwt" in r for r in reasons)

    def test_admit_node_merges_attestation(self):
        att = {
            "tee_evidence_jwt": "eyJ.test",
            "nras_verify_status": "verified",
            "attestation_tier": "attested",
        }
        admitted, details = admit_node(
            "h-att",
            {"runc": "1.2.0", "nvidia_ctk": "1.17.0"},
            "H100",
            attestation=att,
        )
        assert "attestation" in details
        assert details["attestation"]["attestation_tier"] == "attested"


class TestOpsPlan:
    def test_ops_plan_api(self):
        r = client.get("/api/v2/platform/ops-plan")
        assert r.status_code == 200
        plan = r.json()["plan"]
        assert plan["horizon_months"] == 18
        assert "mesh_hosts" in plan["infrastructure"]
        assert "sovereign" not in str(plan).lower()


class TestCapacityPlanner:
    def test_weekday_covariate(self):
        assert weekday_covariate() > 0

    def test_forecast_14d(self):
        now = time.time()
        hist = [(now - 3600 * i, 2 + (i % 3)) for i in range(12)]
        fc = forecast_gpu_demand_14d(hist)
        assert "forecast_workers" in fc
        assert fc["horizon_days"] == 14

    def test_breach_alert_above_band(self):
        fc = {"interval_low": 0, "interval_high": 2}
        alert = demand_breach_alert(10, fc)
        assert alert is not None
        assert alert["kind"] == "demand_above_band"


class TestPelGuardrail:
    def test_should_i_run_endpoint(self, monkeypatch):
        import routes._deps as deps
        import routes.auth as auth
        import api as api_mod

        monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
        monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
        monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)

        from serverless.repo import EndpointCreate, ServerlessRepo
        from tests.fixtures.token_billing_evidence import register_funded_headers

        headers, owner = register_funded_headers(client, "pel-guard")
        repo = ServerlessRepo()
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="guard",
                mode="preset",
                model_ref="Qwen/Qwen3-8B",
                managed_engine="vllm",
            )
        )
        r = client.post(
            "/api/v2/serverless/should-i-run-this",
            headers=headers,
            json={
                "endpoint_id": ep["endpoint_id"],
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 200,
                "duration_hours": 0.05,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "approved" in body
        assert "estimated_token_cost_cad" in body
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)