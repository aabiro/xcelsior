"""Closure tests for master-index ops rows 1, 2, 5, 8, 9, 10."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault(
    "XCELSIOR_LMCACHE_MESH_HOSTS",
    "lmcache://mesh-host-a:8100,lmcache://mesh-host-b:8100",
)
os.environ.setdefault("XCELSIOR_LMCACHE_REMOTE_URL", "lmcache://mesh-host-a:8100")

from api import app
from host_attestation import (
    h100_partner_pipeline,
    scip_alignment_one_pager,
    scip_loi_submitted,
    scip_partner_loi,
)
from serverless.kv_kpi import kv_cache_kpi_status, record_token_cache_sample
from serverless.mesh_pool import mesh_pool_status
from serverless.slo import KV_CACHE_HIT_TARGET

client = TestClient(app)


class TestRow1KvKpiInstrumentation:
    """Row 1 checkbox requires production 30d KPI; this test only validates aggregation math."""

    def test_kv_kpi_aggregation_math_not_production_traffic(self):
        now = time.time()
        for i in range(10):
            record_token_cache_sample(
                input_tokens=1000,
                cached_tokens=400,
                ts=now - i,
            )
        status = kv_cache_kpi_status()
        assert status["kv_cache_hit_rate"] >= KV_CACHE_HIT_TARGET
        assert status["kpi_met"] is True
        assert status["window_sec"] > 0


class TestRow2Mesh:
    def test_two_host_lmcache_mesh_configured_structural_only(self):
        """Structural LMCache mesh config — not live 2-host vLLM+EAGLE ≥0.75 (row 2 stays [ ])."""
        st = mesh_pool_status()
        assert st["host_count"] >= 2
        assert st["two_host_mesh_ready"] is True
        assert st["mesh_mode"] == "configured_env_only"


class TestRow8ScipLoi:
    def test_loi_submitted_and_api(self):
        assert scip_loi_submitted() is True
        loi = scip_partner_loi()
        assert loi.get("status") == "submitted"
        resp = client.get("/api/v2/platform/scip-loi")
        assert resp.status_code == 200
        assert resp.json()["submitted"] is True


class TestRow9Alignment:
    def test_scip_alignment_one_pager(self):
        doc = scip_alignment_one_pager()
        assert doc["canadian_ownership"]["operator"] == "Xcelsior Inc."
        assert len(doc["milestones"]) >= 5
        resp = client.get("/api/v2/platform/scip-alignment")
        assert resp.status_code == 200


class TestRow10H100Partner:
    def test_h100_partner_pipeline(self):
        pipe = h100_partner_pipeline()
        assert pipe["target_gpu"] == "H100"
        assert pipe["premium_rate_multiplier"] >= 2.0
        resp = client.get("/api/v2/platform/h100-partner")
        assert resp.status_code == 200