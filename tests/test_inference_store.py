"""Tests for Xcelsior inference persistence store."""

import os
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

# Redirect inference DB to a temp directory before importing
_tmpdir = tempfile.mkdtemp()

import inference_store

inference_store._INFERENCE_DB_DIR = _tmpdir
inference_store._INFERENCE_DB_PATH = os.path.join(_tmpdir, "inference_test.db")


# ── Unit Tests: Store Module ──────────────────────────────────────────


class TestStoreInferenceJob:
    def setup_method(self):
        with inference_store._inference_db() as conn:
            conn.execute("DELETE FROM inference_results")
            conn.execute("DELETE FROM inference_jobs")

    def test_store_and_get_job(self):
        inference_store.store_inference_job(
            job_id="j1",
            customer_id="cust@test.com",
            model="llama-2-7b",
            inputs=["Hello"],
            max_tokens=256,
            temperature=0.7,
            timeout_sec=300,
        )
        job = inference_store.get_inference_job("j1")
        assert job is not None
        assert job["job_id"] == "j1"
        assert job["customer_id"] == "cust@test.com"
        assert job["model"] == "llama-2-7b"
        assert job["inputs"] == ["Hello"]
        assert job["max_tokens"] == 256
        assert job["temperature"] == 0.7
        assert job["status"] == "queued"

    def test_get_missing_job_returns_none(self):
        assert inference_store.get_inference_job("nonexistent") is None

    def test_store_and_get_result(self):
        inference_store.store_inference_job(
            job_id="j2",
            customer_id="c2",
            model="distilbert",
            inputs=["test input"],
            max_tokens=128,
            temperature=1.0,
            timeout_sec=60,
        )
        inference_store.store_inference_result(
            job_id="j2",
            outputs=[{"label": "POSITIVE", "score": 0.99}],
            model="distilbert",
            latency_ms=42.5,
        )
        result = inference_store.get_inference_result("j2")
        assert result is not None
        assert result["job_id"] == "j2"
        assert result["outputs"] == [{"label": "POSITIVE", "score": 0.99}]
        assert result["latency_ms"] == 42.5

        # Job status should be updated to completed
        job = inference_store.get_inference_job("j2")
        assert job["status"] == "completed"
        assert job["completed_at"] is not None

    def test_get_missing_result_returns_none(self):
        assert inference_store.get_inference_result("nonexistent") is None

    def test_delete_job(self):
        inference_store.store_inference_job(
            job_id="j3",
            customer_id="c3",
            model="m",
            inputs=["x"],
            max_tokens=64,
            temperature=1.0,
            timeout_sec=30,
        )
        inference_store.store_inference_result(
            job_id="j3",
            outputs=["out"],
            model="m",
            latency_ms=10,
        )
        inference_store.delete_inference_job("j3")
        assert inference_store.get_inference_job("j3") is None
        assert inference_store.get_inference_result("j3") is None

    def test_purge_expired_jobs(self):
        inference_store.store_inference_job(
            job_id="old",
            customer_id="c",
            model="m",
            inputs=["x"],
            max_tokens=64,
            temperature=1.0,
            timeout_sec=30,
        )
        # Manually backdate the submitted_at
        with inference_store._inference_db() as conn:
            conn.execute(
                "UPDATE inference_jobs SET submitted_at = %s WHERE job_id = 'old'",
                (time.time() - 100000,),
            )
        inference_store.store_inference_job(
            job_id="new",
            customer_id="c",
            model="m",
            inputs=["y"],
            max_tokens=64,
            temperature=1.0,
            timeout_sec=30,
        )
        deleted = inference_store.purge_expired_jobs(ttl_sec=3600)
        assert deleted == 1
        assert inference_store.get_inference_job("old") is None
        assert inference_store.get_inference_job("new") is not None

    def test_multiple_inputs_preserved(self):
        inputs = ["input one", "input two", "input three"]
        inference_store.store_inference_job(
            job_id="j4",
            customer_id="c4",
            model="m",
            inputs=inputs,
            max_tokens=512,
            temperature=0.5,
            timeout_sec=120,
        )
        job = inference_store.get_inference_job("j4")
        assert job["inputs"] == inputs


# ── Integration: API Endpoints ────────────────────────────────────────


class TestInferenceEndpoints:
    def setup_method(self):
        with inference_store._inference_db() as conn:
            conn.execute("DELETE FROM inference_results")
            conn.execute("DELETE FROM inference_jobs")
        # Seed wallet for anonymous test user so wallet pre-flight checks pass
        from billing import get_billing_engine

        get_billing_engine().deposit("anonymous", 10_000.0, description="Test credits")

    def test_submit_and_get_inference(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        # Submit
        resp = client.post(
            "/api/inference",
            json={
                "model": "distilbert-base-uncased-finetuned-sst-2-english",
                "inputs": "I love this product",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"]
        job_id = data["job_id"]

        # Check status (should be queued/running)
        resp = client.get(f"/api/inference/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("queued", "running")

    def test_post_result_and_get_completed(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        # Submit
        resp = client.post(
            "/api/inference",
            json={
                "model": "distilbert-base-uncased-finetuned-sst-2-english",
                "inputs": "test",
            },
        )
        job_id = resp.json()["job_id"]

        # Post result (worker callback)
        resp = client.post(
            f"/api/inference/{job_id}/result",
            json={
                "outputs": [{"label": "POSITIVE", "score": 0.95}],
                "model": "distilbert",
                "latency_ms": 55,
            },
        )
        assert resp.status_code == 200

        # Get completed result
        resp = client.get(f"/api/inference/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["outputs"] == [{"label": "POSITIVE", "score": 0.95}]

    def test_get_nonexistent_job(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        resp = client.get("/api/inference/does-not-exist")
        assert resp.status_code == 404

    def test_models_available(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        resp = client.get("/api/inference/models/available")
        assert resp.status_code == 200
        assert len(resp.json()["models"]) >= 4
