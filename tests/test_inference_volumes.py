"""Tests for inference engine and volume/cloudburst management.

Pure logic tests (VRAM estimation, token cost, dataclasses, constants)
don't need a database. DB-dependent tests are marked with @pytest.mark.pg.
"""

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from inference import InferenceEngine, InferenceRequest, InferenceResponse


class TestInferenceVRAM:
    """Test VRAM estimation (pure logic, no DB)."""

    def _ie(self):
        """Create engine without connecting to DB."""
        return InferenceEngine.__new__(InferenceEngine)

    def test_7b_model_estimate(self):
        est = self._ie()._estimate_vram_gb("meta-llama/Llama-2-7b-chat-hf")
        assert 4 <= est <= 10

    def test_70b_model_estimate(self):
        est = self._ie()._estimate_vram_gb("meta-llama/Llama-2-70b-chat-hf")
        assert 30 <= est <= 50

    def test_13b_model_estimate(self):
        est = self._ie()._estimate_vram_gb("codellama/CodeLlama-13b-Instruct-hf")
        assert 8 <= est <= 15

    def test_unknown_model_default(self):
        est = self._ie()._estimate_vram_gb("some-unknown-model")
        assert est == 8.0  # Conservative default

    def test_bert_small(self):
        est = self._ie()._estimate_vram_gb("distilbert-base-uncased")
        assert est == 1.0


class TestTokenCost:
    """Test token cost computation (pure math)."""

    def _ie(self):
        return InferenceEngine.__new__(InferenceEngine)

    def test_basic_cost_positive(self):
        cost = self._ie().compute_token_cost(input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_zero_tokens_zero_cost(self):
        cost = self._ie().compute_token_cost(input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_output_more_expensive_per_token(self):
        ie = self._ie()
        input_only = ie.compute_token_cost(input_tokens=1_000_000, output_tokens=0)
        output_only = ie.compute_token_cost(input_tokens=0, output_tokens=1_000_000)
        assert output_only > input_only


class TestInferenceDataclasses:
    """Test dataclass construction."""

    def test_request_defaults(self):
        req = InferenceRequest(
            endpoint_id="ep-1",
            model="test-model",
        )
        assert req.endpoint_id == "ep-1"
        assert req.max_tokens == 2048
        assert req.temperature == pytest.approx(0.7)
        assert req.stream is False
        assert req.request_id.startswith("req-")

    def test_request_custom_values(self):
        req = InferenceRequest(
            request_id="custom-123",
            endpoint_id="ep-2",
            model="llama-7b",
            max_tokens=1024,
            temperature=0.5,
            stream=True,
            user="cust-1",
        )
        assert req.request_id == "custom-123"
        assert req.max_tokens == 1024
        assert req.user == "cust-1"

    def test_response_defaults(self):
        resp = InferenceResponse()
        assert resp.object == "chat.completion"
        assert resp.choices == []
        assert resp.usage == {}
        assert resp.cold_start is False


class TestVolumeEngine:
    """Test persistent volume management."""

    def test_volume_engine_api_surface(self):
        """VolumeEngine should have all required methods."""
        from volumes import VolumeEngine
        required_methods = [
            "create_volume", "attach_volume", "detach_volume",
            "delete_volume", "list_volumes", "get_instance_volumes",
            "detach_all_for_instance",
        ]
        for method in required_methods:
            assert hasattr(VolumeEngine, method), f"Missing method: {method}"


class TestCloudBurst:
    """Test cloud burst auto-scaling engine."""

    def test_instance_type_definitions(self):
        from cloudburst import CLOUD_INSTANCE_TYPES
        assert "aws" in CLOUD_INSTANCE_TYPES
        assert "gcp" in CLOUD_INSTANCE_TYPES
        assert len(CLOUD_INSTANCE_TYPES["aws"]) > 0
        assert len(CLOUD_INSTANCE_TYPES["gcp"]) > 0

    def test_burst_config_defaults(self):
        from cloudburst import BURST_BUDGET_CAD, BURST_MAX_INSTANCES, BURST_QUEUE_THRESHOLD
        assert BURST_BUDGET_CAD > 0
        assert BURST_MAX_INSTANCES > 0
        assert BURST_QUEUE_THRESHOLD > 0

    def test_engine_api_surface(self):
        from cloudburst import CloudBurstEngine
        required_methods = [
            "evaluate_burst_need", "provision_burst_instance",
            "drain_idle_instances", "terminate_instance", "get_burst_status",
        ]
        for method in required_methods:
            assert hasattr(CloudBurstEngine, method), f"Missing method: {method}"
