"""Phase 5 tests: OpenTelemetry instrumentation + LUKS encrypted volume provisioning.

Tests cover:
  - OpenTelemetry integration in api.py (FastAPIInstrumentor, custom spans, W3C propagation)
  - LUKS volume lifecycle in worker_agent.py (provision, attach, detach, destroy, cleanup)
  - otel_span() fallback when opentelemetry is not installed
"""

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── OpenTelemetry Tests ──────────────────────────────────────────────


class TestOpenTelemetryIntegration:
    """Verify OpenTelemetry is wired into the API layer."""

    def test_otel_packages_in_requirements(self):
        """All four OTel packages are in requirements.txt."""
        reqs = Path(__file__).resolve().parent.parent / "requirements.txt"
        text = reqs.read_text()
        for pkg in [
            "opentelemetry-api",
            "opentelemetry-sdk",
            "opentelemetry-instrumentation-fastapi",
            "opentelemetry-exporter-otlp",
        ]:
            assert pkg in text, f"{pkg} missing from requirements.txt"

    def test_otel_span_helper_exists(self):
        """api.py exports an otel_span() helper."""
        import api

        assert hasattr(api, "otel_span"), "otel_span not found in api module"
        assert callable(api.otel_span)

    def test_otel_span_returns_context_manager(self):
        """otel_span() returns a usable context manager even when OTel is disabled."""
        import api

        ctx = api.otel_span("test.span", {"key": "value"})
        # Should work as a context manager regardless of OTel state
        with ctx:
            pass  # No error means it works

    def test_otel_enabled_flag_exists(self):
        """_OTEL_ENABLED flag is defined in api.py."""
        import api

        assert hasattr(api, "_OTEL_ENABLED")
        assert isinstance(api._OTEL_ENABLED, bool)
        assert api._OTEL_ENABLED is True

    def test_otel_tracer_attribute(self):
        """_otel_tracer is defined (may be None if OTel not installed)."""
        import api

        assert hasattr(api, "_otel_tracer")

    def test_fastapi_instrumentor_import_attempted(self):
        """api.py source references FastAPIInstrumentor."""
        source = Path(__file__).resolve().parent.parent / "api" / "api.py"
        text = source.read_text()
        assert "FastAPIInstrumentor" in text
        assert "instrument_app" in text

    def test_w3c_trace_context_propagation(self):
        """api.py sets up W3C Trace Context propagation."""
        source = Path(__file__).resolve().parent.parent / "api" / "api.py"
        text = source.read_text()
        assert "TraceContextTextMapPropagator" in text
        assert "W3CBaggagePropagator" in text
        assert "from opentelemetry.propagate import set_global_textmap" in text
        assert "trace.propagation.tracecontext" in text

    def test_custom_spans_in_endpoints(self):
        """Key endpoints have otel_span() calls."""
        base = Path(__file__).resolve().parent.parent
        routes_dir = base / "api" / "routes"
        all_text = (base / "api" / "api.py").read_text()
        for py in routes_dir.glob("*.py"):
            all_text += py.read_text()
        for span_name in [
            "job.submit",
            "job.status_update",
            "billing.bill_job",
            "webhook.stripe",
            "billing.cycle",
        ]:
            assert (
                span_name in all_text
            ), f"Custom span '{span_name}' not found in routes/ or api.py"

    def test_otlp_exporter_conditional(self):
        """OTLP exporter only activates when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
        source = Path(__file__).resolve().parent.parent / "api" / "api.py"
        text = source.read_text()
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in text
        assert "OTLPSpanExporter" in text

    def test_service_name_resource(self):
        """OpenTelemetry resource sets service name to xcelsior-api."""
        source = Path(__file__).resolve().parent.parent / "api" / "api.py"
        text = source.read_text()
        assert "xcelsior-api" in text


class TestOpenTelemetryGracefulDegradation:
    """Verify OTel gracefully degrades when packages are missing."""

    def test_otel_span_with_none_tracer(self):
        """otel_span returns nullcontext when _otel_tracer is None."""
        import routes._deps as _deps_mod

        original = _deps_mod._otel_tracer
        try:
            _deps_mod._otel_tracer = None
            ctx = _deps_mod.otel_span("test.noop")
            with ctx:
                result = 42
            assert result == 42
        finally:
            _deps_mod._otel_tracer = original

    def test_api_source_has_import_error_handler(self):
        """api.py wraps OTel imports in try/except ImportError."""
        source = Path(__file__).resolve().parent.parent / "api" / "api.py"
        text = source.read_text()
        assert "except ImportError" in text


# ── LUKS Encrypted Volume Tests ──────────────────────────────────────


class TestLUKSVolumeConstants:
    """Verify LUKS volume module constants and functions exist."""

    def test_volume_base_dir_constant(self):
        import worker_agent as wa

        assert hasattr(wa, "VOLUME_BASE_DIR")
        assert "volumes" in wa.VOLUME_BASE_DIR

    def test_volume_key_dir_constant(self):
        import worker_agent as wa

        assert hasattr(wa, "VOLUME_KEY_DIR")
        assert "volume-keys" in wa.VOLUME_KEY_DIR

    def test_provision_function_exists(self):
        import worker_agent as wa

        assert hasattr(wa, "provision_encrypted_volume")
        assert callable(wa.provision_encrypted_volume)

    def test_attach_function_exists(self):
        import worker_agent as wa

        assert hasattr(wa, "attach_encrypted_volume")
        assert callable(wa.attach_encrypted_volume)

    def test_detach_function_exists(self):
        import worker_agent as wa

        assert hasattr(wa, "detach_encrypted_volume")
        assert callable(wa.detach_encrypted_volume)

    def test_destroy_function_exists(self):
        import worker_agent as wa

        assert hasattr(wa, "destroy_encrypted_volume")
        assert callable(wa.destroy_encrypted_volume)

    def test_cleanup_function_exists(self):
        import worker_agent as wa

        assert hasattr(wa, "_cleanup_partial_volume")
        assert callable(wa._cleanup_partial_volume)


class TestLUKSVolumeUsesRootHelper:
    """The worker asks the root helper for every LUKS step (tests/test_worker_privileged.py
    covers what the helper does)."""

    @patch(
        "worker_privileged.call",
        return_value={"ok": True, "mount_point": "/mnt/xcelsior/volumes/v1"},
    )
    def test_provision_requests_helper(self, mock_call):
        from worker_agent import provision_encrypted_volume

        assert provision_encrypted_volume("v1", 10) is True
        mock_call.assert_called_once_with("luks_provision", volume_id="v1", size_gb=10)

    @patch("worker_privileged.call", return_value={"ok": False, "error": "luksFormat failed"})
    def test_provision_failure_returns_false(self, mock_call):
        from worker_agent import provision_encrypted_volume

        assert provision_encrypted_volume("v1", 10) is False

    @patch(
        "worker_privileged.call",
        return_value={"ok": True, "mount_point": "/mnt/xcelsior/volumes/v1"},
    )
    def test_attach_returns_mount_point(self, mock_call):
        from worker_agent import attach_encrypted_volume

        assert attach_encrypted_volume("v1") == "/mnt/xcelsior/volumes/v1"
        mock_call.assert_called_once_with("luks_attach", volume_id="v1")

    @patch(
        "worker_privileged.call", return_value={"ok": False, "refused": True, "error": "not found"}
    )
    def test_attach_failure_returns_none(self, mock_call):
        from worker_agent import attach_encrypted_volume

        assert attach_encrypted_volume("v1") is None

    @patch("worker_privileged.call", return_value={"ok": True})
    def test_detach_and_destroy_request_helper(self, mock_call):
        from worker_agent import destroy_encrypted_volume, detach_encrypted_volume

        assert detach_encrypted_volume("v1") is True
        assert destroy_encrypted_volume("v1") is True
        assert [c.args[0] for c in mock_call.call_args_list] == ["luks_detach", "luks_destroy"]

    @patch("worker_privileged.call", return_value={"ok": False, "error": "shred failed"})
    def test_destroy_failure_returns_false(self, mock_call):
        from worker_agent import destroy_encrypted_volume

        assert destroy_encrypted_volume("v1") is False


class TestRunJobEncryptedVolumes:
    """Verify run_job integrates encrypted volume attach/detach."""

    def test_run_job_docstring_mentions_encrypted(self):
        """run_job docstring references encrypted volumes."""
        import worker_agent as wa

        assert "encrypted" in wa.run_job.__doc__.lower()

    def test_source_references_attach_detach(self):
        """run_job source calls attach_encrypted_volume and detach_encrypted_volume."""
        source = Path(__file__).resolve().parent.parent / "workers" / "worker_agent.py"
        text = source.read_text()
        assert "attach_encrypted_volume" in text
        assert "detach_encrypted_volume" in text
        assert "encrypted_vol_ids" in text
